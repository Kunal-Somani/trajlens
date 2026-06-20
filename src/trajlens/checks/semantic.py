"""SEMANTIC checks (04_CHECK_CATALOG.md §SEMANTIC).

Checks in this module validate the *meaning* of features and metadata, not
just their structural presence:

  SEMANTIC.FEATURE_DIMENSIONALITY  (FAIL) — column widths match declared shape
  SEMANTIC.TASK_INTEGRITY          (FAIL) — every task_index has a non-empty task
  SEMANTIC.CAMERA_INTRINSICS_PLAUSIBLE (INFO) — conditional; see note below
  SEMANTIC.LANGUAGE_PRESENT        (WARN) — episodes have non-empty task strings

Camera intrinsics [verify finding, M6]:
  The LeRobot v3.0 DatasetInfo dataclass (lerobot/datasets/utils.py, current
  main, DatasetInfo.__dataclass_fields__) has no camera intrinsics field.
  The features map (info.json["features"]) stores only the standard feature
  schema: dtype, shape, names.  Neither the DatasetInfo dataclass nor any
  standard LeRobot writer code (dataset_writer.py, dataset_metadata.py)
  writes intrinsic parameters (focal length, principal point, distortion) into
  any standard location.  Camera calibration, when present at all, lives
  outside the dataset schema — typically in robot YAML configs used during
  collection, which are not part of the deposited dataset.

  Implication: the check cannot be WARN against a field that does not exist
  in the standard format.  Per 04_CHECK_CATALOG.md's own bracketed note
  ("if the format does not standardize intrinsics, this check is INFO and
  only runs when a known intrinsics convention is found"), the check is
  implemented as INFO-severity and fires only when a feature with a name
  matching a recognized intrinsics convention is found in the features map.

  Recognized conventions (community datasets only; not part of the standard):
    - Feature name contains "intrinsic" (case-insensitive), e.g.
      "observation.camera_intrinsics" or "intrinsics_matrix"
    - Feature name contains "camera_matrix"
    - Feature name is exactly "K" with shape [3, 3] or [9]

  Source verified against: lerobot/datasets/utils.py DatasetInfo (main branch,
  accessed 2026-06-20); lerobot/datasets/dataset_writer.py; lerobot/datasets/
  dataset_metadata.py.  No intrinsics field exists in the standard schema.
"""

from __future__ import annotations

import structlog

from trajlens.checks.protocol import Check, CheckContext, CheckResult, Severity
from trajlens.checks.registry import registry
from trajlens.model.canonical import CanonicalDataset, FeatureSpec

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# SEMANTIC.FEATURE_DIMENSIONALITY
# ---------------------------------------------------------------------------


class _FeatureDimensionalityCheck:
    id = "SEMANTIC.FEATURE_DIMENSIONALITY"
    severity = Severity.FAIL
    category = "SEMANTIC"
    requires_video = False

    def run(self, ds: CanonicalDataset, ctx: CheckContext) -> CheckResult:
        mismatches: list[str] = []

        for episode in ds:
            pf = ds.parquet_shard_for_episode(episode)
            schema = pf.schema_arrow

            # Collect only features that need width checks to avoid full column reads.
            candidate_features = {
                name: spec
                for name, spec in ds.features.items()
                if spec.dtype != "video" and name in schema.names
            }
            if not candidate_features:
                continue

            # Read only the candidate columns plus episode_index for filtering.
            table = pf.read(  # type: ignore[no-untyped-call]
                columns=["episode_index", *candidate_features.keys()]
            )
            ep_col = table.column("episode_index").to_pylist()
            # Get the first row index belonging to this episode for sampling.
            sample_idx: int | None = None
            for row_i, ep_v in enumerate(ep_col):
                if ep_v == episode.episode_index:
                    sample_idx = row_i
                    break
            if sample_idx is None:
                continue

            for feat_name, feat_spec in candidate_features.items():
                field_idx = schema.get_field_index(feat_name)
                arrow_type = schema.field(field_idx).type
                declared_width = _declared_width(feat_spec)

                # Compute actual width from Arrow type; for variable-length lists,
                # sample the first row from this episode to get the actual length.
                actual_width = _arrow_column_width(arrow_type)
                if actual_width is None:
                    # Variable-length list: sample one actual row value.
                    col_list = table.column(feat_name).to_pylist()
                    sampled = col_list[sample_idx]
                    actual_width = len(sampled) if isinstance(sampled, list) else 1

                if actual_width != declared_width:
                    mismatches.append(
                        f"episode {episode.episode_index}: feature {feat_name!r} "
                        f"has width {actual_width} but declared shape={feat_spec.shape} "
                        f"implies width {declared_width}"
                    )

                # Check names length when present.
                if feat_spec.names is not None and len(feat_spec.names) != declared_width:
                    mismatches.append(
                        f"episode {episode.episode_index}: feature {feat_name!r} "
                        f"names list has {len(feat_spec.names)} entries but "
                        f"declared shape={feat_spec.shape} implies {declared_width}"
                    )

            if mismatches:
                break  # First episode with issues is sufficient signal.

        if mismatches:
            return CheckResult(
                check_id=self.id,
                severity=Severity.FAIL,
                message=(
                    f"Feature dimensionality mismatch ({len(mismatches)} issue(s)): {mismatches[0]}"
                ),
                details={"mismatches": mismatches},
            )
        return CheckResult(
            check_id=self.id,
            severity=Severity.INFO,
            message="All feature column widths match their declared shapes.",
        )


def _declared_width(spec: FeatureSpec) -> int:
    """Total element count from declared shape tuple (product of dims)."""
    result = 1
    for dim in spec.shape:
        result *= dim
    return result


def _arrow_column_width(arrow_type: object) -> int | None:
    """Return element width of an Arrow column type, or None if undecidable."""
    import pyarrow as pa

    t = arrow_type
    if isinstance(t, pa.lib.FixedSizeListType):
        # list_size gives the outer dimension; recurse for nested.
        inner = _arrow_column_width(t.value_type)
        if inner is None:
            return None
        return int(t.list_size) * inner
    if isinstance(t, pa.lib.ListType):
        # Variable-length list — width not statically knowable.
        return None
    # Scalar types map to width 1.
    return 1


FEATURE_DIMENSIONALITY: Check = _FeatureDimensionalityCheck()
registry.register(FEATURE_DIMENSIONALITY)


# ---------------------------------------------------------------------------
# SEMANTIC.TASK_INTEGRITY
# ---------------------------------------------------------------------------


class _TaskIntegrityCheck:
    id = "SEMANTIC.TASK_INTEGRITY"
    severity = Severity.FAIL
    category = "SEMANTIC"
    requires_video = False

    def run(self, ds: CanonicalDataset, ctx: CheckContext) -> CheckResult:
        violations: list[str] = []

        # Collect all task_indices referenced in frame data. ep.tasks gives us
        # the resolved task strings already, but we still need to verify the
        # raw task_index integer values in the Parquet frame rows are all
        # present in the task_table (defined_indices).
        referenced: set[int] = set()

        # Scan per-episode Parquet for referenced task_index values.
        defined_indices: set[int] = set(ds.task_table.keys())
        for episode in ds:
            pf = ds.parquet_shard_for_episode(episode)
            table = pf.read(columns=["episode_index", "task_index"])  # type: ignore[no-untyped-call]
            ep_col = table.column("episode_index").to_pylist()
            ti_col = table.column("task_index").to_pylist()
            for ep_val, ti_val in zip(ep_col, ti_col, strict=True):
                if ep_val == episode.episode_index:
                    referenced.add(int(ti_val))

        # Undefined references.
        undefined = referenced - defined_indices
        for idx in sorted(undefined):
            violations.append(
                f"task_index {idx} is referenced in frame data but not defined in the task table"
            )

        # Empty task descriptions.
        for idx, description in ds.task_table.items():
            if not description or not description.strip():
                violations.append(
                    f"task_index {idx} maps to an empty or whitespace-only task description"
                )

        if violations:
            return CheckResult(
                check_id=self.id,
                severity=Severity.FAIL,
                message=f"Task integrity violated ({len(violations)} issue(s)): {violations[0]}",
                details={"violations": violations},
            )
        return CheckResult(
            check_id=self.id,
            severity=Severity.INFO,
            message=("All task_index references are defined and map to non-empty descriptions."),
        )


TASK_INTEGRITY: Check = _TaskIntegrityCheck()
registry.register(TASK_INTEGRITY)


# ---------------------------------------------------------------------------
# SEMANTIC.CAMERA_INTRINSICS_PLAUSIBLE
# ---------------------------------------------------------------------------
# INFO severity, conditional: only fires when a recognized intrinsics
# convention is found in the features map.  See module docstring for the
# [verify] finding and citation.

# Feature name patterns that suggest camera intrinsics by community convention.
_INTRINSICS_NAME_FRAGMENTS = ("intrinsic", "camera_matrix")
_INTRINSICS_EXACT_K = "K"
# Plausible shapes for a 3x3 camera matrix stored flat or 2-D.
_K_SHAPES: frozenset[tuple[int, ...]] = frozenset({(3, 3), (9,)})


def _is_intrinsics_feature(name: str, spec: FeatureSpec) -> bool:
    """Return True if this feature looks like a camera intrinsics matrix."""
    lower = name.lower()
    if any(frag in lower for frag in _INTRINSICS_NAME_FRAGMENTS):
        return True
    return name == _INTRINSICS_EXACT_K and spec.shape in _K_SHAPES


class _CameraIntrinsicsPlausibleCheck:
    id = "SEMANTIC.CAMERA_INTRINSICS_PLAUSIBLE"
    # INFO because the format does not standardize intrinsics; this check is
    # advisory and only fires on community-convention fields.
    severity = Severity.INFO
    category = "SEMANTIC"
    requires_video = False

    def run(self, ds: CanonicalDataset, ctx: CheckContext) -> CheckResult:
        intrinsics_features = [
            (name, spec) for name, spec in ds.features.items() if _is_intrinsics_feature(name, spec)
        ]

        if not intrinsics_features:
            # No recognized intrinsics field — check is not applicable.
            return CheckResult(
                check_id=self.id,
                severity=Severity.INFO,
                message=(
                    "No camera intrinsics field found in features map "
                    "(the LeRobot standard format does not include intrinsics; "
                    "this check is advisory and skipped)."
                ),
            )

        violations: list[str] = []

        for feat_name, _feat_spec in intrinsics_features:
            # Read values from the first episode and validate plausibility.
            try:
                first_ep = next(iter(ds))
            except StopIteration:
                break

            pf = ds.parquet_shard_for_episode(first_ep)
            table = pf.read(columns=["episode_index", feat_name])  # type: ignore[no-untyped-call]
            ep_col = table.column("episode_index").to_pylist()
            vals_col = table.column(feat_name).to_pylist()
            ep_rows = [
                v for v, ep in zip(vals_col, ep_col, strict=True) if ep == first_ep.episode_index
            ]
            if not ep_rows:
                continue

            # Parse the first frame's intrinsics.  Values may be flat lists or scalars.
            first_val = ep_rows[0]
            if isinstance(first_val, list):
                flat = [float(x) for x in first_val]
            else:
                flat = [float(first_val)]

            if len(flat) < 4:
                violations.append(
                    f"Feature {feat_name!r}: intrinsics value has only {len(flat)} "
                    f"elements (expected at least 4 for fx, fy, cx, cy)"
                )
                continue

            # For a flat [fx, 0, cx, 0, fy, cy, 0, 0, 1] or [3x3] layout
            # the first element is fx and the fifth (index 4) is fy.
            # For a minimal [fx, fy, cx, cy] layout, index 0 and 1 are fx/fy.
            # For a 3x3 flat layout [fx, 0, cx, 0, fy, cy, 0, 0, 1]: fx@0, fy@4, cx@2, cy@5.
            # For a [fx, fy, cx, cy] 4-element compact layout: indices 0,1,2,3.
            fx = flat[0]
            fy = flat[4] if len(flat) >= 9 else flat[1]
            cx = flat[2]
            cy = flat[5] if len(flat) >= 9 else flat[3]

            if fx <= 0 or fy <= 0:
                violations.append(
                    f"Feature {feat_name!r}: focal lengths must be positive "
                    f"(fx={fx:.4f}, fy={fy:.4f})"
                )
            if cx <= 0 or cy <= 0:
                violations.append(
                    f"Feature {feat_name!r}: principal point coordinates must be positive "
                    f"(cx={cx:.4f}, cy={cy:.4f})"
                )

        if violations:
            found_names_bad = [n for n, _ in intrinsics_features]
            return CheckResult(
                check_id=self.id,
                severity=Severity.INFO,
                message=(
                    f"Camera intrinsics field found but values are implausible "
                    f"({len(violations)} issue(s)): {violations[0]}"
                ),
                details={"violations": violations, "intrinsics_features": found_names_bad},
            )

        found_names = [n for n, _ in intrinsics_features]
        return CheckResult(
            check_id=self.id,
            severity=Severity.INFO,
            message=(
                f"Camera intrinsics field(s) {found_names} found and values appear plausible."
            ),
            details={"intrinsics_features": found_names},
        )


CAMERA_INTRINSICS_PLAUSIBLE: Check = _CameraIntrinsicsPlausibleCheck()
registry.register(CAMERA_INTRINSICS_PLAUSIBLE)


# ---------------------------------------------------------------------------
# SEMANTIC.LANGUAGE_PRESENT
# ---------------------------------------------------------------------------


class _LanguagePresentCheck:
    id = "SEMANTIC.LANGUAGE_PRESENT"
    severity = Severity.WARN
    category = "SEMANTIC"
    requires_video = False

    def run(self, ds: CanonicalDataset, ctx: CheckContext) -> CheckResult:
        # An episode has a language description if its tasks list is non-empty
        # and at least one task string is non-empty after stripping whitespace.
        empty_episodes: list[int] = []

        for ep in ds:
            has_language = any(t and t.strip() for t in ep.tasks)
            if not has_language:
                empty_episodes.append(ep.episode_index)

        if empty_episodes:
            sample = empty_episodes[:5]
            return CheckResult(
                check_id=self.id,
                severity=Severity.WARN,
                message=(
                    f"{len(empty_episodes)} episode(s) have no non-empty language task "
                    f"description (sample: {sample}).  Language-conditioned policies "
                    f"require non-empty task strings."
                ),
                details={"empty_episode_indices": empty_episodes},
            )
        return CheckResult(
            check_id=self.id,
            severity=Severity.INFO,
            message="All episodes have at least one non-empty language task description.",
        )


LANGUAGE_PRESENT: Check = _LanguagePresentCheck()
registry.register(LANGUAGE_PRESENT)
