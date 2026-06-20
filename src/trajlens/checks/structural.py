"""STRUCTURAL checks (04_CHECK_CATALOG.md §STRUCTURAL).

All six STRUCTURAL checks are implemented here and registered at module load
time via the singleton registry from checks/registry.py.

  STRUCTURAL.VERSION_DETECTED         (INFO)
  STRUCTURAL.REQUIRED_METADATA_PRESENT (FAIL)
  STRUCTURAL.SCHEMA_CONSISTENCY        (FAIL)
  STRUCTURAL.INDEX_CONTINUITY          (FAIL)
  STRUCTURAL.METADATA_DATA_AGREEMENT   (FAIL) — catches #2401 corruption
  STRUCTURAL.PATH_TEMPLATE_RESOLVES    (FAIL)
"""

from __future__ import annotations

import pyarrow as pa
import structlog

from trajlens.checks.protocol import Check, CheckContext, CheckResult, Severity
from trajlens.checks.registry import registry
from trajlens.model.canonical import CanonicalDataset
from trajlens.sources.paths import safe_join
from trajlens.sources.version import DatasetVersion

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# STRUCTURAL.VERSION_DETECTED
# ---------------------------------------------------------------------------


class _VersionDetectedCheck:
    id = "STRUCTURAL.VERSION_DETECTED"
    severity = Severity.INFO
    category = "STRUCTURAL"
    requires_video = False

    def run(self, ds: CanonicalDataset, ctx: CheckContext) -> CheckResult:
        version_str = ds.version.value
        return CheckResult(
            check_id=self.id,
            severity=Severity.INFO,
            message=f"Detected format version: {version_str}",
            details={"version": version_str},
        )


VERSION_DETECTED: Check = _VersionDetectedCheck()
registry.register(VERSION_DETECTED)


# ---------------------------------------------------------------------------
# STRUCTURAL.REQUIRED_METADATA_PRESENT
# ---------------------------------------------------------------------------

# Files/dirs required per version.  Each entry is (path_parts, is_dir, versions_it_applies_to).
_V3_REQUIRED: list[tuple[tuple[str, ...], bool]] = [
    (("meta", "info.json"), False),
    (("meta", "tasks.parquet"), False),
    (("meta", "episodes"), True),
]
_V2_REQUIRED: list[tuple[tuple[str, ...], bool]] = [
    (("meta", "info.json"), False),
    (("meta", "episodes.jsonl"), False),
    (("meta", "tasks.jsonl"), False),
]


class _RequiredMetadataPresentCheck:
    id = "STRUCTURAL.REQUIRED_METADATA_PRESENT"
    severity = Severity.FAIL
    category = "STRUCTURAL"
    requires_video = False

    def run(self, ds: CanonicalDataset, ctx: CheckContext) -> CheckResult:
        # We need the dataset root to stat files.  The CanonicalDataset carries
        # a stats handle whose root is the dataset root — safe to access here.
        root = ds.stats.root

        required = _V3_REQUIRED if ds.version is DatasetVersion.V3_0 else _V2_REQUIRED
        missing: list[str] = []

        for path_parts, is_dir in required:
            try:
                resolved = safe_join(root, *path_parts)
            except Exception:
                missing.append("/".join(path_parts))
                continue
            exists = resolved.is_dir() if is_dir else resolved.is_file()
            if not exists:
                missing.append("/".join(path_parts))

        if missing:
            return CheckResult(
                check_id=self.id,
                severity=Severity.FAIL,
                message=f"Required metadata missing: {', '.join(missing)}",
                details={"missing": missing},
            )
        return CheckResult(
            check_id=self.id,
            severity=Severity.INFO,
            message="All required metadata files are present.",
        )


REQUIRED_METADATA_PRESENT: Check = _RequiredMetadataPresentCheck()
registry.register(REQUIRED_METADATA_PRESENT)


# ---------------------------------------------------------------------------
# STRUCTURAL.SCHEMA_CONSISTENCY
# ---------------------------------------------------------------------------

# Map lerobot dtype strings to the Arrow data types we expect to see.
# Scalar features with shape [1] are stored as scalars in Arrow (not lists).
_LEROBOT_DTYPE_TO_ARROW: dict[str, pa.DataType] = {
    "float32": pa.float32(),
    "float64": pa.float64(),
    "int64": pa.int64(),
    "int32": pa.int32(),
    "int16": pa.int16(),
    "int8": pa.int8(),
    "uint8": pa.uint8(),
    "bool": pa.bool_(),
    "string": pa.string(),
}


class _SchemaConsistencyCheck:
    id = "STRUCTURAL.SCHEMA_CONSISTENCY"
    severity = Severity.FAIL
    category = "STRUCTURAL"
    requires_video = False

    def run(self, ds: CanonicalDataset, ctx: CheckContext) -> CheckResult:
        mismatches: list[str] = []

        for episode in ds:
            pf = ds.parquet_shard_for_episode(episode)
            actual_schema: pa.Schema = pf.schema_arrow
            actual_names = set(actual_schema.names)

            for feat_name, feat_spec in ds.features.items():
                if feat_spec.dtype == "video":
                    # Video features are not stored as columns in Parquet.
                    continue
                if feat_name not in actual_names:
                    mismatches.append(
                        f"episode {episode.episode_index}: column {feat_name!r} declared in "
                        f"info.json but absent from Parquet shard"
                    )
                    continue

                field_idx = actual_schema.get_field_index(feat_name)
                actual_type = actual_schema.field(field_idx).type
                expected_arrow_type = _LEROBOT_DTYPE_TO_ARROW.get(feat_spec.dtype)
                if expected_arrow_type is not None:
                    # For shape [1] features the value is stored as a scalar.
                    # For shape [N] (N > 1) it's stored as a list/fixed-size-list.
                    if feat_spec.shape == (1,):
                        if actual_type != expected_arrow_type:
                            mismatches.append(
                                f"episode {episode.episode_index}: column {feat_name!r} "
                                f"has dtype {actual_type!s}, expected {expected_arrow_type!s}"
                            )
                    else:
                        # Multi-dim: Arrow stores as list or fixed_size_list; just
                        # check the value_type of the list matches the scalar dtype.
                        if hasattr(actual_type, "value_type"):
                            if actual_type.value_type != expected_arrow_type:
                                mismatches.append(
                                    f"episode {episode.episode_index}: column {feat_name!r} "
                                    f"list element type {actual_type.value_type!s}, "
                                    f"expected {expected_arrow_type!s}"
                                )
                        else:
                            if actual_type != expected_arrow_type:
                                mismatches.append(
                                    f"episode {episode.episode_index}: column {feat_name!r} "
                                    f"has dtype {actual_type!s}, expected {expected_arrow_type!s}"
                                )

            # Early-exit after first shard with mismatches to keep output manageable.
            if mismatches:
                break

        if mismatches:
            return CheckResult(
                check_id=self.id,
                severity=Severity.FAIL,
                message=f"Schema inconsistencies found ({len(mismatches)} issue(s)): "
                f"{mismatches[0]}{'...' if len(mismatches) > 1 else ''}",
                details={"mismatches": mismatches},
            )
        return CheckResult(
            check_id=self.id,
            severity=Severity.INFO,
            message="All Parquet shards conform to the declared feature schema.",
        )


SCHEMA_CONSISTENCY: Check = _SchemaConsistencyCheck()
registry.register(SCHEMA_CONSISTENCY)


# ---------------------------------------------------------------------------
# STRUCTURAL.INDEX_CONTINUITY
# ---------------------------------------------------------------------------


class _IndexContinuityCheck:
    id = "STRUCTURAL.INDEX_CONTINUITY"
    severity = Severity.FAIL
    category = "STRUCTURAL"
    requires_video = False

    def run(self, ds: CanonicalDataset, ctx: CheckContext) -> CheckResult:
        violations: list[str] = []

        # Episode-level: indices must be 0..N-1 contiguously.
        ep_indices = [ep.episode_index for ep in ds]
        expected_ep_indices = list(range(len(ep_indices)))
        if ep_indices != expected_ep_indices:
            violations.append(
                f"Episode indices are not contiguous 0..{len(ep_indices) - 1}: got {ep_indices}"
            )

        # Frame-level: within each episode the Parquet's frame_index column
        # must run 0..length-1; global index must be contiguous too.
        for episode in ds:
            pf = ds.parquet_shard_for_episode(episode)
            table = pf.read(columns=["frame_index", "episode_index", "index"])  # type: ignore[no-untyped-call]

            # Filter to just this episode's rows.
            ep_mask = table.column("episode_index").to_pylist()
            ep_rows_idx = [i for i, v in enumerate(ep_mask) if v == episode.episode_index]

            if len(ep_rows_idx) == 0:
                violations.append(
                    f"Episode {episode.episode_index}: no rows found in Parquet shard"
                )
                continue

            frame_indices = [table.column("frame_index").to_pylist()[i] for i in ep_rows_idx]
            expected_frame_indices = list(range(episode.length))
            if frame_indices != expected_frame_indices:
                violations.append(
                    f"Episode {episode.episode_index}: frame_index does not run "
                    f"0..{episode.length - 1} (got "
                    f"{frame_indices[:5]}{'...' if len(frame_indices) > 5 else ''})"
                )

            global_indices = [table.column("index").to_pylist()[i] for i in ep_rows_idx]
            expected_global = list(
                range(episode.dataset_from_index, episode.dataset_from_index + episode.length)
            )
            if global_indices != expected_global:
                violations.append(
                    f"Episode {episode.episode_index}: global index column is not "
                    f"contiguous from {episode.dataset_from_index} "
                    f"(got {global_indices[:3]}{'...' if len(global_indices) > 3 else ''})"
                )

            if violations:
                break  # Report first episode with issues; don't flood output.

        if violations:
            return CheckResult(
                check_id=self.id,
                severity=Severity.FAIL,
                message=f"Index continuity violated: {violations[0]}",
                details={"violations": violations},
            )
        return CheckResult(
            check_id=self.id,
            severity=Severity.INFO,
            message="Frame and episode indices are contiguous with no gaps or duplicates.",
        )


INDEX_CONTINUITY: Check = _IndexContinuityCheck()
registry.register(INDEX_CONTINUITY)


# ---------------------------------------------------------------------------
# STRUCTURAL.METADATA_DATA_AGREEMENT
# ---------------------------------------------------------------------------
# This is THE check that catches the v2.1->v3.0 corruption bug (#2401).
# It verifies that declared from/to boundaries and episode lengths match
# the actual row counts in the data shards.


class _MetadataDataAgreementCheck:
    id = "STRUCTURAL.METADATA_DATA_AGREEMENT"
    severity = Severity.FAIL
    category = "STRUCTURAL"
    requires_video = False

    def run(self, ds: CanonicalDataset, ctx: CheckContext) -> CheckResult:
        violations: list[str] = []

        # 1. Declared num_episodes must match actual episode count.
        if ds.num_episodes != len(list(ds)):
            violations.append(
                f"Declared num_episodes={ds.num_episodes} but {len(list(ds))} episode records found"
            )

        # 2. Sum of declared lengths must equal declared num_frames (if present).
        declared_total = sum(ep.length for ep in ds)
        if ds.num_frames is not None and declared_total != ds.num_frames:
            violations.append(
                f"Sum of declared episode lengths ({declared_total}) != "
                f"declared total_frames ({ds.num_frames})"
            )

        # 3. Per-episode: actual row count in shard must equal declared length,
        #    and from/to slice boundaries must agree.
        for episode in ds:
            pf = ds.parquet_shard_for_episode(episode)
            table = pf.read(columns=["episode_index"])  # type: ignore[no-untyped-call]

            ep_col = table.column("episode_index").to_pylist()
            actual_rows = sum(1 for v in ep_col if v == episode.episode_index)

            if actual_rows != episode.length:
                violations.append(
                    f"Episode {episode.episode_index}: declared length={episode.length} "
                    f"but actual row count={actual_rows} in shard"
                )

            # from/to must span exactly `length` frames.
            declared_span = episode.dataset_to_index - episode.dataset_from_index
            if declared_span != episode.length:
                violations.append(
                    f"Episode {episode.episode_index}: "
                    f"dataset_to_index - dataset_from_index = {declared_span} "
                    f"!= declared length {episode.length} (from={episode.dataset_from_index}, "
                    f"to={episode.dataset_to_index})"
                )

            if len(violations) >= 5:
                break  # Cap output; first few failures are sufficient signal.

        if violations:
            return CheckResult(
                check_id=self.id,
                severity=Severity.FAIL,
                message=(
                    f"Metadata/data agreement violated ({len(violations)} issue(s)): "
                    f"{violations[0]}"
                ),
                details={"violations": violations},
            )
        return CheckResult(
            check_id=self.id,
            severity=Severity.INFO,
            message=(
                "Declared episode lengths and from/to boundaries agree with "
                "actual Parquet row counts."
            ),
        )


METADATA_DATA_AGREEMENT: Check = _MetadataDataAgreementCheck()
registry.register(METADATA_DATA_AGREEMENT)


# ---------------------------------------------------------------------------
# STRUCTURAL.PATH_TEMPLATE_RESOLVES
# ---------------------------------------------------------------------------


class _PathTemplateResolvesCheck:
    id = "STRUCTURAL.PATH_TEMPLATE_RESOLVES"
    severity = Severity.FAIL
    category = "STRUCTURAL"
    requires_video = False

    def run(self, ds: CanonicalDataset, ctx: CheckContext) -> CheckResult:
        root = ds.stats.root
        missing: list[str] = []

        for episode in ds:
            # Parquet shard
            try:
                pf = ds.parquet_shard_for_episode(episode)
                path = pf.metadata.row_group(0)
                # Just opening the ParquetFile already checks existence (handles.py).
                # Reaching here means the file existed and opened.
                del path
            except Exception as exc:
                missing.append(f"Episode {episode.episode_index} data shard: {exc}")

            # Video shards — one per camera.
            for camera in ds.cameras:
                try:
                    seg = ds.video_segment_for_episode(episode, camera)
                    if not seg.handle.path.is_file():
                        missing.append(
                            f"Episode {episode.episode_index} video shard "
                            f"({camera}): {seg.handle.path}"
                        )
                except Exception as exc:
                    missing.append(f"Episode {episode.episode_index} video shard ({camera}): {exc}")

            if len(missing) >= 10:
                break  # Cap; first batch is enough to diagnose.

        # Also verify meta/stats.json if present (non-fatal if absent).
        try:
            stats_path = safe_join(root, "meta", "stats.json")
            if stats_path.exists() and not stats_path.is_file():
                missing.append("meta/stats.json exists but is not a regular file")
        except Exception as exc:
            missing.append(f"meta/stats.json path check failed: {exc}")

        if missing:
            return CheckResult(
                check_id=self.id,
                severity=Severity.FAIL,
                message=f"Shard resolution failed ({len(missing)} issue(s)): {missing[0]}",
                details={"missing": missing},
            )
        return CheckResult(
            check_id=self.id,
            severity=Severity.INFO,
            message="All declared shard paths resolve to readable files.",
        )


PATH_TEMPLATE_RESOLVES: Check = _PathTemplateResolvesCheck()
registry.register(PATH_TEMPLATE_RESOLVES)
