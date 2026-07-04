"""REPAIR.STATS_RECOMPUTE — fixer for STATISTICAL.STATS_MATCH_DATA.

Detection counterpart: src/trajlens/checks/statistical.py _StatsMatchDataCheck.

The fix: restream all data shards via Welford's online algorithm (same path
the check uses) and rewrite meta/stats.json with the recomputed mean, std, min,
max, and count for every numeric feature.  Only meta/stats.json is touched;
data shards are never opened in write mode.

ADR-004 requirements satisfied here:
  - Copy-on-write: source is never opened in write mode; output_path receives
    the corrected stats.json.
  - Dry-run by default: dry_run() computes the StatsDiff with zero filesystem
    writes.
  - Round-trip: tests in tests/unit/test_stats_recompute.py and
    tests/property/test_repair_properties.py verify repair → re-lint → INFO.

Scope: global meta/stats.json only.  Per-episode stats (inline in
meta/episodes/*.parquet for v3.0, episodes_stats.jsonl for v2.1) are NOT
rewritten by this fixer.  They are targeted by a separate fixer once this one
is stable (per the "small changes" rule in the work log).
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import structlog

from trajlens.checks.statistical import _STATS_RTOL as _CHECK_RTOL
from trajlens.checks.statistical import _relative_error, _stream_feature_columns
from trajlens.errors import RepairError
from trajlens.model.canonical import CanonicalDataset
from trajlens.repair.protocol import Diff, RepairSummary, StatChange
from trajlens.sources.paths import safe_join

log = structlog.get_logger(__name__)

FIXER_ID = "REPAIR.STATS_RECOMPUTE"
CHECK_ID = "STATISTICAL.STATS_MATCH_DATA"

# Relative path components to meta/stats.json — matches StatsHandle.STATS_RELATIVE_PATH.
_STATS_REL: tuple[str, ...] = ("meta", "stats.json")


def _dataset_root(ds: CanonicalDataset) -> Path:
    return ds.stats.root


def _load_stored_stats(root: Path) -> dict[str, Any] | None:
    """Load and return the raw stats dict, or None if the file is absent."""
    path = safe_join(root, *_STATS_REL)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RepairError(f"meta/stats.json is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise RepairError(
            f"meta/stats.json must be a JSON object keyed by feature name; got {type(raw).__name__}"
        )
    return raw


def _recompute_stats(ds: CanonicalDataset) -> dict[str, dict[str, float]]:
    """Stream all data shards once via Welford and return per-feature stats.

    Returns a dict keyed by feature name, each value a dict with keys
    mean, std, min, max, count — the same five fields the lerobot 0.5.2
    writer stores in stats.json.
    """
    accumulators = _stream_feature_columns(ds)
    result: dict[str, dict[str, float]] = {}
    for feat_name, acc in accumulators.items():
        result[feat_name] = {
            "mean": acc.mean,
            "std": acc.std,
            "min": acc.min if not math.isinf(acc.min) else 0.0,
            "max": acc.max if not math.isinf(acc.max) else 0.0,
            "count": float(acc.count),
        }
    return result


class StatsRecomputeFixer:
    """Recompute global meta/stats.json from data shards (ADR-004).

    dry_run() streams all shards once via Welford, diffs against the stored
    stats.json using the same tolerance STATISTICAL.STATS_MATCH_DATA uses,
    and returns a Diff of every mean/std field that deviates beyond that
    tolerance.  apply() copies the source tree to output_path and writes
    the corrected stats.json; no data or video shards are touched.

    Only v3.0 datasets are in scope (v2.x Hub datasets cannot be lazily
    streamed per adapters.py _build_v2 guard).  A RepairError is raised for
    inputs without a stats.json, because writing one from scratch would be
    out of scope for a *repair* operation on a file that is expected to exist.
    """

    fixer_id: str = FIXER_ID
    check_id: str = CHECK_ID

    def dry_run(self, ds: CanonicalDataset) -> Diff:
        """Compute what would change without writing anything.

        Returns a Diff recording every mean/std entry in stats.json that
        deviates from the Welford-recomputed value beyond _CHECK_RTOL.
        Returns a no-op Diff when the dataset is already within tolerance.
        """
        _check_preconditions(ds)
        root = _dataset_root(ds)
        stored = _load_stored_stats(root)

        # Single streaming pass — single-writer assumption: source directory
        # must not be mutated by another process between dry_run() and apply().
        recomputed = _recompute_stats(ds)

        changes: list[StatChange] = []
        for feat_name, new_stats in recomputed.items():
            feat_stored = stored.get(feat_name) if stored else None
            if feat_stored is None or not isinstance(feat_stored, dict):
                continue

            for stat_key in ("mean", "std", "min", "max", "count"):
                stored_val_raw = feat_stored.get(stat_key)
                if stored_val_raw is None:
                    continue
                try:
                    stored_val = float(stored_val_raw)
                except (TypeError, ValueError):
                    continue
                new_val = new_stats[stat_key]
                # count is always an exact integer — any discrepancy is a bug,
                # and relative tolerance would mis-fire on tiny counts.
                if stat_key == "count":
                    deviated = stored_val != new_val
                else:
                    deviated = _relative_error(stored_val, new_val) > _CHECK_RTOL
                if deviated:
                    changes.append(
                        StatChange(
                            feature=feat_name,
                            stat_key=stat_key,
                            old_value=stored_val,
                            new_value=new_val,
                        )
                    )

        diff = Diff(changes=tuple(changes), check_id=CHECK_ID, fixer_id=FIXER_ID)
        if diff.is_noop:
            log.info(
                "stats_recompute.dry_run.noop",
                reason="all stats.json entries match Welford-recomputed values within tolerance",
            )
        else:
            log.info(
                "stats_recompute.dry_run.changes_found",
                num_changes=len(changes),
                features_affected=len({c.feature for c in changes}),
            )
        return diff

    def apply(self, ds: CanonicalDataset, output_path: Path) -> RepairSummary:
        """Write a corrected copy of *ds* to *output_path* (copy-on-write).

        Steps:
          1. Precondition check.
          2. dry_run() to compute the Diff.
          3. Copy the source tree to output_path.
          4. Rewrite meta/stats.json with fully recomputed values.

        Only meta/stats.json is rewritten.  Data shards and video files are
        never opened in write mode.  *output_path* must not be the source root.
        Raises RepairError on any unrecoverable condition.
        """
        _check_preconditions(ds)

        source_root = _dataset_root(ds)
        if output_path.resolve() == source_root.resolve():
            raise RepairError(
                "output_path must not be the source dataset root (ADR-004 "
                f"copy-on-write). Got: {output_path}"
            )

        diff = self.dry_run(ds)

        log.info(
            "stats_recompute.apply.start",
            source=str(source_root),
            output=str(output_path),
            num_changes=len(diff.changes),
        )

        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(source_root, output_path)

        if diff.is_noop:
            log.info("stats_recompute.apply.noop", output=str(output_path))
            return RepairSummary(
                output_path=output_path,
                changes_written=0,
                frames_corrected=0,
            )

        _rewrite_stats_json(ds, output_path)

        stat_changes = [c for c in diff.changes if isinstance(c, StatChange)]
        log.info(
            "stats_recompute.apply.done",
            output=str(output_path),
            features_rewritten=len({c.feature for c in stat_changes}),
            stat_entries_corrected=len(diff.changes),
        )
        return RepairSummary(
            output_path=output_path,
            changes_written=1,
            frames_corrected=len(diff.changes),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_preconditions(ds: CanonicalDataset) -> None:
    """Raise RepairError for any condition that makes repair unsafe or undefined."""
    from trajlens.sources.version import DatasetVersion

    if ds.version is not DatasetVersion.V3_0:
        raise RepairError(
            f"StatsRecomputeFixer only supports v3.0 datasets; "
            f"got version {ds.version!r}. For v2.x datasets, convert to v3.0 first."
        )
    root = _dataset_root(ds)
    stats_path = safe_join(root, *_STATS_REL)
    if not stats_path.is_file():
        raise RepairError(
            "StatsRecomputeFixer requires an existing meta/stats.json to repair; "
            "no such file found. Creating stats.json from scratch is out of scope."
        )


def _rewrite_stats_json(ds: CanonicalDataset, output_root: Path) -> None:
    """Recompute stats and write them to output_root/meta/stats.json.

    Preserves all existing feature entries in stats.json (including min, max,
    count fields) and overwrites only mean and std with the Welford-recomputed
    values.  Features present in stats.json but absent from the data stream
    are left untouched.  Features present in the data but absent from stats.json
    are added with all five fields (mean, std, min, max, count).
    """
    source_root = _dataset_root(ds)
    stored = _load_stored_stats(source_root)
    recomputed = _recompute_stats(ds)

    # Start from the stored dict to preserve fields we don't touch.
    new_stats: dict[str, Any] = dict(stored) if stored else {}

    for feat_name, new_vals in recomputed.items():
        if feat_name in new_stats and isinstance(new_stats[feat_name], dict):
            # Update all five fields so the file is fully consistent.
            entry: dict[str, Any] = dict(new_stats[feat_name])
            entry["mean"] = new_vals["mean"]
            entry["std"] = new_vals["std"]
            entry["min"] = new_vals["min"]
            entry["max"] = new_vals["max"]
            entry["count"] = new_vals["count"]
            new_stats[feat_name] = entry
        else:
            # Feature missing from stats.json — add it in full.
            new_stats[feat_name] = new_vals

    out_path = safe_join(output_root, *_STATS_REL)
    out_path.write_text(json.dumps(new_stats, indent=2))
    log.debug("stats_recompute.stats_json_written", path=str(out_path))
