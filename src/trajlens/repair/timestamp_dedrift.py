"""REPAIR.TIMESTAMP_DEDRIFT — fixer for KNOWNBUG.TIMESTAMP_DRIFT (LeRobot issue #3177).

Detection counterpart: src/trajlens/checks/temporal.py _TimestampDriftCheck.

The fix: rewrite each frame's stored timestamp to the ideal value
``float32(frame_index / fps)`` (or float64, matching the declared dtype).
This is the value the check compares against; after repair, cumulative drift
is exactly zero, guaranteeing KNOWNBUG.TIMESTAMP_DRIFT yields INFO on re-lint.

ADR-004 requirements satisfied here:
  - Copy-on-write: source is never opened in write mode; output_path receives
    the corrected shard(s).
  - Dry-run by default: dry_run() computes the Diff with zero filesystem writes.
  - Round-trip: tests in tests/unit/test_timestamp_dedrift.py and
    tests/property/test_repair_properties.py verify repair → re-lint → INFO.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from trajlens.errors import RepairError
from trajlens.model.canonical import CanonicalDataset
from trajlens.repair.protocol import Diff, FrameChange, RepairSummary
from trajlens.sources.paths import safe_join

log = structlog.get_logger(__name__)

# Shared with _TimestampDriftCheck — never tighten without re-checking the
# decoder's actual tolerance (temporal.py module docstring has the citation).
_DECODER_TOLERANCE_S: float = 1e-4

FIXER_ID = "REPAIR.TIMESTAMP_DEDRIFT"
CHECK_ID = "KNOWNBUG.TIMESTAMP_DRIFT"

# v3.0 data shards live under data/chunk-*/file-*.parquet — this glob pattern
# is grounded in the live lerobot 0.5.2 writer (dataset_writer.py) and the
# fixture builders (tests/fixtures/builders.py _V3Resolver.parquet_shard).
_DATA_SHARD_GLOB = "data/chunk-*/file-*.parquet"


def _quantize_type(ds: CanonicalDataset) -> type[np.float32] | type[np.float64]:
    """Return the numpy scalar type matching the declared timestamp dtype.

    Replicates _TimestampDriftCheck's quantization logic exactly so the
    repaired values pass the check without residual drift.
    """
    declared = ds.features.get("timestamp")
    if declared is not None and declared.dtype == "float64":
        return np.float64
    return np.float32


def _ideal_ts(frame_index: int, fps: int, quantize: type[np.float32] | type[np.float64]) -> float:
    """Return the ideal timestamp for *frame_index*, quantized to the storage dtype."""
    return float(quantize(frame_index / fps))


def _dataset_root(ds: CanonicalDataset) -> Path:
    """Return the dataset root Path from the CanonicalDataset's stats handle."""
    return ds.stats.root


class TimestampDedriftFixer:
    """Recompute all timestamps as ``quantize(frame_index / fps)`` (ADR-004).

    dry_run() produces a Diff of every frame whose stored timestamp differs
    from the ideal value, grouped by Parquet shard path.  apply() copies the
    source tree to *output_path* and rewrites only the affected shard files.

    Only v3.0 datasets are supported because v2.x shard resolution requires
    globbing inside the source root, making copy-on-write complicated, and
    v2.x Hub datasets cannot be lazily streamed (adapters.py _build_v2 guard).
    A RepairError is raised for v2.x inputs.
    """

    fixer_id: str = FIXER_ID
    check_id: str = CHECK_ID

    def dry_run(self, ds: CanonicalDataset) -> Diff:
        """Compute what would change without writing anything.

        Returns a Diff recording every frame whose stored timestamp deviates
        from the ideal ``quantize(frame_index/fps)``.  Returns a no-op Diff if
        the dataset is already drift-free.
        """
        _check_preconditions(ds)

        quantize = _quantize_type(ds)
        fps = ds.fps
        root = _dataset_root(ds)
        changes: list[FrameChange] = []

        # Single glob here — single-writer assumption: source directory must not
        # be mutated by another process between dry_run() and apply().
        shard_paths = sorted(root.glob(_DATA_SHARD_GLOB))
        for shard_abs in shard_paths:
            shard_rel = str(shard_abs.relative_to(root))
            table = pq.read_table(shard_abs)  # type: ignore[no-untyped-call]
            ep_col: list[Any] = table.column("episode_index").to_pylist()
            fi_col: list[Any] = table.column("frame_index").to_pylist()
            ts_col: list[Any] = table.column("timestamp").to_pylist()

            for ep, fi, ts in zip(ep_col, fi_col, ts_col, strict=True):
                frame_index = int(fi)
                stored_ts = float(ts)
                ideal = _ideal_ts(frame_index, fps, quantize)
                if stored_ts != ideal:
                    changes.append(
                        FrameChange(
                            episode_index=int(ep),
                            frame_index=frame_index,
                            shard_path=shard_rel,
                            column="timestamp",
                            old_value=stored_ts,
                            new_value=ideal,
                        )
                    )

        diff = Diff(changes=tuple(changes), check_id=CHECK_ID, fixer_id=FIXER_ID)
        if diff.is_noop:
            log.info(
                "timestamp_dedrift.dry_run.noop",
                reason="all timestamps already match ideal frame_index/fps values",
            )
        else:
            log.info(
                "timestamp_dedrift.dry_run.changes_found",
                num_changes=len(changes),
                num_shards=len({c.shard_path for c in changes}),
            )
        return diff

    def apply(self, ds: CanonicalDataset, output_path: Path) -> RepairSummary:
        """Write a corrected copy of *ds* to *output_path* (copy-on-write).

        Steps:
          1. Precondition check (version, feature presence, fps).
          2. dry_run() to compute the Diff.
          3. Copy the source tree to output_path.
          4. For each affected shard, read → patch timestamp column → write.

        *output_path* must not be the source dataset root.  Raises RepairError
        on any unrecoverable condition; the copy-on-write guarantee holds
        because source shards are never opened in write mode.
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
            "timestamp_dedrift.apply.start",
            source=str(source_root),
            output=str(output_path),
            num_changes=len(diff.changes),
        )

        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(source_root, output_path)

        if diff.is_noop:
            log.info("timestamp_dedrift.apply.noop", output=str(output_path))
            return RepairSummary(
                output_path=output_path,
                changes_written=0,
                frames_corrected=0,
            )

        shards_written = _rewrite_shards(diff, output_root=output_path)

        log.info(
            "timestamp_dedrift.apply.done",
            output=str(output_path),
            shards_written=shards_written,
            frames_corrected=len(diff.changes),
        )
        return RepairSummary(
            output_path=output_path,
            changes_written=shards_written,
            frames_corrected=len(diff.changes),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_preconditions(ds: CanonicalDataset) -> None:
    """Raise RepairError for any condition that makes repair unsafe or undefined."""
    if ds.format_id != "lerobot" or ds.format_version != "3.0":
        raise RepairError(
            f"TimestampDedriftFixer only supports v3.0 datasets; "
            f"got {ds.format_label}. For v2.x datasets, convert to v3.0 first."
        )
    if "frame_index" not in ds.features:
        raise RepairError(
            "TimestampDedriftFixer requires a bare 'frame_index' feature in info.json. "
            "Multi-camera datasets that namespace it (e.g. 'frame_index.<camera>') "
            "are not supported."
        )
    if ds.fps <= 0:
        raise RepairError(f"Dataset fps must be > 0 to compute ideal timestamps; got fps={ds.fps}.")


def _rewrite_shards(diff: Diff, *, output_root: Path) -> int:
    """Rewrite affected Parquet shards in *output_root* with corrected timestamps.

    Groups changes by relative shard_path, reads each output shard once,
    patches the timestamp column via set_column, and writes back.  Source
    shards are never touched.  Returns the number of distinct shard files
    rewritten.
    """
    by_shard: dict[str, list[FrameChange]] = {}
    for change in diff.changes:
        assert isinstance(change, FrameChange), f"unexpected change type: {type(change)}"
        by_shard.setdefault(change.shard_path, []).append(change)

    shards_written = 0
    for rel_shard_path, shard_changes in by_shard.items():
        rel_parts = Path(rel_shard_path).parts
        output_shard = safe_join(output_root, *rel_parts)

        patch: dict[tuple[int, int], float] = {
            (c.episode_index, c.frame_index): c.new_value for c in shard_changes
        }

        table = pq.read_table(output_shard)  # type: ignore[no-untyped-call]
        ts_list: list[Any] = table.column("timestamp").to_pylist()
        ep_list: list[Any] = table.column("episode_index").to_pylist()
        fi_list: list[Any] = table.column("frame_index").to_pylist()

        for row_idx in range(len(ts_list)):
            key = (int(ep_list[row_idx]), int(fi_list[row_idx]))
            if key in patch:
                ts_list[row_idx] = patch[key]

        ts_dtype = table.schema.field("timestamp").type
        new_ts_col = pa.array(ts_list, type=ts_dtype)
        new_table = table.set_column(
            table.schema.get_field_index("timestamp"),
            "timestamp",
            new_ts_col,
        )
        pq.write_table(new_table, output_shard)  # type: ignore[no-untyped-call]
        shards_written += 1
        log.debug(
            "timestamp_dedrift.shard_rewritten",
            shard=str(output_shard),
            rows_patched=len(shard_changes),
        )

    return shards_written
