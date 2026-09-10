"""REPAIR.TASK_INDEX_REPAIR — fixer for SEMANTIC.TASK_INTEGRITY (dangling task_index).

Detection counterpart: src/trajlens/checks/semantic.py _TaskIntegrityCheck.

Ground truth: meta/tasks.parquet (loaded into CanonicalDataset.task_table).
A dangling task_index is a value referenced by a frame data row that is not
a key in task_table.

The "nearest valid task" heuristic (stated once here, restated in the PR
body per the module's own instruction): for a dangling task_index N, the
fixer reassigns it to the defined task_index whose integer value has the
minimum absolute distance to N, i.e. argmin(|defined_index - N|) over
task_table.keys(). Ties (two or more defined indices equidistant from N) are
ambiguous — there is no principled way to prefer one over the other from the
task table alone — so the fixer refuses rather than picking arbitrarily.

This is a bookkeeping repair only, consistent with the project's permanent
repair philosophy (08_ROADMAP.md preamble): it reassigns a dangling
*reference* to an existing, already-defined task. It never invents a task
description, never edits tasks.parquet, and never guesses which real-world
task a frame actually belongs to — that association is unrecoverable from a
dangling integer alone. If no defined task exists at all (empty task_table),
there is nothing to reassign to, so the fixer refuses.

ADR-004 requirements satisfied here:
  - Copy-on-write: source is never opened in write mode; output_path
    receives the corrected shard(s).
  - Dry-run by default: dry_run() computes the Diff with zero filesystem
    writes.
  - Round-trip: tests in tests/unit/test_task_index_repair.py and
    tests/property/test_task_index_repair_properties.py verify
    repair -> re-lint -> INFO on SEMANTIC.TASK_INTEGRITY.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from trajlens.errors import RepairError
from trajlens.model.canonical import CanonicalDataset
from trajlens.repair.protocol import Diff, FrameChange, RepairSummary
from trajlens.sources.paths import safe_join

log = structlog.get_logger(__name__)

FIXER_ID = "REPAIR.TASK_INDEX_REPAIR"
CHECK_ID = "SEMANTIC.TASK_INTEGRITY"

# v3.0 data shards live under data/chunk-*/file-*.parquet — same glob
# timestamp_dedrift.py uses, grounded in the same live lerobot 0.5.2 writer
# and fixture builders (tests/fixtures/builders.py).
_DATA_SHARD_GLOB = "data/chunk-*/file-*.parquet"


def _dataset_root(ds: CanonicalDataset) -> Path:
    return ds.stats.root


def _nearest_valid_task(dangling: int, defined_indices: list[int]) -> int:
    """Return the defined task_index nearest to *dangling* in index space.

    Raises RepairError if two or more defined indices are equidistant (an
    ambiguous reassignment) — see module docstring for the full heuristic
    statement.
    """
    distances = sorted((abs(d - dangling), d) for d in defined_indices)
    best_distance = distances[0][0]
    candidates = [d for dist, d in distances if dist == best_distance]
    if len(candidates) > 1:
        raise RepairError(
            f"task_index {dangling} is equidistant from {len(candidates)} defined task "
            f"indices {sorted(candidates)}; no unambiguous nearest-task reassignment "
            "exists. Refusing to guess which task this frame belongs to."
        )
    return candidates[0]


class TaskIndexRepairFixer:
    """Reassign dangling task_index values to the nearest defined task (ADR-004).

    dry_run() produces a Diff of every frame whose task_index is not a key
    in the dataset's task_table, grouped by Parquet shard path. apply()
    copies the source tree to *output_path* and rewrites only the affected
    shard files.

    Only v3.0 datasets are supported: task_index is a per-frame data-shard
    column only in v3.0 (v2.x resolves tasks per-episode via episodes.jsonl,
    not via a frame-level task_index column trajlens can safely rewrite).
    A RepairError is raised for v2.x inputs.
    """

    fixer_id: str = FIXER_ID
    check_id: str = CHECK_ID
    writable_formats: frozenset[str] = frozenset({"lerobot"})

    def dry_run(self, ds: CanonicalDataset) -> Diff:
        """Compute what would change without writing anything.

        Returns a Diff recording every frame whose task_index is dangling
        (not defined in ds.task_table), reassigned to the nearest defined
        task_index. Returns a no-op Diff if every reference already
        resolves. Raises RepairError if task_table is empty and at least
        one dangling reference exists (nothing to reassign to), or if any
        dangling reference has no unambiguous nearest task.
        """
        _check_preconditions(ds)

        defined_indices = list(ds.task_table.keys())
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
            ti_col: list[Any] = table.column("task_index").to_pylist()

            for ep, fi, ti in zip(ep_col, fi_col, ti_col, strict=True):
                task_index = int(ti)
                if task_index in ds.task_table:
                    continue

                if not defined_indices:
                    raise RepairError(
                        f"task_index {task_index} is dangling (episode {int(ep)}, frame "
                        f"{int(fi)}) but meta/tasks.parquet defines no tasks at all. "
                        "There is no valid task to reassign this frame to."
                    )

                nearest = _nearest_valid_task(task_index, defined_indices)
                changes.append(
                    FrameChange(
                        episode_index=int(ep),
                        frame_index=int(fi),
                        shard_path=shard_rel,
                        column="task_index",
                        old_value=task_index,
                        new_value=nearest,
                    )
                )

        diff = Diff(changes=tuple(changes), check_id=CHECK_ID, fixer_id=FIXER_ID)
        if diff.is_noop:
            log.info(
                "task_index_repair.dry_run.noop",
                reason="every task_index reference already resolves in the task table",
            )
        else:
            log.info(
                "task_index_repair.dry_run.changes_found",
                num_changes=len(changes),
                num_shards=len({c.shard_path for c in changes}),
            )
        return diff

    def apply(self, ds: CanonicalDataset, output_path: Path) -> RepairSummary:
        """Write a corrected copy of *ds* to *output_path* (copy-on-write).

        Steps:
          1. Precondition check (version, task_index feature presence).
          2. dry_run() to compute the Diff.
          3. Copy the source tree to output_path.
          4. For each affected shard, read → patch task_index column → write.

        *output_path* must not be the source dataset root. Raises RepairError
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
            "task_index_repair.apply.start",
            source=str(source_root),
            output=str(output_path),
            num_changes=len(diff.changes),
        )

        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(source_root, output_path)

        if diff.is_noop:
            log.info("task_index_repair.apply.noop", output=str(output_path))
            return RepairSummary(
                output_path=output_path,
                changes_written=0,
                frames_corrected=0,
            )

        shards_written = _rewrite_shards(diff, output_root=output_path)

        log.info(
            "task_index_repair.apply.done",
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
            f"TaskIndexRepairFixer only supports v3.0 datasets; "
            f"got {ds.format_label}. For v2.x datasets, convert to v3.0 first."
        )
    if "task_index" not in ds.features:
        raise RepairError("TaskIndexRepairFixer requires a 'task_index' feature in info.json.")


def _rewrite_shards(diff: Diff, *, output_root: Path) -> int:
    """Rewrite affected Parquet shards in *output_root* with corrected task_index.

    Groups changes by relative shard_path, reads each output shard once,
    patches the task_index column via set_column, and writes back. Source
    shards are never touched. Returns the number of distinct shard files
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

        patch: dict[tuple[int, int], int] = {
            (c.episode_index, c.frame_index): int(c.new_value) for c in shard_changes
        }

        table = pq.read_table(output_shard)  # type: ignore[no-untyped-call]
        ti_list: list[Any] = table.column("task_index").to_pylist()
        ep_list: list[Any] = table.column("episode_index").to_pylist()
        fi_list: list[Any] = table.column("frame_index").to_pylist()

        for row_idx in range(len(ti_list)):
            key = (int(ep_list[row_idx]), int(fi_list[row_idx]))
            if key in patch:
                ti_list[row_idx] = patch[key]

        ti_dtype = table.schema.field("task_index").type
        new_ti_col = pa.array(ti_list, type=ti_dtype)
        new_table = table.set_column(
            table.schema.get_field_index("task_index"),
            "task_index",
            new_ti_col,
        )
        pq.write_table(new_table, output_shard)  # type: ignore[no-untyped-call]
        shards_written += 1
        log.debug(
            "task_index_repair.shard_rewritten",
            shard=str(output_shard),
            rows_patched=len(shard_changes),
        )

    return shards_written
