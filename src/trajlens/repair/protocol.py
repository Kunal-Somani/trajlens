"""Repair Protocol, Diff, and FrameChange types (02_ARCHITECTURE.md §3.5, ADR-004).

Every fixer implements the Fixer Protocol.  The Diff type is the structured
description of what a fixer *would* change; it is produced by dry_run() and
consumed by apply() internally.  Both types live in a leaf module with no
internal fixer imports so every repair/ module can import from here without
cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from trajlens.model.canonical import CanonicalDataset


@dataclass(frozen=True, slots=True)
class FrameChange:
    """A single frame-level change produced by a fixer's dry run.

    episode_index  — which episode contains the changed frame.
    frame_index    — within-episode index of the frame.
    shard_path     — dataset-root-relative path to the Parquet shard.
    column         — the column that would be rewritten.
    old_value      — the value currently stored on disk.
    new_value      — the corrected value the fixer would write.
    """

    episode_index: int
    frame_index: int
    shard_path: str
    column: str
    old_value: float
    new_value: float


@dataclass(frozen=True, slots=True)
class StatChange:
    """A single feature-level stats change produced by a stats fixer's dry run.

    feature        — the feature name whose stat entry would be rewritten.
    stat_key       — the stat field being corrected (e.g. "mean", "std", "min", "max").
    old_value      — the value currently stored in stats.json.
    new_value      — the corrected value the fixer would write.
    """

    feature: str
    stat_key: str
    old_value: float
    new_value: float


@dataclass(frozen=True, slots=True)
class BoundaryChange:
    """A single episode-boundary change produced by a fixer's dry run.

    episode_index  — which episode's declared metadata would be rewritten.
    field          — the metadata field being corrected (e.g. "dataset_from_index",
                     "dataset_to_index", "length").
    old_value      — the value currently stored in episode metadata.
    new_value      — the corrected value the fixer would write, derived from
                     the actual data (never the reverse).
    """

    episode_index: int
    field: str
    old_value: int
    new_value: int


@dataclass(frozen=True, slots=True)
class Diff:
    """Structured description of all changes a fixer would make.

    changes        — ordered list of per-frame, per-stat, or per-boundary
                     changes; empty when the dataset is already clean and the
                     fixer is a no-op.
    check_id       — the check this fixer targets (e.g. "KNOWNBUG.TIMESTAMP_DRIFT").
    fixer_id       — stable identifier for the fixer.
    is_noop        — True when changes is empty; the fixer has nothing to do.
    """

    changes: tuple[FrameChange | StatChange | BoundaryChange, ...]
    check_id: str
    fixer_id: str

    @property
    def is_noop(self) -> bool:
        return len(self.changes) == 0


@dataclass(frozen=True, slots=True)
class RepairSummary:
    """What apply() reports after writing the corrected dataset.

    output_path    — the directory written (never the source path).
    changes_written — number of shard files rewritten.
    frames_corrected — total frame-level changes applied.
    """

    output_path: Path
    changes_written: int
    frames_corrected: int


class Fixer(Protocol):
    """The Fixer Protocol every repair module must satisfy.

    fixer_id       — stable dot-namespaced identifier (e.g. "REPAIR.TIMESTAMP_DEDRIFT").
    check_id       — the check whose findings this fixer clears.

    dry_run() computes all changes without touching the filesystem and returns
    a Diff.  It must never raise; on unrecoverable error raise RepairError.

    apply() reads from *source* (never mutated), writes corrected shards to
    *output_path*, and returns a RepairSummary.  output_path must not be
    the source dataset root (ADR-004 copy-on-write guarantee).
    """

    fixer_id: str
    check_id: str

    def dry_run(self, ds: CanonicalDataset) -> Diff: ...

    def apply(self, ds: CanonicalDataset, output_path: Path) -> RepairSummary: ...
