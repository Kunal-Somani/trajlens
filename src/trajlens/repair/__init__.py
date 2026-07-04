"""Repair engine — fixers that correct identified dataset issues (ADR-004).

Each fixer implements the Fixer Protocol (repair/protocol.py): dry_run()
returns a Diff describing what would change; apply() performs a copy-on-write
repair to an explicit output path.
"""

from trajlens.repair.protocol import Diff, Fixer, FrameChange, RepairSummary, StatChange
from trajlens.repair.stats_recompute import StatsRecomputeFixer
from trajlens.repair.timestamp_dedrift import TimestampDedriftFixer

__all__ = [
    "Diff",
    "Fixer",
    "FrameChange",
    "RepairSummary",
    "StatChange",
    "StatsRecomputeFixer",
    "TimestampDedriftFixer",
]
