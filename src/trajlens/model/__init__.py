"""The canonical dataset model (02_ARCHITECTURE.md §3.2).

A typed, format-neutral in-memory view built from a SourceHandle (sources/).
One adapter per format version translates that version's raw metadata into
CanonicalDataset, so every later consumer (the Check Engine, M4) targets one
shape and never branches on format version again (ADR-002).

This module only represents declared structure; it does not validate it.
Whether the declared structure agrees with the actual data -- index
continuity, timestamp spacing, stats correctness -- is the Check Engine's
job, not this one's.
"""

from trajlens.model.canonical import (
    CanonicalDataset,
    EpisodeRecord,
    FeatureSpec,
    FrameBatch,
    FrameSource,
    ShardResolver,
    VideoSegment,
)
from trajlens.model.lerobot import build_canonical_dataset
from trajlens.model.stats import StatsHandle

__all__ = [
    "CanonicalDataset",
    "EpisodeRecord",
    "FeatureSpec",
    "FrameBatch",
    "FrameSource",
    "ShardResolver",
    "StatsHandle",
    "VideoSegment",
    "build_canonical_dataset",
]
