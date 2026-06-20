"""The canonical dataset model (02_ARCHITECTURE.md §3.2).

A typed, version-agnostic in-memory view built from a SourceHandle (sources/).
One adapter per format version translates that version's raw metadata into
CanonicalDataset, so every later consumer (the Check Engine, M4) targets one
shape and never branches on DatasetVersion again (ADR-002).

This module only represents declared structure; it does not validate it.
Whether the declared structure agrees with the actual data -- index
continuity, timestamp spacing, stats correctness -- is the Check Engine's
job, not this one's.
"""

from trajlens.model.adapters import build_canonical_dataset
from trajlens.model.canonical import (
    CanonicalDataset,
    EpisodeRecord,
    FeatureSpec,
    ShardResolver,
    VideoSegment,
)
from trajlens.model.stats import StatsHandle

__all__ = [
    "CanonicalDataset",
    "EpisodeRecord",
    "FeatureSpec",
    "ShardResolver",
    "StatsHandle",
    "VideoSegment",
    "build_canonical_dataset",
]
