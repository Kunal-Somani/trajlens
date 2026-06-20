"""The canonical, version-agnostic dataset model (02_ARCHITECTURE.md §3.2, ADR-002).

CanonicalDataset is the typed in-memory view every check will eventually
consume. It represents data; it does not judge it -- invariant checking
(schema/index/timestamp/stats agreement) is the Check Engine's job (M4), not
this module's. Building one never reads frame data or video bytes: only
metadata (info.json, episode records, task table) is materialized, and even
that is bounded by the same resource-bound primitives the source layer uses
(sources/bounds.py), independent of what a dataset claims about itself.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import pyarrow.parquet as pq

from trajlens.model.stats import StatsHandle
from trajlens.sources.handles import VideoShardHandle
from trajlens.sources.version import DatasetVersion


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """A single declared feature from info.json's features map."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    names: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """One episode's declared metadata, uniform across format versions.

    dataset_from_index/dataset_to_index are global frame offsets into the
    dataset's tabular data, taken verbatim from v3.0 episode metadata or
    derived from the cumulative sum of declared lengths for v2.x, where the
    format does not store them explicitly. Either way this is what the
    *metadata* declares -- whether it agrees with the actual data shard row
    counts is a Check Engine question (03_DATA_FORMAT_SPEC.md invariant 3),
    not this model's.
    """

    episode_index: int
    length: int
    tasks: tuple[str, ...]
    dataset_from_index: int
    dataset_to_index: int


@dataclass(frozen=True, slots=True)
class VideoSegment:
    """A lazy handle to the video shard covering one episode's camera feed."""

    handle: VideoShardHandle
    from_timestamp: float
    to_timestamp: float


class ShardResolver(Protocol):
    """Version-specific lookup of which shard file holds an episode's payload.

    Implemented once per format version (model/adapters.py) so CanonicalDataset
    itself stays free of version branching.
    """

    def parquet_shard(self, episode: EpisodeRecord) -> pq.ParquetFile: ...

    def video_segment(self, episode: EpisodeRecord, camera: str) -> VideoSegment: ...


@dataclass(frozen=True, slots=True)
class CanonicalDataset:
    """Typed, version-agnostic view of a LeRobotDataset's declared structure.

    Iterating yields EpisodeRecords only -- frame data and video bytes are
    never touched until parquet_shard_for_episode/video_segment_for_episode
    is called for a specific episode, and even then only a handle is
    returned, per 02_ARCHITECTURE.md §3.2 ("the model never materializes full
    video into memory; it yields handles").
    """

    version: DatasetVersion
    fps: int
    features: Mapping[str, FeatureSpec]
    num_episodes: int
    num_frames: int | None
    task_table: Mapping[int, str]
    cameras: tuple[str, ...]
    stats: StatsHandle
    _episodes: Sequence[EpisodeRecord]
    _resolver: ShardResolver

    def __len__(self) -> int:
        return len(self._episodes)

    def __iter__(self) -> Iterator[EpisodeRecord]:
        return iter(self._episodes)

    def episode(self, episode_index: int) -> EpisodeRecord:
        """Return the EpisodeRecord at *episode_index*, raising IndexError if out of range."""
        return self._episodes[episode_index]

    def parquet_shard_for_episode(self, episode: EpisodeRecord) -> pq.ParquetFile:
        """Return a lazy handle to the Parquet shard holding *episode*'s frame data."""
        return self._resolver.parquet_shard(episode)

    def video_segment_for_episode(self, episode: EpisodeRecord, camera: str) -> VideoSegment:
        """Return a lazy handle to the video segment for *episode* on *camera*."""
        return self._resolver.video_segment(episode, camera)
