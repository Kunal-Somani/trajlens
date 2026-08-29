"""LeRobot v3.0 adapter -- translates the sharded raw metadata into a CanonicalDataset.

Split out of model/adapters.py (v0.5 M1-B) so each format version's loading
logic lives in its own module; model/lerobot.py is the entry point that
branches between this module and lerobot_v2.py. Per-version path templates
and column names are grounded in the live lerobot 0.5.2 source (commit
8515d456), not the data format spec's docstring-derived paraphrase -- see
model/__init__.py module docstring for the discrepancy this caught.

Reading here is bounded independent of what info.json declares: the actual
episode-metadata records are counted as they're read and checked against
MAX_DECLARED_EPISODES, because a dataset's episode metadata could in
principle disagree with its own total_episodes (T2 in the threat model).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from trajlens.errors import DatasetFormatError
from trajlens.model.canonical import (
    CanonicalDataset,
    EpisodeRecord,
    FeatureSpec,
    FrameBatch,
    FrameSource,
    VideoSegment,
)
from trajlens.model.lerobot_v2 import (
    _arrow_schema_to_features,
    _camera_keys,
    _parse_features,
    _read_parquet_table,
)
from trajlens.model.stats import StatsHandle
from trajlens.sources.bounds import MAX_DECLARED_EPISODES, check_resource_bound
from trajlens.sources.loader import SourceHandle
from trajlens.sources.paths import safe_join

V3_EPISODES_DIR = ("meta", "episodes")
V3_TASKS_PATH = ("meta", "tasks.parquet")


@dataclass(frozen=True, slots=True)
class LeRobotFrameSourceV3:
    """FrameSource over one v3.0 episode's slice of a (possibly shared) shard.

    v3.0 data shards can hold multiple episodes' frames concatenated, so
    unlike v2.x this filters each row group to the rows whose global "index"
    column falls in [from_index, to_index) -- the same global frame offsets
    EpisodeRecord.dataset_from_index/dataset_to_index declare.
    """

    parquet_file: pq.ParquetFile
    from_index: int
    to_index: int
    length: int

    def schema(self) -> Mapping[str, FeatureSpec]:
        return _arrow_schema_to_features(self.parquet_file.schema_arrow)

    def num_rows(self) -> int:
        return self.length

    def iter_batches(self, columns: Sequence[str] | None = None) -> Iterator[FrameBatch]:
        cols = list(columns) if columns is not None else None
        read_cols = cols if cols is None or "index" in cols else [*cols, "index"]
        for rg_idx in range(self.parquet_file.num_row_groups):
            table = self.parquet_file.read_row_group(  # type: ignore[no-untyped-call]
                rg_idx, columns=read_cols
            )
            index_col = table.column("index").to_numpy()
            mask = (index_col >= self.from_index) & (index_col < self.to_index)
            if not mask.any():
                continue
            filtered = table.filter(pa.array(mask))
            keep = cols if cols is not None else filtered.column_names
            arrays = {name: filtered.column(name).to_numpy(zero_copy_only=False) for name in keep}
            yield FrameBatch(columns=arrays, num_rows=filtered.num_rows)


@dataclass(frozen=True, slots=True)
class _V3VideoLocator:
    chunk_index: int
    file_index: int
    from_timestamp: float
    to_timestamp: float


@dataclass(frozen=True, slots=True)
class _V3Locator:
    data_chunk_index: int
    data_file_index: int
    video: dict[str, _V3VideoLocator]


@dataclass(frozen=True, slots=True)
class _V3Resolver:
    """Resolves shards from the explicit chunk/file columns in episode metadata."""

    handle: SourceHandle
    locators: dict[int, _V3Locator]

    def parquet_shard(self, episode: EpisodeRecord) -> pq.ParquetFile:
        locator = self.locators[episode.episode_index]
        return self.handle.parquet_shard(
            "data",
            f"chunk-{locator.data_chunk_index:03d}",
            f"file-{locator.data_file_index:03d}.parquet",
        )

    def video_segment(self, episode: EpisodeRecord, camera: str) -> VideoSegment:
        locator = self.locators[episode.episode_index]
        try:
            video_locator = locator.video[camera]
        except KeyError:
            raise DatasetFormatError(
                f"episode {episode.episode_index} has no video metadata for camera {camera!r}"
            ) from None
        shard_handle = self.handle.video_shard(
            "videos",
            camera,
            f"chunk-{video_locator.chunk_index:03d}",
            f"file-{video_locator.file_index:03d}.mp4",
        )
        return VideoSegment(
            handle=shard_handle,
            from_timestamp=video_locator.from_timestamp,
            to_timestamp=video_locator.to_timestamp,
        )


@dataclass(frozen=True, slots=True)
class _V3FrameSourceFactory:
    """Picklable Callable[[EpisodeRecord], FrameSource] for CanonicalDataset.

    A plain lambda would not survive pickling into a ProcessPoolExecutor
    worker under --parallel (same constraint as _V3Resolver itself); a
    frozen dataclass with __call__ does, exactly like _V3Resolver already
    proves for parquet_shard/video_segment.
    """

    resolver: _V3Resolver

    def __call__(self, episode: EpisodeRecord) -> FrameSource:
        return LeRobotFrameSourceV3(
            parquet_file=self.resolver.parquet_shard(episode),
            from_index=episode.dataset_from_index,
            to_index=episode.dataset_to_index,
            length=episode.length,
        )


def build_v3(handle: SourceHandle) -> CanonicalDataset:
    """Translate a v3.0 SourceHandle's raw metadata into a CanonicalDataset.

    Raises DatasetFormatError if required metadata files are missing or
    malformed, or if a declared episode count exceeds the resource bound.
    """
    features = _parse_features(dict(handle.info.features))
    cameras = _camera_keys(features)
    task_table = _load_v3_task_table(handle)
    episodes, locators = _load_v3_episodes(handle, cameras)
    resolver = _V3Resolver(handle=handle, locators=locators)
    return CanonicalDataset(
        format_id="lerobot",
        format_version=handle.version.value.removeprefix("v"),
        fps=handle.info.fps,
        features=features,
        num_episodes=len(episodes),
        num_frames=handle.info.total_frames,
        task_table=task_table,
        cameras=cameras,
        stats=StatsHandle(root=handle.root),
        _episodes=episodes,
        _resolver=resolver,
        _frame_source_factory=_V3FrameSourceFactory(resolver=resolver),
    )


def _load_v3_task_table(handle: SourceHandle) -> dict[int, str]:
    table = _read_parquet_table(
        handle.parquet_shard(*V3_TASKS_PATH), shard_label="meta/tasks.parquet"
    )
    try:
        indices = table.column("task_index").to_pylist()
    except KeyError as exc:
        raise DatasetFormatError(f"meta/tasks.parquet is missing required column: {exc}") from exc

    # Real-world schema vs. spec-documented schema discrepancy (confirmed M7,
    # 2026-06-21, against 5 lerobot/* Hub datasets, all codebase_version=v3.0):
    #
    # The spec (03_DATA_FORMAT_SPEC.md §2) documents the schema as:
    #   task_index: int64, task: string
    #
    # Every published lerobot dataset has instead:
    #   task_index: int64, __index_level_0__: string
    #
    # The discrepancy is a Pandas serialization artifact: lerobot stores task
    # descriptions as the DataFrame index (not a named column), so
    # df.to_parquet() serializes it as the anonymous Pandas index column
    # named "__index_level_0__" rather than as a named column "task".
    # This is not a trajlens spec error — it is a Pandas default behaviour in
    # lerobot's writer.
    #
    # We prefer "__index_level_0__" (the real Hub shape) and fall back to
    # "task" (the spec-documented shape, in case lerobot's writer ever calls
    # reset_index() before writing).
    col_names = table.column_names
    if "__index_level_0__" in col_names:
        names = table.column("__index_level_0__").to_pylist()
    elif "task" in col_names:
        names = table.column("task").to_pylist()
    else:
        raise DatasetFormatError(
            "meta/tasks.parquet has no recognisable task-description column: "
            "expected '__index_level_0__' (real Hub schema) or 'task' (spec schema), "
            f"got columns {col_names!r}"
        )
    return dict(zip(indices, names, strict=True))


def _load_v3_episodes(
    handle: SourceHandle, cameras: tuple[str, ...]
) -> tuple[tuple[EpisodeRecord, ...], dict[int, _V3Locator]]:
    # detect_version() already guarantees meta/episodes/ exists as a directory
    # for any v3.0 SourceHandle (sources/version.py), so no existence check
    # is needed here -- only an empty/sparse shard set is this layer's concern.
    episodes_root = safe_join(handle.root, *V3_EPISODES_DIR)
    shard_paths = sorted(episodes_root.glob("chunk-*/file-*.parquet"))
    rows: list[dict[str, Any]] = []
    for shard_path in shard_paths:
        relative_parts = shard_path.relative_to(handle.root).parts
        shard_label = "/".join(relative_parts)
        table = _read_parquet_table(handle.parquet_shard(*relative_parts), shard_label=shard_label)
        rows.extend(table.to_pylist())
        check_resource_bound(len(rows), max_value=MAX_DECLARED_EPISODES, what="episode count")

    rows.sort(key=lambda row: row["episode_index"])

    episodes: list[EpisodeRecord] = []
    locators: dict[int, _V3Locator] = {}
    for row in rows:
        try:
            episode_index = int(row["episode_index"])
            length = int(row["length"])
            tasks = tuple(row["tasks"])
            from_index = int(row["dataset_from_index"])
            to_index = int(row["dataset_to_index"])
            data_chunk_index = int(row["data/chunk_index"])
            data_file_index = int(row["data/file_index"])
        except KeyError as exc:
            raise DatasetFormatError(
                f"a row in meta/episodes/.../*.parquet is missing required column: {exc}"
            ) from exc

        episodes.append(
            EpisodeRecord(
                episode_index=episode_index,
                length=length,
                tasks=tasks,
                dataset_from_index=from_index,
                dataset_to_index=to_index,
            )
        )

        video_locators: dict[str, _V3VideoLocator] = {}
        for camera in cameras:
            try:
                video_locators[camera] = _V3VideoLocator(
                    chunk_index=int(row[f"videos/{camera}/chunk_index"]),
                    file_index=int(row[f"videos/{camera}/file_index"]),
                    from_timestamp=float(row[f"videos/{camera}/from_timestamp"]),
                    to_timestamp=float(row[f"videos/{camera}/to_timestamp"]),
                )
            except KeyError as exc:
                raise DatasetFormatError(
                    f"episode {episode_index} is missing video metadata for "
                    f"camera {camera!r}: {exc}"
                ) from exc

        locators[episode_index] = _V3Locator(
            data_chunk_index=data_chunk_index,
            data_file_index=data_file_index,
            video=video_locators,
        )

    return tuple(episodes), locators
