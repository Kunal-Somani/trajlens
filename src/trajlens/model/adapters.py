"""Version adapters -- translate each format version's raw metadata into a CanonicalDataset.

One function per version (ADR-002): every other part of the system targets
CanonicalDataset and never branches on DatasetVersion again. Per-version path
templates and column names are grounded in the live lerobot 0.5.2 source
(commit 8515d456), not the data format spec's docstring-derived paraphrase --
see model/__init__.py module docstring for the discrepancy this caught.

Reading here is bounded independent of what info.json declares: the actual
episode-metadata records are counted as they're read and checked against
MAX_DECLARED_EPISODES, because a dataset's episode metadata could in
principle disagree with its own total_episodes (T2 in the threat model).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from trajlens.errors import DatasetFormatError
from trajlens.model.canonical import CanonicalDataset, EpisodeRecord, FeatureSpec, VideoSegment
from trajlens.model.stats import StatsHandle
from trajlens.sources.bounds import MAX_DECLARED_EPISODES, check_resource_bound
from trajlens.sources.loader import SourceHandle
from trajlens.sources.paths import safe_join
from trajlens.sources.version import DatasetVersion

V3_EPISODES_DIR = ("meta", "episodes")
V3_TASKS_PATH = ("meta", "tasks.parquet")
LEGACY_EPISODES_PATH = ("meta", "episodes.jsonl")
LEGACY_TASKS_PATH = ("meta", "tasks.jsonl")


def build_canonical_dataset(handle: SourceHandle) -> CanonicalDataset:
    """Translate *handle*'s raw, version-specific metadata into a CanonicalDataset.

    Raises DatasetFormatError if required metadata files are missing or
    malformed, or if a declared episode count exceeds the resource bound.
    """
    if handle.version is DatasetVersion.V3_0:
        return _build_v3(handle)
    return _build_v2(handle)


def _parse_features(raw: dict[str, dict[str, Any]]) -> dict[str, FeatureSpec]:
    features: dict[str, FeatureSpec] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict) or "dtype" not in spec or "shape" not in spec:
            raise DatasetFormatError(
                f"feature {name!r} in info.json is malformed: expected a dict with "
                f"'dtype' and 'shape' keys, got {spec!r}"
            )
        dtype = spec["dtype"]
        shape = spec["shape"]
        names = spec.get("names")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or (not all(isinstance(d, int) for d in shape))
        ):
            raise DatasetFormatError(
                f"feature {name!r} in info.json has an invalid dtype/shape: {spec!r}"
            )
        features[name] = FeatureSpec(
            name=name,
            dtype=dtype,
            shape=tuple(shape),
            names=tuple(names) if names is not None else None,
        )
    return features


def _camera_keys(features: dict[str, FeatureSpec]) -> tuple[str, ...]:
    return tuple(sorted(name for name, spec in features.items() if spec.dtype == "video"))


def _expect_one_match(matches: list[Path], *, what: str) -> Path:
    if not matches:
        raise DatasetFormatError(f"expected exactly one shard file for {what}, found none")
    if len(matches) > 1:
        raise DatasetFormatError(
            f"expected exactly one shard file for {what}, found {len(matches)}: {matches}"
        )
    return matches[0]


# --------------------------------------------------------------------------- v3.0


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


def _build_v3(handle: SourceHandle) -> CanonicalDataset:
    features = _parse_features(dict(handle.info.features))
    cameras = _camera_keys(features)
    task_table = _load_v3_task_table(handle)
    episodes, locators = _load_v3_episodes(handle, cameras)
    return CanonicalDataset(
        version=handle.version,
        fps=handle.info.fps,
        features=features,
        num_episodes=len(episodes),
        num_frames=handle.info.total_frames,
        task_table=task_table,
        cameras=cameras,
        stats=StatsHandle(root=handle.root),
        _episodes=episodes,
        _resolver=_V3Resolver(handle=handle, locators=locators),
    )


def _load_v3_task_table(handle: SourceHandle) -> dict[int, str]:
    # ParquetFile.read() is untyped through pyarrow 24.x; see the matching
    # suppression on ParquetFile's constructor in sources/handles.py.
    table = handle.parquet_shard(*V3_TASKS_PATH).read()  # type: ignore[no-untyped-call]
    try:
        indices = table.column("task_index").to_pylist()
        names = table.column("task").to_pylist()
    except KeyError as exc:
        raise DatasetFormatError(f"meta/tasks.parquet is missing required column: {exc}") from exc
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
        table = handle.parquet_shard(*relative_parts).read()  # type: ignore[no-untyped-call]
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


# --------------------------------------------------------------------------- v2.x


@dataclass(frozen=True, slots=True)
class _V2Resolver:
    """Resolves shards by filename, mirroring how lerobot's own v2.1->v3.0
    converter locates legacy files (convert_dataset_v21_to_v30.py globs
    data/*/*.parquet rather than computing a chunk index from a formula --
    v2.x episode metadata does not carry chunk/file indices at all)."""

    handle: SourceHandle
    fps: int

    def parquet_shard(self, episode: EpisodeRecord) -> pq.ParquetFile:
        filename = f"episode_{episode.episode_index:06d}.parquet"
        matches = sorted(self.handle.root.glob(f"data/chunk-*/{filename}"))
        path = _expect_one_match(matches, what=f"data shard {filename}")
        return self.handle.parquet_shard(*path.relative_to(self.handle.root).parts)

    def video_segment(self, episode: EpisodeRecord, camera: str) -> VideoSegment:
        filename = f"episode_{episode.episode_index:06d}.mp4"
        matches = sorted(self.handle.root.glob(f"videos/chunk-*/{camera}/{filename}"))
        path = _expect_one_match(matches, what=f"video shard {filename} for camera {camera!r}")
        shard_handle = self.handle.video_shard(*path.relative_to(self.handle.root).parts)
        return VideoSegment(
            handle=shard_handle,
            from_timestamp=0.0,
            to_timestamp=episode.length / self.fps,
        )


def _build_v2(handle: SourceHandle) -> CanonicalDataset:
    features = _parse_features(dict(handle.info.features))
    cameras = _camera_keys(features)
    task_table = _load_v2_task_table(handle)
    episodes = _load_v2_episodes(handle)
    return CanonicalDataset(
        version=handle.version,
        fps=handle.info.fps,
        features=features,
        num_episodes=len(episodes),
        num_frames=handle.info.total_frames,
        task_table=task_table,
        cameras=cameras,
        stats=StatsHandle(root=handle.root),
        _episodes=episodes,
        _resolver=_V2Resolver(handle=handle, fps=handle.info.fps),
    )


def _read_jsonl_bounded(path: Path, *, what: str) -> Iterator[dict[str, Any]]:
    count = 0
    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            count += 1
            check_resource_bound(count, max_value=MAX_DECLARED_EPISODES, what=f"{what} count")
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetFormatError(f"{path} contains invalid JSON on a line: {exc}") from exc
            if not isinstance(parsed, dict):
                raise DatasetFormatError(f"{path} has a line that is not a JSON object: {parsed!r}")
            yield parsed


def _load_v2_task_table(handle: SourceHandle) -> dict[int, str]:
    path = safe_join(handle.root, *LEGACY_TASKS_PATH)
    if not path.is_file():
        raise DatasetFormatError("v2.x dataset is missing meta/tasks.jsonl")

    table: dict[int, str] = {}
    for row in _read_jsonl_bounded(path, what="task"):
        try:
            table[int(row["task_index"])] = str(row["task"])
        except KeyError as exc:
            raise DatasetFormatError(f"meta/tasks.jsonl row is missing column: {exc}") from exc
    return table


def _load_v2_episodes(handle: SourceHandle) -> tuple[EpisodeRecord, ...]:
    # detect_version() already guarantees meta/episodes.jsonl exists as a
    # file for any v2.x SourceHandle (sources/version.py).
    path = safe_join(handle.root, *LEGACY_EPISODES_PATH)
    rows = list(_read_jsonl_bounded(path, what="episode"))
    rows.sort(key=lambda row: row["episode_index"])

    episodes: list[EpisodeRecord] = []
    cumulative = 0
    for row in rows:
        try:
            episode_index = int(row["episode_index"])
            length = int(row["length"])
            tasks = tuple(row["tasks"])
        except KeyError as exc:
            raise DatasetFormatError(f"meta/episodes.jsonl row is missing column: {exc}") from exc
        episodes.append(
            EpisodeRecord(
                episode_index=episode_index,
                length=length,
                tasks=tasks,
                dataset_from_index=cumulative,
                dataset_to_index=cumulative + length,
            )
        )
        cumulative += length
    return tuple(episodes)
