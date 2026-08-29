"""LeRobot v2.0/v2.1 adapter -- translates the legacy raw metadata into a CanonicalDataset.

Split out of model/adapters.py (v0.5 M1-B) so each format version's loading
logic lives in its own module; model/lerobot.py is the entry point that
branches between this module and lerobot_v3.py. Per-version path templates
and column names are grounded in the live lerobot 0.5.2 source (commit
8515d456), not the data format spec's docstring-derived paraphrase -- see
model/__init__.py module docstring for the discrepancy this caught.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
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
from trajlens.model.stats import StatsHandle
from trajlens.sources.bounds import MAX_DECLARED_EPISODES, check_resource_bound
from trajlens.sources.loader import SourceHandle
from trajlens.sources.paths import safe_join

LEGACY_EPISODES_PATH = ("meta", "episodes.jsonl")
LEGACY_TASKS_PATH = ("meta", "tasks.jsonl")

_ARROW_SCALAR_TO_DTYPE: dict[pa.DataType, str] = {
    pa.float32(): "float32",
    pa.float64(): "float64",
    pa.int64(): "int64",
    pa.int32(): "int32",
    pa.int16(): "int16",
    pa.int8(): "int8",
    pa.uint8(): "uint8",
    pa.bool_(): "bool",
    pa.string(): "string",
}


def _feature_spec_from_arrow(name: str, arrow_type: pa.DataType) -> FeatureSpec:
    """Map one physical Arrow column to a FeatureSpec, for FrameSource.schema().

    Raises DatasetFormatError for a column type this mapping does not
    recognise (per manual: a check/read that cannot resolve a real answer
    must fail loudly, never guess).
    """
    if isinstance(arrow_type, pa.FixedSizeListType):
        value_dtype = _ARROW_SCALAR_TO_DTYPE.get(arrow_type.value_type)
        if value_dtype is None:
            raise DatasetFormatError(
                f"column {name!r} has an unsupported fixed-size-list element "
                f"type {arrow_type.value_type!r}"
            )
        return FeatureSpec(name=name, dtype=value_dtype, shape=(arrow_type.list_size,), names=None)
    dtype = _ARROW_SCALAR_TO_DTYPE.get(arrow_type)
    if dtype is None:
        raise DatasetFormatError(
            f"column {name!r} has an unsupported Arrow type {arrow_type!r} for FrameSource.schema()"
        )
    return FeatureSpec(name=name, dtype=dtype, shape=(1,), names=None)


def _arrow_schema_to_features(schema: pa.Schema) -> dict[str, FeatureSpec]:
    return {name: _feature_spec_from_arrow(name, schema.field(name).type) for name in schema.names}


def _read_parquet_table(pf: pq.ParquetFile, *, shard_label: str) -> pa.Table:
    """Read *pf*'s full table, wrapping pyarrow row-group decode failures.

    Opening a ParquetFile only reads the footer (sources/handles.py
    open_parquet_shard already wraps that); a corrupt row group can still
    raise ArrowInvalid here, on the actual .read(). A raw ArrowInvalid must
    never reach a CLI user (manual's "errors are typed" rule).
    """
    try:
        return pf.read()  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise DatasetFormatError(
            f"{shard_label} could not be read as Parquet (row-group decode failed: {exc}). "
            "The file may be truncated or corrupted."
        ) from exc


def _flatten_feature_names(names: Any, feature_name: str) -> tuple[str, ...]:
    """Flatten info.json's ``names`` field to a flat tuple of element names.

    LeRobot's own DatasetMetadata.names is typed ``dict[str, list | dict]``:
    a feature's ``names`` may be a flat list (e.g. ``["height", "width"]``)
    or a dict mapping an axis label to a nested list (e.g.
    ``{"motors": ["motor_0", "motor_1"]}``). Both are valid; the element
    count is the list length, or the sum of nested list lengths for dicts.
    """
    if isinstance(names, dict):
        flattened: list[str] = []
        for value in names.values():
            if not isinstance(value, list):
                raise DatasetFormatError(
                    f"feature {feature_name!r} in info.json has a 'names' dict "
                    f"whose value is not a list: {names!r}"
                )
            flattened.extend(value)
        return tuple(flattened)
    return tuple(names)


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
            names=_flatten_feature_names(names, name) if names is not None else None,
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


@dataclass(frozen=True, slots=True)
class LeRobotFrameSourceV2:
    """FrameSource over one v2.x episode's dedicated Parquet shard.

    v2.x is already one-file-per-episode, so every row group belongs to this
    episode -- unlike v3.0, no row filtering by frame index is needed.
    """

    parquet_file: pq.ParquetFile
    length: int

    def schema(self) -> Mapping[str, FeatureSpec]:
        return _arrow_schema_to_features(self.parquet_file.schema_arrow)

    def num_rows(self) -> int:
        return self.length

    def iter_batches(self, columns: Sequence[str] | None = None) -> Iterator[FrameBatch]:
        cols = list(columns) if columns is not None else None
        for rg_idx in range(self.parquet_file.num_row_groups):
            table = self.parquet_file.read_row_group(rg_idx, columns=cols)  # type: ignore[no-untyped-call]
            arrays = {
                name: table.column(name).to_numpy(zero_copy_only=False)
                for name in table.column_names
            }
            yield FrameBatch(columns=arrays, num_rows=table.num_rows)


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


@dataclass(frozen=True, slots=True)
class _V2FrameSourceFactory:
    """Picklable Callable[[EpisodeRecord], FrameSource] for CanonicalDataset.

    A plain lambda would not survive pickling into a ProcessPoolExecutor
    worker under --parallel (same constraint as _V2Resolver itself); a
    frozen dataclass with __call__ does, exactly like _V2Resolver already
    proves for parquet_shard/video_segment.
    """

    resolver: _V2Resolver

    def __call__(self, episode: EpisodeRecord) -> FrameSource:
        return LeRobotFrameSourceV2(
            parquet_file=self.resolver.parquet_shard(episode), length=episode.length
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


def build_v2(handle: SourceHandle) -> CanonicalDataset:
    """Translate a v2.0/v2.1 SourceHandle's raw metadata into a CanonicalDataset.

    Raises DatasetFormatError if required metadata files are missing or
    malformed, or if a declared episode count exceeds the resource bound.
    """
    if handle.repo_id is not None:
        raise DatasetFormatError(
            "v2.x Hub datasets cannot be lazily streamed. Shard paths are implicit and "
            "require a local filesystem to glob. Download the dataset locally to verify it."
        )

    features = _parse_features(dict(handle.info.features))
    cameras = _camera_keys(features)
    task_table = _load_v2_task_table(handle)
    episodes = _load_v2_episodes(handle)
    resolver = _V2Resolver(handle=handle, fps=handle.info.fps)
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
        _frame_source_factory=_V2FrameSourceFactory(resolver=resolver),
    )
