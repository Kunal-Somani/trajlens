"""Tests for build_canonical_dataset and the per-version adapters/resolvers.

Per 00_AGENT_OPERATING_MANUAL.md §5: happy path, at least two failure modes,
and one edge case (empty dataset) for this unit of work.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tests.fixtures.builders import (
    FRAMES_PER_EPISODE,
    build_v2_dataset,
    build_v3_dataset,
    build_v3_dataset_hub_tasks_schema,
)
from trajlens.errors import DatasetFormatError
from trajlens.model import build_canonical_dataset
from trajlens.sources.loader import SourceHandle, SourceLoader
from trajlens.sources.version import DatasetVersion


def _resolve(tmp_path: Path) -> SourceHandle:
    return SourceLoader().resolve(str(tmp_path))


class TestHappyPathV3:
    def test_parses_features_fps_and_episode_count(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=3, camera="top")
        ds = build_canonical_dataset(_resolve(tmp_path))

        assert ds.version is DatasetVersion.V3_0
        assert ds.fps == 30
        assert ds.num_episodes == 3
        assert ds.num_frames == 3 * FRAMES_PER_EPISODE
        assert ds.cameras == ("top",)
        assert "timestamp" in ds.features
        assert ds.features["timestamp"].dtype == "float32"
        assert ds.features["timestamp"].shape == (1,)
        assert ds.task_table == {0: "do the thing"}

    def test_episode_records_have_correct_offsets_and_tasks(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=3)
        ds = build_canonical_dataset(_resolve(tmp_path))

        episodes = list(ds)
        assert len(episodes) == 3
        for i, ep in enumerate(episodes):
            assert ep.episode_index == i
            assert ep.length == FRAMES_PER_EPISODE
            assert ep.tasks == ("do the thing",)
            assert ep.dataset_from_index == i * FRAMES_PER_EPISODE
            assert ep.dataset_to_index == (i + 1) * FRAMES_PER_EPISODE

    def test_lazy_parquet_and_video_shard_access(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=2, camera="wrist")
        ds = build_canonical_dataset(_resolve(tmp_path))

        ep = ds.episode(1)
        shard = ds.parquet_shard_for_episode(ep)
        assert shard.metadata.num_rows == 2 * FRAMES_PER_EPISODE

        segment = ds.video_segment_for_episode(ep, "wrist")
        assert segment.handle.path.is_file()
        assert segment.from_timestamp == FRAMES_PER_EPISODE / 30.0
        assert segment.to_timestamp == 2 * FRAMES_PER_EPISODE / 30.0

    def test_discovers_episodes_across_multiple_shards(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=5, episodes_per_shard=2)
        ds = build_canonical_dataset(_resolve(tmp_path))

        assert ds.num_episodes == 5
        assert [ep.episode_index for ep in ds] == [0, 1, 2, 3, 4]
        assert (tmp_path / "meta" / "episodes" / "chunk-002" / "file-000.parquet").is_file()


class TestHappyPathV2:
    @pytest.mark.parametrize("codebase_version", ["v2.0", "v2.1"])
    def test_parses_v2_dataset(self, tmp_path: Path, codebase_version: str) -> None:
        build_v2_dataset(tmp_path, codebase_version=codebase_version, num_episodes=3, camera="top")
        ds = build_canonical_dataset(_resolve(tmp_path))

        assert ds.version.value == codebase_version
        assert ds.num_episodes == 3
        assert ds.cameras == ("top",)
        episodes = list(ds)
        assert [ep.dataset_from_index for ep in episodes] == [0, 4, 8]
        assert [ep.dataset_to_index for ep in episodes] == [4, 8, 12]

    def test_lazy_parquet_and_video_shard_access(self, tmp_path: Path) -> None:
        build_v2_dataset(tmp_path, codebase_version="v2.1", num_episodes=2, camera="top")
        ds = build_canonical_dataset(_resolve(tmp_path))

        ep = ds.episode(1)
        shard = ds.parquet_shard_for_episode(ep)
        assert shard.metadata.num_rows == FRAMES_PER_EPISODE

        segment = ds.video_segment_for_episode(ep, "top")
        assert segment.handle.path.is_file()
        assert segment.from_timestamp == 0.0
        assert segment.to_timestamp == FRAMES_PER_EPISODE / 30.0


class TestFailureModes:
    def test_malformed_feature_schema_raises_format_error(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=1)
        info_path = tmp_path / "meta" / "info.json"
        raw = json.loads(info_path.read_text())
        raw["features"]["timestamp"] = {"dtype": "float32"}  # missing required 'shape'
        info_path.write_text(json.dumps(raw))

        with pytest.raises(DatasetFormatError, match="malformed"):
            build_canonical_dataset(_resolve(tmp_path))

    def test_invalid_feature_dtype_type_raises_format_error(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=1)
        info_path = tmp_path / "meta" / "info.json"
        raw = json.loads(info_path.read_text())
        raw["features"]["timestamp"] = {"dtype": "float32", "shape": "not-a-list"}
        info_path.write_text(json.dumps(raw))

        with pytest.raises(DatasetFormatError, match="invalid dtype/shape"):
            build_canonical_dataset(_resolve(tmp_path))

    def test_v3_episode_row_missing_column_raises_format_error(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=1, camera="top")
        shard_path = tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(shard_path)
        pq.write_table(table.drop_columns(["length"]), shard_path)

        with pytest.raises(DatasetFormatError, match="missing required column"):
            build_canonical_dataset(_resolve(tmp_path))

    def test_v3_episode_row_missing_camera_column_raises_format_error(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=1, camera="top")
        shard_path = tmp_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(shard_path)
        pq.write_table(table.drop_columns(["videos/top/from_timestamp"]), shard_path)

        with pytest.raises(DatasetFormatError, match="missing video metadata"):
            build_canonical_dataset(_resolve(tmp_path))

    def test_v2_episodes_jsonl_malformed_line_raises_format_error(self, tmp_path: Path) -> None:
        build_v2_dataset(tmp_path, codebase_version="v2.1", num_episodes=1)
        episodes_path = tmp_path / "meta" / "episodes.jsonl"
        episodes_path.write_text(episodes_path.read_text() + "not valid json\n")

        with pytest.raises(DatasetFormatError, match="invalid JSON"):
            build_canonical_dataset(_resolve(tmp_path))

    def test_v2_episodes_jsonl_non_object_line_raises_format_error(self, tmp_path: Path) -> None:
        build_v2_dataset(tmp_path, codebase_version="v2.1", num_episodes=1)
        episodes_path = tmp_path / "meta" / "episodes.jsonl"
        episodes_path.write_text(episodes_path.read_text() + json.dumps([1, 2, 3]) + "\n")

        with pytest.raises(DatasetFormatError, match="not a JSON object"):
            build_canonical_dataset(_resolve(tmp_path))

    def test_missing_v2_tasks_file_raises_format_error(self, tmp_path: Path) -> None:
        build_v2_dataset(tmp_path, codebase_version="v2.1", num_episodes=1)
        (tmp_path / "meta" / "tasks.jsonl").unlink()

        with pytest.raises(DatasetFormatError, match=r"tasks\.jsonl"):
            build_canonical_dataset(_resolve(tmp_path))

    def test_v2_resolver_missing_shard_raises_format_error(self, tmp_path: Path) -> None:
        build_v2_dataset(tmp_path, codebase_version="v2.1", num_episodes=1, camera="top")
        ds = build_canonical_dataset(_resolve(tmp_path))
        ep = ds.episode(0)
        (tmp_path / "data" / "chunk-000" / "episode_000000.parquet").unlink()

        with pytest.raises(DatasetFormatError, match="found none"):
            ds.parquet_shard_for_episode(ep)

    def test_v2_resolver_duplicate_shard_raises_format_error(self, tmp_path: Path) -> None:
        import shutil

        build_v2_dataset(tmp_path, codebase_version="v2.1", num_episodes=1, camera="top")
        ds = build_canonical_dataset(_resolve(tmp_path))
        ep = ds.episode(0)
        duplicate_dir = tmp_path / "data" / "chunk-001"
        duplicate_dir.mkdir()
        shutil.copy(
            tmp_path / "data" / "chunk-000" / "episode_000000.parquet",
            duplicate_dir / "episode_000000.parquet",
        )

        with pytest.raises(DatasetFormatError, match="found 2"):
            ds.parquet_shard_for_episode(ep)

    def test_v3_resolver_missing_camera_video_metadata_raises_format_error(
        self, tmp_path: Path
    ) -> None:
        build_v3_dataset(tmp_path, num_episodes=1, camera="top")
        ds = build_canonical_dataset(_resolve(tmp_path))
        ep = ds.episode(0)

        with pytest.raises(DatasetFormatError, match="no video metadata"):
            ds.video_segment_for_episode(ep, "does-not-exist")


class TestTasksParquetSchemaCompat:
    """Regression tests for the real Hub tasks.parquet schema (Bug 4 / M7 fix).

    All lerobot/* Hub datasets (confirmed 2026-06-21, codebase_version=v3.0)
    write meta/tasks.parquet with columns ``task_index, __index_level_0__``
    (Pandas DataFrame index serialized as an anonymous column) rather than the
    spec-documented ``task_index, task``.  _load_v3_task_table() must accept
    both without raising DatasetFormatError.
    """

    def test_hub_schema_index_level_0_loads_correctly(self, tmp_path: Path) -> None:
        """Happy path: __index_level_0__ (real Hub shape) is accepted and returns task table."""
        build_v3_dataset_hub_tasks_schema(tmp_path, num_episodes=2)
        ds = build_canonical_dataset(_resolve(tmp_path))

        assert ds.task_table == {0: "do the thing"}
        assert ds.version is DatasetVersion.V3_0
        assert ds.num_episodes == 2

    def test_hub_schema_episodes_see_correct_task_descriptions(self, tmp_path: Path) -> None:
        """Episodes loaded from a Hub-schema dataset have the right task string."""
        build_v3_dataset_hub_tasks_schema(tmp_path, num_episodes=3)
        ds = build_canonical_dataset(_resolve(tmp_path))

        for ep in ds:
            # Episode task list is resolved from episode metadata; the task
            # *description* lookup goes through task_table, which came from
            # __index_level_0__.  If the fix is wrong, task_table is empty and
            # episodes would either error or have no task text.
            assert ep.tasks == ("do the thing",)

    def test_spec_schema_task_column_still_accepted(self, tmp_path: Path) -> None:
        """Fallback path: a 'task' column (spec schema) is still accepted."""
        build_v3_dataset(tmp_path, num_episodes=2)
        # build_v3_dataset already writes the spec schema (task_index + task).
        ds = build_canonical_dataset(_resolve(tmp_path))
        assert ds.task_table == {0: "do the thing"}

    def test_unrecognisable_tasks_schema_raises_format_error(self, tmp_path: Path) -> None:
        """Neither __index_level_0__ nor 'task' present → DatasetFormatError with clear message."""
        build_v3_dataset(tmp_path, num_episodes=1)
        tasks_path = tmp_path / "meta" / "tasks.parquet"
        # Write a tasks.parquet that has task_index but no recognisable description column.
        import pyarrow as pa

        bad_table = pa.table(
            {"task_index": pa.array([0], type=pa.int64()), "description": pa.array(["x"])}
        )
        pq.write_table(bad_table, tasks_path)

        with pytest.raises(DatasetFormatError, match="no recognisable task-description column"):
            build_canonical_dataset(_resolve(tmp_path))


class TestEmptyDatasetEdgeCase:
    def test_v3_empty_dataset_has_zero_episodes(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=0)
        ds = build_canonical_dataset(_resolve(tmp_path))

        assert ds.num_episodes == 0
        assert len(ds) == 0
        assert list(ds) == []
        assert ds.num_frames == 0

    def test_v2_empty_dataset_has_zero_episodes(self, tmp_path: Path) -> None:
        build_v2_dataset(tmp_path, codebase_version="v2.1", num_episodes=0)
        ds = build_canonical_dataset(_resolve(tmp_path))

        assert ds.num_episodes == 0
        assert len(ds) == 0
        assert list(ds) == []
