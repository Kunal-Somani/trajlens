"""Tests for build_canonical_dataset and the per-version adapters/resolvers.

Per 00_AGENT_OPERATING_MANUAL.md §5: happy path, at least two failure modes,
and one edge case (empty dataset) for this unit of work.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tests.fixtures.builders import FRAMES_PER_EPISODE, build_v2_dataset, build_v3_dataset
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
