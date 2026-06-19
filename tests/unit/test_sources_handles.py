"""Tests for lazy Parquet/video shard handles."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.builders import build_v3_dataset
from trajlens.errors import DatasetFormatError
from trajlens.sources.handles import open_parquet_shard, open_video_shard


class TestOpenParquetShard:
    def test_opens_lazily_without_reading_rows(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=3)
        shard = open_parquet_shard(tmp_path / "data" / "chunk-000" / "file-000.parquet")
        assert shard.metadata.num_rows == 12  # 3 episodes * 4 frames, from metadata only
        assert "episode_index" in shard.schema_arrow.names

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFormatError, match="expected parquet shard not found"):
            open_parquet_shard(tmp_path / "nope.parquet")


class TestOpenVideoShard:
    def test_returns_handle_without_decoding(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=1, camera="top")
        handle = open_video_shard(tmp_path / "videos" / "top" / "chunk-000" / "file-000.mp4")
        assert handle.path.is_file()

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFormatError, match="expected video shard not found"):
            open_video_shard(tmp_path / "nope.mp4")
