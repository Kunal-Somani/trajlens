"""Tests for lazy Parquet/video shard handles."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.fixtures.builders import build_v3_dataset
from trajlens.errors import DatasetFormatError
from trajlens.sources.handles import open_hub_parquet_shard, open_parquet_shard, open_video_shard


class TestOpenParquetShard:
    def test_opens_lazily_without_reading_rows(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=3)
        shard = open_parquet_shard(tmp_path / "data" / "chunk-000" / "file-000.parquet")
        assert shard.metadata.num_rows == 12  # 3 episodes * 4 frames, from metadata only
        assert "episode_index" in shard.schema_arrow.names

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFormatError, match="expected parquet shard not found"):
            open_parquet_shard(tmp_path / "nope.parquet")

    def test_corrupt_file_raises_typed_error_naming_the_file(self, tmp_path: Path) -> None:
        """A file that exists but isn't valid Parquet must raise DatasetFormatError
        naming the path, never a raw pyarrow.lib.ArrowInvalid traceback."""
        corrupt = tmp_path / "corrupt.parquet"
        corrupt.write_bytes(b"not a valid parquet file")

        with pytest.raises(DatasetFormatError) as exc_info:
            open_parquet_shard(corrupt)

        assert str(corrupt) in str(exc_info.value)
        assert "not a valid Parquet file" in str(exc_info.value)


class TestOpenHubParquetShard:
    def test_opens_with_whole_file_cache(self, tmp_path: Path) -> None:
        # pyarrow's Parquet reader issues dozens of small seek-and-read calls
        # per file (footer, then each column chunk); over HTTP each becomes
        # its own round trip unless the whole (small) shard is fetched in one
        # request. cache_type="all" is what collapses that to a single fetch
        # -- assert it's actually requested, since this is silent and easy to
        # regress on a future refactor of this call site.
        build_v3_dataset(tmp_path, num_episodes=1)
        shard_path = tmp_path / "data" / "chunk-000" / "file-000.parquet"

        fake_fs = MagicMock()
        fake_fs.open.return_value = shard_path.open("rb")

        with patch("huggingface_hub.HfFileSystem", return_value=fake_fs) as fake_fs_cls:
            open_hub_parquet_shard("org/repo", None, "data/chunk-000/file-000.parquet")

        fake_fs_cls.assert_called_once_with(revision=None)
        fake_fs.open.assert_called_once_with(
            "datasets/org/repo/data/chunk-000/file-000.parquet", "rb", cache_type="all"
        )

    def test_missing_shard_raises_format_error(self) -> None:
        with pytest.raises(DatasetFormatError, match="expected parquet shard not found on Hub"):
            open_hub_parquet_shard("org/does-not-exist", None, "data/chunk-000/file-000.parquet")

    def test_corrupt_fetched_shard_raises_typed_error(self, tmp_path: Path) -> None:
        """A shard that fetches successfully but isn't valid Parquet must raise
        DatasetFormatError, never a raw pyarrow.lib.ArrowInvalid traceback."""
        corrupt = tmp_path / "corrupt.parquet"
        corrupt.write_bytes(b"not a valid parquet file")

        fake_fs = MagicMock()
        fake_fs.open.return_value = corrupt.open("rb")

        with (
            patch("huggingface_hub.HfFileSystem", return_value=fake_fs),
            pytest.raises(DatasetFormatError, match="not a valid Parquet file"),
        ):
            open_hub_parquet_shard("org/repo", None, "data/chunk-000/file-000.parquet")


class TestOpenVideoShard:
    def test_returns_handle_without_decoding(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=1, camera="top")
        handle = open_video_shard(tmp_path / "videos" / "top" / "chunk-000" / "file-000.mp4")
        assert handle.path.is_file()

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFormatError, match="expected video shard not found"):
            open_video_shard(tmp_path / "nope.mp4")
