"""Tests for SourceLoader.resolve and SourceHandle.

Hub interactions are mocked here (no network in unit tests, per 05 §5). The
real-Hub check lives in test_sources_loader_integration.py, opt-in only.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.fixtures.builders import build_v2_dataset, build_v3_dataset
from trajlens.errors import (
    DatasetFormatError,
    PathTraversalError,
    ResourceBoundError,
    SourceResolutionError,
)
from trajlens.sources.loader import SourceLoader
from trajlens.sources.version import DatasetVersion


class TestResolveLocal:
    def test_resolves_clean_v3_dataset(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=3)
        handle = SourceLoader().resolve(str(tmp_path))
        assert handle.version is DatasetVersion.V3_0
        assert handle.root == tmp_path.resolve()
        assert handle.info.total_episodes == 3

    def test_resolves_clean_v2_1_dataset(self, tmp_path: Path) -> None:
        build_v2_dataset(tmp_path, codebase_version="v2.1", num_episodes=2)
        handle = SourceLoader().resolve(str(tmp_path))
        assert handle.version is DatasetVersion.V2_1

    def test_missing_info_json_raises_format_error(self, tmp_path: Path) -> None:
        (tmp_path / "meta").mkdir()
        with pytest.raises(DatasetFormatError):
            SourceLoader().resolve(str(tmp_path))

    def test_absurd_episode_count_raises_resource_bound_error(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=1)
        raw = json.loads((tmp_path / "meta" / "info.json").read_text())
        raw["total_episodes"] = 10_000_000_000
        (tmp_path / "meta" / "info.json").write_text(json.dumps(raw))
        with pytest.raises(ResourceBoundError, match="10000000000"):
            SourceLoader().resolve(str(tmp_path))

    def test_absurd_frame_count_raises_resource_bound_error(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=1)
        raw = json.loads((tmp_path / "meta" / "info.json").read_text())
        raw["total_frames"] = 999_999_999_999
        (tmp_path / "meta" / "info.json").write_text(json.dumps(raw))
        with pytest.raises(ResourceBoundError):
            SourceLoader().resolve(str(tmp_path))


class TestResolveHubMocked:
    def test_hub_repo_not_found_raises_source_resolution_error(self, tmp_path: Path) -> None:
        nonexistent_local = tmp_path / "does-not-exist"
        with (
            patch("huggingface_hub.snapshot_download", side_effect=Exception("404")),
            pytest.raises(SourceResolutionError, match="could not resolve"),
        ):
            SourceLoader().resolve(str(nonexistent_local))

    def test_hub_repo_resolves_via_downloaded_snapshot(self, tmp_path: Path) -> None:
        snapshot_dir = tmp_path / "fake-snapshot"
        snapshot_dir.mkdir()
        build_v3_dataset(snapshot_dir, num_episodes=1)

        with patch("huggingface_hub.snapshot_download", return_value=str(snapshot_dir)):
            handle = SourceLoader().resolve("org/fake-repo")

        assert handle.root == snapshot_dir.resolve()
        assert handle.version is DatasetVersion.V3_0


class TestSourceHandleShardAccess:
    def test_parquet_shard_safe_joins_and_opens(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=2)
        handle = SourceLoader().resolve(str(tmp_path))
        shard = handle.parquet_shard("data", "chunk-000", "file-000.parquet")
        assert shard.metadata.num_rows == 8

    def test_video_shard_safe_joins_and_opens(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=1, camera="top")
        handle = SourceLoader().resolve(str(tmp_path))
        video = handle.video_shard("videos", "top", "chunk-000", "file-000.mp4")
        assert video.path.is_file()

    def test_traversal_attempt_via_relative_parts_rejected(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=1)
        handle = SourceLoader().resolve(str(tmp_path))
        with pytest.raises(PathTraversalError):
            handle.parquet_shard("..", "..", "etc", "passwd")
