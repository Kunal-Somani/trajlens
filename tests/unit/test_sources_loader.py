"""Tests for SourceLoader.resolve and SourceHandle.

Hub interactions are mocked here (no network in unit tests, per 05 §5). The
real-Hub check lives in test_sources_loader_integration.py, opt-in only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
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
        with (
            patch("huggingface_hub.snapshot_download", side_effect=Exception("404")),
            pytest.raises(SourceResolutionError, match="could not resolve"),
        ):
            SourceLoader().resolve("org/does-not-exist")

    def test_hub_repo_resolves_via_downloaded_snapshot(self, tmp_path: Path) -> None:
        snapshot_dir = tmp_path / "fake-snapshot"
        snapshot_dir.mkdir()
        build_v3_dataset(snapshot_dir, num_episodes=1)

        def mock_snapshot_download(
            repo_id: str,
            repo_type: str,
            revision: str | None,
            allow_patterns: list[str],
            local_dir: str,
            **kwargs: Any,
        ) -> str:
            for src in snapshot_dir.glob("meta/**/*"):
                if src.is_file():
                    dest = Path(local_dir) / src.relative_to(snapshot_dir)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(src.read_bytes())
            return str(local_dir)

        with patch(
            "huggingface_hub.snapshot_download", side_effect=mock_snapshot_download
        ) as mock_dl:
            handle = SourceLoader().resolve("org/fake-repo")

        # The root is now the computed local_dir, not snapshot_dir directly
        assert handle.root.name == "main"
        assert handle.root.parent.name == "org--fake-repo"
        assert handle.version is DatasetVersion.V3_0

        mock_dl.assert_called_once_with(
            repo_id="org/fake-repo",
            repo_type="dataset",
            revision=None,
            allow_patterns=["meta/**"],
            local_dir=handle.root,
        )


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

    def test_hub_parquet_shard_fetched_once_per_shard_not_per_episode(self, tmp_path: Path) -> None:
        # Real v3.0 Hub datasets pack many episodes into one chunk-*/file-*
        # shard (chunks_size up to 1000). A check that loops `for episode in
        # ds: ds.parquet_shard_for_episode(episode)` must hit the network
        # once per *shard*, not once per *episode* -- otherwise a 125-episode,
        # single-shard dataset pays 125 round trips for data that lives in one
        # file. This is what SourceHandle._parquet_cache exists to prevent.
        from dataclasses import replace

        from trajlens.sources.handles import open_parquet_shard

        build_v3_dataset(tmp_path, num_episodes=10)
        local_handle = SourceLoader().resolve(str(tmp_path))
        hub_handle = replace(local_handle, repo_id="org/repo", revision=None)
        shard_path = local_handle.root / "data" / "chunk-000" / "file-000.parquet"

        with patch(
            "trajlens.sources.handles.open_hub_parquet_shard",
            side_effect=lambda *a, **kw: open_parquet_shard(shard_path),
        ) as mock_open:
            for _ in range(10):  # simulate one parquet_shard() call per episode
                hub_handle.parquet_shard("data", "chunk-000", "file-000.parquet")

        mock_open.assert_called_once()
