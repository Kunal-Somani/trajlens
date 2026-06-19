"""Tests that the synthetic dataset builders produce the shape they claim to.

Per 05_ENGINEERING_STANDARDS.md §5: "Generating these is itself tested so
they stay valid/broken as intended."
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.fixtures.builders import FRAMES_PER_EPISODE, build_v2_dataset, build_v3_dataset


class TestBuildV3Dataset:
    def test_produces_v3_shape(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=3)
        assert (tmp_path / "meta" / "episodes").is_dir()
        assert not (tmp_path / "meta" / "episodes.jsonl").exists()
        info = json.loads((tmp_path / "meta" / "info.json").read_text())
        assert info["codebase_version"] == "v3.0"
        assert info["total_episodes"] == 3
        assert info["total_frames"] == 3 * FRAMES_PER_EPISODE

    def test_data_shard_exists(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=2)
        assert (tmp_path / "data" / "chunk-000" / "file-000.parquet").is_file()


class TestBuildV2Dataset:
    def test_produces_v2_shape(self, tmp_path: Path) -> None:
        build_v2_dataset(tmp_path, codebase_version="v2.1", num_episodes=2)
        assert (tmp_path / "meta" / "episodes.jsonl").is_file()
        assert not (tmp_path / "meta" / "episodes").exists()
        info = json.loads((tmp_path / "meta" / "info.json").read_text())
        assert info["codebase_version"] == "v2.1"

    def test_one_parquet_file_per_episode(self, tmp_path: Path) -> None:
        build_v2_dataset(tmp_path, num_episodes=3)
        for ep in range(3):
            assert (tmp_path / "data" / "chunk-000" / f"episode_{ep:06d}.parquet").is_file()

    def test_v2_0_omits_episodes_stats(self, tmp_path: Path) -> None:
        build_v2_dataset(tmp_path, codebase_version="v2.0", num_episodes=1)
        assert not (tmp_path / "meta" / "episodes_stats.jsonl").exists()
