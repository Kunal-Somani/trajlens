"""Tests for meta/info.json loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fixtures.builders import build_v3_dataset
from trajlens.errors import DatasetFormatError
from trajlens.sources.info import load_info


class TestLoadInfo:
    def test_loads_valid_info(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=2)
        info = load_info(tmp_path)
        assert info.codebase_version == "v3.0"
        assert info.fps == 30
        assert info.total_episodes == 2

    def test_missing_info_file(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFormatError, match="required metadata file not found"):
            load_info(tmp_path)

    def test_invalid_json(self, tmp_path: Path) -> None:
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "info.json").write_text("{not valid json")
        with pytest.raises(DatasetFormatError, match="not valid JSON"):
            load_info(tmp_path)

    def test_missing_required_key(self, tmp_path: Path) -> None:
        (tmp_path / "meta").mkdir()
        # codebase_version and fps present, features omitted entirely.
        (tmp_path / "meta" / "info.json").write_text(
            json.dumps({"codebase_version": "v3.0", "fps": 30})
        )
        with pytest.raises(DatasetFormatError, match="does not match the expected schema"):
            load_info(tmp_path)

    def test_extra_unknown_keys_tolerated(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=1)
        raw = json.loads((tmp_path / "meta" / "info.json").read_text())
        raw["robot_type"] = "so100"
        (tmp_path / "meta" / "info.json").write_text(json.dumps(raw))
        info = load_info(tmp_path)
        assert info.codebase_version == "v3.0"
