"""Tests for StatsHandle, the lazy meta/stats.json accessor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trajlens.errors import DatasetFormatError
from trajlens.model.stats import StatsHandle


class TestStatsHandle:
    def test_loads_valid_stats_json(self, tmp_path: Path) -> None:
        (tmp_path / "meta").mkdir()
        stats = {"timestamp": {"mean": 0.5, "std": 0.1, "min": 0.0, "max": 1.0}}
        (tmp_path / "meta" / "stats.json").write_text(json.dumps(stats))

        loaded = StatsHandle(root=tmp_path).load()
        assert loaded == stats

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        assert StatsHandle(root=tmp_path).load() is None

    def test_invalid_json_raises_format_error(self, tmp_path: Path) -> None:
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "stats.json").write_text("{not valid json")

        with pytest.raises(DatasetFormatError, match="not valid JSON"):
            StatsHandle(root=tmp_path).load()

    def test_non_object_json_raises_format_error(self, tmp_path: Path) -> None:
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "stats.json").write_text("[1, 2, 3]")

        with pytest.raises(DatasetFormatError, match="JSON object"):
            StatsHandle(root=tmp_path).load()
