"""Tests for safe_join (T1 — path traversal containment)."""

from __future__ import annotations

from pathlib import Path

import pytest

from trajlens.errors import PathTraversalError
from trajlens.sources.paths import safe_join


class TestSafeJoin:
    def test_joins_within_root(self, tmp_path: Path) -> None:
        result = safe_join(tmp_path, "meta", "info.json")
        assert result == (tmp_path / "meta" / "info.json").resolve()

    def test_joins_single_multi_segment_part(self, tmp_path: Path) -> None:
        result = safe_join(tmp_path, "data/chunk-000/file-000.parquet")
        assert result == (tmp_path / "data" / "chunk-000" / "file-000.parquet").resolve()

    def test_rejects_dotdot_segment(self, tmp_path: Path) -> None:
        with pytest.raises(PathTraversalError):
            safe_join(tmp_path, "..", "..", "etc", "passwd")

    def test_rejects_dotdot_within_single_part(self, tmp_path: Path) -> None:
        with pytest.raises(PathTraversalError):
            safe_join(tmp_path, "meta/../../etc/passwd")

    def test_rejects_absolute_part_injection(self, tmp_path: Path) -> None:
        # Path(root) / "/etc/passwd" would normally reset to the absolute
        # path under naive pathlib joining; safe_join must not do that.
        result = safe_join(tmp_path, "/etc/passwd")
        assert result == (tmp_path / "etc" / "passwd").resolve()

    def test_empty_parts_are_noops(self, tmp_path: Path) -> None:
        result = safe_join(tmp_path, "", ".", "meta")
        assert result == (tmp_path / "meta").resolve()
