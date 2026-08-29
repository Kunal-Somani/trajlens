"""Unit tests for the FormatAdapter protocol's data types (m1a)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from trajlens.adapters.protocol import Capabilities, FormatMatch, WriteResult


class TestCapabilities:
    def test_frozen_assignment_raises(self) -> None:
        caps = Capabilities(
            readable=True,
            writable=False,
            streamable=True,
            repairable=True,
            lossless=True,
            lost_on_write=(),
        )
        with pytest.raises(FrozenInstanceError):
            caps.readable = False  # type: ignore[misc]

    def test_hashable(self) -> None:
        caps = Capabilities(
            readable=True,
            writable=False,
            streamable=True,
            repairable=True,
            lossless=True,
            lost_on_write=(),
        )
        assert {caps} == {caps}


class TestFormatMatch:
    def test_fields_accessible_and_correct_types(self) -> None:
        match = FormatMatch(
            format_id="lerobot",
            format_version="3.0",
            confidence=1.0,
            evidence="meta/info.json codebase_version=3.0",
        )
        assert match.format_id == "lerobot"
        assert isinstance(match.format_id, str)
        assert match.format_version == "3.0"
        assert isinstance(match.format_version, str)
        assert match.confidence == 1.0
        assert isinstance(match.confidence, float)
        assert match.evidence == "meta/info.json codebase_version=3.0"
        assert isinstance(match.evidence, str)


class TestWriteResult:
    def test_empty_lost_dimensions_is_valid(self, tmp_path: Path) -> None:
        result = WriteResult(
            output_path=tmp_path / "out",
            episodes_written=3,
            frames_written=90,
            lost_dimensions=(),
        )
        assert result.lost_dimensions == ()
        assert result.episodes_written == 3
        assert result.frames_written == 90
