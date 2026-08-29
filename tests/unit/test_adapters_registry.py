"""Unit tests for FormatAdapterRegistry and detect_format() (m1a)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.builders import build_v2_dataset, build_v3_dataset
from trajlens.adapters.protocol import Capabilities, FormatMatch, WriteResult
from trajlens.adapters.registry import detect_format as _real_detect_format
from trajlens.adapters.registry import registry
from trajlens.errors import FormatDetectionError, SourceResolutionError
from trajlens.model.canonical import CanonicalDataset
from trajlens.sources.loader import SourceHandle, SourceLoader


class _AlwaysMatchesAdapter:
    """A throwaway adapter that always matches, to test ambiguity detection."""

    format_id = "test_collider"
    capabilities = Capabilities(
        readable=True,
        writable=False,
        streamable=False,
        repairable=False,
        lossless=True,
        lost_on_write=(),
    )

    def detect(self, h: SourceHandle) -> FormatMatch | None:
        return FormatMatch(
            format_id="test_collider",
            format_version="0",
            confidence=0.5,
            evidence="always matches",
        )

    def load(self, h: SourceHandle) -> CanonicalDataset:
        raise NotImplementedError

    def write(self, ds: CanonicalDataset, out: Path) -> WriteResult:
        raise NotImplementedError


def test_detect_format_v2_0(tmp_path: Path) -> None:
    build_v2_dataset(tmp_path, codebase_version="v2.0")
    handle = SourceLoader().resolve(str(tmp_path))
    match = _real_detect_format(handle)
    assert match.format_id == "lerobot"
    assert match.format_version == "2.0"


def test_detect_format_v2_1(tmp_path: Path) -> None:
    build_v2_dataset(tmp_path, codebase_version="v2.1")
    handle = SourceLoader().resolve(str(tmp_path))
    match = _real_detect_format(handle)
    assert match.format_id == "lerobot"
    assert match.format_version == "2.1"


def test_detect_format_v3_0(tmp_path: Path) -> None:
    build_v3_dataset(tmp_path)
    handle = SourceLoader().resolve(str(tmp_path))
    match = _real_detect_format(handle)
    assert match.format_id == "lerobot"
    assert match.format_version == "3.0"


def test_detect_format_with_no_registered_adapters_raises_source_resolution_error(
    tmp_path: Path,
) -> None:
    build_v3_dataset(tmp_path)
    handle = SourceLoader().resolve(str(tmp_path))

    saved = dict(registry._adapters)
    registry._adapters.clear()
    try:
        with pytest.raises(SourceResolutionError):
            _real_detect_format(handle)
    finally:
        registry._adapters.update(saved)


def test_ambiguous_detection_raises_format_detection_error(tmp_path: Path) -> None:
    build_v3_dataset(tmp_path)
    handle = SourceLoader().resolve(str(tmp_path))

    registry.register(_AlwaysMatchesAdapter())
    try:
        with pytest.raises(FormatDetectionError) as exc_info:
            _real_detect_format(handle)
    finally:
        del registry._adapters["test_collider"]

    message = str(exc_info.value)
    assert "lerobot" in message
    assert "test_collider" in message


def test_ambiguous_detection_message_contains_both_evidence_strings(tmp_path: Path) -> None:
    build_v3_dataset(tmp_path)
    handle = SourceLoader().resolve(str(tmp_path))

    registry.register(_AlwaysMatchesAdapter())
    try:
        with pytest.raises(FormatDetectionError) as exc_info:
            _real_detect_format(handle)
    finally:
        del registry._adapters["test_collider"]

    message = str(exc_info.value)
    assert "meta/info.json codebase_version=3.0" in message
    assert "always matches" in message
