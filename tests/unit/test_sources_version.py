"""Tests for version detection (codebase_version cross-checked against shape)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.fixtures.builders import build_v2_dataset, build_v3_dataset
from trajlens.errors import DatasetFormatError, DatasetVersionError
from trajlens.sources.info import load_info
from trajlens.sources.version import DatasetVersion, detect_version


class TestDetectVersion:
    def test_detects_v2_0(self, tmp_path: Path) -> None:
        build_v2_dataset(tmp_path, codebase_version="v2.0", num_episodes=2)
        info = load_info(tmp_path)
        assert detect_version(tmp_path, info) is DatasetVersion.V2_0

    def test_detects_v2_1(self, tmp_path: Path) -> None:
        build_v2_dataset(tmp_path, codebase_version="v2.1", num_episodes=2)
        info = load_info(tmp_path)
        assert detect_version(tmp_path, info) is DatasetVersion.V2_1

    def test_detects_v3_0(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=2)
        info = load_info(tmp_path)
        assert detect_version(tmp_path, info) is DatasetVersion.V3_0

    def test_unsupported_version_string(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=1)
        raw = json.loads((tmp_path / "meta" / "info.json").read_text())
        raw["codebase_version"] = "v1.99"
        (tmp_path / "meta" / "info.json").write_text(json.dumps(raw))
        info = load_info(tmp_path)
        with pytest.raises(DatasetVersionError, match=re.escape("v1.99")):
            detect_version(tmp_path, info)

    def test_claims_v3_but_has_v2_shape(self, tmp_path: Path) -> None:
        # A v2.1-shaped dataset whose info.json lies and claims v3.0.
        build_v2_dataset(tmp_path, codebase_version="v2.1", num_episodes=1)
        raw = json.loads((tmp_path / "meta" / "info.json").read_text())
        raw["codebase_version"] = "v3.0"
        (tmp_path / "meta" / "info.json").write_text(json.dumps(raw))
        info = load_info(tmp_path)
        with pytest.raises(DatasetFormatError, match="lying about its version"):
            detect_version(tmp_path, info)

    def test_claims_v2_but_has_v3_shape(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=1)
        raw = json.loads((tmp_path / "meta" / "info.json").read_text())
        raw["codebase_version"] = "v2.1"
        (tmp_path / "meta" / "info.json").write_text(json.dumps(raw))
        info = load_info(tmp_path)
        with pytest.raises(DatasetFormatError, match="lying about its version"):
            detect_version(tmp_path, info)
