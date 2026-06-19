"""Opt-in integration test that hits the real Hugging Face Hub.

Excluded from the default run (pytest.ini marker, deselect with
-m "not integration"). lerobot/pusht is a well-known, widely-referenced
public LeRobot dataset; SourceLoader only pulls meta/ to resolve (see
loader.py), so this stays a small download regardless of the dataset's
full size.
"""

from __future__ import annotations

import pytest

from trajlens.sources.loader import SourceLoader
from trajlens.sources.version import DatasetVersion


@pytest.mark.integration
def test_resolves_real_hub_dataset() -> None:
    handle = SourceLoader().resolve("lerobot/pusht")
    assert handle.version in DatasetVersion
    assert handle.info.codebase_version
