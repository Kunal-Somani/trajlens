"""Typed parsing of meta/info.json — the trust boundary for dataset metadata.

Field names are grounded in the live lerobot source (datasets/utils.py
DatasetInfo, utils/constants.py DEFAULT_FEATURES), verified against the
editable lerobot 0.5.2 checkout, not guessed. Per-feature sub-schema
(dtype/shape/names) is intentionally not deeply validated here — that is
the canonical model's job (M3); this layer only confirms info.json is
well-formed enough to detect version and locate shards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from trajlens.errors import DatasetFormatError
from trajlens.sources.paths import safe_join

INFO_RELATIVE_PATH = ("meta", "info.json")


class DatasetInfoModel(BaseModel):
    """Validated view of meta/info.json's required-by-spec fields."""

    model_config = ConfigDict(extra="allow")

    codebase_version: str
    fps: int
    features: dict[str, dict[str, Any]]
    total_episodes: int | None = None
    total_frames: int | None = None


def load_info(root: Path) -> DatasetInfoModel:
    """Load and validate meta/info.json under *root*.

    Raises DatasetFormatError if the file is missing, not valid JSON, or
    does not contain the minimum required keys (codebase_version, fps,
    features) per 03_DATA_FORMAT_SPEC.md §2.
    """
    info_path = safe_join(root, *INFO_RELATIVE_PATH)
    if not info_path.is_file():
        raise DatasetFormatError(
            f"required metadata file not found: {info_path}. "
            f"Every LeRobotDataset must have a meta/info.json."
        )

    try:
        raw = json.loads(info_path.read_text())
    except json.JSONDecodeError as exc:
        raise DatasetFormatError(f"meta/info.json is not valid JSON: {exc}") from exc

    try:
        return DatasetInfoModel.model_validate(raw)
    except ValidationError as exc:
        raise DatasetFormatError(
            f"meta/info.json does not match the expected schema: {exc}"
        ) from exc
