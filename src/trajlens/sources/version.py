"""Format version detection — cross-checks codebase_version against shape.

03_DATA_FORMAT_SPEC.md §2 is explicit that codebase_version alone is not
trustworthy (a malformed or lying info.json is untrusted input per
06_SECURITY_AND_THREAT_MODEL.md §1). detect_version() requires the claimed
version string to agree with the actual directory layout on disk:

  v3.0    meta/episodes/ is a directory of sharded parquet files
  v2.0/1  meta/episodes.jsonl is a single flat file

Path templates verified against the live lerobot 0.5.2 source
(datasets/utils.py: DEFAULT_EPISODES_PATH, LEGACY_EPISODES_PATH).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from trajlens.errors import DatasetFormatError, DatasetVersionError
from trajlens.sources.info import DatasetInfoModel

V3_EPISODES_DIR = ("meta", "episodes")
LEGACY_EPISODES_FILE = ("meta", "episodes.jsonl")


class DatasetVersion(StrEnum):
    V2_0 = "v2.0"
    V2_1 = "v2.1"
    V3_0 = "v3.0"


def detect_version(root: Path, info: DatasetInfoModel) -> DatasetVersion:
    """Detect and validate the dataset's format version under *root*.

    Raises DatasetVersionError if codebase_version is not one of the
    supported strings. Raises DatasetFormatError if codebase_version claims
    a version whose directory shape does not match what's actually on disk.
    """
    try:
        claimed = DatasetVersion(info.codebase_version)
    except ValueError:
        supported = ", ".join(v.value for v in DatasetVersion)
        raise DatasetVersionError(
            f"dataset declares codebase_version {info.codebase_version!r}, "
            f"which trajlens does not support. Supported versions: {supported}."
        ) from None

    has_v3_shape = root.joinpath(*V3_EPISODES_DIR).is_dir()
    has_legacy_shape = root.joinpath(*LEGACY_EPISODES_FILE).is_file()

    if claimed is DatasetVersion.V3_0:
        if not has_v3_shape:
            raise DatasetFormatError(
                "info.json declares codebase_version 'v3.0' but "
                "meta/episodes/ is not a directory of sharded parquet "
                "files as the v3.0 layout requires. The metadata may be "
                "corrupt or lying about its version."
            )
    else:
        if not has_legacy_shape:
            raise DatasetFormatError(
                f"info.json declares codebase_version {claimed.value!r} but "
                f"meta/episodes.jsonl is missing, which the v2.x layout "
                f"requires. The metadata may be corrupt or lying about its "
                f"version."
            )

    return claimed
