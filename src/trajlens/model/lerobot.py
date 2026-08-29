"""LeRobot format entry point -- routes a SourceHandle to its version-specific adapter.

CanonicalDataset and the Check Engine never branch on DatasetVersion again
(ADR-002); this is the one place that does, so build_v2/build_v3 stay free
of it themselves.
"""

from __future__ import annotations

from trajlens.model.canonical import CanonicalDataset
from trajlens.model.lerobot_v2 import build_v2
from trajlens.model.lerobot_v3 import build_v3
from trajlens.sources.loader import SourceHandle
from trajlens.sources.version import DatasetVersion


def build_canonical_dataset(handle: SourceHandle) -> CanonicalDataset:
    """Translate *handle*'s raw, version-specific metadata into a CanonicalDataset.

    Raises DatasetFormatError if required metadata files are missing or
    malformed, or if a declared episode count exceeds the resource bound.
    """
    if handle.version is DatasetVersion.V3_0:
        return build_v3(handle)
    return build_v2(handle)
