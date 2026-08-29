"""LeRobotAdapter: the built-in, and currently only, FormatAdapter.

detect() delegates to the existing sources/version.py version-detection
logic rather than re-implementing it — that logic already cross-checks
codebase_version against on-disk shape (03_DATA_FORMAT_SPEC.md §2), so
confidence is 1.0 for every version: the version field is unambiguous once
validated.
"""

from __future__ import annotations

from pathlib import Path

from trajlens.adapters.protocol import Capabilities, FormatMatch, WriteResult
from trajlens.errors import DatasetVersionError
from trajlens.model.adapters import build_canonical_dataset
from trajlens.model.canonical import CanonicalDataset
from trajlens.sources.loader import SourceHandle
from trajlens.sources.version import DatasetVersion, detect_version

_FORMAT_VERSIONS = {
    DatasetVersion.V2_0: ("2.0", "meta/info.json codebase_version=2.0"),
    DatasetVersion.V2_1: ("2.1", "meta/info.json codebase_version=2.1"),
    DatasetVersion.V3_0: ("3.0", "meta/info.json codebase_version=3.0"),
}


class LeRobotAdapter:
    """FormatAdapter for LeRobotDataset v2.0/v2.1/v3.0."""

    format_id = "lerobot"
    capabilities = Capabilities(
        readable=True,
        writable=False,
        streamable=True,
        repairable=True,
        lossless=True,
        lost_on_write=(),
    )

    def detect(self, h: SourceHandle) -> FormatMatch | None:
        if not (h.root / "meta" / "info.json").is_file():
            return None
        try:
            version = detect_version(h.root, h.info)
        except DatasetVersionError:
            return None
        format_version, evidence = _FORMAT_VERSIONS[version]
        return FormatMatch(
            format_id=self.format_id,
            format_version=format_version,
            confidence=1.0,
            evidence=evidence,
        )

    def load(self, h: SourceHandle) -> CanonicalDataset:
        return build_canonical_dataset(h)

    def write(self, ds: CanonicalDataset, out: Path) -> WriteResult:
        raise NotImplementedError(
            "LeRobot write path is planned for v0.5 M3 — not yet implemented."
        )
