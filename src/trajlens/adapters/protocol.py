"""FormatAdapter protocol: the plug-in surface for dataset formats.

trajlens reads only LeRobot today, but the check engine and canonical model
are already format-agnostic (model/canonical.py). This module defines the
contract a future format (RLDS, HDF5, Zarr, MCAP, rosbag) must implement to
plug in: detect a source, load it into a CanonicalDataset, and optionally
write one back out.

detect() returns None on non-match rather than raising. The registry
(adapters/registry.py) calls every registered adapter's detect() in a loop
and collects the non-None results; if detect() raised on a non-match instead,
that loop would abort on the first adapter that doesn't recognise the source,
never reaching the adapter that actually does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from trajlens.model.canonical import CanonicalDataset
from trajlens.sources.loader import SourceHandle


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What an adapter can do, declared statically per format."""

    readable: bool
    writable: bool
    streamable: bool
    repairable: bool
    lossless: bool
    lost_on_write: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FormatMatch:
    """A positive detection result: one adapter's claim on a source."""

    format_id: str
    format_version: str
    confidence: float
    evidence: str


@dataclass(frozen=True, slots=True)
class WriteResult:
    """The outcome of writing a CanonicalDataset out through an adapter."""

    output_path: Path
    episodes_written: int
    frames_written: int
    lost_dimensions: tuple[str, ...]


class FormatAdapter(Protocol):
    """The contract every format adapter (built-in or community) must satisfy."""

    format_id: str
    capabilities: Capabilities

    def detect(self, h: SourceHandle) -> FormatMatch | None: ...

    def load(self, h: SourceHandle) -> CanonicalDataset: ...

    def write(self, ds: CanonicalDataset, out: Path) -> WriteResult: ...
