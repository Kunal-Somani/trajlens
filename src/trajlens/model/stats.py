"""Lazy access to meta/stats.json -- the global statistics handle.

Mirrors VideoShardHandle's pattern (sources/handles.py): a handle is cheap to
construct and does no I/O until load() is called. Stats correctness
(recomputed vs. stored, within tolerance) is a Check Engine concern (M4);
this module only gets a typed, safe handle to the raw structure in front of
checks, per 03_DATA_FORMAT_SPEC.md invariant 6.

Path and shape verified against the live lerobot 0.5.2 source
(datasets/utils.py STATS_PATH = "meta/stats.json"; io_utils.py write_stats/
load_stats: a JSON object keyed by feature name, each value a dict of
mean/std/min/max/count, present across all three supported versions).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trajlens.errors import DatasetFormatError
from trajlens.sources.paths import safe_join

STATS_RELATIVE_PATH = ("meta", "stats.json")


@dataclass(frozen=True, slots=True)
class StatsHandle:
    """A handle to a dataset's global meta/stats.json. No parsing happens at construction."""

    root: Path

    def load(self) -> Mapping[str, Mapping[str, Any]] | None:
        """Parse and return meta/stats.json, or None if the dataset has none.

        Raises DatasetFormatError if the file exists but is not valid JSON
        or is not a JSON object.
        """
        path = safe_join(self.root, *STATS_RELATIVE_PATH)
        if not path.is_file():
            return None

        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise DatasetFormatError(f"meta/stats.json is not valid JSON: {exc}") from exc

        if not isinstance(raw, dict):
            raise DatasetFormatError(
                f"meta/stats.json must be a JSON object keyed by feature name, "
                f"got {type(raw).__name__}"
            )
        return raw
