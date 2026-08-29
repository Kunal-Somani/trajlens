"""FormatAdapterRegistry: decorator-free, append-only, populated at import time.

Mirrors checks/registry.py's pattern: a singleton registry populated by each
adapter module at import time, queried by detect_format() to resolve a
SourceHandle to exactly one format.

Ambiguity policy: if two or more registered adapters both claim to detect the
same source, detect_format() raises FormatDetectionError rather than picking
one. A tool that silently guesses the format of someone's dataset and is
wrong will corrupt it; we raise and let the caller decide.
"""

from __future__ import annotations

import structlog

from trajlens.adapters.protocol import FormatAdapter, FormatMatch
from trajlens.errors import FormatDetectionError, SourceResolutionError
from trajlens.sources.loader import SourceHandle

log = structlog.get_logger(__name__)


class FormatAdapterRegistry:
    """Append-only registry of FormatAdapter implementations, keyed by format_id."""

    def __init__(self) -> None:
        self._adapters: dict[str, FormatAdapter] = {}

    def register(self, adapter: FormatAdapter) -> FormatAdapter:
        """Register *adapter* under its format_id; raises if the id is already taken."""
        format_id = adapter.format_id
        if format_id in self._adapters:
            raise ValueError(
                f"An adapter with format_id {format_id!r} is already registered.  "
                f"format_ids must be unique — check the module load order."
            )
        self._adapters[format_id] = adapter
        log.debug("adapter.registered", format_id=format_id)
        return adapter

    def get(self, format_id: str) -> FormatAdapter | None:
        """Return the adapter registered under *format_id*, or None if not registered."""
        return self._adapters.get(format_id)

    def all_adapters(self) -> list[FormatAdapter]:
        """Return all registered adapters in stable registration order."""
        return list(self._adapters.values())

    def __len__(self) -> int:
        return len(self._adapters)

    def __contains__(self, format_id: object) -> bool:
        return format_id in self._adapters


# Module-level singleton.
registry = FormatAdapterRegistry()


def detect_format(handle: SourceHandle) -> FormatMatch:
    """Detect *handle*'s format by calling every registered adapter's detect().

    Raises SourceResolutionError if no adapter detects the source. Raises
    FormatDetectionError if two or more adapters both claim it, listing every
    candidate's format_id and evidence in the message.
    """
    matches = [
        match
        for adapter in registry.all_adapters()
        if (match := adapter.detect(handle)) is not None
    ]

    if not matches:
        raise SourceResolutionError(f"no adapter detected the format of {handle.root}")

    if len(matches) == 1:
        return matches[0]

    candidates = "; ".join(f"{m.format_id} ({m.evidence})" for m in matches)
    raise FormatDetectionError(
        f"ambiguous format for {handle.root}: {len(matches)} adapters detected it: "
        f"{candidates}. Resolve the ambiguity explicitly."
    )
