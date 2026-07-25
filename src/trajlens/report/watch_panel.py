"""Watch-mode rich Live summary panel (v0.4 T4).

The panel shows a continuously-updated per-run summary. New FAIL/WARN/ERROR
findings are printed by the caller via console.print() outside the Live
context (so they persist in the scroll buffer) before the panel is refreshed;
PASS episodes are counted but never individually reprinted.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel

from trajlens.checks.protocol import Severity


class WatchSummary:
    """Tracks running PASS/WARN/FAIL/ERROR counts across a watch session."""

    def __init__(self) -> None:
        self.episodes_linted: int = 0
        self._counts: dict[Severity, int] = dict.fromkeys(Severity, 0)
        self._pass_count: int = 0
        self._last_shard: Path | None = None
        self._live: Live | None = None

    @property
    def total_findings(self) -> int:
        return sum(self._counts.values())

    def record(self, shard_path: Path, worst: Severity) -> None:
        """Record one lint outcome for *shard_path* and refresh the panel."""
        self.episodes_linted += 1
        self._last_shard = shard_path
        if worst >= Severity.WARN:
            self._counts[worst] += 1
        else:
            self._pass_count += 1
        if self._live is not None:
            self._live.update(self._render())

    def _render(self) -> Panel:
        last = self._last_shard.name if self._last_shard is not None else "none yet"
        body = Group(
            f"Episodes linted: {self.episodes_linted}  |  "
            f"PASS: {self._pass_count}  "
            f"WARN: {self._counts[Severity.WARN]}  "
            f"FAIL: {self._counts[Severity.FAIL]}  "
            f"ERROR: {self._counts[Severity.ERROR]}",
            f"Last episode: {last}",
        )
        return Panel(body, title="trajlens watch")

    @contextmanager
    def live(self, console: Console) -> Iterator[None]:
        """Run *console* as a rich Live display for the duration of the context."""
        with Live(self._render(), console=console, refresh_per_second=4) as live:
            self._live = live
            try:
                yield
            finally:
                self._live = None
