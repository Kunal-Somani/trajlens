"""Watcher: incremental per-episode lint as a recording rig writes shards (v0.4 T4).

Detects new or modified Parquet/MP4 shards under a local dataset root and
lints each changed episode immediately, discarding shard data after each
lint (O(1)-memory, matching the guarantee established in v0.4 T3). Every
detected path is containment-checked via safe_join before use (06 T1: a
malformed recording script could produce a symlink or traversal path).
"""

from __future__ import annotations

from pathlib import Path

import structlog
import watchfiles
from rich.console import Console

from trajlens.checks import CheckContext, CheckEngine, CheckRegistry, Severity
from trajlens.checks import registry as global_registry
from trajlens.errors import DatasetError, PathTraversalError
from trajlens.model import build_canonical_dataset
from trajlens.report.watch_panel import WatchSummary
from trajlens.sources.loader import SourceLoader
from trajlens.sources.paths import safe_join

log = structlog.get_logger(__name__)

# Data-reading categories run per-episode in watch mode. VIDEO is excluded by
# default (too slow for real-time recording-loop feedback); --deep opts in.
_WATCH_CHECK_CATEGORIES: frozenset[str] = frozenset(
    {"STRUCTURAL", "TEMPORAL", "STATISTICAL", "KNOWNBUG"}
)


def _watch_registry(*, deep: bool) -> CheckRegistry:
    """Build a registry scoped to watch mode's per-episode check subset."""
    categories = _WATCH_CHECK_CATEGORIES | ({"VIDEO"} if deep else set())
    scoped = CheckRegistry()
    for check in global_registry.all_checks():
        if check.category in categories:
            scoped.register(check)
    return scoped


class Watcher:
    """Watches a local dataset root and lints new/modified shards incrementally."""

    def __init__(self, dataset_root: Path, deep: bool = False) -> None:
        self._root = dataset_root
        self._deep = deep
        self._seen: dict[Path, float] = {}
        self._registry = _watch_registry(deep=deep)

    def _detect_changed_shards(self) -> list[Path]:
        """Return shard paths under the dataset root that are new or modified.

        Every candidate is passed through safe_join for containment before
        being considered — a symlink or relative-traversal path produced by a
        malformed recording script is rejected, not followed.
        """
        candidates: list[Path] = []
        for pattern, subdir in ((".parquet", "data"), (".mp4", "videos")):
            base = self._root / subdir
            if not base.is_dir():
                continue
            for path in base.rglob(f"*{pattern}"):
                relative = path.relative_to(self._root)
                try:
                    safe_path = safe_join(self._root, *relative.parts)
                except PathTraversalError:
                    log.error("watch.path_rejected", path=str(path))
                    continue
                candidates.append(safe_path)

        changed: list[Path] = []
        for path in candidates:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if self._seen.get(path) != mtime:
                changed.append(path)
        return changed

    def _lint_shard(self, shard_path: Path, console: Console, summary: WatchSummary) -> None:
        """Reload the dataset and lint the per-episode check subset for one shard."""
        try:
            handle = SourceLoader().resolve(str(self._root))
            ds = build_canonical_dataset(handle)
        except DatasetError as exc:
            console.print(f"[bold red]ERROR[/bold red] could not reload dataset: {exc}")
            summary.record(shard_path, Severity.ERROR)
            self._seen[shard_path] = shard_path.stat().st_mtime
            return

        ctx = CheckContext(deep=self._deep)
        engine = CheckEngine(self._registry)
        results = engine.run(ds, ctx)

        worst = Severity.INFO
        for result in results:
            if result.severity >= Severity.WARN:
                worst = max(worst, result.severity)
                style = {"ERROR": "bold red", "FAIL": "red", "WARN": "yellow"}[result.severity]
                console.print(
                    f"[{style}]{result.severity.value}[/{style}] {result.check_id} "
                    f"({shard_path.name}): {result.message}"
                )

        summary.record(shard_path, worst)
        self._seen[shard_path] = shard_path.stat().st_mtime

    def run(self, console: Console) -> None:
        """Watch the dataset root and lint changed shards until interrupted.

        Exits cleanly on KeyboardInterrupt: prints a summary line and
        returns, no traceback.
        """
        summary = WatchSummary()
        try:
            with summary.live(console):
                for _ in watchfiles.watch(self._root):
                    for shard_path in self._detect_changed_shards():
                        self._lint_shard(shard_path, console, summary)
        except KeyboardInterrupt:
            pass
        console.print(
            f"Watched {summary.episodes_linted} episodes; {summary.total_findings} findings"
        )
