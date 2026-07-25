"""Unit tests for v0.4 T4: mocked watcher, idempotency, Hub-refusal, SIGINT, traversal.

watchfiles.watch is mocked at the Watcher level (Watcher.run patches the
module-level watchfiles.watch generator) so these tests never depend on real
inotify/FSEvents timing.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

from rich.console import Console
from typer.testing import CliRunner

from tests.fixtures.builders import build_v3_dataset
from trajlens.cli import app
from trajlens.watch import Watcher

runner = CliRunner()


def _mute_console() -> Console:
    return Console(quiet=True)


class TestNewShardTriggersLint:
    def test_new_parquet_shard_triggers_lint_for_that_path(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=2)
        watcher = Watcher(tmp_path)
        shard_path = tmp_path / "data" / "chunk-000" / "file-000.parquet"

        def fake_watch(_root: Path) -> Iterator[set[tuple[int, str]]]:
            yield {(1, str(shard_path))}

        with (
            patch("trajlens.watch.watchfiles.watch", side_effect=fake_watch),
            patch.object(watcher, "_lint_shard") as mock_lint,
        ):
            watcher.run(_mute_console())

        called_paths = [call.args[0] for call in mock_lint.call_args_list]
        assert shard_path.resolve() in called_paths


class TestIdempotency:
    def test_unchanged_mtime_is_not_relinted(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=2)
        watcher = Watcher(tmp_path)

        first_pass = watcher._detect_changed_shards()
        assert len(first_pass) > 0
        for path in first_pass:
            watcher._seen[path] = path.stat().st_mtime

        second_pass = watcher._detect_changed_shards()
        assert second_pass == []


class TestHubRefRefusal:
    def test_hub_ref_exits_2_with_usage_error(self) -> None:
        result = runner.invoke(app, ["watch", "org/dataset-name"])
        assert result.exit_code == 2
        assert "Hub" in result.output

    def test_nonexistent_local_path_exits_2(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist"
        result = runner.invoke(app, ["watch", str(missing)])
        assert result.exit_code == 2


class TestSigintCleanExit:
    def test_keyboard_interrupt_exits_cleanly_with_summary(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=1)
        watcher = Watcher(tmp_path)

        def fake_watch(_root: Path) -> Iterator[set[tuple[int, str]]]:
            raise KeyboardInterrupt
            yield  # pragma: no cover — unreachable, makes this a generator

        console = Console(record=True)
        with patch("trajlens.watch.watchfiles.watch", side_effect=fake_watch):
            watcher.run(console)  # no traceback raised — the assertion is that this returns

        output = console.export_text()
        assert "Watched" in output
        assert "findings" in output


class TestPathTraversalRejected:
    def test_symlink_escaping_root_is_excluded(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=1)
        outside = tmp_path.parent / "outside_shard.parquet"
        outside.write_bytes(b"\x00")
        try:
            data_dir = tmp_path / "data" / "chunk-000"
            symlink_path = data_dir / "escape.parquet"
            symlink_path.symlink_to(outside)

            watcher = Watcher(tmp_path)
            changed = watcher._detect_changed_shards()

            assert all(p.resolve() != outside.resolve() for p in changed)
        finally:
            outside.unlink(missing_ok=True)

    def test_dotdot_path_is_rejected_without_crash(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=1)
        watcher = Watcher(tmp_path)

        changed = watcher._detect_changed_shards()

        assert all(".." not in p.parts for p in changed)
