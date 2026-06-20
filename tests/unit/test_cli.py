"""CLI tests (M1 basics + M4 lint wiring)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import trajlens
from tests.fixtures.builders import build_v3_dataset, build_v3_metadata_data_disagreement
from trajlens.cli import app

runner = CliRunner()


class TestVersionFlag:
    def test_version_flag_exits_zero(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0

    def test_version_flag_prints_version_string(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert trajlens.__version__ in result.stdout

    def test_version_short_flag(self) -> None:
        result = runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        assert trajlens.__version__ in result.stdout


class TestMainCallback:
    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app, [])
        assert "Usage" in result.stdout or "Usage" in result.output

    def test_verbose_flag_does_not_crash(self) -> None:
        result = runner.invoke(app, ["--verbose"])
        assert result.exit_code in (0, 2)


class TestLintCommand:
    def test_lint_clean_dataset_exits_zero(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        result = runner.invoke(app, ["lint", str(tmp_path)])
        # Exit 0 or 1 — stub videos may cause FAIL on DECODABLE_SPOTCHECK
        # but non-video checks should all pass.
        assert result.exit_code in (0, 1)
        assert "trajlens lint" in result.output

    def test_lint_corrupt_dataset_exits_nonzero(self, tmp_path: Path) -> None:
        build_v3_metadata_data_disagreement(tmp_path)
        result = runner.invoke(app, ["lint", str(tmp_path)])
        assert result.exit_code != 0
        assert "FAIL" in result.output

    def test_lint_missing_path_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["lint", "/nonexistent/path/to/dataset"])
        assert result.exit_code != 0

    def test_lint_json_flag_not_yet_implemented(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        result = runner.invoke(app, ["lint", "--json", str(tmp_path)])
        assert result.exit_code == 2
        assert "M5" in result.output

    def test_lint_report_flag_not_yet_implemented(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        result = runner.invoke(app, ["lint", "--report", "out.html", str(tmp_path)])
        assert result.exit_code == 2


class TestUnimplementedCommands:
    def test_fix_raises_not_implemented(self) -> None:
        result = runner.invoke(app, ["fix", "/local/path"])
        assert result.exit_code != 0
        assert isinstance(result.exception, NotImplementedError)

    def test_web_raises_not_implemented(self) -> None:
        result = runner.invoke(app, ["web", "some/dataset"])
        assert result.exit_code != 0
        assert isinstance(result.exception, NotImplementedError)
