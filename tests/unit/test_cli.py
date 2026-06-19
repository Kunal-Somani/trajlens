"""CLI tests for the parts of cli.py that are actually implemented at M1."""

from __future__ import annotations

from typer.testing import CliRunner

import trajlens
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


class TestUnimplementedCommandsFailClearly:
    def test_lint_raises_not_implemented(self) -> None:
        result = runner.invoke(app, ["lint", "some/dataset"])
        assert result.exit_code != 0
        assert isinstance(result.exception, NotImplementedError)

    def test_lint_error_message_names_milestone(self) -> None:
        result = runner.invoke(app, ["lint", "some/dataset"])
        assert "M4" in str(result.exception)

    def test_fix_raises_not_implemented(self) -> None:
        result = runner.invoke(app, ["fix", "/local/path"])
        assert result.exit_code != 0
        assert isinstance(result.exception, NotImplementedError)

    def test_web_raises_not_implemented(self) -> None:
        result = runner.invoke(app, ["web", "some/dataset"])
        assert result.exit_code != 0
        assert isinstance(result.exception, NotImplementedError)
