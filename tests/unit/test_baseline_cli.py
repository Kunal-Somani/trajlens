"""CLI integration tests for --baseline, --update-baseline, --show-unchanged."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tests.fixtures.builders import build_v3_real_video
from trajlens.baseline import BaselineStore
from trajlens.checks.protocol import CheckResult, Severity
from trajlens.cli import app

runner = CliRunner()

_FAIL = CheckResult(check_id="STRUCTURAL.X", severity=Severity.FAIL, message="fail")
_WARN = CheckResult(check_id="TEMPORAL.Y", severity=Severity.WARN, message="warn")


class TestBaselineExitCode:
    def test_no_new_findings_exits_0_even_with_existing(self, tmp_path: Path) -> None:
        build_v3_real_video(tmp_path)
        baseline_path = tmp_path / "baseline.json"
        BaselineStore.from_results([_FAIL]).save(baseline_path)

        with patch("trajlens.checks.engine.CheckEngine.run", return_value=[_FAIL]):
            result = runner.invoke(app, ["lint", str(tmp_path), "--baseline", str(baseline_path)])
        assert result.exit_code == 0

    def test_new_findings_drive_exit_code_by_worst_severity(self, tmp_path: Path) -> None:
        build_v3_real_video(tmp_path)
        baseline_path = tmp_path / "baseline.json"
        BaselineStore.from_results([]).save(baseline_path)

        with patch("trajlens.checks.engine.CheckEngine.run", return_value=[_WARN]):
            result = runner.invoke(app, ["lint", str(tmp_path), "--baseline", str(baseline_path)])
        assert result.exit_code == 1

        with patch("trajlens.checks.engine.CheckEngine.run", return_value=[_FAIL]):
            result = runner.invoke(app, ["lint", str(tmp_path), "--baseline", str(baseline_path)])
        assert result.exit_code == 2

    def test_baseline_new_shown_but_unchanged_suppressed_by_default(self, tmp_path: Path) -> None:
        build_v3_real_video(tmp_path)
        baseline_path = tmp_path / "baseline.json"
        BaselineStore.from_results([_FAIL]).save(baseline_path)

        with patch("trajlens.checks.engine.CheckEngine.run", return_value=[_FAIL, _WARN]):
            result = runner.invoke(app, ["lint", str(tmp_path), "--baseline", str(baseline_path)])
        assert "[NEW]" in result.output
        assert "TEMPORAL.Y" in result.output
        assert "[UNCHANGED]" not in result.output


class TestShowUnchangedRequiresBaseline:
    def test_show_unchanged_without_baseline_exits_2(self, tmp_path: Path) -> None:
        build_v3_real_video(tmp_path)
        result = runner.invoke(app, ["lint", str(tmp_path), "--show-unchanged"])
        assert result.exit_code == 2

    def test_show_unchanged_with_baseline_shows_unchanged(self, tmp_path: Path) -> None:
        build_v3_real_video(tmp_path)
        baseline_path = tmp_path / "baseline.json"
        BaselineStore.from_results([_FAIL]).save(baseline_path)

        with patch("trajlens.checks.engine.CheckEngine.run", return_value=[_FAIL]):
            result = runner.invoke(
                app,
                [
                    "lint",
                    str(tmp_path),
                    "--baseline",
                    str(baseline_path),
                    "--show-unchanged",
                ],
            )
        assert "[UNCHANGED]" in result.output


class TestUpdateBaseline:
    def test_update_baseline_writes_file_and_exits_0(self, tmp_path: Path) -> None:
        build_v3_real_video(tmp_path)
        baseline_path = tmp_path / "baseline.json"

        with patch("trajlens.checks.engine.CheckEngine.run", return_value=[_FAIL]):
            result = runner.invoke(
                app, ["lint", str(tmp_path), "--update-baseline", str(baseline_path)]
            )
        assert result.exit_code == 0
        assert baseline_path.is_file()

        loaded = BaselineStore.load(baseline_path)
        assert [f.check_id for f in loaded.findings] == ["STRUCTURAL.X"]

    def test_update_baseline_exits_0_regardless_of_severity(self, tmp_path: Path) -> None:
        build_v3_real_video(tmp_path)
        baseline_path = tmp_path / "baseline.json"

        with patch("trajlens.checks.engine.CheckEngine.run", return_value=[_FAIL]):
            result = runner.invoke(
                app, ["lint", str(tmp_path), "--update-baseline", str(baseline_path)]
            )
        assert result.exit_code == 0
