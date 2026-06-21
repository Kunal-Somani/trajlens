"""CLI tests (M1 basics + M4 lint wiring + M5 report/exit-code contract)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import trajlens
from tests.fixtures.builders import (
    build_v3_dataset,
    build_v3_metadata_data_disagreement,
    build_v3_real_video,
)
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


class TestLintTerminal:
    def test_lint_clean_dataset_outputs_grade(self, tmp_path: Path) -> None:
        build_v3_real_video(tmp_path)
        result = runner.invoke(app, ["lint", str(tmp_path)])
        # Real video -> DECODABLE_SPOTCHECK passes -> exit 0
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_lint_corrupt_dataset_exits_2(self, tmp_path: Path) -> None:
        build_v3_metadata_data_disagreement(tmp_path)
        result = runner.invoke(app, ["lint", str(tmp_path)])
        assert result.exit_code == 2
        assert "FAIL" in result.output

    def test_lint_missing_path_exits_2(self) -> None:
        result = runner.invoke(app, ["lint", "/nonexistent/path/to/dataset"])
        assert result.exit_code == 2

    def test_lint_output_contains_trust_score(self, tmp_path: Path) -> None:
        build_v3_real_video(tmp_path)
        result = runner.invoke(app, ["lint", str(tmp_path)])
        assert "Trust score" in result.output or "trust" in result.output.lower()

    def test_lint_output_contains_version(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        result = runner.invoke(app, ["lint", str(tmp_path)])
        assert "v3.0" in result.output


class TestLintExitCodes:
    """Exit code contract: 0=PASS, 1=WARN, 2=FAIL/ERROR.

    This contract is the CI integration point — an untested exit code is
    an unverified contract. Each exit code is exercised explicitly here.
    """

    def test_exit_0_on_clean_dataset(self, tmp_path: Path) -> None:
        """Real video + clean data -> all checks INFO/PASS -> exit 0."""
        build_v3_real_video(tmp_path)
        result = runner.invoke(app, ["lint", str(tmp_path)])
        assert result.exit_code == 0

    def test_exit_2_on_fail_dataset(self, tmp_path: Path) -> None:
        """Metadata disagreement -> STRUCTURAL.METADATA_DATA_AGREEMENT FAIL -> exit 2."""
        build_v3_metadata_data_disagreement(tmp_path)
        result = runner.invoke(app, ["lint", str(tmp_path)])
        assert result.exit_code == 2

    def test_exit_2_on_load_error(self) -> None:
        """Unresolvable ref -> DatasetError -> exit 2."""
        result = runner.invoke(app, ["lint", "/does/not/exist"])
        assert result.exit_code == 2

    def test_exit_1_on_warn_dataset(self, tmp_path: Path) -> None:
        """Mocked engine returning WARN -> exit 1."""
        from unittest.mock import patch

        from trajlens.checks.protocol import CheckResult, Severity

        build_v3_real_video(tmp_path)
        warn_result = CheckResult(check_id="TEST.WARN", severity=Severity.WARN, message="w")
        with patch("trajlens.checks.engine.CheckEngine.run", return_value=[warn_result]):
            result = runner.invoke(app, ["lint", str(tmp_path)])
            assert result.exit_code == 1


class TestLintJsonOutput:
    def test_json_flag_produces_valid_json(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        result = runner.invoke(app, ["lint", "--json", str(tmp_path)])
        # JSON output is parseable
        data = json.loads(result.output)
        assert "grade" in data
        assert "trust_score" in data
        assert "results" in data

    def test_json_flag_schema_fields(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        result = runner.invoke(app, ["lint", "--json", str(tmp_path)])
        data = json.loads(result.output)
        assert "ref" in data
        assert "version" in data
        assert data["version"] == "v3.0"
        assert "score_formula_version" in data
        assert isinstance(data["trust_score"], int)
        assert isinstance(data["results"], list)

    def test_json_fail_dataset_grade_is_fail(self, tmp_path: Path) -> None:
        build_v3_metadata_data_disagreement(tmp_path)
        result = runner.invoke(app, ["lint", "--json", str(tmp_path)])
        data = json.loads(result.output)
        assert data["grade"] == "FAIL"
        assert data["trust_score"] <= 70

    def test_json_clean_dataset_grade_is_pass(self, tmp_path: Path) -> None:
        build_v3_real_video(tmp_path)
        result = runner.invoke(app, ["lint", "--json", str(tmp_path)])
        data = json.loads(result.output)
        assert data["grade"] == "PASS"
        assert data["trust_score"] == 100

    def test_json_results_have_required_fields(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        result = runner.invoke(app, ["lint", "--json", str(tmp_path)])
        data = json.loads(result.output)
        for r in data["results"]:
            assert "check_id" in r
            assert "severity" in r
            assert "category" in r
            assert "message" in r

    def test_json_load_error_produces_parseable_json(self) -> None:
        """A load-time DatasetError (e.g. v2.x Hub dataset) must still emit
        structured JSON on stdout under --json, not just an stderr message.
        """
        result = runner.invoke(app, ["lint", "--json", "/nonexistent/path/to/dataset"])
        assert result.exit_code == 2
        data = json.loads(result.output)
        assert data["grade"] == "ERROR"
        assert data["results"] == []
        assert data["error_category"]
        assert data["error_message"]

    def test_json_load_error_has_no_stray_stdout_text(self) -> None:
        """stdout under --json must be JSON only, even on load failure."""
        result = runner.invoke(app, ["lint", "--json", "/nonexistent/path/to/dataset"])
        json.loads(result.output)  # raises if anything but JSON is on stdout


class TestLintHtmlReport:
    def test_html_report_creates_file(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        out = tmp_path / "report.html"
        runner.invoke(app, ["lint", "--report", str(out), str(tmp_path)])
        assert out.exists()

    def test_html_report_contains_grade(self, tmp_path: Path) -> None:
        build_v3_real_video(tmp_path)
        out = tmp_path / "report.html"
        runner.invoke(app, ["lint", "--report", str(out), str(tmp_path)])
        content = out.read_text()
        assert "PASS" in content

    def test_html_report_contains_trust_score(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        out = tmp_path / "report.html"
        runner.invoke(app, ["lint", "--report", str(out), str(tmp_path)])
        content = out.read_text()
        assert "Trust score" in content or "trust" in content.lower()

    def test_html_report_is_self_contained(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        out = tmp_path / "report.html"
        runner.invoke(app, ["lint", "--report", str(out), str(tmp_path)])
        content = out.read_text()
        # No external stylesheet or script references
        assert "href=" not in content or "http" not in content
        assert "<style>" in content

    def test_html_report_fail_dataset(self, tmp_path: Path) -> None:
        build_v3_metadata_data_disagreement(tmp_path)
        out = tmp_path / "report.html"
        runner.invoke(app, ["lint", "--report", str(out), str(tmp_path)])
        content = out.read_text()
        assert "FAIL" in content

    def test_html_report_and_terminal_run_together(self, tmp_path: Path) -> None:
        """--report should not suppress terminal output."""
        build_v3_dataset(tmp_path)
        out = tmp_path / "report.html"
        result = runner.invoke(app, ["lint", "--report", str(out), str(tmp_path)])
        assert out.exists()
        assert "trajlens lint" in result.output


class TestLintSarifReport:
    def test_sarif_creates_file(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        out = tmp_path / "results.sarif"
        runner.invoke(app, ["lint", "--sarif", str(out), str(tmp_path)])
        assert out.exists()

    def test_sarif_is_valid_json(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        out = tmp_path / "results.sarif"
        runner.invoke(app, ["lint", "--sarif", str(out), str(tmp_path)])
        data = json.loads(out.read_text())
        # SARIF 2.1.0 required top-level keys
        assert "$schema" in data
        assert data["version"] == "2.1.0"
        assert "runs" in data
        assert isinstance(data["runs"], list)
        assert len(data["runs"]) == 1

    def test_sarif_run_has_tool_and_results(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        out = tmp_path / "results.sarif"
        runner.invoke(app, ["lint", "--sarif", str(out), str(tmp_path)])
        data = json.loads(out.read_text())
        run = data["runs"][0]
        assert "tool" in run
        assert "driver" in run["tool"]
        assert run["tool"]["driver"]["name"] == "trajlens"
        assert "results" in run
        assert isinstance(run["results"], list)

    def test_sarif_results_have_required_fields(self, tmp_path: Path) -> None:
        build_v3_metadata_data_disagreement(tmp_path)
        out = tmp_path / "results.sarif"
        runner.invoke(app, ["lint", "--sarif", str(out), str(tmp_path)])
        data = json.loads(out.read_text())
        for r in data["runs"][0]["results"]:
            assert "ruleId" in r
            assert "level" in r
            assert "message" in r
            assert "text" in r["message"]
            assert "locations" in r

    def test_sarif_fail_maps_to_error_level(self, tmp_path: Path) -> None:
        build_v3_metadata_data_disagreement(tmp_path)
        out = tmp_path / "results.sarif"
        runner.invoke(app, ["lint", "--sarif", str(out), str(tmp_path)])
        data = json.loads(out.read_text())
        levels = {r["level"] for r in data["runs"][0]["results"]}
        assert "error" in levels


class TestUnimplementedCommands:
    def test_fix_raises_not_implemented(self) -> None:
        result = runner.invoke(app, ["fix", "/local/path"])
        assert result.exit_code != 0
        assert isinstance(result.exception, NotImplementedError)

    def test_web_raises_not_implemented(self) -> None:
        result = runner.invoke(app, ["web", "some/dataset"])
        assert result.exit_code != 0
        assert isinstance(result.exception, NotImplementedError)
