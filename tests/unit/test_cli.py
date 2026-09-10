"""CLI tests (M1 basics + M4 lint wiring + M5 report/exit-code contract + v0.2 fix CLI)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import trajlens
from tests.fixtures.builders import (
    build_v3_dataset,
    build_v3_drift_and_wrong_stats,
    build_v3_interleaved_episode_data,
    build_v3_metadata_data_disagreement,
    build_v3_orphan_data_shard,
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
        assert "format_id" in data
        assert data["format_id"] == "lerobot"
        assert "format_version" in data
        assert data["format_version"] == "3.0"
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


class TestWebCommand:
    def test_web_errors_cleanly_when_extra_not_installed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If fastapi/uvicorn (the [web] extra) aren't installed, `web` must
        fail with a typed, actionable error, not a raw ImportError traceback.
        """
        build_v3_dataset(tmp_path)
        real_import = __import__

        def fake_import(
            name: str,
            globals: object = None,
            locals: object = None,
            fromlist: object = (),
            level: int = 0,
        ) -> object:
            if name == "trajlens.web.server":
                raise ModuleNotFoundError("No module named 'fastapi'")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr("builtins.__import__", fake_import)

        result = runner.invoke(app, ["web", str(tmp_path)])

        assert result.exit_code == 2
        assert "[web]" in result.output

    def test_web_errors_cleanly_on_missing_path(self) -> None:
        result = runner.invoke(app, ["web", "/nonexistent/path/to/dataset"])
        assert result.exit_code == 2
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "ERROR" in result.output

    def test_web_serves_and_respects_no_open(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        build_v3_dataset(tmp_path)
        opened: list[str] = []
        monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
        monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)

        result = runner.invoke(app, ["web", str(tmp_path), "--no-open", "--port", "8123"])

        assert result.exit_code == 0
        assert opened == []
        assert "8123" in result.output

    def test_web_opens_browser_without_no_open(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        build_v3_dataset(tmp_path)
        opened: list[str] = []
        monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
        monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)

        result = runner.invoke(app, ["web", str(tmp_path), "--port", "8124"])

        assert result.exit_code == 0
        assert opened == ["http://127.0.0.1:8124/"]


class TestFixDryRunDefault:
    def test_dry_run_default_writes_nothing(self, tmp_path: Path) -> None:
        """No --apply flag: dry-run is the default, and no --out is required."""
        source = tmp_path / "source"
        build_v3_metadata_data_disagreement(source, num_episodes=3)

        result = runner.invoke(app, ["fix", str(source)])

        assert result.exit_code == 0  # REPAIRABLE findings previewed successfully
        assert "would change" in result.output or "CHANGE" in result.output
        assert not (tmp_path / "repaired").exists()

    def test_dry_run_renders_diff(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_metadata_data_disagreement(source, num_episodes=3)

        result = runner.invoke(app, ["fix", str(source)])

        assert "REPAIR.EPISODE_REINDEX" in result.output
        assert "STRUCTURAL.METADATA_DATA_AGREEMENT" in result.output

    def test_dry_run_explicit_flag(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_metadata_data_disagreement(source, num_episodes=3)

        result = runner.invoke(app, ["fix", "--dry-run", str(source)])

        assert result.exit_code == 0
        assert not (tmp_path / "repaired").exists()

    def test_clean_dataset_reports_nothing_to_fix(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_dataset(source, num_episodes=3)

        result = runner.invoke(app, ["fix", str(source)])

        assert result.exit_code == 2  # no REPAIRABLE findings -- nothing to repair
        assert "nothing to fix" in result.output.lower() or "already clean" in result.output.lower()


class TestFixApply:
    def test_apply_without_out_errors_cleanly(self, tmp_path: Path) -> None:
        """--apply requires --out; must fail with a clean message, never a traceback."""
        source = tmp_path / "source"
        build_v3_metadata_data_disagreement(source, num_episodes=3)

        result = runner.invoke(app, ["fix", "--apply", str(source)])

        assert result.exit_code == 2
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "--out" in result.output

    def test_apply_out_equal_to_source_errors_cleanly(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_metadata_data_disagreement(source, num_episodes=3)

        result = runner.invoke(app, ["fix", "--apply", "--out", str(source), str(source)])

        assert result.exit_code == 2
        assert "same path" in result.output.lower() or "copy-on-write" in result.output.lower()

    def test_apply_with_out_produces_clean_reindex_repair(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        out = tmp_path / "repaired"
        build_v3_metadata_data_disagreement(source, num_episodes=3)

        result = runner.invoke(app, ["fix", "--apply", "--out", str(out), str(source)])

        assert result.exit_code == 0  # REPAIRABLE finding applied successfully
        assert out.is_dir()

        relint = runner.invoke(app, ["lint", "--json", str(out)])
        data = json.loads(relint.output)
        agreement = next(
            r for r in data["results"] if r["check_id"] == "STRUCTURAL.METADATA_DATA_AGREEMENT"
        )
        assert agreement["severity"] == "INFO"

    def test_apply_missing_path_exits_2(self, tmp_path: Path) -> None:
        out = tmp_path / "repaired"
        result = runner.invoke(
            app, ["fix", "--apply", "--out", str(out), "/nonexistent/path/to/dataset"]
        )
        assert result.exit_code == 2

    def test_unrepairable_dataset_exits_1_with_clean_message(self, tmp_path: Path) -> None:
        """Interleaved-data fixture (episode_reindex's refusal path) surfaces as
        a clean message, never a Python traceback, and writes no output. Exit 1:
        a REPAIRABLE finding (episode_reindex's target check) failed to apply.
        """
        source = tmp_path / "source"
        out = tmp_path / "repaired"
        build_v3_interleaved_episode_data(source, num_episodes=2)

        result = runner.invoke(app, ["fix", "--apply", "--out", str(out), str(source)])

        assert result.exit_code == 1
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "cannot be safely repaired" in result.output or "interleaved" in result.output
        assert not out.exists()

    def test_unrepairable_dataset_dry_run_also_exits_1(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_interleaved_episode_data(source, num_episodes=2)

        result = runner.invoke(app, ["fix", str(source)])

        assert result.exit_code == 1


class TestFixJson:
    def test_json_dry_run_schema(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_metadata_data_disagreement(source, num_episodes=3)

        result = runner.invoke(app, ["fix", "--json", str(source)])
        data = json.loads(result.output)

        assert data["ref"] == str(source)
        assert data["dry_run"] is True
        assert data["applicable"] is True
        assert data["output_path"] is None
        assert isinstance(data["fixers"], list)
        fixer_ids = {f["fixer_id"] for f in data["fixers"]}
        assert "REPAIR.EPISODE_REINDEX" in fixer_ids

    def test_json_apply_schema(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        out = tmp_path / "repaired"
        build_v3_metadata_data_disagreement(source, num_episodes=3)

        result = runner.invoke(app, ["fix", "--apply", "--out", str(out), "--json", str(source)])
        data = json.loads(result.output)

        assert data["dry_run"] is False
        assert data["output_path"] == str(out)
        reindex = next(f for f in data["fixers"] if f["fixer_id"] == "REPAIR.EPISODE_REINDEX")
        assert reindex["applied"] is True
        assert reindex["frames_corrected"] is not None

    def test_json_clean_dataset_not_applicable(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_dataset(source, num_episodes=3)

        result = runner.invoke(app, ["fix", "--json", str(source)])
        data = json.loads(result.output)

        assert data["applicable"] is False

    def test_json_usage_error_schema(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_metadata_data_disagreement(source, num_episodes=3)

        result = runner.invoke(app, ["fix", "--apply", "--json", str(source)])
        data = json.loads(result.output)

        assert data["error_category"]
        assert data["error_message"]
        assert data["fixers"] == []

    def test_json_unrepairable_error_schema(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        out = tmp_path / "repaired"
        build_v3_interleaved_episode_data(source, num_episodes=2)

        result = runner.invoke(app, ["fix", "--apply", "--out", str(out), "--json", str(source)])
        data = json.loads(result.output)

        assert data["error_category"] == "RepairError"
        assert data["error_message"]
        assert not out.exists()

    def test_json_out_equal_to_source_error_schema(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_metadata_data_disagreement(source, num_episodes=3)

        result = runner.invoke(app, ["fix", "--apply", "--out", str(source), "--json", str(source)])
        data = json.loads(result.output)

        assert data["error_category"] == "UsageError"
        assert data["error_message"]
        assert data["fixers"] == []

    def test_json_load_error_schema(self) -> None:
        result = runner.invoke(app, ["fix", "--json", "/nonexistent/path/to/dataset"])
        data = json.loads(result.output)

        assert data["error_category"]
        assert data["error_message"]
        assert data["fixers"] == []


class TestFixCompositionOrder:
    """The test most likely to expose a fixer-chaining design flaw.

    A dataset with TWO different findings (timestamp drift + stats
    divergence) must have BOTH cleared after fix --apply, with no new
    WARN/FAIL introduced -- proving REPAIR.STATS_RECOMPUTE ran against
    REPAIR.TIMESTAMP_DEDRIFT's already-corrected output, not the original
    drifted data (see repair/orchestrator.py's composition-order rationale).
    """

    def test_drift_and_stats_both_clear_after_apply(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        out = tmp_path / "repaired"
        build_v3_drift_and_wrong_stats(source, num_episodes=3, drift_per_frame=5e-5)

        result = runner.invoke(app, ["fix", "--apply", "--out", str(out), "--json", str(source)])
        assert result.exit_code == 0
        plan = json.loads(result.output)
        fixer_ids = {f["fixer_id"] for f in plan["fixers"]}
        assert "REPAIR.TIMESTAMP_DEDRIFT" in fixer_ids
        assert "REPAIR.STATS_RECOMPUTE" in fixer_ids
        for f in plan["fixers"]:
            if f["fixer_id"] in ("REPAIR.TIMESTAMP_DEDRIFT", "REPAIR.STATS_RECOMPUTE"):
                assert f["applied"] is True
                assert not f["is_noop"]

        pre_relint = runner.invoke(app, ["lint", "--json", str(source)])
        pre_data = json.loads(pre_relint.output)
        pre_bad_ids = {
            r["check_id"] for r in pre_data["results"] if r["severity"] in ("WARN", "FAIL", "ERROR")
        }

        post_relint = runner.invoke(app, ["lint", "--json", str(out)])
        post_data = json.loads(post_relint.output)
        post_by_id = {r["check_id"]: r["severity"] for r in post_data["results"]}

        assert post_by_id["KNOWNBUG.TIMESTAMP_DRIFT"] == "INFO"
        assert post_by_id["STATISTICAL.STATS_MATCH_DATA"] == "INFO"

        post_bad_ids = {
            check_id
            for check_id, severity in post_by_id.items()
            if severity in ("WARN", "FAIL", "ERROR")
        }
        new_findings = post_bad_ids - pre_bad_ids
        assert not new_findings, f"fix --apply introduced new findings: {new_findings}"


class TestFixHelp:
    """--help output is rendered by typer's own rich-based console, not
    render_fix_terminal, so it can't take a console= override the way
    tests/unit/test_fix_report.py's renderer tests can. typer.rich_utils
    bakes FORCE_TERMINAL/COLOR_SYSTEM/MAX_WIDTH into module-level globals
    at import time from GITHUB_ACTIONS/FORCE_COLOR/PY_COLORS env vars
    (typer/rich_utils.py) -- GitHub Actions sets GITHUB_ACTIONS, which
    forces ANSI color codes and can split/style a literal substring like
    "--only" into separate escape-coded spans, so a plain `"--only" in
    result.output` check that passes locally (no such env var set) can
    fail in CI. Fixed here by directly overriding those already-imported
    globals for the duration of the test, forcing the same deterministic,
    uncolored, wide rendering in every environment -- the same "force a
    deterministic console" principle as test_fix_report.py's
    Console(force_terminal=..., width=...), applied at the one point
    where typer's own internal console isn't otherwise reachable. Any
    future test asserting literal substrings against --help (or other
    rich/typer-rendered) output should use this fixture.
    """

    @pytest.fixture(autouse=True)
    def _deterministic_help_rendering(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import typer.rich_utils as rich_utils

        monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", False)
        monkeypatch.setattr(rich_utils, "COLOR_SYSTEM", None)
        monkeypatch.setattr(rich_utils, "MAX_WIDTH", 200)

    def test_help_mentions_mid_chain_noop_semantics(self) -> None:
        result = runner.invoke(app, ["fix", "--help"])
        assert result.exit_code == 0
        assert "no-op if an earlier repair already resolved its finding" in result.output

    def test_help_lists_only_except_quarantine_flags(self) -> None:
        result = runner.invoke(app, ["fix", "--help"])
        assert "--only" in result.output
        assert "--except" in result.output
        assert "--quarantine" in result.output


class TestFixOnlyExcept:
    def test_only_invalid_id_exits_2_with_valid_ids_listed(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_dataset(source, num_episodes=3)

        result = runner.invoke(app, ["fix", str(source), "--only", "not_a_real_fixer"])

        assert result.exit_code == 2
        assert "not_a_real_fixer" in result.output
        assert "REPAIR.EPISODE_REINDEX" in result.output  # valid ids listed
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_except_invalid_id_exits_2(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_dataset(source, num_episodes=3)

        result = runner.invoke(app, ["fix", str(source), "--except", "not_a_real_fixer"])

        assert result.exit_code == 2
        assert "not_a_real_fixer" in result.output

    def test_same_id_in_only_and_except_exits_2_before_any_work(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_dataset(source, num_episodes=3)

        result = runner.invoke(
            app,
            [
                "fix",
                str(source),
                "--only",
                "REPAIR.TASK_INDEX_REPAIR",
                "--except",
                "REPAIR.TASK_INDEX_REPAIR",
            ],
        )

        assert result.exit_code == 2
        assert "REPAIR.TASK_INDEX_REPAIR" in result.output
        assert "contradiction" in result.output.lower()

    def test_invalid_selection_json_error_schema(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_dataset(source, num_episodes=3)

        result = runner.invoke(app, ["fix", str(source), "--only", "not_a_real_fixer", "--json"])
        data = json.loads(result.output)

        assert data["error_category"] == "UsageError"
        assert "not_a_real_fixer" in data["error_message"]
        assert data["fixers"] == []

    def test_only_comma_separated_selects_multiple_fixers(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_orphan_data_shard(source)

        result = runner.invoke(
            app,
            [
                "fix",
                str(source),
                "--json",
                "--only",
                "REPAIR.TASK_INDEX_REPAIR,REPAIR.ORPHAN_SHARD_REPORT",
            ],
        )
        data = json.loads(result.output)

        fixer_ids = {f["fixer_id"] for f in data["fixers"]}
        assert fixer_ids == {"REPAIR.TASK_INDEX_REPAIR", "REPAIR.ORPHAN_SHARD_REPORT"}

    def test_only_reaches_fixer_whose_check_never_fires(self, tmp_path: Path) -> None:
        """REPAIR.VIDEO_METADATA_SYNC's target check is unimplemented, so it can
        never be selected by default -- --only is the only way to invoke it.
        Needs a real decodable video (VideoMetadataSyncFixer's own precondition),
        not build_v3_dataset's placeholder MP4 stub."""
        source = tmp_path / "source"
        build_v3_real_video(source)

        result = runner.invoke(
            app, ["fix", str(source), "--json", "--only", "REPAIR.VIDEO_METADATA_SYNC"]
        )
        data = json.loads(result.output)

        assert data["fixers"][0]["fixer_id"] == "REPAIR.VIDEO_METADATA_SYNC"

    def test_except_excludes_default_selected_fixer(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_metadata_data_disagreement(source, num_episodes=3)

        result = runner.invoke(
            app, ["fix", str(source), "--json", "--except", "REPAIR.EPISODE_REINDEX"]
        )
        data = json.loads(result.output)

        assert data["fixers"] == []
        assert data["applicable"] is False


class TestFixQuarantine:
    def test_apply_with_quarantine_moves_orphan_shard(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        out = tmp_path / "repaired"
        build_v3_orphan_data_shard(source)

        result = runner.invoke(
            app, ["fix", "--apply", "--out", str(out), "--quarantine", str(source)]
        )

        assert result.exit_code == 0
        assert (out / ".trajlens-quarantine" / "quarantine_manifest.json").is_file()

    def test_apply_without_quarantine_leaves_orphan_in_place(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        out = tmp_path / "repaired"
        build_v3_orphan_data_shard(source)

        result = runner.invoke(app, ["fix", "--apply", "--out", str(out), str(source)])

        assert result.exit_code == 0
        assert not (out / ".trajlens-quarantine").exists()
        assert (out / "data" / "chunk-000" / "file-001.parquet").is_file()
