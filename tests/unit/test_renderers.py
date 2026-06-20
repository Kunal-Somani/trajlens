"""Unit tests for report renderers (terminal, JSON, HTML, SARIF).

Tests call the render functions directly without going through the CLI
so failures localize to the specific renderer, not the CLI layer.
"""

from __future__ import annotations

import json

from rich.console import Console

from trajlens.checks.protocol import CheckResult, Severity
from trajlens.report.html_report import render_html
from trajlens.report.json_report import render_json
from trajlens.report.sarif import render_sarif
from trajlens.report.terminal import render_terminal
from trajlens.sources.version import DatasetVersion

_CLEAN = [
    CheckResult(check_id="STRUCTURAL.VERSION_DETECTED", severity=Severity.INFO, message="v3.0"),
    CheckResult(check_id="TEMPORAL.TIMESTAMP_MONOTONIC", severity=Severity.INFO, message="ok"),
]

_WITH_FAIL = [
    CheckResult(check_id="STRUCTURAL.VERSION_DETECTED", severity=Severity.INFO, message="v3.0"),
    CheckResult(
        check_id="STRUCTURAL.METADATA_DATA_AGREEMENT",
        severity=Severity.FAIL,
        message="span mismatch on episode 0",
    ),
]

_WITH_WARN = [
    CheckResult(
        check_id="TEMPORAL.TIMESTAMP_SPACING",
        severity=Severity.WARN,
        message="spacing deviation",
    ),
]

_WITH_ERROR = [
    CheckResult(check_id="TEST.CRASH", severity=Severity.ERROR, message="check could not run"),
]

_VER = DatasetVersion.V3_0


class TestTerminalRenderer:
    def _render(self, results: list[CheckResult]) -> str:
        buf = Console(file=None, record=True, highlight=False, markup=False)
        render_terminal("test/ref", _VER, 5, 100, results, console=buf)
        return buf.export_text()

    def test_clean_shows_pass(self) -> None:
        out = self._render(_CLEAN)
        assert "PASS" in out

    def test_fail_shows_fail(self) -> None:
        out = self._render(_WITH_FAIL)
        assert "FAIL" in out

    def test_shows_check_id(self) -> None:
        out = self._render(_WITH_FAIL)
        assert "STRUCTURAL.METADATA_DATA_AGREEMENT" in out

    def test_shows_trust_score(self) -> None:
        out = self._render(_CLEAN)
        assert "100" in out  # clean dataset scores 100

    def test_fail_trust_score_is_70(self) -> None:
        out = self._render(_WITH_FAIL)
        # 1 FAIL -> 100 - 30 = 70
        assert "70" in out

    def test_shows_ref(self) -> None:
        out = self._render(_CLEAN)
        assert "test/ref" in out

    def test_shows_version(self) -> None:
        out = self._render(_CLEAN)
        assert "v3.0" in out

    def test_num_frames_none_shows_unknown(self) -> None:
        buf = Console(file=None, record=True, highlight=False, markup=False)
        render_terminal("test/ref", _VER, 5, None, _CLEAN, console=buf)
        out = buf.export_text()
        assert "unknown" in out

    def test_warn_shows_warn(self) -> None:
        out = self._render(_WITH_WARN)
        assert "WARN" in out

    def test_error_shows_error(self) -> None:
        out = self._render(_WITH_ERROR)
        assert "ERROR" in out


class TestJsonRenderer:
    def test_clean_dataset_grade_pass(self) -> None:
        doc = json.loads(render_json("r", _VER, 1, 10, _CLEAN))
        assert doc["grade"] == "PASS"

    def test_fail_dataset_grade_fail(self) -> None:
        doc = json.loads(render_json("r", _VER, 1, 10, _WITH_FAIL))
        assert doc["grade"] == "FAIL"

    def test_warn_dataset_grade_warn(self) -> None:
        doc = json.loads(render_json("r", _VER, 1, 10, _WITH_WARN))
        assert doc["grade"] == "WARN"

    def test_error_dataset_grade_error(self) -> None:
        doc = json.loads(render_json("r", _VER, 1, 10, _WITH_ERROR))
        assert doc["grade"] == "ERROR"

    def test_trust_score_clean_is_100(self) -> None:
        doc = json.loads(render_json("r", _VER, 1, 10, _CLEAN))
        assert doc["trust_score"] == 100

    def test_trust_score_fail_is_70(self) -> None:
        doc = json.loads(render_json("r", _VER, 1, 10, _WITH_FAIL))
        assert doc["trust_score"] == 70

    def test_schema_has_required_fields(self) -> None:
        doc = json.loads(render_json("my/ref", _VER, 3, 90, _CLEAN))
        assert doc["ref"] == "my/ref"
        assert doc["version"] == "v3.0"
        assert "score_formula_version" in doc
        assert doc["num_episodes"] == 3
        assert doc["num_frames"] == 90
        assert isinstance(doc["results"], list)

    def test_results_have_category(self) -> None:
        doc = json.loads(render_json("r", _VER, 1, 10, _WITH_FAIL))
        result_entry = next(
            r for r in doc["results"] if r["check_id"] == "STRUCTURAL.METADATA_DATA_AGREEMENT"
        )
        assert result_entry["category"] == "STRUCTURAL"

    def test_num_frames_none_serializes(self) -> None:
        doc = json.loads(render_json("r", _VER, 1, None, _CLEAN))
        assert doc["num_frames"] is None


class TestHtmlRenderer:
    def test_clean_contains_pass(self) -> None:
        html = render_html("r", _VER, 1, 10, _CLEAN)
        assert "PASS" in html

    def test_fail_contains_fail(self) -> None:
        html = render_html("r", _VER, 1, 10, _WITH_FAIL)
        assert "FAIL" in html

    def test_has_inline_style(self) -> None:
        html = render_html("r", _VER, 1, 10, _CLEAN)
        assert "<style>" in html

    def test_no_external_links(self) -> None:
        html = render_html("r", _VER, 1, 10, _CLEAN)
        # No external stylesheet references (link rel=stylesheet)
        assert 'rel="stylesheet"' not in html
        assert "rel='stylesheet'" not in html

    def test_contains_trust_score(self) -> None:
        html = render_html("r", _VER, 1, 10, _CLEAN)
        assert "100" in html  # trust score 100

    def test_contains_check_id(self) -> None:
        html = render_html("r", _VER, 1, 10, _WITH_FAIL)
        assert "STRUCTURAL.METADATA_DATA_AGREEMENT" in html

    def test_ref_is_escaped(self) -> None:
        html = render_html("<script>alert(1)</script>", _VER, 1, 10, _CLEAN)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_message_is_escaped(self) -> None:
        results = [
            CheckResult(
                check_id="TEST.X",
                severity=Severity.INFO,
                message='<img src=x onerror="alert(1)">',
            )
        ]
        html = render_html("r", _VER, 1, 10, results)
        assert '<img src=x onerror="alert(1)">' not in html

    def test_num_frames_none_shows_unknown(self) -> None:
        html = render_html("r", _VER, 1, None, _CLEAN)
        assert "unknown" in html


class TestSarifRenderer:
    def _parse(self, results: list[CheckResult]) -> dict[str, object]:
        return json.loads(render_sarif("test/ref", _VER, 1, 10, results))  # type: ignore[return-value]

    def test_top_level_schema_key(self) -> None:
        doc = self._parse(_CLEAN)
        assert "$schema" in doc
        assert "sarif" in str(doc["$schema"]).lower()

    def test_version_is_2_1_0(self) -> None:
        doc = self._parse(_CLEAN)
        assert doc["version"] == "2.1.0"

    def test_runs_is_list_of_one(self) -> None:
        doc = self._parse(_CLEAN)
        assert isinstance(doc["runs"], list)
        assert len(doc["runs"]) == 1

    def test_tool_driver_name(self) -> None:
        doc = self._parse(_CLEAN)
        assert doc["runs"][0]["tool"]["driver"]["name"] == "trajlens"  # type: ignore[index]

    def test_results_present(self) -> None:
        doc = self._parse(_WITH_FAIL)
        assert "results" in doc["runs"][0]  # type: ignore[index]
        assert len(doc["runs"][0]["results"]) > 0  # type: ignore[index]

    def test_result_has_rule_id(self) -> None:
        doc = self._parse(_WITH_FAIL)
        for r in doc["runs"][0]["results"]:  # type: ignore[index]
            assert "ruleId" in r

    def test_result_has_message_text(self) -> None:
        doc = self._parse(_WITH_FAIL)
        for r in doc["runs"][0]["results"]:  # type: ignore[index]
            assert "message" in r
            assert "text" in r["message"]

    def test_result_has_locations(self) -> None:
        doc = self._parse(_WITH_FAIL)
        for r in doc["runs"][0]["results"]:  # type: ignore[index]
            assert "locations" in r
            assert len(r["locations"]) > 0

    def test_fail_maps_to_error_level(self) -> None:
        doc = self._parse(_WITH_FAIL)
        levels = {r["level"] for r in doc["runs"][0]["results"]}  # type: ignore[index]
        assert "error" in levels

    def test_warn_maps_to_warning_level(self) -> None:
        doc = self._parse(_WITH_WARN)
        levels = {r["level"] for r in doc["runs"][0]["results"]}  # type: ignore[index]
        assert "warning" in levels

    def test_info_maps_to_note_level(self) -> None:
        doc = self._parse(_CLEAN)
        levels = {r["level"] for r in doc["runs"][0]["results"]}  # type: ignore[index]
        assert "note" in levels

    def test_rules_match_result_ids(self) -> None:
        doc = self._parse(_WITH_FAIL)
        run = doc["runs"][0]  # type: ignore[index]
        rule_ids = {r["id"] for r in run["tool"]["driver"]["rules"]}
        result_ids = {r["ruleId"] for r in run["results"]}
        assert result_ids.issubset(rule_ids)

    def test_empty_results_produces_valid_sarif(self) -> None:
        doc = self._parse([])
        assert doc["version"] == "2.1.0"
        assert doc["runs"][0]["results"] == []  # type: ignore[index]
