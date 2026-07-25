"""SARIF schema validation and exit-code contract tests (08_ROADMAP.md v0.4 T1).

The SARIF schema fixture is embedded under tests/fixtures/ — no network call in
this suite (05 §5). Fetched once from the OASIS sarif-spec repo, main branch,
sarif-2.1/schema/sarif-schema-2.1.0.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
from typer.testing import CliRunner

from tests.fixtures.builders import (
    build_v3_metadata_data_disagreement,
    build_v3_real_video,
)
from trajlens.checks.protocol import CheckResult, Severity
from trajlens.cli import app
from trajlens.report.sarif import render_sarif
from trajlens.sources.version import DatasetVersion

_SCHEMA_PATH = Path(__file__).parent.parent / "fixtures" / "sarif-schema-2.1.0.json"
_VER = DatasetVersion.V3_0

runner = CliRunner()


def _r(severity: Severity, check_id: str = "TEST.X") -> CheckResult:
    return CheckResult(check_id=check_id, severity=severity, message="test finding")


class TestSarifSchemaValidation:
    def test_clean_results_validate_against_sarif_schema(self) -> None:
        schema = json.loads(_SCHEMA_PATH.read_text())
        doc = json.loads(render_sarif("test/ref", _VER, 1, 10, []))
        jsonschema.validate(instance=doc, schema=schema)

    def test_mixed_severity_results_validate_against_sarif_schema(self) -> None:
        schema = json.loads(_SCHEMA_PATH.read_text())
        results = [
            _r(Severity.ERROR, "TEST.ERROR"),
            _r(Severity.FAIL, "TEST.FAIL"),
            _r(Severity.WARN, "TEST.WARN"),
            _r(Severity.INFO, "TEST.INFO"),
        ]
        doc = json.loads(render_sarif("test/ref", _VER, 3, 100, results))
        jsonschema.validate(instance=doc, schema=schema)


class TestExitCodeRegressionGuard:
    """Exit codes are a stable contract (08_ROADMAP.md T1 DoD): must not change
    without a major version bump, since the GitHub Action maps them to
    success/neutral/failure."""

    def test_clean_dataset_exits_0(self, tmp_path: Path) -> None:
        build_v3_real_video(tmp_path)
        result = runner.invoke(app, ["lint", str(tmp_path)])
        assert result.exit_code == 0

    def test_warn_dataset_exits_1(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        build_v3_real_video(tmp_path)
        warn_result = CheckResult(check_id="TEST.WARN", severity=Severity.WARN, message="warn")
        with patch("trajlens.checks.engine.CheckEngine.run", return_value=[warn_result]):
            result = runner.invoke(app, ["lint", str(tmp_path)])
        assert result.exit_code == 1

    def test_fail_dataset_exits_2(self, tmp_path: Path) -> None:
        build_v3_metadata_data_disagreement(tmp_path)
        result = runner.invoke(app, ["lint", str(tmp_path)])
        assert result.exit_code == 2


class TestGithubActionDocs:
    def test_docs_file_exists_with_copy_pasteable_snippet(self) -> None:
        docs_path = Path(__file__).parent.parent.parent / "docs" / "github-action.md"
        assert docs_path.exists()
        content = docs_path.read_text()
        assert "uses: Kunal-Somani/trajlens-action@v1" in content
        assert "dataset-ref" in content
