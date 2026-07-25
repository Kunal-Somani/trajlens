"""Perf suite (v0.4 T3): exercises trajlens at scale, gated behind -m perf.

Slow and not part of the default gate (scripts/run_tests.sh passes
-m "not perf"). Run manually: pytest tests/perf/ -m perf -v
Timing numbers belong in docs/performance.md, not test assertions here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.fixtures.builders import build_large_synthetic_dataset
from trajlens.cli import app

runner = CliRunner()


@pytest.mark.perf
class TestScaleLint:
    def test_lint_completes_on_large_dataset(self, tmp_path: Path) -> None:
        root = tmp_path / "large_dataset"
        root.mkdir()
        build_large_synthetic_dataset(root, num_episodes=100, rows_per_episode=10_000)

        result = runner.invoke(app, ["lint", str(root), "--json"])

        # trajlens lint exits via sys.exit(); exit codes 0/1/2 all mean the
        # run completed and produced a report (severity-driven, not a crash).
        assert result.exit_code in (0, 1, 2)
        payload = json.loads(result.stdout)
        assert len(payload["results"]) > 0
