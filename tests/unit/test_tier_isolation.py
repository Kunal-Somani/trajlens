"""Invariant tests for Check.tier isolation and Check.formats scope (M1-C, ADR-011).

Two invariants are asserted structurally, not just by inspection:
  1. A QUALITY-tier check's findings can never move the trust score or grade,
     even when the check actually runs and actually fails, alongside the
     real built-in check suite.
  2. A format-scoped check is skipped (not run) against a dataset whose
     format_id is outside its declared formats, and runs normally when it is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.fixtures.builders import build_v3_dataset
from trajlens.checks.engine import CheckEngine
from trajlens.checks.protocol import Check, CheckContext, CheckResult, Severity, Tier
from trajlens.checks.registry import CheckRegistry
from trajlens.checks.registry import registry as global_registry
from trajlens.model import build_canonical_dataset
from trajlens.report.json_report import render_json
from trajlens.report.trust_score import compute_trust_score
from trajlens.sources.loader import SourceLoader

CTX = CheckContext(deep=False)


def _load(root: Path) -> Any:
    handle = SourceLoader().resolve(str(root))
    return build_canonical_dataset(handle)


class _AlwaysFailQualityCheck:
    id = "QUALITY.TEST_SYNTHETIC_ISOLATION"
    tier = Tier.QUALITY
    formats: frozenset[str] | None = None
    severity = Severity.FAIL
    category = "QUALITY"
    requires_video = False
    thread_safe = True

    def run(self, ds: Any, ctx: Any) -> CheckResult:
        return CheckResult(
            check_id=self.id,
            severity=Severity.FAIL,
            message="synthetic quality fail for tier isolation test",
            details={},
        )


def _grade_from(results: list[CheckResult]) -> str:
    doc = render_json("test/ref", "lerobot", "3.0", 3, 12, results)
    import json

    return str(json.loads(doc)["grade"])


def test_quality_finding_cannot_affect_integrity_score(tmp_path: Path) -> None:
    build_v3_dataset(tmp_path, num_episodes=3)
    ds = _load(tmp_path)

    baseline_engine = CheckEngine(global_registry)
    baseline_result = baseline_engine.run(ds, CTX)
    baseline_score = compute_trust_score(list(baseline_result.results))
    baseline_grade = _grade_from(list(baseline_result.results))

    reg = CheckRegistry()
    for check in global_registry.all_checks():
        reg.register(check)
    quality_check: Check = _AlwaysFailQualityCheck()  # type: ignore[assignment]
    reg.register(quality_check)

    engine = CheckEngine(reg)
    engine_result = engine.run(ds, CTX)

    quality_score = compute_trust_score(list(engine_result.results))
    quality_grade = _grade_from(list(engine_result.results))

    assert quality_score == baseline_score, (
        f"a QUALITY-tier FAIL must not move the trust score: "
        f"baseline={baseline_score}, with_quality_check={quality_score}"
    )
    assert quality_grade == baseline_grade, (
        f"a QUALITY-tier FAIL must not move the grade: "
        f"baseline={baseline_grade!r}, with_quality_check={quality_grade!r}"
    )

    fail_finding = next(
        (r for r in engine_result.results if r.check_id == "QUALITY.TEST_SYNTHETIC_ISOLATION"),
        None,
    )
    assert fail_finding is not None, "the synthetic QUALITY check must actually have run"
    assert fail_finding.severity is Severity.FAIL
    assert fail_finding.tier is Tier.QUALITY


class _RldsOnlyCheck:
    id = "TEST.RLDS_ONLY_FORMAT_SCOPE"
    tier = Tier.INTEGRITY
    formats: frozenset[str] | None = frozenset({"rlds"})
    severity = Severity.FAIL
    category = "TEST"
    requires_video = False
    thread_safe = True

    def run(self, ds: Any, ctx: Any) -> CheckResult:  # pragma: no cover
        raise AssertionError("should not run against a lerobot-format dataset")


class _LerobotOnlyCheck:
    id = "TEST.LEROBOT_ONLY_FORMAT_SCOPE"
    tier = Tier.INTEGRITY
    formats: frozenset[str] | None = frozenset({"lerobot"})
    severity = Severity.INFO
    category = "TEST"
    requires_video = False
    thread_safe = True

    def run(self, ds: Any, ctx: Any) -> CheckResult:
        return CheckResult(check_id=self.id, severity=Severity.INFO, message="ran")


def test_format_scoped_check_is_skipped_on_wrong_format(tmp_path: Path) -> None:
    build_v3_dataset(tmp_path, num_episodes=2)
    ds = _load(tmp_path)
    assert ds.format_id == "lerobot"

    reg = CheckRegistry()
    check: Check = _RldsOnlyCheck()  # type: ignore[assignment]
    reg.register(check)

    engine_result = CheckEngine(reg).run(ds, CTX)

    assert "TEST.RLDS_ONLY_FORMAT_SCOPE" in engine_result.skipped
    assert all(r.check_id != "TEST.RLDS_ONLY_FORMAT_SCOPE" for r in engine_result.results)


def test_format_scoped_check_runs_on_matching_format(tmp_path: Path) -> None:
    build_v3_dataset(tmp_path, num_episodes=2)
    ds = _load(tmp_path)
    assert ds.format_id == "lerobot"

    reg = CheckRegistry()
    check: Check = _LerobotOnlyCheck()  # type: ignore[assignment]
    reg.register(check)

    engine_result = CheckEngine(reg).run(ds, CTX)

    assert "TEST.LEROBOT_ONLY_FORMAT_SCOPE" not in engine_result.skipped
    assert any(r.check_id == "TEST.LEROBOT_ONLY_FORMAT_SCOPE" for r in engine_result.results)
