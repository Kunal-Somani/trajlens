"""Regression tests: a baseline must not suppress a dataset getting WORSE.

The baseline's identity key is (check_id, episode_index, shard_path). But no
check populates details["shard_path"], and only three populate per_episode --
so for 14 of the 17 checks the identity collapses to (check_id, None, None).
Combined with a diff that treated "identity present in baseline" as
"unchanged", one baselined entry suppressed that check *entirely and
permanently*:

  - SCHEMA_CONSISTENCY baselined at 1 mismatch stayed "unchanged" at 500.
  - A check escalating WARN -> FAIL stayed "unchanged" (severity was not
    compared at all).

Both are exactly the regressions CI exists to catch, and both passed green.

The fix persists severity and a magnitude (finding_count) alongside each
identity and compares them explicitly: a baseline is a ceiling on accepted
badness, and anything above that ceiling surfaces as new. Findings that
shrink or soften still count as unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trajlens.baseline import BASELINE_SCHEMA_VERSION, BaselineStore
from trajlens.checks.protocol import CheckResult, Severity
from trajlens.errors import DatasetFormatError


def _result(
    check_id: str = "STRUCTURAL.SCHEMA_CONSISTENCY",
    *,
    severity: Severity = Severity.FAIL,
    total: int | None = None,
    episodes: list[int] | None = None,
) -> CheckResult:
    details: dict[str, object] = {}
    if total is not None:
        details["total_violations"] = total
    return CheckResult(
        check_id=check_id,
        severity=severity,
        message="test",
        details=details,
        per_episode=dict.fromkeys(episodes, "x") if episodes else None,
    )


class TestMagnitudeRegressionSurfaces:
    def test_growing_violation_count_is_reported_as_new(self) -> None:
        """1 mismatch accepted, 500 present now -> must NOT be 'unchanged'."""
        baseline = BaselineStore.from_results([_result(total=1)])
        diff = baseline.diff([_result(total=500)])

        assert len(diff.new) == 1, "a 500x worse finding must surface as new"
        assert diff.unchanged == []

    def test_identical_count_is_unchanged(self) -> None:
        baseline = BaselineStore.from_results([_result(total=7)])
        diff = baseline.diff([_result(total=7)])

        assert diff.new == []
        assert len(diff.unchanged) == 1

    def test_shrinking_count_is_still_unchanged_not_new(self) -> None:
        """A baseline is a ceiling on accepted badness, not an exact
        snapshot -- an improvement must not be reported as a new finding."""
        baseline = BaselineStore.from_results([_result(total=10)])
        diff = baseline.diff([_result(total=3)])

        assert diff.new == []
        assert len(diff.unchanged) == 1

    def test_growing_affected_episode_count_is_reported_as_new(self) -> None:
        """For checks with no total_violations, per-episode breadth is the
        magnitude. Note both share identity (check_id, min(episodes)=0, None),
        so this is genuinely testing magnitude, not identity change."""
        baseline = BaselineStore.from_results([_result(episodes=[0, 1])])
        diff = baseline.diff([_result(episodes=[0, 1, 2, 3, 4])])

        assert len(diff.new) == 1
        assert diff.unchanged == []


class TestSeverityEscalationSurfaces:
    def test_warn_escalating_to_fail_is_reported_as_new(self) -> None:
        """Severity was not compared at all before -- a check going
        WARN -> FAIL on the same identity reported 'unchanged'."""
        baseline = BaselineStore.from_results([_result(severity=Severity.WARN, total=1)])
        diff = baseline.diff([_result(severity=Severity.FAIL, total=1)])

        assert len(diff.new) == 1
        assert diff.new[0].severity is Severity.FAIL

    def test_fail_softening_to_warn_is_unchanged(self) -> None:
        baseline = BaselineStore.from_results([_result(severity=Severity.FAIL, total=1)])
        diff = baseline.diff([_result(severity=Severity.WARN, total=1)])

        assert diff.new == []
        assert len(diff.unchanged) == 1


class TestInfoResultsAreNotBaselined:
    def test_info_results_are_excluded_from_the_snapshot(self) -> None:
        """A baseline records accepted *problems*. Recording INFO (passing)
        results meant a check going green later showed up as a new finding,
        because its identity changes once its per-episode map disappears."""
        store = BaselineStore.from_results(
            [
                _result(severity=Severity.INFO),
                _result("TEMPORAL.TIMESTAMP_MONOTONIC", severity=Severity.WARN),
            ]
        )
        assert [f.check_id for f in store.findings] == ["TEMPORAL.TIMESTAMP_MONOTONIC"]

    def test_a_check_going_green_is_resolved_not_new(self, tmp_path: Path) -> None:
        """End-to-end: FAIL accepted, later INFO -> 'resolved', never 'new'."""
        baseline = BaselineStore.from_results([_result(episodes=[3], total=1)])
        diff = baseline.diff([_result(severity=Severity.INFO)])

        assert diff.new == [], "a check that now passes must never be a 'new finding'"
        assert len(diff.resolved) == 1


class TestPersistedMagnitudeRoundTrips:
    def test_severity_and_count_survive_save_and_load(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        BaselineStore.from_results([_result(severity=Severity.WARN, total=42)]).save(path)

        loaded = BaselineStore.load(path)
        assert loaded.findings[0].severity == "WARN"
        assert loaded.findings[0].finding_count == 42

        # And the loaded ceiling still detects a regression above it.
        assert len(loaded.diff([_result(severity=Severity.WARN, total=43)]).new) == 1
        assert loaded.diff([_result(severity=Severity.WARN, total=42)]).new == []


class TestSchemaVersionBump:
    def test_schema_version_is_2(self) -> None:
        """The identity contract gained severity/finding_count, so the
        version must have been bumped rather than silently reinterpreting
        existing v1 baseline files."""
        assert BASELINE_SCHEMA_VERSION == "2"

    def test_a_v1_baseline_file_is_rejected_with_an_actionable_message(
        self, tmp_path: Path
    ) -> None:
        import json

        path = tmp_path / "baseline.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "findings": [{"check_id": "STRUCTURAL.SCHEMA_CONSISTENCY"}],
                }
            )
        )
        with pytest.raises(DatasetFormatError, match="--update-baseline"):
            BaselineStore.load(path)
