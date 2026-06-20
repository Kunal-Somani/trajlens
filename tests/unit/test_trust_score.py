"""Unit tests for report/trust_score.py.

Validates every cap behaviour and boundary case per 07_EVALUATION_AND_ACCURACY.md §3.
"""

from __future__ import annotations

from trajlens.checks.protocol import CheckResult, Severity
from trajlens.report.trust_score import (
    _FAIL_CAP,
    _FAIL_PENALTY,
    _WARN_CAP,
    _WARN_PENALTY,
    SCORE_FORMULA_VERSION,
    compute_trust_score,
)


def _r(severity: Severity, check_id: str = "TEST.X") -> CheckResult:
    return CheckResult(check_id=check_id, severity=severity, message="test")


class TestTrustScoreClean:
    def test_no_results_scores_100(self) -> None:
        assert compute_trust_score([]) == 100

    def test_all_info_scores_100(self) -> None:
        results = [_r(Severity.INFO)] * 10
        assert compute_trust_score(results) == 100


class TestTrustScoreFailPenalty:
    def test_one_fail_deducts_30(self) -> None:
        assert compute_trust_score([_r(Severity.FAIL)]) == 70

    def test_two_fails_deducts_60(self) -> None:
        assert compute_trust_score([_r(Severity.FAIL)] * 2) == 40

    def test_three_fails_caps_at_60_not_90(self) -> None:
        # Cap is 60 total from FAILs. 3 * 30 = 90 would undercharge without the cap.
        assert compute_trust_score([_r(Severity.FAIL)] * 3) == 40

    def test_many_fails_still_caps_at_60(self) -> None:
        assert compute_trust_score([_r(Severity.FAIL)] * 10) == 40

    def test_fail_cap_value_matches_constant(self) -> None:
        # Prove the cap constant is used correctly: _FAIL_CAP / _FAIL_PENALTY FAILs
        # and one more should produce the same score.
        at_cap = _FAIL_CAP // _FAIL_PENALTY
        score_at_cap = compute_trust_score([_r(Severity.FAIL)] * at_cap)
        score_over_cap = compute_trust_score([_r(Severity.FAIL)] * (at_cap + 1))
        assert score_at_cap == score_over_cap


class TestTrustScoreWarnPenalty:
    def test_one_warn_deducts_5(self) -> None:
        assert compute_trust_score([_r(Severity.WARN)]) == 95

    def test_four_warns_deducts_20(self) -> None:
        assert compute_trust_score([_r(Severity.WARN)] * 4) == 80

    def test_five_warns_caps_at_20_not_25(self) -> None:
        # Cap is 20 total from WARNs. 5 * 5 = 25 without the cap.
        assert compute_trust_score([_r(Severity.WARN)] * 5) == 80

    def test_many_warns_still_caps_at_20(self) -> None:
        assert compute_trust_score([_r(Severity.WARN)] * 100) == 80

    def test_warn_cap_value_matches_constant(self) -> None:
        at_cap = _WARN_CAP // _WARN_PENALTY
        score_at_cap = compute_trust_score([_r(Severity.WARN)] * at_cap)
        score_over_cap = compute_trust_score([_r(Severity.WARN)] * (at_cap + 1))
        assert score_at_cap == score_over_cap


class TestTrustScoreErrorPenalty:
    def test_one_error_deducts_10(self) -> None:
        assert compute_trust_score([_r(Severity.ERROR)]) == 90

    def test_two_errors_deducts_20(self) -> None:
        assert compute_trust_score([_r(Severity.ERROR)] * 2) == 80

    def test_errors_have_no_cap(self) -> None:
        # ERRORs are not capped — ten ERRORs means the score floors at 0.
        assert compute_trust_score([_r(Severity.ERROR)] * 10) == 0


class TestTrustScoreCombined:
    def test_fail_and_warn_combined(self) -> None:
        # 1 FAIL (-30) + 1 WARN (-5) = 65
        results = [_r(Severity.FAIL), _r(Severity.WARN)]
        assert compute_trust_score(results) == 65

    def test_all_three_combined(self) -> None:
        # 1 FAIL (-30) + 1 WARN (-5) + 1 ERROR (-10) = 55
        results = [_r(Severity.FAIL), _r(Severity.WARN), _r(Severity.ERROR)]
        assert compute_trust_score(results) == 55

    def test_caps_applied_independently_then_combined(self) -> None:
        # 3 FAILs (capped at -60) + 5 WARNs (capped at -20) = 20
        results = [_r(Severity.FAIL)] * 3 + [_r(Severity.WARN)] * 5
        assert compute_trust_score(results) == 20

    def test_score_never_goes_below_zero(self) -> None:
        results = [_r(Severity.FAIL)] * 10 + [_r(Severity.WARN)] * 10 + [_r(Severity.ERROR)] * 10
        assert compute_trust_score(results) == 0

    def test_mixed_with_info_ignored(self) -> None:
        # INFO results contribute 0 deduction.
        results = [_r(Severity.FAIL), _r(Severity.INFO), _r(Severity.INFO)]
        assert compute_trust_score(results) == 70


class TestFormulaVersion:
    def test_formula_version_is_string(self) -> None:
        assert isinstance(SCORE_FORMULA_VERSION, str)

    def test_formula_version_is_1_0(self) -> None:
        # If this test fails someone changed the version without updating these tests.
        assert SCORE_FORMULA_VERSION == "1.0"
