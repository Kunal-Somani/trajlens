"""Trust score computation (07_EVALUATION_AND_ACCURACY.md §3).

Formula (exact, versioned):

    base = 100
    for each FAIL result:   base -= 30   (cap at -60 total from FAILs)
    for each WARN result:   base -= 5    (cap at -20 total from WARNs)
    for each ERROR result:  base -= 10   (check could not run — uncertainty penalty)
    score = max(0, base)

SCORE_FORMULA_VERSION must be bumped on any formula change, per 07 §3:
"if it changes, the version bumps and old scores are not compared to new ones."
"""

from __future__ import annotations

from trajlens.checks.protocol import CheckResult, Severity

SCORE_FORMULA_VERSION = "1.0"

_FAIL_PENALTY = 30
_WARN_PENALTY = 5
_ERROR_PENALTY = 10

_FAIL_CAP = 60
_WARN_CAP = 20


def compute_trust_score(results: list[CheckResult]) -> int:
    """Return a trust score in [0, 100] for the given check results.

    The score is a coarse signal alongside the categorical grade — it never
    overrides the grade.  A dataset with any FAIL has score <= 70.
    """
    fail_count = sum(1 for r in results if r.severity is Severity.FAIL)
    warn_count = sum(1 for r in results if r.severity is Severity.WARN)
    error_count = sum(1 for r in results if r.severity is Severity.ERROR)

    fail_deduction = min(fail_count * _FAIL_PENALTY, _FAIL_CAP)
    warn_deduction = min(warn_count * _WARN_PENALTY, _WARN_CAP)
    error_deduction = error_count * _ERROR_PENALTY

    score = 100 - fail_deduction - warn_deduction - error_deduction
    return max(0, score)
