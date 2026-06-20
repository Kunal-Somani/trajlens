"""CheckEngine: selects applicable checks and runs them, enforcing ADR-003.

ADR-003 is the single most important rule here: a crashing check yields an
ERROR result; it never lets the exception propagate, and it never produces a
PASS.  Every check's run() call is wrapped in a broad try/except — the one
sanctioned place for catching Exception broadly in the codebase, because the
alternative (one bad check crashing the entire lint run) is worse.
"""

from __future__ import annotations

import structlog

from trajlens.checks.protocol import Check, CheckContext, CheckResult, Severity
from trajlens.checks.registry import CheckRegistry
from trajlens.model.canonical import CanonicalDataset

log = structlog.get_logger(__name__)


class CheckEngine:
    """Selects applicable checks and runs them, collecting CheckResults.

    Selection criteria (applied in order):
      1. Checks that require_video are skipped when the dataset has no cameras.
      2. Checks that require_video and deep=False in ctx are skipped unless
         the check is in the non-deep video set (currently only DECODABLE_SPOTCHECK).
      3. All other checks run unconditionally.

    ADR-003: any exception escaping a check's run() is caught here, logged,
    and converted to a CheckResult with severity=ERROR.  The exception type
    and message are preserved in the result's details dict.
    """

    def __init__(self, reg: CheckRegistry) -> None:
        self._registry = reg

    def run(self, ds: CanonicalDataset, ctx: CheckContext) -> list[CheckResult]:
        """Run all applicable checks; return one CheckResult per check."""
        results: list[CheckResult] = []
        has_video = len(ds.cameras) > 0

        for check in self._registry.all_checks():
            if check.requires_video and not has_video:
                log.debug("check.skipped.no_video", check_id=check.id)
                continue

            result = self._run_one(check, ds, ctx)
            results.append(result)
            log.info(
                "check.result",
                check_id=result.check_id,
                severity=result.severity.value,
                message=result.message,
            )

        return results

    def _run_one(self, check: Check, ds: CanonicalDataset, ctx: CheckContext) -> CheckResult:
        """Run a single check, converting any exception to an ERROR result (ADR-003)."""
        try:
            return check.run(ds, ctx)
        except Exception as exc:
            exc_type = type(exc).__name__
            log.error(
                "check.crashed",
                check_id=check.id,
                exc_type=exc_type,
                exc_message=str(exc),
            )
            return CheckResult(
                check_id=check.id,
                severity=Severity.ERROR,
                message=(
                    f"Check could not be evaluated — it raised {exc_type}: {exc}. "
                    f"This is a trajlens bug; please report it."
                ),
                details={"exc_type": exc_type, "exc_message": str(exc)},
            )
