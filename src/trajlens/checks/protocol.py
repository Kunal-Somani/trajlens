"""Check Protocol, CheckResult, CheckContext, and Severity enum (02_ARCHITECTURE.md §3.3).

These are the core value types the entire check system depends on.  They live
in a leaf module with no internal imports so every other checks/ module can
import from here without creating cycles.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class Severity(enum.StrEnum):
    """Categorical outcome level for a single check result.

    ERROR  — the check itself could not be evaluated (ADR-003: never PASS on error).
    FAIL   — dataset is unsafe to train on as-is.
    WARN   — degraded or suspicious but potentially usable.
    INFO   — advisory / informational.
    """

    ERROR = "ERROR"
    FAIL = "FAIL"
    WARN = "WARN"
    INFO = "INFO"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return _SEVERITY_ORDER[self] < _SEVERITY_ORDER[other]

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return _SEVERITY_ORDER[self] <= _SEVERITY_ORDER[other]

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return _SEVERITY_ORDER[self] > _SEVERITY_ORDER[other]

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return _SEVERITY_ORDER[self] >= _SEVERITY_ORDER[other]


# Higher number = more severe.  ERROR > FAIL > WARN > INFO.
_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.WARN: 1,
    Severity.FAIL: 2,
    Severity.ERROR: 3,
}


@dataclass(frozen=True, slots=True)
class CheckContext:
    """Runtime context passed to every check's run() method.

    deep: when True, the engine was invoked with --deep; checks that gate
    expensive work behind this flag should inspect it.
    """

    deep: bool = False


@dataclass(frozen=True, slots=True)
class CheckResult:
    """The outcome of running a single check against a dataset.

    check_id  — the stable identifier of the check that produced this result.
    severity  — the categorical verdict (ERROR/FAIL/WARN/INFO).
    message   — a human-readable summary; must be actionable and non-empty.
    details   — optional structured detail dict for machine consumers (JSON output).
    """

    check_id: str
    severity: Severity
    message: str
    details: dict[str, object] = field(default_factory=dict)


# Import after Severity/CheckResult/CheckContext are defined to avoid a
# forward-reference issue — CanonicalDataset is declared in model/, which
# imports nothing from checks/.
from trajlens.model.canonical import CanonicalDataset  # noqa: E402


@runtime_checkable
class Check(Protocol):
    """The Check Protocol every built-in and community check must satisfy.

    id            — stable dot-namespaced identifier (e.g. "TEMPORAL.TIMESTAMP_MONOTONIC").
    severity      — the Severity that will be emitted on a failure finding.
    category      — the top-level category string ("STRUCTURAL", "TEMPORAL", etc.).
    requires_video — if True the engine will skip this check when no video is present.

    run() must return a CheckResult.  It must never raise; catch everything
    internally and return a CheckResult(severity=ERROR, ...) per ADR-003.
    """

    id: str
    severity: Severity
    category: str
    requires_video: bool

    def run(self, ds: CanonicalDataset, ctx: CheckContext) -> CheckResult: ...
