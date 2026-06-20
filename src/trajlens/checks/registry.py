"""CheckRegistry: decorator-based, append-only, populated at import time.

This is the one sanctioned place for global mutable state in trajlens
(05_ENGINEERING_STANDARDS.md §2).  The registry is append-only — once a check
is registered its id cannot be overwritten.  The singleton ``registry`` is what
the CheckEngine queries; individual check modules call
``registry.register(MyCheckClass)`` at module level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from trajlens.checks.protocol import Check

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)


class CheckRegistry:
    """Append-only registry of Check implementations, keyed by their id."""

    def __init__(self) -> None:
        self._checks: dict[str, Check] = {}

    def register(self, check_instance: Check) -> Check:
        """Register *check_instance* under its id; raises if the id is already taken.

        Intended as a module-level decorator target:

            @registry.register
            class MyCheck:
                id = "CATEGORY.NAME"
                ...

        The decorated class/instance is returned unchanged so it can still be
        imported directly from its defining module.
        """
        check_id = check_instance.id
        if check_id in self._checks:
            raise ValueError(
                f"A check with id {check_id!r} is already registered.  "
                f"Check ids must be unique — check the module load order."
            )
        self._checks[check_id] = check_instance
        log.debug("check.registered", check_id=check_id, severity=check_instance.severity.value)
        return check_instance

    def all_checks(self) -> list[Check]:
        """Return all registered checks in stable registration order."""
        return list(self._checks.values())

    def get(self, check_id: str) -> Check | None:
        """Return the check with *check_id*, or None if not registered."""
        return self._checks.get(check_id)

    def __len__(self) -> int:
        return len(self._checks)

    def __contains__(self, check_id: object) -> bool:
        return check_id in self._checks


# Module-level singleton — the only global mutable state in trajlens.
registry = CheckRegistry()
