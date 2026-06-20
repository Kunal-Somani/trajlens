"""Check sub-package: protocol, registry, engine, and individual check modules.

Importing this package registers all built-in checks via each sub-module's
module-level @registry.register decorator calls.  The registration happens at
import time, which is the one sanctioned place for append-only global mutable
state in trajlens (05_ENGINEERING_STANDARDS.md §2).

Public surface:
  - Check (Protocol)
  - CheckResult, CheckContext, Severity
  - CheckRegistry, registry (the singleton)
  - CheckEngine
"""

# Import check modules so their @registry.register decorators fire.
import trajlens.checks.semantic
import trajlens.checks.statistical
import trajlens.checks.structural
import trajlens.checks.temporal
import trajlens.checks.video  # noqa: F401
from trajlens.checks.engine import CheckEngine
from trajlens.checks.protocol import Check, CheckContext, CheckResult, Severity
from trajlens.checks.registry import CheckRegistry, registry

__all__ = [
    "Check",
    "CheckContext",
    "CheckEngine",
    "CheckRegistry",
    "CheckResult",
    "Severity",
    "registry",
]
