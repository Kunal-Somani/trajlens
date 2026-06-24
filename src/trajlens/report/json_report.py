"""JSON renderer for lint results (--json flag).

Schema:
  {
    "ref": str,
    "version": str,
    "trust_score": int,
    "score_formula_version": str,
    "grade": "PASS" | "WARN" | "FAIL" | "ERROR",
    "num_episodes": int,
    "num_frames": int,
    "results": [
      {
        "check_id": str,
        "severity": str,
        "category": str,
        "message": str
      },
      ...
    ]
  }

Exit codes (enforced by CLI, not this module):
  0 = all INFO (PASS)
  1 = any WARN, no FAIL/ERROR
  2 = any FAIL or ERROR
"""

from __future__ import annotations

import json

from trajlens.checks.protocol import CheckResult, Severity
from trajlens.report.trust_score import SCORE_FORMULA_VERSION, compute_trust_score
from trajlens.sources.version import DatasetVersion


def _grade(worst: Severity) -> str:
    if worst >= Severity.ERROR:
        return "ERROR"
    if worst >= Severity.FAIL:
        return "FAIL"
    if worst >= Severity.WARN:
        return "WARN"
    return "PASS"


def render_json(
    ref: str,
    version: DatasetVersion,
    num_episodes: int,
    num_frames: int | None,
    results: list[CheckResult],
) -> str:
    """Return a JSON string representing the lint report."""
    worst = max((r.severity for r in results), default=Severity.INFO)
    score = compute_trust_score(results)

    payload: dict[str, object] = {
        "ref": ref,
        "version": version.value,
        "trust_score": score,
        "score_formula_version": SCORE_FORMULA_VERSION,
        "grade": _grade(worst),
        "num_episodes": num_episodes,
        "num_frames": num_frames,
        "results": [
            {
                "check_id": r.check_id,
                "severity": r.severity.value,
                "category": r.check_id.split(".")[0],
                "message": r.message,
            }
            for r in results
        ],
    }
    return json.dumps(payload, indent=2)


def render_json_load_error(ref: str, error_category: str, message: str) -> str:
    """Return a JSON string for a dataset that failed to load before any checks ran.

    Mirrors the schema of :func:`render_json` (ref, grade, results) so
    consumers parsing ``--json`` output never have to special-case a missing
    key — ``results`` is just empty and ``error_category``/``error_message``
    explain why.
    """
    payload: dict[str, object] = {
        "ref": ref,
        "version": None,
        "trust_score": None,
        "score_formula_version": SCORE_FORMULA_VERSION,
        "grade": "ERROR",
        "num_episodes": None,
        "num_frames": None,
        "error_category": error_category,
        "error_message": message,
        "results": [],
    }
    return json.dumps(payload, indent=2)
