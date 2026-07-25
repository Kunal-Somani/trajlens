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
        "message": str,
        "details": dict,  # check-specific structured detail, shape varies by check_id
        "per_episode": dict[str, str] | absent  # episode_index (as string) -> finding;
                                                 # present only when the check populated it
      },
      ...
    ],
    "episodes": {  # absent entirely if no check produced per-episode data
      "worst": [
        {
          "episode_index": int,
          "finding_count": int,
          "trust_contribution": int,
          "finding_counts_by_check": dict[str, int]
        },
        ...  # up to 5, worst first
      ]
    },
    "baseline": {  # present only when --baseline was used
      "new": [ ... ],       # same shape as "results" entries
      "resolved": [{"check_id": str, "episode_index": int | None, "shard_path": str | None}, ...],
      "unchanged": [ ... ]  # same shape as "results" entries
    }
  }

Exit codes (enforced by CLI, not this module):
  0 = all INFO (PASS)
  1 = any WARN, no FAIL/ERROR
  2 = any FAIL or ERROR
"""

from __future__ import annotations

import json

from trajlens.baseline import BaselineDiff
from trajlens.checks.protocol import CheckResult, Severity
from trajlens.report.episodes import worst_episodes
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


def _result_entry(r: CheckResult) -> dict[str, object]:
    entry: dict[str, object] = {
        "check_id": r.check_id,
        "severity": r.severity.value,
        "category": r.check_id.split(".")[0],
        "message": r.message,
        "details": r.details,
    }
    if r.per_episode is not None:
        entry["per_episode"] = {str(k): v for k, v in r.per_episode.items()}
    return entry


def render_json(
    ref: str,
    version: DatasetVersion,
    num_episodes: int,
    num_frames: int | None,
    results: list[CheckResult],
    *,
    baseline_diff: BaselineDiff | None = None,
) -> str:
    """Return a JSON string representing the lint report.

    When baseline_diff is given, adds a top-level "baseline" key with
    "new"/"resolved"/"unchanged" lists (additive; all other keys unchanged).
    """
    worst = max((r.severity for r in results), default=Severity.INFO)
    score = compute_trust_score(results)

    result_entries = [_result_entry(r) for r in results]

    payload: dict[str, object] = {
        "ref": ref,
        "version": version.value,
        "trust_score": score,
        "score_formula_version": SCORE_FORMULA_VERSION,
        "grade": _grade(worst),
        "num_episodes": num_episodes,
        "num_frames": num_frames,
        "results": result_entries,
    }

    worst_eps = worst_episodes(results)
    if worst_eps:
        payload["episodes"] = {
            "worst": [
                {
                    "episode_index": e.episode_index,
                    "finding_count": e.finding_count,
                    "trust_contribution": e.trust_contribution,
                    "finding_counts_by_check": e.finding_counts_by_check,
                }
                for e in worst_eps
            ]
        }

    if baseline_diff is not None:
        payload["baseline"] = {
            "new": [_result_entry(r) for r in baseline_diff.new],
            "resolved": [
                {
                    "check_id": f.check_id,
                    "episode_index": f.episode_index,
                    "shard_path": f.shard_path,
                }
                for f in baseline_diff.resolved
            ],
            "unchanged": [_result_entry(r) for r in baseline_diff.unchanged],
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
