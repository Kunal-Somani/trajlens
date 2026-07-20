"""Redacted single-file finding summary for `trajlens lint --share` (v0.3 T1).

Designed for pasting into a GitHub issue. Contains only trajlens version,
trust score, grade, format version, per-check finding counts, and the
worst-5 episodes list (episode indices + counts only, no free text). It
never includes CheckResult.message, .details, or .per_episode values,
because those are check-authored free text that can embed local filesystem
paths (e.g. STRUCTURAL.PATH_TEMPLATE_RESOLVES messages quote shard paths) --
counts and stable IDs are the only fields guaranteed path-free.
"""

from __future__ import annotations

import json

import trajlens
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


def render_share(
    version: DatasetVersion,
    results: list[CheckResult],
    *,
    dataset_ref: str | None = None,
) -> str:
    """Return a redacted single-file JSON summary suitable for a GitHub issue.

    No local paths, no environment info: only counts, ids, and the dataset's
    declared format version are included. *dataset_ref* identifies which
    dataset this is about without leaking filesystem structure: callers must
    pass the Hub repo id (full `org/name`) for Hub refs, or the basename only
    -- never any parent path component -- for local paths.
    """
    worst = max((r.severity for r in results), default=Severity.INFO)
    score = compute_trust_score(results)

    finding_counts: dict[str, int] = {}
    for r in results:
        if r.severity is Severity.INFO:
            continue
        finding_counts[r.check_id] = finding_counts.get(r.check_id, 0) + 1

    payload: dict[str, object] = {
        "trajlens_version": trajlens.__version__,
        "trust_score": score,
        "score_formula_version": SCORE_FORMULA_VERSION,
        "grade": _grade(worst),
        "format_version": version.value,
        "finding_counts": finding_counts,
    }
    if dataset_ref is not None:
        payload["dataset_ref"] = dataset_ref

    worst_eps = worst_episodes(results)
    if worst_eps:
        payload["worst_episodes"] = [
            {
                "episode_index": e.episode_index,
                "finding_count": e.finding_count,
                "trust_contribution": e.trust_contribution,
            }
            for e in worst_eps
        ]

    return json.dumps(payload, indent=2)
