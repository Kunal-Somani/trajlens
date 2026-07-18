"""Per-episode aggregation for lint results (v0.3 T1).

Derives an episodes-view summary from CheckResult.per_episode data: which
episodes have findings, how many per check, and a worst-5 ranking. Pure
aggregation over already-computed check output — it never re-runs checks or
changes any check's dataset-level verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

from trajlens.checks.protocol import CheckResult, Severity

# Trust-contribution weight per severity, mirroring trust_score.py's
# per-result penalties (FAIL=30, WARN=5, ERROR=10) so an episode's
# contribution is legible against the same scale as the dataset trust score.
_SEVERITY_WEIGHT: dict[Severity, int] = {
    Severity.FAIL: 30,
    Severity.WARN: 5,
    Severity.ERROR: 10,
    Severity.INFO: 0,
}


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    """Aggregated per-episode finding data for one episode."""

    episode_index: int
    finding_count: int
    trust_contribution: int
    finding_counts_by_check: dict[str, int]


def build_episode_summaries(results: list[CheckResult]) -> list[EpisodeSummary]:
    """Aggregate every CheckResult.per_episode entry into one summary per episode.

    Returns episodes sorted worst-first (highest trust_contribution, ties
    broken by finding_count, then by episode_index ascending for determinism).
    Empty if no check in *results* populated per_episode.
    """
    finding_counts: dict[int, int] = {}
    trust_contributions: dict[int, int] = {}
    finding_counts_by_check: dict[int, dict[str, int]] = {}

    for result in results:
        if not result.per_episode:
            continue
        weight = _SEVERITY_WEIGHT[result.severity]
        for episode_index in result.per_episode:
            finding_counts[episode_index] = finding_counts.get(episode_index, 0) + 1
            trust_contributions[episode_index] = trust_contributions.get(episode_index, 0) + weight
            by_check = finding_counts_by_check.setdefault(episode_index, {})
            by_check[result.check_id] = by_check.get(result.check_id, 0) + 1

    summaries = [
        EpisodeSummary(
            episode_index=episode_index,
            finding_count=finding_counts[episode_index],
            trust_contribution=trust_contributions[episode_index],
            finding_counts_by_check=finding_counts_by_check[episode_index],
        )
        for episode_index in finding_counts
    ]
    summaries.sort(key=lambda s: (-s.trust_contribution, -s.finding_count, s.episode_index))
    return summaries


def worst_episodes(results: list[CheckResult], limit: int = 5) -> list[EpisodeSummary]:
    """Return up to *limit* episodes with the highest trust contribution, worst first."""
    return build_episode_summaries(results)[:limit]
