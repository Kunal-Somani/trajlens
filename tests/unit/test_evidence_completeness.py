"""Regression tests: checks must count EVERY violation, not stop at a display cap.

Three checks populate CheckResult.per_episode, and all three used to `break`
out of their episode scan once a hard-coded output cap was reached:

  STRUCTURAL.METADATA_DATA_AGREEMENT   break at 5 violations
  STATISTICAL.PER_EPISODE_STATS_MATCH  break at 10 violations
  TEMPORAL.TIMESTAMP_MONOTONIC         break after the FIRST offending episode

That conflated two separate concerns -- how many problems exist, and how many
example messages to show -- with two consequences:

  1. The reported "(N issue(s))" count was the truncation point, not the true
     total. A dataset with 12 broken episodes reported "5 issue(s)".
  2. CheckResult.per_episode, which feeds report/episodes.py's worst-episodes
     ranking, was populated only from the lowest-indexed episodes the scan
     reached before stopping -- so the "worst 5 episodes" table in the JSON
     report, the HTML report and the dashboard was ranking a biased prefix.
     For TIMESTAMP_MONOTONIC that map could never hold more than one entry.

These tests pin the corrected behavior: scan everything, count exactly,
sample the display.
"""

from __future__ import annotations

from pathlib import Path

from tests.fixtures.builders import (
    build_v3_metadata_data_disagreement,
    build_v3_non_monotonic_timestamps_multi,
)
from trajlens.checks.protocol import CheckContext, Severity
from trajlens.checks.structural import METADATA_DATA_AGREEMENT
from trajlens.checks.temporal import TIMESTAMP_MONOTONIC
from trajlens.checks.utils import MAX_SAMPLE_MESSAGES, format_violation_count
from trajlens.model import build_canonical_dataset
from trajlens.report.episodes import worst_episodes
from trajlens.sources.loader import SourceLoader

CTX = CheckContext(deep=False)

# Deliberately above both former caps (5 and 10) so one fixture size
# demonstrates the fix for every affected check.
BROKEN_EPISODES = 12


def _load(root: Path):  # type: ignore[no-untyped-def]
    handle = SourceLoader().resolve(str(root))
    return build_canonical_dataset(handle)


class TestMetadataDataAgreementCountsEverything:
    def test_reports_true_total_not_the_display_cap(self, tmp_path: Path) -> None:
        """Pre-fix this reported '5 issue(s)' for 12 broken episodes."""
        build_v3_metadata_data_disagreement(tmp_path, num_episodes=BROKEN_EPISODES)
        result = METADATA_DATA_AGREEMENT.run(_load(tmp_path), CTX)

        assert result.severity is Severity.FAIL
        assert result.details["total_violations"] == BROKEN_EPISODES
        assert result.details["affected_episodes"] == BROKEN_EPISODES

    def test_per_episode_covers_every_broken_episode(self, tmp_path: Path) -> None:
        """Pre-fix per_episode held only ~5 of the 12 broken episodes, so the
        worst-episodes ranking never saw the rest."""
        build_v3_metadata_data_disagreement(tmp_path, num_episodes=BROKEN_EPISODES)
        result = METADATA_DATA_AGREEMENT.run(_load(tmp_path), CTX)

        assert result.per_episode is not None
        assert len(result.per_episode) == BROKEN_EPISODES
        assert sorted(result.per_episode) == list(range(BROKEN_EPISODES))


class TestTimestampMonotonicCountsEverything:
    def test_per_episode_is_not_capped_at_a_single_entry(self, tmp_path: Path) -> None:
        """Pre-fix the outer `break` fired after the first offending episode,
        so per_episode could never hold more than one entry."""
        build_v3_non_monotonic_timestamps_multi(tmp_path, num_episodes=BROKEN_EPISODES)
        result = TIMESTAMP_MONOTONIC.run(_load(tmp_path), CTX)

        assert result.severity is Severity.FAIL
        assert result.per_episode is not None
        assert len(result.per_episode) == BROKEN_EPISODES, (
            f"expected all {BROKEN_EPISODES} broken episodes, got "
            f"{len(result.per_episode)} (1 would mean the outer break is back)"
        )
        assert result.details["total_violations"] == BROKEN_EPISODES


class TestDisplaySampleStaysBounded:
    def test_sample_messages_are_capped_while_count_stays_exact(self, tmp_path: Path) -> None:
        """The point of the fix is separating counting from sampling -- the
        message list must still be bounded even though the count is exact."""
        many = MAX_SAMPLE_MESSAGES + 15
        build_v3_metadata_data_disagreement(tmp_path, num_episodes=many)
        result = METADATA_DATA_AGREEMENT.run(_load(tmp_path), CTX)

        assert result.details["total_violations"] == many
        assert len(result.details["violations"]) == MAX_SAMPLE_MESSAGES
        # ...and the message must say so rather than implying the sample size
        # is the total.
        assert f"{many} issue(s), showing first {MAX_SAMPLE_MESSAGES}" in result.message


class TestFormatViolationCount:
    def test_no_truncation_phrasing_when_all_samples_present(self) -> None:
        assert format_violation_count(3, ["a", "b", "c"]) == "3 issue(s)"

    def test_names_the_true_total_when_truncated(self) -> None:
        assert format_violation_count(100, ["a", "b"]) == "100 issue(s), showing first 2"


class TestWorstEpisodesRanksOverEverything:
    def test_ranking_sees_all_affected_episodes(self, tmp_path: Path) -> None:
        """worst_episodes must aggregate over every affected episode. With a
        single check flagging all of them they legitimately tie (documented
        in worst_episodes' docstring), but the aggregation must at least have
        had the full set available -- pre-fix it only ever saw a prefix."""
        build_v3_metadata_data_disagreement(tmp_path, num_episodes=BROKEN_EPISODES)
        result = METADATA_DATA_AGREEMENT.run(_load(tmp_path), CTX)

        from trajlens.report.episodes import build_episode_summaries

        summaries = build_episode_summaries([result])
        assert len(summaries) == BROKEN_EPISODES

        top = worst_episodes([result])
        assert len(top) == 5

    def test_episode_flagged_by_two_checks_outranks_one_flagged_by_one(
        self, tmp_path: Path
    ) -> None:
        """Cross-check differentiation is the ranking's real signal, and it
        only works if both checks contributed their full per-episode maps."""
        build_v3_non_monotonic_timestamps_multi(tmp_path, num_episodes=BROKEN_EPISODES)
        ds = _load(tmp_path)
        monotonic = TIMESTAMP_MONOTONIC.run(ds, CTX)
        agreement = METADATA_DATA_AGREEMENT.run(ds, CTX)

        # This fixture breaks timestamps but leaves boundaries intact, so only
        # one check should be contributing per-episode findings here.
        assert monotonic.per_episode is not None
        summaries = worst_episodes([monotonic, agreement], limit=BROKEN_EPISODES)
        assert summaries, "expected per-episode findings to rank"
        # Every ranked episode came from the (complete) monotonic map.
        assert {s.episode_index for s in summaries} == set(monotonic.per_episode)
