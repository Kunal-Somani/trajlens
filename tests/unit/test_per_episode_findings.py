"""Tests for v0.3 T1: per-episode findings on CheckResult.

Covers the three checks given per-episode attribution
(STRUCTURAL.METADATA_DATA_AGREEMENT, TEMPORAL.TIMESTAMP_MONOTONIC,
STATISTICAL.PER_EPISODE_STATS_MATCH), the --json schema, the --share
redaction contract, and the episodes report-layer aggregation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pyarrow as pa
import pyarrow.parquet as pq

from tests.fixtures.builders import (
    build_v3_dataset,
    build_v3_metadata_data_disagreement,
    build_v3_non_monotonic_timestamps,
)
from trajlens.checks.protocol import CheckContext, CheckResult, Severity
from trajlens.checks.statistical import PER_EPISODE_STATS_MATCH
from trajlens.checks.structural import METADATA_DATA_AGREEMENT
from trajlens.checks.temporal import TIMESTAMP_MONOTONIC
from trajlens.model import build_canonical_dataset
from trajlens.report.episodes import build_episode_summaries, worst_episodes
from trajlens.report.json_report import render_json
from trajlens.report.share_report import render_share
from trajlens.sources.loader import SourceLoader
from trajlens.sources.version import DatasetVersion

CTX = CheckContext(deep=False)


def _load(root: Path):  # type: ignore[no-untyped-def]
    handle = SourceLoader().resolve(str(root))
    return build_canonical_dataset(handle)


def _add_per_episode_stats(root: Path, *, corrupt_episode: int, corrupt_mean: float) -> None:
    """Add inline v3.0 stats/timestamp/{mean,std} columns to episode metadata,
    with one episode's mean deliberately wrong (all others correct)."""
    ep_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(ep_path)
    ep_indices = table.column("episode_index").to_pylist()
    lengths = table.column("length").to_pylist()

    means = []
    stds = []
    for ep_idx, length in zip(ep_indices, lengths, strict=True):
        ts = [f / 30.0 for f in range(length)]
        mean = sum(ts) / len(ts)
        std = (sum((x - mean) ** 2 for x in ts) / len(ts)) ** 0.5
        if ep_idx == corrupt_episode:
            mean = corrupt_mean
        means.append(mean)
        stds.append(std)

    table = table.append_column("stats/timestamp/mean", pa.array(means, type=pa.float64()))
    table = table.append_column("stats/timestamp/std", pa.array(stds, type=pa.float64()))
    pq.write_table(table, ep_path)


class TestMetadataDataAgreementPerEpisode:
    def test_clean_dataset_no_per_episode(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        result = METADATA_DATA_AGREEMENT.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO
        assert result.per_episode is None

    def test_corrupted_dataset_populates_per_episode(self, tmp_path: Path) -> None:
        build_v3_metadata_data_disagreement(tmp_path, num_episodes=3)
        result = METADATA_DATA_AGREEMENT.run(_load(tmp_path), CTX)
        assert result.severity is Severity.FAIL
        assert result.per_episode is not None
        assert 0 in result.per_episode
        assert "dataset_to_index" in result.per_episode[0]

    def test_per_episode_keys_are_ints(self, tmp_path: Path) -> None:
        build_v3_metadata_data_disagreement(tmp_path, num_episodes=3)
        result = METADATA_DATA_AGREEMENT.run(_load(tmp_path), CTX)
        assert result.per_episode is not None
        assert all(isinstance(k, int) for k in result.per_episode)


class TestTimestampMonotonicPerEpisode:
    def test_clean_dataset_no_per_episode(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        result = TIMESTAMP_MONOTONIC.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO
        assert result.per_episode is None

    def test_reversed_timestamps_populates_per_episode(self, tmp_path: Path) -> None:
        build_v3_non_monotonic_timestamps(tmp_path)
        result = TIMESTAMP_MONOTONIC.run(_load(tmp_path), CTX)
        assert result.severity is Severity.FAIL
        assert result.per_episode is not None
        assert 0 in result.per_episode
        assert "not strictly increasing" in result.per_episode[0]


class TestPerEpisodeStatsMatchPerEpisode:
    def test_clean_dataset_no_per_episode(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=3)
        _add_per_episode_stats(tmp_path, corrupt_episode=-1, corrupt_mean=0.0)
        result = PER_EPISODE_STATS_MATCH.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO
        assert result.per_episode is None

    def test_diverged_episode_populates_per_episode(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=3)
        _add_per_episode_stats(tmp_path, corrupt_episode=1, corrupt_mean=999.0)
        result = PER_EPISODE_STATS_MATCH.run(_load(tmp_path), CTX)
        assert result.severity is Severity.WARN
        assert result.per_episode is not None
        assert 1 in result.per_episode
        assert 0 not in result.per_episode
        assert 2 not in result.per_episode


class TestEpisodesReportLayer:
    _RESULTS: ClassVar[list[CheckResult]] = [
        CheckResult(
            check_id="STRUCTURAL.METADATA_DATA_AGREEMENT",
            severity=Severity.FAIL,
            message="bad",
            per_episode={0: "ep0 finding", 2: "ep2 finding"},
        ),
        CheckResult(
            check_id="TEMPORAL.TIMESTAMP_MONOTONIC",
            severity=Severity.FAIL,
            message="bad",
            per_episode={0: "ep0 finding 2"},
        ),
        CheckResult(
            check_id="STATISTICAL.PER_EPISODE_STATS_MATCH",
            severity=Severity.WARN,
            message="bad",
            per_episode={1: "ep1 finding"},
        ),
        CheckResult(check_id="STRUCTURAL.VERSION_DETECTED", severity=Severity.INFO, message="ok"),
    ]

    def test_worst_episode_is_highest_trust_contribution(self) -> None:
        summaries = build_episode_summaries(self._RESULTS)
        # episode 0: two FAILs = 60; episode 2: one FAIL = 30; episode 1: one WARN = 5
        assert summaries[0].episode_index == 0
        assert summaries[0].trust_contribution == 60
        assert summaries[0].finding_count == 2

    def test_worst_episodes_respects_limit(self) -> None:
        top = worst_episodes(self._RESULTS, limit=2)
        assert len(top) == 2
        assert top[0].episode_index == 0

    def test_no_per_episode_data_returns_empty(self) -> None:
        results = [CheckResult(check_id="X.Y", severity=Severity.FAIL, message="m")]
        assert build_episode_summaries(results) == []
        assert worst_episodes(results) == []

    def test_tie_break_is_deterministic_by_episode_index(self) -> None:
        results = [
            CheckResult(
                check_id="A.B",
                severity=Severity.WARN,
                message="m",
                per_episode={5: "x", 1: "y"},
            ),
        ]
        summaries = build_episode_summaries(results)
        assert [s.episode_index for s in summaries] == [1, 5]

    def test_finding_counts_by_check_populated(self) -> None:
        summaries = build_episode_summaries(self._RESULTS)
        ep0 = next(s for s in summaries if s.episode_index == 0)
        assert ep0.finding_counts_by_check == {
            "STRUCTURAL.METADATA_DATA_AGREEMENT": 1,
            "TEMPORAL.TIMESTAMP_MONOTONIC": 1,
        }


class TestJsonSchema:
    def test_per_episode_present_where_populated(self, tmp_path: Path) -> None:
        build_v3_metadata_data_disagreement(tmp_path, num_episodes=3)
        ds = _load(tmp_path)
        results = [METADATA_DATA_AGREEMENT.run(ds, CTX)]
        payload = json.loads(render_json("ref", ds.version, 3, 300, results))
        result_entry = payload["results"][0]
        assert "per_episode" in result_entry
        assert "0" in result_entry["per_episode"]  # JSON keys are always strings

    def test_per_episode_absent_where_not_populated(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        ds = _load(tmp_path)
        results = [METADATA_DATA_AGREEMENT.run(ds, CTX)]
        payload = json.loads(render_json("ref", ds.version, 3, 300, results))
        assert "per_episode" not in payload["results"][0]

    def test_episodes_key_present_when_per_episode_data_exists(self, tmp_path: Path) -> None:
        build_v3_metadata_data_disagreement(tmp_path, num_episodes=3)
        ds = _load(tmp_path)
        results = [METADATA_DATA_AGREEMENT.run(ds, CTX)]
        payload = json.loads(render_json("ref", ds.version, 3, 300, results))
        assert "episodes" in payload
        assert len(payload["episodes"]["worst"]) >= 1

    def test_episodes_key_absent_when_no_per_episode_data(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        ds = _load(tmp_path)
        results = [METADATA_DATA_AGREEMENT.run(ds, CTX)]
        payload = json.loads(render_json("ref", ds.version, 3, 300, results))
        assert "episodes" not in payload

    def test_existing_schema_fields_unchanged(self, tmp_path: Path) -> None:
        """Additive-only: every pre-v0.3 top-level key is still present with the same shape."""
        build_v3_dataset(tmp_path)
        ds = _load(tmp_path)
        results = [METADATA_DATA_AGREEMENT.run(ds, CTX)]
        payload = json.loads(render_json("ref", ds.version, 3, 300, results))
        for key in (
            "ref",
            "version",
            "trust_score",
            "score_formula_version",
            "grade",
            "num_episodes",
            "num_frames",
            "results",
        ):
            assert key in payload
        for key in ("check_id", "severity", "category", "message", "details"):
            assert key in payload["results"][0]


class TestSharePathRedaction:
    def test_share_output_is_valid_json(self, tmp_path: Path) -> None:
        build_v3_metadata_data_disagreement(tmp_path, num_episodes=3)
        ds = _load(tmp_path)
        results = [METADATA_DATA_AGREEMENT.run(ds, CTX)]
        doc = render_share(ds.version, results)
        parsed = json.loads(doc)  # round-trips
        assert parsed["grade"] == "FAIL"

    def test_share_output_has_no_absolute_paths(self, tmp_path: Path) -> None:
        # tmp_path itself is an absolute path -- prove it never leaks through.
        build_v3_metadata_data_disagreement(tmp_path, num_episodes=3)
        ds = _load(tmp_path)
        results = [METADATA_DATA_AGREEMENT.run(ds, CTX)]
        doc = render_share(ds.version, results)

        def _scan(value: object) -> None:
            if isinstance(value, str):
                assert not value.startswith("/")
                assert not value.startswith("~")
                assert not value.startswith("/home")
                assert str(tmp_path) not in value
            elif isinstance(value, dict):
                for v in value.values():
                    _scan(v)
            elif isinstance(value, list):
                for v in value:
                    _scan(v)

        _scan(json.loads(doc))

    def test_share_output_contains_no_message_or_details_text(self, tmp_path: Path) -> None:
        """--share must never leak check-authored free text (messages can embed paths)."""
        build_v3_metadata_data_disagreement(tmp_path, num_episodes=3)
        ds = _load(tmp_path)
        results = [METADATA_DATA_AGREEMENT.run(ds, CTX)]
        doc = render_share(ds.version, results)
        assert results[0].message not in doc

    def test_share_output_zero_findings_dataset(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        ds = _load(tmp_path)
        results = [METADATA_DATA_AGREEMENT.run(ds, CTX)]
        doc = render_share(ds.version, results)
        parsed = json.loads(doc)
        assert parsed["grade"] == "PASS"
        assert parsed["finding_counts"] == {}
        assert "worst_episodes" not in parsed

    def test_share_grep_zero_absolute_paths(self, tmp_path: Path) -> None:
        """Exact contract from the task spec: grep -E '^(/|~|/home)' must find nothing per line."""
        import re

        build_v3_metadata_data_disagreement(tmp_path, num_episodes=3)
        ds = _load(tmp_path)
        results = [METADATA_DATA_AGREEMENT.run(ds, CTX)]
        doc = render_share(ds.version, results)
        for line in doc.splitlines():
            assert re.match(r"^(/|~|/home)", line) is None


class TestShareVersionHandling:
    def test_share_uses_dataset_version(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        ds = _load(tmp_path)
        doc = render_share(ds.version, [])
        parsed = json.loads(doc)
        assert parsed["format_version"] == DatasetVersion.V3_0.value
