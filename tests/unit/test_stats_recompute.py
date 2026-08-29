"""Unit tests for StatsRecomputeFixer (REPAIR.STATS_RECOMPUTE).

Coverage per 05_ENGINEERING_STANDARDS.md §5 and ADR-004:
  - Happy path + mandatory round-trip test: wrong stats → fixer → re-lint → PASS.
  - Two failure modes (missing stats.json, v2 dataset rejected).
  - Edge cases: already-clean dataset (no-op).
  - Apply-only-touches-stats-json test (mtime-based).
  - Dry-run zero-write test (mtime-based).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fixtures.builders import (
    build_v2_dataset,
    build_v3_dataset,
    build_v3_with_correct_stats,
    build_v3_with_wrong_count,
    build_v3_with_wrong_max,
    build_v3_with_wrong_stats,
)
from trajlens.checks import CheckEngine, registry
from trajlens.checks.protocol import CheckContext, Severity
from trajlens.checks.statistical import STATS_MATCH_DATA
from trajlens.errors import RepairError
from trajlens.model import build_canonical_dataset
from trajlens.repair.protocol import Diff, StatChange
from trajlens.repair.stats_recompute import (
    CHECK_ID,
    FIXER_ID,
    StatsRecomputeFixer,
    _recompute_stats,
)
from trajlens.sources.loader import SourceLoader

CTX = CheckContext(deep=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(root: Path):  # type: ignore[no-untyped-def]
    handle = SourceLoader().resolve(str(root))
    return build_canonical_dataset(handle)


def _has_stats_finding(root: Path) -> bool:
    ds = _load(root)
    result = STATS_MATCH_DATA.run(ds, CTX)
    return result.severity is Severity.FAIL


# ---------------------------------------------------------------------------
# Fixer identity / metadata
# ---------------------------------------------------------------------------


class TestFixerMetadata:
    def test_ids(self) -> None:
        fixer = StatsRecomputeFixer()
        assert fixer.fixer_id == FIXER_ID
        assert fixer.check_id == CHECK_ID
        assert fixer.fixer_id == "REPAIR.STATS_RECOMPUTE"
        assert fixer.check_id == "STATISTICAL.STATS_MATCH_DATA"


# ---------------------------------------------------------------------------
# Happy path + mandatory ADR-004 round-trip test
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_repair_clears_stats_finding(self, tmp_path: Path) -> None:
        """ADR-004 mandatory round-trip: repair → re-lint → finding gone."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_with_wrong_stats(source)

        assert _has_stats_finding(source), "fixture must trigger STATISTICAL.STATS_MATCH_DATA"

        fixer = StatsRecomputeFixer()
        ds = _load(source)
        summary = fixer.apply(ds, output)

        assert summary.output_path == output
        assert summary.changes_written == 1
        assert summary.frames_corrected > 0

        assert not _has_stats_finding(output), (
            "STATISTICAL.STATS_MATCH_DATA must be INFO after repair"
        )

    def test_dry_run_produces_no_filesystem_writes(self, tmp_path: Path) -> None:
        """dry_run() must not write, create, or modify any file."""
        source = tmp_path / "source"
        build_v3_with_wrong_stats(source)

        before = {p: p.stat().st_mtime for p in source.rglob("*") if p.is_file()}

        fixer = StatsRecomputeFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        after = {p: p.stat().st_mtime for p in source.rglob("*") if p.is_file()}

        assert before == after, "dry_run() must not touch any file on disk"
        assert not diff.is_noop, "wrong-stats fixture must produce a non-empty diff"

    def test_dry_run_diff_contents(self, tmp_path: Path) -> None:
        """Diff records correct feature/stat_key and old/new values."""
        source = tmp_path / "source"
        build_v3_with_wrong_stats(source)

        fixer = StatsRecomputeFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        assert isinstance(diff, Diff)
        assert diff.check_id == CHECK_ID
        assert diff.fixer_id == FIXER_ID
        assert len(diff.changes) > 0

        for change in diff.changes:
            assert isinstance(change, StatChange)
            assert change.stat_key in ("mean", "std", "min", "max", "count")
            assert change.feature != ""

    def test_no_new_finding_introduced(self, tmp_path: Path) -> None:
        """Repair must not introduce any new FAIL/WARN that wasn't already present.

        Uses the full default check suite so that side-effects on adjacent
        checks are caught — a scoped single-check run would miss those regressions.
        """
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_with_wrong_stats(source)

        engine = CheckEngine(registry)

        ds_source = _load(source)
        pre_results = engine.run(ds_source, CTX).results
        pre_fail_ids = {
            r.check_id
            for r in pre_results
            if r.severity >= Severity.WARN and r.check_id != CHECK_ID
        }

        fixer = StatsRecomputeFixer()
        fixer.apply(ds_source, output)

        ds_fixed = _load(output)
        post_results = engine.run(ds_fixed, CTX).results

        drift_post = next((r for r in post_results if r.check_id == CHECK_ID), None)
        assert drift_post is None or drift_post.severity < Severity.WARN, (
            f"{CHECK_ID} must not be WARN/FAIL after repair; got {drift_post}"
        )

        post_fail_ids = {
            r.check_id
            for r in post_results
            if r.severity >= Severity.WARN and r.check_id != CHECK_ID
        }
        new_findings = post_fail_ids - pre_fail_ids
        assert not new_findings, (
            f"repair introduced new WARN/FAIL findings not present in source: {new_findings}"
        )

    def test_repaired_stats_json_matches_recomputed(self, tmp_path: Path) -> None:
        """After repair, stats.json values match the Welford-recomputed values."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_with_wrong_stats(source)

        ds = _load(source)
        expected = _recompute_stats(ds)

        fixer = StatsRecomputeFixer()
        fixer.apply(ds, output)

        stats_path = output / "meta" / "stats.json"
        written = json.loads(stats_path.read_text())

        for feat, stats in expected.items():
            assert feat in written, f"feature {feat!r} missing from repaired stats.json"
            assert abs(written[feat]["mean"] - stats["mean"]) < 1e-9, (
                f"mean mismatch for {feat!r}: {written[feat]['mean']} vs {stats['mean']}"
            )
            assert abs(written[feat]["std"] - stats["std"]) < 1e-9, (
                f"std mismatch for {feat!r}: {written[feat]['std']} vs {stats['std']}"
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_clean_dataset_is_noop(self, tmp_path: Path) -> None:
        """A dataset already within tolerance must produce an empty Diff."""
        source = tmp_path / "source"
        build_v3_with_correct_stats(source)

        fixer = StatsRecomputeFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        assert diff.is_noop, "clean dataset must yield a noop diff"
        assert len(diff.changes) == 0

    def test_clean_dataset_apply_is_noop_summary(self, tmp_path: Path) -> None:
        """apply() on a clean dataset copies the tree but reports zero changes."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_with_correct_stats(source)

        fixer = StatsRecomputeFixer()
        ds = _load(source)
        summary = fixer.apply(ds, output)

        assert summary.frames_corrected == 0
        assert summary.changes_written == 0
        assert output.is_dir()

    def test_output_must_not_equal_source(self, tmp_path: Path) -> None:
        """apply() must raise RepairError when output_path == source root."""
        source = tmp_path / "source"
        build_v3_with_correct_stats(source)

        fixer = StatsRecomputeFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match="copy-on-write"):
            fixer.apply(ds, source)


# ---------------------------------------------------------------------------
# Apply only touches stats.json
# ---------------------------------------------------------------------------


class TestApplyOnlyTouchesStatsJson:
    def test_apply_only_rewrites_stats_json(self, tmp_path: Path) -> None:
        """apply() must produce an output where only meta/stats.json differs from source.

        After apply(), every file in the output tree should have the same content
        as the corresponding source file, except meta/stats.json which must differ.
        """
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_with_wrong_stats(source)

        fixer = StatsRecomputeFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)
        assert not diff.is_noop

        fixer.apply(ds, output)

        stats_rel = "meta/stats.json"
        for src_file in source.rglob("*"):
            if not src_file.is_file():
                continue
            rel = str(src_file.relative_to(source))
            out_file = output / rel
            assert out_file.is_file(), f"output missing file: {rel}"
            if rel == stats_rel:
                assert src_file.read_bytes() != out_file.read_bytes(), (
                    "meta/stats.json content must differ after repair"
                )
            else:
                assert src_file.read_bytes() == out_file.read_bytes(), (
                    f"non-stats file content changed unexpectedly: {rel}"
                )

    def test_stats_json_content_changes_after_apply(self, tmp_path: Path) -> None:
        """The content of meta/stats.json must differ between source and repaired."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_with_wrong_stats(source)

        original_stats = json.loads((source / "meta" / "stats.json").read_text())

        fixer = StatsRecomputeFixer()
        ds = _load(source)
        fixer.apply(ds, output)

        repaired_stats = json.loads((output / "meta" / "stats.json").read_text())

        # The wrong mean must have been corrected.
        assert original_stats["timestamp"]["mean"] != repaired_stats["timestamp"]["mean"], (
            "stats.json mean must change after repair"
        )


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_missing_stats_json_raises_repair_error(self, tmp_path: Path) -> None:
        """A dataset without meta/stats.json must raise RepairError."""
        source = tmp_path / "source"
        # build_v3_dataset does NOT write stats.json.
        build_v3_dataset(source, num_episodes=3)

        fixer = StatsRecomputeFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match=r"stats\.json"):
            fixer.dry_run(ds)

    def test_corrupt_stats_json_raises_repair_error(self, tmp_path: Path) -> None:
        """A corrupt (non-JSON) stats.json must raise RepairError."""
        source = tmp_path / "source"
        build_v3_with_correct_stats(source)
        (source / "meta" / "stats.json").write_text("NOT_VALID_JSON{{{{")

        fixer = StatsRecomputeFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match="not valid JSON"):
            fixer.dry_run(ds)

    def test_v2_dataset_raises_repair_error(self, tmp_path: Path) -> None:
        """v2.x datasets are rejected with a clear RepairError."""
        source = tmp_path / "source"
        build_v2_dataset(source, codebase_version="v2.1", num_episodes=3)

        fixer = StatsRecomputeFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match=r"v3\.0"):
            fixer.dry_run(ds)


# ---------------------------------------------------------------------------
# Corrupted max — false-pass regression guard
# ---------------------------------------------------------------------------


class TestCorruptedMax:
    """Guard against the false-pass bug: correct mean/std but corrupted max.

    Before the fix, dry_run() only diffed mean/std so a corrupt max produced a
    noop Diff and apply() silently skipped the rewrite.
    """

    def test_dry_run_detects_corrupted_max(self, tmp_path: Path) -> None:
        """dry_run() must report a non-noop Diff when only max is corrupted."""
        source = tmp_path / "source"
        build_v3_with_wrong_max(source)

        fixer = StatsRecomputeFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        assert not diff.is_noop, "corrupted max must produce a non-empty diff"
        max_changes = [c for c in diff.changes if isinstance(c, StatChange) and c.stat_key == "max"]
        assert len(max_changes) >= 1, "diff must contain at least one StatChange for 'max'"
        assert max_changes[0].feature == "timestamp"
        assert max_changes[0].old_value == pytest.approx(9.9)

    def test_apply_corrects_corrupted_max(self, tmp_path: Path) -> None:
        """apply() must rewrite stats.json and clear the max corruption."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_with_wrong_max(source)

        fixer = StatsRecomputeFixer()
        ds = _load(source)
        summary = fixer.apply(ds, output)

        assert summary.changes_written == 1
        assert summary.frames_corrected > 0

        written = json.loads((output / "meta" / "stats.json").read_text())
        # Correct max is 3/30 ≈ 0.1; the corrupt value was 9.9.
        assert written["timestamp"]["max"] == pytest.approx(3 / 30.0, rel=1e-4)

    def test_repaired_max_matches_recomputed(self, tmp_path: Path) -> None:
        """After repair, max in stats.json equals the Welford-recomputed value."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_with_wrong_max(source)

        ds = _load(source)
        expected = _recompute_stats(ds)

        fixer = StatsRecomputeFixer()
        fixer.apply(ds, output)

        written = json.loads((output / "meta" / "stats.json").read_text())
        assert written["timestamp"]["max"] == pytest.approx(expected["timestamp"]["max"], rel=1e-9)


# ---------------------------------------------------------------------------
# Corrupted count — exact-match path guard
# ---------------------------------------------------------------------------


class TestCorruptedCount:
    """Guard the exact-match comparison path used for count in dry_run()."""

    def test_dry_run_detects_corrupted_count(self, tmp_path: Path) -> None:
        """dry_run() must report a non-noop Diff when only count is wrong."""
        source = tmp_path / "source"
        build_v3_with_wrong_count(source)

        fixer = StatsRecomputeFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        assert not diff.is_noop, "corrupted count must produce a non-empty diff"
        count_changes = [
            c for c in diff.changes if isinstance(c, StatChange) and c.stat_key == "count"
        ]
        assert len(count_changes) >= 1, "diff must contain at least one StatChange for 'count'"
        assert count_changes[0].old_value == pytest.approx(11.0)
        assert count_changes[0].new_value == pytest.approx(12.0)

    def test_apply_corrects_corrupted_count(self, tmp_path: Path) -> None:
        """apply() must rewrite stats.json with the correct frame count."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_with_wrong_count(source)

        fixer = StatsRecomputeFixer()
        ds = _load(source)
        summary = fixer.apply(ds, output)

        assert summary.changes_written == 1

        written = json.loads((output / "meta" / "stats.json").read_text())
        # 3 episodes x 4 frames = 12.
        assert written["timestamp"]["count"] == pytest.approx(12.0)
