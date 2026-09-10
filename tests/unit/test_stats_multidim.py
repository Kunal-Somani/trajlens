"""Regression tests: STATISTICAL.STATS_MATCH_DATA / REPAIR.STATS_RECOMPUTE
must handle multi-dimensional features per-dimension, never pooled.

Prior to this fix, checks/statistical.py::_stream_feature_columns kept one
WelfordAccumulator per *feature name*, folding every element of a multi-dim
feature (e.g. a 3-DoF or 7-DoF ``action``) into a single pooled accumulator.
STATISTICAL.STATS_MATCH_DATA then compared that one pooled value against
every stored per-dimension mean/std in turn -- so a dataset with correct,
distinct per-dimension stats produced a spurious FAIL on every dimension.
REPAIR.STATS_RECOMPUTE inherited the same accumulator and, on --apply,
overwrote a correct per-dimension stats.json entry with a pooled scalar.

These tests pin the corrected behavior: per-dimension accumulators, aligned
comparison, and a fixer that leaves correct per-dimension stats untouched.
"""

from __future__ import annotations

from pathlib import Path

from tests.fixtures.builders import (
    build_v3_multidim_action_correct_stats,
    build_v3_multidim_action_wrong_stats,
)
from trajlens.checks.protocol import CheckContext, Severity
from trajlens.checks.statistical import STATS_MATCH_DATA
from trajlens.model import build_canonical_dataset
from trajlens.repair.protocol import StatChange
from trajlens.repair.stats_recompute import StatsRecomputeFixer
from trajlens.sources.loader import SourceLoader

CTX = CheckContext(deep=False)


def _load(root: Path):  # type: ignore[no-untyped-def]
    handle = SourceLoader().resolve(str(root))
    return build_canonical_dataset(handle)


class TestMultiDimStatsMatchData:
    def test_correct_per_dimension_stats_pass(self, tmp_path: Path) -> None:
        """Three distinct, correct per-dimension means must not FAIL.

        Pre-fix: this fixture FAILed on all 3 dimensions because every
        dimension was compared against one pooled cross-dimension mean.
        """
        build_v3_multidim_action_correct_stats(tmp_path)
        ds = _load(tmp_path)
        result = STATS_MATCH_DATA.run(ds, CTX)
        assert result.severity is Severity.INFO, result.message

    def test_corrupted_single_dimension_is_named_precisely(self, tmp_path: Path) -> None:
        """Only the corrupted dimension is reported; the other two are not."""
        build_v3_multidim_action_wrong_stats(tmp_path)
        ds = _load(tmp_path)
        result = STATS_MATCH_DATA.run(ds, CTX)
        assert result.severity is Severity.FAIL
        assert "'action'[1]" in result.message
        assert "'action'[0]" not in result.message
        assert "'action'[2]" not in result.message

    def test_dimension_count_mismatch_is_its_own_finding(self, tmp_path: Path) -> None:
        """Stored stats with the wrong element count is flagged distinctly,
        not silently zipped against a mismatched-length data list."""
        import json

        build_v3_multidim_action_correct_stats(tmp_path)
        stats_path = tmp_path / "meta" / "stats.json"
        stats = json.loads(stats_path.read_text())
        stats["action"]["mean"] = stats["action"]["mean"][:2]  # drop one dimension
        stats["action"]["std"] = stats["action"]["std"][:2]
        stats_path.write_text(json.dumps(stats))

        ds = _load(tmp_path)
        result = STATS_MATCH_DATA.run(ds, CTX)
        assert result.severity is Severity.FAIL
        assert "dimension count mismatch" in result.message


class TestMultiDimStatsRecomputeFixer:
    def test_correct_per_dimension_stats_are_a_noop(self, tmp_path: Path) -> None:
        """The fixer must not propose changes to already-correct per-dim stats.

        Pre-fix: dry_run's `float(stored_val_raw)` raised on a list and was
        silently swallowed, so multi-dim mismatches were invisible to the
        fixer; but apply() unconditionally overwrote every feature's entry
        with a pooled scalar regardless, corrupting correct data.
        """
        build_v3_multidim_action_correct_stats(tmp_path)
        ds = _load(tmp_path)
        diff = StatsRecomputeFixer().dry_run(ds)
        assert diff.is_noop, diff.changes

    def test_corrupted_dimension_yields_one_precise_change(self, tmp_path: Path) -> None:
        """Exactly one StatChange for the corrupted dimension, scalar-valued."""
        build_v3_multidim_action_wrong_stats(tmp_path)
        ds = _load(tmp_path)
        diff = StatsRecomputeFixer().dry_run(ds)
        assert not diff.is_noop
        mean_changes = [
            c for c in diff.changes if isinstance(c, StatChange) and c.stat_key == "mean[1]"
        ]
        assert len(mean_changes) == 1
        assert isinstance(mean_changes[0].old_value, float)
        assert isinstance(mean_changes[0].new_value, float)

    def test_apply_preserves_list_shape_in_stats_json(self, tmp_path: Path) -> None:
        """apply() must write mean/std back as a 3-element list, never a pooled scalar."""
        import json

        build_v3_multidim_action_wrong_stats(tmp_path)
        ds = _load(tmp_path)
        out = tmp_path.parent / "repaired"
        StatsRecomputeFixer().apply(ds, out)

        repaired_stats = json.loads((out / "meta" / "stats.json").read_text())
        assert isinstance(repaired_stats["action"]["mean"], list)
        assert len(repaired_stats["action"]["mean"]) == 3
        assert isinstance(repaired_stats["action"]["std"], list)
        assert len(repaired_stats["action"]["std"]) == 3

        # And the repaired dataset must now pass the check.
        repaired_ds = _load(out)
        result = STATS_MATCH_DATA.run(repaired_ds, CTX)
        assert result.severity is Severity.INFO, result.message
