"""Unit tests for M6: Welford, SEMANTIC, and STATISTICAL checks.

Coverage targets per 05_ENGINEERING_STANDARDS.md §5:
  - Every new check: passing fixture, failing fixture, at least one edge case.
  - Welford: correct mean/variance/std on known inputs, NaN handling, empty stream.
  - Each SEMANTIC check: clean passes as INFO/WARN-free, failure fixture fires at
    correct severity, edge cases (zero episodes, missing column).
  - Each STATISTICAL check: correct-stats fixture passes, wrong-stats fires,
    edge cases (no stats.json, zero episodes).
  - Precision/recall verification: each failing fixture fires exactly one check
    (the target one) and each clean fixture fires only INFO on the target check.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, ClassVar

import pytest

from tests.fixtures.builders import (
    build_v3_all_nan_action,
    build_v3_all_zero_action,
    build_v3_constant_action,
    build_v3_dataset,
    build_v3_empty_task_description,
    build_v3_missing_task,
    build_v3_no_language,
    build_v3_varying_action,
    build_v3_with_action,
    build_v3_with_action_names_dict,
    build_v3_with_correct_stats,
    build_v3_with_intrinsics_implausible,
    build_v3_with_intrinsics_plausible,
    build_v3_with_per_episode_stats,
    build_v3_with_wrong_stats,
    build_v3_wrong_feature_shape,
)
from trajlens.checks.protocol import CheckContext, Severity
from trajlens.checks.semantic import (
    CAMERA_INTRINSICS_PLAUSIBLE,
    FEATURE_DIMENSIONALITY,
    LANGUAGE_PRESENT,
    TASK_INTEGRITY,
)
from trajlens.checks.statistical import (
    PER_EPISODE_STATS_MATCH,
    STATS_MATCH_DATA,
    VALUE_SANITY,
)
from trajlens.checks.welford import WelfordAccumulator
from trajlens.model import build_canonical_dataset
from trajlens.sources.loader import SourceLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(root: Path) -> Any:
    handle = SourceLoader().resolve(str(root))
    return build_canonical_dataset(handle)


CTX = CheckContext(deep=False)


# ---------------------------------------------------------------------------
# Welford accumulator unit tests
# ---------------------------------------------------------------------------


class TestWelfordAccumulator:
    def test_empty_accumulator(self) -> None:
        acc = WelfordAccumulator()
        assert acc.count == 0
        assert acc.mean == 0.0
        assert acc.variance == 0.0
        assert acc.std == 0.0
        assert math.isinf(acc.min)
        assert math.isinf(-acc.max)

    def test_single_value(self) -> None:
        acc = WelfordAccumulator()
        acc.update(5.0)
        assert acc.count == 1
        assert acc.mean == pytest.approx(5.0)
        assert acc.variance == 0.0
        assert acc.std == 0.0
        assert acc.min == 5.0
        assert acc.max == 5.0

    def test_two_values(self) -> None:
        acc = WelfordAccumulator()
        acc.update(2.0)
        acc.update(4.0)
        assert acc.mean == pytest.approx(3.0)
        # Population variance: ((2-3)^2 + (4-3)^2) / 2 = 1.0
        assert acc.variance == pytest.approx(1.0)
        assert acc.std == pytest.approx(1.0)

    def test_known_values(self) -> None:
        """Verify against analytically known mean/std for [1, 2, 3, 4, 5]."""
        acc = WelfordAccumulator()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            acc.update(v)
        assert acc.count == 5
        assert acc.mean == pytest.approx(3.0)
        # Population std = sqrt(2)
        assert acc.std == pytest.approx(math.sqrt(2.0), rel=1e-10)
        assert acc.min == 1.0
        assert acc.max == 5.0

    def test_nan_bumps_count_but_not_mean(self) -> None:
        acc = WelfordAccumulator()
        acc.update(1.0)
        acc.update(math.nan)
        acc.update(3.0)
        assert acc.count == 3  # NaN still bumps count
        # Welford: NaN does not affect _mean recurrence, but count includes it.
        # After update(1.0): count=1, mean=1.0
        # After update(nan): count=2, mean unchanged=1.0 (NaN skips recurrence)
        # After update(3.0): count=3, delta=3-1=2, mean=1+2/3=5/3
        assert acc.mean == pytest.approx(5 / 3)

    def test_large_stream_consistent_with_naive(self) -> None:
        """On a moderately long stream, Welford should match the naive formula."""
        import random

        random.seed(42)
        values = [random.gauss(5.0, 2.0) for _ in range(1000)]
        acc = WelfordAccumulator()
        for v in values:
            acc.update(v)
        naive_mean = sum(values) / len(values)
        naive_var = sum((v - naive_mean) ** 2 for v in values) / len(values)
        assert acc.mean == pytest.approx(naive_mean, rel=1e-9)
        assert acc.variance == pytest.approx(naive_var, rel=1e-9)

    def test_constant_stream_zero_variance(self) -> None:
        acc = WelfordAccumulator()
        for _ in range(10):
            acc.update(7.0)
        assert acc.mean == pytest.approx(7.0)
        assert acc.variance == pytest.approx(0.0, abs=1e-14)
        assert acc.std == pytest.approx(0.0, abs=1e-14)


# ---------------------------------------------------------------------------
# SEMANTIC.FEATURE_DIMENSIONALITY
# ---------------------------------------------------------------------------


class TestFeatureDimensionality:
    def test_clean_scalar_features_pass(self, tmp_path: Path) -> None:
        """Default fixture has only shape=[1] scalar features — all pass."""
        build_v3_dataset(tmp_path)
        result = FEATURE_DIMENSIONALITY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO

    def test_correct_action_shape_passes(self, tmp_path: Path) -> None:
        """action shape=[3] with 3 names and 3-element Parquet column passes."""
        build_v3_with_action(tmp_path)
        result = FEATURE_DIMENSIONALITY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO

    def test_wrong_action_shape_fails(self, tmp_path: Path) -> None:
        """action declared shape=[7] but Parquet stores 3 elements — FAIL."""
        build_v3_wrong_feature_shape(tmp_path)
        result = FEATURE_DIMENSIONALITY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.FAIL
        assert "action" in result.message

    def test_zero_episodes_passes(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=0)
        result = FEATURE_DIMENSIONALITY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO

    def test_dict_shaped_names_with_correct_count_passes(self, tmp_path: Path) -> None:
        """Regression test: dict-shaped names (real lerobot/pusht format).

        action declares names={"motors": ["j0","j1","j2"]} (3 elements nested
        under one key) against shape=[3]. Must not be miscounted as 1 (the
        dict's key count) -- the nested list's length is what matters.
        """
        build_v3_with_action_names_dict(tmp_path)
        result = FEATURE_DIMENSIONALITY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO


# ---------------------------------------------------------------------------
# SEMANTIC.TASK_INTEGRITY
# ---------------------------------------------------------------------------


class TestTaskIntegrity:
    def test_clean_passes(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        result = TASK_INTEGRITY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO

    def test_undefined_task_index_fails(self, tmp_path: Path) -> None:
        """Frame data references task_index=99 not in task table — FAIL."""
        build_v3_missing_task(tmp_path)
        result = TASK_INTEGRITY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.FAIL
        assert "99" in result.message or any("99" in v for v in result.details["violations"])

    def test_empty_task_description_fails(self, tmp_path: Path) -> None:
        """Task table has task_index=0 -> "" (empty string) — FAIL."""
        build_v3_empty_task_description(tmp_path)
        result = TASK_INTEGRITY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.FAIL
        assert "empty" in result.message.lower() or any(
            "empty" in v.lower() for v in result.details["violations"]
        )

    def test_zero_episodes_passes(self, tmp_path: Path) -> None:
        """Zero-episode dataset has no frame data to reference tasks — passes."""
        build_v3_dataset(tmp_path, num_episodes=0)
        result = TASK_INTEGRITY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO


# ---------------------------------------------------------------------------
# SEMANTIC.CAMERA_INTRINSICS_PLAUSIBLE
# ---------------------------------------------------------------------------


class TestCameraIntrinsicsPlausible:
    def test_no_intrinsics_field_skipped(self, tmp_path: Path) -> None:
        """Standard fixture has no intrinsics feature — check reports skipped (INFO)."""
        build_v3_dataset(tmp_path)
        result = CAMERA_INTRINSICS_PLAUSIBLE.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO
        assert "skipped" in result.message.lower() or "not found" in result.message.lower()

    def test_plausible_intrinsics_pass(self, tmp_path: Path) -> None:
        """Plausible K matrix (positive fx, fy, cx, cy) — INFO with no violations."""
        build_v3_with_intrinsics_plausible(tmp_path)
        result = CAMERA_INTRINSICS_PLAUSIBLE.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO
        # The check should either say found+plausible or skip — not report violations.
        assert "violations" not in result.details or len(result.details.get("violations", [])) == 0

    def test_implausible_intrinsics_reported(self, tmp_path: Path) -> None:
        """Negative focal lengths — INFO with violations noted."""
        build_v3_with_intrinsics_implausible(tmp_path)
        result = CAMERA_INTRINSICS_PLAUSIBLE.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO  # INFO, not WARN (format doesn't standardize)
        assert len(result.details.get("violations", [])) > 0

    def test_zero_episodes_skipped_gracefully(self, tmp_path: Path) -> None:
        """Zero-episode dataset — check cannot validate values but does not crash."""
        build_v3_dataset(tmp_path, num_episodes=0)
        result = CAMERA_INTRINSICS_PLAUSIBLE.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO


# ---------------------------------------------------------------------------
# SEMANTIC.LANGUAGE_PRESENT
# ---------------------------------------------------------------------------


class TestLanguagePresent:
    def test_clean_with_tasks_passes(self, tmp_path: Path) -> None:
        """Default fixture has non-empty task descriptions — INFO."""
        build_v3_dataset(tmp_path)
        result = LANGUAGE_PRESENT.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO

    def test_empty_task_strings_warns(self, tmp_path: Path) -> None:
        """All episodes have empty task description — WARN."""
        build_v3_no_language(tmp_path)
        result = LANGUAGE_PRESENT.run(_load(tmp_path), CTX)
        assert result.severity is Severity.WARN
        assert "episode" in result.message.lower()

    def test_zero_episodes_passes(self, tmp_path: Path) -> None:
        """No episodes means no episodes lack descriptions — INFO."""
        build_v3_dataset(tmp_path, num_episodes=0)
        result = LANGUAGE_PRESENT.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO


# ---------------------------------------------------------------------------
# STATISTICAL.STATS_MATCH_DATA
# ---------------------------------------------------------------------------


class TestStatsMatchData:
    def test_no_stats_json_skipped(self, tmp_path: Path) -> None:
        """Dataset with no meta/stats.json — check skips gracefully (INFO)."""
        build_v3_dataset(tmp_path)
        result = STATS_MATCH_DATA.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO
        assert "skipped" in result.message.lower()

    def test_correct_stats_pass(self, tmp_path: Path) -> None:
        """stats.json generated from the same data — should pass within tolerance."""
        build_v3_with_correct_stats(tmp_path)
        result = STATS_MATCH_DATA.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO
        assert "matches" in result.message.lower() or "within tolerance" in result.message.lower()

    def test_wrong_stats_fail(self, tmp_path: Path) -> None:
        """stats.json mean is 0.9, true mean ~0.05 — FAIL with relative error >> rtol."""
        build_v3_with_wrong_stats(tmp_path)
        result = STATS_MATCH_DATA.run(_load(tmp_path), CTX)
        assert result.severity is Severity.FAIL
        assert "timestamp" in result.message
        assert len(result.details["violations"]) >= 1

    def test_zero_episodes_with_no_stats_skipped(self, tmp_path: Path) -> None:
        """Zero episodes, no stats.json — passes (nothing to compare)."""
        build_v3_dataset(tmp_path, num_episodes=0)
        result = STATS_MATCH_DATA.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO


# ---------------------------------------------------------------------------
# STATISTICAL.PER_EPISODE_STATS_MATCH
# ---------------------------------------------------------------------------


class TestPerEpisodeStatsMatch:
    def test_no_per_episode_stats_skipped(self, tmp_path: Path) -> None:
        """Standard v3.0 fixture has no stats/* columns — check skips (INFO)."""
        build_v3_dataset(tmp_path)
        result = PER_EPISODE_STATS_MATCH.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO

    def test_correct_per_episode_stats_pass(self, tmp_path: Path) -> None:
        """Episode metadata has correct stats/* columns — passes (INFO)."""
        build_v3_with_per_episode_stats(tmp_path, corrupt=False)
        result = PER_EPISODE_STATS_MATCH.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO

    def test_wrong_per_episode_stats_warn(self, tmp_path: Path) -> None:
        """Episode metadata has incorrect stats (9.9) — WARN."""
        build_v3_with_per_episode_stats(tmp_path, corrupt=True)
        result = PER_EPISODE_STATS_MATCH.run(_load(tmp_path), CTX)
        assert result.severity is Severity.WARN
        assert "timestamp" in result.message or any(
            "timestamp" in v for v in result.details["violations"]
        )

    def test_zero_episodes_passes(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=0)
        result = PER_EPISODE_STATS_MATCH.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO


# ---------------------------------------------------------------------------
# STATISTICAL.VALUE_SANITY
# ---------------------------------------------------------------------------


class TestValueSanity:
    def test_no_state_action_columns_skipped(self, tmp_path: Path) -> None:
        """Default fixture has no action/state columns — INFO."""
        build_v3_dataset(tmp_path)
        result = VALUE_SANITY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO
        assert "no state/action" in result.message.lower()

    def test_varying_action_passes(self, tmp_path: Path) -> None:
        """action column with varying values — INFO (clean)."""
        build_v3_varying_action(tmp_path)
        result = VALUE_SANITY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO

    def test_all_nan_action_fails(self, tmp_path: Path) -> None:
        """action column entirely NaN — FAIL (most severe)."""
        build_v3_all_nan_action(tmp_path)
        result = VALUE_SANITY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.FAIL
        assert "NaN" in result.message
        assert len(result.details["fail_items"]) >= 1

    def test_constant_nonzero_action_warns(self, tmp_path: Path) -> None:
        """action column constant at 0.5 across every episode — WARN."""
        build_v3_constant_action(tmp_path)
        result = VALUE_SANITY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.WARN
        assert len(result.details["warn_items"]) >= 1

    def test_all_zero_action_warns(self, tmp_path: Path) -> None:
        """action column all-zero across every episode — WARN."""
        build_v3_all_zero_action(tmp_path)
        result = VALUE_SANITY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.WARN

    def test_zero_episodes_passes(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=0)
        result = VALUE_SANITY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO


# ---------------------------------------------------------------------------
# Precision/recall verification (qualitative, per 07 §2 and task instructions)
#
# For each failing fixture we confirm:
#   (a) The TARGET check fires at the expected severity.
#   (b) Other checks from the same milestone do NOT fire at FAIL/WARN on this
#       fixture (verifying the fixture is precision-isolated: it fails exactly
#       one check and does not accidentally trip unrelated checks).
# ---------------------------------------------------------------------------


class TestPrecisionRecallVerification:
    """Verify each fixture fails exactly the intended check and no other M6 checks."""

    ALL_NEW_CHECKS: ClassVar[list[object]] = [
        FEATURE_DIMENSIONALITY,
        TASK_INTEGRITY,
        CAMERA_INTRINSICS_PLAUSIBLE,
        LANGUAGE_PRESENT,
        STATS_MATCH_DATA,
        PER_EPISODE_STATS_MATCH,
        VALUE_SANITY,
    ]

    def _run_all(self, root: Path) -> dict[str, Severity]:
        ds = _load(root)
        return {chk.id: chk.run(ds, CTX).severity for chk in self.ALL_NEW_CHECKS}

    def test_wrong_feature_shape_only_trips_feature_dimensionality(self, tmp_path: Path) -> None:
        build_v3_wrong_feature_shape(tmp_path)
        results = self._run_all(tmp_path)
        assert results["SEMANTIC.FEATURE_DIMENSIONALITY"] is Severity.FAIL
        # Other M6 checks must not FAIL on this fixture.
        for check_id, sev in results.items():
            if check_id != "SEMANTIC.FEATURE_DIMENSIONALITY":
                assert sev is not Severity.FAIL, (
                    f"{check_id} unexpectedly FAIL on wrong-shape fixture"
                )

    def test_missing_task_only_trips_task_integrity(self, tmp_path: Path) -> None:
        build_v3_missing_task(tmp_path)
        results = self._run_all(tmp_path)
        assert results["SEMANTIC.TASK_INTEGRITY"] is Severity.FAIL
        for check_id, sev in results.items():
            if check_id != "SEMANTIC.TASK_INTEGRITY":
                assert sev is not Severity.FAIL, (
                    f"{check_id} unexpectedly FAIL on missing-task fixture"
                )

    def test_no_language_only_trips_language_present(self, tmp_path: Path) -> None:
        build_v3_no_language(tmp_path)
        results = self._run_all(tmp_path)
        assert results["SEMANTIC.LANGUAGE_PRESENT"] is Severity.WARN
        # Only LANGUAGE_PRESENT and TASK_INTEGRITY (empty task) should fire.
        # TASK_INTEGRITY also fires because empty string = bad description.
        for check_id, sev in results.items():
            if check_id not in ("SEMANTIC.LANGUAGE_PRESENT", "SEMANTIC.TASK_INTEGRITY"):
                assert sev is not Severity.FAIL, (
                    f"{check_id} unexpectedly FAIL on no-language fixture"
                )
                assert sev is not Severity.WARN, (
                    f"{check_id} unexpectedly WARN on no-language fixture"
                )

    def test_wrong_stats_only_trips_stats_match_data(self, tmp_path: Path) -> None:
        build_v3_with_wrong_stats(tmp_path)
        results = self._run_all(tmp_path)
        assert results["STATISTICAL.STATS_MATCH_DATA"] is Severity.FAIL
        for check_id, sev in results.items():
            if check_id != "STATISTICAL.STATS_MATCH_DATA":
                assert sev is not Severity.FAIL, (
                    f"{check_id} unexpectedly FAIL on wrong-stats fixture"
                )

    def test_all_nan_action_only_trips_value_sanity(self, tmp_path: Path) -> None:
        build_v3_all_nan_action(tmp_path)
        results = self._run_all(tmp_path)
        assert results["STATISTICAL.VALUE_SANITY"] is Severity.FAIL
        for check_id, sev in results.items():
            if check_id != "STATISTICAL.VALUE_SANITY":
                assert sev is not Severity.FAIL, f"{check_id} unexpectedly FAIL on all-nan fixture"

    def test_constant_action_only_trips_value_sanity_as_warn(self, tmp_path: Path) -> None:
        build_v3_constant_action(tmp_path)
        results = self._run_all(tmp_path)
        assert results["STATISTICAL.VALUE_SANITY"] is Severity.WARN
        # No other new check should FAIL or WARN on this fixture.
        for check_id, sev in results.items():
            if check_id != "STATISTICAL.VALUE_SANITY":
                assert sev is not Severity.FAIL, (
                    f"{check_id} unexpectedly FAIL on constant-action fixture"
                )
                assert sev is not Severity.WARN, (
                    f"{check_id} unexpectedly WARN on constant-action fixture"
                )

    def test_wrong_per_episode_stats_only_trips_per_episode_stats(self, tmp_path: Path) -> None:
        build_v3_with_per_episode_stats(tmp_path, corrupt=True)
        results = self._run_all(tmp_path)
        assert results["STATISTICAL.PER_EPISODE_STATS_MATCH"] is Severity.WARN
        for check_id, sev in results.items():
            if check_id != "STATISTICAL.PER_EPISODE_STATS_MATCH":
                assert sev is not Severity.FAIL, (
                    f"{check_id} unexpectedly FAIL on corrupt-per-episode-stats fixture"
                )
                assert sev is not Severity.WARN, (
                    f"{check_id} unexpectedly WARN on corrupt-per-episode-stats fixture"
                )
