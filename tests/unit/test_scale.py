"""Unit tests for v0.4 T3: determinism, worker-exception, thread_safe, allocation-bomb.

Fast, no large fixture — see tests/perf/test_scale.py for the perf suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.builders import build_v3_dataset
from trajlens.checks.engine import CheckEngine
from trajlens.checks.protocol import Check, CheckContext, CheckResult, Severity, Tier
from trajlens.checks.registry import CheckRegistry
from trajlens.checks.registry import registry as global_registry
from trajlens.model import build_canonical_dataset
from trajlens.sources.loader import SourceLoader

CTX = CheckContext(deep=False)


def _load(root: Path) -> Any:
    handle = SourceLoader().resolve(str(root))
    return build_canonical_dataset(handle)


# ---------------------------------------------------------------------------
# Module-level check fixtures.
#
# ProcessPoolExecutor pickles the Check instance to send it to a worker, so
# any check used in a --parallel test must be a module-level, importable
# object -- a class defined inside a test function/method cannot be pickled.
# ---------------------------------------------------------------------------


class _CrashingCheck:
    id = "TEST.SCALE_CRASH"
    severity = Severity.FAIL
    category = "TEST"
    requires_video = False
    thread_safe = True
    tier = Tier.INTEGRITY
    formats: frozenset[str] | None = None

    def run(self, ds: Any, ctx: Any) -> CheckResult:
        raise RuntimeError("intentional crash for --parallel worker-exception test")


class _SerialOnlyCheck:
    id = "TEST.SCALE_SERIAL_ONLY"
    severity = Severity.INFO
    category = "TEST"
    requires_video = False
    thread_safe = False
    tier = Tier.INTEGRITY
    formats: frozenset[str] | None = None

    def run(self, ds: Any, ctx: Any) -> CheckResult:
        return CheckResult(
            check_id=self.id, severity=Severity.INFO, message="ran in serial fallback"
        )


SCALE_CRASH: Check = _CrashingCheck()  # type: ignore[assignment]
SCALE_SERIAL_ONLY: Check = _SerialOnlyCheck()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Determinism: serial and parallel modes produce identical CheckResult sets.
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_serial_and_parallel_produce_identical_results(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=5)
        ds = _load(tmp_path)
        engine = CheckEngine(global_registry)

        serial_results = engine.run(ds, CTX, parallel=1).results
        parallel_results = engine.run(ds, CTX, parallel=2).results

        serial_set = {(r.check_id, r.severity) for r in serial_results}
        parallel_set = {(r.check_id, r.severity) for r in parallel_results}
        assert serial_set == parallel_set
        assert [r.check_id for r in serial_results] == [r.check_id for r in parallel_results]


# ---------------------------------------------------------------------------
# Worker exception -> ERROR, never abort, never PASS (ADR-003 regression
# guard for the parallel path).
# ---------------------------------------------------------------------------


class TestWorkerException:
    def test_crashing_check_in_worker_yields_error(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=2)
        ds = _load(tmp_path)
        reg = CheckRegistry()
        reg.register(SCALE_CRASH)
        engine = CheckEngine(reg)

        results = engine.run(ds, CTX, parallel=2).results

        assert len(results) == 1
        assert results[0].check_id == "TEST.SCALE_CRASH"
        assert results[0].severity is Severity.ERROR
        assert results[0].severity is not Severity.FAIL
        assert "RuntimeError" in results[0].message


# ---------------------------------------------------------------------------
# thread_safe=False forces the serial fallback regardless of --parallel.
# ---------------------------------------------------------------------------


class TestThreadSafeFallback:
    def test_thread_unsafe_check_runs_serial_under_parallel(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=2)
        ds = _load(tmp_path)
        reg = CheckRegistry()
        reg.register(SCALE_SERIAL_ONLY)
        engine = CheckEngine(reg)

        serial_results = engine.run(ds, CTX, parallel=1).results
        parallel_results = engine.run(ds, CTX, parallel=4).results

        assert len(parallel_results) == 1
        assert parallel_results[0].check_id == "TEST.SCALE_SERIAL_ONLY"
        assert parallel_results[0].severity is Severity.INFO
        assert parallel_results[0].message == serial_results[0].message


# ---------------------------------------------------------------------------
# _MAX_SHARD_ROWS ceiling: an oversized declared shard yields ERROR for
# every data-reading check rather than attempting to process it.
# ---------------------------------------------------------------------------


class TestAllocationBombCeiling:
    def test_oversized_shard_yields_error_for_data_reading_checks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # FRAMES_PER_EPISODE is 4 per builders.py; 3 episodes = 12 rows in the
        # single shard. Patch the ceiling below that so the fixture itself
        # becomes the "oversized" shard without writing a 10M-row file.
        monkeypatch.setattr("trajlens.checks.engine._MAX_SHARD_ROWS", 2)

        build_v3_dataset(tmp_path, num_episodes=3)
        ds = _load(tmp_path)
        engine = CheckEngine(global_registry)

        results = engine.run(ds, CTX).results

        from trajlens.checks.engine import _DATA_READING_CHECK_IDS

        by_id = {r.check_id: r for r in results}
        for check_id in _DATA_READING_CHECK_IDS:
            if check_id not in by_id:
                continue  # e.g. requires_video and skipped for a no-camera dataset
            result = by_id[check_id]
            assert result.severity is Severity.ERROR, (
                f"{check_id} should be ERROR under the oversized-shard ceiling, "
                f"got {result.severity}"
            )
            assert "_MAX_SHARD_ROWS" in result.message

    def test_normal_shard_is_unaffected_by_ceiling(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=3)
        ds = _load(tmp_path)
        engine = CheckEngine(global_registry)

        results = engine.run(ds, CTX).results

        assert all("_MAX_SHARD_ROWS" not in r.message for r in results)
