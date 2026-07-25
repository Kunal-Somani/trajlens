"""CheckEngine: selects applicable checks and runs them, enforcing ADR-003.

ADR-003 is the single most important rule here: a crashing check yields an
ERROR result; it never lets the exception propagate, and it never produces a
PASS.  Every check's run() call is wrapped in a broad try/except — the one
sanctioned place for catching Exception broadly in the codebase, because the
alternative (one bad check crashing the entire lint run) is worse.

O(1)-memory audit (v0.4 T3, 05_ENGINEERING_STANDARDS.md §6). Memory bound is
expressed relative to one shard's row count, since shard size is a
dataset-declared quantity with a hard ceiling (_MAX_SHARD_ROWS below), not an
absolute byte figure. No check in checks/ ever materializes more than one
shard's worth of its needed columns in memory at a time — none accumulate
data across shards.

  STRUCTURAL.VERSION_DETECTED          O(1) — reads only ds.version, no shard I/O.
  STRUCTURAL.SCHEMA_CONSISTENCY        O(shard rows) per shard — pf.schema_arrow
                                        is footer metadata only; no column read.
  STRUCTURAL.INDEX_CONTINUITY          O(shard rows) per shard — via
                                        ShardColumnCache, one shard resident at a time.
  STRUCTURAL.METADATA_DATA_AGREEMENT   O(shard rows) per shard — via ShardColumnCache.
  STRUCTURAL.PATH_TEMPLATE_RESOLVES    O(1) per episode — opens footer/existence
                                        only, never reads column data.
  STRUCTURAL.ORPHAN_SHARD              O(shard rows) per shard — delegates to
                                        repair/orphan_shard_report.find_orphan_shards(),
                                        which reads locator columns per shard.
  STATISTICAL.STATS_MATCH_DATA         O(shard rows) per shard — Welford
                                        accumulators are O(1) state per feature;
                                        _stream_feature_columns reads one shard's
                                        columns at a time via pf.read(), folds into
                                        the accumulator, and discards the shard.
  STATISTICAL.PER_EPISODE_STATS_MATCH  O(shard rows) per shard — same streaming
                                        pattern, re-run per episode.
  STATISTICAL.VALUE_SANITY             O(shard rows) per shard — reads target
                                        columns for one shard via pf.read(), scans,
                                        discards before the next shard.
  TEMPORAL.TIMESTAMP_MONOTONIC         O(shard rows) per shard — via ShardColumnCache.
  TEMPORAL.TIMESTAMP_SPACING           O(shard rows) per shard — via ShardColumnCache.
  KNOWNBUG.TIMESTAMP_DRIFT             O(shard rows) per shard — via ShardColumnCache;
                                        cumulative_drift is a single running float.
  SEMANTIC.FEATURE_DIMENSIONALITY      O(shard rows) per shard — via ShardColumnCache.
  SEMANTIC.TASK_INTEGRITY              O(shard rows) per shard — via ShardColumnCache;
                                        referenced/defined_indices are O(distinct
                                        task_index values), bounded by task count.
  SEMANTIC.CAMERA_INTRINSICS_PLAUSIBLE O(1) — reads exactly one shard, first
                                        episode only, then returns.
  SEMANTIC.LANGUAGE_PRESENT            O(1) per episode — reads ep.tasks from
                                        already-materialized episode metadata,
                                        no shard I/O.
  VIDEO.DECODABLE_SPOTCHECK            O(1) per shard — _MAX_FRAMES_PER_DECODE
                                        caps PyAV frame decode per shard regardless
                                        of declared video length (video.py).

ShardColumnCache (checks/utils.py) and pf.read(columns=...) both read the
requested columns of exactly one shard into memory at a time; the memory
footprint scales with that shard's row count, never with dataset size or
episode count. This is a deliberate choice, not an oversight: switching to
ParquetFile.iter_batches() would re-read the same shard once per episode it
contains for Hub-hosted multi-episode shards, which is the O(N^2) HTTP-read
pathology ShardColumnCache's own docstring exists to avoid. _MAX_SHARD_ROWS
below is what makes "one shard's worth" a bounded quantity instead of an
unbounded one.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import structlog

from trajlens.checks.protocol import Check, CheckContext, CheckResult, Severity
from trajlens.checks.registry import CheckRegistry
from trajlens.model.canonical import CanonicalDataset

log = structlog.get_logger(__name__)

# 10M rows * ~100 bytes/row = ~1 GB; datasets declaring more are malformed or
# adversarial — fail closed per 06 T2.
_MAX_SHARD_ROWS: int = 10_000_000

# Checks that read shard data and must be skipped (not run) when a shard
# exceeds _MAX_SHARD_ROWS, rather than attempting to process it. Structural
# checks that only touch footer metadata (VERSION_DETECTED,
# PATH_TEMPLATE_RESOLVES) or already-materialized episode metadata
# (LANGUAGE_PRESENT) are not data-reading in this sense and are exempt.
_DATA_READING_CHECK_IDS: frozenset[str] = frozenset(
    {
        "STRUCTURAL.SCHEMA_CONSISTENCY",
        "STRUCTURAL.INDEX_CONTINUITY",
        "STRUCTURAL.METADATA_DATA_AGREEMENT",
        "STRUCTURAL.ORPHAN_SHARD",
        "STATISTICAL.STATS_MATCH_DATA",
        "STATISTICAL.PER_EPISODE_STATS_MATCH",
        "STATISTICAL.VALUE_SANITY",
        "TEMPORAL.TIMESTAMP_MONOTONIC",
        "TEMPORAL.TIMESTAMP_SPACING",
        "KNOWNBUG.TIMESTAMP_DRIFT",
        "SEMANTIC.FEATURE_DIMENSIONALITY",
        "SEMANTIC.TASK_INTEGRITY",
        "SEMANTIC.CAMERA_INTRINSICS_PLAUSIBLE",
    }
)


def _run_check_in_worker(check: Check, ds: CanonicalDataset, ctx: CheckContext) -> CheckResult:
    """Run one check in a ProcessPoolExecutor worker, converting exceptions to ERROR.

    Module-level so it is picklable as the executor.submit target. Mirrors
    CheckEngine._run_one's ADR-003 handling: a worker exception must never
    abort the run or surface as a bare traceback, and must never produce PASS.
    """
    try:
        return check.run(ds, ctx)
    except Exception as exc:
        exc_type = type(exc).__name__
        return CheckResult(
            check_id=check.id,
            severity=Severity.ERROR,
            message=f"check raised unhandled exception in worker: {exc_type}: {exc}",
            details={"exc_type": exc_type, "exc_message": str(exc)},
        )


def _oversized_shard_rows(ds: CanonicalDataset) -> int | None:
    """Return the row count of the first shard exceeding _MAX_SHARD_ROWS, if any.

    Checks each episode's resolved data shard via its ParquetFile footer
    metadata (num_rows), which is read at open time and costs no column I/O
    (sources/handles.py). This runs before any data-reading check so a
    dataset that declares an absurd shard is rejected up front rather than
    discovered mid-check.
    """
    for episode in ds:
        pf = ds.parquet_shard_for_episode(episode)
        num_rows = pf.metadata.num_rows
        if num_rows > _MAX_SHARD_ROWS:
            return int(num_rows)
    return None


class CheckEngine:
    """Selects applicable checks and runs them, collecting CheckResults.

    Selection criteria (applied in order):
      1. Checks that require_video are skipped when the dataset has no cameras.
      2. Checks that require_video and deep=False in ctx are skipped unless
         the check is in the non-deep video set (currently only DECODABLE_SPOTCHECK).
      3. All other checks run unconditionally.

    ADR-003: any exception escaping a check's run() is caught here, logged,
    and converted to a CheckResult with severity=ERROR.  The exception type
    and message are preserved in the result's details dict.
    """

    def __init__(self, reg: CheckRegistry) -> None:
        self._registry = reg

    def run(
        self, ds: CanonicalDataset, ctx: CheckContext, *, parallel: int = 1
    ) -> list[CheckResult]:
        """Run all applicable checks; return one CheckResult per check.

        parallel: number of worker processes. 1 (default) is fully serial and
        behaves exactly as before this option existed. Values >1 run
        thread_safe checks in a ProcessPoolExecutor with
        min(parallel, os.cpu_count() or 1) workers; checks declaring
        thread_safe=False always run in the serial fallback regardless of
        this value. Result order matches registration order in both modes
        (the determinism invariant: serial and parallel produce identical
        CheckResult sets).
        """
        has_video = len(ds.cameras) > 0
        oversized_rows = _oversized_shard_rows(ds)

        # runnable holds (result-slot index, check) pairs so the serial and
        # parallel branches below write back to the correct position in
        # results even though skipped checks (no-video, oversized-shard)
        # occupy slots in between runnable entries.
        runnable: list[tuple[int, Check]] = []
        results: list[CheckResult | None] = []
        for check in self._registry.all_checks():
            if check.requires_video and not has_video:
                log.debug("check.skipped.no_video", check_id=check.id)
                continue

            if oversized_rows is not None and check.id in _DATA_READING_CHECK_IDS:
                result = CheckResult(
                    check_id=check.id,
                    severity=Severity.ERROR,
                    message=(
                        f"dataset declares a shard with {oversized_rows} rows exceeding "
                        f"_MAX_SHARD_ROWS={_MAX_SHARD_ROWS}; skipping to prevent "
                        f"allocation bomb"
                    ),
                )
                log.error(
                    "check.skipped.oversized_shard",
                    check_id=check.id,
                    shard_rows=oversized_rows,
                    max_shard_rows=_MAX_SHARD_ROWS,
                )
                results.append(result)
                continue

            results.append(None)
            runnable.append((len(results) - 1, check))

        if parallel > 1:
            self._run_parallel(runnable, ds, ctx, results, parallel=parallel)
        else:
            for i, check in runnable:
                results[i] = self._run_one(check, ds, ctx)

        final_results: list[CheckResult] = []
        for maybe_result in results:
            if maybe_result is None:
                raise AssertionError("check result slot was not filled by any run branch")
            final_results.append(maybe_result)
            log.info(
                "check.result",
                check_id=maybe_result.check_id,
                severity=maybe_result.severity.value,
                message=maybe_result.message,
            )

        return final_results

    def _run_parallel(
        self,
        runnable: list[tuple[int, Check]],
        ds: CanonicalDataset,
        ctx: CheckContext,
        results: list[CheckResult | None],
        *,
        parallel: int,
    ) -> None:
        """Fill *results* in place: thread_safe checks in worker processes, others serial.

        Each entry in *runnable* is (result-slot index, check); the index is
        threaded through explicitly so writes land in the correct slot
        regardless of how many other checks were skipped (no-video,
        oversized-shard) ahead of a given entry.
        """
        worker_count = min(parallel, os.cpu_count() or 1)
        parallel_entries = [(i, c) for i, c in runnable if c.thread_safe]
        serial_entries = [(i, c) for i, c in runnable if not c.thread_safe]

        for i, check in serial_entries:
            results[i] = self._run_one(check, ds, ctx)

        if not parallel_entries:
            return

        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_run_check_in_worker, check, ds, ctx): i
                for i, check in parallel_entries
            }
            for future, i in futures.items():
                results[i] = future.result()

    def _run_one(self, check: Check, ds: CanonicalDataset, ctx: CheckContext) -> CheckResult:
        """Run a single check, converting any exception to an ERROR result (ADR-003)."""
        try:
            return check.run(ds, ctx)
        except Exception as exc:
            exc_type = type(exc).__name__
            log.error(
                "check.crashed",
                check_id=check.id,
                exc_type=exc_type,
                exc_message=str(exc),
            )
            return CheckResult(
                check_id=check.id,
                severity=Severity.ERROR,
                message=(
                    f"Check could not be evaluated — it raised {exc_type}: {exc}. "
                    f"This is a trajlens bug; please report it."
                ),
                details={"exc_type": exc_type, "exc_message": str(exc)},
            )
