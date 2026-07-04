"""Unit tests for TimestampDedriftFixer (REPAIR.TIMESTAMP_DEDRIFT).

Coverage per 05_ENGINEERING_STANDARDS.md §5 and ADR-004:
  - Happy path + mandatory round-trip test: drifted → fixer → re-lint → PASS.
  - At least two failure modes (v2 dataset rejected, no frame_index rejected).
  - Edge cases: already-clean dataset (no-op), single-episode dataset,
    zero-episode dataset, output == source raises RepairError.
  - Corrupt shard failure mode: _rewrite_shards propagates read errors.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tests.fixtures.builders import (
    build_v2_dataset,
    build_v3_dataset,
    build_v3_dataset_no_frame_index,
    build_v3_timestamp_drift,
)
from trajlens.checks import CheckEngine, registry
from trajlens.checks.protocol import CheckContext, Severity
from trajlens.checks.temporal import TIMESTAMP_DRIFT
from trajlens.errors import RepairError
from trajlens.model import build_canonical_dataset
from trajlens.repair.protocol import Diff, FrameChange
from trajlens.repair.timestamp_dedrift import (
    CHECK_ID,
    FIXER_ID,
    TimestampDedriftFixer,
    _ideal_ts,
    _quantize_type,
    _rewrite_shards,
)
from trajlens.sources.loader import SourceLoader

CTX = CheckContext(deep=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(root: Path):  # type: ignore[no-untyped-def]
    handle = SourceLoader().resolve(str(root))
    return build_canonical_dataset(handle)


def _has_drift_finding(root: Path) -> bool:
    ds = _load(root)
    result = TIMESTAMP_DRIFT.run(ds, CTX)
    return result.severity is Severity.FAIL


# ---------------------------------------------------------------------------
# Fixer identity / metadata
# ---------------------------------------------------------------------------


class TestFixerMetadata:
    def test_ids(self) -> None:
        fixer = TimestampDedriftFixer()
        assert fixer.fixer_id == FIXER_ID
        assert fixer.check_id == CHECK_ID
        assert fixer.fixer_id == "REPAIR.TIMESTAMP_DEDRIFT"
        assert fixer.check_id == "KNOWNBUG.TIMESTAMP_DRIFT"


# ---------------------------------------------------------------------------
# Happy path + mandatory ADR-004 round-trip test
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_repair_clears_drift_finding(self, tmp_path: Path) -> None:
        """ADR-004 mandatory round-trip: repair → re-lint → finding gone."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_timestamp_drift(source, num_episodes=5, drift_per_frame=5e-5)

        # Confirm the drift finding fires before repair.
        assert _has_drift_finding(source), "fixture must trigger KNOWNBUG.TIMESTAMP_DRIFT"

        fixer = TimestampDedriftFixer()
        ds = _load(source)
        summary = fixer.apply(ds, output)

        assert summary.output_path == output
        assert summary.frames_corrected > 0
        assert summary.changes_written >= 1

        # Re-lint the repaired output — the targeted finding must be gone.
        assert not _has_drift_finding(output), "KNOWNBUG.TIMESTAMP_DRIFT must be INFO after repair"

    def test_dry_run_produces_no_filesystem_writes(self, tmp_path: Path) -> None:
        """dry_run() must not write, create, or modify any file."""
        source = tmp_path / "source"
        build_v3_timestamp_drift(source, num_episodes=5, drift_per_frame=5e-5)

        before = {p: p.stat().st_mtime for p in source.rglob("*") if p.is_file()}

        fixer = TimestampDedriftFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        after = {p: p.stat().st_mtime for p in source.rglob("*") if p.is_file()}

        assert before == after, "dry_run() must not touch any file on disk"
        assert not diff.is_noop, "drifted fixture must produce a non-empty diff"

    def test_dry_run_diff_contents(self, tmp_path: Path) -> None:
        """Diff records correct episode/frame indices and old/new values."""
        source = tmp_path / "source"
        build_v3_timestamp_drift(source, num_episodes=3, drift_per_frame=5e-5)

        fixer = TimestampDedriftFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        assert isinstance(diff, Diff)
        assert diff.check_id == CHECK_ID
        assert diff.fixer_id == FIXER_ID
        assert len(diff.changes) > 0

        quantize = _quantize_type(ds)
        for change in diff.changes:
            assert isinstance(change, FrameChange)
            assert change.column == "timestamp"
            expected_ideal = _ideal_ts(change.frame_index, ds.fps, quantize)
            assert change.new_value == expected_ideal, (
                f"new_value={change.new_value} != ideal={expected_ideal} "
                f"for episode={change.episode_index} frame={change.frame_index}"
            )

    def test_no_new_finding_introduced(self, tmp_path: Path) -> None:
        """Repair must not introduce any new FAIL or ERROR findings on re-lint."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_timestamp_drift(source, num_episodes=5, drift_per_frame=5e-5)

        fixer = TimestampDedriftFixer()
        ds = _load(source)
        fixer.apply(ds, output)

        ds_fixed = _load(output)
        engine = CheckEngine(registry)
        results = engine.run(ds_fixed, CTX)

        new_fails = [
            r
            for r in results
            if r.severity >= Severity.FAIL and not r.check_id.startswith("VIDEO.")
        ]
        assert not new_fails, (
            f"repair introduced new FAIL/ERROR findings: "
            f"{[(r.check_id, r.message) for r in new_fails]}"
        )

    def test_repaired_timestamps_match_ideal(self, tmp_path: Path) -> None:
        """Every timestamp in the repaired shard equals frame_index / fps (quantized)."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_timestamp_drift(source, num_episodes=3, drift_per_frame=1e-4)

        fixer = TimestampDedriftFixer()
        ds = _load(source)
        fixer.apply(ds, output)

        ds_fixed = _load(output)
        quantize = _quantize_type(ds_fixed)
        fps = ds_fixed.fps

        shard = output / "data" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(shard)
        ts_list = table.column("timestamp").to_pylist()
        fi_list = table.column("frame_index").to_pylist()

        for fi, ts in zip(fi_list, ts_list, strict=True):
            ideal = float(quantize(int(fi) / fps))
            assert ts == ideal, f"frame_index={fi}: ts={ts} != ideal={ideal}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_clean_dataset_is_noop(self, tmp_path: Path) -> None:
        """A dataset already free of drift must produce an empty Diff."""
        source = tmp_path / "source"
        build_v3_dataset(source, num_episodes=3)

        fixer = TimestampDedriftFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        assert diff.is_noop, "clean dataset must yield a noop diff"
        assert len(diff.changes) == 0

    def test_clean_dataset_apply_is_noop_summary(self, tmp_path: Path) -> None:
        """apply() on a clean dataset copies the tree but reports zero changes."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_dataset(source, num_episodes=3)

        fixer = TimestampDedriftFixer()
        ds = _load(source)
        summary = fixer.apply(ds, output)

        assert summary.frames_corrected == 0
        assert summary.changes_written == 0
        assert output.is_dir()

    def test_single_episode_dataset(self, tmp_path: Path) -> None:
        """Single-episode dataset: round-trip must still work end-to-end."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_timestamp_drift(source, num_episodes=1, drift_per_frame=5e-5)

        fixer = TimestampDedriftFixer()
        ds = _load(source)
        fixer.apply(ds, output)

        assert not _has_drift_finding(output)

    def test_zero_episodes_is_noop(self, tmp_path: Path) -> None:
        """A dataset with zero episodes must be a no-op and not raise."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_dataset(source, num_episodes=0)

        fixer = TimestampDedriftFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        assert diff.is_noop
        summary = fixer.apply(ds, output)
        assert summary.frames_corrected == 0

    def test_output_must_not_equal_source(self, tmp_path: Path) -> None:
        """apply() must raise RepairError when output_path == source root."""
        source = tmp_path / "source"
        build_v3_dataset(source, num_episodes=3)

        fixer = TimestampDedriftFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match="copy-on-write"):
            fixer.apply(ds, source)


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_v2_dataset_raises_repair_error(self, tmp_path: Path) -> None:
        """v2.x datasets are rejected with a clear RepairError."""
        source = tmp_path / "source"
        build_v2_dataset(source, codebase_version="v2.1", num_episodes=3)

        fixer = TimestampDedriftFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match=r"v3\.0"):
            fixer.dry_run(ds)

    def test_no_frame_index_feature_raises_repair_error(self, tmp_path: Path) -> None:
        """Dataset without a bare 'frame_index' feature must raise RepairError."""
        source = tmp_path / "source"
        build_v3_dataset_no_frame_index(source, num_episodes=3)

        fixer = TimestampDedriftFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match="frame_index"):
            fixer.dry_run(ds)

    def test_corrupt_output_shard_raises(self, tmp_path: Path) -> None:
        """If a copied output shard is unreadable, _rewrite_shards must propagate the error.

        trajlens does not swallow Parquet read errors from its own copy — the
        caller sees the failure clearly (fail-closed, per ADR-003 spirit).
        """
        import shutil as _shutil

        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_timestamp_drift(source, num_episodes=3, drift_per_frame=5e-5)

        fixer = TimestampDedriftFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)
        assert not diff.is_noop

        _shutil.copytree(source, output)
        shard = output / "data" / "chunk-000" / "file-000.parquet"
        shard.write_bytes(b"CORRUPTED_NOT_PARQUET")

        with pytest.raises(Exception):  # noqa: B017
            _rewrite_shards(diff, output_root=output)
