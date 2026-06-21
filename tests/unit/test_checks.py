"""Unit tests for M4: Check Protocol, Registry, Engine, and all implemented checks.

Coverage targets (per 05_ENGINEERING_STANDARDS.md §5):
  - Every check: passing fixture, failing fixture, at least one edge case.
  - ADR-003: crashing check => ERROR, not propagated exception.
  - Registry: duplicate-id rejection.
  - Engine: no-video skipping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.builders import (
    build_v2_dataset,
    build_v3_bad_timestamp_spacing,
    build_v3_corrupt_video,
    build_v3_dataset,
    build_v3_long_episode_no_drift,
    build_v3_metadata_data_disagreement,
    build_v3_missing_shard,
    build_v3_non_monotonic_timestamps,
    build_v3_noncontiguous_indices,
    build_v3_real_video,
    build_v3_timestamp_drift,
    build_v3_wrong_schema,
)
from trajlens.checks.engine import CheckEngine
from trajlens.checks.protocol import Check, CheckContext, CheckResult, Severity
from trajlens.checks.registry import CheckRegistry
from trajlens.checks.structural import (
    INDEX_CONTINUITY,
    METADATA_DATA_AGREEMENT,
    PATH_TEMPLATE_RESOLVES,
    SCHEMA_CONSISTENCY,
    VERSION_DETECTED,
)
from trajlens.checks.temporal import (
    TIMESTAMP_DRIFT,
    TIMESTAMP_MONOTONIC,
    TIMESTAMP_SPACING,
)
from trajlens.checks.video import DECODABLE_SPOTCHECK
from trajlens.model import build_canonical_dataset
from trajlens.sources.loader import SourceLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(root: Path) -> Any:
    handle = SourceLoader().resolve(str(root))
    return build_canonical_dataset(handle)


CTX = CheckContext(deep=False)
CTX_DEEP = CheckContext(deep=True)


# ---------------------------------------------------------------------------
# Protocol & Severity
# ---------------------------------------------------------------------------


class TestSeverityOrdering:
    def test_ordering(self) -> None:
        assert Severity.INFO < Severity.WARN < Severity.FAIL < Severity.ERROR

    def test_worst_of(self) -> None:
        results = [Severity.WARN, Severity.INFO, Severity.FAIL, Severity.WARN]
        assert max(results) is Severity.FAIL


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestCheckRegistry:
    def test_register_and_retrieve(self) -> None:
        reg = CheckRegistry()

        class _Dummy:
            id = "TEST.DUMMY"
            severity = Severity.INFO
            category = "TEST"
            requires_video = False

            def run(self, ds: Any, ctx: Any) -> CheckResult:
                return CheckResult(check_id=self.id, severity=Severity.INFO, message="ok")

        inst = _Dummy()
        reg.register(inst)
        assert "TEST.DUMMY" in reg
        assert reg.get("TEST.DUMMY") is inst
        assert len(reg) == 1

    def test_duplicate_id_rejected(self) -> None:
        reg = CheckRegistry()

        class _Dup:
            id = "TEST.DUP"
            severity = Severity.INFO
            category = "TEST"
            requires_video = False

            def run(self, ds: Any, ctx: Any) -> CheckResult:  # pragma: no cover
                return CheckResult(check_id=self.id, severity=Severity.INFO, message="ok")

        reg.register(_Dup())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_Dup())

    def test_all_checks_stable_order(self) -> None:
        reg = CheckRegistry()
        ids = []
        for i in range(5):

            class _C:
                id = f"TEST.C{i}"
                severity = Severity.INFO
                category = "TEST"
                requires_video = False

                def run(self, ds: Any, ctx: Any) -> CheckResult:  # pragma: no cover
                    return CheckResult(check_id=self.id, severity=Severity.INFO, message="")

            _c = _C()
            _c.id = f"TEST.C{i}"
            reg.register(_c)
            ids.append(f"TEST.C{i}")
        assert [c.id for c in reg.all_checks()] == ids


# ---------------------------------------------------------------------------
# Engine — ADR-003
# ---------------------------------------------------------------------------


class TestCheckEngine:
    def _make_crashing_check(self) -> Check:
        class _Crash:
            id = "TEST.CRASH"
            severity = Severity.FAIL
            category = "TEST"
            requires_video = False

            def run(self, ds: Any, ctx: Any) -> CheckResult:
                raise RuntimeError("intentional crash for ADR-003 test")

        return _Crash()  # type: ignore[return-value]

    def _make_ok_check(self) -> Check:
        class _Ok:
            id = "TEST.OK"
            severity = Severity.INFO
            category = "TEST"
            requires_video = False

            def run(self, ds: Any, ctx: Any) -> CheckResult:
                return CheckResult(check_id=self.id, severity=Severity.INFO, message="all good")

        return _Ok()  # type: ignore[return-value]

    def test_crashing_check_yields_error_not_exception(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        ds = _load(tmp_path)
        reg = CheckRegistry()
        reg.register(self._make_crashing_check())
        engine = CheckEngine(reg)
        results = engine.run(ds, CTX)
        assert len(results) == 1
        assert results[0].severity is Severity.ERROR
        assert "RuntimeError" in results[0].message
        assert results[0].details["exc_type"] == "RuntimeError"

    def test_engine_skips_video_check_when_no_cameras(self, tmp_path: Path) -> None:
        # Dataset without video features.
        import json

        import pyarrow as pa
        import pyarrow.parquet as pq

        (tmp_path / "meta").mkdir()
        info = {
            "codebase_version": "v3.0",
            "fps": 30,
            "features": {
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
            "total_episodes": 1,
            "total_frames": 2,
        }
        (tmp_path / "meta" / "info.json").write_text(json.dumps(info))

        tasks_table = pa.table(
            {"task_index": pa.array([0], type=pa.int64()), "task": pa.array(["do thing"])}
        )
        pq.write_table(tasks_table, tmp_path / "meta" / "tasks.parquet")

        ep_dir = tmp_path / "meta" / "episodes" / "chunk-000"
        ep_dir.mkdir(parents=True)
        ep_table = pa.table(
            {
                "episode_index": pa.array([0], type=pa.int64()),
                "tasks": pa.array([["do thing"]], type=pa.list_(pa.string())),
                "length": pa.array([2], type=pa.int64()),
                "data/chunk_index": pa.array([0], type=pa.int64()),
                "data/file_index": pa.array([0], type=pa.int64()),
                "dataset_from_index": pa.array([0], type=pa.int64()),
                "dataset_to_index": pa.array([2], type=pa.int64()),
            }
        )
        pq.write_table(ep_table, ep_dir / "file-000.parquet")

        data_dir = tmp_path / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        data_table = pa.table(
            {
                "timestamp": pa.array([0.0, 1 / 30.0], type=pa.float32()),
                "frame_index": pa.array([0, 1], type=pa.int64()),
                "episode_index": pa.array([0, 0], type=pa.int64()),
                "index": pa.array([0, 1], type=pa.int64()),
                "task_index": pa.array([0, 0], type=pa.int64()),
            }
        )
        pq.write_table(data_table, data_dir / "file-000.parquet")

        ds = _load(tmp_path)
        assert ds.cameras == ()

        reg = CheckRegistry()

        class _VideoOnly:
            id = "TEST.VIDEO_ONLY"
            severity = Severity.FAIL
            category = "TEST"
            requires_video = True

            def run(self, ds2: Any, ctx: Any) -> CheckResult:  # pragma: no cover
                raise AssertionError("should not be called")

        reg.register(_VideoOnly())  # type: ignore[arg-type]
        engine = CheckEngine(reg)
        results = engine.run(ds, CTX)
        assert results == []  # skipped because no cameras

    def test_ok_check_passes_through(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        ds = _load(tmp_path)
        reg = CheckRegistry()
        reg.register(self._make_ok_check())
        engine = CheckEngine(reg)
        results = engine.run(ds, CTX)
        assert len(results) == 1
        assert results[0].severity is Severity.INFO


# ---------------------------------------------------------------------------
# STRUCTURAL.VERSION_DETECTED
# ---------------------------------------------------------------------------


class TestVersionDetected:
    def test_v3_reports_version(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        result = VERSION_DETECTED.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO
        assert "v3.0" in result.message
        assert result.details["version"] == "v3.0"

    def test_v2_reports_version(self, tmp_path: Path) -> None:
        build_v2_dataset(tmp_path, codebase_version="v2.1")
        result = VERSION_DETECTED.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO
        assert "v2.1" in result.message

    def test_zero_episodes(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=0)
        result = VERSION_DETECTED.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO


# ---------------------------------------------------------------------------
# STRUCTURAL.SCHEMA_CONSISTENCY
# ---------------------------------------------------------------------------


class TestSchemaConsistency:
    def test_clean_passes(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        result = SCHEMA_CONSISTENCY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO

    def test_wrong_dtype_fails(self, tmp_path: Path) -> None:
        build_v3_wrong_schema(tmp_path)
        result = SCHEMA_CONSISTENCY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.FAIL
        assert "timestamp" in result.message

    def test_zero_episodes_passes(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=0)
        result = SCHEMA_CONSISTENCY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO


# ---------------------------------------------------------------------------
# STRUCTURAL.INDEX_CONTINUITY
# ---------------------------------------------------------------------------


class TestIndexContinuity:
    def test_clean_passes(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        result = INDEX_CONTINUITY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO

    def test_noncontiguous_frame_index_fails(self, tmp_path: Path) -> None:
        build_v3_noncontiguous_indices(tmp_path)
        result = INDEX_CONTINUITY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.FAIL

    def test_single_episode_passes(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=1)
        result = INDEX_CONTINUITY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO

    def test_zero_episodes_passes(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=0)
        result = INDEX_CONTINUITY.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO


# ---------------------------------------------------------------------------
# STRUCTURAL.METADATA_DATA_AGREEMENT
# ---------------------------------------------------------------------------


class TestMetadataDataAgreement:
    def test_clean_passes(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        result = METADATA_DATA_AGREEMENT.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO

    def test_corrupted_to_index_fails(self, tmp_path: Path) -> None:
        """Directly replicates the #2401 corruption: from/to span > actual rows."""
        build_v3_metadata_data_disagreement(tmp_path)
        result = METADATA_DATA_AGREEMENT.run(_load(tmp_path), CTX)
        assert result.severity is Severity.FAIL
        # Check the message references the boundary mismatch.
        assert any(
            kw in result.message or any(kw in v for v in result.details["violations"])
            for kw in ["dataset_to_index", "to_index", "span", "length"]
        )

    def test_single_episode_clean_passes(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=1)
        result = METADATA_DATA_AGREEMENT.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO

    def test_zero_episodes_passes(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=0)
        result = METADATA_DATA_AGREEMENT.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO

    def test_v2_clean_passes(self, tmp_path: Path) -> None:
        build_v2_dataset(tmp_path)
        result = METADATA_DATA_AGREEMENT.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO


# ---------------------------------------------------------------------------
# STRUCTURAL.PATH_TEMPLATE_RESOLVES
# ---------------------------------------------------------------------------


class TestPathTemplateResolves:
    def test_clean_passes(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        result = PATH_TEMPLATE_RESOLVES.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO

    def test_missing_shard_fails(self, tmp_path: Path) -> None:
        build_v3_missing_shard(tmp_path)
        result = PATH_TEMPLATE_RESOLVES.run(_load(tmp_path), CTX)
        assert result.severity is Severity.FAIL


# ---------------------------------------------------------------------------
# TEMPORAL.TIMESTAMP_MONOTONIC
# ---------------------------------------------------------------------------


class TestTimestampMonotonic:
    def test_clean_passes(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        result = TIMESTAMP_MONOTONIC.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO

    def test_reversed_timestamps_fail(self, tmp_path: Path) -> None:
        build_v3_non_monotonic_timestamps(tmp_path)
        result = TIMESTAMP_MONOTONIC.run(_load(tmp_path), CTX)
        assert result.severity is Severity.FAIL
        assert "not strictly increasing" in result.message

    def test_single_frame_episode_passes(self, tmp_path: Path) -> None:
        # Single-frame episode is trivially monotonic.
        import json

        import pyarrow as pa
        import pyarrow.parquet as pq

        from tests.fixtures.builders import DEFAULT_FEATURES, _video_feature

        n = 1
        (tmp_path / "meta").mkdir()
        info = {
            "codebase_version": "v3.0",
            "fps": 30,
            "features": {**DEFAULT_FEATURES, **_video_feature("top")},
            "total_episodes": 1,
            "total_frames": n,
        }
        (tmp_path / "meta" / "info.json").write_text(json.dumps(info))

        tasks_t = pa.table(
            {"task_index": pa.array([0], type=pa.int64()), "task": pa.array(["do thing"])}
        )
        pq.write_table(tasks_t, tmp_path / "meta" / "tasks.parquet")

        ep_dir = tmp_path / "meta" / "episodes" / "chunk-000"
        ep_dir.mkdir(parents=True)
        ep_t = pa.table(
            {
                "episode_index": pa.array([0], type=pa.int64()),
                "tasks": pa.array([["do thing"]], type=pa.list_(pa.string())),
                "length": pa.array([1], type=pa.int64()),
                "data/chunk_index": pa.array([0], type=pa.int64()),
                "data/file_index": pa.array([0], type=pa.int64()),
                "dataset_from_index": pa.array([0], type=pa.int64()),
                "dataset_to_index": pa.array([1], type=pa.int64()),
                "videos/top/chunk_index": pa.array([0], type=pa.int64()),
                "videos/top/file_index": pa.array([0], type=pa.int64()),
                "videos/top/from_timestamp": pa.array([0.0]),
                "videos/top/to_timestamp": pa.array([1 / 30.0]),
            }
        )
        pq.write_table(ep_t, ep_dir / "file-000.parquet")

        data_dir = tmp_path / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        data_t = pa.table(
            {
                "timestamp": pa.array([0.0], type=pa.float32()),
                "frame_index": pa.array([0], type=pa.int64()),
                "episode_index": pa.array([0], type=pa.int64()),
                "index": pa.array([0], type=pa.int64()),
                "task_index": pa.array([0], type=pa.int64()),
            }
        )
        pq.write_table(data_t, data_dir / "file-000.parquet")

        video_dir = tmp_path / "videos" / "top" / "chunk-000"
        video_dir.mkdir(parents=True)
        (video_dir / "file-000.mp4").write_bytes(b"\x00")

        result = TIMESTAMP_MONOTONIC.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO


# ---------------------------------------------------------------------------
# TEMPORAL.TIMESTAMP_SPACING
# ---------------------------------------------------------------------------


class TestTimestampSpacing:
    def test_clean_passes(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        result = TIMESTAMP_SPACING.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO

    def test_large_gap_fails(self, tmp_path: Path) -> None:
        build_v3_bad_timestamp_spacing(tmp_path, gap_multiple=3.0)
        result = TIMESTAMP_SPACING.run(_load(tmp_path), CTX)
        assert result.severity is Severity.FAIL

    def test_zero_episodes_passes(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=0)
        result = TIMESTAMP_SPACING.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO


# ---------------------------------------------------------------------------
# KNOWNBUG.TIMESTAMP_DRIFT
# ---------------------------------------------------------------------------


class TestTimestampDrift:
    def test_clean_passes(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path)
        result = TIMESTAMP_DRIFT.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO

    def test_drifted_fails(self, tmp_path: Path) -> None:
        """#3177 fingerprint: cumulative drift exceeds decoder tolerance."""
        build_v3_timestamp_drift(tmp_path, num_episodes=5, drift_per_frame=5e-5)
        result = TIMESTAMP_DRIFT.run(_load(tmp_path), CTX)
        assert result.severity is Severity.FAIL
        assert "#3177" in result.message
        assert result.details["lerobot_issue"] == "#3177"

    def test_zero_episodes_passes(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=0)
        result = TIMESTAMP_DRIFT.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO

    def test_no_real_drift_at_fps_with_no_exact_float32_passes(self, tmp_path: Path) -> None:
        """Regression test: pure float32 storage quantization must not false-fire.

        Mirrors the real lerobot/pusht false positive: fps=10 has no exact
        float32 representation, and with enough frames/episode the rounding
        error alone used to cross the 1e-4 tolerance with no real drift
        present. Comparing against a float32-quantized ideal value (matching
        what's actually stored on disk) must keep cumulative drift at ~0.
        """
        build_v3_long_episode_no_drift(tmp_path, fps=10, frames_per_episode=125, num_episodes=5)
        result = TIMESTAMP_DRIFT.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO
        assert result.details["cumulative_drift_s"] < 1e-6

    def test_drifted_fails_with_many_frames_per_episode(self, tmp_path: Path) -> None:
        """Real drift must still be caught even with the float32-quantized comparison."""
        build_v3_timestamp_drift(tmp_path, num_episodes=5, drift_per_frame=1e-4)
        result = TIMESTAMP_DRIFT.run(_load(tmp_path), CTX)
        assert result.severity is Severity.FAIL
        assert "#3177" in result.message


# ---------------------------------------------------------------------------
# VIDEO.DECODABLE_SPOTCHECK
# ---------------------------------------------------------------------------


class TestDecodableSpotcheck:
    def test_no_cameras_skipped_by_engine(self, tmp_path: Path) -> None:
        """Engine skips requires_video=True checks when no cameras present."""
        import json

        import pyarrow as pa
        import pyarrow.parquet as pq

        (tmp_path / "meta").mkdir()
        info = {
            "codebase_version": "v3.0",
            "fps": 30,
            "features": {
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
            "total_episodes": 1,
            "total_frames": 2,
        }
        (tmp_path / "meta" / "info.json").write_text(json.dumps(info))
        tasks_t = pa.table({"task_index": pa.array([0], type=pa.int64()), "task": pa.array(["t"])})
        pq.write_table(tasks_t, tmp_path / "meta" / "tasks.parquet")
        ep_dir = tmp_path / "meta" / "episodes" / "chunk-000"
        ep_dir.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "episode_index": pa.array([0], type=pa.int64()),
                    "tasks": pa.array([["t"]], type=pa.list_(pa.string())),
                    "length": pa.array([2], type=pa.int64()),
                    "data/chunk_index": pa.array([0], type=pa.int64()),
                    "data/file_index": pa.array([0], type=pa.int64()),
                    "dataset_from_index": pa.array([0], type=pa.int64()),
                    "dataset_to_index": pa.array([2], type=pa.int64()),
                }
            ),
            ep_dir / "file-000.parquet",
        )
        data_dir = tmp_path / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "timestamp": pa.array([0.0, 1 / 30.0], type=pa.float32()),
                    "frame_index": pa.array([0, 1], type=pa.int64()),
                    "episode_index": pa.array([0, 0], type=pa.int64()),
                    "index": pa.array([0, 1], type=pa.int64()),
                    "task_index": pa.array([0, 0], type=pa.int64()),
                }
            ),
            data_dir / "file-000.parquet",
        )
        ds = _load(tmp_path)
        assert ds.cameras == ()
        assert DECODABLE_SPOTCHECK.requires_video is True
        # Running directly with no cameras: the dataset has no video segments to
        # iterate, so no failures should be emitted.
        result = DECODABLE_SPOTCHECK.run(ds, CTX)
        assert result.severity is Severity.INFO

    def test_corrupt_video_fails(self, tmp_path: Path) -> None:
        build_v3_corrupt_video(tmp_path)
        result = DECODABLE_SPOTCHECK.run(_load(tmp_path), CTX)
        assert result.severity is Severity.FAIL
        assert len(result.details["failures"]) >= 1

    def test_real_video_passes(self, tmp_path: Path) -> None:
        """Success path: a genuinely decodable MP4 must yield INFO.

        This exercises _decode_frame_at_position through its full decode loop
        (av.open → stream → decode), not just the exception handler.
        build_v3_real_video writes a real libx264-encoded shard; PyAV can
        open it and yield frames, so all three spot-check positions succeed.
        """
        build_v3_real_video(tmp_path)
        result = DECODABLE_SPOTCHECK.run(_load(tmp_path), CTX)
        assert result.severity is Severity.INFO
        assert "successfully" in result.message
