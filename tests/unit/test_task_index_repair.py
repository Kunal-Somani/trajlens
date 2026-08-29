"""Unit tests for TaskIndexRepairFixer (REPAIR.TASK_INDEX_REPAIR).

Coverage per 05_ENGINEERING_STANDARDS.md §5 and ADR-004:
  - Happy path + mandatory round-trip test: dangling task_index -> fixer ->
    re-lint -> SEMANTIC.TASK_INTEGRITY clears, full CheckEngine set-diff.
  - Byte-identity: every file outside data/ shards is untouched.
  - Refusal: empty tasks.parquet -> RepairError, zero output files written.
  - Refusal: ambiguous (equidistant candidates) -> RepairError, zero output
    files written.
  - Dry-run zero-write (mtime pattern).
  - No-op: already-valid dataset -> noop Diff.
  - Failure modes: v2.x dataset rejected, missing task_index feature rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.fixtures.builders import (
    build_v2_dataset,
    build_v3_dataset,
    build_v3_missing_task,
)
from trajlens.checks import CheckEngine, registry
from trajlens.checks.protocol import CheckContext, Severity
from trajlens.checks.semantic import TASK_INTEGRITY
from trajlens.errors import RepairError
from trajlens.model import build_canonical_dataset
from trajlens.repair.protocol import Diff, FrameChange
from trajlens.repair.task_index_repair import (
    CHECK_ID,
    FIXER_ID,
    TaskIndexRepairFixer,
    _nearest_valid_task,
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


def _has_task_integrity_finding(root: Path) -> bool:
    ds = _load(root)
    result = TASK_INTEGRITY.run(ds, CTX)
    return result.severity is Severity.FAIL


def _write_empty_tasks_table(root: Path) -> None:
    """Overwrite meta/tasks.parquet with zero rows (no defined tasks at all)."""
    tasks_path = root / "meta" / "tasks.parquet"
    empty = pa.table(
        {"task_index": pa.array([], type=pa.int64()), "task": pa.array([], type=pa.string())}
    )
    pq.write_table(empty, tasks_path)


def _build_two_task_dataset_with_dangling_between(root: Path) -> None:
    """v3.0 dataset with task_index 0 and 10 defined; last frame references 5 (equidistant)."""
    build_v3_dataset(root, num_episodes=1)
    tasks_path = root / "meta" / "tasks.parquet"
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0, 10], type=pa.int64()),
                "task": pa.array(["do thing zero", "do thing ten"]),
            }
        ),
        tasks_path,
    )
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    ti_col = old.column("task_index").to_pylist()
    ti_col[-1] = 5  # equidistant from 0 and 10
    new = old.set_column(
        old.schema.get_field_index("task_index"), "task_index", pa.array(ti_col, type=pa.int64())
    )
    pq.write_table(new, data_path)


def _build_v3_dataset_no_task_index_feature(root: Path, *, camera: str = "top") -> None:
    """v3.0 dataset whose declared features lack 'task_index' (mirrors no_frame_index pattern)."""
    build_v3_dataset(root, camera=camera)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    del info["features"]["task_index"]
    info_path.write_text(json.dumps(info))


# ---------------------------------------------------------------------------
# Fixer identity / metadata
# ---------------------------------------------------------------------------


class TestFixerMetadata:
    def test_ids(self) -> None:
        fixer = TaskIndexRepairFixer()
        assert fixer.fixer_id == FIXER_ID
        assert fixer.check_id == CHECK_ID
        assert fixer.fixer_id == "REPAIR.TASK_INDEX_REPAIR"
        assert fixer.check_id == "SEMANTIC.TASK_INTEGRITY"


# ---------------------------------------------------------------------------
# Nearest-valid-task heuristic (pure function)
# ---------------------------------------------------------------------------


class TestNearestValidTask:
    def test_picks_strictly_nearest(self) -> None:
        assert _nearest_valid_task(7, [0, 5, 10]) == 5

    def test_exact_match_not_reachable_but_returns_self_if_present(self) -> None:
        assert _nearest_valid_task(5, [0, 5, 10]) == 5

    def test_equidistant_raises_repair_error(self) -> None:
        with pytest.raises(RepairError, match="equidistant"):
            _nearest_valid_task(5, [0, 10])


# ---------------------------------------------------------------------------
# Happy path + mandatory ADR-004 round-trip test
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_repair_clears_task_integrity_finding(self, tmp_path: Path) -> None:
        """ADR-004 mandatory round-trip: repair -> re-lint -> finding gone."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_missing_task(source)

        assert _has_task_integrity_finding(source), "fixture must trigger SEMANTIC.TASK_INTEGRITY"

        fixer = TaskIndexRepairFixer()
        ds = _load(source)
        summary = fixer.apply(ds, output)

        assert summary.output_path == output
        assert summary.frames_corrected > 0
        assert summary.changes_written >= 1

        assert not _has_task_integrity_finding(output), (
            "SEMANTIC.TASK_INTEGRITY must be INFO after repair"
        )

    def test_no_new_finding_introduced_full_check_engine(self, tmp_path: Path) -> None:
        """Full CheckEngine set-diff: no new WARN/FAIL introduced by repair."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_missing_task(source)

        engine = CheckEngine(registry)

        ds_source = _load(source)
        pre_results = engine.run(ds_source, CTX).results
        pre_fail_ids = {
            r.check_id
            for r in pre_results
            if r.severity >= Severity.WARN and r.check_id != CHECK_ID
        }

        fixer = TaskIndexRepairFixer()
        fixer.apply(ds_source, output)

        ds_fixed = _load(output)
        post_results = engine.run(ds_fixed, CTX).results

        task_post = next((r for r in post_results if r.check_id == CHECK_ID), None)
        assert task_post is None or task_post.severity < Severity.WARN

        post_fail_ids = {
            r.check_id
            for r in post_results
            if r.severity >= Severity.WARN and r.check_id != CHECK_ID
        }
        new_findings = post_fail_ids - pre_fail_ids
        assert not new_findings, f"repair introduced new findings: {new_findings}"

    def test_dry_run_produces_no_filesystem_writes(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_missing_task(source)

        before = {p: p.stat().st_mtime for p in source.rglob("*") if p.is_file()}

        fixer = TaskIndexRepairFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        after = {p: p.stat().st_mtime for p in source.rglob("*") if p.is_file()}

        assert before == after, "dry_run() must not touch any file on disk"
        assert not diff.is_noop

    def test_dry_run_diff_contents(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_missing_task(source)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        assert isinstance(diff, Diff)
        assert diff.check_id == CHECK_ID
        assert diff.fixer_id == FIXER_ID
        assert len(diff.changes) == 1

        change = diff.changes[0]
        assert isinstance(change, FrameChange)
        assert change.column == "task_index"
        assert change.old_value == 99
        assert change.new_value == 0  # only defined task_index is 0


# ---------------------------------------------------------------------------
# Byte-identity outside data/ shards
# ---------------------------------------------------------------------------


class TestByteIdentity:
    def test_only_data_shards_change_everything_else_byte_identical(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_missing_task(source)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)
        fixer.apply(ds, output)

        source_files = {p.relative_to(source): p for p in source.rglob("*") if p.is_file()}
        output_files = {p.relative_to(output): p for p in output.rglob("*") if p.is_file()}
        assert set(source_files) == set(output_files), "apply() must not add or remove files"

        for rel, src_path in source_files.items():
            out_path = output_files[rel]
            if str(rel).startswith("data/"):
                continue  # the rewritten shard(s) are expected to differ
            assert src_path.read_bytes() == out_path.read_bytes(), (
                f"{rel} differs between source and repaired output, but is outside data/"
            )


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class TestRefusals:
    def test_empty_tasks_table_raises_repair_error_and_writes_nothing(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_missing_task(source)
        _write_empty_tasks_table(source)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match="no tasks at all"):
            fixer.dry_run(ds)

        with pytest.raises(RepairError):
            fixer.apply(ds, output)
        assert not output.exists(), "apply() must write nothing on refusal"

    def test_ambiguous_equidistant_raises_repair_error_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        _build_two_task_dataset_with_dangling_between(source)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match="equidistant"):
            fixer.dry_run(ds)

        with pytest.raises(RepairError):
            fixer.apply(ds, output)
        assert not output.exists(), "apply() must write nothing on refusal"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_clean_dataset_is_noop(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_dataset(source, num_episodes=3)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        assert diff.is_noop
        assert len(diff.changes) == 0

    def test_clean_dataset_apply_is_noop_summary(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_dataset(source, num_episodes=3)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)
        summary = fixer.apply(ds, output)

        assert summary.frames_corrected == 0
        assert summary.changes_written == 0
        assert output.is_dir()

    def test_zero_episodes_is_noop(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_dataset(source, num_episodes=0)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        assert diff.is_noop
        summary = fixer.apply(ds, output)
        assert summary.frames_corrected == 0

    def test_output_must_not_equal_source(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_dataset(source, num_episodes=3)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match="copy-on-write"):
            fixer.apply(ds, source)


# ---------------------------------------------------------------------------
# apply() only rewrites diff shards
# ---------------------------------------------------------------------------


class TestApplyOnlyTouchesDiffShards:
    def test_apply_only_rewrites_shards_in_diff(self, tmp_path: Path) -> None:
        import shutil as _shutil

        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_missing_task(source)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)
        assert not diff.is_noop

        diff_shard_relpaths = {c.shard_path for c in diff.changes}

        _shutil.copytree(source, output)
        baseline_mtimes = {p: p.stat().st_mtime for p in output.rglob("*.parquet")}

        _rewrite_shards(diff, output_root=output)

        for shard_abs, before_mtime in baseline_mtimes.items():
            rel = str(shard_abs.relative_to(output))
            after_mtime = shard_abs.stat().st_mtime
            if after_mtime != before_mtime:
                assert rel in diff_shard_relpaths, (
                    f"shard '{rel}' rewritten but not in diff.changes"
                )


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_v2_dataset_raises_repair_error(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v2_dataset(source, codebase_version="v2.1", num_episodes=3)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match=r"v3\.0"):
            fixer.dry_run(ds)

    def test_no_task_index_feature_raises_repair_error(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        _build_v3_dataset_no_task_index_feature(source)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match="task_index"):
            fixer.dry_run(ds)

    def test_corrupt_output_shard_raises(self, tmp_path: Path) -> None:
        import shutil as _shutil

        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_missing_task(source)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)
        assert not diff.is_noop

        _shutil.copytree(source, output)
        shard = output / "data" / "chunk-000" / "file-000.parquet"
        shard.write_bytes(b"CORRUPTED_NOT_PARQUET")

        with pytest.raises(Exception):  # noqa: B017
            _rewrite_shards(diff, output_root=output)
