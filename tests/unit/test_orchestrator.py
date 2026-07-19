"""Unit tests for repair/orchestrator.py (fixer selection, composition, chaining).

CLI-level behavior (exit codes, --json rendering, usage errors) is covered in
tests/unit/test_cli.py; these tests exercise the library layer directly, per
ADR-001 (library-first) -- the orchestrator must be independently testable
without going through the CLI.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.fixtures.builders import (
    build_v3_dataset,
    build_v3_drift_and_wrong_stats,
    build_v3_drift_fixed_by_dedrift_incidentally_clears_stats,
    build_v3_interleaved_episode_data,
    build_v3_metadata_data_disagreement,
    build_v3_orphan_data_shard,
    build_v3_timestamp_drift,
)
from trajlens.checks import CheckEngine, registry
from trajlens.checks.protocol import CheckContext
from trajlens.errors import RepairError
from trajlens.model import build_canonical_dataset
from trajlens.repair.orchestrator import (
    ALL_FIXER_IDS,
    build_fixer_order,
    refuse_if_hub_ref,
    run_apply,
    run_dry_run,
    select_applicable_fixers,
    validate_fixer_selection,
)
from trajlens.sources.loader import SourceLoader

CTX = CheckContext(deep=False)


def _load(root: Path):  # type: ignore[no-untyped-def]
    handle = SourceLoader().resolve(str(root))
    return build_canonical_dataset(handle)


def _lint(ds):  # type: ignore[no-untyped-def]
    return CheckEngine(registry).run(ds, CTX)


def _build_dangling_task_index(root: Path) -> None:
    """A v3.0 dataset with one dangling task_index (2), unambiguously nearest to
    the single defined task_index=0. Triggers SEMANTIC.TASK_INTEGRITY FAIL."""
    build_v3_dataset(root, num_episodes=2)
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_path)
    ti_col = table.column("task_index").to_pylist()
    ti_col[0] = 2
    new_table = table.set_column(
        table.schema.get_field_index("task_index"), "task_index", pa.array(ti_col, type=pa.int64())
    )
    pq.write_table(new_table, data_path)


class TestSelectApplicableFixers:
    def test_clean_dataset_selects_nothing(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=3)
        ds = _load(tmp_path)
        fixers = select_applicable_fixers(_lint(ds))
        assert fixers == []

    def test_reindex_only_selects_episode_reindex(self, tmp_path: Path) -> None:
        build_v3_metadata_data_disagreement(tmp_path, num_episodes=3)
        ds = _load(tmp_path)
        fixers = select_applicable_fixers(_lint(ds))
        fixer_ids = [f.fixer_id for f in fixers]
        assert fixer_ids == ["REPAIR.EPISODE_REINDEX"]

    def test_drift_only_selects_timestamp_dedrift(self, tmp_path: Path) -> None:
        build_v3_timestamp_drift(tmp_path, num_episodes=3, drift_per_frame=5e-5)
        ds = _load(tmp_path)
        fixers = select_applicable_fixers(_lint(ds))
        fixer_ids = [f.fixer_id for f in fixers]
        assert fixer_ids == ["REPAIR.TIMESTAMP_DEDRIFT"]

    def test_selection_order_is_fixed_not_discovery_order(self, tmp_path: Path) -> None:
        """Order must always be episode_reindex, timestamp_dedrift, stats_recompute
        regardless of which findings happen to fire, or in what order lint reports them.
        """
        build_v3_drift_and_wrong_stats(tmp_path, num_episodes=3, drift_per_frame=5e-5)
        ds = _load(tmp_path)
        fixers = select_applicable_fixers(_lint(ds))
        fixer_ids = [f.fixer_id for f in fixers]
        assert fixer_ids == ["REPAIR.TIMESTAMP_DEDRIFT", "REPAIR.STATS_RECOMPUTE"]


class TestRunDryRun:
    def test_clean_dataset_not_applicable(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=3)
        ds = _load(tmp_path)
        plan = run_dry_run(ds, _lint(ds))
        assert plan.applicable is False
        assert plan.outcomes == ()
        assert plan.output_path is None
        assert plan.applied is False

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        build_v3_metadata_data_disagreement(tmp_path, num_episodes=3)
        before = {p: p.stat().st_mtime for p in tmp_path.rglob("*") if p.is_file()}

        ds = _load(tmp_path)
        plan = run_dry_run(ds, _lint(ds))

        after = {p: p.stat().st_mtime for p in tmp_path.rglob("*") if p.is_file()}
        assert before == after
        assert plan.applicable is True
        assert all(o.summary is None for o in plan.outcomes)

    def test_dry_run_unrepairable_raises(self, tmp_path: Path) -> None:
        build_v3_interleaved_episode_data(tmp_path, num_episodes=2)
        ds = _load(tmp_path)
        with pytest.raises(RepairError):
            run_dry_run(ds, _lint(ds))


class TestRunApply:
    def test_clean_dataset_is_pure_copy(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        out = tmp_path / "repaired"
        build_v3_dataset(source, num_episodes=3)

        ds = _load(source)
        plan = run_apply(ds, _lint(ds), out)

        assert plan.applicable is False
        assert plan.applied is True
        assert out.is_dir()
        source_files = {p.relative_to(source) for p in source.rglob("*") if p.is_file()}
        out_files = {p.relative_to(out) for p in out.rglob("*") if p.is_file()}
        assert source_files == out_files

    def test_single_fixer_applies_and_writes_to_out(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        out = tmp_path / "repaired"
        build_v3_metadata_data_disagreement(source, num_episodes=3)

        ds = _load(source)
        plan = run_apply(ds, _lint(ds), out)

        assert plan.applicable is True
        assert len(plan.outcomes) == 1
        assert plan.outcomes[0].fixer_id == "REPAIR.EPISODE_REINDEX"
        assert plan.outcomes[0].summary is not None
        assert not _has_finding(out, "STRUCTURAL.METADATA_DATA_AGREEMENT")

    def test_unrepairable_raises_and_writes_nothing(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        out = tmp_path / "repaired"
        build_v3_interleaved_episode_data(source, num_episodes=2)

        ds = _load(source)
        with pytest.raises(RepairError):
            run_apply(ds, _lint(ds), out)

        assert not out.exists()

    def test_no_leftover_temp_dirs_after_successful_chain(self, tmp_path: Path) -> None:
        """Chaining two fixers must not leave temp directories behind."""
        import tempfile

        source = tmp_path / "source"
        out = tmp_path / "repaired"
        build_v3_drift_and_wrong_stats(source, num_episodes=3, drift_per_frame=5e-5)

        before_tmp = set(Path(tempfile.gettempdir()).glob("trajlens-fix-*"))
        ds = _load(source)
        run_apply(ds, _lint(ds), out)
        after_tmp = set(Path(tempfile.gettempdir()).glob("trajlens-fix-*"))

        assert after_tmp == before_tmp, "run_apply leaked a temp directory"

    def test_no_leftover_temp_dirs_after_failed_chain(self, tmp_path: Path) -> None:
        """A mid-chain RepairError must still clean up any temp dirs already created."""
        import tempfile

        source = tmp_path / "source"
        out = tmp_path / "repaired"
        # episode_reindex fires and would normally write; but data is also
        # interleaved, so episode_reindex itself raises inside the chain.
        build_v3_interleaved_episode_data(source, num_episodes=2)

        before_tmp = set(Path(tempfile.gettempdir()).glob("trajlens-fix-*"))
        ds = _load(source)
        with pytest.raises(RepairError):
            run_apply(ds, _lint(ds), out)
        after_tmp = set(Path(tempfile.gettempdir()).glob("trajlens-fix-*"))

        assert after_tmp == before_tmp, "run_apply leaked a temp directory on failure"

    def test_mid_chain_fixer_noop_after_earlier_fixer_ran(self, tmp_path: Path) -> None:
        """A fixer selected because its check fired FAIL initially may still
        turn out to be a noop once it runs against an EARLIER fixer's
        already-corrected output -- e.g. timestamp_dedrift incidentally
        brings stats back within STATISTICAL.STATS_MATCH_DATA's tolerance.
        This must be reported as a skipped (summary=None) outcome, not
        applied, and must not error.
        """
        source = tmp_path / "source"
        out = tmp_path / "repaired"
        build_v3_drift_fixed_by_dedrift_incidentally_clears_stats(source, num_episodes=3)

        ds = _load(source)
        results = _lint(ds)
        pre_fail_ids = {r.check_id for r in results if r.severity.value == "FAIL"}
        assert "KNOWNBUG.TIMESTAMP_DRIFT" in pre_fail_ids
        assert "STATISTICAL.STATS_MATCH_DATA" in pre_fail_ids

        plan = run_apply(ds, results, out)

        by_fixer = {o.fixer_id: o for o in plan.outcomes}
        assert by_fixer["REPAIR.TIMESTAMP_DEDRIFT"].summary is not None
        assert not by_fixer["REPAIR.TIMESTAMP_DEDRIFT"].diff.is_noop
        assert by_fixer["REPAIR.STATS_RECOMPUTE"].summary is None
        assert by_fixer["REPAIR.STATS_RECOMPUTE"].diff.is_noop

        assert not _has_finding(out, "KNOWNBUG.TIMESTAMP_DRIFT")
        assert not _has_finding(out, "STATISTICAL.STATS_MATCH_DATA")


def _has_finding(root: Path, check_id: str) -> bool:
    ds = _load(root)
    results = _lint(ds)
    result = next(r for r in results if r.check_id == check_id)
    return result.severity.value == "FAIL"


class TestRefuseIfHubRef:
    def test_local_ref_does_not_raise(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=3)
        handle = SourceLoader().resolve(str(tmp_path))
        refuse_if_hub_ref(handle, str(tmp_path))  # must not raise

    def test_hub_ref_raises_repair_error(self, tmp_path: Path) -> None:
        from trajlens.sources.loader import SourceHandle
        from trajlens.sources.version import DatasetVersion

        build_v3_dataset(tmp_path, num_episodes=3)
        handle = SourceLoader().resolve(str(tmp_path))
        # Synthesize a Hub-shaped handle without a network call: same root,
        # but with repo_id set, mirroring what SourceLoader.resolve() returns
        # for a real Hub ref (sources/loader.py resolve()).
        hub_handle = SourceHandle(
            root=handle.root,
            version=DatasetVersion.V3_0,
            info=handle.info,
            repo_id="org/some-dataset",
            revision=None,
        )
        with pytest.raises(RepairError, match="Hugging Face Hub"):
            refuse_if_hub_ref(hub_handle, "org/some-dataset")


class TestAllFixerIds:
    def test_includes_all_six_fixers(self) -> None:
        assert set(ALL_FIXER_IDS) == {
            "REPAIR.EPISODE_REINDEX",
            "REPAIR.TIMESTAMP_DEDRIFT",
            "REPAIR.STATS_RECOMPUTE",
            "REPAIR.TASK_INDEX_REPAIR",
            "REPAIR.VIDEO_METADATA_SYNC",
            "REPAIR.ORPHAN_SHARD_REPORT",
        }


class TestValidateFixerSelection:
    def test_empty_selection_is_valid(self) -> None:
        assert validate_fixer_selection(frozenset(), frozenset()) is None

    def test_valid_only_is_valid(self) -> None:
        assert (
            validate_fixer_selection(frozenset({"REPAIR.TASK_INDEX_REPAIR"}), frozenset()) is None
        )

    def test_unknown_only_id_returns_message_naming_it(self) -> None:
        msg = validate_fixer_selection(frozenset({"NOT_A_REAL_FIXER"}), frozenset())
        assert msg is not None
        assert "NOT_A_REAL_FIXER" in msg
        assert "REPAIR.TASK_INDEX_REPAIR" in msg  # valid ids listed

    def test_unknown_except_id_returns_message_naming_it(self) -> None:
        msg = validate_fixer_selection(frozenset(), frozenset({"NOT_A_REAL_FIXER"}))
        assert msg is not None
        assert "NOT_A_REAL_FIXER" in msg

    def test_same_id_in_both_only_and_except_is_contradiction(self) -> None:
        msg = validate_fixer_selection(
            frozenset({"REPAIR.TASK_INDEX_REPAIR"}), frozenset({"REPAIR.TASK_INDEX_REPAIR"})
        )
        assert msg is not None
        assert "REPAIR.TASK_INDEX_REPAIR" in msg
        assert "contradiction" in msg

    def test_unknown_id_reported_before_contradiction(self) -> None:
        """An id that is both unknown AND would be contradictory (if valid)
        must be reported as unknown, not silently treated as a contradiction."""
        msg = validate_fixer_selection(frozenset({"BOGUS"}), frozenset({"BOGUS"}))
        assert msg is not None
        assert "BOGUS" in msg
        assert "contradiction" not in msg


class TestBuildFixerOrder:
    def test_default_quarantine_false(self) -> None:
        order = build_fixer_order()
        orphan_fixers = [f for f in order if f.fixer_id == "REPAIR.ORPHAN_SHARD_REPORT"]
        assert len(orphan_fixers) == 1
        assert orphan_fixers[0].quarantine is False  # type: ignore[attr-defined]

    def test_quarantine_true_only_affects_orphan_fixer(self) -> None:
        order = build_fixer_order(quarantine=True)
        assert [f.fixer_id for f in order] == list(ALL_FIXER_IDS)
        orphan_fixers = [f for f in order if f.fixer_id == "REPAIR.ORPHAN_SHARD_REPORT"]
        assert orphan_fixers[0].quarantine is True  # type: ignore[attr-defined]


class TestOnlyExceptSelection:
    def test_only_selects_exact_fixer_regardless_of_findings(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=3)  # clean dataset, no findings
        ds = _load(tmp_path)
        fixers = select_applicable_fixers(_lint(ds), only=frozenset({"REPAIR.VIDEO_METADATA_SYNC"}))
        assert [f.fixer_id for f in fixers] == ["REPAIR.VIDEO_METADATA_SYNC"]

    def test_only_preserves_fixer_order_for_multiple_ids(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=3)
        ds = _load(tmp_path)
        fixers = select_applicable_fixers(
            _lint(ds),
            only=frozenset({"REPAIR.ORPHAN_SHARD_REPORT", "REPAIR.TASK_INDEX_REPAIR"}),
        )
        # _FIXER_ORDER has task_index_repair before orphan_shard_report.
        assert [f.fixer_id for f in fixers] == [
            "REPAIR.TASK_INDEX_REPAIR",
            "REPAIR.ORPHAN_SHARD_REPORT",
        ]

    def test_except_removes_from_warn_plus_selection(self, tmp_path: Path) -> None:
        build_v3_metadata_data_disagreement(tmp_path, num_episodes=3)
        ds = _load(tmp_path)
        fixers = select_applicable_fixers(_lint(ds), except_=frozenset({"REPAIR.EPISODE_REINDEX"}))
        assert fixers == []

    def test_except_removes_from_only_selection(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=3)
        ds = _load(tmp_path)
        fixers = select_applicable_fixers(
            _lint(ds),
            only=frozenset({"REPAIR.TASK_INDEX_REPAIR", "REPAIR.ORPHAN_SHARD_REPORT"}),
            except_=frozenset({"REPAIR.ORPHAN_SHARD_REPORT"}),
        )
        assert [f.fixer_id for f in fixers] == ["REPAIR.TASK_INDEX_REPAIR"]

    def test_task_index_repair_selected_by_default_when_fired(self, tmp_path: Path) -> None:
        _build_dangling_task_index(tmp_path)
        ds = _load(tmp_path)
        fixers = select_applicable_fixers(_lint(ds))
        assert [f.fixer_id for f in fixers] == ["REPAIR.TASK_INDEX_REPAIR"]

    def test_orphan_shard_report_selected_by_default_when_fired(self, tmp_path: Path) -> None:
        build_v3_orphan_data_shard(tmp_path)
        ds = _load(tmp_path)
        fixers = select_applicable_fixers(_lint(ds))
        assert [f.fixer_id for f in fixers] == ["REPAIR.ORPHAN_SHARD_REPORT"]

    def test_video_metadata_sync_never_selected_by_default(self, tmp_path: Path) -> None:
        """VIDEO.RESOLUTION_FPS_MATCH is not implemented, so its severity never
        appears in lint results -- video_metadata_sync is unreachable except
        via explicit --only, on any fixture."""
        build_v3_dataset(tmp_path, num_episodes=3)
        ds = _load(tmp_path)
        fixers = select_applicable_fixers(_lint(ds))
        assert "REPAIR.VIDEO_METADATA_SYNC" not in [f.fixer_id for f in fixers]

    def test_run_dry_run_with_only_task_index_repair(self, tmp_path: Path) -> None:
        _build_dangling_task_index(tmp_path)
        ds = _load(tmp_path)
        plan = run_dry_run(ds, _lint(ds), only=frozenset({"REPAIR.TASK_INDEX_REPAIR"}))
        assert plan.applicable is True
        assert [o.fixer_id for o in plan.outcomes] == ["REPAIR.TASK_INDEX_REPAIR"]

    def test_run_apply_with_quarantine_moves_orphan_shard(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        out = tmp_path / "repaired"
        build_v3_orphan_data_shard(source)

        ds = _load(source)
        plan = run_apply(ds, _lint(ds), out, fixer_order=build_fixer_order(quarantine=True))

        assert plan.applicable is True
        orphan_outcome = next(
            o for o in plan.outcomes if o.fixer_id == "REPAIR.ORPHAN_SHARD_REPORT"
        )
        assert orphan_outcome.summary is not None
        assert orphan_outcome.summary.changes_written == 1
        assert (out / ".trajlens-quarantine" / "quarantine_manifest.json").is_file()

    def test_run_apply_without_quarantine_reports_only(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        out = tmp_path / "repaired"
        build_v3_orphan_data_shard(source)

        ds = _load(source)
        plan = run_apply(ds, _lint(ds), out)  # default fixer_order: quarantine=False

        orphan_outcome = next(
            o for o in plan.outcomes if o.fixer_id == "REPAIR.ORPHAN_SHARD_REPORT"
        )
        assert orphan_outcome.summary is not None
        assert orphan_outcome.summary.changes_written == 0
        assert not (out / ".trajlens-quarantine").exists()
