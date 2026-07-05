"""Unit tests for repair/orchestrator.py (fixer selection, composition, chaining).

CLI-level behavior (exit codes, --json rendering, usage errors) is covered in
tests/unit/test_cli.py; these tests exercise the library layer directly, per
ADR-001 (library-first) -- the orchestrator must be independently testable
without going through the CLI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.builders import (
    build_v3_dataset,
    build_v3_drift_and_wrong_stats,
    build_v3_drift_fixed_by_dedrift_incidentally_clears_stats,
    build_v3_interleaved_episode_data,
    build_v3_metadata_data_disagreement,
    build_v3_timestamp_drift,
)
from trajlens.checks import CheckEngine, registry
from trajlens.checks.protocol import CheckContext
from trajlens.errors import RepairError
from trajlens.model import build_canonical_dataset
from trajlens.repair.orchestrator import (
    refuse_if_hub_ref,
    run_apply,
    run_dry_run,
    select_applicable_fixers,
)
from trajlens.sources.loader import SourceLoader

CTX = CheckContext(deep=False)


def _load(root: Path):  # type: ignore[no-untyped-def]
    handle = SourceLoader().resolve(str(root))
    return build_canonical_dataset(handle)


def _lint(ds):  # type: ignore[no-untyped-def]
    return CheckEngine(registry).run(ds, CTX)


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
