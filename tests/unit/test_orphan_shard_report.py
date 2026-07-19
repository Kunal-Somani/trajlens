"""Unit tests for OrphanShardReportFixer (REPAIR.ORPHAN_SHARD_REPORT).

Coverage per 05_ENGINEERING_STANDARDS.md §5, ADR-004, and 08_ROADMAP.md's
T2c DoD:
  - dry_run returns exactly the planted orphan set, relative paths only,
    zero filesystem writes.
  - apply() without quarantine: no files moved, correct message, exit clean.
  - apply() with quarantine: orphans under .trajlens-quarantine/, manifest
    present and correct, every non-orphan file byte-identical to source.
  - grep guard: no user-facing string contains "delete".
  - no-op: clean dataset -> noop Diff, apply is a pure copy.
  - refusal: all-shards-orphaned (empty episode metadata) -> RepairError.
  - refusal: path traversal via a malicious camera key -> PathTraversalError.
  - round-trip: orphan fixture -> quarantine apply -> re-lint -> STRUCTURAL.
    ORPHAN_SHARD reports clean, no new FAIL/WARN via full CheckEngine set-diff.
  - clean-fixture no-false-positive + trust score on build_v3_real_video stays 100.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from tests.fixtures.builders import (
    build_v2_dataset,
    build_v3_all_shards_orphaned,
    build_v3_no_orphan_shards,
    build_v3_orphan_data_shard,
    build_v3_orphan_traversal_attempt,
    build_v3_orphan_video_shard,
    build_v3_real_video,
)
from trajlens.checks import CheckEngine, registry
from trajlens.checks.protocol import CheckContext, Severity
from trajlens.errors import PathTraversalError, RepairError
from trajlens.model import build_canonical_dataset
from trajlens.repair import orphan_shard_report as osr_module
from trajlens.repair.orphan_shard_report import (
    CHECK_ID,
    FIXER_ID,
    OrphanShardReportFixer,
    find_orphan_shards,
)
from trajlens.repair.protocol import Diff, FeatureFieldChange
from trajlens.sources.loader import SourceLoader

CTX = CheckContext(deep=False)


def _load(root: Path):  # type: ignore[no-untyped-def]
    handle = SourceLoader().resolve(str(root))
    return build_canonical_dataset(handle)


def _content_tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


# ---------------------------------------------------------------------------
# Fixer identity / metadata
# ---------------------------------------------------------------------------


class TestFixerMetadata:
    def test_ids(self) -> None:
        fixer = OrphanShardReportFixer()
        assert fixer.fixer_id == FIXER_ID
        assert fixer.check_id == CHECK_ID
        assert fixer.fixer_id == "REPAIR.ORPHAN_SHARD_REPORT"
        assert fixer.check_id == "STRUCTURAL.ORPHAN_SHARD"

    def test_default_quarantine_is_false(self) -> None:
        assert OrphanShardReportFixer().quarantine is False


# ---------------------------------------------------------------------------
# dry_run: exact orphan set, relative paths, zero writes
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_finds_exact_orphan_data_shard(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_orphan_data_shard(source)

        orphans = find_orphan_shards(_load(source))

        assert len(orphans) == 1
        assert orphans[0].relative_path == "data/chunk-000/file-001.parquet"

    def test_finds_exact_orphan_video_shard(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_orphan_video_shard(source, camera="top")

        orphans = find_orphan_shards(_load(source))

        assert len(orphans) == 1
        assert orphans[0].relative_path == "videos/top/chunk-000/file-001.mp4"

    def test_relative_paths_never_absolute(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_orphan_data_shard(source)

        orphans = find_orphan_shards(_load(source))

        for o in orphans:
            assert not Path(o.relative_path).is_absolute()
            assert str(source) not in o.relative_path

    def test_dry_run_produces_no_filesystem_writes(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_orphan_data_shard(source)

        before = {p: p.stat().st_mtime for p in source.rglob("*") if p.is_file()}
        before_content = _content_tree(source)

        fixer = OrphanShardReportFixer()
        diff = fixer.dry_run(_load(source))

        after = {p: p.stat().st_mtime for p in source.rglob("*") if p.is_file()}
        after_content = _content_tree(source)

        assert before == after, "dry_run() must not touch any file's mtime"
        assert before_content == after_content, "dry_run() must not change any file's content"
        assert not diff.is_noop

    def test_dry_run_zero_writes_even_with_quarantine_option_set(self, tmp_path: Path) -> None:
        """dry_run() never writes regardless of the quarantine option (Protocol contract)."""
        source = tmp_path / "source"
        build_v3_orphan_data_shard(source)

        before_content = _content_tree(source)
        fixer = OrphanShardReportFixer(quarantine=True)
        diff = fixer.dry_run(_load(source))
        after_content = _content_tree(source)

        assert before_content == after_content
        assert not diff.is_noop

    def test_dry_run_diff_contents(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_orphan_data_shard(source)

        fixer = OrphanShardReportFixer()
        diff = fixer.dry_run(_load(source))

        assert isinstance(diff, Diff)
        assert diff.check_id == CHECK_ID
        assert diff.fixer_id == FIXER_ID
        assert len(diff.changes) == 1
        change = diff.changes[0]
        assert isinstance(change, FeatureFieldChange)
        assert change.feature == "data/chunk-000/file-001.parquet"


# ---------------------------------------------------------------------------
# apply() without quarantine: report-only
# ---------------------------------------------------------------------------


class TestApplyWithoutQuarantine:
    def test_no_files_moved(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_orphan_data_shard(source)

        fixer = OrphanShardReportFixer(quarantine=False)
        summary = fixer.apply(_load(source), output)

        assert summary.changes_written == 0
        orphan_path = output / "data" / "chunk-000" / "file-001.parquet"
        assert orphan_path.is_file(), "orphan shard must remain in its original location"
        assert not (output / ".trajlens-quarantine").exists()

    def test_output_is_full_copy(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_orphan_data_shard(source)

        fixer = OrphanShardReportFixer(quarantine=False)
        fixer.apply(_load(source), output)

        source_files = {str(p.relative_to(source)) for p in source.rglob("*") if p.is_file()}
        output_files = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
        assert source_files == output_files


# ---------------------------------------------------------------------------
# apply() with quarantine: move, manifest, byte-identity
# ---------------------------------------------------------------------------


class TestApplyWithQuarantine:
    def test_orphan_moved_under_quarantine_dir(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_orphan_data_shard(source)

        fixer = OrphanShardReportFixer(quarantine=True)
        summary = fixer.apply(_load(source), output)

        assert summary.changes_written == 1
        original_location = output / "data" / "chunk-000" / "file-001.parquet"
        quarantined_location = (
            output / ".trajlens-quarantine" / "data" / "chunk-000" / "file-001.parquet"
        )
        assert not original_location.exists()
        assert quarantined_location.is_file()

    def test_manifest_present_and_correct(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_orphan_data_shard(source)

        fixer = OrphanShardReportFixer(quarantine=True)
        fixer.apply(_load(source), output)

        manifest_path = output / ".trajlens-quarantine" / "quarantine_manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text())
        assert len(manifest) == 1
        assert manifest[0]["relative_path"] == "data/chunk-000/file-001.parquet"
        assert manifest[0]["reason"] == "unreferenced by episode metadata"
        assert manifest[0]["size_bytes"] > 0

    def test_non_orphan_files_byte_identical(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_orphan_data_shard(source)

        fixer = OrphanShardReportFixer(quarantine=True)
        fixer.apply(_load(source), output)

        orphan_rel = "data/chunk-000/file-001.parquet"
        for src_file in source.rglob("*"):
            if not src_file.is_file():
                continue
            rel = str(src_file.relative_to(source))
            if rel == orphan_rel:
                continue
            out_file = output / rel
            assert out_file.is_file(), f"non-orphan file missing in output: {rel}"
            assert src_file.read_bytes() == out_file.read_bytes(), (
                f"non-orphan file content changed unexpectedly: {rel}"
            )

    def test_source_never_modified(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_orphan_data_shard(source)

        before_content = _content_tree(source)
        fixer = OrphanShardReportFixer(quarantine=True)
        fixer.apply(_load(source), output)
        after_content = _content_tree(source)

        assert before_content == after_content, "source tree must never be modified"

    def test_apply_overwrites_preexisting_output_dir(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_orphan_data_shard(source)
        output.mkdir()
        (output / "stale.txt").write_text("leftover from a previous run")

        fixer = OrphanShardReportFixer(quarantine=True)
        fixer.apply(_load(source), output)

        assert not (output / "stale.txt").exists()
        assert (output / ".trajlens-quarantine" / "quarantine_manifest.json").is_file()

    def test_quarantine_true_on_clean_dataset_is_noop(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_no_orphan_shards(source)

        fixer = OrphanShardReportFixer(quarantine=True)
        summary = fixer.apply(_load(source), output)

        assert summary.changes_written == 0
        assert not (output / ".trajlens-quarantine").exists()


# ---------------------------------------------------------------------------
# grep guard: no user-facing string mentions "delete"
# ---------------------------------------------------------------------------


class TestNoDeleteLanguage:
    def test_no_delete_in_user_facing_strings(self) -> None:
        """Greps only user-facing string literals: log event/kwargs, RepairError
        messages, and RepairSummary-adjacent output -- not the module's own
        prose docstrings/comments, which describe the implementation to
        developers and are never shown to a CLI user."""
        tree = ast.parse(inspect.getsource(osr_module))
        offending: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                is_log_call = (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "log"
                )
                is_repair_error = isinstance(func, ast.Name) and func.id == "RepairError"
                if not (is_log_call or is_repair_error):
                    continue
                for arg_node in ast.walk(node):
                    if (
                        isinstance(arg_node, ast.Constant)
                        and isinstance(arg_node.value, str)
                        and "delete" in arg_node.value.lower()
                    ):
                        offending.append(arg_node.value)
                    if isinstance(arg_node, ast.JoinedStr):
                        for value in arg_node.values:
                            if (
                                isinstance(value, ast.Constant)
                                and isinstance(value.value, str)
                                and "delete" in value.value.lower()
                            ):
                                offending.append(value.value)

        assert not offending, f"user-facing strings must never contain 'delete': {offending}"


# ---------------------------------------------------------------------------
# No-op: clean dataset
# ---------------------------------------------------------------------------


class TestCleanDatasetNoop:
    def test_clean_dataset_is_noop(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_no_orphan_shards(source)

        fixer = OrphanShardReportFixer()
        diff = fixer.dry_run(_load(source))

        assert diff.is_noop
        assert len(diff.changes) == 0

    def test_clean_dataset_apply_is_pure_copy(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_no_orphan_shards(source)

        fixer = OrphanShardReportFixer(quarantine=True)
        summary = fixer.apply(_load(source), output)

        assert summary.changes_written == 0
        source_content = _content_tree(source)
        output_content = _content_tree(output)
        assert source_content == output_content


# ---------------------------------------------------------------------------
# Refusal / edge cases
# ---------------------------------------------------------------------------


class TestRefusals:
    def test_all_shards_orphaned_refuses(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_all_shards_orphaned(source)

        with pytest.raises(RepairError, match="episode metadata"):
            find_orphan_shards(_load(source))

    def test_path_traversal_camera_key_refuses(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_orphan_traversal_attempt(source)

        with pytest.raises(PathTraversalError):
            find_orphan_shards(_load(source))

    def test_v2_dataset_raises_repair_error(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v2_dataset(source, codebase_version="v2.1", num_episodes=3)

        with pytest.raises(RepairError, match=r"v3\.0"):
            find_orphan_shards(_load(source))

    def test_output_must_not_equal_source(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_orphan_data_shard(source)

        fixer = OrphanShardReportFixer()
        with pytest.raises(RepairError, match="copy-on-write"):
            fixer.apply(_load(source), source)

    def test_unreadable_episode_shard_fails_closed_at_load_time(self, tmp_path: Path) -> None:
        """An unreadable episode shard is refused before a CanonicalDataset ever
        exists -- build_canonical_dataset() is the enforcement boundary here,
        not find_orphan_shards() (same reasoning as STRUCTURAL.
        REQUIRED_METADATA_PRESENT's catalog note)."""
        source = tmp_path / "source"
        build_v3_orphan_data_shard(source)
        episodes_shard = source / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        episodes_shard.write_bytes(b"not a valid parquet file")

        with pytest.raises(Exception):  # noqa: B017 - pyarrow's own ArrowInvalid, not ours to wrap
            _load(source)

    def test_malformed_locator_columns_fails_closed_at_load_time(self, tmp_path: Path) -> None:
        import pyarrow.parquet as pq

        from trajlens.errors import DatasetFormatError

        source = tmp_path / "source"
        build_v3_orphan_data_shard(source)
        episodes_shard = source / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(episodes_shard)
        table = table.drop(["data/chunk_index"])
        pq.write_table(table, episodes_shard)

        with pytest.raises(DatasetFormatError, match="data/chunk_index"):
            _load(source)


# ---------------------------------------------------------------------------
# Round-trip: report -> quarantine -> re-lint -> clean
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_quarantine_clears_orphan_shard_finding_no_new_findings(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_orphan_data_shard(source)

        engine = CheckEngine(registry)
        ds_source = _load(source)
        pre_results = engine.run(ds_source, CTX)
        pre_ids = {r.check_id for r in pre_results if r.severity >= Severity.WARN}
        assert CHECK_ID in pre_ids

        fixer = OrphanShardReportFixer(quarantine=True)
        fixer.apply(ds_source, output)

        ds_fixed = _load(output)
        post_results = engine.run(ds_fixed, CTX)
        post_ids = {r.check_id for r in post_results if r.severity >= Severity.WARN}

        assert CHECK_ID not in post_ids, "STRUCTURAL.ORPHAN_SHARD must be clear after quarantine"
        new_findings = post_ids - pre_ids
        assert not new_findings, f"quarantine introduced new findings: {new_findings}"

        post_diff = fixer.dry_run(ds_fixed)
        assert post_diff.is_noop, "fixer's own dry_run() must be a noop against quarantined output"


# ---------------------------------------------------------------------------
# No false positive on a known-clean real-video dataset; trust score unaffected
# ---------------------------------------------------------------------------


class TestNoFalsePositive:
    def test_real_video_fixture_reports_no_orphans(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_real_video(source)

        orphans = find_orphan_shards(_load(source))
        assert orphans == []

    def test_trust_score_on_real_video_stays_100(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_real_video(source)

        engine = CheckEngine(registry)
        results = engine.run(_load(source), CTX)
        orphan_result = next(r for r in results if r.check_id == CHECK_ID)
        assert orphan_result.severity == Severity.INFO

        worst = max((r.severity for r in results), default=Severity.INFO)
        assert worst == Severity.INFO, (
            f"build_v3_real_video must remain fully clean; got worst severity {worst}"
        )
