"""Unit tests for EpisodeReindexFixer (REPAIR.EPISODE_REINDEX).

Coverage per 05_ENGINEERING_STANDARDS.md §5 and ADR-004, stricter than the
previous two fixers because a bug here can WRITE the same silent corruption
it claims to repair (06_SECURITY_AND_THREAT_MODEL.md T9):
  - Happy path + mandatory round-trip test: #2401-corrupted → fixer → re-lint → PASS.
  - Byte-identity: data/ and videos/ are untouched by apply().
  - SEMANTIC correctness: repaired boundaries select the RIGHT frames, not
    merely counts that add up (the guard against recreating #2401).
  - Refusal: internally-inconsistent data → RepairError, zero output written.
  - Dry-run zero-write (mtime pattern).
  - No-op: already-consistent dataset → noop Diff, apply is a pure copy.
  - Two failure modes: missing episode metadata, corrupt episode metadata parquet.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.fixtures.builders import (
    build_v2_dataset,
    build_v3_corrupt_episode_metadata,
    build_v3_dataset,
    build_v3_interleaved_episode_data,
    build_v3_metadata_data_disagreement,
    build_v3_metadata_from_index_wrong,
    build_v3_metadata_length_wrong,
    build_v3_missing_episode_metadata,
    build_v3_noncontiguous_index_column,
    build_v3_overlapping_index_ranges,
)
from trajlens.checks import CheckEngine, registry
from trajlens.checks.protocol import CheckContext, Severity
from trajlens.checks.structural import METADATA_DATA_AGREEMENT
from trajlens.errors import DatasetFormatError, RepairError
from trajlens.model import build_canonical_dataset
from trajlens.repair.episode_reindex import CHECK_ID, FIXER_ID, EpisodeReindexFixer
from trajlens.repair.protocol import BoundaryChange, Diff
from trajlens.sources.loader import SourceLoader

CTX = CheckContext(deep=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(root: Path):  # type: ignore[no-untyped-def]
    handle = SourceLoader().resolve(str(root))
    return build_canonical_dataset(handle)


def _has_agreement_finding(root: Path) -> bool:
    ds = _load(root)
    result = METADATA_DATA_AGREEMENT.run(ds, CTX)
    return result.severity is Severity.FAIL


def _all_files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


# ---------------------------------------------------------------------------
# Fixer identity / metadata
# ---------------------------------------------------------------------------


class TestFixerMetadata:
    def test_ids(self) -> None:
        fixer = EpisodeReindexFixer()
        assert fixer.fixer_id == FIXER_ID
        assert fixer.check_id == CHECK_ID
        assert fixer.fixer_id == "REPAIR.EPISODE_REINDEX"
        assert fixer.check_id == "STRUCTURAL.METADATA_DATA_AGREEMENT"


# ---------------------------------------------------------------------------
# Happy path + mandatory ADR-004 round-trip test
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_repair_clears_agreement_finding(self, tmp_path: Path) -> None:
        """ADR-004 mandatory round-trip: #2401 corruption -> repair -> re-lint -> finding gone."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_metadata_data_disagreement(source, num_episodes=3)

        assert _has_agreement_finding(source), (
            "fixture must trigger STRUCTURAL.METADATA_DATA_AGREEMENT"
        )

        fixer = EpisodeReindexFixer()
        ds = _load(source)
        summary = fixer.apply(ds, output)

        assert summary.output_path == output
        assert summary.changes_written >= 1
        assert summary.frames_corrected > 0

        assert not _has_agreement_finding(output), (
            "STRUCTURAL.METADATA_DATA_AGREEMENT must be INFO after repair"
        )

    def test_dry_run_produces_no_filesystem_writes(self, tmp_path: Path) -> None:
        """dry_run() must not write, create, or modify any file."""
        source = tmp_path / "source"
        build_v3_metadata_data_disagreement(source, num_episodes=3)

        before = {p: p.stat().st_mtime for p in source.rglob("*") if p.is_file()}

        fixer = EpisodeReindexFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        after = {p: p.stat().st_mtime for p in source.rglob("*") if p.is_file()}

        assert before == after, "dry_run() must not touch any file on disk"
        assert not diff.is_noop, "#2401 fixture must produce a non-empty diff"

    def test_dry_run_diff_contents(self, tmp_path: Path) -> None:
        """Diff records correct episode_index/field/old_value/new_value."""
        source = tmp_path / "source"
        build_v3_metadata_data_disagreement(source, num_episodes=3)

        fixer = EpisodeReindexFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        assert isinstance(diff, Diff)
        assert diff.check_id == CHECK_ID
        assert diff.fixer_id == FIXER_ID
        assert len(diff.changes) > 0

        for change in diff.changes:
            assert isinstance(change, BoundaryChange)
            assert change.field in ("dataset_from_index", "dataset_to_index", "length")

        # The fixture corrupts dataset_to_index by +1 for every episode.
        to_changes = [c for c in diff.changes if c.field == "dataset_to_index"]
        assert len(to_changes) == 3
        for c in to_changes:
            assert c.old_value == c.new_value + 1

    def test_no_new_finding_introduced(self, tmp_path: Path) -> None:
        """Repair must not introduce any new FAIL/WARN that wasn't already present.

        Uses the full default check suite so that side-effects on adjacent
        checks (e.g. INDEX_CONTINUITY, PATH_TEMPLATE_RESOLVES) are caught.
        """
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_metadata_data_disagreement(source, num_episodes=3)

        engine = CheckEngine(registry)

        ds_source = _load(source)
        pre_results = engine.run(ds_source, CTX)
        pre_fail_ids = {
            r.check_id
            for r in pre_results
            if r.severity >= Severity.WARN and r.check_id != CHECK_ID
        }

        fixer = EpisodeReindexFixer()
        fixer.apply(ds_source, output)

        ds_fixed = _load(output)
        post_results = engine.run(ds_fixed, CTX)

        agreement_post = next((r for r in post_results if r.check_id == CHECK_ID), None)
        assert agreement_post is None or agreement_post.severity < Severity.WARN, (
            f"{CHECK_ID} must not be WARN/FAIL after repair; got {agreement_post}"
        )

        post_fail_ids = {
            r.check_id
            for r in post_results
            if r.severity >= Severity.WARN and r.check_id != CHECK_ID
        }
        new_findings = post_fail_ids - pre_fail_ids
        assert not new_findings, (
            f"repair introduced new WARN/FAIL findings not present in source: {new_findings}"
        )


# ---------------------------------------------------------------------------
# Field coverage — from_index and length corrected independently of to_index
# ---------------------------------------------------------------------------


class TestFieldCoverage:
    def test_corrupted_from_index_is_corrected(self, tmp_path: Path) -> None:
        """A wrong dataset_from_index (not to_index) must be detected and fixed."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_metadata_from_index_wrong(source, num_episodes=3)

        assert _has_agreement_finding(source)

        fixer = EpisodeReindexFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)
        assert not diff.is_noop
        from_changes = [c for c in diff.changes if c.field == "dataset_from_index"]
        assert len(from_changes) >= 1

        fixer.apply(ds, output)
        assert not _has_agreement_finding(output)

    def test_corrupted_length_is_corrected(self, tmp_path: Path) -> None:
        """A wrong declared length (independent of from/to) must be detected and fixed."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_metadata_length_wrong(source, num_episodes=3)

        assert _has_agreement_finding(source)

        fixer = EpisodeReindexFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)
        assert not diff.is_noop
        length_changes = [c for c in diff.changes if c.field == "length"]
        assert len(length_changes) >= 1

        fixer.apply(ds, output)
        assert not _has_agreement_finding(output)

    def test_multi_shard_episode_metadata_only_rewrites_affected_shard(
        self, tmp_path: Path
    ) -> None:
        """With episode metadata split across multiple shards, only the shard
        containing the corrupted episode's row is rewritten.

        Exercises the skip-unaffected-shard branch in _rewrite_episode_shards.
        """
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_dataset(source, num_episodes=4, episodes_per_shard=2)

        ep_path = source / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        old = pq.read_table(ep_path)
        to_col = old.column("dataset_to_index").to_pylist()
        corrupted_to = [v + 1 for v in to_col]
        new = old.set_column(
            old.schema.get_field_index("dataset_to_index"),
            "dataset_to_index",
            pa.array(corrupted_to, type=pa.int64()),
        )
        pq.write_table(new, ep_path)

        assert _has_agreement_finding(source)

        fixer = EpisodeReindexFixer()
        ds = _load(source)
        fixer.apply(ds, output)

        assert not _has_agreement_finding(output)

        unaffected_shard = "meta/episodes/chunk-001/file-000.parquet"
        assert (source / unaffected_shard).read_bytes() == (
            output / unaffected_shard
        ).read_bytes(), "shard with no corrupted episodes must be byte-identical"

    def test_apply_overwrites_existing_output_directory(self, tmp_path: Path) -> None:
        """apply() must remove and replace a pre-existing output_path, not merge into it."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_metadata_data_disagreement(source, num_episodes=3)

        output.mkdir()
        (output / "stale_marker.txt").write_text("leftover from a previous run")

        fixer = EpisodeReindexFixer()
        ds = _load(source)
        fixer.apply(ds, output)

        assert not (output / "stale_marker.txt").exists(), (
            "apply() must fully replace a pre-existing output directory"
        )
        assert not _has_agreement_finding(output)


# ---------------------------------------------------------------------------
# Byte-identity outside episode metadata
# ---------------------------------------------------------------------------


class TestByteIdentity:
    def test_data_and_video_shards_are_byte_identical(self, tmp_path: Path) -> None:
        """apply() must produce data/ and videos/ trees byte-identical to source.

        Only meta/episodes/.../*.parquet may differ. A fixer that touches
        anything under data/ or videos/ has violated the copy-on-write
        guarantee this fixer exists to uphold.
        """
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_metadata_data_disagreement(source, num_episodes=3)

        fixer = EpisodeReindexFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)
        assert not diff.is_noop
        fixer.apply(ds, output)

        source_files = _all_files(source)
        output_files = _all_files(output)
        assert set(source_files) == set(output_files), "output tree must mirror source tree"

        for rel, src_bytes in source_files.items():
            if rel.startswith("data/") or rel.startswith("videos/"):
                assert output_files[rel] == src_bytes, (
                    f"non-metadata file changed unexpectedly: {rel}"
                )

    def test_only_episode_metadata_shard_differs(self, tmp_path: Path) -> None:
        """Every file outside meta/episodes/ must be byte-identical; that shard must differ."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_metadata_data_disagreement(source, num_episodes=3)

        fixer = EpisodeReindexFixer()
        ds = _load(source)
        fixer.apply(ds, output)

        source_files = _all_files(source)
        output_files = _all_files(output)

        changed = {rel for rel in source_files if source_files[rel] != output_files.get(rel)}
        assert changed, "expected at least one changed file (the episode metadata shard)"
        for rel in changed:
            assert rel.startswith("meta/episodes/"), (
                f"unexpected file changed outside meta/episodes/: {rel}"
            )


# ---------------------------------------------------------------------------
# Semantic correctness — the guard against recreating #2401
# ---------------------------------------------------------------------------


class TestSemanticCorrectness:
    def test_repaired_boundaries_select_correct_frames(self, tmp_path: Path) -> None:
        """After repair, each episode's new from/to boundary selects ONLY its own rows.

        This is the critical guard: a fixer that makes counts add up but
        assigns the wrong from/to range to an episode would still pass
        METADATA_DATA_AGREEMENT's count-based check while recreating the
        exact silent #2401 corruption. This test proves semantic correctness,
        not just check-passing.
        """
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_metadata_data_disagreement(source, num_episodes=4)

        fixer = EpisodeReindexFixer()
        ds = _load(source)
        fixer.apply(ds, output)

        ds_fixed = _load(output)

        data_path = output / "data" / "chunk-000" / "file-000.parquet"
        table = pq.read_table(data_path)
        ep_col = table.column("episode_index").to_pylist()
        idx_col = table.column("index").to_pylist()
        idx_to_ep = dict(zip(idx_col, ep_col, strict=True))

        for episode in ds_fixed:
            selected_indices = range(episode.dataset_from_index, episode.dataset_to_index)
            assert len(list(selected_indices)) == episode.length
            for global_idx in selected_indices:
                assert idx_to_ep[global_idx] == episode.episode_index, (
                    f"episode {episode.episode_index}'s boundary "
                    f"[{episode.dataset_from_index}, {episode.dataset_to_index}) selects "
                    f"row with index={global_idx} belonging to episode "
                    f"{idx_to_ep[global_idx]} instead"
                )


# ---------------------------------------------------------------------------
# Refusal path — first-class feature, not an afterthought
# ---------------------------------------------------------------------------


class TestRefusal:
    def test_interleaved_data_raises_repair_error(self, tmp_path: Path) -> None:
        """Internally-inconsistent data (interleaved episode_index) must be refused.

        No consistent boundary assignment exists for interleaved rows -- the
        fixer must fail closed with RepairError, never guess.
        """
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_interleaved_episode_data(source, num_episodes=2)

        fixer = EpisodeReindexFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match="interleaved"):
            fixer.dry_run(ds)

        with pytest.raises(RepairError):
            fixer.apply(ds, output)

        assert not output.exists(), "apply() must not write any output when refusing to repair"

    def test_refusal_message_is_actionable(self, tmp_path: Path) -> None:
        """The RepairError message must name the offending episode, not just 'error'."""
        source = tmp_path / "source"
        build_v3_interleaved_episode_data(source, num_episodes=2)

        fixer = EpisodeReindexFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match=r"episode \d+"):
            fixer.dry_run(ds)

    def test_noncontiguous_index_column_raises_repair_error(self, tmp_path: Path) -> None:
        """A gap in the data's global 'index' column must be refused, not guessed.

        A gap means no single [from, to) range can describe the episode's
        rows without also spanning a row that isn't its own.
        """
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_noncontiguous_index_column(source, num_episodes=2)

        fixer = EpisodeReindexFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match="not contiguous"):
            fixer.dry_run(ds)

        with pytest.raises(RepairError):
            fixer.apply(ds, output)
        assert not output.exists()

    def test_overlapping_index_ranges_raise_repair_error(self, tmp_path: Path) -> None:
        """Two episodes whose data claims overlapping global positions must be refused."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_overlapping_index_ranges(source, num_episodes=2)

        fixer = EpisodeReindexFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match="overlap"):
            fixer.dry_run(ds)

        with pytest.raises(RepairError):
            fixer.apply(ds, output)
        assert not output.exists()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_clean_dataset_is_noop(self, tmp_path: Path) -> None:
        """A dataset already in agreement must produce an empty Diff."""
        source = tmp_path / "source"
        build_v3_dataset(source, num_episodes=3)

        fixer = EpisodeReindexFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        assert diff.is_noop, "clean dataset must yield a noop diff"
        assert len(diff.changes) == 0

    def test_clean_dataset_apply_is_pure_copy(self, tmp_path: Path) -> None:
        """apply() on a clean dataset copies the tree unchanged with zero reported changes."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_dataset(source, num_episodes=3)

        fixer = EpisodeReindexFixer()
        ds = _load(source)
        summary = fixer.apply(ds, output)

        assert summary.frames_corrected == 0
        assert summary.changes_written == 0
        assert output.is_dir()

        source_files = _all_files(source)
        output_files = _all_files(output)
        assert source_files == output_files, "clean dataset apply() must be a pure copy"

    def test_output_must_not_equal_source(self, tmp_path: Path) -> None:
        """apply() must raise RepairError when output_path == source root."""
        source = tmp_path / "source"
        build_v3_dataset(source, num_episodes=3)

        fixer = EpisodeReindexFixer()
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

        fixer = EpisodeReindexFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match=r"v3\.0"):
            fixer.dry_run(ds)

    def test_missing_episode_metadata_raises_repair_error(self, tmp_path: Path) -> None:
        """Data referencing an episode absent from episode metadata must be refused."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_missing_episode_metadata(source)

        fixer = EpisodeReindexFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match="not declared in episode metadata"):
            fixer.dry_run(ds)

        with pytest.raises(RepairError):
            fixer.apply(ds, output)
        assert not output.exists()

    def test_corrupt_episode_metadata_raises(self, tmp_path: Path) -> None:
        """A corrupt (non-Parquet) episode-metadata shard must raise, not silently pass.

        The corruption surfaces when model/adapters.py tries to build the
        CanonicalDataset itself (episode metadata is read before any fixer
        runs), so this is a DatasetFormatError-shaped failure from the
        loading layer -- confirmed here so the fixer's test suite documents
        the full failure surface for a corrupt dataset, per the task's
        two-failure-mode requirement.
        """
        source = tmp_path / "source"
        build_v3_corrupt_episode_metadata(source)

        with pytest.raises((RepairError, DatasetFormatError, pa.lib.ArrowInvalid)):
            _load(source)
