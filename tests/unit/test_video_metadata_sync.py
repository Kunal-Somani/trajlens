"""Unit tests for VideoMetadataSyncFixer (REPAIR.VIDEO_METADATA_SYNC).

Coverage per 05_ENGINEERING_STANDARDS.md §5 and ADR-004:
  - Happy path + mandatory round-trip test: mismatched fps -> fixer -> re-lint
    -> full CheckEngine set-diff shows no new FAIL/WARN and the fixer's own
    dry_run() is a noop against the repaired output (the target invariant
    VIDEO.RESOLUTION_FPS_MATCH names is not yet an implemented check to
    re-run directly -- see module docstring).
  - Failure modes: undecodable video, no video feature declared, v2 dataset rejected.
  - Edge case: already-consistent dataset (no-op).
  - Byte-identity outside meta/info.json.
  - Dry-run zero-write test (mtime-based, content-based).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fixtures.builders import (
    build_v2_dataset,
    build_v3_corrupt_video,
    build_v3_no_video_feature,
    build_v3_video_fps_match,
    build_v3_video_fps_mismatch,
)
from trajlens.checks import CheckEngine, registry
from trajlens.checks.protocol import CheckContext, Severity
from trajlens.errors import RepairError
from trajlens.model import build_canonical_dataset
from trajlens.repair.protocol import Diff, FeatureFieldChange
from trajlens.repair.video_metadata_sync import CHECK_ID, FIXER_ID, VideoMetadataSyncFixer
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
        fixer = VideoMetadataSyncFixer()
        assert fixer.fixer_id == FIXER_ID
        assert fixer.check_id == CHECK_ID
        assert fixer.fixer_id == "REPAIR.VIDEO_METADATA_SYNC"
        assert fixer.check_id == "VIDEO.RESOLUTION_FPS_MATCH"


# ---------------------------------------------------------------------------
# Happy path + mandatory ADR-004 round-trip test
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_repair_clears_fps_mismatch_and_no_new_findings(self, tmp_path: Path) -> None:
        """ADR-004 mandatory round-trip: repair -> re-lint -> no new FAIL/WARN, fixer converges."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_video_fps_mismatch(source, declared_fps=30, container_fps=24)

        engine = CheckEngine(registry)
        ds_source = _load(source)
        pre_results = engine.run(ds_source, CTX).results
        pre_fail_ids = {r.check_id for r in pre_results if r.severity >= Severity.WARN}

        fixer = VideoMetadataSyncFixer()
        diff = fixer.dry_run(ds_source)
        assert not diff.is_noop, "fps-mismatch fixture must trigger a non-empty diff"

        summary = fixer.apply(ds_source, output)
        assert summary.output_path == output
        assert summary.changes_written == 1

        ds_fixed = _load(output)
        post_diff = fixer.dry_run(ds_fixed)
        assert post_diff.is_noop, "fixer's own dry_run() must be a noop against repaired output"

        post_results = engine.run(ds_fixed, CTX).results
        post_fail_ids = {r.check_id for r in post_results if r.severity >= Severity.WARN}
        new_findings = post_fail_ids - pre_fail_ids
        assert not new_findings, (
            f"repair introduced new WARN/FAIL findings not present in source: {new_findings}"
        )

    def test_dry_run_produces_no_filesystem_writes(self, tmp_path: Path) -> None:
        """dry_run() must not write, create, or modify any file."""
        source = tmp_path / "source"
        build_v3_video_fps_mismatch(source, declared_fps=30, container_fps=24)

        before = {p: p.stat().st_mtime for p in source.rglob("*") if p.is_file()}
        before_content = _content_tree(source)

        fixer = VideoMetadataSyncFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        after = {p: p.stat().st_mtime for p in source.rglob("*") if p.is_file()}
        after_content = _content_tree(source)

        assert before == after, "dry_run() must not touch any file's mtime"
        assert before_content == after_content, "dry_run() must not change any file's content"
        assert not diff.is_noop

    def test_dry_run_diff_contents(self, tmp_path: Path) -> None:
        """Diff records correct field and old/new fps values."""
        source = tmp_path / "source"
        build_v3_video_fps_mismatch(source, declared_fps=30, container_fps=24)

        fixer = VideoMetadataSyncFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        assert isinstance(diff, Diff)
        assert diff.check_id == CHECK_ID
        assert diff.fixer_id == FIXER_ID
        assert len(diff.changes) == 1

        change = diff.changes[0]
        assert isinstance(change, FeatureFieldChange)
        assert change.field == "fps"
        assert change.old_value == pytest.approx(30.0)
        assert change.new_value == pytest.approx(24.0, rel=1e-2)

    def test_repaired_info_json_fps_matches_container(self, tmp_path: Path) -> None:
        """After repair, info.json's fps equals the video container's average_rate."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_video_fps_mismatch(source, declared_fps=30, container_fps=24)

        fixer = VideoMetadataSyncFixer()
        ds = _load(source)
        fixer.apply(ds, output)

        info = json.loads((output / "meta" / "info.json").read_text())
        assert info["fps"] == pytest.approx(24.0, rel=1e-2)


# ---------------------------------------------------------------------------
# Byte-identity outside meta/info.json
# ---------------------------------------------------------------------------


class TestByteIdentity:
    def test_apply_only_rewrites_info_json(self, tmp_path: Path) -> None:
        """apply() must produce an output where only meta/info.json differs from source."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_video_fps_mismatch(source, declared_fps=30, container_fps=24)

        fixer = VideoMetadataSyncFixer()
        ds = _load(source)
        fixer.apply(ds, output)

        info_rel = "meta/info.json"
        for src_file in source.rglob("*"):
            if not src_file.is_file():
                continue
            rel = str(src_file.relative_to(source))
            out_file = output / rel
            assert out_file.is_file(), f"output missing file: {rel}"
            if rel == info_rel:
                assert src_file.read_bytes() != out_file.read_bytes(), (
                    "meta/info.json content must differ after repair"
                )
            else:
                assert src_file.read_bytes() == out_file.read_bytes(), (
                    f"non-info.json file content changed unexpectedly: {rel}"
                )

    def test_video_bytes_byte_identical(self, tmp_path: Path) -> None:
        """The video shard itself must be byte-for-byte identical after repair."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_video_fps_mismatch(source, declared_fps=30, container_fps=24)

        video_rel = Path("videos") / "top" / "chunk-000" / "file-000.mp4"
        source_bytes = (source / video_rel).read_bytes()

        fixer = VideoMetadataSyncFixer()
        ds = _load(source)
        fixer.apply(ds, output)

        output_bytes = (output / video_rel).read_bytes()
        assert source_bytes == output_bytes, "video shard bytes must be untouched by repair"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_clean_dataset_is_noop(self, tmp_path: Path) -> None:
        """A dataset whose declared fps already matches the container must yield an empty Diff."""
        source = tmp_path / "source"
        build_v3_video_fps_match(source, fps=30)

        fixer = VideoMetadataSyncFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        assert diff.is_noop, "fps-matching dataset must yield a noop diff"
        assert len(diff.changes) == 0

    def test_clean_dataset_apply_is_noop_summary(self, tmp_path: Path) -> None:
        """apply() on a clean dataset copies the tree but reports zero changes."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_video_fps_match(source, fps=30)

        fixer = VideoMetadataSyncFixer()
        ds = _load(source)
        summary = fixer.apply(ds, output)

        assert summary.changes_written == 0
        assert summary.frames_corrected == 0
        assert output.is_dir()

    def test_output_must_not_equal_source(self, tmp_path: Path) -> None:
        """apply() must raise RepairError when output_path == source root."""
        source = tmp_path / "source"
        build_v3_video_fps_match(source, fps=30)

        fixer = VideoMetadataSyncFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match="copy-on-write"):
            fixer.apply(ds, source)


# ---------------------------------------------------------------------------
# Failure modes / refusals
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_undecodable_video_raises_repair_error_zero_output(self, tmp_path: Path) -> None:
        """A corrupt/undecodable video shard must raise RepairError with zero output writes."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_corrupt_video(source)

        fixer = VideoMetadataSyncFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match="could not be decoded"):
            fixer.dry_run(ds)
        assert not output.exists(), "dry_run() failure must not create any output"

        with pytest.raises(RepairError, match="could not be decoded"):
            fixer.apply(ds, output)
        assert not output.exists(), "apply() failure must leave zero output files written"

    def test_no_video_feature_raises_repair_error(self, tmp_path: Path) -> None:
        """A dataset declaring no video feature at all must raise RepairError."""
        source = tmp_path / "source"
        build_v3_no_video_feature(source)

        fixer = VideoMetadataSyncFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match="video feature"):
            fixer.dry_run(ds)

    def test_v2_dataset_raises_repair_error(self, tmp_path: Path) -> None:
        """v2.x datasets are rejected with a clear RepairError."""
        source = tmp_path / "source"
        build_v2_dataset(source, codebase_version="v2.1", num_episodes=3)

        fixer = VideoMetadataSyncFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match=r"v3\.0"):
            fixer.dry_run(ds)
