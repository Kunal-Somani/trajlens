"""REPAIR.VIDEO_METADATA_SYNC — fixer for VIDEO.RESOLUTION_FPS_MATCH.

Detection counterpart: src/trajlens/checks/video.py _DecodableSpotcheckCheck
(VIDEO.RESOLUTION_FPS_MATCH itself is catalog-only, not yet implemented as a
check; this fixer targets the invariant that check id names — LEROBOT.md §3
invariant 5 — using the same PyAV entry point (av.open()) DECODABLE_SPOTCHECK
already uses in-process).

Ground truth: the video container's own frame rate, read via PyAV
(``av.open(path).streams.video[0].average_rate``). The container is what the
decoder actually reads at train time, so a mismatch there is the real
corruption; the info.json declaration is what gets corrected.

Scope note (found during implementation, see PR body): 03_SPECS/LEROBOT.md's
verified info.json key set (DatasetInfo dataclass, lerobot 0.5.2) has exactly
one dataset-global ``fps`` field — there is no per-camera/per-feature fps and
no per-feature frame-count field. ``total_frames`` is a dataset-global
tabular row count, not a video-container-derived value, and does not
correspond 1:1 to any single video shard's frame count. This fixer therefore
only compares/corrects the single global ``fps`` field in info.json against
the video container's average_rate; frame-count is out of scope because no
declared field exists for it to correct. Re-muxing/re-encoding the video
itself is also out of scope — only info.json is ever rewritten.

ADR-004 requirements satisfied here:
  - Copy-on-write: source is never opened in write mode; output_path receives
    the corrected info.json.
  - Dry-run by default: dry_run() computes the Diff with zero filesystem writes.
  - Round-trip: tests in tests/repair/test_video_metadata_sync.py and
    tests/property/test_video_metadata_sync_properties.py verify
    repair → re-lint → no regression.
"""

from __future__ import annotations

import json
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import structlog

from trajlens.errors import RepairError
from trajlens.model.canonical import CanonicalDataset
from trajlens.repair.protocol import Diff, FeatureFieldChange, RepairSummary
from trajlens.sources.paths import safe_join

log = structlog.get_logger(__name__)

FIXER_ID = "REPAIR.VIDEO_METADATA_SYNC"
CHECK_ID = "VIDEO.RESOLUTION_FPS_MATCH"

_INFO_REL: tuple[str, ...] = ("meta", "info.json")

# Relative tolerance for comparing declared fps against the container's
# average_rate. Matches STATISTICAL.STATS_MATCH_DATA's rtol convention
# (checks/statistical.py _STATS_RTOL) so a fps fixer and a stats fixer don't
# apply different notions of "close enough" to the same dataset.
_FPS_RTOL: float = 1e-4


def _dataset_root(ds: CanonicalDataset) -> Path:
    return ds.stats.root


def _read_container_fps(video_path: Path) -> Fraction:
    """Open *video_path* via PyAV (same call DECODABLE_SPOTCHECK uses) and return its fps.

    Raises RepairError if the container cannot be opened, has no video
    stream, or declares a zero/undefined frame rate.
    """
    try:
        with av.open(str(video_path)) as container:
            if not container.streams.video:
                raise RepairError(f"video shard {video_path} has no video stream to read fps from")
            stream = container.streams.video[0]
            rate = stream.average_rate
            if rate is None or rate <= 0:
                raise RepairError(
                    f"video shard {video_path} does not declare a usable average frame rate "
                    f"in its container metadata (got {rate!r})"
                )
            return Fraction(rate)
    except RepairError:
        raise
    except Exception as exc:
        # T5 (06_SECURITY_AND_THREAT_MODEL.md): malformed media must never
        # crash the process, regardless of the specific exception PyAV raises
        # for a given corruption (container/codec/IO errors all land here).
        raise RepairError(f"video shard {video_path} could not be decoded: {exc}") from exc


def _first_video_shard(ds: CanonicalDataset, root: Path) -> tuple[str, Path]:
    """Return (camera, path) for one representative video shard.

    Uses the first episode's first camera segment, mirroring
    DECODABLE_SPOTCHECK's own "gather unique shards" traversal but stopping
    at the first one found, since fps is a dataset-global property and every
    shard for a fixed-rate dataset is expected to agree.

    Callers must have already checked ds.cameras is non-empty
    (_check_preconditions does this before this function is ever reached).
    """
    episodes = list(ds)
    if not episodes:
        raise RepairError("dataset has no episodes; cannot resolve a video shard to read")
    camera = ds.cameras[0]
    try:
        segment = ds.video_segment_for_episode(episodes[0], camera)
    except Exception as exc:
        raise RepairError(f"could not resolve video segment for camera {camera!r}: {exc}") from exc
    if not segment.handle.is_local:
        raise RepairError(
            f"video shard for camera {camera!r} is not local (Hub streaming); "
            "REPAIR.VIDEO_METADATA_SYNC requires a local dataset"
        )
    path = segment.handle.path
    assert isinstance(path, Path)
    if not path.is_file():
        raise RepairError(f"video shard not found on disk: {path}")
    return camera, safe_join(root, *path.relative_to(root).parts)


def _load_info(root: Path) -> dict[str, Any]:
    path = safe_join(root, *_INFO_REL)
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RepairError(f"meta/info.json is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise RepairError(f"meta/info.json must be a JSON object; got {type(raw).__name__}")
    return raw


class VideoMetadataSyncFixer:
    """Rewrite info.json's declared ``fps`` to match the video container (ADR-004).

    dry_run() reads one representative video shard's container fps via PyAV
    and compares it to info.json's declared ``fps``. apply() copies the
    source tree to *output_path* and rewrites only meta/info.json.

    Only v3.0 datasets are supported (v2.x per-episode video shards would
    require reading every shard to confirm dataset-wide agreement, which is
    out of scope here — see timestamp_dedrift.py for the same v3.0-only
    rationale pattern). A RepairError is raised for v2.x inputs, undecodable
    video, or a dataset declaring no video feature.
    """

    fixer_id: str = FIXER_ID
    check_id: str = CHECK_ID

    def dry_run(self, ds: CanonicalDataset) -> Diff:
        """Compute what would change without writing anything.

        Returns a Diff with at most one FeatureFieldChange (field="fps") when
        the container's average_rate disagrees with info.json's declared fps
        beyond _FPS_RTOL. Returns a no-op Diff if they already agree.
        """
        _check_preconditions(ds)

        root = _dataset_root(ds)
        _camera, shard_path = _first_video_shard(ds, root)
        container_fps = float(_read_container_fps(shard_path))

        declared_fps = float(ds.fps)
        rel_err = abs(container_fps - declared_fps) / max(abs(container_fps), 1e-12)

        changes: list[FeatureFieldChange] = []
        if rel_err > _FPS_RTOL:
            changes.append(
                FeatureFieldChange(
                    feature="fps",
                    field="fps",
                    old_value=declared_fps,
                    new_value=container_fps,
                )
            )

        diff = Diff(changes=tuple(changes), check_id=CHECK_ID, fixer_id=FIXER_ID)
        if diff.is_noop:
            log.info(
                "video_metadata_sync.dry_run.noop",
                reason="declared fps already matches video container average_rate",
            )
        else:
            log.info(
                "video_metadata_sync.dry_run.changes_found",
                declared_fps=declared_fps,
                container_fps=container_fps,
            )
        return diff

    def apply(self, ds: CanonicalDataset, output_path: Path) -> RepairSummary:
        """Write a corrected copy of *ds* to *output_path* (copy-on-write).

        Steps:
          1. Precondition check (version, video feature presence).
          2. dry_run() to compute the Diff.
          3. Copy the source tree to output_path.
          4. If changed, rewrite meta/info.json's fps field.

        Only meta/info.json is rewritten. Video bytes are never opened in
        write mode. *output_path* must not be the source dataset root.
        """
        _check_preconditions(ds)

        source_root = _dataset_root(ds)
        if output_path.resolve() == source_root.resolve():
            raise RepairError(
                "output_path must not be the source dataset root (ADR-004 "
                f"copy-on-write). Got: {output_path}"
            )

        diff = self.dry_run(ds)

        log.info(
            "video_metadata_sync.apply.start",
            source=str(source_root),
            output=str(output_path),
            num_changes=len(diff.changes),
        )

        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(source_root, output_path)

        if diff.is_noop:
            log.info("video_metadata_sync.apply.noop", output=str(output_path))
            return RepairSummary(
                output_path=output_path,
                changes_written=0,
                frames_corrected=0,
            )

        fps_change = next(c for c in diff.changes if isinstance(c, FeatureFieldChange))
        _rewrite_info_json(output_path, fps_change)

        log.info(
            "video_metadata_sync.apply.done",
            output=str(output_path),
            old_fps=fps_change.old_value,
            new_fps=fps_change.new_value,
        )
        return RepairSummary(
            output_path=output_path,
            changes_written=1,
            frames_corrected=0,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_preconditions(ds: CanonicalDataset) -> None:
    """Raise RepairError for any condition that makes repair unsafe or undefined."""
    if ds.format_id != "lerobot" or ds.format_version != "3.0":
        raise RepairError(
            f"VideoMetadataSyncFixer only supports v3.0 datasets; "
            f"got {ds.format_label}. For v2.x datasets, convert to v3.0 first."
        )
    if not ds.cameras:
        raise RepairError(
            "VideoMetadataSyncFixer requires at least one declared video feature "
            "in info.json; this dataset declares none."
        )


def _rewrite_info_json(output_root: Path, change: FeatureFieldChange) -> None:
    """Rewrite output_root/meta/info.json's fps field to change.new_value."""
    path = safe_join(output_root, *_INFO_REL)
    info = _load_info(output_root)
    info["fps"] = change.new_value
    path.write_text(json.dumps(info))
    log.debug("video_metadata_sync.info_json_rewritten", path=str(path))
