"""REPAIR.ORPHAN_SHARD_REPORT — fixer for STRUCTURAL.ORPHAN_SHARD.

Detection counterpart: src/trajlens/checks/structural.py _OrphanShardCheck,
which calls find_orphan_shards() directly (this module owns the detection
logic; the check is a thin wrapper).

Ground truth: the set of data/video shard paths every v3.0 episode record
resolves to, constructed the same way model/adapters.py's _V3Resolver builds
them (data/chunk-{n:03d}/file-{n:03d}.parquet,
videos/{camera}/chunk-{n:03d}/file-{n:03d}.mp4) — read directly from
meta/episodes/chunk-*/file-*.parquet rather than through
parquet_shard_for_episode()/video_segment_for_episode(), because those
handles do not expose the resolved path back out (pq.ParquetFile has no
filename attribute once opened; VideoShardHandle does, but there is no
symmetric parquet equivalent). Reading the raw locator columns keeps orphan
detection consistent with how STRUCTURAL.PATH_TEMPLATE_RESOLVES's forward
direction is reasoned about, without inventing a second path-resolution
scheme.

This is deliberately the safest fixer in the suite: it is a *report*, not a
silent auto-fix. Deletion is irreversible and permanently out of bounds
(08_ROADMAP.md's repair philosophy) — this module never deletes anything,
anywhere, under any option. apply() without quarantine=True moves nothing
and only reports; apply() with quarantine=True moves orphans to a sidecar
directory (<output>/.trajlens-quarantine/), never removes the source, and
writes a manifest of exactly what moved and why.

ADR-004 requirements satisfied here:
  - Copy-on-write: source is never opened in write mode; output_path
    receives the full copy, with orphans relocated within it.
  - Dry-run by default: dry_run() computes the Diff with zero filesystem
    writes, regardless of the quarantine option.
  - Round-trip: tests in tests/unit/test_orphan_shard_report.py and
    tests/property/test_orphan_shard_report_properties.py verify
    report -> quarantine -> re-lint -> shard no longer flagged.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import structlog

from trajlens.errors import RepairError
from trajlens.model.canonical import CanonicalDataset
from trajlens.repair.protocol import Diff, FeatureFieldChange, RepairSummary
from trajlens.sources.paths import safe_join
from trajlens.sources.version import DatasetVersion

log = structlog.get_logger(__name__)

FIXER_ID = "REPAIR.ORPHAN_SHARD_REPORT"
CHECK_ID = "STRUCTURAL.ORPHAN_SHARD"

# Same glob task_index_repair.py/timestamp_dedrift.py use to discover data
# shards; meta/episodes/chunk-*/file-*.parquet (model/adapters.py.
# _load_v3_episodes' glob) is inlined directly in _referenced_shard_paths.
_DATA_GLOB = "data/chunk-*/file-*.parquet"
_QUARANTINE_DIR = ".trajlens-quarantine"


@dataclass(frozen=True, slots=True)
class OrphanShard:
    """One shard on disk that no episode record references.

    relative_path — dataset-root-relative path, POSIX-separated, never
                    absolute (an absolute path in a report is a redaction
                    leak — see lint --share's precedent).
    size_bytes     — the file's size on disk at detection time.
    """

    relative_path: str
    size_bytes: int


def _dataset_root(ds: CanonicalDataset) -> Path:
    return ds.stats.root


def _referenced_shard_paths(root: Path, cameras: tuple[str, ...]) -> set[Path]:
    """Return the set of absolute shard paths every episode record resolves to.

    Reads meta/episodes/chunk-*/file-*.parquet directly (rather than through
    CanonicalDataset's episode-by-episode resolver) so a single pass yields
    every referenced data and video shard path, deduplicated by construction
    since multiple episodes legitimately share one v3.0 shard file.

    Row shape and readability are not re-validated here: reaching this
    function at all requires a CanonicalDataset to already exist, and
    build_canonical_dataset() (model/adapters.py _load_v3_episodes) already
    raises DatasetFormatError for an unreadable or malformed episode shard
    before a CanonicalDataset is ever returned -- the same enforcement-
    boundary reasoning 04_CHECK_CATALOG.md's STRUCTURAL.
    REQUIRED_METADATA_PRESENT note documents for hard-required metadata
    files. The one condition that reaches here as a genuine ambiguity is an
    empty episode-shard set (zero episodes declared, so there is no ground
    truth to anchor "referenced" against at all), which raises RepairError.
    """
    episodes_root = safe_join(root, "meta", "episodes")
    shard_paths = sorted(episodes_root.glob("chunk-*/file-*.parquet"))
    if not shard_paths:
        raise RepairError(
            "no episode metadata shards found under meta/episodes/; cannot establish "
            "the referenced-shard set with confidence, refusing to guess which data/video "
            "shards are orphaned"
        )

    referenced: set[Path] = set()
    for shard_path in shard_paths:
        table = pq.read_table(shard_path)  # type: ignore[no-untyped-call]
        for row in table.to_pylist():
            data_path = safe_join(
                root,
                "data",
                f"chunk-{int(row['data/chunk_index']):03d}",
                f"file-{int(row['data/file_index']):03d}.parquet",
            )
            referenced.add(data_path)

            for camera in cameras:
                video_path = safe_join(
                    root,
                    "videos",
                    camera,
                    f"chunk-{int(row[f'videos/{camera}/chunk_index']):03d}",
                    f"file-{int(row[f'videos/{camera}/file_index']):03d}.mp4",
                )
                referenced.add(video_path)

    return referenced


def _on_disk_shard_paths(root: Path, cameras: tuple[str, ...]) -> set[Path]:
    """Enumerate actual data/video shard files on disk (the forward inventory)."""
    paths: set[Path] = set(root.glob(_DATA_GLOB))
    for camera in cameras:
        paths.update(root.glob(f"videos/{camera}/chunk-*/file-*.mp4"))
    return paths


def find_orphan_shards(ds: CanonicalDataset) -> list[OrphanShard]:
    """Return every data/video shard on disk that no episode record references.

    v3.0 only. Raises RepairError if episode metadata cannot be read with
    confidence (ambiguous/unreadable), per the fail-closed refusal rule.
    """
    if ds.version is not DatasetVersion.V3_0:
        raise RepairError(
            f"orphan shard detection only supports v3.0 datasets; got version {ds.version!r}. "
            "v2.x's one-file-per-episode naming has no metadata-vs-disk indirection to diff."
        )

    root = _dataset_root(ds)
    referenced = _referenced_shard_paths(root, ds.cameras)
    on_disk = _on_disk_shard_paths(root, ds.cameras)

    orphans: list[OrphanShard] = []
    for path in sorted(on_disk - referenced):
        rel = path.relative_to(root).as_posix()
        orphans.append(OrphanShard(relative_path=rel, size_bytes=path.stat().st_size))
    return orphans


class OrphanShardReportFixer:
    """Report (and optionally quarantine) data/video shards unreferenced by episode metadata.

    dry_run() always just reports — it never writes, regardless of the
    quarantine option, per the Protocol's dry_run contract.

    apply() without quarantine=True moves no files: it copies the source
    tree unchanged and returns immediately with a RepairSummary of zero
    changes. apply() with quarantine=True copies the source tree, then moves
    each orphan shard under <output>/.trajlens-quarantine/ (mirroring its
    original relative path) and writes a manifest recording what moved and
    why. Deletion is never implemented anywhere in this fixer.
    """

    fixer_id: str = FIXER_ID
    check_id: str = CHECK_ID

    def __init__(self, *, quarantine: bool = False) -> None:
        self.quarantine = quarantine

    def dry_run(self, ds: CanonicalDataset) -> Diff:
        """Compute the orphan-shard report without writing anything.

        Returns a Diff with one FeatureFieldChange per orphan shard found
        (feature=relative_path, field="location", old_value=size_bytes,
        new_value=size_bytes — the report records presence and size, not a
        value correction, since nothing about the shard's own content is
        being changed). Returns a no-op Diff if no orphans are found.
        """
        orphans = find_orphan_shards(ds)
        changes = tuple(
            FeatureFieldChange(
                feature=o.relative_path,
                field="location",
                old_value=o.size_bytes,
                new_value=o.size_bytes,
            )
            for o in orphans
        )
        diff = Diff(changes=changes, check_id=CHECK_ID, fixer_id=FIXER_ID)
        if diff.is_noop:
            log.info(
                "orphan_shard_report.dry_run.noop",
                reason="every data/video shard on disk is referenced by episode metadata",
            )
        else:
            log.info(
                "orphan_shard_report.dry_run.orphans_found",
                num_orphans=len(changes),
                paths=[c.feature for c in changes],
            )
        return diff

    def apply(self, ds: CanonicalDataset, output_path: Path) -> RepairSummary:
        """Copy the source tree to *output_path*; quarantine orphans if requested.

        Without quarantine=True: copies the source tree unchanged and
        returns a zero-change RepairSummary. No files are moved.

        With quarantine=True: copies the source tree, then moves each orphan
        shard from its original location to
        <output>/.trajlens-quarantine/<relative_path>, and writes
        <output>/.trajlens-quarantine/quarantine_manifest.json listing each
        moved file (relative path, size, reason). *output_path* must not be
        the source dataset root.
        """
        source_root = _dataset_root(ds)
        if output_path.resolve() == source_root.resolve():
            raise RepairError(
                "output_path must not be the source dataset root (ADR-004 "
                f"copy-on-write). Got: {output_path}"
            )

        orphans = find_orphan_shards(ds)

        log.info(
            "orphan_shard_report.apply.start",
            source=str(source_root),
            output=str(output_path),
            num_orphans=len(orphans),
            quarantine=self.quarantine,
        )

        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(source_root, output_path)

        if not self.quarantine or not orphans:
            if self.quarantine:
                log.info("orphan_shard_report.apply.noop", output=str(output_path))
            else:
                log.info(
                    "orphan_shard_report.apply.report_only",
                    output=str(output_path),
                    num_orphans=len(orphans),
                    message=(
                        f"No files moved. Re-run with --quarantine to move orphan shards to "
                        f"{output_path}/{_QUARANTINE_DIR}/."
                    ),
                )
            return RepairSummary(
                output_path=output_path,
                changes_written=0,
                frames_corrected=0,
            )

        _quarantine_orphans(output_path, orphans)

        log.info(
            "orphan_shard_report.apply.done",
            output=str(output_path),
            num_quarantined=len(orphans),
        )
        return RepairSummary(
            output_path=output_path,
            changes_written=len(orphans),
            frames_corrected=0,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _quarantine_orphans(output_root: Path, orphans: list[OrphanShard]) -> None:
    """Move each orphan shard under output_root into the quarantine sidecar dir.

    Never deletes: shutil.move() relocates the file, it is not removed from
    the dataset tree, only relocated within it. The manifest records exactly
    what moved and why, per the fixer's report-first design.
    """
    quarantine_root = safe_join(output_root, _QUARANTINE_DIR)
    quarantine_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    for orphan in orphans:
        src = safe_join(output_root, *Path(orphan.relative_path).parts)
        dst = safe_join(quarantine_root, *Path(orphan.relative_path).parts)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        manifest.append(
            {
                "relative_path": orphan.relative_path,
                "size_bytes": orphan.size_bytes,
                "reason": "unreferenced by episode metadata",
            }
        )
        log.debug("orphan_shard_report.shard_quarantined", path=orphan.relative_path)

    manifest_path = quarantine_root / "quarantine_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
