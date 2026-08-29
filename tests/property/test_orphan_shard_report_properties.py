"""Property tests for OrphanShardReportFixer (05_ENGINEERING_STANDARDS.md §5).

Invariant 1: dry_run() identifies exactly the planted orphan set -- no more,
no less -- across random combinations of extra unreferenced data/video
shard file indices.

Invariant 2: quarantine -> re-lint clears STRUCTURAL.ORPHAN_SHARD and
introduces no new WARN/FAIL finding. Matches the style of
tests/property/test_video_metadata_sync_properties.py.
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from tests.fixtures.builders import build_v3_dataset
from trajlens.checks import CheckEngine, registry
from trajlens.checks.protocol import CheckContext, Severity
from trajlens.model import build_canonical_dataset
from trajlens.repair.orphan_shard_report import CHECK_ID, OrphanShardReportFixer, find_orphan_shards
from trajlens.sources.loader import SourceLoader

CTX = CheckContext(deep=False)

# File indices 1..9 for extra, unreferenced shards -- 0 is always the
# referenced shard build_v3_dataset writes, so these never collide with it.
ORPHAN_DATA_FILE_INDICES = st.sets(st.integers(min_value=1, max_value=9), min_size=0, max_size=4)
ORPHAN_VIDEO_FILE_INDICES = st.sets(st.integers(min_value=1, max_value=9), min_size=0, max_size=4)


def _plant_orphans(
    root: Path, data_indices: set[int], video_indices: set[int], *, camera: str = "top"
) -> set[str]:
    build_v3_dataset(root, num_episodes=2, camera=camera)

    expected: set[str] = set()
    data_dir = root / "data" / "chunk-000"
    for idx in data_indices:
        path = data_dir / f"file-{idx:03d}.parquet"
        path.write_bytes(b"\x00")
        expected.add(str(path.relative_to(root)))

    video_dir = root / "videos" / camera / "chunk-000"
    for idx in video_indices:
        path = video_dir / f"file-{idx:03d}.mp4"
        path.write_bytes(b"\x00")
        expected.add(str(path.relative_to(root)))

    return expected


@given(data_indices=ORPHAN_DATA_FILE_INDICES, video_indices=ORPHAN_VIDEO_FILE_INDICES)
@settings(max_examples=20, deadline=None)
def test_dry_run_identifies_exactly_the_planted_orphan_set(
    tmp_path_factory,  # type: ignore[no-untyped-def]
    data_indices: set[int],
    video_indices: set[int],
) -> None:
    root = tmp_path_factory.mktemp("prop-orphan-shard-report")
    source = root / "source"
    expected = _plant_orphans(source, data_indices, video_indices)

    handle = SourceLoader().resolve(str(source))
    ds = build_canonical_dataset(handle)

    orphans = find_orphan_shards(ds)
    found = {o.relative_path for o in orphans}

    assert found == expected, (
        f"orphan set mismatch: expected {expected}, found {found} "
        f"(data_indices={data_indices}, video_indices={video_indices})"
    )


@given(data_indices=ORPHAN_DATA_FILE_INDICES, video_indices=ORPHAN_VIDEO_FILE_INDICES)
@settings(max_examples=15, deadline=None)
def test_quarantine_converges_and_introduces_no_new_findings(
    tmp_path_factory,  # type: ignore[no-untyped-def]
    data_indices: set[int],
    video_indices: set[int],
) -> None:
    root = tmp_path_factory.mktemp("prop-orphan-shard-report-quarantine")
    source = root / "source"
    output = root / "repaired"
    expected = _plant_orphans(source, data_indices, video_indices)

    handle = SourceLoader().resolve(str(source))
    ds_source = build_canonical_dataset(handle)

    engine = CheckEngine(registry)
    pre_results = engine.run(ds_source, CTX).results
    pre_ids = {r.check_id for r in pre_results if r.severity >= Severity.WARN}

    if not expected:
        assert CHECK_ID not in pre_ids
        return

    assert CHECK_ID in pre_ids

    fixer = OrphanShardReportFixer(quarantine=True)
    fixer.apply(ds_source, output)

    handle_fixed = SourceLoader().resolve(str(output))
    ds_fixed = build_canonical_dataset(handle_fixed)

    post_diff = fixer.dry_run(ds_fixed)
    assert post_diff.is_noop, (
        f"fixer must converge to a noop diff after quarantine "
        f"(data_indices={data_indices}, video_indices={video_indices})"
    )

    post_results = engine.run(ds_fixed, CTX).results
    post_ids = {r.check_id for r in post_results if r.severity >= Severity.WARN}
    assert CHECK_ID not in post_ids
    new_findings = post_ids - pre_ids
    assert not new_findings, (
        f"quarantine introduced new WARN/FAIL findings not present in source: {new_findings} "
        f"(data_indices={data_indices}, video_indices={video_indices})"
    )
