"""Property tests for VideoMetadataSyncFixer (05_ENGINEERING_STANDARDS.md §5).

Invariant: "repair converges" -- after apply(), the fixer's own dry_run()
against the repaired output is always a noop, and the repair never introduces
a new WARN/FAIL finding from the rest of the check suite. Matches the style
of tests/property/test_task_index_repair_properties.py.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from tests.fixtures.builders import build_v3_video_fps_mismatch
from trajlens.checks import CheckEngine, registry
from trajlens.checks.protocol import CheckContext, Severity
from trajlens.model import build_canonical_dataset
from trajlens.repair.video_metadata_sync import VideoMetadataSyncFixer
from trajlens.sources.loader import SourceLoader

CTX = CheckContext(deep=False)

DECLARED_FPS = st.integers(min_value=5, max_value=60)
CONTAINER_FPS = st.integers(min_value=5, max_value=60)


@given(declared_fps=DECLARED_FPS, container_fps=CONTAINER_FPS)
@settings(max_examples=15, deadline=None)
def test_repair_converges_and_introduces_no_new_findings(
    tmp_path_factory,  # type: ignore[no-untyped-def]
    declared_fps: int,
    container_fps: int,
) -> None:
    """Invariant: repair -> fixer's own dry_run() is a noop, no new WARN/FAIL elsewhere."""
    root = tmp_path_factory.mktemp("prop-video-metadata-sync")
    source = root / "source"
    output = root / "repaired"

    build_v3_video_fps_mismatch(source, declared_fps=declared_fps, container_fps=container_fps)

    handle = SourceLoader().resolve(str(source))
    ds_source = build_canonical_dataset(handle)

    fixer = VideoMetadataSyncFixer()
    pre_diff = fixer.dry_run(ds_source)
    if declared_fps == container_fps:
        assert pre_diff.is_noop
        return

    engine = CheckEngine(registry)
    pre_results = engine.run(ds_source, CTX)
    pre_fail_ids = {r.check_id for r in pre_results if r.severity >= Severity.WARN}

    fixer.apply(ds_source, output)

    handle_fixed = SourceLoader().resolve(str(output))
    ds_fixed = build_canonical_dataset(handle_fixed)

    post_diff = fixer.dry_run(ds_fixed)
    assert post_diff.is_noop, (
        f"fixer must converge to a noop diff after apply() "
        f"(declared_fps={declared_fps}, container_fps={container_fps})"
    )

    post_results = engine.run(ds_fixed, CTX)
    post_fail_ids = {r.check_id for r in post_results if r.severity >= Severity.WARN}
    new_findings = post_fail_ids - pre_fail_ids
    assert not new_findings, (
        f"repair introduced new WARN/FAIL findings not present in source: {new_findings} "
        f"(declared_fps={declared_fps}, container_fps={container_fps})"
    )
