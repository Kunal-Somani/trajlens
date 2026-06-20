"""Property tests for M4 checks (05_ENGINEERING_STANDARDS.md §5).

Invariant expressed: "a freshly-written clean dataset passes all checks."
This is the explicit property test named in the spec.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.fixtures.builders import build_v2_dataset, build_v3_dataset
from trajlens.checks import CheckContext, CheckEngine, Severity, registry
from trajlens.model import build_canonical_dataset
from trajlens.sources.loader import SourceLoader

CODEBASE_VERSIONS = st.sampled_from(["v2.0", "v2.1", "v3.0"])
NUM_EPISODES = st.integers(min_value=0, max_value=5)


@given(codebase_version=CODEBASE_VERSIONS, num_episodes=NUM_EPISODES)
@settings(max_examples=20, deadline=None)
def test_clean_dataset_passes_all_non_video_checks(
    tmp_path_factory: pytest.TempPathFactory,
    codebase_version: str,
    num_episodes: int,
) -> None:
    """A freshly-written clean dataset must produce no FAIL or ERROR results.

    VIDEO.DECODABLE_SPOTCHECK may emit FAIL on the stub b'\\x00' video bytes
    that fixture builders produce (not real MP4), so we skip it from the clean
    invariant — the fixture's purpose is shape-correctness, not real-video
    decodability.  All non-video checks must pass cleanly.
    """
    root = tmp_path_factory.mktemp("clean-property")
    if codebase_version == "v3.0":
        build_v3_dataset(root, num_episodes=num_episodes)
    else:
        build_v2_dataset(root, codebase_version=codebase_version, num_episodes=num_episodes)

    handle = SourceLoader().resolve(str(root))
    ds = build_canonical_dataset(handle)
    engine = CheckEngine(registry)
    ctx = CheckContext(deep=False)

    results = engine.run(ds, ctx)
    non_video_results = [r for r in results if not r.check_id.startswith("VIDEO.")]

    worst = max((r.severity for r in non_video_results), default=Severity.INFO)
    fail_results = [r for r in non_video_results if r.severity >= Severity.FAIL]
    assert not fail_results, (
        f"Clean {codebase_version} dataset with {num_episodes} episodes "
        f"triggered FAIL/ERROR: {[(r.check_id, r.message) for r in fail_results]}"
    )
    assert worst <= Severity.WARN, f"Unexpected severity {worst} on clean dataset"
