"""Property tests for the canonical dataset model (05_ENGINEERING_STANDARDS.md §5).

tmp_path is function-scoped, which hypothesis's health checks reject for
@given-decorated tests, so each example gets its own directory via
tmp_path_factory instead.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.fixtures.builders import FRAMES_PER_EPISODE, build_v2_dataset, build_v3_dataset
from trajlens.model import build_canonical_dataset
from trajlens.sources.loader import SourceLoader

CODEBASE_VERSIONS = st.sampled_from(["v2.0", "v2.1", "v3.0"])
NUM_EPISODES = st.integers(min_value=0, max_value=6)


@given(codebase_version=CODEBASE_VERSIONS, num_episodes=NUM_EPISODES)
@settings(max_examples=30, deadline=None)
def test_episode_count_matches_what_the_dataset_declares(
    tmp_path_factory: pytest.TempPathFactory, codebase_version: str, num_episodes: int
) -> None:
    root = tmp_path_factory.mktemp("canonical-model-property")
    if codebase_version == "v3.0":
        build_v3_dataset(root, num_episodes=num_episodes)
    else:
        build_v2_dataset(root, codebase_version=codebase_version, num_episodes=num_episodes)

    handle = SourceLoader().resolve(str(root))
    ds = build_canonical_dataset(handle)

    assert ds.num_episodes == num_episodes
    assert len(ds) == num_episodes
    assert [ep.episode_index for ep in ds] == list(range(num_episodes))


@given(codebase_version=CODEBASE_VERSIONS, num_episodes=NUM_EPISODES)
@settings(max_examples=30, deadline=None)
def test_episode_frame_offsets_are_contiguous(
    tmp_path_factory: pytest.TempPathFactory, codebase_version: str, num_episodes: int
) -> None:
    root = tmp_path_factory.mktemp("canonical-model-property")
    if codebase_version == "v3.0":
        build_v3_dataset(root, num_episodes=num_episodes)
    else:
        build_v2_dataset(root, codebase_version=codebase_version, num_episodes=num_episodes)

    handle = SourceLoader().resolve(str(root))
    ds = build_canonical_dataset(handle)

    episodes = list(ds)
    expected_from = 0
    for ep in episodes:
        assert ep.dataset_from_index == expected_from
        assert ep.dataset_to_index == expected_from + FRAMES_PER_EPISODE
        expected_from = ep.dataset_to_index
    assert expected_from == num_episodes * FRAMES_PER_EPISODE
