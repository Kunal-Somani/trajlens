"""Property tests for EpisodeReindexFixer (05_ENGINEERING_STANDARDS.md §5).

Invariant: "repair then full re-lint clears STRUCTURAL.METADATA_DATA_AGREEMENT,
and repaired boundaries select the semantically correct frames."

Matches the style of test_stats_repair_properties.py:
  - @given over corruption-shape parameters
  - @settings with deadline=None (disk I/O is inherently variable)
  - Full-engine set-diff comparison pattern
  - An additional semantic-correctness assertion per example, since a fixer
    that merely makes counts agree without assigning the right frames would
    recreate the #2401 corruption it claims to repair (06_SECURITY_AND_THREAT_MODEL.md T9).
"""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.fixtures.builders import build_v3_metadata_data_disagreement
from trajlens.checks import CheckEngine, registry
from trajlens.checks.protocol import CheckContext, Severity
from trajlens.checks.structural import METADATA_DATA_AGREEMENT
from trajlens.model import build_canonical_dataset
from trajlens.repair.episode_reindex import CHECK_ID, EpisodeReindexFixer
from trajlens.sources.loader import SourceLoader

CTX = CheckContext(deep=False)

NUM_EPISODES = st.integers(min_value=1, max_value=6)


@given(num_episodes=NUM_EPISODES)
@settings(max_examples=15, deadline=None)
def test_repair_then_relint_clears_metadata_data_agreement(
    tmp_path_factory: pytest.TempPathFactory,
    num_episodes: int,
) -> None:
    """Invariant: repair -> full re-lint clears STRUCTURAL.METADATA_DATA_AGREEMENT,
    adds no new findings, and every episode's repaired boundary selects only its
    own rows.
    """
    root = tmp_path_factory.mktemp("prop-episodereindex")
    source = root / "source"
    output = root / "repaired"

    build_v3_metadata_data_disagreement(source, num_episodes=num_episodes)

    handle = SourceLoader().resolve(str(source))
    ds_source = build_canonical_dataset(handle)

    pre_result = METADATA_DATA_AGREEMENT.run(ds_source, CTX)
    if pre_result.severity is not Severity.FAIL:
        # The fixture is a no-op for this shape (e.g. zero episodes) -- hypothesis
        # will try other values.
        return

    engine = CheckEngine(registry)

    pre_results = engine.run(ds_source, CTX).results
    pre_fail_ids = {
        r.check_id for r in pre_results if r.severity >= Severity.WARN and r.check_id != CHECK_ID
    }

    fixer = EpisodeReindexFixer()
    fixer.apply(ds_source, output)

    handle_fixed = SourceLoader().resolve(str(output))
    ds_fixed = build_canonical_dataset(handle_fixed)
    post_results = engine.run(ds_fixed, CTX).results

    agreement_post = next((r for r in post_results if r.check_id == CHECK_ID), None)
    assert agreement_post is None or agreement_post.severity < Severity.WARN, (
        f"STRUCTURAL.METADATA_DATA_AGREEMENT must be INFO after repair "
        f"(num_episodes={num_episodes}). Got severity={agreement_post.severity!r}: "
        f"{agreement_post.message}"
    )

    post_fail_ids = {
        r.check_id for r in post_results if r.severity >= Severity.WARN and r.check_id != CHECK_ID
    }
    new_findings = post_fail_ids - pre_fail_ids
    assert not new_findings, (
        f"repair introduced new WARN/FAIL findings not present in source: {new_findings} "
        f"(num_episodes={num_episodes})"
    )

    # Semantic correctness: every repaired boundary must select ONLY that
    # episode's own rows -- the guard against recreating #2401's silent
    # wrong-episode-assignment while merely making counts add up.
    data_path = output / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_path)
    ep_col = table.column("episode_index").to_pylist()
    idx_col = table.column("index").to_pylist()
    idx_to_ep = dict(zip(idx_col, ep_col, strict=True))

    for episode in ds_fixed:
        selected = range(episode.dataset_from_index, episode.dataset_to_index)
        assert len(list(selected)) == episode.length, (
            f"episode {episode.episode_index}: boundary span != declared length "
            f"(num_episodes={num_episodes})"
        )
        for global_idx in selected:
            assert idx_to_ep[global_idx] == episode.episode_index, (
                f"episode {episode.episode_index}'s repaired boundary selects row "
                f"index={global_idx} belonging to episode {idx_to_ep[global_idx]} instead "
                f"(num_episodes={num_episodes})"
            )
