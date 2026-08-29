"""Property tests for TaskIndexRepairFixer (05_ENGINEERING_STANDARDS.md §5).

Invariant: "repair then re-lint clears SEMANTIC.TASK_INTEGRITY's dangling
task_index findings." Matches the style of tests/property/test_repair_properties.py.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.fixtures.builders import build_v3_dataset
from trajlens.checks import CheckEngine, registry
from trajlens.checks.protocol import CheckContext, Severity
from trajlens.checks.semantic import TASK_INTEGRITY
from trajlens.model import build_canonical_dataset
from trajlens.repair.task_index_repair import CHECK_ID, TaskIndexRepairFixer
from trajlens.sources.loader import SourceLoader

CTX = CheckContext(deep=False)

NUM_EPISODES = st.integers(min_value=1, max_value=4)
# Dangling task_index far outside the single defined task_index=0, so the
# nearest-valid-task heuristic is always unambiguous (only one candidate).
DANGLING_TASK_INDEX = st.integers(min_value=1, max_value=500)
NUM_DANGLING_FRAMES = st.integers(min_value=1, max_value=3)


def _make_dangling(root, num_episodes: int, dangling_value: int, num_dangling_frames: int) -> None:  # type: ignore[no-untyped-def]
    build_v3_dataset(root, num_episodes=num_episodes)
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    ti_col = old.column("task_index").to_pylist()
    n = min(num_dangling_frames, len(ti_col))
    for i in range(n):
        ti_col[i] = dangling_value
    new = old.set_column(
        old.schema.get_field_index("task_index"), "task_index", pa.array(ti_col, type=pa.int64())
    )
    pq.write_table(new, data_path)


@given(
    num_episodes=NUM_EPISODES,
    dangling_value=DANGLING_TASK_INDEX,
    num_dangling_frames=NUM_DANGLING_FRAMES,
)
@settings(max_examples=15, deadline=None)
def test_repair_then_relint_clears_task_integrity(
    tmp_path_factory: pytest.TempPathFactory,
    num_episodes: int,
    dangling_value: int,
    num_dangling_frames: int,
) -> None:
    """Invariant: repair -> full re-lint clears SEMANTIC.TASK_INTEGRITY, adds no new findings."""
    root = tmp_path_factory.mktemp("prop-task-index-repair")
    source = root / "source"
    output = root / "repaired"

    _make_dangling(source, num_episodes, dangling_value, num_dangling_frames)

    handle = SourceLoader().resolve(str(source))
    ds_source = build_canonical_dataset(handle)

    pre_result = TASK_INTEGRITY.run(ds_source, CTX)
    if pre_result.severity is not Severity.FAIL:
        return  # dangling_value happened to collide with a defined index — skip.

    engine = CheckEngine(registry)
    pre_results = engine.run(ds_source, CTX).results
    pre_fail_ids = {
        r.check_id for r in pre_results if r.severity >= Severity.WARN and r.check_id != CHECK_ID
    }

    fixer = TaskIndexRepairFixer()
    fixer.apply(ds_source, output)

    handle_fixed = SourceLoader().resolve(str(output))
    ds_fixed = build_canonical_dataset(handle_fixed)
    post_results = engine.run(ds_fixed, CTX).results

    task_post = next((r for r in post_results if r.check_id == CHECK_ID), None)
    assert task_post is None or task_post.severity < Severity.WARN, (
        f"SEMANTIC.TASK_INTEGRITY must be INFO after repair "
        f"(num_episodes={num_episodes}, dangling_value={dangling_value}). "
        f"Got: {task_post}"
    )

    post_fail_ids = {
        r.check_id for r in post_results if r.severity >= Severity.WARN and r.check_id != CHECK_ID
    }
    new_findings = post_fail_ids - pre_fail_ids
    assert not new_findings, (
        f"repair introduced new WARN/FAIL findings not present in source: {new_findings} "
        f"(num_episodes={num_episodes}, dangling_value={dangling_value})"
    )
