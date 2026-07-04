"""Property tests for StatsRecomputeFixer (05_ENGINEERING_STANDARDS.md §5).

Invariant: "repair then full re-lint clears STATISTICAL.STATS_MATCH_DATA."

Matches the style of test_repair_properties.py:
  - @given over stats corruption parameters
  - @settings with deadline=None (disk I/O is inherently variable)
  - Full-engine set-diff comparison pattern (same as fixed timestamp_dedrift tests)
"""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.fixtures.builders import build_v3_with_correct_stats
from trajlens.checks import CheckEngine, registry
from trajlens.checks.protocol import CheckContext, Severity
from trajlens.checks.statistical import STATS_MATCH_DATA
from trajlens.model import build_canonical_dataset
from trajlens.repair.stats_recompute import CHECK_ID, StatsRecomputeFixer
from trajlens.sources.loader import SourceLoader

CTX = CheckContext(deep=False)

# Strategy: corrupt the stored mean by adding a large offset (well beyond rtol=1e-4).
MEAN_OFFSET = st.floats(min_value=0.5, max_value=5.0, allow_nan=False)


@given(mean_offset=MEAN_OFFSET)
@settings(max_examples=10, deadline=None)
def test_repair_then_relint_clears_stats_match_data(
    tmp_path_factory: pytest.TempPathFactory,
    mean_offset: float,
) -> None:
    """Invariant: repair → full re-lint clears STATISTICAL.STATS_MATCH_DATA.

    Uses the full default check suite so that any side-effect on adjacent
    checks is caught — a single-check run would miss those regressions.
    """
    root = tmp_path_factory.mktemp("prop-statsrecompute")
    source = root / "source"
    output = root / "repaired"

    build_v3_with_correct_stats(source)

    # Corrupt the stored mean by a large offset to guarantee the check fires.
    stats_path = source / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    for feat in stats:
        if isinstance(stats[feat], dict) and "mean" in stats[feat]:
            stats[feat]["mean"] = float(stats[feat]["mean"]) + mean_offset
    stats_path.write_text(json.dumps(stats))

    handle = SourceLoader().resolve(str(source))
    ds_source = build_canonical_dataset(handle)

    # Only run the property when the fixture actually triggers the check.
    pre_result = STATS_MATCH_DATA.run(ds_source, CTX)
    if pre_result.severity is not Severity.FAIL:
        return

    engine = CheckEngine(registry)

    pre_results = engine.run(ds_source, CTX)
    pre_fail_ids = {
        r.check_id for r in pre_results if r.severity >= Severity.WARN and r.check_id != CHECK_ID
    }

    fixer = StatsRecomputeFixer()
    fixer.apply(ds_source, output)

    handle_fixed = SourceLoader().resolve(str(output))
    ds_fixed = build_canonical_dataset(handle_fixed)
    post_results = engine.run(ds_fixed, CTX)

    stats_post = next((r for r in post_results if r.check_id == CHECK_ID), None)
    assert stats_post is None or stats_post.severity < Severity.WARN, (
        f"STATISTICAL.STATS_MATCH_DATA must be INFO after repair "
        f"(mean_offset={mean_offset}). "
        f"Got severity={stats_post.severity!r}: {stats_post.message}"
    )

    post_fail_ids = {
        r.check_id for r in post_results if r.severity >= Severity.WARN and r.check_id != CHECK_ID
    }
    new_findings = post_fail_ids - pre_fail_ids
    assert not new_findings, (
        f"repair introduced new WARN/FAIL findings not present in source: {new_findings} "
        f"(mean_offset={mean_offset})"
    )
