"""Property tests for TimestampDedriftFixer (05_ENGINEERING_STANDARDS.md §5).

Invariant: "repair then re-lint clears KNOWNBUG.TIMESTAMP_DRIFT."

Matches the style of tests/property/test_check_properties.py:
  - @given over dataset shape parameters
  - @settings with deadline=None (disk I/O is inherently variable)
  - Structured assertion with a descriptive failure message
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.fixtures.builders import build_v3_timestamp_drift
from trajlens.checks import CheckEngine, registry
from trajlens.checks.protocol import CheckContext, Severity
from trajlens.checks.temporal import TIMESTAMP_DRIFT
from trajlens.model import build_canonical_dataset
from trajlens.repair.timestamp_dedrift import CHECK_ID, TimestampDedriftFixer
from trajlens.sources.loader import SourceLoader

CTX = CheckContext(deep=False)

# Strategy bounds: small datasets so tests stay fast; enough drift to fire.
NUM_EPISODES = st.integers(min_value=1, max_value=6)
DRIFT_PER_FRAME = st.floats(min_value=5e-5, max_value=2e-4, allow_nan=False)


@given(num_episodes=NUM_EPISODES, drift_per_frame=DRIFT_PER_FRAME)
@settings(max_examples=15, deadline=None)
def test_repair_then_relint_clears_timestamp_drift(
    tmp_path_factory: pytest.TempPathFactory,
    num_episodes: int,
    drift_per_frame: float,
) -> None:
    """Invariant: repair → full re-lint clears KNOWNBUG.TIMESTAMP_DRIFT and adds no new findings.

    Uses the full default check suite so that any side-effect on adjacent
    checks (e.g. TEMPORAL.TIMESTAMP_SPACING, STRUCTURAL.METADATA_DATA_AGREEMENT)
    is caught — a single-check run would miss those regressions.
    """
    root = tmp_path_factory.mktemp("prop-dedrift")
    source = root / "source"
    output = root / "repaired"

    build_v3_timestamp_drift(source, num_episodes=num_episodes, drift_per_frame=drift_per_frame)

    handle = SourceLoader().resolve(str(source))
    ds_source = build_canonical_dataset(handle)

    # Only run the property when the fixture actually triggers the check.
    pre_result = TIMESTAMP_DRIFT.run(ds_source, CTX)
    if pre_result.severity is not Severity.FAIL:
        # The drift was too small to fire — hypothesis will try other values.
        return

    engine = CheckEngine(registry)

    # Capture pre-repair WARN/FAIL check-ids (excluding the drift finding itself).
    pre_results = engine.run(ds_source, CTX).results
    pre_fail_ids = {
        r.check_id for r in pre_results if r.severity >= Severity.WARN and r.check_id != CHECK_ID
    }

    fixer = TimestampDedriftFixer()
    fixer.apply(ds_source, output)

    handle_fixed = SourceLoader().resolve(str(output))
    ds_fixed = build_canonical_dataset(handle_fixed)
    post_results = engine.run(ds_fixed, CTX).results

    # The drift finding must be gone (INFO or better).
    drift_post = next((r for r in post_results if r.check_id == CHECK_ID), None)
    assert drift_post is None or drift_post.severity < Severity.WARN, (
        f"KNOWNBUG.TIMESTAMP_DRIFT must be INFO after repair "
        f"(num_episodes={num_episodes}, drift_per_frame={drift_per_frame}). "
        f"Got severity={drift_post.severity!r}: {drift_post.message}"
    )

    # No new WARN/FAIL findings that weren't already present before repair.
    post_fail_ids = {
        r.check_id for r in post_results if r.severity >= Severity.WARN and r.check_id != CHECK_ID
    }
    new_findings = post_fail_ids - pre_fail_ids
    assert not new_findings, (
        f"repair introduced new WARN/FAIL findings not present in source: {new_findings} "
        f"(num_episodes={num_episodes}, drift_per_frame={drift_per_frame})"
    )
