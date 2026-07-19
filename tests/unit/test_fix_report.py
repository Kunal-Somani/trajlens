"""Unit tests for report/fix_report.py (`trajlens fix` terminal/JSON renderers).

Constructs FixPlan/FixerOutcome/Diff directly rather than going through the
CLI or orchestrator, so every render branch (per-change-type summaries,
noop vs. applicable, truncation past 5 changes) is reachable without needing
a fixture dataset that happens to trigger it.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from trajlens.repair.orchestrator import FixerOutcome, FixPlan
from trajlens.repair.protocol import (
    BoundaryChange,
    Diff,
    FeatureFieldChange,
    FrameChange,
    RepairSummary,
    StatChange,
)
from trajlens.report.fix_report import render_fix_json, render_fix_json_error, render_fix_terminal


def _render(plan: FixPlan) -> str:
    # width=200: wide enough that no assertion string gets split across a
    # wrapped line (rich's default 80-col wrap can otherwise fragment a
    # short phrase like "no changes needed" across two lines).
    buf = Console(file=None, record=True, highlight=False, markup=False, width=200)
    render_fix_terminal("test/ref", plan, console=buf)
    return buf.export_text()


class TestChangeSummaryBranches:
    """Each Diff.changes element type has its own _change_summary branch."""

    def test_frame_change_summary(self) -> None:
        diff = Diff(
            changes=(
                FrameChange(
                    episode_index=0,
                    frame_index=1,
                    shard_path="data/chunk-000/file-000.parquet",
                    column="task_index",
                    old_value=2,
                    new_value=0,
                ),
            ),
            check_id="SEMANTIC.TASK_INTEGRITY",
            fixer_id="REPAIR.TASK_INDEX_REPAIR",
        )
        plan = FixPlan(
            applicable=True,
            outcomes=(
                FixerOutcome(
                    fixer_id="REPAIR.TASK_INDEX_REPAIR",
                    check_id="SEMANTIC.TASK_INTEGRITY",
                    diff=diff,
                    summary=None,
                ),
            ),
            output_path=None,
            applied=False,
        )
        out = _render(plan)
        assert "episode 0 frame 1" in out
        assert "task_index" in out
        assert "2 -> 0" in out

    def test_stat_change_summary(self) -> None:
        diff = Diff(
            changes=(
                StatChange(feature="timestamp", stat_key="mean", old_value=1.5, new_value=1.0),
            ),
            check_id="STATISTICAL.STATS_MATCH_DATA",
            fixer_id="REPAIR.STATS_RECOMPUTE",
        )
        plan = FixPlan(
            applicable=True,
            outcomes=(
                FixerOutcome(
                    fixer_id="REPAIR.STATS_RECOMPUTE",
                    check_id="STATISTICAL.STATS_MATCH_DATA",
                    diff=diff,
                    summary=None,
                ),
            ),
            output_path=None,
            applied=False,
        )
        out = _render(plan)
        assert "timestamp.mean" in out
        assert "1.5 -> 1.0" in out

    def test_feature_field_change_summary(self) -> None:
        diff = Diff(
            changes=(
                FeatureFieldChange(feature="fps", field="fps", old_value=30.0, new_value=24.0),
            ),
            check_id="VIDEO.RESOLUTION_FPS_MATCH",
            fixer_id="REPAIR.VIDEO_METADATA_SYNC",
        )
        plan = FixPlan(
            applicable=True,
            outcomes=(
                FixerOutcome(
                    fixer_id="REPAIR.VIDEO_METADATA_SYNC",
                    check_id="VIDEO.RESOLUTION_FPS_MATCH",
                    diff=diff,
                    summary=None,
                ),
            ),
            output_path=None,
            applied=False,
        )
        out = _render(plan)
        assert "fps.fps" in out
        assert "30.0 -> 24.0" in out

    def test_boundary_change_summary(self) -> None:
        diff = Diff(
            changes=(
                BoundaryChange(
                    episode_index=2, field="dataset_from_index", old_value=8, new_value=4
                ),
            ),
            check_id="STRUCTURAL.METADATA_DATA_AGREEMENT",
            fixer_id="REPAIR.EPISODE_REINDEX",
        )
        plan = FixPlan(
            applicable=True,
            outcomes=(
                FixerOutcome(
                    fixer_id="REPAIR.EPISODE_REINDEX",
                    check_id="STRUCTURAL.METADATA_DATA_AGREEMENT",
                    diff=diff,
                    summary=None,
                ),
            ),
            output_path=None,
            applied=False,
        )
        out = _render(plan)
        assert "episode 2 (dataset_from_index)" in out
        assert "8 -> 4" in out


class TestTerminalRendererBranches:
    def test_no_outcomes_at_all_shows_nothing_to_fix(self) -> None:
        plan = FixPlan(applicable=False, outcomes=(), output_path=None, applied=False)
        out = _render(plan)
        assert "No applicable fixers" in out

    def test_outcomes_but_all_noop_shows_already_clean(self) -> None:
        diff = Diff(
            changes=(), check_id="SEMANTIC.TASK_INTEGRITY", fixer_id="REPAIR.TASK_INDEX_REPAIR"
        )
        plan = FixPlan(
            applicable=False,
            outcomes=(
                FixerOutcome(
                    fixer_id="REPAIR.TASK_INDEX_REPAIR",
                    check_id="SEMANTIC.TASK_INTEGRITY",
                    diff=diff,
                    summary=None,
                ),
            ),
            output_path=None,
            applied=False,
        )
        out = _render(plan)
        assert "already clean" in out
        assert "Nothing to fix" in out

    def test_single_noop_outcome_among_others_prints_ok_line(self) -> None:
        """A mixed plan (one noop fixer, one with changes) must print the
        noop fixer's "OK ... no changes needed" line, not skip it."""
        noop_diff = Diff(
            changes=(), check_id="SEMANTIC.TASK_INTEGRITY", fixer_id="REPAIR.TASK_INDEX_REPAIR"
        )
        change_diff = Diff(
            changes=(
                StatChange(feature="timestamp", stat_key="mean", old_value=1.5, new_value=1.0),
            ),
            check_id="STATISTICAL.STATS_MATCH_DATA",
            fixer_id="REPAIR.STATS_RECOMPUTE",
        )
        plan = FixPlan(
            applicable=True,
            outcomes=(
                FixerOutcome(
                    fixer_id="REPAIR.TASK_INDEX_REPAIR",
                    check_id="SEMANTIC.TASK_INTEGRITY",
                    diff=noop_diff,
                    summary=None,
                ),
                FixerOutcome(
                    fixer_id="REPAIR.STATS_RECOMPUTE",
                    check_id="STATISTICAL.STATS_MATCH_DATA",
                    diff=change_diff,
                    summary=None,
                ),
            ),
            output_path=None,
            applied=False,
        )
        out = _render(plan)
        assert "OK" in out
        assert "REPAIR.TASK_INDEX_REPAIR" in out
        assert "no changes needed" in out

    def test_more_than_five_changes_shows_truncation_message(self) -> None:
        changes = tuple(
            StatChange(feature=f"feature_{i}", stat_key="mean", old_value=1.0, new_value=2.0)
            for i in range(8)
        )
        diff = Diff(
            changes=changes,
            check_id="STATISTICAL.STATS_MATCH_DATA",
            fixer_id="REPAIR.STATS_RECOMPUTE",
        )
        plan = FixPlan(
            applicable=True,
            outcomes=(
                FixerOutcome(
                    fixer_id="REPAIR.STATS_RECOMPUTE",
                    check_id="STATISTICAL.STATS_MATCH_DATA",
                    diff=diff,
                    summary=None,
                ),
            ),
            output_path=None,
            applied=False,
        )
        out = _render(plan)
        assert "... and 3 more" in out

    def test_applied_plan_shows_output_path(self, tmp_path: Path) -> None:
        out_path = tmp_path / "out"
        diff = Diff(
            changes=(
                StatChange(feature="timestamp", stat_key="mean", old_value=1.5, new_value=1.0),
            ),
            check_id="STATISTICAL.STATS_MATCH_DATA",
            fixer_id="REPAIR.STATS_RECOMPUTE",
        )
        summary = RepairSummary(output_path=out_path, changes_written=1, frames_corrected=0)
        plan = FixPlan(
            applicable=True,
            outcomes=(
                FixerOutcome(
                    fixer_id="REPAIR.STATS_RECOMPUTE",
                    check_id="STATISTICAL.STATS_MATCH_DATA",
                    diff=diff,
                    summary=summary,
                ),
            ),
            output_path=out_path,
            applied=True,
        )
        out = _render(plan)
        assert "Repaired dataset written to" in out
        assert str(out_path) in out

    def test_dry_run_plan_shows_rerun_hint(self) -> None:
        diff = Diff(
            changes=(
                StatChange(feature="timestamp", stat_key="mean", old_value=1.5, new_value=1.0),
            ),
            check_id="STATISTICAL.STATS_MATCH_DATA",
            fixer_id="REPAIR.STATS_RECOMPUTE",
        )
        plan = FixPlan(
            applicable=True,
            outcomes=(
                FixerOutcome(
                    fixer_id="REPAIR.STATS_RECOMPUTE",
                    check_id="STATISTICAL.STATS_MATCH_DATA",
                    diff=diff,
                    summary=None,
                ),
            ),
            output_path=None,
            applied=False,
        )
        out = _render(plan)
        assert "Dry run only" in out
        assert "--apply --out" in out


class TestJsonRendererBranches:
    def test_applied_outcome_includes_summary_fields(self, tmp_path: Path) -> None:
        out_path = tmp_path / "out"
        diff = Diff(
            changes=(
                StatChange(feature="timestamp", stat_key="mean", old_value=1.5, new_value=1.0),
            ),
            check_id="STATISTICAL.STATS_MATCH_DATA",
            fixer_id="REPAIR.STATS_RECOMPUTE",
        )
        summary = RepairSummary(output_path=out_path, changes_written=1, frames_corrected=4)
        plan = FixPlan(
            applicable=True,
            outcomes=(
                FixerOutcome(
                    fixer_id="REPAIR.STATS_RECOMPUTE",
                    check_id="STATISTICAL.STATS_MATCH_DATA",
                    diff=diff,
                    summary=summary,
                ),
            ),
            output_path=out_path,
            applied=True,
        )
        data = json.loads(render_fix_json("test/ref", plan))
        fixer = data["fixers"][0]
        assert fixer["applied"] is True
        assert fixer["frames_corrected"] == 4
        assert fixer["changes_written"] == 1

    def test_dry_run_outcome_has_null_summary_fields(self) -> None:
        diff = Diff(
            changes=(), check_id="SEMANTIC.TASK_INTEGRITY", fixer_id="REPAIR.TASK_INDEX_REPAIR"
        )
        plan = FixPlan(
            applicable=False,
            outcomes=(
                FixerOutcome(
                    fixer_id="REPAIR.TASK_INDEX_REPAIR",
                    check_id="SEMANTIC.TASK_INTEGRITY",
                    diff=diff,
                    summary=None,
                ),
            ),
            output_path=None,
            applied=False,
        )
        data = json.loads(render_fix_json("test/ref", plan))
        fixer = data["fixers"][0]
        assert fixer["applied"] is False
        assert fixer["frames_corrected"] is None
        assert fixer["changes_written"] is None


class TestJsonErrorRenderer:
    def test_error_schema_has_no_fixers(self) -> None:
        data = json.loads(render_fix_json_error("test/ref", "RepairError", "could not repair"))
        assert data["error_category"] == "RepairError"
        assert data["error_message"] == "could not repair"
        assert data["fixers"] == []
        assert data["dry_run"] is None
        assert data["applicable"] is None
