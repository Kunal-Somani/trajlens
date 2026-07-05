"""Renderers for `trajlens fix` output — terminal (rich) and --json.

Mirrors report/terminal.py and report/json_report.py's conventions: plain
functions taking structured data (here, a repair.FixPlan) and either printing
(terminal) or returning a JSON string (--json). No repair logic lives here —
this module only renders a FixPlan that orchestrator.py already computed.

JSON schema:
  {
    "ref": str,
    "dry_run": bool,
    "applicable": bool,
    "output_path": str | null,
    "fixers": [
      {
        "fixer_id": str,
        "check_id": str,
        "is_noop": bool,
        "num_changes": int,
        "applied": bool,
        "frames_corrected": int | null,   # from RepairSummary, only when applied
        "changes_written": int | null
      },
      ...
    ]
  }
"""

from __future__ import annotations

import json

from rich.console import Console

from trajlens.repair.orchestrator import FixPlan
from trajlens.repair.protocol import BoundaryChange, FrameChange, StatChange


def _change_summary(change: FrameChange | StatChange | BoundaryChange) -> str:
    # Parentheses, not brackets: rich.Console.print() treats [...] as markup
    # tag syntax and silently swallows unrecognized tags (e.g. a column or
    # field name like "[timestamp]" or "[dataset_to_index]" would vanish).
    if isinstance(change, FrameChange):
        return (
            f"episode {change.episode_index} frame {change.frame_index} "
            f"({change.column}): {change.old_value} -> {change.new_value}"
        )
    if isinstance(change, StatChange):
        return f"{change.feature}.{change.stat_key}: {change.old_value} -> {change.new_value}"
    return (
        f"episode {change.episode_index} ({change.field}): {change.old_value} -> {change.new_value}"
    )


def render_fix_terminal(
    ref: str,
    plan: FixPlan,
    *,
    console: Console | None = None,
) -> None:
    """Print a color-coded fix report (dry-run diff or applied summary)."""
    con = console or Console()
    mode = "apply" if plan.applied else "dry-run"

    con.print()
    con.print(f"[bold]trajlens fix:[/bold] {ref}  [dim]({mode})[/dim]")
    con.print()

    if not plan.outcomes:
        con.print("  [green]No applicable fixers — nothing to fix.[/green]")
        con.print()
        return

    if not plan.applicable:
        con.print("  [green]All applicable fixers report the dataset is already clean.[/green]")
        con.print("  Nothing to fix.")
        con.print()
        return

    for outcome in plan.outcomes:
        if outcome.diff.is_noop:
            con.print(
                f"  [green]OK[/green]     {outcome.fixer_id} ({outcome.check_id}): "
                "no changes needed"
            )
            continue

        verb = "applied" if outcome.summary is not None else "would change"
        con.print(
            f"  [yellow]CHANGE[/yellow] {outcome.fixer_id} ({outcome.check_id}): "
            f"{len(outcome.diff.changes)} change(s) {verb}"
        )
        preview = outcome.diff.changes[:5]
        for change in preview:
            con.print(f"           {_change_summary(change)}")
        remaining = len(outcome.diff.changes) - len(preview)
        if remaining > 0:
            con.print(f"           ... and {remaining} more")

    con.print()
    if plan.applied:
        con.print(f"  Repaired dataset written to: {plan.output_path}")
    else:
        con.print("  Dry run only — nothing was written.")
        con.print("  Re-run with --apply --out <path> to write.")
    con.print()


def render_fix_json(ref: str, plan: FixPlan) -> str:
    """Return a JSON string representing the fix report."""
    payload: dict[str, object] = {
        "ref": ref,
        "dry_run": not plan.applied,
        "applicable": plan.applicable,
        "output_path": str(plan.output_path) if plan.output_path is not None else None,
        "fixers": [
            {
                "fixer_id": outcome.fixer_id,
                "check_id": outcome.check_id,
                "is_noop": outcome.diff.is_noop,
                "num_changes": len(outcome.diff.changes),
                "applied": outcome.summary is not None,
                "frames_corrected": (
                    outcome.summary.frames_corrected if outcome.summary is not None else None
                ),
                "changes_written": (
                    outcome.summary.changes_written if outcome.summary is not None else None
                ),
            }
            for outcome in plan.outcomes
        ],
    }
    return json.dumps(payload, indent=2)


def render_fix_json_error(ref: str, error_category: str, message: str) -> str:
    """Return a JSON string for a fix run that could not proceed (load or RepairError).

    Mirrors report/json_report.py's render_json_load_error schema shape so
    --json output is never ambiguous between "ran and found nothing" and
    "could not run at all."
    """
    payload: dict[str, object] = {
        "ref": ref,
        "dry_run": None,
        "applicable": None,
        "output_path": None,
        "error_category": error_category,
        "error_message": message,
        "fixers": [],
    }
    return json.dumps(payload, indent=2)
