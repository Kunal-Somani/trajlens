"""Terminal renderer for lint results using rich.

Uses `rich` for color-coded severity output and a summary table.
rich is already a transitive dependency of typer (typer[all] ships rich);
we declare it directly because we call its public API.
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.console import Console
from rich.table import Table
from rich.text import Text

from trajlens.baseline import BaselineDiff
from trajlens.checks.protocol import CheckResult, Severity, Tier
from trajlens.report.trust_score import SCORE_FORMULA_VERSION, compute_trust_score, integrity_only


def _severity_style(severity: Severity) -> str:
    return {
        Severity.ERROR: "bold red",
        Severity.FAIL: "red",
        Severity.WARN: "yellow",
        Severity.INFO: "green",
    }[severity]


def _severity_label(severity: Severity) -> str:
    return {
        Severity.ERROR: "✖ ERROR",
        Severity.FAIL: "✖ FAIL",
        Severity.WARN: "⚠ WARN",
        Severity.INFO: "✔ INFO",
    }[severity]


def _grade(worst: Severity) -> tuple[str, str]:
    """Return (label, style) for the overall grade."""
    if worst >= Severity.ERROR:
        return "ERROR", "bold red"
    if worst >= Severity.FAIL:
        return "FAIL  — unsafe to train on", "red"
    if worst >= Severity.WARN:
        return "WARN  — usable with caution", "yellow"
    return "PASS", "bold green"


def _render_baseline_findings(con: Console, diff: BaselineDiff, *, show_unchanged: bool) -> None:
    for result in diff.new:
        label = _severity_label(result.severity)
        con.print(
            f"  [red][NEW][/red]  [{_severity_style(result.severity)}]{label}[/] {result.check_id}"
        )
        con.print(f"           {result.message}")
    for finding in diff.resolved:
        con.print(f"  [green][RESOLVED][/green]  {finding.check_id}")
    if show_unchanged:
        for result in diff.unchanged:
            label = _severity_label(result.severity)
            style = _severity_style(result.severity)
            con.print(f"  [UNCHANGED]  [{style}]{label}[/{style}] {result.check_id}")
            con.print(f"           {result.message}")
    con.print()
    con.print(
        f"  baseline: {len(diff.new)} new, {len(diff.resolved)} resolved, "
        f"{len(diff.unchanged)} unchanged"
    )
    con.print()


def render_terminal(
    ref: str,
    format_id: str,
    format_version: str,
    num_episodes: int,
    num_frames: int | None,
    results: list[CheckResult],
    *,
    console: Console | None = None,
    baseline_diff: BaselineDiff | None = None,
    show_unchanged: bool = False,
    skipped_checks: Sequence[str] = (),
) -> None:
    """Print a color-coded lint report to the terminal.

    When baseline_diff is given, each result is prefixed with [NEW]
    (red), [RESOLVED] (green), or [UNCHANGED] (suppressed unless
    show_unchanged) instead of the plain per-finding listing.

    Findings are grouped INTEGRITY tier first, then QUALITY tier (advisory,
    never affects the trust score or grade). skipped_checks lists check_ids
    the engine skipped due to format scope; a summary line is printed for
    them when non-empty.
    """
    con = console or Console()

    con.print()
    con.print(f"[bold]trajlens lint:[/bold] {ref}")
    con.print(f"  format   : {format_id} v{format_version}")
    con.print(f"  episodes : {num_episodes}")
    frames_str = str(num_frames) if num_frames is not None else "unknown"
    con.print(f"  frames   : {frames_str}")
    con.print()

    if baseline_diff is not None:
        _render_baseline_findings(con, baseline_diff, show_unchanged=show_unchanged)
    elif results:
        integrity_results = [r for r in results if r.tier is Tier.INTEGRITY]
        quality_results = [r for r in results if r.tier is Tier.QUALITY]

        for result in integrity_results:
            style = _severity_style(result.severity)
            label = _severity_label(result.severity)
            con.print(f"  [{style}]{label}[/{style}]  {result.check_id}")
            con.print(f"           {result.message}")

        if quality_results:
            con.print()
            con.print("[bold]Quality findings (advisory — do not affect grade)[/bold]")
            for result in quality_results:
                style = _severity_style(result.severity)
                label = _severity_label(result.severity)
                con.print(f"  [{style}]{label}[/{style}]  {result.check_id}")
                con.print(f"           {result.message}")

        con.print()

    if skipped_checks:
        con.print(f"  {len(skipped_checks)} checks skipped (format scope)")
        con.print()

    counts: dict[Severity, int] = dict.fromkeys(Severity, 0)
    for r in results:
        counts[r.severity] += 1

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Severity")
    table.add_column("Count", justify="right")
    for sev in (Severity.FAIL, Severity.WARN, Severity.ERROR, Severity.INFO):
        style = _severity_style(sev)
        table.add_row(Text(sev.value, style=style), str(counts[sev]))
    con.print(table)
    con.print()

    score = compute_trust_score(results)
    worst = max((r.severity for r in integrity_only(results)), default=Severity.INFO)
    grade_label, grade_style = _grade(worst)

    con.print(f"  Trust score : {score}/100  (formula v{SCORE_FORMULA_VERSION})")
    con.print(f"  Grade       : [{grade_style}]{grade_label}[/{grade_style}]")
    con.print()
