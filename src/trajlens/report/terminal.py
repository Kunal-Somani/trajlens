"""Terminal renderer for lint results using rich.

Uses `rich` for color-coded severity output and a summary table.
rich is already a transitive dependency of typer (typer[all] ships rich);
we declare it directly because we call its public API.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.text import Text

from trajlens.checks.protocol import CheckResult, Severity
from trajlens.report.trust_score import SCORE_FORMULA_VERSION, compute_trust_score
from trajlens.sources.version import DatasetVersion


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


def render_terminal(
    ref: str,
    version: DatasetVersion,
    num_episodes: int,
    num_frames: int | None,
    results: list[CheckResult],
    *,
    console: Console | None = None,
) -> None:
    """Print a color-coded lint report to the terminal."""
    con = console or Console()

    con.print()
    con.print(f"[bold]trajlens lint:[/bold] {ref}")
    con.print(f"  version  : {version.value}")
    con.print(f"  episodes : {num_episodes}")
    frames_str = str(num_frames) if num_frames is not None else "unknown"
    con.print(f"  frames   : {frames_str}")
    con.print()

    if results:
        for result in results:
            style = _severity_style(result.severity)
            label = _severity_label(result.severity)
            con.print(f"  [{style}]{label}[/{style}]  {result.check_id}")
            con.print(f"           {result.message}")
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
    worst = max((r.severity for r in results), default=Severity.INFO)
    grade_label, grade_style = _grade(worst)

    con.print(f"  Trust score : {score}/100  (formula v{SCORE_FORMULA_VERSION})")
    con.print(f"  Grade       : [{grade_style}]{grade_label}[/{grade_style}]")
    con.print()
