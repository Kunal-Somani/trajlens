"""trajlens command-line interface.

Entry point: `trajlens` (registered in pyproject.toml [project.scripts]).
All subcommands are defined here; heavy logic lives in the library modules.
"""

from __future__ import annotations

from typing import Annotated

import typer

import trajlens
from trajlens.logging import configure_logging

app = typer.Typer(
    name="trajlens",
    help="The quality and synthesis layer for the open robot-learning data ecosystem.",
    add_completion=False,
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"trajlens {trajlens.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """trajlens — lint, fix, and synthesize clean LeRobot datasets."""
    configure_logging(level="DEBUG" if verbose else "WARNING")


@app.command()
def lint(
    ref: Annotated[
        str,
        typer.Argument(help="Local path or Hugging Face Hub repo id (org/name)."),
    ],
    deep: Annotated[
        bool,
        typer.Option("--deep", help="Full video decode (slow; default is spot-check)."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON report to stdout."),
    ] = False,
    report: Annotated[
        str | None,
        typer.Option("--report", help="Write HTML report to this path."),
    ] = None,
    sarif: Annotated[
        str | None,
        typer.Option("--sarif", help="Write SARIF 2.1.0 report to this path."),
    ] = None,
) -> None:
    """Validate a LeRobotDataset and report its quality grade."""
    import sys
    from pathlib import Path

    from trajlens.checks import CheckContext, CheckEngine, Severity, registry
    from trajlens.checks.protocol import CheckResult
    from trajlens.errors import DatasetError
    from trajlens.model import build_canonical_dataset
    from trajlens.report import render_html, render_json, render_sarif, render_terminal
    from trajlens.sources.loader import SourceLoader

    try:
        handle = SourceLoader().resolve(ref)
        ds = build_canonical_dataset(handle)
    except DatasetError as exc:
        typer.echo(f"ERROR: Could not load dataset {ref!r}: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    ctx = CheckContext(deep=deep)
    engine = CheckEngine(registry)
    results: list[CheckResult] = engine.run(ds, ctx)

    worst = max((r.severity for r in results), default=Severity.INFO)

    if json_output:
        typer.echo(render_json(ref, ds.version, ds.num_episodes, ds.num_frames, results))
    else:
        render_terminal(ref, ds.version, ds.num_episodes, ds.num_frames, results)

    if report is not None:
        html = render_html(ref, ds.version, ds.num_episodes, ds.num_frames, results)
        Path(report).write_text(html, encoding="utf-8")

    if sarif is not None:
        sarif_doc = render_sarif(ref, ds.version, ds.num_episodes, ds.num_frames, results)
        Path(sarif).write_text(sarif_doc, encoding="utf-8")

    if worst >= Severity.FAIL or worst >= Severity.ERROR:
        sys.exit(2)
    elif worst >= Severity.WARN:
        sys.exit(1)
    else:
        sys.exit(0)


@app.command()
def fix(
    ref: Annotated[
        str,
        typer.Argument(help="Local path to the dataset to repair."),
    ],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run/--apply", help="Preview changes without writing (default)."),
    ] = True,
    out: Annotated[
        str | None,
        typer.Option("--out", help="Output path for the repaired dataset copy."),
    ] = None,
) -> None:
    """Repair issues found by lint (copy-on-write; dry-run by default)."""
    raise NotImplementedError(  # pragma: no cover — stub until v0.2
        "trajlens fix is not yet implemented (v0.2 milestone)."
    )


@app.command()
def web(
    ref: Annotated[
        str,
        typer.Argument(help="Local path or Hub repo id to visualise."),
    ],
    port: Annotated[
        int,
        typer.Option("--port", help="Port to serve the dashboard on."),
    ] = 8000,
) -> None:
    """Open the web dashboard for a dataset lint report."""
    raise NotImplementedError(  # pragma: no cover — stub until v0.2
        "trajlens web is not yet implemented (v0.2 milestone)."
    )
