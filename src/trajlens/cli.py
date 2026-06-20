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
) -> None:
    """Validate a LeRobotDataset and report its quality grade."""
    import sys

    from trajlens.checks import CheckContext, CheckEngine, Severity, registry
    from trajlens.checks.protocol import CheckResult
    from trajlens.errors import DatasetError
    from trajlens.model import build_canonical_dataset
    from trajlens.sources.loader import SourceLoader

    if json_output or report:
        typer.echo(
            "ERROR: --json and --report output are M5 scope and not yet implemented. "
            "Run without those flags for plain terminal output.",
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        handle = SourceLoader().resolve(ref)
        ds = build_canonical_dataset(handle)
    except DatasetError as exc:
        typer.echo(f"ERROR: Could not load dataset {ref!r}: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    ctx = CheckContext(deep=deep)
    engine = CheckEngine(registry)
    results: list[CheckResult] = engine.run(ds, ctx)

    _severity_symbol = {
        Severity.ERROR: "✖ ERROR",
        Severity.FAIL: "✖ FAIL ",
        Severity.WARN: "⚠ WARN ",
        Severity.INFO: "✔ INFO ",
    }

    typer.echo(f"\ntrajlens lint: {ref}")
    typer.echo(f"  version : {ds.version.value}")
    typer.echo(f"  episodes: {ds.num_episodes}")
    typer.echo(f"  frames  : {ds.num_frames}")
    typer.echo("")

    counts: dict[Severity, int] = dict.fromkeys(Severity, 0)
    for result in results:
        counts[result.severity] += 1
        sym = _severity_symbol[result.severity]
        typer.echo(f"  {sym}  {result.check_id}")
        typer.echo(f"           {result.message}")

    typer.echo("")
    typer.echo(
        f"  Summary: {counts[Severity.FAIL]} FAIL, {counts[Severity.WARN]} WARN, "
        f"{counts[Severity.ERROR]} ERROR, {counts[Severity.INFO]} INFO"
    )

    worst = max((r.severity for r in results), default=Severity.INFO)
    if worst >= Severity.ERROR:
        typer.echo("  Grade  : ERROR (checks could not evaluate)")
        sys.exit(3)
    elif worst >= Severity.FAIL:
        typer.echo("  Grade  : FAIL (unsafe to train on)")
        sys.exit(1)
    elif worst >= Severity.WARN:
        typer.echo("  Grade  : WARN (usable with caution)")
        sys.exit(0)
    else:
        typer.echo("  Grade  : PASS")
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
