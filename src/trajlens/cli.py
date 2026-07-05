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
    from trajlens.report import (
        render_html,
        render_json,
        render_json_load_error,
        render_sarif,
        render_terminal,
    )
    from trajlens.sources.loader import SourceLoader

    try:
        handle = SourceLoader().resolve(ref)
        ds = build_canonical_dataset(handle)
    except DatasetError as exc:
        if json_output:
            typer.echo(render_json_load_error(ref, type(exc).__name__, str(exc)))
        else:
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
        typer.Argument(help="Local path or Hugging Face Hub repo id (org/name)."),
    ],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run/--apply", help="Preview changes without writing (default)."),
    ] = True,
    out: Annotated[
        str | None,
        typer.Option(
            "--out", help="Output path for the repaired dataset copy. Required with --apply."
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON report to stdout."),
    ] = False,
) -> None:
    """Repair issues found by lint (copy-on-write; dry-run by default).

    Runs lint first, determines which fixers apply to the findings, then
    either previews (dry-run, default) or applies (--apply) each applicable
    fixer's repair. Only local datasets can be repaired: a Hub ref's data and
    video shards are streamed on demand, never fully present on disk to copy.

    Exit codes:
      0 = nothing to fix (dataset already agrees with data for every
          applicable check)
      1 = fixes proposed (dry-run) or applied (--apply) successfully
      2 = could not fix: dataset failed to load, invalid usage (--apply
          without --out, or --out same as the source), or a fixer refused to
          repair (RepairError) because the underlying data is itself
          internally inconsistent
    """
    import sys
    from pathlib import Path

    from trajlens.checks import CheckContext, CheckEngine, registry
    from trajlens.errors import DatasetError, RepairError
    from trajlens.model import build_canonical_dataset
    from trajlens.repair.orchestrator import refuse_if_hub_ref, run_apply, run_dry_run
    from trajlens.report import render_fix_json, render_fix_json_error, render_fix_terminal
    from trajlens.sources.loader import SourceLoader

    if not dry_run and out is None:
        message = "--apply requires --out <path> (never writes to an implicit location)."
        if json_output:
            typer.echo(render_fix_json_error(ref, "UsageError", message))
        else:
            typer.echo(f"ERROR: {message}", err=True)
        raise typer.Exit(code=2)

    if out is not None:
        source_candidate = Path(ref)
        if source_candidate.exists() and source_candidate.resolve() == Path(out).resolve():
            message = "--out must not be the same path as the source dataset (copy-on-write)."
            if json_output:
                typer.echo(render_fix_json_error(ref, "UsageError", message))
            else:
                typer.echo(f"ERROR: {message}", err=True)
            raise typer.Exit(code=2)

    try:
        handle = SourceLoader().resolve(ref)
        refuse_if_hub_ref(handle, ref)
        ds = build_canonical_dataset(handle)
    except DatasetError as exc:
        if json_output:
            typer.echo(render_fix_json_error(ref, type(exc).__name__, str(exc)))
        else:
            typer.echo(f"ERROR: Could not load dataset {ref!r}: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except RepairError as exc:
        if json_output:
            typer.echo(render_fix_json_error(ref, type(exc).__name__, str(exc)))
        else:
            typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    ctx = CheckContext(deep=False)
    engine = CheckEngine(registry)
    results = engine.run(ds, ctx)

    try:
        if dry_run:
            plan = run_dry_run(ds, results)
        else:
            assert out is not None  # guarded above
            plan = run_apply(ds, results, Path(out))
    except RepairError as exc:
        if json_output:
            typer.echo(render_fix_json_error(ref, type(exc).__name__, str(exc)))
        else:
            typer.echo(f"ERROR: Could not repair {ref!r}: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if json_output:
        typer.echo(render_fix_json(ref, plan))
    else:
        render_fix_terminal(ref, plan)

    sys.exit(1 if plan.applicable else 0)


@app.command()
def web(
    ref: Annotated[
        str,
        typer.Argument(help="Local path or Hugging Face Hub repo id (org/name)."),
    ],
    port: Annotated[
        int,
        typer.Option("--port", help="Port to serve the dashboard on."),
    ] = 8000,
    no_open: Annotated[
        bool,
        typer.Option("--no-open", help="Do not open a browser automatically."),
    ] = False,
) -> None:
    """Lint a dataset once and serve a read-only dashboard of the report.

    Read-only over the library: this never invokes fix, never writes
    anything. Binds to 127.0.0.1 only. A Hub ref is allowed (lint already
    streams Hub data read-only); the dataset is resolved once at launch and
    the server never accepts a path, ref, or dataset id from the browser.

    Exit codes:
      0 = server ran and was stopped normally (Ctrl-C)
      2 = dataset could not be resolved or loaded, or the [web] extra is
          not installed
    """
    from trajlens.errors import DatasetError

    try:
        from trajlens.web.server import dashboard_url, serve
    except ModuleNotFoundError as exc:
        message = "trajlens web requires the [web] extra: pip install 'trajlens[web]'"
        typer.echo(f"ERROR: {message}", err=True)
        raise typer.Exit(code=2) from exc

    try:
        url = dashboard_url(port)
        typer.echo(f"trajlens dashboard: {url}")
        serve(ref, port=port, open_browser=not no_open)
    except DatasetError as exc:
        typer.echo(f"ERROR: Could not load dataset {ref!r}: {exc}", err=True)
        raise typer.Exit(code=2) from exc
