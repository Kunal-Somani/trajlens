"""trajlens command-line interface.

Entry point: `trajlens` (registered in pyproject.toml [project.scripts]).
All subcommands are defined here; heavy logic lives in the library modules.
"""

from __future__ import annotations

from typing import Annotated, Optional

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
        Optional[bool],
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
        Optional[str],
        typer.Option("--report", help="Write HTML report to this path."),
    ] = None,
) -> None:
    """Validate a LeRobotDataset and report its quality grade."""
    # Implemented in M4–M6. Raises NotImplementedError so CI build-smoke test
    # (`trajlens --version`) passes while `trajlens lint` is correctly not-done.
    raise NotImplementedError(
        "trajlens lint is not yet implemented (M4 milestone). "
        "Run `trajlens --version` to confirm the installation is working."
    )


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
        Optional[str],
        typer.Option("--out", help="Output path for the repaired dataset copy."),
    ] = None,
) -> None:
    """Repair issues found by lint (copy-on-write; dry-run by default)."""
    raise NotImplementedError(
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
    raise NotImplementedError(
        "trajlens web is not yet implemented (v0.2 milestone)."
    )
