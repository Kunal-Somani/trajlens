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
    share: Annotated[
        bool,
        typer.Option(
            "--share",
            help="Print a redacted finding summary to stdout, for pasting into a GitHub issue.",
        ),
    ] = False,
    share_out: Annotated[
        str | None,
        typer.Option(
            "--share-out", help="Write the --share summary to this path instead of stdout."
        ),
    ] = None,
    baseline: Annotated[
        str | None,
        typer.Option(
            "--baseline",
            help=(
                "Compare results against this baseline file; exit code is "
                "driven by new findings only."
            ),
        ),
    ] = None,
    update_baseline: Annotated[
        str | None,
        typer.Option(
            "--update-baseline",
            help="Write current results to this baseline file and exit 0.",
        ),
    ] = None,
    show_unchanged: Annotated[
        bool,
        typer.Option(
            "--show-unchanged",
            help="With --baseline, also show unchanged findings (suppressed by default).",
        ),
    ] = False,
    parallel: Annotated[
        int,
        typer.Option(
            "--parallel",
            help=(
                "Number of parallel worker processes (default: 1, serial). "
                "Values >1 use ProcessPoolExecutor."
            ),
        ),
    ] = 1,
) -> None:
    """Validate a LeRobotDataset and report its quality grade."""
    import sys
    from pathlib import Path

    from trajlens.baseline import BaselineStore
    from trajlens.checks import CheckContext, CheckEngine, Severity, registry
    from trajlens.checks.protocol import CheckResult
    from trajlens.errors import DatasetError, DatasetFormatError
    from trajlens.model import build_canonical_dataset
    from trajlens.report import (
        render_html,
        render_json,
        render_json_load_error,
        render_sarif,
        render_share,
        render_terminal,
    )
    from trajlens.report.trust_score import integrity_only
    from trajlens.sources.loader import SourceLoader

    if show_unchanged and baseline is None:
        typer.echo("ERROR: --show-unchanged requires --baseline.", err=True)
        raise typer.Exit(code=2)

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
    engine_result = engine.run(ds, ctx, parallel=parallel)
    results: list[CheckResult] = list(engine_result.results)
    skipped_checks = engine_result.skipped

    if update_baseline is not None:
        BaselineStore.from_results(results).save(Path(update_baseline))

    baseline_diff = None
    if baseline is not None:
        try:
            store = BaselineStore.load(Path(baseline))
        except DatasetFormatError as exc:
            if json_output:
                typer.echo(render_json_load_error(ref, type(exc).__name__, str(exc)))
            else:
                typer.echo(f"ERROR: Could not load baseline {baseline!r}: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        baseline_diff = store.diff(results)

    worst = (
        max((r.severity for r in integrity_only(baseline_diff.new)), default=Severity.INFO)
        if baseline_diff is not None
        else max((r.severity for r in integrity_only(results)), default=Severity.INFO)
    )

    if json_output:
        typer.echo(
            render_json(
                ref,
                ds.format_id,
                ds.format_version,
                ds.num_episodes,
                ds.num_frames,
                results,
                baseline_diff=baseline_diff,
                skipped_checks=skipped_checks,
            )
        )
    else:
        render_terminal(
            ref,
            ds.format_id,
            ds.format_version,
            ds.num_episodes,
            ds.num_frames,
            results,
            baseline_diff=baseline_diff,
            show_unchanged=show_unchanged,
            skipped_checks=skipped_checks,
        )

    if update_baseline is not None:
        sys.exit(0)

    if report is not None:
        html = render_html(
            ref, ds.format_id, ds.format_version, ds.num_episodes, ds.num_frames, results
        )
        Path(report).write_text(html, encoding="utf-8")

    if sarif is not None:
        sarif_doc = render_sarif(
            ref,
            ds.format_id,
            ds.format_version,
            ds.num_episodes,
            ds.num_frames,
            results,
            skipped_checks=skipped_checks,
        )
        Path(sarif).write_text(sarif_doc, encoding="utf-8")

    if share or share_out is not None:
        dataset_ref = handle.repo_id if handle.repo_id is not None else Path(ref).name
        share_doc = render_share(ds.format_id, ds.format_version, results, dataset_ref=dataset_ref)
        if share_out is not None:
            Path(share_out).write_text(share_doc, encoding="utf-8")
        else:
            typer.echo(share_doc)

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
    only: Annotated[
        str | None,
        typer.Option(
            "--only",
            help=(
                "Comma-separated fixer id(s) to run, bypassing the normal "
                "WARN+ selection threshold (e.g. --only REPAIR.TASK_INDEX_REPAIR)."
            ),
        ),
    ] = None,
    except_: Annotated[
        str | None,
        typer.Option(
            "--except",
            help="Comma-separated fixer id(s) to exclude from the otherwise-applicable set.",
        ),
    ] = None,
    quarantine: Annotated[
        bool,
        typer.Option(
            "--quarantine",
            help=(
                "REPAIR.ORPHAN_SHARD_REPORT only: move orphan shards to "
                "<output>/.trajlens-quarantine/ instead of just reporting them."
            ),
        ),
    ] = False,
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

    A fixer may report no-op if an earlier repair already resolved its finding.

    Exit codes (M1-D: gated on RepairPlan.repairable -- see
    repair/orchestrator.py's plan() and repair/protocol.py's Repairability --
    never on DETECTED_NOT_WRITABLE or NO_FIXER findings, which are reported
    but do not affect the exit code):
      0 = every REPAIRABLE finding was applied (--apply) or previewed
          (dry-run) successfully
      1 = at least one REPAIRABLE finding failed to apply (a fixer raised
          RepairError because the underlying data is itself internally
          inconsistent)
      2 = no REPAIRABLE findings existed (nothing to repair), or the run
          could not start at all: dataset failed to load, or invalid usage
          (--apply without --out, --out same as the source, or an
          invalid/conflicting --only/--except selection)
    Future maintainers: do not change these codes' meaning silently -- other
    tooling (CI gates, scripts) depends on the 0/1/2 split above.
    """
    import sys
    from pathlib import Path

    from trajlens.checks import CheckContext, CheckEngine, registry
    from trajlens.errors import DatasetError, RepairError
    from trajlens.model import build_canonical_dataset
    from trajlens.repair.orchestrator import (
        build_fixer_order,
        refuse_if_hub_ref,
        run_apply,
        run_dry_run,
        validate_fixer_selection,
    )
    from trajlens.repair.orchestrator import plan as compute_repair_plan
    from trajlens.report import render_fix_json, render_fix_json_error, render_fix_terminal
    from trajlens.sources.loader import SourceLoader

    def _fail_usage(message: str) -> None:
        if json_output:
            typer.echo(render_fix_json_error(ref, "UsageError", message))
        else:
            typer.echo(f"ERROR: {message}", err=True)
        raise typer.Exit(code=2)

    if not dry_run and out is None:
        _fail_usage("--apply requires --out <path> (never writes to an implicit location).")

    if out is not None:
        source_candidate = Path(ref)
        if source_candidate.exists() and source_candidate.resolve() == Path(out).resolve():
            _fail_usage("--out must not be the same path as the source dataset (copy-on-write).")

    only_ids = frozenset(i.strip() for i in only.split(",") if i.strip()) if only else frozenset()
    except_ids = (
        frozenset(i.strip() for i in except_.split(",") if i.strip()) if except_ else frozenset()
    )
    selection_error = validate_fixer_selection(only_ids, except_ids)
    if selection_error is not None:
        _fail_usage(selection_error)

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
    results = list(engine.run(ds, ctx).results)
    fixer_order = build_fixer_order(quarantine=quarantine)

    try:
        # plan()'s dry_run() calls for each REPAIRABLE finding can themselves
        # raise RepairError (e.g. episode_reindex refusing interleaved data),
        # so it shares this try/except with run_dry_run/run_apply below rather
        # than running unguarded.
        repair_plan = compute_repair_plan(ds, results, fixer_order=fixer_order)
        if dry_run:
            fix_plan = run_dry_run(
                ds, results, fixer_order=fixer_order, only=only_ids, except_=except_ids
            )
        else:
            assert out is not None  # guarded above
            fix_plan = run_apply(
                ds, results, Path(out), fixer_order=fixer_order, only=only_ids, except_=except_ids
            )
    except RepairError as exc:
        if json_output:
            typer.echo(render_fix_json_error(ref, type(exc).__name__, str(exc)))
        else:
            typer.echo(f"ERROR: Could not repair {ref!r}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(render_fix_json(ref, fix_plan, repair_plan=repair_plan))
    else:
        render_fix_terminal(ref, fix_plan, repair_plan=repair_plan, format_id=ds.format_id)

    sys.exit(0 if repair_plan.repairable else 2)


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


@app.command()
def watch(
    ref: Annotated[
        str,
        typer.Argument(
            help=(
                "Local path to a dataset directory being recorded. Hub refs are "
                "not supported — Hub is immutable post-upload."
            )
        ),
    ],
    deep: Annotated[
        bool,
        typer.Option("--deep", help="Include video checks in each episode lint (slower)."),
    ] = False,
) -> None:
    """Lint episodes incrementally as a recording rig writes them. Exits cleanly on Ctrl-C."""
    from pathlib import Path

    from rich.console import Console

    from trajlens.watch import Watcher

    parts = ref.split("/")
    if not ref.startswith("/") and len(parts) == 2 and all(parts):
        typer.echo(
            f"ERROR: {ref!r} looks like a Hugging Face Hub ref, which watch does not "
            f"support — Hub is immutable post-upload. Pass a local dataset directory.",
            err=True,
        )
        raise typer.Exit(code=2)

    local_path = Path(ref)
    if not local_path.exists() or not local_path.is_dir():
        typer.echo(f"ERROR: {ref!r} is not an existing local directory.", err=True)
        raise typer.Exit(code=2)

    console = Console()
    watcher = Watcher(local_path, deep=deep)
    watcher.run(console)
