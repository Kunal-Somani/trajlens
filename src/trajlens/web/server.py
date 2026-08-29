"""Resolves a dataset, lints it once, and serves the read-only dashboard.

Binds to 127.0.0.1 only (T10: local by default, no flag to widen the bind
in this version). The dataset is resolved once here, at CLI launch time,
and never again — the server has no route that accepts a path, ref, or
dataset id from the browser (06_SECURITY_AND_THREAT_MODEL.md T10).
"""

from __future__ import annotations

import webbrowser

import uvicorn

from trajlens.checks import CheckContext, CheckEngine, registry
from trajlens.checks.protocol import CheckResult
from trajlens.model import build_canonical_dataset
from trajlens.report import render_json
from trajlens.sources.loader import SourceLoader
from trajlens.web.app import create_app

_BIND_HOST = "127.0.0.1"


def build_report_json(ref: str) -> str:
    """Resolve *ref*, run the default (non-deep) check set, render JSON.

    Raises DatasetError if the dataset cannot be resolved or loaded — the
    caller (CLI) is responsible for converting that into a clean exit, same
    as `lint` and `fix` already do.
    """
    handle = SourceLoader().resolve(ref)
    ds = build_canonical_dataset(handle)
    ctx = CheckContext(deep=False)
    engine = CheckEngine(registry)
    results: list[CheckResult] = engine.run(ds, ctx)
    return render_json(
        ref, ds.format_id, ds.format_version, ds.num_episodes, ds.num_frames, results
    )


def dashboard_url(port: int) -> str:
    """Return the localhost URL the dashboard will be served at."""
    return f"http://{_BIND_HOST}:{port}/"


def serve(ref: str, *, port: int, open_browser: bool) -> None:
    """Lint *ref* once and serve the dashboard on 127.0.0.1:*port* until killed.

    Raises DatasetError if the dataset cannot be resolved or loaded. The
    caller is responsible for printing the URL before calling this, since
    uvicorn.run() blocks until the server is killed.
    """
    report_json = build_report_json(ref)
    app = create_app(report_json)

    if open_browser:
        webbrowser.open(dashboard_url(port))

    uvicorn.run(app, host=_BIND_HOST, port=port, log_level="warning")
