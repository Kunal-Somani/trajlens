"""FastAPI app factory for the read-only lint dashboard (02_ARCHITECTURE.md §3.4).

Exactly two routes: the static dashboard shell and GET /api/report. The
report JSON is computed once, before the app is built, and served verbatim
from memory — no route accepts a path, ref, or dataset id from the client
(06_SECURITY_AND_THREAT_MODEL.md T10). This is a read-only shell over the
library (ADR-001): it never imports repair/ and never writes anything.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse

_STATIC_DIR = Path(__file__).parent / "static"
_INDEX_HTML = _STATIC_DIR / "index.html"

# No unsafe-inline, no unsafe-eval, no remote origins: the dashboard is fully
# self-contained and must never depend on or leak to third-party hosts (T10).
_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "font-src 'self'; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "frame-ancestors 'none'"
)


def create_app(report_json: str) -> FastAPI:
    """Build the dashboard app bound to one already-computed report.

    *report_json* is the exact string trajlens' own JSON renderer produced
    for the dataset resolved at CLI launch. The app never re-resolves a
    dataset, never accepts one from a request, and exposes no route that
    takes a path/ref/id parameter.
    """
    app = FastAPI(
        title="trajlens dashboard",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def _security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = _CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/api/report")
    def get_report() -> Response:
        """Return the lint report computed at server startup, verbatim."""
        return Response(content=report_json, media_type="application/json")

    @app.get("/")
    def get_index() -> FileResponse:
        """Serve the single-file static dashboard."""
        return FileResponse(_INDEX_HTML)

    return app
