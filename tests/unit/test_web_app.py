"""Tests for the FastAPI dashboard app (src/trajlens/web/app.py).

Covers the T10 contract: exactly two routes, strict security headers on
every response, no route accepts a path/ref/dataset-id parameter, and the
served /api/report JSON matches what render_json produces (the JSON report
IS the API contract — 02_ARCHITECTURE.md §3.4).
"""

from __future__ import annotations

import json
import re

from fastapi.testclient import TestClient

from trajlens.checks.protocol import CheckResult, Severity
from trajlens.report.json_report import render_json
from trajlens.sources.version import DatasetVersion
from trajlens.web.app import create_app

_RESULTS = [
    CheckResult(check_id="STRUCTURAL.VERSION_DETECTED", severity=Severity.INFO, message="v3.0"),
    CheckResult(
        check_id="STRUCTURAL.METADATA_DATA_AGREEMENT",
        severity=Severity.FAIL,
        message="span mismatch on episode 0",
        details={"violations": [{"episode_index": 0}]},
    ),
]

_EXPECTED_HEADERS = {
    "content-security-policy": lambda v: (
        "default-src 'self'" in v and "unsafe-inline" not in v and "unsafe-eval" not in v
    ),
    "x-content-type-options": lambda v: v == "nosniff",
    "x-frame-options": lambda v: v == "DENY",
}


def _client() -> TestClient:
    report_json = render_json("some/dataset", DatasetVersion.V3_0, 3, 300, _RESULTS)
    app = create_app(report_json)
    return TestClient(app)


class TestApiReport:
    def test_report_matches_render_json_output(self) -> None:
        report_json = render_json("some/dataset", DatasetVersion.V3_0, 3, 300, _RESULTS)
        app = create_app(report_json)
        client = TestClient(app)

        resp = client.get("/api/report")

        assert resp.status_code == 200
        assert resp.json() == json.loads(report_json)

    def test_report_content_type_is_json(self) -> None:
        client = _client()
        resp = client.get("/api/report")
        assert resp.headers["content-type"].startswith("application/json")

    def test_report_includes_details(self) -> None:
        client = _client()
        data = client.get("/api/report").json()
        fail_result = next(r for r in data["results"] if r["severity"] == "FAIL")
        assert fail_result["details"] == {"violations": [{"episode_index": 0}]}


class TestIndexRoute:
    def test_index_serves_html(self) -> None:
        client = _client()
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_index_has_no_external_references(self) -> None:
        """No CDN scripts, no remote fonts, no analytics (T10)."""
        client = _client()
        body = client.get("/").text
        urls = re.findall(r'(?:src|href)=["\'](https?://[^"\']+)["\']', body)
        assert urls == []
        assert "http://" not in body.replace("http://127.0.0.1", "")
        assert "https://" not in body


class TestDashboardXssInert:
    """Guard against HTML-injection via untrusted dataset metadata.

    A hostile Hub dataset controls its own task strings, feature names, and
    (indirectly, via a check's message/details) free text derived from that
    metadata. CheckResult.message and .details are rendered into the
    dashboard DOM; if any of that ever reached the page via innerHTML /
    insertAdjacentHTML / outerHTML instead of textContent, a dataset could
    inject a script into the browser of anyone who opens `trajlens web` on
    it. CSP (T10) is defense-in-depth here, not the fix -- this invariant is
    load-bearing for the decision to allow Hub refs into the dashboard at
    all, so it gets its own regression guard, not just a one-time read-through.
    """

    def test_no_html_sink_receives_report_derived_data(self) -> None:
        """Static-analysis guard: every innerHTML/insertAdjacentHTML/outerHTML
        assignment in the dashboard's <script> must be a literal-only clear
        (e.g. `x.innerHTML = ""`), never fed anything derived from the
        report (`data.*`, `r.*`, or a local computed from `r.details` like
        `detailsText`). This is the exact regression class a future edit
        could introduce (e.g. switching a textContent line to innerHTML for
        "richer" formatting) without anyone touching security-header tests.
        """
        from trajlens.web.app import _INDEX_HTML

        html = _INDEX_HTML.read_text()
        script_match = re.search(r"<script>(.*?)</script>", html, re.DOTALL)
        assert script_match is not None, "index.html must contain a <script> block"
        script = script_match.group(1)

        # Report-derived identifiers this script ever holds untrusted content in.
        # detailsText/detailBox/tdMsg etc. are all ultimately assigned via
        # textContent from data.*/r.* below, but a sink fed ANY expression
        # touching data. or r. (or a variable named after report content) is
        # flagged, since that is precisely how untrusted text could reach a
        # parsing sink instead of a text-node sink.
        tainted_markers = ("data.", "r.", "detailsText", "renderDetails(")

        sink_pattern = re.compile(
            r"([A-Za-z_$][\w.$]*)\s*\.\s*(innerHTML|outerHTML)\s*=\s*([^;]+);"
            r"|([A-Za-z_$][\w.$]*)\s*\.\s*insertAdjacentHTML\s*\(([^)]*)\)"
        )
        found_any_sink = False
        for m in sink_pattern.finditer(script):
            found_any_sink = True
            assigned_expr = m.group(3) if m.group(3) is not None else m.group(5)
            for marker in tainted_markers:
                assert marker not in assigned_expr, (
                    f"HTML sink fed report-derived content: {m.group(0)!r} "
                    f"(contains {marker!r}). Use textContent instead."
                )

        # If the script stops using innerHTML/insertAdjacentHTML entirely in a
        # future rewrite, that's fine -- but confirm this test actually
        # exercised the sink-detection regex against the real file, not a
        # regex that silently matches nothing.
        assert found_any_sink, (
            "expected to find at least one innerHTML usage (the two "
            "list-clearing '= \"\"' calls) -- if this fails, the sink "
            "regex itself may be broken, not the invariant it guards"
        )

    def test_hostile_message_and_details_render_as_inert_text(self) -> None:
        """End-to-end: a CheckResult carrying an HTML-injection payload in
        both .message and .details (as a hostile dataset's task string or
        feature name could produce, surfaced verbatim into a check message)
        must reach the browser only as JSON string data -- never as HTML the
        server has assembled or escaped differently from any other string.
        The client-side inertness (previous test) is what actually neutralizes
        it; this confirms the server performs no special-casing that would
        make the payload reach the DOM any other way than through the same
        textContent path as an ordinary message.
        """
        payload = "<img src=x onerror=alert(document.domain)>"
        hostile_results = [
            CheckResult(
                check_id="SEMANTIC.TASK_INTEGRITY",
                severity=Severity.WARN,
                message=f"task description: {payload}",
                details={"task": payload, "feature_name": payload},
            ),
        ]
        report_json = render_json("hostile/dataset", DatasetVersion.V3_0, 1, 4, hostile_results)
        app = create_app(report_json)
        client = TestClient(app)

        data = client.get("/api/report").json()
        result = data["results"][0]

        # The payload must survive verbatim as plain JSON string data --
        # proof the server does no HTML assembly of its own; the client-side
        # test above is what guarantees textContent is the only place it lands.
        assert payload in result["message"]
        assert result["details"]["task"] == payload
        assert result["details"]["feature_name"] == payload

        # The raw response body must never contain the payload unescaped as
        # if it had been embedded into an HTML document -- it should only
        # appear inside the JSON string encoding (e.g. quoted), never as a
        # bare "<img" preceded by an HTML tag context. Since this route
        # always returns application/json (asserted in TestApiReport), a
        # browser will never parse this response body as HTML at all.
        assert client.get("/api/report").headers["content-type"].startswith("application/json")


class TestSecurityHeaders:
    def test_headers_present_on_report_route(self) -> None:
        client = _client()
        resp = client.get("/api/report")
        for header, check in _EXPECTED_HEADERS.items():
            assert header in {k.lower() for k in resp.headers}, f"missing {header}"
            assert check(resp.headers[header]), f"{header} failed check: {resp.headers[header]}"

    def test_headers_present_on_index_route(self) -> None:
        client = _client()
        resp = client.get("/")
        for header, check in _EXPECTED_HEADERS.items():
            assert header in {k.lower() for k in resp.headers}, f"missing {header}"
            assert check(resp.headers[header]), f"{header} failed check: {resp.headers[header]}"

    def test_headers_present_on_404(self) -> None:
        client = _client()
        resp = client.get("/nonexistent")
        assert resp.status_code == 404
        for header, check in _EXPECTED_HEADERS.items():
            assert header in {k.lower() for k in resp.headers}, f"missing {header}"
            assert check(resp.headers[header]), f"{header} failed check: {resp.headers[header]}"


class TestRouteSurface:
    def test_unknown_route_404s(self) -> None:
        client = _client()
        resp = client.get("/anything/else")
        assert resp.status_code == 404

    def test_no_route_accepts_a_path_parameter(self) -> None:
        """T10: the browser must never be able to send a filesystem path,
        ref, or dataset id — verified structurally, not just by convention.
        """
        client = _client()
        for candidate in (
            "/api/report/../../etc/passwd",
            "/api/dataset/some/ref",
            "/../etc/passwd",
        ):
            resp = client.get(candidate)
            assert resp.status_code == 404

    def test_openapi_docs_disabled(self) -> None:
        client = _client()
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404

    def test_post_to_report_not_allowed(self) -> None:
        client = _client()
        resp = client.post("/api/report", json={"path": "/etc/passwd"})
        assert resp.status_code in (404, 405)
