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
