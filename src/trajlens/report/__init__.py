"""Report builder — renders CheckResults to terminal, JSON, HTML, and SARIF."""

from __future__ import annotations

from trajlens.report.html_report import render_html
from trajlens.report.json_report import render_json
from trajlens.report.sarif import render_sarif
from trajlens.report.terminal import render_terminal
from trajlens.report.trust_score import SCORE_FORMULA_VERSION, compute_trust_score

__all__ = [
    "SCORE_FORMULA_VERSION",
    "compute_trust_score",
    "render_html",
    "render_json",
    "render_sarif",
    "render_terminal",
]
