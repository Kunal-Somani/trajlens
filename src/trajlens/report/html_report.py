"""HTML renderer for lint results (--report path.html).

Self-contained: all CSS is inlined, no external assets. Safe for standalone
viewing without a web server. Content is escaped via html.escape() — the
dataset ref and messages are untrusted strings that must not inject HTML.
"""

from __future__ import annotations

import html

from trajlens.checks.protocol import CheckResult, Severity
from trajlens.report.trust_score import SCORE_FORMULA_VERSION, compute_trust_score
from trajlens.sources.version import DatasetVersion

_CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    background: #0f1117;
    color: #e2e8f0;
    margin: 0;
    padding: 2rem;
    line-height: 1.6;
}
h1 { color: #f8fafc; font-size: 1.4rem; margin-bottom: 0.25rem; }
.meta { color: #94a3b8; font-size: 0.9rem; margin-bottom: 2rem; }
table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }
th { text-align: left; color: #94a3b8; font-weight: 600;
     border-bottom: 1px solid #334155; padding: 0.5rem 1rem; }
td { padding: 0.5rem 1rem; border-bottom: 1px solid #1e293b; }
.sev-ERROR  { color: #f87171; font-weight: bold; }
.sev-FAIL   { color: #f87171; }
.sev-WARN   { color: #fbbf24; }
.sev-INFO   { color: #34d399; }
.summary { background: #1e293b; border-radius: 0.5rem; padding: 1.5rem;
           display: flex; gap: 2rem; align-items: center; }
.score { font-size: 2.5rem; font-weight: bold; }
.grade { font-size: 1.2rem; }
.grade-PASS  { color: #34d399; }
.grade-WARN  { color: #fbbf24; }
.grade-FAIL  { color: #f87171; }
.grade-ERROR { color: #f87171; font-weight: bold; }
"""


def _grade_label(worst: Severity) -> tuple[str, str]:
    """Return (grade_str, css_class)."""
    if worst >= Severity.ERROR:
        return "ERROR", "grade-ERROR"
    if worst >= Severity.FAIL:
        return "FAIL — unsafe to train on", "grade-FAIL"
    if worst >= Severity.WARN:
        return "WARN — usable with caution", "grade-WARN"
    return "PASS", "grade-PASS"


def render_html(
    ref: str,
    version: DatasetVersion,
    num_episodes: int,
    num_frames: int | None,
    results: list[CheckResult],
) -> str:
    """Return a self-contained HTML document representing the lint report."""
    worst = max((r.severity for r in results), default=Severity.INFO)
    score = compute_trust_score(results)
    grade_label, grade_class = _grade_label(worst)

    counts: dict[Severity, int] = dict.fromkeys(Severity, 0)
    for r in results:
        counts[r.severity] += 1

    def e(s: str | int | None) -> str:
        return html.escape(str(s) if s is not None else "unknown")

    rows = "\n".join(
        f"<tr>"
        f"<td class='sev-{e(r.severity.value)}'>{e(r.severity.value)}</td>"
        f"<td>{e(r.check_id)}</td>"
        f"<td>{e(r.message)}</td>"
        f"</tr>"
        for r in results
    )

    count_rows = "\n".join(
        f"<tr><td class='sev-{e(sev.value)}'>{e(sev.value)}</td><td>{e(counts[sev])}</td></tr>"
        for sev in (Severity.FAIL, Severity.WARN, Severity.ERROR, Severity.INFO)
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>trajlens lint — {e(ref)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>trajlens lint: {e(ref)}</h1>
<p class="meta">
  Version: {e(version.value)} &nbsp;|&nbsp;
  Episodes: {e(num_episodes)} &nbsp;|&nbsp;
  Frames: {e(num_frames)}
</p>

<div class="summary">
  <div>
    <div class="score">{e(score)}<span style="font-size:1rem;color:#94a3b8">/100</span></div>
    <div style="color:#94a3b8;font-size:0.8rem">
      Trust score (formula v{e(SCORE_FORMULA_VERSION)})</div>
  </div>
  <div>
    <div class="grade {e(grade_class)}">{e(grade_label)}</div>
  </div>
</div>

<h2 style="margin-top:2rem">Check results</h2>
<table>
<tr>
  <th>Count</th><th>Severity</th>
</tr>
{count_rows}
</table>

<table>
<tr>
  <th>Severity</th><th>Check ID</th><th>Message</th>
</tr>
{rows}
</table>

</body>
</html>"""
