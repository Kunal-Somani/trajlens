"""HTML renderer for lint results (--report path.html).

Self-contained: all CSS is inlined, no external assets. Safe for standalone
viewing without a web server. Content is escaped via html.escape() — the
dataset ref and messages are untrusted strings that must not inject HTML.
"""

from __future__ import annotations

import html

from trajlens.checks.protocol import CheckResult, Severity
from trajlens.repair.orchestrator import FixPlan, RepairPlan
from trajlens.report.trust_score import SCORE_FORMULA_VERSION, compute_trust_score, integrity_only

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
.fix-noop   { color: #34d399; }
.fix-change { color: #fbbf24; }
.fix-detected-not-writable { color: #94a3b8; font-style: italic; }
.fix-no-fixer { color: #94a3b8; }
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
    format_id: str,
    format_version: str,
    num_episodes: int,
    num_frames: int | None,
    results: list[CheckResult],
) -> str:
    """Return a self-contained HTML document representing the lint report."""
    worst = max((r.severity for r in integrity_only(results)), default=Severity.INFO)
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
  Format: {e(format_id)} v{e(format_version)} &nbsp;|&nbsp;
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


def render_fix_html(
    ref: str,
    plan: FixPlan,
    *,
    repair_plan: RepairPlan | None = None,
    format_id: str | None = None,
) -> str:
    """Return a self-contained HTML document representing a `trajlens fix` report.

    Same three-state rendering as report/fix_report.py's terminal renderer,
    styled consistently with render_html: REPAIRABLE outcomes (existing
    plan.outcomes diff), then DETECTED_NOT_WRITABLE, then NO_FIXER --
    repair_plan/format_id default to None so the schema degrades gracefully
    to just the REPAIRABLE section when no RepairPlan was computed.
    """

    def e(s: str | int | None) -> str:
        return html.escape(str(s) if s is not None else "unknown")

    mode = "apply" if plan.applied else "dry-run"

    if not plan.outcomes:
        outcomes_html = "<p>No applicable fixers — nothing to fix.</p>"
    elif not plan.applicable:
        outcomes_html = (
            "<p>All applicable fixers report the dataset is already clean. Nothing to fix.</p>"
        )
    else:
        rows = []
        for outcome in plan.outcomes:
            if outcome.diff.is_noop:
                rows.append(
                    f"<li class='fix-noop'>OK — {e(outcome.fixer_id)} ({e(outcome.check_id)}): "
                    "no changes needed</li>"
                )
                continue
            verb = "applied" if outcome.summary is not None else "would change"
            rows.append(
                f"<li class='fix-change'>CHANGE — {e(outcome.fixer_id)} ({e(outcome.check_id)}): "
                f"{e(len(outcome.diff.changes))} change(s) {e(verb)}</li>"
            )
        outcomes_html = f"<ul>{''.join(rows)}</ul>"
        if plan.applied:
            outcomes_html += f"<p>Repaired dataset written to: {e(str(plan.output_path))}</p>"
        else:
            outcomes_html += "<p>Dry run only — nothing was written.</p>"

    detected_html = ""
    no_fixer_html = ""
    if repair_plan is not None:
        if repair_plan.detected_not_writable:
            items = "".join(
                f"<li class='fix-detected-not-writable'>{e(check_id)}: detected — not "
                f"auto-repairable in {e(format_id)} yet</li>"
                for check_id in repair_plan.detected_not_writable
            )
            detected_html = f"<h2>Detected, not auto-repairable</h2><ul>{items}</ul>"
        if repair_plan.no_fixer:
            items = "".join(
                f"<li class='fix-no-fixer'>{e(check_id)}: no fixer exists for this finding</li>"
                for check_id in repair_plan.no_fixer
            )
            no_fixer_html = f"<h2>No fixer available</h2><ul>{items}</ul>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>trajlens fix — {e(ref)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>trajlens fix: {e(ref)}</h1>
<p class="meta">mode: {e(mode)}</p>

<h2>Repairable</h2>
{outcomes_html}
{detected_html}
{no_fixer_html}

</body>
</html>"""
