#!/usr/bin/env python3
"""Audit public LeRobot datasets on the Hugging Face Hub.

Pulls a sample of public datasets tagged 'lerobot' via the Hub search API,
runs ``trajlens lint --json`` on each in a subprocess with a per-dataset
timeout, and writes a machine-readable JSON results file and a human-readable
summary.

Security properties (06_SECURITY_AND_THREAT_MODEL.md):
- Each dataset runs in an isolated subprocess so a hostile or malformed
  dataset cannot crash or hang the audit process itself.
- No ``--deep`` flag is ever passed — full video is never downloaded.
- Timed-out datasets appear in results as ERROR (never silently dropped).
- All subprocess output is consumed via communicate() with the timeout;
  no shell=True, no string-built shell commands.

Usage::

    pip install "trajlens[hub]"
    python scripts/audit_hub.py --limit 100 --out audit_results.json

Run ``python scripts/audit_hub.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per-dataset subprocess timeout in seconds.  30 s matches the performance
# target stated in 01_VISION_AND_PRD.md §5 ("under 30 seconds on a laptop"
# for a 100-episode dataset); we give a 2x buffer (60 s) for network latency
# on the meta/ pull from the Hub, which is the only I/O in the non-deep path.
# Datasets that exceed this are classified as ERROR and counted, never dropped.
DATASET_TIMEOUT_SECONDS = 60

# Hub search tag filter (07_EVALUATION_AND_ACCURACY.md §4).
HUB_TAG = "lerobot"

# How much larger a pool to fetch (relative to --limit) before shuffling and
# truncating, so a random sample isn't just the API's fixed head-of-list.
_MAX_POOL_MULTIPLIER = 10
# Absolute cap on pool size regardless of --limit, to bound a single Hub call.
_MAX_POOL_SIZE = 2000

# Check IDs that map to known bug fingerprints (07 §4).
KNOWN_BUG_CHECKS = {
    "KNOWNBUG.TIMESTAMP_DRIFT": "#3177",
    "STRUCTURAL.METADATA_DATA_AGREEMENT": "#2401",
}

_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Hub dataset enumeration
# ---------------------------------------------------------------------------


def _list_hub_datasets(limit: int, token: str | None) -> list[str]:
    """Return up to *limit* public Hub dataset repo ids tagged with 'lerobot'.

    Uses the huggingface_hub list_datasets API (not the search API, which
    requires a different token scope) so it works with or without a token.

    The Hub API returns datasets in a fixed, deterministic order when no
    explicit ``sort`` is passed (confirmed empirically: repeated calls with
    identical args return identical order). Truncating that fixed order with
    ``limit`` alone means every run audits the same head-of-list datasets —
    not a random sample, but "whatever wave of uploads happens to be first in
    the API's ordering." We fetch a larger pool (capped at
    ``_MAX_POOL_MULTIPLIER`` x limit) and shuffle client-side before
    truncating, so repeated runs see a genuinely different cross-section.
    """
    try:
        from huggingface_hub import list_datasets
    except ImportError:
        print(
            "ERROR: huggingface_hub is not installed.\n"
            "Install it with: pip install 'trajlens[hub]'",
            file=sys.stderr,
        )
        sys.exit(1)

    pool_size = min(limit * _MAX_POOL_MULTIPLIER, _MAX_POOL_SIZE)
    pool = [ds.id for ds in list_datasets(filter=HUB_TAG, limit=pool_size, token=token)]
    random.shuffle(pool)
    return pool[:limit]


# ---------------------------------------------------------------------------
# Per-dataset lint
# ---------------------------------------------------------------------------


def _lint_dataset(repo_id: str, python: str, timeout: int) -> dict[str, Any]:
    """Run ``trajlens lint --json`` on *repo_id* in a subprocess.

    Returns a result dict with keys:
      repo_id, status (PASS|WARN|FAIL|ERROR|TIMEOUT),
      trust_score, results (list of check result dicts), error_message,
      duration_s.

    A dataset is NEVER silently dropped: timeouts and subprocess failures
    both produce an ERROR-class entry in the aggregate (ADR-003 spirit).
    """
    start = time.monotonic()
    result: dict[str, Any] = {
        "repo_id": repo_id,
        "status": "ERROR",
        "trust_score": None,
        "results": [],
        "error_message": None,
        "duration_s": None,
    }
    try:
        proc = subprocess.run(
            [python, "-m", "trajlens", "lint", "--json", repo_id],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.monotonic() - start
        result["duration_s"] = round(duration, 2)

        if proc.stdout.strip():
            try:
                lint_json = json.loads(proc.stdout)
                result["results"] = lint_json.get("results", [])
                result["trust_score"] = lint_json.get("trust_score")
                exit_code = proc.returncode
                grade = lint_json.get("grade")
                # Exit code 2 covers both a real check FAIL and a load-time
                # error (e.g. unsupported v2.x Hub format) -- the JSON
                # "grade" field is what actually distinguishes them; the CLI
                # sets grade="ERROR" with error_category/error_message and
                # an empty results list for the latter case.
                if grade == "ERROR" and lint_json.get("error_category"):
                    result["status"] = "ERROR"
                    result["error_message"] = lint_json.get("error_message")
                elif exit_code == 0:
                    result["status"] = "PASS"
                elif exit_code == 1:
                    result["status"] = "WARN"
                elif exit_code == 2:
                    result["status"] = "FAIL"
                else:
                    result["status"] = "ERROR"
                    result["error_message"] = (
                        f"unexpected exit code {exit_code}; stderr: {proc.stderr[:500]}"
                    )
            except json.JSONDecodeError:
                result["status"] = "ERROR"
                result["error_message"] = (
                    f"could not parse JSON output; stdout[:200]={proc.stdout[:200]!r}"
                )
        else:
            result["status"] = "ERROR"
            result["error_message"] = (
                f"empty stdout (exit {proc.returncode}); stderr: {proc.stderr[:500]}"
            )

    except subprocess.TimeoutExpired:
        duration = time.monotonic() - start
        result["duration_s"] = round(duration, 2)
        result["status"] = "TIMEOUT"
        result["error_message"] = f"exceeded {timeout}s timeout"

    except Exception as exc:
        duration = time.monotonic() - start
        result["duration_s"] = round(duration, 2)
        result["status"] = "ERROR"
        result["error_message"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute aggregate statistics from a list of per-dataset result dicts."""
    counts: dict[str, int] = {"PASS": 0, "WARN": 0, "FAIL": 0, "ERROR": 0, "TIMEOUT": 0}
    check_fire_counts: dict[str, int] = {}
    known_bug_counts: dict[str, int] = dict.fromkeys(KNOWN_BUG_CHECKS, 0)
    total_duration = 0.0

    for r in results:
        status = r["status"]
        counts[status] = counts.get(status, 0) + 1
        total_duration += r.get("duration_s") or 0.0

        for check_result in r.get("results", []):
            check_id = check_result.get("check_id", "")
            severity = check_result.get("severity", "")
            if severity in ("FAIL", "WARN", "ERROR"):
                check_fire_counts[check_id] = check_fire_counts.get(check_id, 0) + 1
            if check_id in known_bug_counts and severity == "FAIL":
                known_bug_counts[check_id] += 1

    total = len(results)
    non_error = counts["PASS"] + counts["WARN"] + counts["FAIL"]
    fail_or_worse = counts["FAIL"] + counts["ERROR"] + counts["TIMEOUT"]

    top_checks = sorted(check_fire_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]

    return {
        "total_datasets": total,
        "counts": counts,
        "pct_fail_or_worse": round(100 * fail_or_worse / total, 1) if total else 0,
        "pct_any_issue": (round(100 * (total - counts["PASS"]) / total, 1) if total else 0),
        "top_checks_by_fire_count": [{"check_id": k, "fire_count": v} for k, v in top_checks],
        "known_bug_prevalence": {
            check_id: {
                "bug_ref": bug_ref,
                "fire_count": known_bug_counts[check_id],
                "pct_of_linted": (
                    round(100 * known_bug_counts[check_id] / non_error, 1) if non_error else 0
                ),
            }
            for check_id, bug_ref in KNOWN_BUG_CHECKS.items()
        },
        "mean_duration_s": round(total_duration / total, 2) if total else 0,
        "datasets_exceeding_timeout": counts["TIMEOUT"],
    }


# ---------------------------------------------------------------------------
# Summary renderer
# ---------------------------------------------------------------------------


def _render_summary(agg: dict[str, Any], repo_ids: list[str]) -> str:
    lines = [
        "=" * 70,
        "  trajlens Hub Audit Summary",
        f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",  # noqa: UP017
        "=" * 70,
        "",
        f"  Datasets audited : {agg['total_datasets']}",
        f"  PASS             : {agg['counts']['PASS']}",
        f"  WARN             : {agg['counts']['WARN']}",
        f"  FAIL             : {agg['counts']['FAIL']}",
        f"  ERROR/TIMEOUT    : {agg['counts'].get('ERROR', 0) + agg['counts'].get('TIMEOUT', 0)}",
        "",
        f"  Datasets with any issue : {agg['pct_any_issue']}%",
        f"  Datasets FAIL or worse  : {agg['pct_fail_or_worse']}%",
        f"  Mean lint duration      : {agg['mean_duration_s']}s",
        "",
        "  Known bug prevalence:",
    ]
    for check_id, info in agg["known_bug_prevalence"].items():
        lines.append(
            f"    {check_id} ({info['bug_ref']}): "
            f"{info['fire_count']} datasets "
            f"({info['pct_of_linted']}% of successfully linted)"
        )
    lines.append("")
    lines.append("  Top checks by fire count (FAIL/WARN/ERROR):")
    for entry in agg["top_checks_by_fire_count"]:
        lines.append(f"    {entry['check_id']:<45} {entry['fire_count']}")
    lines += ["", "=" * 70]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="audit_hub",
        description=(
            "Audit public LeRobot datasets on the Hugging Face Hub.\n\n"
            "Requires: pip install 'trajlens[hub]'\n\n"
            "This script pulls a sample of public datasets tagged 'lerobot', "
            "runs 'trajlens lint --json' on each in a subprocess with a "
            f"{DATASET_TIMEOUT_SECONDS}s timeout, and writes aggregate results."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        metavar="N",
        help="Number of datasets to audit (default: 100).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("audit_results.json"),
        metavar="PATH",
        help="Path to write the machine-readable JSON results (default: audit_results.json).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to write the human-readable summary (default: stdout only).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        metavar="HF_TOKEN",
        help=(
            "Hugging Face API token for higher rate limits. "
            "Defaults to the token from 'huggingface-cli login' if present. "
            "Never logged or written to disk by this script."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DATASET_TIMEOUT_SECONDS,
        metavar="SECS",
        help=f"Per-dataset subprocess timeout in seconds (default: {DATASET_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        metavar="PATH",
        help=(
            "Python interpreter to use for the lint subprocesses "
            "(default: the interpreter running this script). "
            "Must have trajlens installed."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"trajlens audit_hub v{_VERSION}")
    print(f"Fetching up to {args.limit} public datasets tagged '{HUB_TAG}' …")
    repo_ids = _list_hub_datasets(limit=args.limit, token=args.token)
    print(f"Found {len(repo_ids)} datasets. Starting lint …\n")

    all_results: list[dict[str, Any]] = []
    for i, repo_id in enumerate(repo_ids, start=1):
        print(f"  [{i:>3}/{len(repo_ids)}] {repo_id}", end=" … ", flush=True)
        result = _lint_dataset(repo_id, python=args.python, timeout=args.timeout)
        all_results.append(result)
        status_str = result["status"]
        duration = result.get("duration_s")
        dur_str = f"{duration:.1f}s" if duration is not None else "?"
        print(f"{status_str} ({dur_str})")

    agg = _aggregate(all_results)
    summary = _render_summary(agg, repo_ids)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "trajlens_version": _VERSION,
        "hub_tag_filter": HUB_TAG,
        "timeout_seconds": args.timeout,
        "aggregate": agg,
        "datasets": all_results,
    }

    args.out.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to: {args.out}")

    print("\n" + summary)

    if args.summary is not None:
        args.summary.write_text(summary)
        print(f"Summary written to: {args.summary}")


if __name__ == "__main__":
    main()
