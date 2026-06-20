"""SARIF 2.1.0 renderer for lint results (02_ARCHITECTURE.md §3.4).

Produces a valid SARIF 2.1.0 JSON document per the OASIS schema at:
  https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1.0/schema/sarif-schema-2.1.0.json

GitHub Code Scanning requirements satisfied:
  - Top-level $schema, version ("2.1.0"), runs array.
  - tool.driver with name and version.
  - rules array (reportingDescriptor objects) — one per unique check_id seen.
  - results array with ruleId, level, message.text, and locations.

Location mapping: SARIF requires a physical location (file URI + line).
trajlens findings are dataset-level, not file:line findings.  We map each
result to the dataset root directory at line 1.  This is the correct idiom
for tool-level findings without source-code attribution and is accepted by
GitHub's code-scanning upload action.

Severity mapping to SARIF levels:
  ERROR -> "error"   (check could not run)
  FAIL  -> "error"   (unsafe to train on)
  WARN  -> "warning"
  INFO  -> "note"
"""

from __future__ import annotations

import json
from pathlib import Path

import trajlens
from trajlens.checks.protocol import CheckResult, Severity
from trajlens.report.trust_score import SCORE_FORMULA_VERSION, compute_trust_score
from trajlens.sources.version import DatasetVersion

_SARIF_SCHEMA = (
    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main"
    "/sarif-2.1.0/schema/sarif-schema-2.1.0.json"
)


def _sarif_level(severity: Severity) -> str:
    return {
        Severity.ERROR: "error",
        Severity.FAIL: "error",
        Severity.WARN: "warning",
        Severity.INFO: "note",
    }[severity]


def render_sarif(
    ref: str,
    version: DatasetVersion,
    num_episodes: int,
    num_frames: int | None,
    results: list[CheckResult],
) -> str:
    """Return a SARIF 2.1.0 JSON string for the lint results.

    The returned string is valid SARIF 2.1.0 and is accepted by the
    GitHub Code Scanning upload action (upload-sarif).
    """
    score = compute_trust_score(results)

    seen_rule_ids: list[str] = []
    for r in results:
        if r.check_id not in seen_rule_ids:
            seen_rule_ids.append(r.check_id)

    rules = [
        {
            "id": rule_id,
            "name": rule_id.replace(".", "_"),
            "shortDescription": {"text": rule_id},
        }
        for rule_id in seen_rule_ids
    ]

    # Dataset root URI for location mapping.  For Hub refs without a local
    # path, use "dataset://<ref>" as a synthetic URI so SARIF remains valid.
    dataset_path = Path(ref)
    if dataset_path.is_dir():
        location_uri = dataset_path.resolve().as_uri() + "/"
    else:
        location_uri = f"dataset://{ref}/"

    sarif_results = [
        {
            "ruleId": r.check_id,
            "level": _sarif_level(r.severity),
            "message": {"text": r.message},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": location_uri,
                            "uriBaseId": "%SRCROOT%",
                        },
                        "region": {"startLine": 1},
                    }
                }
            ],
        }
        for r in results
    ]

    document: dict[str, object] = {
        "$schema": _SARIF_SCHEMA,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "trajlens",
                        "version": trajlens.__version__,
                        "informationUri": "https://github.com/yourusername/trajlens",
                        "rules": rules,
                    }
                },
                "results": sarif_results,
                "properties": {
                    "dataset_ref": ref,
                    "dataset_version": version.value,
                    "num_episodes": num_episodes,
                    "num_frames": num_frames,
                    "trust_score": score,
                    "score_formula_version": SCORE_FORMULA_VERSION,
                },
            }
        ],
    }
    return json.dumps(document, indent=2)
