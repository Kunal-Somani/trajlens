"""BaselineStore — read/write/diff `.trajlens-baseline.json` (adoption unlock).

A baseline snapshots the current findings for a dataset so CI can fail only
on *new* findings, not on the pre-existing backlog. The baseline file is
user-controlled and committed to their repo, so it is a trust boundary
(06_SECURITY_AND_THREAT_MODEL.md): validated via Pydantic before any value is
acted on, and a malformed file fails closed with DatasetFormatError rather
than crashing or silently producing a clean result.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from trajlens.checks.protocol import CheckResult, Severity
from trajlens.errors import DatasetFormatError

BASELINE_SCHEMA_VERSION = "2"

# The identity key is (check_id, episode_index, shard_path). It is what makes
# a finding "the same" across two lint runs so an unchanged dataset doesn't
# re-report existing findings as new. This tuple is a versioned contract:
# changing which fields participate in identity changes what counts as
# "the same finding," so any change to it must bump BASELINE_SCHEMA_VERSION
# rather than silently reinterpreting old baseline files.
#
# Identity alone is deliberately NOT sufficient to call a finding
# "unchanged". No check populates details["shard_path"], and only three
# populate per_episode, so for most checks the identity collapses to
# (check_id, None, None) -- meaning a baseline entry would suppress that
# check entirely, no matter how much worse the dataset got. A dataset whose
# SCHEMA_CONSISTENCY findings went from 1 to 500, or whose check escalated
# from WARN to FAIL, would still have reported "unchanged" and passed CI.
#
# So severity and a magnitude (finding_count) are persisted alongside the
# identity and compared explicitly on diff: a baseline records how much of a
# problem was accepted, and anything worse than that is surfaced as new.
# Schema bumped 1 -> 2 for these two added fields; BaselineStore.load()
# already rejects a mismatched version with an actionable "regenerate with
# --update-baseline" message.
IdentityKey = tuple[str, int | None, str | None]

# Severity rank for regression comparison. Mirrors checks/protocol.py's
# _SEVERITY_ORDER; duplicated as a plain str->int map rather than imported so
# a persisted baseline is compared against stable strings, not against enum
# identity that a future reordering could silently change the meaning of.
_SEVERITY_RANK: dict[str, int] = {"INFO": 0, "WARN": 1, "FAIL": 2, "ERROR": 3}


class FindingKey(BaseModel):
    """A single finding's identity and accepted magnitude, as persisted.

    severity/finding_count are not part of identity() -- they are the
    "how bad was it when accepted" attributes compared against the current
    run to decide whether an identified finding has regressed.
    """

    check_id: str
    episode_index: int | None = None
    shard_path: str | None = None
    severity: str = "FAIL"
    finding_count: int = 1

    def identity(self) -> IdentityKey:
        return (self.check_id, self.episode_index, self.shard_path)


class BaselineFile(BaseModel):
    """On-disk schema for `.trajlens-baseline.json`."""

    schema_version: str
    findings: list[FindingKey]


class BaselineDiff(BaseModel):
    """Result of comparing current findings against a loaded baseline."""

    model_config = {"arbitrary_types_allowed": True}

    new: list[CheckResult]
    resolved: list[FindingKey]
    unchanged: list[CheckResult]


def _result_identity(result: CheckResult) -> IdentityKey:
    episode_index: int | None = None
    shard_path: str | None = None
    if result.per_episode:
        # The lowest-indexed affected episode, as a stable discriminator.
        # min() rather than next(iter(...)) so identity does not depend on
        # dict insertion order -- which is the order the check happened to
        # scan in, and therefore not guaranteed stable across releases.
        episode_index = min(result.per_episode)
    details_shard = result.details.get("shard_path")
    if isinstance(details_shard, str):
        shard_path = details_shard
    return (result.check_id, episode_index, shard_path)


def _result_finding_count(result: CheckResult) -> int:
    """Return the magnitude of *result* -- how much of this problem exists.

    Prefers the check's own exact total where it reports one (the three
    per-episode-capable checks record details["total_violations"], which is
    an exact count independent of any display cap). Falls back to how many
    episodes the finding touches, then to 1 for a check that reports no
    quantitative signal at all -- for those, a baseline still cannot detect
    a worsening, which is a limit of the check's own reporting rather than
    of the baseline, and is honest: a magnitude is never invented here.
    """
    total = result.details.get("total_violations")
    if isinstance(total, int):
        return total
    if result.per_episode:
        return len(result.per_episode)
    return 1


def _has_regressed(current: CheckResult, accepted: FindingKey) -> bool:
    """True if *current* is worse than what the baseline *accepted*.

    Worse means either escalated in severity (e.g. WARN -> FAIL) or grown in
    magnitude. A finding that shrank or softened is still "unchanged" -- a
    baseline is a ceiling on accepted badness, not an exact snapshot.
    """
    current_rank = _SEVERITY_RANK.get(current.severity.value, 0)
    accepted_rank = _SEVERITY_RANK.get(accepted.severity, 0)
    if current_rank > accepted_rank:
        return True
    return _result_finding_count(current) > accepted.finding_count


class BaselineStore:
    """Loads, saves, and diffs a `.trajlens-baseline.json` file."""

    def __init__(self, findings: list[FindingKey]) -> None:
        self._findings = findings

    @property
    def findings(self) -> list[FindingKey]:
        return self._findings

    @classmethod
    def load(cls, path: Path) -> BaselineStore:
        """Load and validate a baseline file.

        Raises DatasetFormatError if the file is missing, not valid JSON,
        has a mismatched schema_version, or is missing required fields.
        Never crashes on untrusted input.
        """
        if not path.is_file():
            raise DatasetFormatError(f"baseline file not found: {path}")

        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise DatasetFormatError(f"baseline file is not valid JSON: {path}: {exc}") from exc

        raw_version = raw.get("schema_version") if isinstance(raw, dict) else None
        if raw_version != BASELINE_SCHEMA_VERSION:
            raise DatasetFormatError(
                f"baseline file schema_version mismatch: file has "
                f"{raw_version!r}, this version of trajlens requires "
                f"{BASELINE_SCHEMA_VERSION!r}. Regenerate the baseline with "
                f"--update-baseline."
            )

        try:
            parsed = BaselineFile.model_validate(raw)
        except ValidationError as exc:
            raise DatasetFormatError(
                f"baseline file does not match the expected schema: {path}: {exc}"
            ) from exc

        return cls(parsed.findings)

    def save(self, path: Path) -> None:
        """Write this store's findings to *path* as schema_version-tagged JSON."""
        payload = BaselineFile(schema_version=BASELINE_SCHEMA_VERSION, findings=self._findings)
        path.write_text(json.dumps(payload.model_dump(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_results(cls, results: list[CheckResult]) -> BaselineStore:
        """Snapshot the actionable findings in *results*, with their magnitude.

        Only WARN-and-above results are recorded. A baseline exists to say
        "these problems are known and accepted"; an INFO result is a check
        that passed, and recording it made a later run report the check
        going green as a *new finding* (its identity changes once the
        per-episode discriminator disappears). Filtering here and in diff()
        keeps both sides talking about problems only.
        """
        findings = [
            FindingKey(
                check_id=r.check_id,
                episode_index=_result_identity(r)[1],
                shard_path=_result_identity(r)[2],
                severity=r.severity.value,
                finding_count=_result_finding_count(r),
            )
            for r in results
            if r.severity >= Severity.WARN
        ]
        return cls(findings)

    def diff(self, current: list[CheckResult]) -> BaselineDiff:
        """Compare *current* results against this baseline.

        new       — not in the baseline at all, OR present but now worse than
                    the baseline accepted (escalated severity, or a larger
                    finding count). A regression must surface: suppressing it
                    is how a baselined check silently hid a dataset getting
                    dramatically worse.
        resolved  — in baseline, no longer reported.
        unchanged — present and no worse than the baseline accepted.

        Only WARN+ results participate, matching from_results().
        """
        actionable = [r for r in current if r.severity >= Severity.WARN]
        baseline_by_identity = {f.identity(): f for f in self._findings}
        current_identities = {_result_identity(r) for r in actionable}

        new: list[CheckResult] = []
        unchanged: list[CheckResult] = []
        for result in actionable:
            accepted = baseline_by_identity.get(_result_identity(result))
            if accepted is None or _has_regressed(result, accepted):
                new.append(result)
            else:
                unchanged.append(result)

        resolved = [f for f in self._findings if f.identity() not in current_identities]

        return BaselineDiff(new=new, resolved=resolved, unchanged=unchanged)
