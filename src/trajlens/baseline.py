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

from trajlens.checks.protocol import CheckResult
from trajlens.errors import DatasetFormatError

BASELINE_SCHEMA_VERSION = "1"

# The identity key is (check_id, episode_index, shard_path). It is what makes
# a finding "the same" across two lint runs so an unchanged dataset doesn't
# re-report existing findings as new. This tuple is a versioned contract:
# changing which fields participate in identity changes what counts as
# "the same finding," so any change to it must bump BASELINE_SCHEMA_VERSION
# rather than silently reinterpreting old baseline files.
IdentityKey = tuple[str, int | None, str | None]


class FindingKey(BaseModel):
    """A single finding's identity, as persisted in the baseline file."""

    check_id: str
    episode_index: int | None = None
    shard_path: str | None = None

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
        episode_index = next(iter(result.per_episode))
    details_shard = result.details.get("shard_path")
    if isinstance(details_shard, str):
        shard_path = details_shard
    return (result.check_id, episode_index, shard_path)


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
        """Build a store snapshotting exactly the given results, no more/less."""
        findings = [
            FindingKey(
                check_id=key[0],
                episode_index=key[1],
                shard_path=key[2],
            )
            for key in (_result_identity(r) for r in results)
        ]
        return cls(findings)

    def diff(self, current: list[CheckResult]) -> BaselineDiff:
        """Compare *current* results against this baseline.

        new       — in current, not in baseline.
        resolved  — in baseline, not in current.
        unchanged — in both.
        """
        baseline_identities = {f.identity() for f in self._findings}
        current_identities = {_result_identity(r) for r in current}

        new = [r for r in current if _result_identity(r) not in baseline_identities]
        unchanged = [r for r in current if _result_identity(r) in baseline_identities]
        resolved = [f for f in self._findings if f.identity() not in current_identities]

        return BaselineDiff(new=new, resolved=resolved, unchanged=unchanged)
