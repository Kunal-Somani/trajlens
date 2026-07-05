"""Fixer orchestration for `trajlens fix` (ADR-001, ADR-004).

This module is the library-level "which fixers apply, in what order, and how
do their copy-on-write outputs chain" logic.  The CLI (cli.py) is a thin
shell over run_dry_run()/run_apply(); no repair or selection logic lives
there, per ADR-001 (library-first).

Fixer selection: a fixer is "applicable" when its check_id appears in the
lint results at severity WARN or above -- the same threshold the fixers'
own round-trip tests already use to mean "this finding is present."

Composition order and the copy-chaining design (read before changing either):

Every Fixer.apply(ds, output_path) takes a CanonicalDataset bound to its own
source root and unconditionally copies that ENTIRE source tree to
output_path before rewriting its narrow slice (ADR-004 copy-on-write). There
is no "operate in place on an already-copied tree" mode in the Fixer
Protocol -- and there cannot safely be one without changing the protocol,
because apply() also refuses output_path == source root precisely so a
fixer can never be pointed at something that could be mistaken for the
original (see each fixer's "must not be the source dataset root" guard).

Given that contract, chaining N applicable fixers means N full tree copies,
each into its own temporary directory, with only the LAST fixer's output
landing at the user's requested --out:

    source --[fixer_1.apply]--> tmp_1 --[fixer_2.apply]--> tmp_2 --> ... --> out

Each tmp_i is reloaded as a fresh CanonicalDataset before being handed to the
next fixer -- a fixer only ever sees a CanonicalDataset whose declared
metadata matches the bytes actually on disk at that path, never a stale view
of a tree another fixer has since rewritten out from under it. This is the
only design consistent with the existing Fixer Protocol's guarantees; a
"one copy, rewrite in place" design would require changing apply()'s
signature or relaxing its source-root guard, which is a protocol change all
three fixers depend on, not a CLI detail (flagged in the PR, not made
silently).

Fixed fixer order -- episode_reindex, then timestamp_dedrift, then
stats_recompute -- chosen because stats_recompute streams data column
values (including timestamp) to recompute mean/std/min/max
(checks/statistical.py _stream_feature_columns), so it must run AFTER
timestamp_dedrift or it would compute "correct" statistics over still-wrong
timestamps. episode_reindex has no such dependency: _stream_feature_columns
filters rows by the row-level episode_index column directly, never by
dataset_from_index/dataset_to_index, so reindexing does not change what
stats_recompute streams. episode_reindex is placed first only because
"structural boundaries agree with data" is the more fundamental invariant to
restore first; timestamp_dedrift and episode_reindex have no dependency on
each other in either order.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from trajlens.checks.protocol import CheckResult, Severity
from trajlens.errors import RepairError
from trajlens.model import CanonicalDataset, build_canonical_dataset
from trajlens.repair.episode_reindex import EpisodeReindexFixer
from trajlens.repair.protocol import Diff, Fixer, RepairSummary
from trajlens.repair.stats_recompute import StatsRecomputeFixer
from trajlens.repair.timestamp_dedrift import TimestampDedriftFixer
from trajlens.sources.loader import SourceHandle, SourceLoader

# Fixed application order -- see module docstring for why this order and not
# applicable-fixer-discovery order.
_FIXER_ORDER: tuple[Fixer, ...] = (
    EpisodeReindexFixer(),
    TimestampDedriftFixer(),
    StatsRecomputeFixer(),
)


@dataclass(frozen=True, slots=True)
class FixerOutcome:
    """One applicable fixer's result within a fix run.

    fixer_id / check_id  — identify the fixer and the finding it targets.
    diff                 — the Diff computed by dry_run() (always populated,
                           even when --apply also ran, since apply() calls
                           dry_run() internally and this mirrors that Diff).
    summary              — set only when --apply ran; None for dry-run.
    """

    fixer_id: str
    check_id: str
    diff: Diff
    summary: RepairSummary | None


@dataclass(frozen=True, slots=True)
class FixPlan:
    """The result of a `trajlens fix` run: which fixers applied and what happened.

    applicable       — True if at least one fixer had a non-noop Diff.
    outcomes         — one FixerOutcome per applicable fixer, in application order.
    output_path      — final --out path (only meaningful when applied=True).
    applied          — True if --apply was requested and fixers actually ran apply().
    """

    applicable: bool
    outcomes: tuple[FixerOutcome, ...]
    output_path: Path | None
    applied: bool


def select_applicable_fixers(results: list[CheckResult]) -> list[Fixer]:
    """Return the fixers (in fixed application order) whose check fired at WARN+.

    Mirrors the threshold the fixers' own round-trip tests use for "this
    finding is present": severity >= Severity.WARN.
    """
    fired_check_ids = {r.check_id for r in results if r.severity >= Severity.WARN}
    return [fixer for fixer in _FIXER_ORDER if fixer.check_id in fired_check_ids]


def run_dry_run(ds: CanonicalDataset, results: list[CheckResult]) -> FixPlan:
    """Compute the combined Diff for every applicable fixer; writes nothing.

    Each fixer's dry_run() is called against the ORIGINAL ds -- dry-run never
    chains, because chaining is only meaningful when fixers actually write
    (a later fixer's dry_run() against an unwritten hypothetical output would
    just be re-deriving the same Diff against the same unmodified source).
    A RepairError from any fixer propagates to the caller (the CLI renders it
    as a clean user-facing message, per the manual's error rule).
    """
    fixers = select_applicable_fixers(results)
    outcomes = tuple(
        FixerOutcome(
            fixer_id=fixer.fixer_id,
            check_id=fixer.check_id,
            diff=fixer.dry_run(ds),
            summary=None,
        )
        for fixer in fixers
    )
    applicable = any(not o.diff.is_noop for o in outcomes)
    return FixPlan(applicable=applicable, outcomes=outcomes, output_path=None, applied=False)


def run_apply(ds: CanonicalDataset, results: list[CheckResult], output_path: Path) -> FixPlan:
    """Apply every applicable fixer in sequence, chaining through temp copies.

    See the module docstring for the full composition-order rationale. Each
    fixer in turn: reads its Diff via dry_run() against the current dataset
    view, skips (no copy, no temp dir) if it is a no-op, otherwise apply()s
    into a fresh temp directory and the result is reloaded as the next
    fixer's input. The LAST fixer to actually write lands its output at
    *output_path* directly, so a fully clean dataset (all fixers no-op) still
    ends with a plain copy at *output_path* -- 03_DATA_FORMAT_SPEC.md's
    round-trip contract that "no findings" means the output is identical to
    the source, never a partially-written or missing tree.

    Raises RepairError immediately (writing nothing further, and cleaning up
    its own temp directories) if any fixer refuses to repair -- fail-closed,
    per 06_SECURITY_AND_THREAT_MODEL.md T9. Temp directories from fixers
    that already ran successfully before the failure are still cleaned up.
    """
    fixers = select_applicable_fixers(results)
    # Diffs must be computed up front (each against the ORIGINAL ds, before any
    # fixer has written anything) purely to know which fixer is the LAST one
    # that will actually write -- so that fixer's apply() can target
    # output_path directly instead of a temp dir that then has to be copied
    # again. The loop below re-derives each fixer's real diff against
    # current_ds (necessary once the chain has actually written something);
    # for the first fixer this duplicates one dry_run() call against an
    # unmodified ds, accepted for a clear, non-special-cased loop body.
    upfront_diffs = [fixer.dry_run(ds) for fixer in fixers]
    last_writing_index = max((i for i, d in enumerate(upfront_diffs) if not d.is_noop), default=-1)

    outcomes: list[FixerOutcome] = []
    current_ds = ds
    temp_dirs: list[str] = []

    try:
        for i, fixer in enumerate(fixers):
            diff = fixer.dry_run(current_ds)

            if diff.is_noop:
                outcomes.append(
                    FixerOutcome(
                        fixer_id=fixer.fixer_id, check_id=fixer.check_id, diff=diff, summary=None
                    )
                )
                continue

            if i == last_writing_index:
                target = output_path
            else:
                tmp_dir = tempfile.mkdtemp(prefix="trajlens-fix-")
                temp_dirs.append(tmp_dir)
                target = Path(tmp_dir) / "dataset"

            summary = fixer.apply(current_ds, target)
            outcomes.append(
                FixerOutcome(
                    fixer_id=fixer.fixer_id, check_id=fixer.check_id, diff=diff, summary=summary
                )
            )
            current_ds = _reload(target)

        # No applicable fixer ever wrote (either none are applicable, or a
        # fixer's diff against the chained ds turned out noop even though its
        # diff against the original ds was not, e.g. an earlier fixer already
        # incidentally fixed it): output_path must still receive a plain copy
        # so --apply always produces a complete dataset at --out.
        if not output_path.exists():
            shutil.copytree(_dataset_root(current_ds), output_path)

        applicable = any(not o.diff.is_noop for o in outcomes)
        return FixPlan(
            applicable=applicable, outcomes=tuple(outcomes), output_path=output_path, applied=True
        )
    finally:
        for tmp_dir in temp_dirs:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _dataset_root(ds: CanonicalDataset) -> Path:
    return ds.stats.root


def _reload(root: Path) -> CanonicalDataset:
    """Reload *root* as a fresh CanonicalDataset for the next fixer in the chain.

    Always a local path (chaining only ever happens through temp directories
    or --out, never a Hub ref), so SourceLoader resolves it as a plain local
    directory with no network access.
    """
    handle = SourceLoader().resolve(str(root))
    return build_canonical_dataset(handle)


def refuse_if_hub_ref(handle: SourceHandle, ref: str) -> None:
    """Raise RepairError if *ref* resolved to a Hub dataset, not a local path.

    Fixer.apply() unconditionally shutil.copytree()s the ENTIRE source root.
    For a Hub ref, SourceHandle.root is a local cache directory that only
    ever contains meta/** (sources/loader.py _resolve_hub uses
    allow_patterns=["meta/**"]) -- data/ and videos/ shards are streamed
    per-shard over HTTP and are never present under root at all. Copying
    that root would silently produce a "repaired" dataset missing all of its
    data and video shards. trajlens fix only operates on local datasets;
    for a Hub ref, download it locally first (this is the same restriction
    v2.x Hub datasets already have for a different reason -- see
    sources/loader.py _build_v2 guard).
    """
    if handle.repo_id is not None:
        raise RepairError(
            f"{ref!r} resolved to a Hugging Face Hub dataset. `trajlens fix` only "
            "repairs local datasets, because a Hub ref's data/video shards are "
            "streamed on demand and never fully present on disk to copy. "
            "Download the dataset locally first (e.g. with `huggingface-cli download` "
            "or `snapshot_download`), then run `trajlens fix` against the local path."
        )
