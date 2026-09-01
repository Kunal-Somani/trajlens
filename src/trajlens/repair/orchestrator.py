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

Fixed fixer order -- episode_reindex, timestamp_dedrift, stats_recompute,
task_index_repair, video_metadata_sync, orphan_shard_report -- chosen
because stats_recompute streams data column values (including timestamp)
to recompute mean/std/min/max (checks/statistical.py
_stream_feature_columns), so it must run AFTER timestamp_dedrift or it
would compute "correct" statistics over still-wrong timestamps.
episode_reindex has no such dependency: _stream_feature_columns filters
rows by the row-level episode_index column directly, never by
dataset_from_index/dataset_to_index, so reindexing does not change what
stats_recompute streams. episode_reindex is placed first only because
"structural boundaries agree with data" is the more fundamental invariant to
restore first; timestamp_dedrift and episode_reindex have no dependency on
each other in either order.

task_index_repair (rewrites task_index only), video_metadata_sync
(rewrites meta/info.json's fps only), and orphan_shard_report (only ever
moves already-unreferenced files) are appended after the original three:
none of them read or write a column or field any other fixer touches, so
they have no ordering dependency on the first three or each other --
placed last simply because they were added later, not because order matters
for them.

Note: video_metadata_sync's target check, VIDEO.RESOLUTION_FPS_MATCH, is
not yet implemented (checks/video.py defers it to a later milestone), so
this fixer can never be selected automatically by select_applicable_fixers's
WARN+-in-results threshold -- it is only reachable via explicit --only
selection today. This is a known, pre-existing gap in the check suite, not
a bug in fixer selection.
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
from trajlens.repair.orphan_shard_report import OrphanShardReportFixer
from trajlens.repair.protocol import Diff, Fixer, RepairSummary
from trajlens.repair.stats_recompute import StatsRecomputeFixer
from trajlens.repair.task_index_repair import TaskIndexRepairFixer
from trajlens.repair.timestamp_dedrift import TimestampDedriftFixer
from trajlens.repair.video_metadata_sync import VideoMetadataSyncFixer
from trajlens.sources.loader import SourceHandle, SourceLoader

# Fixed application order -- see module docstring for why this order and not
# applicable-fixer-discovery order. quarantine=False here is the always-safe
# default; a caller wanting --quarantine passes its own OrphanShardReportFixer
# instance to select_applicable_fixers instead of relying on this tuple (see
# build_fixer_order()).
_FIXER_ORDER: tuple[Fixer, ...] = (
    EpisodeReindexFixer(),
    TimestampDedriftFixer(),
    StatsRecomputeFixer(),
    TaskIndexRepairFixer(),
    VideoMetadataSyncFixer(),
    OrphanShardReportFixer(),
)

ALL_FIXER_IDS: tuple[str, ...] = tuple(fixer.fixer_id for fixer in _FIXER_ORDER)


def build_fixer_order(*, quarantine: bool = False) -> tuple[Fixer, ...]:
    """Return the fixed fixer order, substituting a quarantine-enabled
    OrphanShardReportFixer when *quarantine* is True.

    A separate function rather than a mutable module-level tuple: _FIXER_ORDER
    must stay a stable, side-effect-free constant (every other caller,
    including tests, relies on it never changing), so a caller that wants
    --quarantine builds its own copy of the order instead.
    """
    if not quarantine:
        return _FIXER_ORDER
    return tuple(
        OrphanShardReportFixer(quarantine=True)
        if isinstance(fixer, OrphanShardReportFixer)
        else fixer
        for fixer in _FIXER_ORDER
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


@dataclass(frozen=True, slots=True)
class PlannedFix:
    """One finding classified REPAIRABLE by plan(): a fixer exists and can write ds.format_id.

    fixer_id / check_id — identify the fixer and the finding it targets.
    diff                 — the Diff computed by dry_run() against ds, so
                           report renderers can show the planned change
                           without a second pass over the dataset.
    """

    fixer_id: str
    check_id: str
    diff: Diff


@dataclass(frozen=True, slots=True)
class RepairPlan:
    """Three-way classification of every WARN+ finding in a lint run (M1-D).

    repairable             — a fixer exists and can write ds.format_id.
    detected_not_writable  — check_ids with a fixer whose writable_formats
                              excludes ds.format_id.
    no_fixer                — check_ids with no fixer targeting them at all.

    See repair/protocol.py's Repairability enum for the same three states.
    """

    repairable: tuple[PlannedFix, ...]
    detected_not_writable: tuple[str, ...]
    no_fixer: tuple[str, ...]


def plan(
    ds: CanonicalDataset,
    results: list[CheckResult],
    *,
    fixer_order: tuple[Fixer, ...] = _FIXER_ORDER,
) -> RepairPlan:
    """Classify every WARN+ finding as REPAIRABLE, DETECTED_NOT_WRITABLE, or NO_FIXER.

    Mirrors select_applicable_fixers's WARN+ threshold for "this finding is
    present", then additionally checks each matching fixer's writable_formats
    against ds.format_id -- the distinction select_applicable_fixers itself
    does not make (it is only ever called today against ds's own format,
    which happens to always be writable, so this check has been silently
    absent from fixer selection until now).
    """
    fixers_by_check_id = {fixer.check_id: fixer for fixer in fixer_order}
    fired_check_ids = sorted({r.check_id for r in results if r.severity >= Severity.WARN})

    repairable: list[PlannedFix] = []
    detected_not_writable: list[str] = []
    no_fixer: list[str] = []

    for check_id in fired_check_ids:
        fixer = fixers_by_check_id.get(check_id)
        if fixer is None:
            no_fixer.append(check_id)
        elif ds.format_id in fixer.writable_formats:
            repairable.append(
                PlannedFix(fixer_id=fixer.fixer_id, check_id=check_id, diff=fixer.dry_run(ds))
            )
        else:
            detected_not_writable.append(check_id)

    return RepairPlan(
        repairable=tuple(repairable),
        detected_not_writable=tuple(detected_not_writable),
        no_fixer=tuple(no_fixer),
    )


def select_applicable_fixers(
    results: list[CheckResult],
    *,
    fixer_order: tuple[Fixer, ...] = _FIXER_ORDER,
    only: frozenset[str] | None = None,
    except_: frozenset[str] | None = None,
) -> list[Fixer]:
    """Return the fixers (in *fixer_order*) to run for this `fix` invocation.

    Default selection (only/except_ both None or empty): every fixer whose
    check fired at WARN+ -- mirrors the threshold the fixers' own round-trip
    tests use for "this finding is present".

    *only*, if non-empty, replaces the WARN+ threshold entirely: exactly the
    named fixer ids run, regardless of whether their check fired. This is the
    only way to reach a fixer whose target check is not yet implemented (e.g.
    REPAIR.VIDEO_METADATA_SYNC -- see module docstring) or to preview a
    specific fixer's effect in isolation.

    *except_*, if non-empty, removes the named fixer ids from whichever set
    --only or the WARN+ threshold would otherwise have selected.

    Both parameters take fixer ids already validated by the caller (CLI
    validates --only/--except against ALL_FIXER_IDS and rejects contradictions
    before calling into lint/fix at all); this function does not re-validate.
    """
    if only:
        base = [fixer for fixer in fixer_order if fixer.fixer_id in only]
    else:
        fired_check_ids = {r.check_id for r in results if r.severity >= Severity.WARN}
        base = [fixer for fixer in fixer_order if fixer.check_id in fired_check_ids]

    if except_:
        base = [fixer for fixer in base if fixer.fixer_id not in except_]

    return base


def run_dry_run(
    ds: CanonicalDataset,
    results: list[CheckResult],
    *,
    fixer_order: tuple[Fixer, ...] = _FIXER_ORDER,
    only: frozenset[str] | None = None,
    except_: frozenset[str] | None = None,
) -> FixPlan:
    """Compute the combined Diff for every applicable fixer; writes nothing.

    Each fixer's dry_run() is called against the ORIGINAL ds -- dry-run never
    chains, because chaining is only meaningful when fixers actually write
    (a later fixer's dry_run() against an unwritten hypothetical output would
    just be re-deriving the same Diff against the same unmodified source).
    A RepairError from any fixer propagates to the caller (the CLI renders it
    as a clean user-facing message, per the manual's error rule).
    """
    fixers = select_applicable_fixers(results, fixer_order=fixer_order, only=only, except_=except_)
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


def run_apply(
    ds: CanonicalDataset,
    results: list[CheckResult],
    output_path: Path,
    *,
    fixer_order: tuple[Fixer, ...] = _FIXER_ORDER,
    only: frozenset[str] | None = None,
    except_: frozenset[str] | None = None,
) -> FixPlan:
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
    fixers = select_applicable_fixers(results, fixer_order=fixer_order, only=only, except_=except_)
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


def validate_fixer_selection(only: frozenset[str], except_: frozenset[str]) -> str | None:
    """Validate --only/--except fixer ids before any lint or fixer work starts.

    Returns a clean, actionable error message (naming the invalid or
    conflicting id(s) and listing ALL_FIXER_IDS) if selection is invalid;
    returns None if selection is valid. The caller (cli.py) is responsible
    for exiting with code 2 and printing the message -- this function never
    raises or exits itself, matching every other CLI usage-error check in
    fix() (e.g. --apply without --out).
    """
    known = set(ALL_FIXER_IDS)
    valid_ids_list = ", ".join(ALL_FIXER_IDS)

    unknown_only = only - known
    unknown_except = except_ - known
    unknown = unknown_only | unknown_except
    if unknown:
        return f"unknown fixer id(s) {sorted(unknown)!r}. Valid fixer ids: {valid_ids_list}."

    conflicting = only & except_
    if conflicting:
        return (
            f"fixer id(s) {sorted(conflicting)!r} appear in both --only and --except, "
            "which is a contradiction. Remove the id from one of the two."
        )

    return None
