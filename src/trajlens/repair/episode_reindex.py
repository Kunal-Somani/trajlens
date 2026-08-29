"""REPAIR.EPISODE_REINDEX — fixer for STRUCTURAL.METADATA_DATA_AGREEMENT (#2401).

Detection counterpart: src/trajlens/checks/structural.py _MetadataDataAgreementCheck.

The fix: the actual data is ground truth.  Every data row already carries its
own true global position in the ``index`` column and its true episode
assignment in the ``episode_index`` column.  This fixer derives, per episode,
the true ``dataset_from_index``/``dataset_to_index``/``length`` directly from
those columns and rewrites only the episode-metadata Parquet shard(s) so the
declared boundaries agree with the data.  Data and video shards are never
opened in write mode, and no row is ever reordered, dropped, or reassigned to
a different episode_index — the direction of repair is metadata -> matches ->
data, never the reverse (per 06_SECURITY_AND_THREAT_MODEL.md T9).

Fail-closed guarantee: if the data itself is internally inconsistent -- an
episode_index's ``index`` values are non-contiguous, two episodes' ranges
overlap, or an episode's rows are interleaved with another episode's within a
shard -- there is no single consistent boundary assignment that could repair
the dataset, so this fixer raises RepairError and writes nothing.  A guessed
repair here would silently recreate the exact #2401 corruption it claims to
fix, which is the catastrophic case this fixer exists to prevent.

ADR-004 requirements satisfied here:
  - Copy-on-write: source is never opened in write mode; output_path receives
    the corrected episode-metadata shard(s).
  - Dry-run by default: dry_run() computes the Diff with zero filesystem
    writes.
  - Round-trip: tests in tests/unit/test_episode_reindex.py and
    tests/property/test_episode_reindex_properties.py verify repair ->
    re-lint -> INFO, plus a semantic-correctness assertion that repaired
    boundaries select the RIGHT frames, not merely consistent counts.

KNOWN SCOPE BOUNDARY -- correctness is conditional on the ``index`` column
being trustworthy: this fixer's entire ground-truth derivation rests on each
data row's ``index`` value being the writer's real, original global position
(03_DATA_FORMAT_SPEC.md §2 confirms this against the live lerobot 0.5.2
writer). If a dataset's ``index`` column is itself corrupted in a way that is
*internally self-consistent* per episode -- contiguous, non-overlapping
ranges, just not what the writer actually assigned (e.g. two episodes'
``index`` ranges silently swapped) -- this fixer has no independent signal to
detect that and will "repair" the episode metadata to agree with the
corrupted ``index`` values. This is an accepted, documented design boundary,
not a bug: there is no second independent ground-truth column in the v3.0
format to cross-check ``index`` against. A missing or structurally
inconsistent ``index`` column (absent from the schema, non-contiguous,
overlapping, or interleaved) IS caught and raises RepairError -- see
_derive_true_episode_ranges. Only self-consistent-but-wrong corruption of
``index`` itself falls outside this fixer's detection.
"""

from __future__ import annotations

import shutil
from itertools import pairwise
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from trajlens.errors import RepairError
from trajlens.model.canonical import CanonicalDataset
from trajlens.repair.protocol import BoundaryChange, Diff, RepairSummary
from trajlens.sources.paths import safe_join

log = structlog.get_logger(__name__)

FIXER_ID = "REPAIR.EPISODE_REINDEX"
CHECK_ID = "STRUCTURAL.METADATA_DATA_AGREEMENT"

# v3.0 episode-metadata shards live under meta/episodes/chunk-*/file-*.parquet
# -- same glob model/adapters.py _load_v3_episodes uses to discover them.
_EPISODES_DIR = ("meta", "episodes")


class EpisodeReindexFixer:
    """Recompute dataset_from_index/dataset_to_index/length from data (ADR-004).

    dry_run() derives each episode's true row range from the data shards'
    ``episode_index`` and ``index`` columns -- the same columns
    STRUCTURAL.METADATA_DATA_AGREEMENT's row-count check reads -- and diffs
    them against the declared episode-metadata values.  apply() copies the
    source tree to *output_path* and rewrites only the affected
    meta/episodes/.../*.parquet shard(s); data and video shards are always
    byte-identical to source.

    Only v3.0 datasets are supported: v2.x episode metadata does not store
    explicit from/to indices at all (model/adapters.py _load_v2_episodes
    derives them from a cumulative sum of declared lengths, so there is
    nothing independent to reconcile against).  A RepairError is raised for
    v2.x inputs.
    """

    fixer_id: str = FIXER_ID
    check_id: str = CHECK_ID

    def dry_run(self, ds: CanonicalDataset) -> Diff:
        """Compute what would change without writing anything.

        Returns a Diff recording every dataset_from_index/dataset_to_index/
        length field that disagrees with the true value derived from data.
        Raises RepairError if the data itself admits no consistent boundary
        assignment (see module docstring) -- never guesses a repair in that
        case.  Returns a no-op Diff if the dataset already agrees with data.
        """
        _check_preconditions(ds)

        true_ranges = _derive_true_episode_ranges(ds)

        declared_indices = {episode.episode_index for episode in ds}
        data_only_indices = set(true_ranges) - declared_indices
        if data_only_indices:
            raise RepairError(
                f"data shard(s) contain rows for episode_index {sorted(data_only_indices)} "
                "that are not declared in episode metadata. There is no EpisodeRecord to "
                "correct for these frames. This dataset cannot be safely repaired."
            )

        changes: list[BoundaryChange] = []
        for episode in ds:
            true_range = true_ranges.get(episode.episode_index)
            if true_range is None:
                raise RepairError(
                    f"episode {episode.episode_index} is declared in episode metadata "
                    "but has no rows in any data shard; cannot derive a true boundary "
                    "for it. This dataset cannot be safely repaired."
                )
            true_from, true_to, true_length = true_range

            if episode.dataset_from_index != true_from:
                changes.append(
                    BoundaryChange(
                        episode_index=episode.episode_index,
                        field="dataset_from_index",
                        old_value=episode.dataset_from_index,
                        new_value=true_from,
                    )
                )
            if episode.dataset_to_index != true_to:
                changes.append(
                    BoundaryChange(
                        episode_index=episode.episode_index,
                        field="dataset_to_index",
                        old_value=episode.dataset_to_index,
                        new_value=true_to,
                    )
                )
            if episode.length != true_length:
                changes.append(
                    BoundaryChange(
                        episode_index=episode.episode_index,
                        field="length",
                        old_value=episode.length,
                        new_value=true_length,
                    )
                )

        diff = Diff(changes=tuple(changes), check_id=CHECK_ID, fixer_id=FIXER_ID)
        if diff.is_noop:
            log.info(
                "episode_reindex.dry_run.noop",
                reason="declared boundaries already agree with data",
            )
        else:
            log.info(
                "episode_reindex.dry_run.changes_found",
                num_changes=len(changes),
                num_episodes_affected=len({c.episode_index for c in changes}),
            )
        return diff

    def apply(self, ds: CanonicalDataset, output_path: Path) -> RepairSummary:
        """Write a corrected copy of *ds* to *output_path* (copy-on-write).

        Steps:
          1. Precondition check (version).
          2. dry_run() to compute the Diff (raises RepairError and writes
             nothing if the data is internally inconsistent).
          3. Copy the source tree to output_path.
          4. Rewrite only the affected meta/episodes/.../*.parquet shard(s)
             with corrected dataset_from_index/dataset_to_index/length.

        *output_path* must not be the source dataset root.  Data and video
        shards are never opened in write mode -- only episode-metadata
        shards are rewritten, and only the three boundary fields within them.
        """
        _check_preconditions(ds)

        source_root = _dataset_root(ds)
        if output_path.resolve() == source_root.resolve():
            raise RepairError(
                "output_path must not be the source dataset root (ADR-004 "
                f"copy-on-write). Got: {output_path}"
            )

        diff = self.dry_run(ds)

        log.info(
            "episode_reindex.apply.start",
            source=str(source_root),
            output=str(output_path),
            num_changes=len(diff.changes),
        )

        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(source_root, output_path)

        if diff.is_noop:
            log.info("episode_reindex.apply.noop", output=str(output_path))
            return RepairSummary(
                output_path=output_path,
                changes_written=0,
                frames_corrected=0,
            )

        shards_written = _rewrite_episode_shards(diff, output_root=output_path)

        boundary_changes = [c for c in diff.changes if isinstance(c, BoundaryChange)]
        log.info(
            "episode_reindex.apply.done",
            output=str(output_path),
            shards_written=shards_written,
            episodes_corrected=len({c.episode_index for c in boundary_changes}),
        )
        return RepairSummary(
            output_path=output_path,
            changes_written=shards_written,
            frames_corrected=len(diff.changes),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dataset_root(ds: CanonicalDataset) -> Path:
    return ds.stats.root


def _check_preconditions(ds: CanonicalDataset) -> None:
    """Raise RepairError for any condition that makes repair unsafe or undefined."""
    if ds.format_id != "lerobot" or ds.format_version != "3.0":
        raise RepairError(
            f"EpisodeReindexFixer only supports v3.0 datasets; "
            f"got {ds.format_label}. v2.x episode metadata does not store "
            "explicit dataset_from_index/dataset_to_index values to reconcile."
        )


def _derive_true_episode_ranges(ds: CanonicalDataset) -> dict[int, tuple[int, int, int]]:
    """Derive each episode's true (from_index, to_index, length) from data.

    Reads each distinct data shard exactly once (dedup by resolved path, since
    v3.0 shards hold many episodes), using the same ``episode_index`` and
    ``index`` columns STRUCTURAL.METADATA_DATA_AGREEMENT reads. ``index`` is
    each row's own declared true global position (written by the lerobot
    writer at frame-write time) -- it is not derived from the from/to
    boundaries this fixer is correcting, so it is a valid independent source
    of truth.

    Raises RepairError (writing nothing) if the data is internally
    inconsistent in any way that makes a single boundary assignment
    impossible: an episode's rows are interleaved with another episode's
    within a shard, an episode's ``index`` values are non-contiguous, or two
    episodes' derived ranges overlap.
    """
    # Single pass over distinct shards -- single-writer assumption: source
    # directory must not be mutated by another process between dry_run() and
    # apply(). SourceHandle.parquet_shard() memoizes by relative path (see
    # sources/loader.py), so the same physical shard always yields the same
    # ParquetFile object; dedup on object identity reads each shard once even
    # though many episodes resolve to it.
    seen_shard_ids: set[int] = set()
    ep_index_values: dict[int, list[int]] = {}
    ep_row_counts: dict[int, int] = {}

    for episode in ds:
        pf = ds.parquet_shard_for_episode(episode)
        if id(pf) in seen_shard_ids:
            continue
        seen_shard_ids.add(id(pf))

        shard_columns = pf.schema_arrow.names
        missing = [c for c in ("episode_index", "index") if c not in shard_columns]
        if missing:
            raise RepairError(
                f"episode {episode.episode_index}'s data shard is missing required "
                f"column(s) {missing}; cannot derive a true boundary without them. "
                "This dataset cannot be safely repaired."
            )

        table = pf.read(columns=["episode_index", "index"])  # type: ignore[no-untyped-call]
        ep_col: list[Any] = table.column("episode_index").to_pylist()
        idx_col: list[Any] = table.column("index").to_pylist()

        _check_shard_not_interleaved(ep_col)

        for ep_val, idx_val in zip(ep_col, idx_col, strict=True):
            ep_key = int(ep_val)
            ep_index_values.setdefault(ep_key, []).append(int(idx_val))
            ep_row_counts[ep_key] = ep_row_counts.get(ep_key, 0) + 1

    true_ranges: dict[int, tuple[int, int, int]] = {}
    for ep_key, idx_values in ep_index_values.items():
        length = ep_row_counts[ep_key]
        lo = min(idx_values)
        hi = max(idx_values)
        if hi - lo + 1 != length or len(set(idx_values)) != length:
            raise RepairError(
                f"episode {ep_key}: data rows' 'index' column is not contiguous "
                f"(min={lo}, max={hi}, row_count={length}). No single "
                "dataset_from_index/dataset_to_index assignment can represent this "
                "episode's data. This dataset cannot be safely repaired -- the "
                "underlying data is internally inconsistent, not just the metadata."
            )
        true_ranges[ep_key] = (lo, hi + 1, length)

    _check_ranges_do_not_overlap(true_ranges)

    return true_ranges


def _check_shard_not_interleaved(ep_col: list[Any]) -> None:
    """Raise RepairError if any episode_index value appears in >1 contiguous run.

    If a shard's rows go e.g. [0, 0, 1, 0, 1, 1], episode 0's rows cannot be
    described by any single contiguous range -- no from/to boundary can
    represent it without either mis-including episode 1's rows or excluding
    some of episode 0's own rows.
    """
    seen_and_closed: set[int] = set()
    current: int | None = None
    for ep_val in ep_col:
        ep_key = int(ep_val)
        if ep_key == current:
            continue
        if current is not None:
            seen_and_closed.add(current)
        if ep_key in seen_and_closed:
            raise RepairError(
                f"episode {ep_key}: rows in a data shard are interleaved with another "
                "episode's rows (not a single contiguous run). No single "
                "dataset_from_index/dataset_to_index assignment can represent this "
                "episode's data. This dataset cannot be safely repaired -- the "
                "underlying data is internally inconsistent, not just the metadata."
            )
        current = ep_key


def _check_ranges_do_not_overlap(ranges: dict[int, tuple[int, int, int]]) -> None:
    """Raise RepairError if any two episodes' derived [from, to) ranges overlap."""
    ordered = sorted(ranges.items(), key=lambda kv: kv[1][0])
    for (ep_a, (_, to_a, _)), (ep_b, (from_b, _, _)) in pairwise(ordered):
        if from_b < to_a:
            raise RepairError(
                f"episodes {ep_a} and {ep_b}: derived data ranges overlap "
                f"(episode {ep_a} ends at index {to_a}, episode {ep_b} starts at "
                f"index {from_b}). No single boundary assignment can represent this "
                "data. This dataset cannot be safely repaired -- the underlying data "
                "is internally inconsistent, not just the metadata."
            )


def _rewrite_episode_shards(diff: Diff, *, output_root: Path) -> int:
    """Rewrite affected meta/episodes/.../*.parquet shards with corrected boundaries.

    Groups changes by episode_index, discovers every episode-metadata shard
    under output_root (same glob model/adapters.py uses), reads each shard
    exactly once, patches only the dataset_from_index/dataset_to_index/length
    columns for affected rows, and writes back.  Data and video shards are
    never touched.  Returns the number of distinct shard files rewritten.
    """
    by_episode: dict[int, dict[str, int]] = {}
    for change in diff.changes:
        assert isinstance(change, BoundaryChange), f"unexpected change type: {type(change)}"
        by_episode.setdefault(change.episode_index, {})[change.field] = int(change.new_value)

    episodes_root = safe_join(output_root, *_EPISODES_DIR)
    shard_paths = sorted(episodes_root.glob("chunk-*/file-*.parquet"))

    shards_written = 0
    for shard_path in shard_paths:
        table = pq.read_table(shard_path)  # type: ignore[no-untyped-call]
        ep_col: list[Any] = table.column("episode_index").to_pylist()

        if not any(int(ep) in by_episode for ep in ep_col):
            continue

        from_col: list[Any] = table.column("dataset_from_index").to_pylist()
        to_col: list[Any] = table.column("dataset_to_index").to_pylist()
        length_col: list[Any] = table.column("length").to_pylist()

        for row_idx, ep_val in enumerate(ep_col):
            patch = by_episode.get(int(ep_val))
            if patch is None:
                continue
            if "dataset_from_index" in patch:
                from_col[row_idx] = patch["dataset_from_index"]
            if "dataset_to_index" in patch:
                to_col[row_idx] = patch["dataset_to_index"]
            if "length" in patch:
                length_col[row_idx] = patch["length"]

        new_table = table
        for col_name, new_values in (
            ("dataset_from_index", from_col),
            ("dataset_to_index", to_col),
            ("length", length_col),
        ):
            col_type = new_table.schema.field(col_name).type
            new_table = new_table.set_column(
                new_table.schema.get_field_index(col_name),
                col_name,
                pa.array(new_values, type=col_type),
            )

        pq.write_table(new_table, shard_path)  # type: ignore[no-untyped-call]
        shards_written += 1
        log.debug(
            "episode_reindex.shard_rewritten",
            shard=str(shard_path),
            episodes_patched=len({int(ep) for ep in ep_col if int(ep) in by_episode}),
        )

    return shards_written
