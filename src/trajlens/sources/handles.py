"""Lazy access primitives for Parquet and video shards.

Parquet shards are opened via pyarrow.parquet.ParquetFile, which reads only
the footer/schema on open and streams row groups on demand — the whole file
is never materialized in memory (05 §6). Video shards get a handle only;
decoding is PyAV's job at a later milestone, never this layer's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from trajlens.errors import DatasetFormatError


def open_parquet_shard(path: Path) -> pq.ParquetFile:
    """Open a Parquet shard lazily; row groups stream on read, never the whole file."""
    if not path.is_file():
        raise DatasetFormatError(f"expected parquet shard not found: {path}")
    # pyarrow's ParquetFile constructor remains untyped through 24.x despite
    # partial py.typed support added in pyarrow 19+; the per-module mypy
    # override (disallow_untyped_calls = false) does not suppress this for
    # reasons not fully understood -- suppressing at the call site instead.
    return pq.ParquetFile(path)  # type: ignore[no-untyped-call]


@dataclass(frozen=True, slots=True)
class VideoShardHandle:
    """A handle to a video shard file. No decode happens at this layer."""

    path: Path | str

    @property
    def is_local(self) -> bool:
        return isinstance(self.path, Path)


def open_video_shard(path: Path) -> VideoShardHandle:
    """Return a handle to a video shard; raises if the file does not exist."""
    if not path.is_file():
        raise DatasetFormatError(f"expected video shard not found: {path}")
    return VideoShardHandle(path=path)


def open_hub_parquet_shard(repo_id: str, revision: str | None, path: str) -> pq.ParquetFile:
    """Open a Parquet shard lazily over HTTP from the Hugging Face Hub.

    NOTE on pre_buffer=True (investigated 2026-06-21, M7):
    PyArrow's pre_buffer=True coalesces scattered per-column-chunk HTTP Range
    requests into fewer, parallel requests via a background I/O thread pool.
    Benchmark results from direct shard reads:
      - 835KB shard (aloha): 4.05s → 0.66s (6.1x faster)
      - 1.1MB shard (xarm):  3.65s → 1.93s (1.9x faster)
      - 54MB shard (pepijn): 25.5s → 6.5s  (3.9x faster)

    However, the lint pipeline opens the same shard 7 times sequentially (once
    per ShardColumnCache instance, one per check).  In the multi-open pattern,
    pre_buffer=True does NOT help: the background thread pool startup overhead
    is paid 7 times, network jitter dominates per-open latency, and measured
    totals (default 47.6s vs pre_buffer 49.4s) show no consistent benefit.

    The correct fix for the multi-open problem is to share a single
    pq.ParquetFile per shard across all checks (i.e. pass the ParquetFile into
    ShardColumnCache rather than re-opening via parquet_shard_for_episode).
    That architectural change is tracked separately.  Do not add blanket
    pre_buffer=True here until that refactor lands, or measure it against the
    full multi-open scenario rather than single-shard isolates.
    """
    try:
        from huggingface_hub import HfFileSystem
    except ImportError as exc:
        raise DatasetFormatError("huggingface_hub is required to load Hub datasets") from exc

    fs = HfFileSystem(revision=revision)
    try:
        f = fs.open(f"datasets/{repo_id}/{path}", "rb")
        return pq.ParquetFile(f)  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise DatasetFormatError(
            f"expected parquet shard not found on Hub: {repo_id}/{path}"
        ) from exc


def open_hub_video_shard(repo_id: str, revision: str | None, path: str) -> VideoShardHandle:
    """Return an HTTP URL handle to a video shard on the Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_url
    except ImportError as exc:
        raise DatasetFormatError("huggingface_hub is required to load Hub datasets") from exc

    url = hf_hub_url(repo_id, filename=path, repo_type="dataset", revision=revision)
    return VideoShardHandle(path=url)
