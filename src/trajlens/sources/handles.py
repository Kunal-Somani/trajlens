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

    path: Path


def open_video_shard(path: Path) -> VideoShardHandle:
    """Return a handle to a video shard; raises if the file does not exist."""
    if not path.is_file():
        raise DatasetFormatError(f"expected video shard not found: {path}")
    return VideoShardHandle(path=path)
