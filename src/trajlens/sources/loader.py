"""SourceLoader — resolves a dataset reference into a readable handle.

This is the only place trajlens performs filesystem or network I/O for
reading a dataset (02_ARCHITECTURE.md §3.1). Hub access is explicit,
opt-in, and goes only through huggingface_hub (T7) — it is only attempted
when the reference does not resolve to a local directory, and the import
itself is lazy so the optional [hub] extra is never required for local-only
use.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from trajlens.errors import SourceResolutionError
from trajlens.sources.bounds import MAX_DECLARED_EPISODES, MAX_DECLARED_FRAMES, check_resource_bound
from trajlens.sources.handles import VideoShardHandle, open_parquet_shard, open_video_shard
from trajlens.sources.info import DatasetInfoModel, load_info
from trajlens.sources.paths import safe_join
from trajlens.sources.version import DatasetVersion, detect_version

_HUB_CACHE_ROOT = Path.home() / ".cache" / "trajlens" / "hub"


@dataclass(frozen=True, slots=True)
class SourceHandle:
    """A resolved, version-detected dataset, ready for lazy reads."""

    root: Path
    version: DatasetVersion
    info: DatasetInfoModel

    def parquet_shard(self, *relative_parts: str) -> pq.ParquetFile:
        """Open a Parquet shard by path relative to the dataset root, safely joined."""
        path = safe_join(self.root, *relative_parts)
        return open_parquet_shard(path)

    def video_shard(self, *relative_parts: str) -> VideoShardHandle:
        """Open a video shard handle by path relative to the dataset root, safely joined."""
        path = safe_join(self.root, *relative_parts)
        return open_video_shard(path)


class SourceLoader:
    """Resolves a local path or Hub repo id (org/name) into a SourceHandle."""

    def resolve(self, ref: str, *, revision: str | None = None) -> SourceHandle:
        """Resolve *ref* to a SourceHandle, detecting version and checking bounds.

        Raises SourceResolutionError if *ref* is neither an existing local
        directory nor a reachable Hub dataset repo.
        """
        candidate = Path(ref)
        root = candidate.resolve() if candidate.is_dir() else self._resolve_hub(ref, revision)

        info = load_info(root)
        version = detect_version(root, info)

        if info.total_episodes is not None:
            check_resource_bound(
                info.total_episodes, max_value=MAX_DECLARED_EPISODES, what="episode count"
            )
        if info.total_frames is not None:
            check_resource_bound(
                info.total_frames, max_value=MAX_DECLARED_FRAMES, what="frame count"
            )

        return SourceHandle(root=root, version=version, info=info)

    def _resolve_hub(self, repo_id: str, revision: str | None) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SourceResolutionError(
                f"{repo_id!r} is not a local path, and the optional Hub "
                f"dependency is not installed. Install it with "
                f"`pip install trajlens[hub]` to load datasets from the "
                f"Hugging Face Hub."
            ) from exc

        # huggingface_hub's default cache stores files as symlinks into a
        # content-addressed blobs/ dir that is a *sibling* of the snapshot
        # directory it returns, not a descendant. safe_join's containment
        # check correctly treats that as an escape, since it can't tell HF's
        # own cache symlinks apart from a hostile dataset-committed symlink.
        # local_dir sidesteps this: huggingface_hub 0.36 copies real files
        # into it (file_download.py _hf_hub_download_to_local_dir) instead of
        # symlinking, so the returned root behaves like a plain local dir.
        # Keyed by revision so a later resolve() at a different revision
        # doesn't silently serve stale cached content.
        local_dir = _HUB_CACHE_ROOT / repo_id.replace("/", "--") / (revision or "main")

        try:
            # Resolving only needs meta/ to detect version and parse info.json;
            # data/video shards are fetched lazily when actually opened. Mirrors
            # lerobot's own metadata-only pull (dataset_metadata.py _pull_from_repo).
            snapshot_path = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                allow_patterns="meta/",
                local_dir=local_dir,
            )
        except Exception as exc:
            raise SourceResolutionError(
                f"could not resolve {repo_id!r}: it is not a local directory "
                f"and could not be downloaded from the Hugging Face Hub "
                f"({exc})."
            ) from exc
        return Path(snapshot_path)
