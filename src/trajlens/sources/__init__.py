"""Dataset source layer.

The only place trajlens performs filesystem or network I/O for reading a
dataset (02_ARCHITECTURE.md §3.1). Resolves a local path or Hub repo id into
a version-detected SourceHandle with lazy, bounded access to metadata,
Parquet shards, and video shards.
"""

from trajlens.sources.bounds import MAX_DECLARED_EPISODES, MAX_DECLARED_FRAMES, check_resource_bound
from trajlens.sources.handles import VideoShardHandle, open_parquet_shard, open_video_shard
from trajlens.sources.info import DatasetInfoModel, load_info
from trajlens.sources.loader import SourceHandle, SourceLoader
from trajlens.sources.paths import safe_join
from trajlens.sources.version import DatasetVersion, detect_version

__all__ = [
    "MAX_DECLARED_EPISODES",
    "MAX_DECLARED_FRAMES",
    "DatasetInfoModel",
    "DatasetVersion",
    "SourceHandle",
    "SourceLoader",
    "VideoShardHandle",
    "check_resource_bound",
    "detect_version",
    "load_info",
    "open_parquet_shard",
    "open_video_shard",
    "safe_join",
]
