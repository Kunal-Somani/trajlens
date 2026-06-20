"""Builders for tiny synthetic LeRobotDataset fixtures used by sources/ and model/ tests.

Datasets are generated on the fly into a tmp_path rather than committed as
binary blobs, so the fixture's correctness is exercised by every test that
consumes it. Field names and path templates mirror the live lerobot 0.5.2
source (commit 8515d456), verified directly against:
  - src/lerobot/datasets/dataset_writer.py (_save_episode_data, _save_episode_video)
  - src/lerobot/datasets/dataset_metadata.py (save_episode)
  - tests/fixtures/dataset_factories.py

The v3.0 episodes metadata schema uses slash-namespaced column names
(``data/chunk_index``, ``videos/{camera}/from_timestamp``, etc.), not the
underscore-joined names (``data_chunk_index``) that appear in a stale
docstring elsewhere in the lerobot source — the actual writer code was
checked, not the docstring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

# Mirrors lerobot.utils.constants.DEFAULT_FEATURES exactly.
DEFAULT_FEATURES: dict[str, dict[str, Any]] = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}

FRAMES_PER_EPISODE = 4
DEFAULT_TASK = "do the thing"


def _video_feature(camera: str) -> dict[str, dict[str, Any]]:
    # The features-dict key is itself the video path segment in the real
    # lerobot writer (get_video_keys() in convert_dataset_v21_to_v30.py
    # filters features by dtype == "video" and uses the key verbatim as
    # video_key) -- so the fixture's feature name and path segment must match.
    return {camera: {"dtype": "video", "shape": [3, 64, 64], "names": None}}


def _write_frames_table(num_episodes: int) -> pa.Table:
    timestamps, frame_idx, ep_idx, idx, task_idx = [], [], [], [], []
    frame = 0
    for ep in range(num_episodes):
        for f in range(FRAMES_PER_EPISODE):
            timestamps.append(f / 30.0)
            frame_idx.append(f)
            ep_idx.append(ep)
            idx.append(frame)
            task_idx.append(0)
            frame += 1
    return pa.table(
        {
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_idx, type=pa.int64()),
            "episode_index": pa.array(ep_idx, type=pa.int64()),
            "index": pa.array(idx, type=pa.int64()),
            "task_index": pa.array(task_idx, type=pa.int64()),
        }
    )


def build_v3_dataset(
    root: Path,
    *,
    num_episodes: int = 3,
    camera: str = "top",
    episodes_per_shard: int | None = None,
) -> None:
    """Build a tiny, valid v3.0-shaped dataset under root.

    Episode metadata columns mirror the real writer's flat (slash-namespaced)
    schema. By default all episodes land in a single meta/episodes shard and a
    single data/video shard; pass episodes_per_shard to spread them across
    multiple chunk-*/file-*.parquet shards, exercising the multi-shard
    discovery path real large datasets require.
    """
    total_frames = num_episodes * FRAMES_PER_EPISODE
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "features": {**DEFAULT_FEATURES, **_video_feature(camera)},
        "total_episodes": num_episodes,
        "total_frames": total_frames,
    }
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(_write_frames_table(num_episodes), data_dir / "file-000.parquet")

    shard_size = episodes_per_shard or max(num_episodes, 1)
    episode_rows: list[dict[str, Any]] = []
    for ep in range(num_episodes):
        from_idx = ep * FRAMES_PER_EPISODE
        episode_rows.append(
            {
                "episode_index": ep,
                "tasks": [DEFAULT_TASK],
                "length": FRAMES_PER_EPISODE,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": from_idx,
                "dataset_to_index": from_idx + FRAMES_PER_EPISODE,
                "meta/episodes/chunk_index": ep // shard_size,
                "meta/episodes/file_index": 0,
                f"videos/{camera}/chunk_index": 0,
                f"videos/{camera}/file_index": 0,
                f"videos/{camera}/from_timestamp": from_idx / 30.0,
                f"videos/{camera}/to_timestamp": (from_idx + FRAMES_PER_EPISODE) / 30.0,
            }
        )

    for shard_start in range(0, max(num_episodes, 1), shard_size):
        shard_rows = episode_rows[shard_start : shard_start + shard_size]
        chunk_index = shard_start // shard_size
        episodes_dir = root / "meta" / "episodes" / f"chunk-{chunk_index:03d}"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        if shard_rows:
            columns = {key: [row[key] for row in shard_rows] for key in shard_rows[0]}
            episodes_table = pa.table(columns)
        else:
            episodes_table = pa.table(
                {
                    "episode_index": pa.array([], type=pa.int64()),
                    "tasks": pa.array([], type=pa.list_(pa.string())),
                    "length": pa.array([], type=pa.int64()),
                    "data/chunk_index": pa.array([], type=pa.int64()),
                    "data/file_index": pa.array([], type=pa.int64()),
                    "dataset_from_index": pa.array([], type=pa.int64()),
                    "dataset_to_index": pa.array([], type=pa.int64()),
                    "meta/episodes/chunk_index": pa.array([], type=pa.int64()),
                    "meta/episodes/file_index": pa.array([], type=pa.int64()),
                    f"videos/{camera}/chunk_index": pa.array([], type=pa.int64()),
                    f"videos/{camera}/file_index": pa.array([], type=pa.int64()),
                    f"videos/{camera}/from_timestamp": pa.array([], type=pa.float64()),
                    f"videos/{camera}/to_timestamp": pa.array([], type=pa.float64()),
                }
            )
        pq.write_table(episodes_table, episodes_dir / "file-000.parquet")

    tasks_table = pa.table(
        {"task_index": pa.array([0], type=pa.int64()), "task": pa.array([DEFAULT_TASK])}
    )
    pq.write_table(tasks_table, root / "meta" / "tasks.parquet")

    video_dir = root / "videos" / camera / "chunk-000"
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "file-000.mp4").write_bytes(b"\x00")


def build_v2_dataset(
    root: Path, *, codebase_version: str = "v2.1", num_episodes: int = 3, camera: str = "top"
) -> None:
    """Build a tiny, valid v2.0/v2.1-shaped (one file per episode) dataset under root."""
    total_frames = num_episodes * FRAMES_PER_EPISODE
    info = {
        "codebase_version": codebase_version,
        "fps": 30,
        "features": {**DEFAULT_FEATURES, **_video_feature(camera)},
        "total_episodes": num_episodes,
        "total_frames": total_frames,
    }
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))

    episode_lines = [
        json.dumps({"episode_index": ep, "tasks": [DEFAULT_TASK], "length": FRAMES_PER_EPISODE})
        for ep in range(num_episodes)
    ]
    (root / "meta" / "episodes.jsonl").write_text(
        "\n".join(episode_lines) + ("\n" if episode_lines else "")
    )
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": DEFAULT_TASK}) + "\n"
    )
    if codebase_version == "v2.1":
        stats_lines = [json.dumps({"episode_index": ep, "stats": {}}) for ep in range(num_episodes)]
        (root / "meta" / "episodes_stats.jsonl").write_text(
            "\n".join(stats_lines) + ("\n" if stats_lines else "")
        )

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir = root / "videos" / "chunk-000" / camera
    video_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(num_episodes):
        frames = pa.table(
            {
                "timestamp": pa.array(
                    [f / 30.0 for f in range(FRAMES_PER_EPISODE)], type=pa.float32()
                ),
                "frame_index": pa.array(list(range(FRAMES_PER_EPISODE)), type=pa.int64()),
                "episode_index": pa.array([ep] * FRAMES_PER_EPISODE, type=pa.int64()),
                "index": pa.array(
                    list(range(ep * FRAMES_PER_EPISODE, (ep + 1) * FRAMES_PER_EPISODE)),
                    type=pa.int64(),
                ),
                "task_index": pa.array([0] * FRAMES_PER_EPISODE, type=pa.int64()),
            }
        )
        pq.write_table(frames, data_dir / f"episode_{ep:06d}.parquet")
        (video_dir / f"episode_{ep:06d}.mp4").write_bytes(b"\x00")
