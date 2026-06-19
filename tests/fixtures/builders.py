"""Builders for tiny synthetic LeRobotDataset fixtures used by sources/ tests.

Datasets are generated on the fly into a tmp_path rather than committed as
binary blobs, so the fixture's correctness is exercised by every test that
consumes it. Field names and path templates mirror the live lerobot 0.5.2
source (verified, see src/trajlens/sources/info.py and version.py).
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


def build_v3_dataset(root: Path, *, num_episodes: int = 3, camera: str = "top") -> None:
    """Build a tiny, valid v3.0-shaped dataset under root."""
    total_frames = num_episodes * FRAMES_PER_EPISODE
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "features": DEFAULT_FEATURES,
        "total_episodes": num_episodes,
        "total_frames": total_frames,
    }
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(_write_frames_table(num_episodes), data_dir / "file-000.parquet")

    episodes_dir = root / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    episodes_table = pa.table(
        {
            "episode_index": pa.array(list(range(num_episodes)), type=pa.int64()),
            "length": pa.array([FRAMES_PER_EPISODE] * num_episodes, type=pa.int64()),
        }
    )
    pq.write_table(episodes_table, episodes_dir / "file-000.parquet")

    tasks_table = pa.table(
        {"task_index": pa.array([0], type=pa.int64()), "task": pa.array(["do the thing"])}
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
        "features": DEFAULT_FEATURES,
        "total_episodes": num_episodes,
        "total_frames": total_frames,
    }
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))

    episode_lines = [
        json.dumps({"episode_index": ep, "tasks": ["do the thing"], "length": FRAMES_PER_EPISODE})
        for ep in range(num_episodes)
    ]
    (root / "meta" / "episodes.jsonl").write_text("\n".join(episode_lines) + "\n")
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "do the thing"}) + "\n"
    )
    if codebase_version == "v2.1":
        stats_lines = [json.dumps({"episode_index": ep, "stats": {}}) for ep in range(num_episodes)]
        (root / "meta" / "episodes_stats.jsonl").write_text("\n".join(stats_lines) + "\n")

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
