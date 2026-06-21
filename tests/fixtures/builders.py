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


def build_v3_dataset_hub_tasks_schema(
    root: Path,
    *,
    num_episodes: int = 3,
    camera: str = "top",
) -> None:
    """Build a v3.0 dataset whose tasks.parquet uses the real Hub schema.

    All lerobot/* datasets published on the Hugging Face Hub (confirmed
    against 5 datasets, 2026-06-21, codebase_version=v3.0) write
    ``meta/tasks.parquet`` with columns::

        task_index: int64
        __index_level_0__: string   <-- NOT "task"

    This is a Pandas serialization artifact: lerobot stores task descriptions
    as the DataFrame index rather than a named column, so ``df.to_parquet()``
    emits the anonymous Pandas index column ``__index_level_0__`` instead of
    a column named ``task``.  The spec (03_DATA_FORMAT_SPEC.md §2) originally
    documented only the intended schema (``task_index, task``); this fixture
    captures the real-world shape so trajlens cannot regress silently.
    """
    build_v3_dataset(root, num_episodes=num_episodes, camera=camera)
    tasks_path = root / "meta" / "tasks.parquet"
    # Overwrite the tasks table with the real Hub column layout.
    hub_tasks_table = pa.table(
        {
            "task_index": pa.array([0], type=pa.int64()),
            # The task description is serialized as the Pandas DataFrame index,
            # which Parquet names "__index_level_0__" by default.
            "__index_level_0__": pa.array([DEFAULT_TASK], type=pa.string()),
        }
    )
    pq.write_table(hub_tasks_table, tasks_path)


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


def build_v3_wrong_schema(root: Path, *, camera: str = "top") -> None:
    """Build a v3.0 dataset where 'timestamp' has the wrong Arrow dtype.

    Triggers SCHEMA_CONSISTENCY FAIL.
    """
    build_v3_dataset(root, camera=camera)
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    n_rows = old.num_rows
    # Write timestamp as int64 (expected float32) to trigger the dtype mismatch.
    new = old.set_column(
        old.schema.get_field_index("timestamp"),
        "timestamp",
        pa.array(list(range(n_rows)), type=pa.int64()),
    )
    pq.write_table(new, data_path)


def build_v3_noncontiguous_indices(root: Path, *, camera: str = "top") -> None:
    """Build a v3.0 dataset where frame_index is not 0-based within each episode."""
    build_v3_dataset(root, camera=camera)
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    shifted = [v + 1 for v in old.column("frame_index").to_pylist()]
    new = old.set_column(
        old.schema.get_field_index("frame_index"),
        "frame_index",
        pa.array(shifted, type=pa.int64()),
    )
    pq.write_table(new, data_path)


def build_v3_metadata_data_disagreement(
    root: Path, *, camera: str = "top", num_episodes: int = 3
) -> None:
    """Build a v3.0 dataset where episode metadata from/to boundaries are wrong.

    Replicates the #2401 corruption pattern: metadata claims each episode has
    FRAMES_PER_EPISODE rows but the declared to_index spans one extra row.
    """
    build_v3_dataset(root, num_episodes=num_episodes, camera=camera)
    episodes_root = root / "meta" / "episodes" / "chunk-000"
    ep_path = episodes_root / "file-000.parquet"
    old = pq.read_table(ep_path)
    to_col = old.column("dataset_to_index").to_pylist()
    corrupted_to = [v + 1 for v in to_col]
    new = old.set_column(
        old.schema.get_field_index("dataset_to_index"),
        "dataset_to_index",
        pa.array(corrupted_to, type=pa.int64()),
    )
    pq.write_table(new, ep_path)


def build_v3_non_monotonic_timestamps(root: Path, *, camera: str = "top") -> None:
    """Build a v3.0 dataset where timestamps are not strictly increasing within episode 0."""
    build_v3_dataset(root, camera=camera)
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    ts = old.column("timestamp").to_pylist()
    ep = old.column("episode_index").to_pylist()
    ep0_indices = [i for i, e in enumerate(ep) if e == 0]
    if len(ep0_indices) >= 2:
        ts[ep0_indices[1]], ts[ep0_indices[0]] = ts[ep0_indices[0]], ts[ep0_indices[1]]
    new = old.set_column(
        old.schema.get_field_index("timestamp"),
        "timestamp",
        pa.array(ts, type=pa.float32()),
    )
    pq.write_table(new, data_path)


def build_v3_bad_timestamp_spacing(
    root: Path, *, camera: str = "top", gap_multiple: float = 3.0
) -> None:
    """Build a v3.0 dataset with a large timestamp gap that exceeds 1 frame duration.

    Triggers TEMPORAL.TIMESTAMP_SPACING FAIL.
    """
    build_v3_dataset(root, camera=camera)
    fps = 30
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    ts = old.column("timestamp").to_pylist()
    ep = old.column("episode_index").to_pylist()
    ep0_indices = [i for i, e in enumerate(ep) if e == 0]
    if len(ep0_indices) >= 2:
        gap_s = gap_multiple / fps
        for j in ep0_indices[1:]:
            ts[j] = ts[j] + gap_s
    new = old.set_column(
        old.schema.get_field_index("timestamp"),
        "timestamp",
        pa.array(ts, type=pa.float32()),
    )
    pq.write_table(new, data_path)


def build_v3_timestamp_drift(
    root: Path,
    *,
    camera: str = "top",
    num_episodes: int = 5,
    drift_per_frame: float = 5e-5,
) -> None:
    """Build a v3.0 dataset with accumulating timestamp drift (#3177 fingerprint).

    Each frame's timestamp gains an additional ``drift_per_frame`` seconds,
    causing cumulative drift to exceed the decoder tolerance (1e-4 s).
    """
    build_v3_dataset(root, num_episodes=num_episodes, camera=camera)
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    ts = old.column("timestamp").to_pylist()
    new_ts = [float(t) + (i * drift_per_frame) for i, t in enumerate(ts)]
    new = old.set_column(
        old.schema.get_field_index("timestamp"),
        "timestamp",
        pa.array(new_ts, type=pa.float32()),
    )
    pq.write_table(new, data_path)


def build_v3_missing_shard(root: Path, *, camera: str = "top") -> None:
    """Build a v3.0 dataset where the data shard is deleted.

    Triggers PATH_TEMPLATE_RESOLVES FAIL.
    """
    build_v3_dataset(root, camera=camera)
    (root / "data" / "chunk-000" / "file-000.parquet").unlink()


def build_v3_corrupt_video(root: Path, *, camera: str = "top") -> None:
    """Build a v3.0 dataset with a corrupt video shard (triggers VIDEO.DECODABLE_SPOTCHECK FAIL)."""
    build_v3_dataset(root, camera=camera)
    (root / "videos" / camera / "chunk-000" / "file-000.mp4").write_bytes(
        b"NOT_A_VALID_MP4_FILE_JUST_GARBAGE_BYTES"
    )


def _write_real_mp4(path: Path, *, num_frames: int = 5, fps: int = 30) -> None:
    """Write a minimal but genuinely decodable MP4 using PyAV.

    Produces ``num_frames`` solid-colour frames at 16x16 (the smallest
    even resolution yuv420p accepts) encoded with libx264/ultrafast.
    No numpy: the RGB frame buffer is filled via ctypes.

    Codec choice mirrors LeRobot's default encoder
    (lerobot/datasets/video_utils.py encode_video_frames → libx264).
    """
    import ctypes

    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "ultrafast"}
        for i in range(num_frames):
            frame = av.VideoFrame(16, 16, "rgb24")
            frame.pts = i
            # Solid blue, varying slightly per frame so the encoder doesn't
            # collapse to a single I-frame and skip decoding.
            blue = min(255, 80 + i * 30)
            buf = (ctypes.c_uint8 * (16 * 16 * 3))()
            for j in range(16 * 16):
                buf[j * 3] = 0
                buf[j * 3 + 1] = 0
                buf[j * 3 + 2] = blue
            frame.planes[0].update(bytes(buf))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def build_v3_real_video(root: Path, *, camera: str = "top") -> None:
    """Build a v3.0 dataset with a genuinely decodable MP4 video shard.

    Replaces the placeholder b'\\x00' stub written by build_v3_dataset with
    a real libx264-encoded MP4 that PyAV can open and decode.  Used to
    exercise VIDEO.DECODABLE_SPOTCHECK's success path.
    """
    build_v3_dataset(root, camera=camera)
    video_path = root / "videos" / camera / "chunk-000" / "file-000.mp4"
    _write_real_mp4(video_path)


def build_v3_wrong_feature_shape(root: Path, *, camera: str = "top") -> None:
    """Build a v3.0 dataset where 'action' column has wrong declared shape.

    info.json declares action shape=[7] (names has 7 entries) but the Parquet
    column only has 3 elements per row.  Triggers SEMANTIC.FEATURE_DIMENSIONALITY FAIL.
    Only this check fires; all STRUCTURAL checks pass because the column exists
    and the declared dtype is correct.
    """
    build_v3_dataset(root, camera=camera)
    # Add a multi-element action column to info.json with shape mismatch.
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    # Declare action with 7 names but write 3 elements per row.
    info["features"]["action"] = {
        "dtype": "float32",
        "shape": [7],
        "names": ["j0", "j1", "j2", "j3", "j4", "j5", "j6"],
    }
    info_path.write_text(json.dumps(info))

    # Rewrite Parquet with a 3-element list instead of 7.
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    n_rows = old.num_rows
    action_col = pa.array(
        [[0.1, 0.2, 0.3]] * n_rows,
        type=pa.list_(pa.float32()),
    )
    new = old.append_column(pa.field("action", pa.list_(pa.float32())), action_col)
    pq.write_table(new, data_path)


def build_v3_with_action(root: Path, *, camera: str = "top") -> None:
    """Build a v3.0 dataset where action matches its declared shape exactly.

    Used as the clean-passes fixture for SEMANTIC.FEATURE_DIMENSIONALITY.
    action has shape=[3] and names=["j0","j1","j2"]; Parquet stores 3 floats.
    """
    build_v3_dataset(root, camera=camera)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"] = {
        "dtype": "float32",
        "shape": [3],
        "names": ["j0", "j1", "j2"],
    }
    info_path.write_text(json.dumps(info))

    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    n_rows = old.num_rows
    action_col = pa.array(
        [[0.1, 0.2, 0.3]] * n_rows,
        type=pa.list_(pa.float32()),
    )
    new = old.append_column(pa.field("action", pa.list_(pa.float32())), action_col)
    pq.write_table(new, data_path)


def build_v3_missing_task(root: Path, *, camera: str = "top") -> None:
    """Build a v3.0 dataset where frame data references a task_index not in tasks.parquet.

    task_index=99 appears in one frame row but only task_index=0 is defined.
    Triggers SEMANTIC.TASK_INTEGRITY FAIL.
    """
    build_v3_dataset(root, camera=camera)
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    # Change the last row's task_index to 99 (undefined).
    ti_col = old.column("task_index").to_pylist()
    ti_col[-1] = 99
    new = old.set_column(
        old.schema.get_field_index("task_index"),
        "task_index",
        pa.array(ti_col, type=pa.int64()),
    )
    pq.write_table(new, data_path)


def build_v3_empty_task_description(root: Path, *, camera: str = "top") -> None:
    """Build a v3.0 dataset where the task description is an empty string.

    All task_index references are valid but the mapped description is "".
    Triggers SEMANTIC.TASK_INTEGRITY FAIL.
    """
    build_v3_dataset(root, camera=camera)
    tasks_path = root / "meta" / "tasks.parquet"
    tasks_table = pa.table({"task_index": pa.array([0], type=pa.int64()), "task": pa.array([""])})
    pq.write_table(tasks_table, tasks_path)


def build_v3_no_language(root: Path, *, camera: str = "top") -> None:
    """Build a v3.0 dataset where episodes have empty string task descriptions.

    The task table has task_index=0 -> "" so all episodes lack a language label.
    Triggers SEMANTIC.LANGUAGE_PRESENT WARN.
    """
    build_v3_dataset(root, camera=camera)
    # Replace tasks in both the tasks table and episode metadata.
    tasks_path = root / "meta" / "tasks.parquet"
    pq.write_table(
        pa.table({"task_index": pa.array([0], type=pa.int64()), "task": pa.array([""])}),
        tasks_path,
    )
    # Update episode metadata to use empty task strings.
    ep_dir = root / "meta" / "episodes" / "chunk-000"
    ep_path = ep_dir / "file-000.parquet"
    old_ep = pq.read_table(ep_path)
    n_ep = old_ep.num_rows
    new_ep = old_ep.set_column(
        old_ep.schema.get_field_index("tasks"),
        "tasks",
        pa.array([[""] for _ in range(n_ep)], type=pa.list_(pa.string())),
    )
    pq.write_table(new_ep, ep_path)


def build_v3_with_intrinsics_plausible(root: Path, *, camera: str = "top") -> None:
    """Build a v3.0 dataset with a plausible camera_intrinsics column.

    Adds an 'observation.camera_intrinsics' feature with shape [9] containing
    a reasonable K matrix.  SEMANTIC.CAMERA_INTRINSICS_PLAUSIBLE should pass
    (return INFO with no violations).
    """
    build_v3_dataset(root, camera=camera)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.camera_intrinsics"] = {
        "dtype": "float32",
        "shape": [9],
        "names": None,
    }
    info_path.write_text(json.dumps(info))

    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    n_rows = old.num_rows
    # Plausible K: fx=500, 0, cx=320, 0, fy=500, cy=240, 0, 0, 1
    k_row = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
    k_col = pa.array([k_row] * n_rows, type=pa.list_(pa.float32()))
    new = old.append_column(
        pa.field("observation.camera_intrinsics", pa.list_(pa.float32())), k_col
    )
    pq.write_table(new, data_path)


def build_v3_with_intrinsics_implausible(root: Path, *, camera: str = "top") -> None:
    """Build a v3.0 dataset with an implausible camera_intrinsics column.

    Focal lengths are negative, which is physically impossible.
    SEMANTIC.CAMERA_INTRINSICS_PLAUSIBLE should fire (INFO with violations).
    """
    build_v3_dataset(root, camera=camera)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.camera_intrinsics"] = {
        "dtype": "float32",
        "shape": [9],
        "names": None,
    }
    info_path.write_text(json.dumps(info))

    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    n_rows = old.num_rows
    # Implausible K: negative focal lengths.
    k_row = [-500.0, 0.0, 320.0, 0.0, -500.0, 240.0, 0.0, 0.0, 1.0]
    k_col = pa.array([k_row] * n_rows, type=pa.list_(pa.float32()))
    new = old.append_column(
        pa.field("observation.camera_intrinsics", pa.list_(pa.float32())), k_col
    )
    pq.write_table(new, data_path)


# ---------------------------------------------------------------------------
# M6 STATISTICAL fixture builders
# ---------------------------------------------------------------------------


def _write_stats_json(root: Path, stats: dict[str, Any]) -> None:
    """Write meta/stats.json with the given stats dict."""
    (root / "meta" / "stats.json").write_text(json.dumps(stats))


def build_v3_with_correct_stats(root: Path, *, camera: str = "top") -> None:
    """Build a v3.0 dataset with stats.json matching the actual data.

    The timestamp column in the fixture has values [0, 1/30, 2/30, 3/30] per
    episode (3 episodes).  We compute the correct global mean/std and write
    them into stats.json so STATISTICAL.STATS_MATCH_DATA passes.
    """
    build_v3_dataset(root, camera=camera)
    # All timestamps: 0, 1/30, 2/30, 3/30 repeated num_episodes times.
    # Mean = (0 + 1/30 + 2/30 + 3/30) / 4 = 6/(30*4) = 0.05
    # But stored as float32 so we match the actual stored precision.
    all_ts = [float(f / 30.0) for ep in range(3) for f in range(FRAMES_PER_EPISODE)]
    n = len(all_ts)
    mean = sum(all_ts) / n
    variance = sum((x - mean) ** 2 for x in all_ts) / n
    import math as _math

    std = _math.sqrt(variance)
    _write_stats_json(
        root,
        {
            "timestamp": {
                "mean": mean,
                "std": std,
                "min": min(all_ts),
                "max": max(all_ts),
                "count": n,
            }
        },
    )


def build_v3_with_wrong_stats(root: Path, *, camera: str = "top") -> None:
    """Build a v3.0 dataset where stats.json has wrong mean for 'timestamp'.

    The true mean of the timestamp column is ~0.05 but stats.json claims 0.9
    (a ~1700% relative error).  Triggers STATISTICAL.STATS_MATCH_DATA FAIL.
    Only this check fires; structural checks pass because the data is valid.
    """
    build_v3_with_correct_stats(root, camera=camera)
    # Overwrite with a wrong mean.
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["timestamp"]["mean"] = 0.9  # Correct is ~0.05; delta >> rtol.
    stats_path.write_text(json.dumps(stats))


def build_v3_with_per_episode_stats(
    root: Path, *, camera: str = "top", corrupt: bool = False
) -> None:
    """Build a v3.0 dataset with per-episode stats inline in episode metadata.

    When corrupt=False: stats match the data (STATISTICAL.PER_EPISODE_STATS_MATCH passes).
    When corrupt=True:  stats are wrong (WARN fires).
    """
    build_v3_dataset(root, camera=camera)

    ep_dir = root / "meta" / "episodes" / "chunk-000"
    ep_path = ep_dir / "file-000.parquet"
    old_ep = pq.read_table(ep_path)

    # Per-episode timestamp stats: each episode has [0, 1/30, 2/30, 3/30].
    n_ep = old_ep.num_rows
    ep_ts_mean = sum(f / 30.0 for f in range(FRAMES_PER_EPISODE)) / FRAMES_PER_EPISODE
    ep_ts_var = sum((f / 30.0 - ep_ts_mean) ** 2 for f in range(FRAMES_PER_EPISODE))
    ep_ts_var /= FRAMES_PER_EPISODE
    import math as _math

    ep_ts_std = _math.sqrt(ep_ts_var)

    mean_val = 9.9 if corrupt else ep_ts_mean
    std_val = 9.9 if corrupt else ep_ts_std

    mean_col = pa.array([mean_val] * n_ep, type=pa.float64())
    std_col = pa.array([std_val] * n_ep, type=pa.float64())

    new_ep = old_ep
    new_ep = new_ep.append_column(pa.field("stats/timestamp/mean", pa.float64()), mean_col)
    new_ep = new_ep.append_column(pa.field("stats/timestamp/std", pa.float64()), std_col)
    pq.write_table(new_ep, ep_path)


def build_v3_all_nan_action(root: Path, *, camera: str = "top") -> None:
    """Build a v3.0 dataset where the 'action' column is entirely NaN.

    Triggers STATISTICAL.VALUE_SANITY FAIL (all-NaN case).
    """
    build_v3_dataset(root, camera=camera)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"] = {"dtype": "float32", "shape": [1], "names": None}
    info_path.write_text(json.dumps(info))

    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    n_rows = old.num_rows
    import math as _math

    action_col = pa.array([_math.nan] * n_rows, type=pa.float32())
    new = old.append_column(pa.field("action", pa.float32()), action_col)
    pq.write_table(new, data_path)


def build_v3_constant_action(root: Path, *, camera: str = "top") -> None:
    """Build a v3.0 dataset where the 'action' column is constant across an episode.

    Triggers STATISTICAL.VALUE_SANITY WARN (constant value, may be implausible).
    Action is 0.5 for every frame — not zero and not NaN, so severity is WARN.
    """
    build_v3_dataset(root, camera=camera)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"] = {"dtype": "float32", "shape": [1], "names": None}
    info_path.write_text(json.dumps(info))

    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    n_rows = old.num_rows
    action_col = pa.array([0.5] * n_rows, type=pa.float32())
    new = old.append_column(pa.field("action", pa.float32()), action_col)
    pq.write_table(new, data_path)


def build_v3_all_zero_action(root: Path, *, camera: str = "top") -> None:
    """Build a v3.0 dataset where the 'action' column is all-zero across an episode.

    Triggers STATISTICAL.VALUE_SANITY WARN (all-zero case).
    """
    build_v3_dataset(root, camera=camera)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"] = {"dtype": "float32", "shape": [1], "names": None}
    info_path.write_text(json.dumps(info))

    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    n_rows = old.num_rows
    action_col = pa.array([0.0] * n_rows, type=pa.float32())
    new = old.append_column(pa.field("action", pa.float32()), action_col)
    pq.write_table(new, data_path)


def build_v3_varying_action(root: Path, *, camera: str = "top") -> None:
    """Build a v3.0 dataset where 'action' varies sensibly across frames.

    Used as the clean-passes fixture for STATISTICAL.VALUE_SANITY.
    Action values cycle through [0.1, 0.2, 0.3, 0.4] — non-constant, non-NaN.
    """
    build_v3_dataset(root, camera=camera)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"] = {"dtype": "float32", "shape": [1], "names": None}
    info_path.write_text(json.dumps(info))

    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    n_rows = old.num_rows
    action_col = pa.array([0.1 + (i % 4) * 0.1 for i in range(n_rows)], type=pa.float32())
    new = old.append_column(pa.field("action", pa.float32()), action_col)
    pq.write_table(new, data_path)
