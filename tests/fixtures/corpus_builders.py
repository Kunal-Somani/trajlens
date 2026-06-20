"""Additional labeled synthetic fixtures for 07_EVALUATION_AND_ACCURACY.md §1.

Corpus accounting (07 §1 table, synthetic-only scope for M6):

  Category                          Target  Pre-M6  M6-new  M6-total  This file adds
  -------                           ------  ------  ------  --------  --------------
  Clean, fully valid                20+       3       0        3        17  → 20
  Timestamp drift (#3177)           10+       1       0        1         9  → 10
  v2.1→v3.0 index corruption        10+       1       0        1         9  → 10
  Schema mismatch                    5+       1       1        2         3  →  5
  Missing metadata / bad paths       5+       2       0        2         3  →  5
  Stats divergence                   5+       0       2        2         3  →  5
  Video decode failure               5+       1       1        2         3  →  5
  -------                           ------                             ---
  Synthetic total                   60+                               25    38  → 63

Out-of-scope for synthetic milestone (M7 / post-launch):
  Video/data frame count mismatch    5+  (no check yet; VIDEO.FRAME_COUNT_ALIGNMENT deferred)
  Real Hub datasets, PASS           20+  (audit_hub.py, M7)
  Real Hub datasets, FAIL           20+  (audit_hub.py, M7)

Every builder in this file is one *independent labeled dataset instance*:
each produces a distinct on-disk layout with a unique ground-truth label
(the check that should fire, and whether the expected result is PASS/FAIL).
Multiple builders may reuse similar logic with different parameters — that
is intentional; parameter variation is what exercises threshold boundaries.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tests.fixtures.builders import (
    FRAMES_PER_EPISODE,
    _write_real_mp4,
    build_v2_dataset,
    build_v3_dataset,
    build_v3_metadata_data_disagreement,
    build_v3_timestamp_drift,
    build_v3_with_correct_stats,
)

# ---------------------------------------------------------------------------
# Category 1: Clean, fully valid datasets  (need 17 more → total 20)
# ---------------------------------------------------------------------------
# Varied episode counts, fps, num_cameras, and v2/v3 to give diversity.


def build_v3_clean_1ep(root: Path) -> None:
    """Clean v3.0 — 1 episode. Corpus label: PASS all checks."""
    build_v3_dataset(root, num_episodes=1)


def build_v3_clean_5ep(root: Path) -> None:
    """Clean v3.0 — 5 episodes. Corpus label: PASS all checks."""
    build_v3_dataset(root, num_episodes=5)


def build_v3_clean_10ep(root: Path) -> None:
    """Clean v3.0 — 10 episodes. Corpus label: PASS all checks."""
    build_v3_dataset(root, num_episodes=10)


def build_v3_clean_20ep(root: Path) -> None:
    """Clean v3.0 — 20 episodes. Corpus label: PASS all checks."""
    build_v3_dataset(root, num_episodes=20)


def build_v3_clean_50ep(root: Path) -> None:
    """Clean v3.0 — 50 episodes. Corpus label: PASS all checks."""
    build_v3_dataset(root, num_episodes=50)


def build_v3_clean_multishard(root: Path) -> None:
    """Clean v3.0 — 6 episodes spread across 2 episode shards.

    Exercises the multi-shard shard-resolver path.  Corpus label: PASS.
    """
    build_v3_dataset(root, num_episodes=6, episodes_per_shard=3)


def build_v3_clean_two_cameras(root: Path) -> None:
    """Clean v3.0 — 3 episodes, two cameras ('top' and 'wrist').

    Both cameras appear in info.json features AND in the episode metadata
    Parquet (videos/wrist/* columns).  Corpus label: PASS.
    """
    build_v3_dataset(root, num_episodes=3, camera="top")
    # Add wrist feature to info.json.
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["wrist"] = {"dtype": "video", "shape": [3, 64, 64], "names": None}
    info_path.write_text(json.dumps(info))
    # Add wrist video stub.
    wrist_dir = root / "videos" / "wrist" / "chunk-000"
    wrist_dir.mkdir(parents=True, exist_ok=True)
    (wrist_dir / "file-000.mp4").write_bytes(b"\x00")
    # Add wrist columns to the episode metadata Parquet (required by adapter).
    ep_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    old_ep = pq.read_table(ep_path)
    n_ep = old_ep.num_rows
    new_ep = old_ep
    for col_name, col_val in [
        ("videos/wrist/chunk_index", pa.array([0] * n_ep, type=pa.int64())),
        ("videos/wrist/file_index", pa.array([0] * n_ep, type=pa.int64())),
        (
            "videos/wrist/from_timestamp",
            pa.array([float(i * 4) / 30.0 for i in range(n_ep)], type=pa.float64()),
        ),
        (
            "videos/wrist/to_timestamp",
            pa.array([float((i + 1) * 4) / 30.0 for i in range(n_ep)], type=pa.float64()),
        ),
    ]:
        new_ep = new_ep.append_column(pa.field(col_name, col_val.type), col_val)
    pq.write_table(new_ep, ep_path)


def build_v2_clean_v20(root: Path) -> None:
    """Clean v2.0 dataset. Corpus label: PASS all checks."""
    build_v2_dataset(root, codebase_version="v2.0", num_episodes=3)


def build_v2_clean_v21_5ep(root: Path) -> None:
    """Clean v2.1 dataset — 5 episodes. Corpus label: PASS all checks."""
    build_v2_dataset(root, codebase_version="v2.1", num_episodes=5)


def build_v2_clean_v21_10ep(root: Path) -> None:
    """Clean v2.1 dataset — 10 episodes. Corpus label: PASS all checks."""
    build_v2_dataset(root, codebase_version="v2.1", num_episodes=10)


def build_v3_clean_with_correct_stats_5ep(root: Path) -> None:
    """Clean v3.0 — 5 episodes with correct stats.json. Corpus label: PASS."""
    build_v3_dataset(root, num_episodes=5)
    _write_correct_stats(root, num_episodes=5)


def build_v3_clean_with_correct_stats_10ep(root: Path) -> None:
    """Clean v3.0 — 10 episodes with correct stats.json. Corpus label: PASS."""
    build_v3_dataset(root, num_episodes=10)
    _write_correct_stats(root, num_episodes=10)


def build_v3_clean_0ep(root: Path) -> None:
    """Clean v3.0 — 0 episodes (empty dataset). Corpus label: PASS."""
    build_v3_dataset(root, num_episodes=0)


def build_v3_clean_large_fps(root: Path) -> None:
    """Clean v3.0 — 3 episodes, fps=60 declared in info.json. Corpus label: PASS."""
    build_v3_dataset(root, num_episodes=3)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["fps"] = 60
    info_path.write_text(json.dumps(info))


def build_v3_clean_single_frame_ep(root: Path) -> None:
    """Clean v3.0 — 3 episodes each with 1 frame (edge case for spacing checks).

    Reuses build_v3_dataset but overwrites data with 1 frame per episode.
    Corpus label: PASS (single-frame episode is trivially monotonic/no-spacing).
    """
    build_v3_dataset(root, num_episodes=3)
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    rows = []
    for ep in range(3):
        rows.append(
            {
                "timestamp": 0.0,
                "frame_index": 0,
                "episode_index": ep,
                "index": ep,
                "task_index": 0,
            }
        )
    table = pa.table(
        {
            "timestamp": pa.array([r["timestamp"] for r in rows], type=pa.float32()),
            "frame_index": pa.array([r["frame_index"] for r in rows], type=pa.int64()),
            "episode_index": pa.array([r["episode_index"] for r in rows], type=pa.int64()),
            "index": pa.array([r["index"] for r in rows], type=pa.int64()),
            "task_index": pa.array([r["task_index"] for r in rows], type=pa.int64()),
        }
    )
    pq.write_table(table, data_path)


def build_v3_clean_with_action(root: Path) -> None:
    """Clean v3.0 — 3 episodes with valid action column (shape matches).

    Corpus label: PASS SEMANTIC.FEATURE_DIMENSIONALITY.
    """
    build_v3_dataset(root, num_episodes=3)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"] = {"dtype": "float32", "shape": [6], "names": None}
    info_path.write_text(json.dumps(info))
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    n = old.num_rows
    col = pa.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]] * n, type=pa.list_(pa.float32()))
    pq.write_table(old.append_column(pa.field("action", pa.list_(pa.float32())), col), data_path)


def build_v3_clean_robot_type(root: Path) -> None:
    """Clean v3.0 — robot_type field present. Corpus label: PASS."""
    build_v3_dataset(root, num_episodes=3)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["robot_type"] = "so100"
    info_path.write_text(json.dumps(info))


def _write_correct_stats(root: Path, *, num_episodes: int) -> None:
    """Write a correct stats.json for a standard build_v3_dataset fixture."""
    all_ts = [float(f / 30.0) for _ in range(num_episodes) for f in range(FRAMES_PER_EPISODE)]
    n = len(all_ts)
    if n == 0:
        return
    mean = sum(all_ts) / n
    variance = sum((x - mean) ** 2 for x in all_ts) / n
    std = math.sqrt(variance)
    stats = {
        "timestamp": {
            "mean": mean,
            "std": std,
            "min": min(all_ts),
            "max": max(all_ts),
            "count": n,
        }
    }
    (root / "meta" / "stats.json").write_text(json.dumps(stats))


# ---------------------------------------------------------------------------
# Category 2: Timestamp drift (#3177)  (need 9 more → total 10)
# ---------------------------------------------------------------------------
# Varied episode counts and drift magnitudes, including a sub-threshold case.


def build_v3_drift_3ep_mild(root: Path) -> None:
    """Timestamp drift — 3 episodes, drift=5e-5/frame (above threshold). FAIL."""
    build_v3_timestamp_drift(root, num_episodes=3, drift_per_frame=5e-5)


def build_v3_drift_5ep_moderate(root: Path) -> None:
    """Timestamp drift — 5 episodes, drift=1e-4/frame. FAIL."""
    build_v3_timestamp_drift(root, num_episodes=5, drift_per_frame=1e-4)


def build_v3_drift_10ep_heavy(root: Path) -> None:
    """Timestamp drift — 10 episodes, drift=2e-4/frame. FAIL."""
    build_v3_timestamp_drift(root, num_episodes=10, drift_per_frame=2e-4)


def build_v3_drift_20ep(root: Path) -> None:
    """Timestamp drift — 20 episodes, drift=5e-5/frame. FAIL."""
    build_v3_timestamp_drift(root, num_episodes=20, drift_per_frame=5e-5)


def build_v3_drift_50ep(root: Path) -> None:
    """Timestamp drift — 50 episodes, drift=5e-5/frame. FAIL (large dataset)."""
    build_v3_timestamp_drift(root, num_episodes=50, drift_per_frame=5e-5)


def build_v3_drift_subthreshold(root: Path) -> None:
    """Timestamp drift — 5 episodes, drift=1e-7/frame (below tolerance). PASS.

    Tests that the check does NOT fire false positives at sub-threshold drift.
    """
    build_v3_timestamp_drift(root, num_episodes=5, drift_per_frame=1e-7)


def build_v3_drift_last_episode_only(root: Path) -> None:
    """Timestamp drift injected only in the last episode. FAIL.

    Verifies the check scans all episodes, not just the first.
    """
    build_v3_dataset(root, num_episodes=5)
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    ts = old.column("timestamp").to_pylist()
    ep = old.column("episode_index").to_pylist()
    last_ep = max(ep)
    last_indices = [i for i, e in enumerate(ep) if e == last_ep]
    for rank, idx in enumerate(last_indices):
        ts[idx] = ts[idx] + rank * 5e-5
    new = old.set_column(
        old.schema.get_field_index("timestamp"),
        "timestamp",
        pa.array(ts, type=pa.float32()),
    )
    pq.write_table(new, data_path)


def build_v3_drift_first_episode_only(root: Path) -> None:
    """Timestamp drift injected only in the first episode. FAIL."""
    build_v3_dataset(root, num_episodes=5)
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    ts = old.column("timestamp").to_pylist()
    ep = old.column("episode_index").to_pylist()
    first_indices = [i for i, e in enumerate(ep) if e == 0]
    for rank, idx in enumerate(first_indices):
        ts[idx] = ts[idx] + rank * 5e-5
    new = old.set_column(
        old.schema.get_field_index("timestamp"),
        "timestamp",
        pa.array(ts, type=pa.float32()),
    )
    pq.write_table(new, data_path)


def build_v3_drift_large_constant_offset(root: Path) -> None:
    """Episode 0 timestamps all shifted by +1.0 — large constant offset.

    The TIMESTAMP_DRIFT check measures cumulative |ts - expected| residual;
    a constant +1.0 offset per frame exceeds the 1e-4 s tolerance and
    correctly FAILS.  Corpus label: FAIL (large constant offset IS detected
    by #3177 fingerprint — it produces the same video-seek error).
    """
    build_v3_dataset(root, num_episodes=3)
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    ts = old.column("timestamp").to_pylist()
    ep = old.column("episode_index").to_pylist()
    for i, e in enumerate(ep):
        if e == 0:
            ts[i] = ts[i] + 1.0  # constant offset, not growing
    new = old.set_column(
        old.schema.get_field_index("timestamp"),
        "timestamp",
        pa.array(ts, type=pa.float32()),
    )
    pq.write_table(new, data_path)


# ---------------------------------------------------------------------------
# Category 3: v2.1→v3.0 index corruption (#2401)  (need 9 more → total 10)
# ---------------------------------------------------------------------------


def build_v3_corruption_from_index_too_low(root: Path) -> None:
    """#2401: dataset_from_index is 1 less than actual start. FAIL."""
    build_v3_dataset(root, num_episodes=3)
    ep_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(ep_path)
    from_col = old.column("dataset_from_index").to_pylist()
    corrupted = [max(0, v - 1) for v in from_col]
    new = old.set_column(
        old.schema.get_field_index("dataset_from_index"),
        "dataset_from_index",
        pa.array(corrupted, type=pa.int64()),
    )
    pq.write_table(new, ep_path)


def build_v3_corruption_5ep(root: Path) -> None:
    """#2401: to_index off by 1, 5 episodes. FAIL."""
    build_v3_metadata_data_disagreement(root, num_episodes=5)


def build_v3_corruption_10ep(root: Path) -> None:
    """#2401: to_index off by 1, 10 episodes. FAIL."""
    build_v3_metadata_data_disagreement(root, num_episodes=10)


def build_v3_corruption_20ep(root: Path) -> None:
    """#2401: to_index off by 1, 20 episodes. FAIL."""
    build_v3_metadata_data_disagreement(root, num_episodes=20)


def build_v3_corruption_to_index_too_high_by_2(root: Path) -> None:
    """#2401: to_index off by 2 (not just 1). FAIL."""
    build_v3_dataset(root, num_episodes=3)
    ep_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(ep_path)
    to_col = old.column("dataset_to_index").to_pylist()
    new = old.set_column(
        old.schema.get_field_index("dataset_to_index"),
        "dataset_to_index",
        pa.array([v + 2 for v in to_col], type=pa.int64()),
    )
    pq.write_table(new, ep_path)


def build_v3_corruption_episode_length_mismatch(root: Path) -> None:
    """#2401: episode 'length' field disagrees with from/to span. FAIL."""
    build_v3_dataset(root, num_episodes=3)
    ep_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(ep_path)
    length_col = old.column("length").to_pylist()
    new = old.set_column(
        old.schema.get_field_index("length"),
        "length",
        pa.array([v + 1 for v in length_col], type=pa.int64()),
    )
    pq.write_table(new, ep_path)


def build_v3_corruption_multishard_3ep(root: Path) -> None:
    """#2401: to_index off by 1, 6 episodes spread across 2 shards. FAIL."""
    build_v3_dataset(root, num_episodes=6, episodes_per_shard=3)
    for chunk in range(2):
        ep_path = root / "meta" / "episodes" / f"chunk-{chunk:03d}" / "file-000.parquet"
        if not ep_path.exists():
            continue
        old = pq.read_table(ep_path)
        to_col = old.column("dataset_to_index").to_pylist()
        new = old.set_column(
            old.schema.get_field_index("dataset_to_index"),
            "dataset_to_index",
            pa.array([v + 1 for v in to_col], type=pa.int64()),
        )
        pq.write_table(new, ep_path)


def build_v3_corruption_partial_only_last_ep(root: Path) -> None:
    """#2401: only the last episode's to_index is wrong. FAIL (partial corruption)."""
    build_v3_dataset(root, num_episodes=5)
    ep_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(ep_path)
    to_col = old.column("dataset_to_index").to_pylist()
    to_col[-1] += 1
    new = old.set_column(
        old.schema.get_field_index("dataset_to_index"),
        "dataset_to_index",
        pa.array(to_col, type=pa.int64()),
    )
    pq.write_table(new, ep_path)


def build_v3_clean_boundary_exact(root: Path) -> None:
    """Clean boundaries: to_index == from_index + FRAMES_PER_EPISODE exactly. PASS."""
    build_v3_dataset(root, num_episodes=5)


# ---------------------------------------------------------------------------
# Category 4: Schema mismatch  (need 3 more → total 5)
# ---------------------------------------------------------------------------


def build_v3_schema_frame_index_wrong_dtype(root: Path) -> None:
    """frame_index stored as float32 instead of int64. FAIL SCHEMA_CONSISTENCY."""
    build_v3_dataset(root)
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    n = old.num_rows
    new = old.set_column(
        old.schema.get_field_index("frame_index"),
        "frame_index",
        pa.array([float(v) for v in range(n)], type=pa.float32()),
    )
    pq.write_table(new, data_path)


def build_v3_schema_episode_index_wrong_dtype(root: Path) -> None:
    """episode_index stored as float32 instead of int64. FAIL SCHEMA_CONSISTENCY."""
    build_v3_dataset(root)
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    ep_list = old.column("episode_index").to_pylist()
    new = old.set_column(
        old.schema.get_field_index("episode_index"),
        "episode_index",
        pa.array([float(v) for v in ep_list], type=pa.float32()),
    )
    pq.write_table(new, data_path)


def build_v3_schema_task_index_wrong_dtype(root: Path) -> None:
    """task_index stored as int32 instead of int64. FAIL SCHEMA_CONSISTENCY."""
    build_v3_dataset(root)
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    n = old.num_rows
    new = old.set_column(
        old.schema.get_field_index("task_index"),
        "task_index",
        pa.array([0] * n, type=pa.int32()),
    )
    pq.write_table(new, data_path)


# ---------------------------------------------------------------------------
# Category 5: Missing metadata / bad paths  (need 3 more → total 5)
# ---------------------------------------------------------------------------


def build_v3_missing_episodes_dir(root: Path) -> None:
    """meta/episodes/ directory removed entirely. FAIL PATH_TEMPLATE_RESOLVES."""
    build_v3_dataset(root)
    import shutil

    shutil.rmtree(root / "meta" / "episodes")


def build_v3_missing_tasks_parquet(root: Path) -> None:
    """meta/tasks.parquet deleted. Triggers loader/structural error. FAIL."""
    build_v3_dataset(root)
    (root / "meta" / "tasks.parquet").unlink()


def build_v3_missing_video_shard(root: Path, *, camera: str = "top") -> None:
    """Video shard deleted (not data shard). FAIL VIDEO.DECODABLE_SPOTCHECK."""
    build_v3_dataset(root, camera=camera)
    (root / "videos" / camera / "chunk-000" / "file-000.mp4").unlink()


# ---------------------------------------------------------------------------
# Category 6: Stats divergence  (need 3 more → total 5)
# ---------------------------------------------------------------------------


def build_v3_stats_wrong_std(root: Path) -> None:
    """stats.json std is wrong (100x too large). FAIL STATS_MATCH_DATA."""
    build_v3_with_correct_stats(root)
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["timestamp"]["std"] = stats["timestamp"]["std"] * 100.0
    stats_path.write_text(json.dumps(stats))


def build_v3_stats_wrong_min(root: Path) -> None:
    """stats.json min is wrong (-99). Only mean/std are checked; PASS STATS_MATCH_DATA.

    Verifies the check doesn't false-positive on min/max fields it doesn't validate.
    """
    build_v3_with_correct_stats(root)
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["timestamp"]["min"] = -99.0
    stats_path.write_text(json.dumps(stats))


def build_v3_stats_diverged_5ep(root: Path) -> None:
    """5-episode dataset with wrong stats.json mean. FAIL STATS_MATCH_DATA."""
    build_v3_dataset(root, num_episodes=5)
    _write_correct_stats(root, num_episodes=5)
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["timestamp"]["mean"] = 9.9
    stats_path.write_text(json.dumps(stats))


# ---------------------------------------------------------------------------
# Category 7: Video decode failure  (need 3 more → total 5)
# ---------------------------------------------------------------------------


def build_v3_video_truncated(root: Path, *, camera: str = "top") -> None:
    """MP4 shard is a valid header but truncated mid-stream. FAIL DECODABLE_SPOTCHECK."""
    build_v3_dataset(root, camera=camera)
    # Write the first 12 bytes of a real ftyp box then stop — truncated.
    (root / "videos" / camera / "chunk-000" / "file-000.mp4").write_bytes(
        b"\x00\x00\x00\x0cftypisom" + b"\x00" * 8
    )


def build_v3_video_empty_file(root: Path, *, camera: str = "top") -> None:
    """MP4 shard is a zero-byte file. FAIL DECODABLE_SPOTCHECK."""
    build_v3_dataset(root, camera=camera)
    (root / "videos" / camera / "chunk-000" / "file-000.mp4").write_bytes(b"")


def build_v3_video_real_two_cameras(root: Path) -> None:
    """Two real decodable MP4 shards (top + wrist). PASS DECODABLE_SPOTCHECK."""
    build_v3_clean_two_cameras(root)
    _write_real_mp4(root / "videos" / "top" / "chunk-000" / "file-000.mp4")
    _write_real_mp4(root / "videos" / "wrist" / "chunk-000" / "file-000.mp4")
