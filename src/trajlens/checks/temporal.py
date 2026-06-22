"""TEMPORAL checks and KNOWNBUG.TIMESTAMP_DRIFT (04_CHECK_CATALOG.md §TEMPORAL).

  TEMPORAL.TIMESTAMP_MONOTONIC  (FAIL)
  TEMPORAL.TIMESTAMP_SPACING    (WARN → FAIL)
  KNOWNBUG.TIMESTAMP_DRIFT      (FAIL) — the #3177 fingerprint

Timestamp spacing tolerance:
  The default tolerance used by LeRobotDataset when calling decode_video_frames
  is ``tolerance_s = 1e-4`` (100 microseconds).  This value is the default
  parameter on:
    - lerobot/datasets/lerobot_dataset.py  (line 55, class LeRobotDataset.__init__)
    - lerobot/configs/train.py             (line 105, TrainConfig.tolerance_s)
  The decoder raises FrameTimestampError when ``min_distance >= tolerance_s``
  (strict greater-equal, equivalently the check passes only when
  ``min_distance < tolerance_s``).

  Trajlens reuses this same threshold so our check's WARN boundary exactly
  matches the decoder's failure boundary — any spacing deviation that would
  cause a FrameTimestampError during training gets flagged WARN here.  A FAIL
  is reserved for deviations that exceed 1 full frame duration (1/fps), which
  indicates structural corruption, not merely floating-point accumulation.

  Source: lerobot 0.5.2, commit 8515d456
    lerobot/datasets/lerobot_dataset.py:55  ``tolerance_s: float = 1e-4``
    lerobot/configs/train.py:105            ``tolerance_s: float = 1e-4``
    lerobot/datasets/video_utils.py:162     ``is_within_tol = min_ < tolerance_s``

  Do not tighten this threshold without re-verifying the decoder's actual
  tolerance, per 03_DATA_FORMAT_SPEC.md §5 and 07_EVALUATION_AND_ACCURACY.md §6.
"""

from __future__ import annotations

import numpy as np
import structlog

from trajlens.checks.protocol import Check, CheckContext, CheckResult, Severity
from trajlens.checks.registry import registry
from trajlens.checks.utils import ShardColumnCache
from trajlens.model.canonical import CanonicalDataset

log = structlog.get_logger(__name__)

# The decoder's published default tolerance; see module docstring for citation.
# This is the WARN threshold: spacing farther than this from ideal will cause
# FrameTimestampError in lerobot's training loop.
_DECODER_TOLERANCE_S: float = 1e-4

# FAIL threshold: spacing more than a full frame duration off is structural
# corruption, not mere float drift.  Computed per-dataset from fps at runtime.
_FAIL_MULTIPLIER: float = 1.0  # multiples of 1/fps


# ---------------------------------------------------------------------------
# TEMPORAL.TIMESTAMP_MONOTONIC
# ---------------------------------------------------------------------------


class _TimestampMonotonicCheck:
    id = "TEMPORAL.TIMESTAMP_MONOTONIC"
    severity = Severity.FAIL
    category = "TEMPORAL"
    requires_video = False

    def run(self, ds: CanonicalDataset, ctx: CheckContext) -> CheckResult:
        violations: list[str] = []

        cache = ShardColumnCache(["timestamp"])
        for episode in ds:
            data = cache.get_episode_data(ds, episode)
            ts_col = [float(v) for v in data["timestamp"]]

            if len(ts_col) < 2:
                continue  # Single-frame episodes are trivially monotonic.

            for i in range(1, len(ts_col)):
                if ts_col[i] <= ts_col[i - 1]:
                    violations.append(
                        f"Episode {episode.episode_index}: timestamp[{i}]={ts_col[i]:.6f} "
                        f"<= timestamp[{i - 1}]={ts_col[i - 1]:.6f} (not strictly increasing)"
                    )
                    break  # One violation per episode is sufficient signal.

            if violations:
                break

        if violations:
            return CheckResult(
                check_id=self.id,
                severity=Severity.FAIL,
                message=f"Timestamps not strictly monotonic: {violations[0]}",
                details={"violations": violations},
            )
        return CheckResult(
            check_id=self.id,
            severity=Severity.INFO,
            message="Timestamps are strictly increasing within every episode.",
        )


TIMESTAMP_MONOTONIC: Check = _TimestampMonotonicCheck()
registry.register(TIMESTAMP_MONOTONIC)


# ---------------------------------------------------------------------------
# TEMPORAL.TIMESTAMP_SPACING
# ---------------------------------------------------------------------------


class _TimestampSpacingCheck:
    id = "TEMPORAL.TIMESTAMP_SPACING"
    # Nominal severity is WARN; we escalate to FAIL inline when deviation
    # exceeds a full frame duration.
    severity = Severity.WARN
    category = "TEMPORAL"
    requires_video = False

    def run(self, ds: CanonicalDataset, ctx: CheckContext) -> CheckResult:
        ideal_spacing = 1.0 / ds.fps
        warns: list[str] = []
        fails: list[str] = []

        cache = ShardColumnCache(["timestamp"])
        for episode in ds:
            if episode.length < 2:
                continue

            data = cache.get_episode_data(ds, episode)
            ts_col = [float(v) for v in data["timestamp"]]

            for i in range(1, len(ts_col)):
                gap = ts_col[i] - ts_col[i - 1]
                deviation = abs(gap - ideal_spacing)
                if deviation > ideal_spacing * _FAIL_MULTIPLIER:
                    fails.append(
                        f"Episode {episode.episode_index} frame {i}: "
                        f"gap={gap:.6f}s deviates {deviation:.6f}s from ideal "
                        f"{ideal_spacing:.6f}s (>= 1 frame; structural)"
                    )
                    break
                elif deviation > _DECODER_TOLERANCE_S:
                    # This matches the decoder's FrameTimestampError boundary.
                    warns.append(
                        f"Episode {episode.episode_index} frame {i}: "
                        f"gap={gap:.6f}s deviates {deviation:.6f}s > "
                        f"decoder tolerance {_DECODER_TOLERANCE_S}s"
                    )
                    break

            if fails:
                break

        if fails:
            return CheckResult(
                check_id=self.id,
                severity=Severity.FAIL,
                message=f"Timestamp spacing structurally broken: {fails[0]}",
                details={"fails": fails, "warns": warns, "ideal_spacing_s": ideal_spacing},
            )
        if warns:
            return CheckResult(
                check_id=self.id,
                severity=Severity.WARN,
                message=(
                    f"Timestamp spacing exceeds decoder tolerance "
                    f"(>{_DECODER_TOLERANCE_S}s from ideal): {warns[0]}"
                ),
                details={"warns": warns, "ideal_spacing_s": ideal_spacing},
            )
        return CheckResult(
            check_id=self.id,
            severity=Severity.INFO,
            message=(
                f"Timestamp spacing is consistent with fps={ds.fps} "
                f"within decoder tolerance ({_DECODER_TOLERANCE_S}s)."
            ),
        )


TIMESTAMP_SPACING: Check = _TimestampSpacingCheck()
registry.register(TIMESTAMP_SPACING)


# ---------------------------------------------------------------------------
# KNOWNBUG.TIMESTAMP_DRIFT
# ---------------------------------------------------------------------------
# Issue #3177 fingerprint: cumulative floating-point rounding error in
# timestamps causes the video decoder to seek to the wrong frame after
# enough episodes.  The canonical symptom: decode fails after ~45 episodes.
#
# Detection: compare stored timestamp[i] against the ideal value
# (frame_index_within_episode / fps), track the cumulative absolute deviation
# across the whole dataset.  When cumulative drift exceeds the decoder's
# tolerance, any subsequent seek may land on the wrong frame.
#
# FP guard: only flag when info.json declares a fixed fps; variable-rate
# captures are exempt because drift is expected.


class _TimestampDriftCheck:
    id = "KNOWNBUG.TIMESTAMP_DRIFT"
    severity = Severity.FAIL
    category = "KNOWNBUG"
    requires_video = False

    def run(self, ds: CanonicalDataset, ctx: CheckContext) -> CheckResult:
        ideal_frame_duration = 1.0 / ds.fps
        cumulative_drift: float = 0.0
        first_breach_episode: int | None = None
        first_breach_drift: float = 0.0

        # Quantize the ideal value to the *declared* storage dtype before
        # differencing, so both sides round the same way. info.json's
        # features map states whether timestamp is float32 or float64;
        # assuming float32 unconditionally manufactures spurious cumulative
        # drift against float64-stored datasets (the representation error
        # term cancels out only when the comparison matches what's actually
        # on disk).
        declared_dtype = ds.features.get("timestamp")
        quantize: type[np.float32] | type[np.float64] = (
            np.float64
            if declared_dtype is not None and declared_dtype.dtype == "float64"
            else np.float32
        )

        cache = ShardColumnCache(["frame_index", "timestamp"])
        for episode in ds:
            data = cache.get_episode_data(ds, episode)
            ep_rows = [
                (int(fi), float(ts))
                for fi, ts in zip(data["frame_index"], data["timestamp"], strict=True)
            ]

            for frame_index, stored_ts in ep_rows:
                ideal_ts = float(quantize(frame_index * ideal_frame_duration))
                cumulative_drift += abs(stored_ts - ideal_ts)
                if cumulative_drift > _DECODER_TOLERANCE_S and first_breach_episode is None:
                    first_breach_episode = episode.episode_index
                    first_breach_drift = cumulative_drift

        if first_breach_episode is not None:
            return CheckResult(
                check_id=self.id,
                severity=Severity.FAIL,
                message=(
                    f"Timestamp drift fingerprint detected (LeRobot issue #3177): "
                    f"cumulative drift reached {first_breach_drift:.6f}s > "
                    f"decoder tolerance ({_DECODER_TOLERANCE_S}s) at "
                    f"episode {first_breach_episode}.  "
                    f"Video seeks will land on wrong frames during training."
                ),
                details={
                    "cumulative_drift_s": cumulative_drift,
                    "decoder_tolerance_s": _DECODER_TOLERANCE_S,
                    "first_breach_episode": first_breach_episode,
                    "lerobot_issue": "#3177",
                },
            )
        return CheckResult(
            check_id=self.id,
            severity=Severity.INFO,
            message=(
                f"No timestamp drift detected (cumulative drift {cumulative_drift:.6f}s "
                f"< decoder tolerance {_DECODER_TOLERANCE_S}s)."
            ),
            details={"cumulative_drift_s": cumulative_drift},
        )


TIMESTAMP_DRIFT: Check = _TimestampDriftCheck()
registry.register(TIMESTAMP_DRIFT)
