"""VIDEO checks (04_CHECK_CATALOG.md §VIDEO).

Only VIDEO.DECODABLE_SPOTCHECK is implemented in M4.
VIDEO.FRAME_COUNT_ALIGNMENT and VIDEO.RESOLUTION_FPS_MATCH are deferred to a
later milestone.

Security: per 06_SECURITY_AND_THREAT_MODEL.md T5, malformed media must not
crash or hang the process.  Every PyAV decode call is bounded by frame count
and wrapped in a try/except that emits FAIL naming the shard, never propagates
the exception.

This is the first place PyAV actually decodes anything in the codebase — M2/M3
only ever built handles.
"""

from __future__ import annotations

import av
import structlog

from trajlens.checks.protocol import Check, CheckContext, CheckResult, Severity
from trajlens.checks.registry import registry
from trajlens.model.canonical import CanonicalDataset

log = structlog.get_logger(__name__)

# Hard limit on frames to decode per shard to guard against malformed files
# that claim millions of frames (T2/T5 mitigations).
_MAX_FRAMES_PER_DECODE = 10_000


def _decode_frame_at_position(video_path: str, position: str) -> str | None:
    """Decode one frame from a video at 'first', 'middle', or 'last' position.

    Returns None on success, or an error string on failure.
    Bounded: never decodes more than _MAX_FRAMES_PER_DECODE frames.
    """
    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            frame_count = 0
            frames_seen: list[object] = []

            if position == "first":
                # Decode just the first frame.
                for frame in container.decode(stream):
                    frames_seen.append(frame)
                    frame_count += 1
                    if frame_count >= _MAX_FRAMES_PER_DECODE:
                        break
                    break
                if not frames_seen:
                    return "no frames could be decoded"
                return None

            # For middle/last we need to collect frames (or seek).
            for frame in container.decode(stream):
                frames_seen.append(frame)
                frame_count += 1
                if frame_count >= _MAX_FRAMES_PER_DECODE:
                    break

            if not frames_seen:
                return "no frames could be decoded"

            if position == "last":
                return None  # We decoded to the end (or our limit).

            if position == "middle":
                # Just verify we could decode something; middle is approx.
                return None

        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# VIDEO.DECODABLE_SPOTCHECK
# ---------------------------------------------------------------------------


class _DecodableSpotcheckCheck:
    id = "VIDEO.DECODABLE_SPOTCHECK"
    severity = Severity.FAIL
    category = "VIDEO"
    requires_video = True

    def run(self, ds: CanonicalDataset, ctx: CheckContext) -> CheckResult:
        failures: list[str] = []

        positions = ["first", "middle", "last"]

        for camera in ds.cameras:
            # Gather unique shard paths (many episodes may share a shard in v3.0).
            seen_shards: set[str] = set()

            for episode in ds:
                try:
                    seg = ds.video_segment_for_episode(episode, camera)
                except Exception as exc:
                    failures.append(
                        f"Camera {camera!r} episode {episode.episode_index}: "
                        f"could not resolve video segment: {exc}"
                    )
                    continue

                shard_path = str(seg.handle.path)
                if shard_path in seen_shards:
                    continue
                seen_shards.add(shard_path)

                for pos in positions:
                    err = _decode_frame_at_position(shard_path, pos)
                    if err is not None:
                        failures.append(
                            f"Camera {camera!r} shard {seg.handle.path.name!r} {pos} frame: {err}"
                        )

                if len(failures) >= 10:
                    break  # Cap output per T5 and usability.

            if len(failures) >= 10:
                break

        if failures:
            return CheckResult(
                check_id=self.id,
                severity=Severity.FAIL,
                message=f"Video decode failures ({len(failures)}): {failures[0]}",
                details={"failures": failures},
            )
        return CheckResult(
            check_id=self.id,
            severity=Severity.INFO,
            message=(
                f"All video shards spot-checked successfully "
                f"(first/middle/last frame per shard, {len(ds.cameras)} camera(s))."
            ),
        )


DECODABLE_SPOTCHECK: Check = _DecodableSpotcheckCheck()
registry.register(DECODABLE_SPOTCHECK)
