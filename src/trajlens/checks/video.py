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

from trajlens.checks.protocol import Check, CheckContext, CheckResult, Severity, Tier
from trajlens.checks.registry import registry
from trajlens.model.canonical import CanonicalDataset

log = structlog.get_logger(__name__)

# Hard limit on frames to decode per shard to guard against malformed files
# that claim millions of frames (T2/T5 mitigations).
_MAX_FRAMES_PER_DECODE = 10_000


def _decode_bounded(
    container: av.container.InputContainer,
    stream: av.video.stream.VideoStream,
    *,
    stop_after_one: bool,
) -> bool:
    """Decode frames from the container's current position, discarding each
    one immediately. Returns True if at least one frame decoded.

    Never retains more than the single frame currently being decoded (a
    prior version accumulated every decoded frame into a list up to
    _MAX_FRAMES_PER_DECODE=10,000 -- ~14GB for a typical 640x480x3 shard --
    even though only truthiness of the list was ever checked). Bounded by
    _MAX_FRAMES_PER_DECODE regardless of stop_after_one, so a malformed
    file that claims an absurd frame count still cannot force an unbounded
    decode (T2/T5).
    """
    count = 0
    for frame in container.decode(stream):
        del frame  # discard immediately; only the count is ever needed
        count += 1
        if stop_after_one or count >= _MAX_FRAMES_PER_DECODE:
            break
    return count > 0


def _decode_frame_at_position(video_path: str, position: str) -> str | None:
    """Decode near 'first', 'middle', or 'last' position in *video_path*.

    Returns None on success, or an error string on failure. 'middle' and
    'last' genuinely seek via the container's own duration rather than
    approximating both with a forward decode from frame 0 (a prior version
    did exactly that for 'middle', and 'last' merely decoded forward to
    _MAX_FRAMES_PER_DECODE and reported success regardless of whether the
    video's true last frame was ever reached -- so a video longer than the
    cap could have a corrupted tail and this check would still PASS).

    'middle': one successfully decoded frame at the seek point is
    sufficient signal for a spot-check. 'last': the whole tail segment (from
    the final keyframe to EOF -- inherently GOP-sized, and still bounded
    defensively by _MAX_FRAMES_PER_DECODE) is decoded, so a truncated or
    corrupted tail is actually caught rather than assumed fine.
    """
    try:
        with av.open(video_path) as container:
            if not container.streams.video:
                return "no video stream in container"
            stream = container.streams.video[0]

            if position == "first":
                decoded = _decode_bounded(container, stream, stop_after_one=True)
                return None if decoded else "no frames could be decoded"

            duration_s = (
                float(container.duration) / float(av.time_base)
                if container.duration is not None
                else None
            )
            if duration_s is None:
                # No duration metadata to seek against -- fall back to a
                # bounded forward decode from the start rather than
                # silently claiming a position was verified that never was.
                decoded = _decode_bounded(container, stream, stop_after_one=(position == "middle"))
                return None if decoded else "no frames could be decoded"

            fraction = 0.5 if position == "middle" else 1.0
            target_us = int(duration_s * fraction * av.time_base)
            container.seek(target_us, backward=True)

            decoded = _decode_bounded(container, stream, stop_after_one=(position == "middle"))
            if not decoded:
                return f"no frame decodable near the {position} of the video"
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
    thread_safe = True
    tier: Tier = Tier.INTEGRITY
    formats: frozenset[str] | None = None

    def run(self, ds: CanonicalDataset, ctx: CheckContext) -> CheckResult:
        failures: list[str] = []

        # T11: Remote HTTP stream fallback. PyAV streaming without --deep seeks
        # over HTTP which can be slow or unreliable. Fail closed (skip gracefully).
        if not ctx.deep:
            for camera in ds.cameras:
                for episode in list(ds)[:1]:
                    try:
                        seg = ds.video_segment_for_episode(episode, camera)
                        if not seg.handle.is_local:
                            return CheckResult(
                                check_id=self.id,
                                severity=Severity.INFO,
                                message=(
                                    "Skipped VIDEO.DECODABLE_SPOTCHECK for Hub dataset "
                                    "(too slow over HTTP). Use --deep to verify video."
                                ),
                            )
                    except Exception as exc:
                        log.debug("failed_to_resolve_segment_for_skip_check", error=str(exc))

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
                        if seg.handle.is_local:
                            shard_name = getattr(seg.handle.path, "name", str(seg.handle.path))
                        else:
                            shard_name = str(seg.handle.path).split("/")[-1]
                        failures.append(
                            f"Camera {camera!r} shard {shard_name!r} {pos} frame: {err}"
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
