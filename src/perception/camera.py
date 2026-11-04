"""
camera.py
─────────
Wrapper for Intel RealSense D455 camera + a MockCamera for sim/test.

D455Camera lazy-imports pyrealsense2 → this module is importable on machines
without the RealSense SDK (only MockCamera works in that case).

Both cameras expose the same interface:
    .intrinsics  → dict {fx, fy, ppx, ppy, width, height}
    .get_frame() → (rgb, depth_m) — depth_m in metres, or (None, None)
    .last_frame_clock → instants of the frame get_frame returned last (frame_clock)
    .stop()
"""
from __future__ import annotations

import ctypes
import logging
import sys
import time

import numpy as np

logger = logging.getLogger(__name__)

# 100 ns intervals from 1601-01-01 (FILETIME) to 1970-01-01 (Unix epoch).
_FILETIME_UNIX_EPOCH = 116444736000000000
if sys.platform == "win32":
    _precise_filetime = ctypes.windll.kernel32.GetSystemTimePreciseAsFileTime
    _filetime = ctypes.c_ulonglong()


def system_ms() -> float:
    """Wall-clock time in ms since the Unix epoch, read the way librealsense reads
    its system time (std::chrono::system_clock, GetSystemTimePreciseAsFileTime
    under MSVC). CPython 3.10's time.time() reads the coarser
    GetSystemTimeAsFileTime on Windows, so it is not used there."""
    if sys.platform == "win32":
        _precise_filetime(ctypes.byref(_filetime))
        return (_filetime.value - _FILETIME_UNIX_EPOCH) / 1e4
    return time.time() * 1e3


def frame_clock(frames, t_got: float, rs, now_ms=system_ms, perf=time.perf_counter) -> dict:
    """Instants of one frameset on the time.perf_counter axis, in seconds.

    got      wait_for_frames returned the frameset (t_got).
    arrival  the colour frame reached librealsense from the USB backend
             (TIME_OF_ARRIVAL, system clock), after exposure, readout and the
             transfer to the host; None when the frame does not carry it.
    capture  the best estimate of the colour exposure: with GLOBAL_TIME
             timestamps (Windows metadata registered, global time on) the frame
             timestamp, the start of readout, moved back to mid-exposure by
             FRAME_TIMESTAMP - SENSOR_TIMESTAMP when both are present; otherwise
             the arrival; otherwise got. src names the one used. Only global time
             approaches the exposure, with an accuracy librealsense does not state;
             arrival and got trail it by the camera-to-host latency, so the age
             they give is a lower bound.
    System-clock times are mapped through a (system, perf_counter) pair read
    here: the narrowest of three, since the ctypes call releases the GIL and
    another thread can widen one. Only reads: the frameset is not changed.
    """
    pairs = []
    for _ in range(3):
        a = perf()
        s = now_ms()
        b = perf()
        pairs.append((b - a, 0.5 * (a + b), s))
    _, mid, s = min(pairs)

    def to_perf(ms):
        return None if ms is None else mid + (float(ms) - s) / 1e3

    md = rs.frame_metadata_value

    def meta(frame, key):
        value = getattr(md, key)
        return frame.get_frame_metadata(value) if frame.supports_frame_metadata(value) else None

    color = frames.get_color_frame()
    domain = str(color.get_frame_timestamp_domain()).rsplit(".", 1)[-1]
    out = {"got": t_got, "domain": domain, "arrival": to_perf(meta(color, "time_of_arrival"))}
    if domain == "global_time":
        ts = float(color.get_timestamp())
        f_us, s_us = meta(color, "frame_timestamp"), meta(color, "sensor_timestamp")
        if f_us is not None and s_us is not None:
            ts -= (f_us - s_us) / 1e3
            out["src"] = "global_mid"
        else:
            out["src"] = "global_readout"
        out["capture"] = to_perf(ts)
    elif out["arrival"] is not None:
        out["capture"], out["src"] = out["arrival"], "arrival"
    else:
        out["capture"], out["src"] = t_got, "got"
    return out


class D455Camera:
    """RealSense D455 with depth→color alignment and depth noise filters."""

    def __init__(
        self,
        color_size: tuple[int, int] = (1280, 720),
        depth_size: tuple[int, int] = (848, 480),
        fps: int = 30,
    ) -> None:
        import pyrealsense2 as rs  # lazy import

        self._rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, *color_size, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, *depth_size, rs.format.z16, fps)
        self.profile = self.pipeline.start(cfg)
        # If any post-start setup fails, stop the pipeline before propagating —
        # otherwise the started pipeline + claimed USB device leak (the caller falls
        # back to Mock and the device stays locked until process exit).
        try:
            self.align = rs.align(rs.stream.color)
            self.spatial = rs.spatial_filter()
            self.temporal = rs.temporal_filter()
            self.hole_filling = rs.hole_filling_filter()

            color_stream = self.profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            self.intrinsics = {
                "fx": intr.fx, "fy": intr.fy,
                "ppx": intr.ppx, "ppy": intr.ppy,
                # Reported, not applied: calibration logs them and warns if non-zero.
                "model": str(intr.model), "coeffs": [float(c) for c in intr.coeffs],
                "width": intr.width, "height": intr.height,
            }
            self.depth_scale = (
                self.profile.get_device().first_depth_sensor().get_depth_scale()
            )
        except Exception:
            try:
                self.pipeline.stop()
            except Exception:                          # noqa: BLE001
                pass
            raise
        logger.info("D455 started — intrinsics fx=%.3f fy=%.3f ppx=%.3f ppy=%.3f depth_scale=%.6f",
                    intr.fx, intr.fy, intr.ppx, intr.ppy, self.depth_scale)

    def get_frame(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Capture one (rgb, depth_m) pair. depth_m in metres, aligned and filtered."""
        frames = self.pipeline.wait_for_frames(timeout_ms=2000)
        t_got = time.perf_counter()
        try:
            self.last_frame_clock = frame_clock(frames, t_got, self._rs)
        except Exception as e:                 # noqa: BLE001 - timing never costs a frame
            logger.debug("frame_clock: %s", e)
            self.last_frame_clock = {"got": t_got, "capture": t_got, "src": "got"}
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None

        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        # NOTE: hole_filling is intentionally NOT applied — it FABRICATES depth in
        # no-return regions (e.g. by copying a neighbour), which then passes the
        # zero-depth rejection in postprocess.masked_depth and corrupts the grasp
        # depth with invented data. Leaving holes as 0 lets masked_depth drop them.
        # (Re-enable only for a display-only colormap, on a separate copy.)

        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()) * self.depth_scale
        return rgb, depth

    def stop(self) -> None:
        """Stop the RealSense pipeline."""
        self.pipeline.stop()
        logger.info("D455 stopped")


class MockCamera:
    """Simulated camera — returns synthetic frames. Use for test/sim without hardware.

    Args:
        rgb_frames: List of RGB images cycled in order. None → black frame.
        depth_frames: Corresponding list of depth images (metres). None → flat 0.8 m plane.
        intrinsics: Intrinsics dict. None → D455-style defaults.
    """

    DEFAULT_INTRINSICS = {
        "fx": 640.0, "fy": 640.0, "ppx": 640.0, "ppy": 360.0,
        "width": 1280, "height": 720,
    }

    def __init__(
        self,
        rgb_frames: list[np.ndarray] | None = None,
        depth_frames: list[np.ndarray] | None = None,
        intrinsics: dict[str, float] | None = None,
        clock=None,
        latency_s: float = 0.0,
    ) -> None:
        self.intrinsics = intrinsics or dict(self.DEFAULT_INTRINSICS)
        h, w = self.intrinsics["height"], self.intrinsics["width"]
        self._rgb = rgb_frames or [np.zeros((h, w, 3), np.uint8)]
        self._depth = depth_frames or [np.full((h, w), 0.8, np.float32)]
        self._i = 0
        # Stand-in for frame_clock: the frame is 'got' when get_frame runs and was
        # captured latency_s before, on the clock given (a test's fake clock).
        self._clock = clock or time.perf_counter
        self._latency_s = float(latency_s)
        self.last_frame_clock: dict | None = None

    def get_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the next frame pair (cyclic)."""
        rgb = self._rgb[self._i % len(self._rgb)]
        depth = self._depth[self._i % len(self._depth)]
        self._i += 1
        t = self._clock()
        self.last_frame_clock = {"got": t, "capture": t - self._latency_s, "src": "mock"}
        return rgb, depth

    def stop(self) -> None:
        """No-op — keeps the interface consistent with D455Camera."""
