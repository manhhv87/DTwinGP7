"""
camera.py
─────────
Wrapper for Intel RealSense D455 camera + a MockCamera for sim/test.

D455Camera lazy-imports pyrealsense2 → this module is importable on machines
without the RealSense SDK (only MockCamera works in that case).

Both cameras expose the same interface:
    .intrinsics  → dict {fx, fy, ppx, ppy, width, height}
    .get_frame() → (rgb, depth_m) — depth_m in metres, or (None, None)
    .stop()
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


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
        logger.info("D455 started — intrinsics fx=%.1f fy=%.1f", intr.fx, intr.fy)

    def get_frame(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Capture one (rgb, depth_m) pair. depth_m in metres, aligned and filtered."""
        frames = self.pipeline.wait_for_frames(timeout_ms=2000)
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
    ) -> None:
        self.intrinsics = intrinsics or dict(self.DEFAULT_INTRINSICS)
        h, w = self.intrinsics["height"], self.intrinsics["width"]
        self._rgb = rgb_frames or [np.zeros((h, w, 3), np.uint8)]
        self._depth = depth_frames or [np.full((h, w), 0.8, np.float32)]
        self._i = 0

    def get_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the next frame pair (cyclic)."""
        rgb = self._rgb[self._i % len(self._rgb)]
        depth = self._depth[self._i % len(self._depth)]
        self._i += 1
        return rgb, depth

    def stop(self) -> None:
        """No-op — keeps the interface consistent with D455Camera."""
