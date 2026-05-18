"""
camera.py
─────────
Wrapper cho camera Intel RealSense D455 + một MockCamera cho sim/test.

D455Camera lazy-import pyrealsense2 → module này import được trên máy không
có RealSense SDK (chỉ MockCamera là dùng được khi đó).

Cả hai camera đều cung cấp cùng interface:
    .intrinsics  → dict {fx, fy, ppx, ppy, width, height}
    .get_frame() → (rgb, depth_m) — depth_m đơn vị mét, hoặc (None, None)
    .stop()
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class D455Camera:
    """RealSense D455 với align depth→color + bộ lọc giảm nhiễu depth."""

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
        logger.info("D455 started — intrinsics fx=%.1f fy=%.1f", intr.fx, intr.fy)

    def get_frame(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Lấy 1 cặp (rgb, depth_m). depth_m đơn vị mét, đã align + lọc."""
        frames = self.pipeline.wait_for_frames(timeout_ms=2000)
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None

        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)

        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()) * self.depth_scale
        return rgb, depth

    def stop(self) -> None:
        """Dừng pipeline RealSense."""
        self.pipeline.stop()
        logger.info("D455 stopped")


class MockCamera:
    """Camera giả lập — trả frame tổng hợp. Dùng cho test/sim không phần cứng.

    Args:
        rgb_frames: List ảnh RGB lặp tuần hoàn. None → ảnh đen.
        depth_frames: List ảnh depth (mét) tương ứng. None → mặt phẳng 0.8m.
        intrinsics: Dict intrinsics. None → giá trị mặc định kiểu D455.
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
        """Trả cặp frame kế tiếp (lặp vòng)."""
        rgb = self._rgb[self._i % len(self._rgb)]
        depth = self._depth[self._i % len(self._depth)]
        self._i += 1
        return rgb, depth

    def stop(self) -> None:
        """No-op — giữ interface đồng nhất với D455Camera."""
