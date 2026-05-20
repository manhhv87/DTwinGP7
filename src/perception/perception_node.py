"""
perception_node.py
──────────────────
Node perception chạy nền: camera → detector → pose extractor → queue.

Kiến trúc: chạy một thread vòng lặp liên tục, đẩy message detection vào
queue cho Orchestrator tiêu thụ. Queue có maxsize → tự bỏ frame cũ nếu
orchestrator xử lý chậm (luôn dùng detection mới nhất).

Message đẩy vào queue:
    {
        "timestamp": float,
        "objects": list[dict],   # mỗi dict có 'class_name', 'pose_camera', ...
        "fps": float,
    }
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

from .detector import field_dict
from .postprocess import PoseExtractor

logger = logging.getLogger(__name__)


class PerceptionNode:
    """Vòng lặp perception chạy trên thread riêng.

    Args:
        camera: Đối tượng có .get_frame() và .intrinsics (D455Camera/MockCamera).
        detector: Đối tượng có .detect(rgb) (ObjectDetector/MockDetector).
        output_queue: Queue nhận message detection.
    """

    def __init__(
        self,
        camera: Any,
        detector: Any,
        output_queue: queue.Queue,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.queue = output_queue
        self.extractor = PoseExtractor(camera.intrinsics)

        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Khởi động vòng lặp perception trên thread nền."""
        if self._running:
            logger.warning("PerceptionNode đã chạy rồi")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("PerceptionNode started")

    def stop(self) -> None:
        """Dừng vòng lặp và camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.camera.stop()
        logger.info("PerceptionNode stopped")

    def process_once(self) -> dict[str, Any] | None:
        """Xử lý đúng một frame, trả về message detection.

        Tách riêng khỏi vòng lặp để test được mà không cần thread.
        """
        t0 = time.time()
        rgb, depth = self.camera.get_frame()
        if rgb is None or depth is None:
            return None

        detections = self.detector.detect(rgb)
        objects: list[dict[str, Any]] = []
        for det in detections:
            enriched = self.extractor.extract(field_dict(det), depth)
            if enriched is not None and enriched["pose_camera"] is not None:
                objects.append(enriched)

        dt = max(time.time() - t0, 1e-6)
        return {"timestamp": time.time(), "objects": objects, "fps": 1.0 / dt}

    def _loop(self) -> None:
        """Vòng lặp nền: liên tục process_once → đẩy vào queue."""
        while self._running:
            msg = self.process_once()
            if msg is None:
                # Không có frame (camera lỗi hoặc timeout) — nghỉ ngắn để
                # không busy-spin CPU.
                time.sleep(0.01)
                continue
            try:
                self.queue.put_nowait(msg)
            except queue.Full:
                # Bỏ frame cũ nhất, giữ frame mới.
                try:
                    self.queue.get_nowait()
                    self.queue.put_nowait(msg)
                except queue.Empty:
                    pass
