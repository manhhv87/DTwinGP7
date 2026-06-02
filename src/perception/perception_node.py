"""
perception_node.py
──────────────────
Background perception node: camera → detector → pose extractor → queue.

Architecture: runs a continuous loop thread, pushing detection messages into
a queue for the Orchestrator to consume. Queue has maxsize → old frames are
dropped automatically if the orchestrator is slow (always uses the latest detection).

Message pushed to queue:
    {
        "timestamp": float,
        "objects": list[dict],   # each dict has 'class_name', 'pose_camera', ...
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
    """Perception loop running on a dedicated thread.

    Args:
        camera: Object with .get_frame() and .intrinsics (D455Camera/MockCamera).
        detector: Object with .detect(rgb) (ObjectDetector/MockDetector).
        output_queue: Queue that receives detection messages.
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
        """Start the perception loop on a background thread."""
        if self._running:
            logger.warning("PerceptionNode is already running")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("PerceptionNode started")

    def stop(self) -> None:
        """Stop the loop and the camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.camera.stop()
        logger.info("PerceptionNode stopped")

    def process_once(self) -> dict[str, Any] | None:
        """Process exactly one frame and return a detection message.

        Decoupled from the loop so it can be tested without a thread.
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
        """Background loop: continuously call process_once → push to queue."""
        while self._running:
            msg = self.process_once()
            if msg is None:
                # No frame (camera error or timeout) — short sleep to avoid
                # busy-spinning the CPU.
                time.sleep(0.01)
                continue
            try:
                self.queue.put_nowait(msg)
            except queue.Full:
                # Drop the oldest frame, keep the new one.
                try:
                    self.queue.get_nowait()
                    self.queue.put_nowait(msg)
                except queue.Empty:
                    pass
