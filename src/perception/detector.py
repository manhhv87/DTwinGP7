"""
detector.py
───────────
Inference wrapper for YOLOv8-seg + a MockDetector for sim/test.

ObjectDetector lazy-imports ultralytics → this module can be imported on a
machine without ultralytics installed (only MockDetector is usable then).

Both detectors share the same interface:
    .detect(rgb) → list[Detection]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Class names — must match the order used during training (data/splits/dataset.yaml).
DEFAULT_CLASS_NAMES = ["tray", "bottle", "cup", "bolt"]


@dataclass
class Detection:
    """A single detected object in the image."""

    class_id: int
    class_name: str
    confidence: float
    mask: np.ndarray                       # binary mask (H, W)
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    # Fields below are filled in by PoseExtractor after detection:
    pose_camera: tuple[float, float, float, float] | None = None
    pixel_uv: tuple[int, int] | None = None
    mask_area: int = 0
    # True height of the object (mm) — sim mode reads from STL bounds. Real mode
    # can compute from depth gradient or class lookup. Used for adaptive
    # grasp_depth_offset in the orchestrator.
    height_mm: float | None = None


def field_dict(det: Detection) -> dict:
    """Convert Detection → dict (keeping the keys the orchestrator needs)."""
    return {
        "class_id": det.class_id,
        "class_name": det.class_name,
        "confidence": det.confidence,
        "mask": det.mask,
        "bbox": det.bbox,
        "pose_camera": det.pose_camera,
        "pixel_uv": det.pixel_uv,
        "mask_area": det.mask_area,
        "height_mm": det.height_mm,
    }


class ObjectDetector:
    """YOLOv8-seg inference wrapper.

    Used for INFERENCE only with pre-trained weights — the model is trained on
    a separate machine (Linux GPU); this repo only receives the weight file.
    Supports both .pt and .onnx files (ultralytics auto-detects by extension).
    """

    def __init__(
        self,
        model_path: str = "models/yolov8s-seg_best.pt",
        conf: float = 0.5,
        iou: float = 0.45,
        class_names: list[str] | None = None,
    ) -> None:
        # Must be a real weights file — do NOT let ultralytics auto-download a
        # COCO model when the path is wrong (that would produce wrong classes).
        weights = Path(model_path)
        if not weights.exists():
            raise FileNotFoundError(
                f"Weight file not found: {model_path}\n"
                f"The model is trained on a Linux machine — copy best.pt (or .onnx) "
                f"here and set the correct path.\n"
                f"See HUONG_DAN_CAI_DAT.md section 'Transferring weights to the inference machine'."
            )

        from ultralytics import YOLO  # lazy import

        self.model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou
        # Prefer class names from the model itself (self.model.names) → automatically
        # correct for any task/model. Fall back only when names are unavailable.
        model_names = getattr(self.model, "names", None)
        if class_names:
            self.class_names = list(class_names)
        elif isinstance(model_names, dict) and model_names:
            self.class_names = [model_names[k] for k in sorted(model_names)]
        elif model_names:
            self.class_names = list(model_names)
        else:
            self.class_names = DEFAULT_CLASS_NAMES
        logger.info("YOLO loaded: %s (conf=%.2f, iou=%.2f) classes=%s",
                    model_path, conf, iou, self.class_names)

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        """Run inference and return a list of Detection objects."""
        # retina_masks=True → masks come back at the ORIGINAL image resolution, not
        # the letterboxed model-input (square, padded) size. Without it, masks.data
        # are e.g. 640×640-with-bars for a 1280×720 input, and a later naive resize to
        # the depth shape distorts the centroid/PCA-yaw and the deprojected 3D point.
        results = self.model(
            rgb, conf=self.conf, iou=self.iou, retina_masks=True, verbose=False)[0]
        detections: list[Detection] = []

        if results.masks is None:
            return detections

        for i in range(len(results.boxes)):
            cls_id = int(results.boxes.cls[i])
            cls_name = (self.class_names[cls_id]
                        if 0 <= cls_id < len(self.class_names) else str(cls_id))
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=float(results.boxes.conf[i]),
                    mask=results.masks.data[i].cpu().numpy().astype(np.uint8),
                    bbox=tuple(results.boxes.xyxy[i].cpu().numpy().tolist()),
                )
            )
        return detections


class MockDetector:
    """Simulated detector — returns a fixed scripted detection scenario.

    Used to test the orchestrator/perception_node without a model or GPU.

    Args:
        scripted: List of list[Detection] — each detect() call returns the next entry.
    """

    def __init__(self, scripted: list[list[Detection]] | None = None) -> None:
        self._scripted = scripted or [[]]
        self._i = 0

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        """Return the next scripted detection scenario (cycles through the list)."""
        out = self._scripted[self._i % len(self._scripted)]
        self._i += 1
        return out

    @staticmethod
    def make_detection(
        class_name: str = "bottle",
        confidence: float = 0.95,
        mask_hw: tuple[int, int] = (720, 1280),
        mask_box: tuple[int, int, int, int] = (600, 320, 680, 420),
    ) -> Detection:
        """Convenience helper to create a Detection with a rectangular mask."""
        mask = np.zeros(mask_hw, np.uint8)
        x1, y1, x2, y2 = mask_box
        mask[y1:y2, x1:x2] = 1
        cls_id = (
            DEFAULT_CLASS_NAMES.index(class_name)
            if class_name in DEFAULT_CLASS_NAMES
            else 0
        )
        return Detection(
            class_id=cls_id,
            class_name=class_name,
            confidence=confidence,
            mask=mask,
            bbox=(float(x1), float(y1), float(x2), float(y2)),
        )
