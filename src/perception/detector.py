"""
detector.py
───────────
Wrapper inference cho YOLOv8-seg + một MockDetector cho sim/test.

ObjectDetector lazy-import ultralytics → module này import được trên máy
chưa cài ultralytics (chỉ MockDetector dùng được khi đó).

Cả hai detector cùng interface:
    .detect(rgb) → list[Detection]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Tên class — phải khớp thứ tự khi train (data/splits/dataset.yaml).
DEFAULT_CLASS_NAMES = ["tray", "bottle", "cup", "bolt"]


@dataclass
class Detection:
    """Một vật được phát hiện trong ảnh."""

    class_id: int
    class_name: str
    confidence: float
    mask: np.ndarray                       # mask nhị phân (H, W)
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    # Các trường dưới do PoseExtractor điền sau:
    pose_camera: tuple[float, float, float, float] | None = None
    pixel_uv: tuple[int, int] | None = None
    mask_area: int = 0
    # Chiều cao thật của vật (mm) — sim mode lấy từ STL bounds. Real mode
    # có thể tính từ depth gradient hoặc lookup class. Dùng cho adaptive
    # grasp_depth_offset trong orchestrator.
    height_mm: float | None = None


def field_dict(det: Detection) -> dict:
    """Chuyển Detection → dict (giữ các key mà orchestrator cần)."""
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

    Chỉ dùng để INFERENCE bằng trọng số đã train sẵn — model được train ở
    máy khác (Linux GPU), repo này chỉ nhận file trọng số. Hỗ trợ cả file
    .pt lẫn .onnx (ultralytics tự nhận dạng theo đuôi file).
    """

    def __init__(
        self,
        model_path: str = "models/yolov8s-seg_best.pt",
        conf: float = 0.5,
        iou: float = 0.45,
        class_names: list[str] | None = None,
    ) -> None:
        # Phải là file trọng số có thật — KHÔNG để ultralytics tự tải về
        # model COCO khi đường dẫn sai (sẽ cho kết quả sai class).
        weights = Path(model_path)
        if not weights.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file trọng số: {model_path}\n"
                f"Model được train trên máy Linux — copy file best.pt (hoặc .onnx) "
                f"về và đặt đúng đường dẫn này.\n"
                f"Xem HUONG_DAN_CAI_DAT.md mục 'Đưa trọng số về máy chạy'."
            )

        from ultralytics import YOLO  # lazy import

        self.model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        logger.info("YOLO loaded: %s (conf=%.2f, iou=%.2f)", model_path, conf, iou)

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        """Chạy inference, trả về danh sách Detection."""
        results = self.model(rgb, conf=self.conf, iou=self.iou, verbose=False)[0]
        detections: list[Detection] = []

        if results.masks is None:
            return detections

        for i in range(len(results.boxes)):
            cls_id = int(results.boxes.cls[i])
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=self.class_names[cls_id],
                    confidence=float(results.boxes.conf[i]),
                    mask=results.masks.data[i].cpu().numpy().astype(np.uint8),
                    bbox=tuple(results.boxes.xyxy[i].cpu().numpy().tolist()),
                )
            )
        return detections


class MockDetector:
    """Detector giả lập — trả về một kịch bản detection cố định.

    Dùng cho test orchestrator/perception_node mà không cần model + GPU.

    Args:
        scripted: List các list[Detection] — mỗi lần detect() trả phần tử kế.
    """

    def __init__(self, scripted: list[list[Detection]] | None = None) -> None:
        self._scripted = scripted or [[]]
        self._i = 0

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        """Trả về kịch bản detection kế tiếp (lặp vòng)."""
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
        """Tiện ích tạo nhanh một Detection với mask hình chữ nhật."""
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
