"""
perception — Thị giác: camera D455 → YOLOv8-seg → pose 3D của vật.

Public API:
    D455Camera, MockCamera
    ObjectDetector, MockDetector, Detection
    PoseExtractor + các hàm postprocess
    PerceptionNode
"""
from .camera import D455Camera, MockCamera
from .detector import DEFAULT_CLASS_NAMES, Detection, MockDetector, ObjectDetector
from .perception_node import PerceptionNode
from .postprocess import (
    PoseExtractor,
    deproject_pixel,
    mask_centroid,
    mask_pca_yaw,
    masked_depth,
    resize_mask,
)

__all__ = [
    "D455Camera",
    "MockCamera",
    "ObjectDetector",
    "MockDetector",
    "Detection",
    "DEFAULT_CLASS_NAMES",
    "PerceptionNode",
    "PoseExtractor",
    "mask_centroid",
    "mask_pca_yaw",
    "masked_depth",
    "resize_mask",
    "deproject_pixel",
]
