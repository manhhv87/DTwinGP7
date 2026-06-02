"""
perception — Vision pipeline: D455 camera → YOLOv8-seg → 3D object pose.

Public API:
    D455Camera, MockCamera
    ObjectDetector, MockDetector, Detection
    PoseExtractor + postprocess helpers
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
