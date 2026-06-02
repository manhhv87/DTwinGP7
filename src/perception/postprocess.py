"""
postprocess.py
──────────────
Extract 3D pose of an object from a segmentation mask + depth image.

Pipeline for each detection:
    mask (2D)  →  centroid (u, v)
    depth+mask →  z (median, noise-robust)
    (u, v, z)  →  deproject → (x, y, z) in camera frame
    mask (2D)  →  PCA → yaw

Entirely pure-numpy → testable without a camera. `cv2` is used only for
resize_mask and is lazy-imported.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Minimum number of mask pixels to consider a detection valid.
MIN_MASK_PIXELS = 25


def resize_mask(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask to the target (height, width) using nearest-neighbor interpolation.

    YOLOv8 returns masks at a different resolution than the original image,
    so they must be resized before combining with the depth image.
    """
    if mask.shape == target_hw:
        return mask
    import cv2  # lazy import

    resized = cv2.resize(
        mask.astype(np.uint8),
        (target_hw[1], target_hw[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized


def mask_centroid(mask: np.ndarray) -> tuple[int, int] | None:
    """Compute the centroid (u, v) of a binary mask.

    Returns:
        (u, v) pixel coordinate, or None if the mask is too small.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < MIN_MASK_PIXELS:
        return None
    return int(round(xs.mean())), int(round(ys.mean()))


def masked_depth(
    depth_image: np.ndarray,
    mask: np.ndarray,
    method: str = "median",
) -> float | None:
    """Return a representative depth value over the masked region.

    Args:
        depth_image: Depth image (in metres), shape (H, W).
        mask: Binary mask with the same shape.
        method: "median" (default, noise-robust) or "mean".

    Returns:
        Depth (metres), or None if there are not enough valid depth pixels.
    """
    if depth_image.shape != mask.shape:
        mask = resize_mask(mask, depth_image.shape)

    values = depth_image[mask > 0]
    values = values[values > 0]  # discard pixels with no depth
    if len(values) < 5:
        return None
    return float(np.median(values) if method == "median" else np.mean(values))


def mask_pca_yaw(mask: np.ndarray) -> float:
    """Estimate the object's yaw via PCA on mask pixels.

    The principal component (largest eigenvalue) aligns with the object's long axis.

    Returns:
        Yaw (radians), in the range (-pi/2, pi/2].
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < MIN_MASK_PIXELS:
        return 0.0

    pts = np.column_stack([xs, ys]).astype(float)
    pts -= pts.mean(axis=0)

    # 2x2 covariance matrix → eigenvector corresponding to the largest eigenvalue.
    cov = np.cov(pts, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, np.argmax(eigvals)]

    yaw = np.arctan2(major_axis[1], major_axis[0])
    # Normalise to (-pi/2, pi/2] because an axis has 180° symmetry.
    if yaw > np.pi / 2:
        yaw -= np.pi
    elif yaw <= -np.pi / 2:
        yaw += np.pi
    return float(yaw)


def deproject_pixel(
    intrinsics: dict[str, float],
    u: float,
    v: float,
    z_m: float,
) -> np.ndarray:
    """Deproject pixel (u, v) + depth → 3D point in camera frame.

    Uses the pinhole model. Equivalent to rs2_deproject_pixel_to_point but
    pure-numpy → testable without the RealSense SDK.

    Args:
        intrinsics: Dict {fx, fy, ppx, ppy} (pixels).
        u, v: Pixel coordinates.
        z_m: Depth at that pixel (metres).

    Returns:
        Point (x, y, z) in camera frame, in mm.
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    ppx, ppy = intrinsics["ppx"], intrinsics["ppy"]
    x = (u - ppx) * z_m / fx
    y = (v - ppy) * z_m / fy
    return np.array([x, y, z_m]) * 1000.0  # metres → mm


class PoseExtractor:
    """Combine the above helpers to extract a 3D pose from a single detection.

    Pure logic — accepts intrinsics as a dict; holds no camera reference.
    """

    def __init__(self, intrinsics: dict[str, float]) -> None:
        self.intrinsics = intrinsics

    def extract(
        self,
        detection: dict[str, Any],
        depth_image: np.ndarray,
    ) -> dict[str, Any] | None:
        """Augment a detection with the 'pose_camera' = (x, y, z, yaw) field.

        Returns:
            Detection dict with pose added, or None if the pose cannot be extracted.
        """
        mask = detection["mask"]
        if mask.shape != depth_image.shape:
            mask = resize_mask(mask, depth_image.shape)

        centroid = mask_centroid(mask)
        if centroid is None:
            return None
        u, v = centroid

        z_m = masked_depth(depth_image, mask)
        if z_m is None:
            return None

        xyz_cam = deproject_pixel(self.intrinsics, u, v, z_m)
        yaw = mask_pca_yaw(mask)

        detection["pose_camera"] = (
            float(xyz_cam[0]), float(xyz_cam[1]), float(xyz_cam[2]), yaw,
        )
        detection["pixel_uv"] = (u, v)
        detection["mask_area"] = int(np.sum(mask > 0))
        return detection
