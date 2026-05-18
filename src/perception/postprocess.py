"""
postprocess.py
──────────────
Trích xuất pose 3D của vật từ mask segmentation + depth image.

Pipeline cho mỗi detection:
    mask (2D)  →  centroid (u, v)
    depth+mask →  z (median, chống nhiễu)
    (u, v, z)  →  deproject → (x, y, z) trong camera frame
    mask (2D)  →  PCA → yaw

Toàn bộ là pure-numpy → testable không cần camera. `cv2` chỉ dùng cho
resize_mask và được lazy-import.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Số pixel tối thiểu của mask để coi là detection hợp lệ.
MIN_MASK_PIXELS = 25


def resize_mask(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize mask nhị phân về (height, width) mục tiêu (nearest neighbor).

    YOLOv8 trả mask ở độ phân giải khác ảnh gốc → cần resize trước khi
    kết hợp với depth image.
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
    """Tính tâm (u, v) của mask nhị phân.

    Returns:
        (u, v) pixel coordinate, hoặc None nếu mask quá nhỏ.
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
    """Lấy giá trị depth đại diện trên vùng mask.

    Args:
        depth_image: Ảnh depth (đơn vị mét), shape (H, W).
        mask: Mask nhị phân cùng shape.
        method: "median" (mặc định, chống nhiễu) hoặc "mean".

    Returns:
        Depth (mét) hoặc None nếu không đủ pixel có depth hợp lệ.
    """
    if depth_image.shape != mask.shape:
        mask = resize_mask(mask, depth_image.shape)

    values = depth_image[mask > 0]
    values = values[values > 0]  # loại pixel không có depth
    if len(values) < 5:
        return None
    return float(np.median(values) if method == "median" else np.mean(values))


def mask_pca_yaw(mask: np.ndarray) -> float:
    """Ước lượng yaw của vật bằng PCA trên các pixel mask.

    Trục chính (principal component lớn nhất) = hướng dài của vật.

    Returns:
        Yaw (radian), trong khoảng (-pi/2, pi/2].
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < MIN_MASK_PIXELS:
        return 0.0

    pts = np.column_stack([xs, ys]).astype(float)
    pts -= pts.mean(axis=0)

    # Hiệp phương sai 2x2 → eigenvector ứng với eigenvalue lớn nhất.
    cov = np.cov(pts, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, np.argmax(eigvals)]

    yaw = np.arctan2(major_axis[1], major_axis[0])
    # Chuẩn hoá về (-pi/2, pi/2] vì trục là đối xứng 180°.
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
    """Deproject pixel (u, v) + depth → điểm 3D trong camera frame.

    Dùng mô hình pinhole. Tương đương rs2_deproject_pixel_to_point nhưng
    pure-numpy → testable không cần RealSense SDK.

    Args:
        intrinsics: Dict {fx, fy, ppx, ppy} (pixel).
        u, v: Toạ độ pixel.
        z_m: Depth tại pixel đó (mét).

    Returns:
        Điểm (x, y, z) trong camera frame, đơn vị mm.
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    ppx, ppy = intrinsics["ppx"], intrinsics["ppy"]
    x = (u - ppx) * z_m / fx
    y = (v - ppy) * z_m / fy
    return np.array([x, y, z_m]) * 1000.0  # mét → mm


class PoseExtractor:
    """Kết hợp các hàm trên để trích pose 3D từ một detection.

    Pure logic — nhận intrinsics dạng dict, không giữ tham chiếu camera.
    """

    def __init__(self, intrinsics: dict[str, float]) -> None:
        self.intrinsics = intrinsics

    def extract(
        self,
        detection: dict[str, Any],
        depth_image: np.ndarray,
    ) -> dict[str, Any] | None:
        """Bổ sung trường 'pose_camera' = (x, y, z, yaw) vào detection.

        Returns:
            Detection đã thêm pose, hoặc None nếu không trích được pose.
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
