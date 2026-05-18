"""
coord_conv.py
─────────────
Coordinate transform utilities cho pipeline pick-and-place.

Toàn bộ hàm là pure-numpy → testable mà không cần RoboDK/RealSense.

Quy ước thống nhất toàn dự án:
  - Đơn vị độ dài: mm
  - Đơn vị góc: radian (trừ khi tên hàm ghi rõ _deg)
  - Ma trận biến đổi: 4x4 homogeneous, T_A^B nghĩa là "frame A biểu diễn trong frame B"
  - T_BC = T_base^camera... thực chất là camera-trong-base (camera-to-base):
        p_base = T_BC @ p_camera
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ───── Ma trận xoay/tịnh tiến cơ bản (4x4) ─────


def transl(x: float, y: float, z: float) -> np.ndarray:
    """Ma trận tịnh tiến thuần (mm)."""
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def rotx(angle_rad: float) -> np.ndarray:
    """Ma trận xoay quanh trục X."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    T = np.eye(4)
    T[1, 1], T[1, 2] = c, -s
    T[2, 1], T[2, 2] = s, c
    return T


def rotz(angle_rad: float) -> np.ndarray:
    """Ma trận xoay quanh trục Z."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    T = np.eye(4)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    return T


# ───── Phép biến đổi ─────


def transform_point(p_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Biến đổi 1 điểm 3D qua ma trận T (4x4).

    Args:
        p_xyz: Điểm (x, y, z).
        T: Ma trận biến đổi 4x4.

    Returns:
        Điểm sau biến đổi (3,).
    """
    p = np.asarray(p_xyz, dtype=float).reshape(3)
    p_h = np.append(p, 1.0)
    return (np.asarray(T, dtype=float) @ p_h)[:3]


def camera_to_base(xyz_cam: np.ndarray, T_BC: np.ndarray) -> np.ndarray:
    """Chuyển 1 điểm từ camera frame sang base frame robot.

    Args:
        xyz_cam: Điểm (x, y, z) trong camera frame (mm).
        T_BC: Ma trận camera-to-base 4x4 (output hand-eye calibration).

    Returns:
        Điểm (x, y, z) trong base frame robot (mm).

    Example:
        >>> T = np.eye(4); T[:3, 3] = [400, 0, 850]
        >>> camera_to_base(np.array([0, 0, 0]), T)
        array([400.,   0., 850.])
    """
    return transform_point(xyz_cam, T_BC)


def make_grasp_pose(
    xyz_base_mm: np.ndarray,
    yaw_rad: float = 0.0,
    yaw_offset_deg: float = 0.0,
) -> np.ndarray:
    """Dựng pose SE(3) cho gripper gắp top-down tại (xyz, yaw).

    Gripper hướng thẳng xuống (trục Z tool song song -Z base), xoay quanh
    trục thẳng đứng theo yaw của vật.

    Args:
        xyz_base_mm: Vị trí gắp trong base frame (mm).
        yaw_rad: Góc yaw của vật (radian, từ PCA trên mask).
        yaw_offset_deg: Bù lệch yaw giữa trục PCA và trục mở của gripper
            — giá trị này được tinh chỉnh trong pha tuning (mục 9).

    Returns:
        Ma trận pose 4x4 (mm).
    """
    xyz = np.asarray(xyz_base_mm, dtype=float).reshape(3)
    total_yaw = yaw_rad + np.deg2rad(yaw_offset_deg)
    # Tịnh tiến → xoay yaw quanh Z base → lật gripper hướng xuống.
    return transl(*xyz) @ rotz(total_yaw) @ rotx(np.pi)


# ───── I/O calibration ─────


def load_calibration(path: str | Path) -> np.ndarray:
    """Load ma trận hand-eye T_BC từ file .npy.

    Raises:
        FileNotFoundError: File không tồn tại.
        ValueError: Ma trận không phải 4x4.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"File calibration không tồn tại: {p}\n"
            f"Chạy scripts/02_run_calibration.py để tạo."
        )
    T = np.load(p)
    if T.shape != (4, 4):
        raise ValueError(f"Ma trận calibration phải 4x4, nhận {T.shape}")
    return T


def save_calibration(path: str | Path, T_BC: np.ndarray) -> None:
    """Lưu ma trận hand-eye T_BC ra file .npy."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    T = np.asarray(T_BC, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"Ma trận calibration phải 4x4, nhận {T.shape}")
    np.save(p, T)
