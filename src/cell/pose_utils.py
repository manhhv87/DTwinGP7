"""
pose_utils.py
─────────────
Helper functions cho pose math.

Tách riêng để:
  - Testable mà không cần RoboDK đang chạy
  - Có thể dùng cho mục đích khác (visualization, debugging)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def make_homogeneous(
    xyz_mm: Sequence[float],
    rpy_deg: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Build 4x4 homogeneous transform matrix.

    Args:
        xyz_mm: Translation (x, y, z) in millimeters.
        rpy_deg: Rotation (roll, pitch, yaw) in degrees. Applied as Rz*Ry*Rx (XYZ extrinsic).

    Returns:
        4x4 numpy array (float64), units in mm.

    Example:
        >>> T = make_homogeneous([100, 200, 300], [0, 0, 90])
        >>> T[:3, 3]
        array([100., 200., 300.])
    """
    if len(xyz_mm) != 3:
        raise ValueError(f"xyz_mm must have 3 elements, got {len(xyz_mm)}")
    if len(rpy_deg) != 3:
        raise ValueError(f"rpy_deg must have 3 elements, got {len(rpy_deg)}")

    # Convert RPY to rotation matrix
    R = _rpy_to_matrix(*[np.deg2rad(a) for a in rpy_deg])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz_mm
    return T


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll-pitch-yaw (radians) → 3x3 rotation matrix.

    Convention: R = Rz(yaw) * Ry(pitch) * Rx(roll) (intrinsic XYZ, same as RoboDK)
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def matrix_to_rpy_xyz(T: np.ndarray) -> tuple[list[float], list[float]]:
    """Inverse of make_homogeneous: extract (xyz_mm, rpy_deg) from 4x4 matrix.

    Args:
        T: 4x4 homogeneous transform.

    Returns:
        (xyz_mm, rpy_deg) tuple of lists.
    """
    if T.shape != (4, 4):
        raise ValueError(f"T must be 4x4, got shape {T.shape}")

    xyz_mm = T[:3, 3].tolist()

    R = T[:3, :3]
    # Extract RPY (ZYX intrinsic = XYZ extrinsic)
    pitch = np.arcsin(-R[2, 0])
    if np.cos(pitch) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])

    rpy_deg = [np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)]
    return xyz_mm, rpy_deg


def robodk_pose_to_matrix(robodk_pose) -> np.ndarray:
    """Convert RoboDK Mat (pose) → numpy 4x4 array.

    RoboDK trả về Mat object qua .Pose() — phải convert.
    """
    rows = robodk_pose.Rows()
    return np.array(rows).reshape(4, 4)


def matrix_to_robodk_pose(T: np.ndarray):
    """Convert numpy 4x4 → RoboDK Mat (lazy import để testable không RoboDK)."""
    from robodk.robomath import Mat
    return Mat(T.tolist())
