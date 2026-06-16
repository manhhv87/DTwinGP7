"""
coord_conv.py
─────────────
Coordinate transform utilities for the pick-and-place pipeline.

All functions are pure-numpy → testable without RoboDK/RealSense.

Project-wide conventions:
  - Length unit: mm
  - Angle unit: radian (unless the function name explicitly says _deg)
  - Transform matrices: 4x4 homogeneous, T_A^B means "frame A expressed in frame B"
  - T_BC = T_base^camera... effectively camera-in-base (camera-to-base):
        p_base = T_BC @ p_camera
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ───── Basic rotation/translation matrices (4x4) ─────


def transl(x: float, y: float, z: float) -> np.ndarray:
    """Pure translation matrix (mm)."""
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def rotx(angle_rad: float) -> np.ndarray:
    """Rotation matrix about the X axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    T = np.eye(4)
    T[1, 1], T[1, 2] = c, -s
    T[2, 1], T[2, 2] = s, c
    return T


def rotz(angle_rad: float) -> np.ndarray:
    """Rotation matrix about the Z axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    T = np.eye(4)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    return T


# ───── Transforms ─────


def transform_point(p_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform a single 3D point through a 4x4 matrix T.

    Args:
        p_xyz: Point (x, y, z).
        T: 4x4 transform matrix.

    Returns:
        Transformed point (3,).
    """
    p = np.asarray(p_xyz, dtype=float).reshape(3)
    p_h = np.append(p, 1.0)
    return (np.asarray(T, dtype=float) @ p_h)[:3]


def camera_to_base(xyz_cam: np.ndarray, T_BC: np.ndarray) -> np.ndarray:
    """Convert a point from the camera frame to the robot base frame.

    Args:
        xyz_cam: Point (x, y, z) in the camera frame (mm).
        T_BC: 4x4 camera-to-base matrix (output of hand-eye calibration).

    Returns:
        Point (x, y, z) in the robot base frame (mm).

    Example:
        >>> T = np.eye(4); T[:3, 3] = [400, 0, 850]
        >>> camera_to_base(np.array([0, 0, 0]), T)
        array([400.,   0., 850.])
    """
    return transform_point(xyz_cam, T_BC)


def camera_yaw_to_base(yaw_cam_rad: float, T_BC: np.ndarray) -> float:
    """Rotate an in-plane (image-frame) PCA grasp yaw into the base frame via the
    hand-eye rotation R_BC. SINGLE SOURCE OF TRUTH for the yaw transform — both the
    orchestrator pick path and the GUI camera-teach path must use this so they can
    never diverge (skipping it leaves the gripper off by the camera mounting
    rotation + image handedness, mis-aligning the jaws with the object long axis).
    """
    R_BC = np.asarray(T_BC, dtype=float)[:3, :3]
    d = R_BC @ np.array([np.cos(yaw_cam_rad), np.sin(yaw_cam_rad), 0.0])
    return float(np.arctan2(d[1], d[0]))


def make_grasp_pose(
    xyz_base_mm: np.ndarray,
    yaw_rad: float = 0.0,
    yaw_offset_deg: float = 0.0,
) -> np.ndarray:
    """Build an SE(3) pose for a top-down gripper grasp at (xyz, yaw).

    The gripper points straight down (tool Z axis parallel to -Z base) and
    is rotated about the vertical axis by the object yaw.

    Args:
        xyz_base_mm: Grasp position in the base frame (mm).
        yaw_rad: Object yaw angle (radians, from PCA on the mask).
        yaw_offset_deg: Yaw offset between the PCA axis and the gripper opening
            axis — tuned during the calibration phase (section 9).

    Returns:
        4x4 pose matrix (mm).
    """
    xyz = np.asarray(xyz_base_mm, dtype=float).reshape(3)
    total_yaw = yaw_rad + np.deg2rad(yaw_offset_deg)
    # Translate → rotate yaw about base Z → flip gripper downward.
    return transl(*xyz) @ rotz(total_yaw) @ rotx(np.pi)


# ───── I/O calibration ─────


def load_calibration(path: str | Path) -> np.ndarray:
    """Load the hand-eye matrix T_BC from a .npy file.

    Raises:
        FileNotFoundError: File does not exist.
        ValueError: Matrix is not 4x4.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {p}\n"
            f"Run scripts/02_run_calibration.py to generate it."
        )
    T = np.load(p)
    if T.shape != (4, 4):
        raise ValueError(f"Calibration matrix must be 4x4, got {T.shape}")
    return T


def save_calibration(path: str | Path, T_BC: np.ndarray) -> None:
    """Save the hand-eye matrix T_BC to a .npy file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    T = np.asarray(T_BC, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"Calibration matrix must be 4x4, got {T.shape}")
    np.save(p, T)
