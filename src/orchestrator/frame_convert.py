"""
frame_convert.py
────────────────
Frame conversion utilities cho HSE Cartesian motion path.

3 frame chính:
  - WORLD: orchestrator native (cell layout, hand-eye `T_BC` output)
  - ROBOT BASE: robot's own frame (origin tại J1 axis, what HSE BASE expects)
  - USER FRAME: optional, define qua YRC1000 teach pendant (chưa dùng)

Convention rotation: Yaskawa HSE BASE coordinate dùng **XYZ-fixed** Tait-Bryan
(= ZYX intrinsic): R = Rx(Rx) · Ry(Ry) · Rz(Rz) applied in sequence.

Reference:
  - Yaskawa INFORM Language Manual (RE-CKI-A464), §Cartesian Position
  - Confirm bằng test thực tế lần đầu trên robot (touch test 3 pose orthogonal)
"""
from __future__ import annotations

import numpy as np

from src.cell.pose_utils import make_homogeneous


def world_to_robot_base(
    T_world: np.ndarray,
    robot_base_xyz_mm: tuple[float, float, float],
    robot_base_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Convert pose 4x4 từ world frame → robot base frame.

    HSE BASE coordinate frame có origin tại J1 axis của robot. Cell config
    đặt robot trên pedestal world Z=630 (vd) → cần subtract base pose.

    Math: T_world = T_base_in_world @ T_robot
        → T_robot = inv(T_base_in_world) @ T_world

    Args:
        T_world: 4x4 pose trong world frame (mm).
        robot_base_xyz_mm: Robot J1 position trong world (mm). Lấy từ
            `cell_config.robot.pose.xyz_mm`.
        robot_base_rpy_deg: Robot base rotation trong world (degrees XYZ-fixed).
            Default (0,0,0) — robot upright, không xoay base.

    Returns:
        T_robot_base: 4x4 pose trong robot base frame.
    """
    T_world = np.asarray(T_world, dtype=float)
    if T_world.shape != (4, 4):
        raise ValueError(f"T_world phải 4x4, got {T_world.shape}")

    T_base_in_world = make_homogeneous(robot_base_xyz_mm, robot_base_rpy_deg)
    # inv của rigid transform: R^T, -R^T @ t
    R = T_base_in_world[:3, :3]
    t = T_base_in_world[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv @ T_world


def matrix_to_rpy_yaskawa(R: np.ndarray) -> tuple[float, float, float]:
    """Decompose 3x3 rotation matrix → (Rx, Ry, Rz) degrees XYZ-fixed convention.

    Yaskawa HSE BASE pose dùng XYZ-fixed Tait-Bryan:
        R = Rx(α) · Ry(β) · Rz(γ)         (applied left-to-right)
        equivalent to Rz then Ry then Rx (intrinsic ZYX)

    Solve:
        β = atan2(-R[2,0], sqrt(R[0,0]² + R[1,0]²))
        γ = atan2(R[1,0]/cos β, R[0,0]/cos β)         (Rz)
        α = atan2(R[2,1]/cos β, R[2,2]/cos β)         (Rx)

    Gimbal lock khi β = ±90° (cos β = 0) → α + γ degenerate, set γ = 0.

    Returns:
        (rx_deg, ry_deg, rz_deg).
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"R phải 3x3, got {R.shape}")

    sy = -R[2, 0]                                # = -R[2,0] = sin(β)
    cy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))

    if cy > 1e-6:
        rx = float(np.arctan2(R[2, 1], R[2, 2]))
        ry = float(np.arctan2(sy, cy))
        rz = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        # Gimbal lock: pitch = ±90°. Set rz=0, solve rx.
        rx = float(np.arctan2(-R[1, 2], R[1, 1]))
        ry = float(np.arctan2(sy, cy))
        rz = 0.0

    return float(np.rad2deg(rx)), float(np.rad2deg(ry)), float(np.rad2deg(rz))


def rpy_yaskawa_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Compose 3x3 rotation matrix từ Yaskawa XYZ-fixed RPY.

    Yaskawa convention: rotation áp dụng theo thứ tự về AXIS CỐ ĐỊNH (world):
        1. Rotate about world Z by rz
        2. Rotate about world Y by ry
        3. Rotate about world X by rx
    → R = Rx · Ry · Rz khi áp dụng theo intrinsic = Rz · Ry · Rx khi áp dụng
       extrinsic. Chuẩn industrial robotics = R = Rz · Ry · Rx (equivalent
       to scipy euler 'xyz' extrinsic).

    Inverse của `matrix_to_rpy_yaskawa`. Round-trip rpy → R → rpy phải khớp
    (trừ gimbal lock ry = ±90°).
    """
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    # XYZ-fixed (extrinsic): R = Rz · Ry · Rx
    return Rz @ Ry @ Rx


def matrix_to_xyzrpy_yaskawa(
    T: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Decompose 4x4 transform → (x_mm, y_mm, z_mm, Rx_deg, Ry_deg, Rz_deg).

    One-liner cho HSE encode flow:
        T_base = world_to_robot_base(T_world, base_xyz, base_rpy)
        x, y, z, rx, ry, rz = matrix_to_xyzrpy_yaskawa(T_base)
        payload = encode_cartesian_position(x, y, z, rx, ry, rz, tool_no=1)
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"T phải 4x4, got {T.shape}")
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    rx, ry, rz = matrix_to_rpy_yaskawa(T[:3, :3])
    return float(x), float(y), float(z), rx, ry, rz
