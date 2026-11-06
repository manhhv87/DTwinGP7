"""
frame_convert.py
────────────────
Frame conversion utilities for HSE Cartesian motion path.

3 primary frames:
  - WORLD: orchestrator native (cell layout, hand-eye `T_BC` output)
  - ROBOT BASE: robot's own frame (origin at J1 axis, what HSE BASE expects)
  - USER FRAME: optional, defined via YRC1000 teach pendant (not yet used)

Rotation convention: Yaskawa HSE BASE coordinate uses **XYZ-fixed** Tait-Bryan
(= ZYX intrinsic). The composed matrix is **R = Rz · Ry · Rx** (extrinsic XYZ),
exactly as implemented in `rpy_yaskawa_to_matrix`. ⚠ Do NOT "correct" this to
Rx·Ry·Rz to match an old docstring — that would mirror every commanded orientation.

Reference:
  - Yaskawa INFORM Language Manual (RE-CKI-A464), §Cartesian Position
  - Verify against first real robot test (touch test 3 orthogonal poses)
"""
from __future__ import annotations

import numpy as np

from src.cell.pose_utils import make_homogeneous


def world_to_robot_base(
    T_world: np.ndarray,
    robot_base_xyz_mm: tuple[float, float, float],
    robot_base_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Convert a 4x4 pose from world frame → robot base frame.

    HSE BASE coordinate frame has its origin at the robot's J1 axis. Cell config
    places the robot on a pedestal at world Z=630 (e.g.) → need to subtract base pose.

    Math: T_world = T_base_in_world @ T_robot
        → T_robot = inv(T_base_in_world) @ T_world

    Args:
        T_world: 4x4 pose in world frame (mm).
        robot_base_xyz_mm: Robot J1 position in world (mm). Taken from
            `cell_config.robot.pose.xyz_mm`.
        robot_base_rpy_deg: Robot base rotation in world (degrees XYZ-fixed).
            Default (0,0,0) — robot upright, no base rotation.

    Returns:
        T_robot_base: 4x4 pose in robot base frame.
    """
    T_world = np.asarray(T_world, dtype=float)
    if T_world.shape != (4, 4):
        raise ValueError(f"T_world must be 4x4, got {T_world.shape}")

    T_base_in_world = make_homogeneous(robot_base_xyz_mm, robot_base_rpy_deg)
    # inverse of rigid transform: R^T, -R^T @ t
    R = T_base_in_world[:3, :3]
    t = T_base_in_world[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv @ T_world


def matrix_to_rpy_yaskawa(R: np.ndarray) -> tuple[float, float, float]:
    """Decompose 3x3 rotation matrix → (Rx, Ry, Rz) degrees XYZ-fixed convention.

    Yaskawa HSE BASE pose uses XYZ-fixed (extrinsic) Tait-Bryan, which equals the
    intrinsic ZYX composition that this code actually solves:
        R = Rz(γ) · Ry(β) · Rx(α)
        (extrinsic XYZ = intrinsic ZYX — rotate about fixed X, then Y, then Z)

    Solve:
        β = atan2(-R[2,0], sqrt(R[0,0]² + R[1,0]²))
        γ = atan2(R[1,0]/cos β, R[0,0]/cos β)         (Rz)
        α = atan2(R[2,1]/cos β, R[2,2]/cos β)         (Rx)

    Gimbal lock when β = ±90° (cos β = 0) → α + γ degenerate, set γ = 0.

    Returns:
        (rx_deg, ry_deg, rz_deg).
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got {R.shape}")

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
    """Compose 3x3 rotation matrix from Yaskawa XYZ-fixed RPY.

    Yaskawa convention: rotation applied in order about FIXED AXES (world):
        1. Rotate about world Z by rz
        2. Rotate about world Y by ry
        3. Rotate about world X by rx
    → R = Rx · Ry · Rz when applied intrinsically = Rz · Ry · Rx when applied
       extrinsically. Industrial robotics standard = R = Rz · Ry · Rx (equivalent
       to scipy euler 'xyz' extrinsic).

    Inverse of `matrix_to_rpy_yaskawa`. Round-trip rpy → R → rpy must match
    (except at gimbal lock ry = ±90°).
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

    One-liner for HSE encode flow:
        T_base = world_to_robot_base(T_world, base_xyz, base_rpy)
        x, y, z, rx, ry, rz = matrix_to_xyzrpy_yaskawa(T_base)
        payload = encode_cartesian_position(x, y, z, rx, ry, rz, tool_no=1)
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"T must be 4x4, got {T.shape}")
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    rx, ry, rz = matrix_to_rpy_yaskawa(T[:3, :3])
    return float(x), float(y), float(z), rx, ry, rz


def place_approach_base(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
) -> np.ndarray:
    """P003, the point above the place, from P000, P001 and P002 (YRC BASE, mm).

    X, Y and orientation are P002's. Z is P002's raised by the approach height
    of the pick (P001.Z - P000.Z), but never below P001, so the joint move from
    P001 to P003 does not descend.

    A part lifted off a support is set down lower than it was picked, by the
    support's height less the release margin, and P002 plus the approach height
    then lies under P001 by as much. DTWINRUN reaches P003 with a joint move,
    which spreads that drop over the transfer while the part still overlaps its
    support: on 14 Oct 2026 (B3, approach 100 mm, a plastic box lifted off the
    wood box) the forward kinematics of the logged joints put the plastic's
    bottom 9.1 mm under the wood's top while their footprints overlapped. Held
    at P001's height the transfer is level, and the part descends only on the
    straight move from P003 to P002. A part picked from the table is set down
    within 1 mm of the height it was picked at (the table's tilt between the
    cards and the place), so its P003 moves by at most that.

    The orchestrator plans with this P003 and the HSE backend writes it, so the
    two cannot differ.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p3 = np.array(p2, dtype=float)
    p3[2, 3] = max(float(p3[2, 3]) + float(p1[2, 3] - p0[2, 3]), float(p1[2, 3]))
    return p3
