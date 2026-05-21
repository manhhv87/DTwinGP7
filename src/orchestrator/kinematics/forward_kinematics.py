"""
forward_kinematics.py
─────────────────────
Pure-numpy forward kinematics cho 6R serial robot dùng Modified DH.

KHÔNG phụ thuộc RoboDK — chạy độc lập + testable. Tốc độ ~50µs/call trên CPU
modern → predict 100-sample trajectory < 5ms.
"""
from __future__ import annotations

import numpy as np

from .dh_model import DHLink, RobotDHModel


def _dh_transform(link: DHLink, joint_rad: float) -> np.ndarray:
    """Modified DH 4x4 transform cho 1 link.

    Per Craig 1986 convention:
        T = Rot_x(alpha) · Trans_x(a) · Rot_z(theta + offset) · Trans_z(d)
    """
    theta = float(joint_rad) + link.theta_offset
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(link.alpha), np.sin(link.alpha)

    return np.array([
        [ct,       -st,       0.0,      link.a],
        [st * ca,  ct * ca,   -sa,      -sa * link.d],
        [st * sa,  ct * sa,   ca,       ca * link.d],
        [0.0,      0.0,       0.0,      1.0],
    ])


def _base_transform(model: RobotDHModel) -> np.ndarray:
    """Compose base frame transform từ base_xyz + base_rpy."""
    roll, pitch, yaw = model.base_rpy_rad
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    # Rzyx = Rz(yaw) · Ry(pitch) · Rx(roll)
    R = np.array([
        [cy * cp,  cy * sp * sr - sy * cr,  cy * sp * cr + sy * sr],
        [sy * cp,  sy * sp * sr + cy * cr,  sy * sp * cr - cy * sr],
        [-sp,      cp * sr,                 cp * cr],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = model.base_xyz_mm
    return T


def forward_kinematics(
    model: RobotDHModel,
    joints_rad: list[float] | tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    """Compute TCP pose 4x4 trong WORLD frame từ joint angles (radian).

    Returns:
        T_world_tcp: 4x4 homogeneous transform. Translation mm.

    Raises:
        ValueError: Joint count không khớp model.
    """
    joints = np.asarray(joints_rad, dtype=float).flatten()
    if len(joints) != model.num_joints():
        raise ValueError(
            f"Expected {model.num_joints()} joints, got {len(joints)}"
        )

    T = _base_transform(model)
    for link, q in zip(model.links, joints):
        T = T @ _dh_transform(link, q)

    # Apply TCP offset theo Z tool (cuối cùng) nếu có
    if model.tool_offset_mm != 0.0:
        tool = np.eye(4)
        tool[2, 3] = model.tool_offset_mm
        T = T @ tool

    return T


def joint_positions(
    model: RobotDHModel,
    joints_rad: list[float] | tuple[float, ...] | np.ndarray,
) -> list[np.ndarray]:
    """Compute vị trí (x, y, z) của TỪNG joint origin trong world frame.

    Hữu ích cho:
      - Visualize skeleton trong matplotlib 3D
      - Self-collision check (segment between consecutive joints)

    Returns:
        List 7 điểm 3D: [base, J1, J2, J3, J4, J5, J6, TCP] — len = num_joints + 2
        (bao gồm base origin và TCP cuối).
    """
    joints = np.asarray(joints_rad, dtype=float).flatten()
    if len(joints) != model.num_joints():
        raise ValueError(
            f"Expected {model.num_joints()} joints, got {len(joints)}"
        )

    positions: list[np.ndarray] = []
    T = _base_transform(model)
    positions.append(T[:3, 3].copy())             # base origin

    for link, q in zip(model.links, joints):
        T = T @ _dh_transform(link, q)
        positions.append(T[:3, 3].copy())          # joint i origin

    if model.tool_offset_mm != 0.0:
        tool = np.eye(4)
        tool[2, 3] = model.tool_offset_mm
        T = T @ tool
        positions.append(T[:3, 3].copy())          # TCP

    return positions
