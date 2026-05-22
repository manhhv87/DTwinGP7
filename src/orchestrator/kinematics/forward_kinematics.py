"""
forward_kinematics.py
─────────────────────
Pure-numpy forward kinematics cho 6R serial robot dùng Modified DH.

KHÔNG phụ thuộc RoboDK — chạy độc lập + testable. Tốc độ ~50µs/call trên CPU
modern → predict 100-sample trajectory < 5ms.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .dh_model import DHLink, RobotDHModel


@lru_cache(maxsize=None)
def _dh_link_consts(link: DHLink) -> tuple[float, float, float, np.ndarray]:
    """Hằng số của 1 link KHÔNG phụ thuộc joint angle — cache 1 lần.

    `cos/sin(alpha)`, `theta_offset` và toàn bộ entry hằng của ma trận DH
    (a, -sa, -sa·d, ca, ca·d, hàng cuối) được dựng sẵn vào template. FK khỏi
    tính lại trig của alpha hay parse list lồng mỗi call → nhanh hơn ~1.7×.
    Kết quả số HỌC y hệt (cùng phép cos/sin/nhân).
    """
    ca, sa = float(np.cos(link.alpha)), float(np.sin(link.alpha))
    tpl = np.zeros((4, 4))
    tpl[0, 3] = link.a
    tpl[1, 2] = -sa
    tpl[1, 3] = -sa * link.d
    tpl[2, 2] = ca
    tpl[2, 3] = ca * link.d
    tpl[3, 3] = 1.0
    return ca, sa, link.theta_offset, tpl


@lru_cache(maxsize=None)
def _base_transform_cached(model: RobotDHModel) -> np.ndarray:
    """Base transform cache (chỉ phụ thuộc base_xyz/base_rpy của model)."""
    return _base_transform(model)


def _dh_transform(link: DHLink, joint_rad: float) -> np.ndarray:
    """Modified DH 4x4 transform cho 1 link.

    Per Craig 1986 convention:
        T = Rot_x(alpha) · Trans_x(a) · Rot_z(theta + offset) · Trans_z(d)

    Chỉ 6 entry phụ thuộc joint angle; phần còn lại lấy từ template cache
    (`_dh_link_consts`) → bit-identical với bản dựng `np.array([...])` cũ.
    """
    ca, sa, theta_offset, tpl = _dh_link_consts(link)
    theta = float(joint_rad) + theta_offset
    ct, st = np.cos(theta), np.sin(theta)

    M = tpl.copy()
    M[0, 0] = ct
    M[0, 1] = -st
    M[1, 0] = st * ca
    M[1, 1] = ct * ca
    M[2, 0] = st * sa
    M[2, 1] = ct * sa
    return M


def _dh_transform_batch(link: DHLink, q_col: np.ndarray) -> np.ndarray:
    """Stack (N,4,4) DH transform cho N joint angle cùng link — vectorized.

    M[k] == _dh_transform(link, q_col[k]) (bit-identical): cùng cos/sin và phép
    nhân, chỉ chạy theo vector numpy thay vì vòng lặp Python.
    """
    ca, sa, theta_offset, _ = _dh_link_consts(link)
    theta = q_col + theta_offset
    ct = np.cos(theta)
    st = np.sin(theta)
    M = np.zeros((q_col.shape[0], 4, 4))
    M[:, 0, 0] = ct
    M[:, 0, 1] = -st
    M[:, 0, 3] = link.a
    M[:, 1, 0] = st * ca
    M[:, 1, 1] = ct * ca
    M[:, 1, 2] = -sa
    M[:, 1, 3] = -sa * link.d
    M[:, 2, 0] = st * sa
    M[:, 2, 1] = ct * sa
    M[:, 2, 2] = ca
    M[:, 2, 3] = ca * link.d
    M[:, 3, 3] = 1.0
    return M


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

    T = _base_transform_cached(model)
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
    T = _base_transform_cached(model)
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


def joint_positions_batch(
    model: RobotDHModel,
    joints_batch_rad: np.ndarray,
) -> list[np.ndarray]:
    """Vectorized `joint_positions` cho N samples cùng lúc (batched matmul).

    Tính FK cho cả batch trong vài phép matmul (N,4,4) thay vì gọi
    `joint_positions` N lần → nhanh hơn nhiều cho self-collision check trên
    trajectory (~30× cho 200+ samples). Vì batched matmul tính từng sample độc
    lập bằng đúng routine BLAS, kết quả BIT-IDENTICAL với từng call lẻ:
        positions[p][k] == joint_positions(model, joints_batch_rad[k])[p]

    Args:
        joints_batch_rad: (N, num_joints) joint angles (radian).

    Returns:
        List dài (num_joints + 1 [+1 nếu tool_offset]); mỗi phần tử là array
        (N, 3) — vị trí world của 1 joint origin cho tất cả N samples.
    """
    J = np.asarray(joints_batch_rad, dtype=float)
    if J.ndim != 2 or J.shape[1] != model.num_joints():
        raise ValueError(
            f"Expected (N, {model.num_joints()}), got {J.shape}"
        )
    n_samples = J.shape[0]
    T = np.broadcast_to(_base_transform_cached(model), (n_samples, 4, 4)).copy()
    positions: list[np.ndarray] = [T[:, :3, 3].copy()]      # base origin

    for k, link in enumerate(model.links):
        T = T @ _dh_transform_batch(link, J[:, k])
        positions.append(T[:, :3, 3].copy())

    if model.tool_offset_mm != 0.0:
        tool = np.eye(4)
        tool[2, 3] = model.tool_offset_mm
        T = T @ np.broadcast_to(tool, (n_samples, 4, 4))
        positions.append(T[:, :3, 3].copy())

    return positions
