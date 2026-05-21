"""
dh_model.py
───────────
Denavit-Hartenberg parameters cho Yaskawa GP7 + framework cho mọi 6R robot.

Quy ước MODIFIED DH (Yaskawa, Craig 1986):
    For link i, transform from frame i-1 to frame i:
        T_i = Rot_x(alpha_{i-1}) · Trans_x(a_{i-1}) · Rot_z(theta_i) · Trans_z(d_i)

⚠ DH parameters cho GP7 dưới đây là từ datasheet Yaskawa HW1474564. VERIFY lại
với datasheet riêng của model variant bạn dùng (GP7, GP7L, GP8, ... có khác).
Sai DH → forward kinematics lệch → predictive simulation vô dụng.

Joint convention Yaskawa: [S, L, U, R, B, T] (6-DOF revolute).
Unit: mm cho length, radian cho góc (toàn module).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DHLink:
    """Modified DH parameters cho 1 link + joint limit.

    Args:
        a: Link length (mm) — distance giữa Z_{i-1} và Z_i theo X_{i-1}.
        alpha: Link twist (rad) — góc giữa Z_{i-1} và Z_i.
        d: Link offset (mm) — distance giữa O_{i-1} và Z_i theo Z_{i-1}.
        theta_offset: Joint angle offset (rad) — hằng số cộng vào joint value.
        joint_min: Joint limit min (rad).
        joint_max: Joint limit max (rad).
    """

    a: float
    alpha: float
    d: float
    theta_offset: float
    joint_min: float
    joint_max: float


@dataclass(frozen=True)
class RobotDHModel:
    """6-DOF serial robot DH model.

    Args:
        name: Tên robot (vd "Yaskawa GP7").
        links: List 6 DHLink, theo thứ tự S→L→U→R→B→T cho Yaskawa.
        base_xyz_mm: Vị trí base frame trong world (mm).
        base_rpy_rad: Rotation base frame (roll, pitch, yaw radian).
        tool_offset_mm: TCP offset từ flange theo Z tool (mm).
    """

    name: str
    links: tuple[DHLink, ...]
    base_xyz_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tool_offset_mm: float = 0.0

    def num_joints(self) -> int:
        return len(self.links)


def gp7_default(
    base_xyz_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    base_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tool_offset_mm: float = 0.0,
) -> RobotDHModel:
    """Yaskawa GP7 DH default values.

    Spec source: datasheet HW1474564 (public).
    Reach max ~927mm, payload 7kg, repeatability ±0.02mm.

    ⚠ VERIFY với máy thật trước khi dùng cho safety-critical predict. Sai
    pulse_per_deg trong hse_protocol.py cũng làm sai actual joint angle.
    """
    deg = np.deg2rad
    # Joint limits từ datasheet (deg → rad)
    links = (
        # S (J1): base rotation
        DHLink(a=150.0,  alpha=deg(-90),  d=330.0,
               theta_offset=0.0,
               joint_min=deg(-170), joint_max=deg(170)),
        # L (J2): shoulder
        DHLink(a=760.0,  alpha=0.0,       d=0.0,
               theta_offset=deg(-90),
               joint_min=deg(-90),  joint_max=deg(155)),
        # U (J3): elbow
        DHLink(a=140.0,  alpha=deg(-90),  d=0.0,
               theta_offset=0.0,
               joint_min=deg(-175), joint_max=deg(240)),
        # R (J4): wrist roll
        DHLink(a=0.0,    alpha=deg(90),   d=795.0,
               theta_offset=0.0,
               joint_min=deg(-180), joint_max=deg(180)),
        # B (J5): wrist pitch
        DHLink(a=0.0,    alpha=deg(-90),  d=0.0,
               theta_offset=0.0,
               joint_min=deg(-135), joint_max=deg(135)),
        # T (J6): wrist yaw + flange
        DHLink(a=0.0,    alpha=0.0,       d=80.0,
               theta_offset=deg(180),
               joint_min=deg(-360), joint_max=deg(360)),
    )
    return RobotDHModel(
        name="Yaskawa GP7",
        links=links,
        base_xyz_mm=base_xyz_mm,
        base_rpy_rad=base_rpy_rad,
        tool_offset_mm=tool_offset_mm,
    )
