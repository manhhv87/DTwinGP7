"""
dh_model.py
───────────
Denavit-Hartenberg parameters for the Yaskawa GP7 + framework for any 6R robot.

MODIFIED DH convention (Yaskawa, Craig 1986):
    For link i, transform from frame i-1 to frame i:
        T_i = Rot_x(alpha_{i-1}) · Trans_x(a_{i-1}) · Rot_z(theta_i) · Trans_z(d_i)

⚠ DH parameters for the GP7 below are taken from Yaskawa datasheet HW1474564. VERIFY
against the datasheet for your specific model variant (GP7, GP7L, GP8, ... may differ).
Wrong DH → skewed forward kinematics → predictive simulation is useless.

Joint convention Yaskawa: [S, L, U, R, B, T] (6-DOF revolute).
Units: mm for length, radians for angles (entire module).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DHLink:
    """Modified DH parameters for one link + joint limits.

    Args:
        a: Link length (mm) — distance between Z_{i-1} and Z_i along X_{i-1}.
        alpha: Link twist (rad) — angle between Z_{i-1} and Z_i.
        d: Link offset (mm) — distance between O_{i-1} and Z_i along Z_{i-1}.
        theta_offset: Joint angle offset (rad) — constant added to the joint value.
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
        name: Robot name (e.g. "Yaskawa GP7").
        links: List of 6 DHLinks in order S→L→U→R→B→T for Yaskawa.
        base_xyz_mm: Base frame position in world (mm).
        base_rpy_rad: Base frame rotation (roll, pitch, yaw in radians).
        tool_offset_mm: TCP offset from flange along tool Z (mm).
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
    """Yaskawa GP7 DH default values — Modified DH (Craig 1986).

    Physical dimensions (Yaskawa GP7 datasheet HW1474564 + ROS-Industrial
    motoman_gp7_support URDF):
        d1  = 330 mm    Base height (J1 axis to J2 axis vertical)
        a1  = 40  mm    J1 horizontal offset (J1 axis to J2 axis)
        a2  = 445 mm    Upper arm length (J2 to J3)
        d4  = 440 mm    Forearm length (J3 to J5)
        d6  = 80  mm    Wrist to flange
        Reach max ~927mm, payload 7kg, repeatability ±0.02mm

    Modified DH convention (link i's params are alpha_{i-1}, a_{i-1}, d_i):
        Each transform = Rot_x(alpha) · Trans_x(a) · Rot_z(theta) · Trans_z(d)

    | i | alpha_{i-1} | a_{i-1} | d_i | theta_offset_i |
    |---|-------------|---------|-----|----------------|
    | 1 |  0          |  0      | d1  |  0             |
    | 2 | -π/2        |  a1     |  0  | -π/2           |  ← home arm horizontal
    | 3 |  0          |  a2     |  0  |  0             |
    | 4 | -π/2        |  0      | d4  |  0             |
    | 5 |  π/2        |  0      |  0  |  0             |
    | 6 | -π/2        |  0      | d6  |  0             |

    Verified via `scripts/13_verify_vs_robodk.py` vs the RoboDK GP7 model — max
    position diff < 5mm when base_xyz matches. (The URDF chain in `urdf_chain.py`
    gives a more exact match — 0.00mm — and is the default used in the orchestrator.)
    """
    deg = np.deg2rad
    # GP7 dimensions (mm)
    d1, a1, a2, d4, d6 = 330.0, 40.0, 445.0, 440.0, 80.0

    # Joint limits synced with the URDF chain (gp7_urdf in urdf_chain.py) —
    # verified 1:1 against RoboDK Yaskawa-GP7.robot. The old DH used stale values
    # (J2 ±[-90,155], J3 [-175,240], J4 ±180) → IK failed at home pose because
    # J4=-181.93° fell outside the limit. Source-of-truth = URDF.
    links = (
        # S (J1): base rotation about world Z
        DHLink(a=0.0,    alpha=0.0,        d=d1,
               theta_offset=0.0,
               joint_min=deg(-170), joint_max=deg(170)),
        # L (J2): shoulder — rotate about Y after Rx(-90°) twist
        DHLink(a=a1,     alpha=deg(-90),   d=0.0,
               theta_offset=deg(-90),
               joint_min=deg(-65),  joint_max=deg(145)),
        # U (J3): elbow — same axis as L (no twist)
        DHLink(a=a2,     alpha=0.0,        d=0.0,
               theta_offset=0.0,
               joint_min=deg(-70), joint_max=deg(190)),   # GP7 datasheet (was -116/+255)
        # R (J4): wrist roll — twist axis to forearm
        DHLink(a=0.0,    alpha=deg(-90),   d=d4,
               theta_offset=0.0,
               joint_min=deg(-190), joint_max=deg(190)),
        # B (J5): wrist pitch
        DHLink(a=0.0,    alpha=deg(90),    d=0.0,
               theta_offset=0.0,
               joint_min=deg(-135), joint_max=deg(135)),
        # T (J6): wrist yaw + flange
        DHLink(a=0.0,    alpha=deg(-90),   d=d6,
               theta_offset=0.0,
               joint_min=deg(-360), joint_max=deg(360)),
    )
    return RobotDHModel(
        name="Yaskawa GP7",
        links=links,
        base_xyz_mm=base_xyz_mm,
        base_rpy_rad=base_rpy_rad,
        tool_offset_mm=tool_offset_mm,
    )
