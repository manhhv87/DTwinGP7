"""
kinematics — Pure-Python forward kinematics + trajectory simulation for GP7.

This module has NO dependency on RoboDK — runs on plain numpy. Used for:
  - Predictive simulation (UC1, UC2): preview trajectory before execution
  - Replay mode (UC4): visualize CSV telemetry offline
  - Future IK solver client-side (foundation)
"""
from .dh_model import DHLink, RobotDHModel, gp7_default
from .forward_kinematics import forward_kinematics, joint_positions
from .inverse_kinematics import inverse_kinematics, inverse_kinematics_seeded
from .trajectory import (
    TrajectorySample,
    interpolate_joints,
    check_joint_limits,
    check_self_collision_spheres,
)

__all__ = [
    "DHLink",
    "RobotDHModel",
    "gp7_default",
    "forward_kinematics",
    "joint_positions",
    "inverse_kinematics",
    "inverse_kinematics_seeded",
    "TrajectorySample",
    "interpolate_joints",
    "check_joint_limits",
    "check_self_collision_spheres",
]
