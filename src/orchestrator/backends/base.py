"""
base.py
───────
RobotBackend Protocol — interface contract every backend must satisfy.

Design: duck-typed against a subset of the RoboDK Item API. SimRobot and new
backends (HSE, MotoROS2, ...) all mimic this interface → Orchestrator calls
`self.robot.MoveJ()`, `self.robot.setDO()`... without knowing which backend is active.

Methods NOT implementable by every backend (e.g. SolveIK in a pure HSE backend):
  - Mark Optional in Protocol
  - Backend raises NotImplementedError with a hint to use a different backend

Pose conventions (uniform across the entire project):
  - Pose 4x4 in base/world frame (mm).
  - Joints: list[float], 6 elements (S, L, U, R, B, T for GP7), unit: degrees.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RobotBackend(Protocol):
    """Minimal interface required by the Orchestrator. Every backend must implement these methods.

    Note: method names are capitalized to match the RoboDK Item API, avoiding orchestrator refactoring.
    """

    # ─── Query ───
    def Joints(self) -> Any:
        """Current joints (6 elements, degrees). RoboDK returns `Mat`, others return list."""

    def JointsHome(self) -> Any:
        """Joints home (rest pose)."""

    def Valid(self) -> bool:
        """Whether the backend is operational."""

    # ─── Motion ───
    def MoveJ(self, target: Any) -> None:
        """Joint move to target (4x4 pose or 6-element joint list)."""

    def MoveL(self, target: Any) -> None:
        """Linear (Cartesian) move to target. Raises if no feasible IK solution exists."""

    def MoveJ_Test(self, j_start: Any, target: Any, *args: Any) -> int:
        """RoboDK convention: 0 = OK, >0 = collision at joint #, <0 = unreachable."""

    def SolveIK(self, pose: Any, joints_approx: Any = None) -> Any:
        """Inverse kinematics: pose → joints. Returns None if no feasible solution."""

    # ─── I/O ───
    def setDO(self, index: int, value: int) -> None:
        """Set a digital output (gripper, fixture, signal lamp...)."""

    def setSpeed(self, linear_mm_s: float, joint_deg_s: float = -1) -> None:
        """Set max speed for the next move. -1 = keep current value."""

    # ─── Misc (RoboDK-specific, optional) ───
    def Parent(self) -> Any:
        """RoboDK parent frame. HSE/SimRobot backends return None."""
