"""
base.py
───────
RobotBackend Protocol — hợp đồng interface mọi backend phải đáp ứng.

Thiết kế: duck-typed theo subset của RoboDK Item API. SimRobot và backend
mới (HSE, MotoROS2, ...) đều mimic interface này → Orchestrator gọi nguyên
văn `self.robot.MoveJ()`, `self.robot.setDO()`... mà không biết backend nào.

Method nào KHÔNG phải mọi backend implement được (vd SolveIK trong HSE thuần
không có) thì:
  - Mark Optional trong Protocol
  - Backend raise NotImplementedError với hint dùng backend khác

Pose conventions (THỐNG NHẤT toàn project):
  - Pose 4x4 trong base/world frame (mm).
  - Joints: list[float], 6 phần tử (S, L, U, R, B, T cho GP7), đơn vị degrees.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RobotBackend(Protocol):
    """Interface tối thiểu Orchestrator cần. Mọi backend phải có các method này.

    Note: chữ hoa method name khớp RoboDK Item API để khỏi refactor orchestrator.
    """

    # ─── Query ───
    def Joints(self) -> Any:
        """Joints hiện tại (6 phần tử, độ). RoboDK trả `Mat`, others trả list."""

    def JointsHome(self) -> Any:
        """Joints home (rest pose)."""

    def Valid(self) -> bool:
        """Backend có hoạt động không."""

    # ─── Motion ───
    def MoveJ(self, target: Any) -> None:
        """Joint move tới target (pose 4x4 hoặc joint list 6 phần tử)."""

    def MoveL(self, target: Any) -> None:
        """Linear (Cartesian) move tới target. Raise nếu không có IK feasible."""

    def MoveJ_Test(self, j_start: Any, target: Any, *args: Any) -> int:
        """RoboDK convention: 0 = OK, >0 = collision tại joint #, <0 = unreachable."""

    def SolveIK(self, pose: Any, joints_approx: Any = None) -> Any:
        """Inverse kinematics: pose → joints. None nếu không feasible."""

    # ─── I/O ───
    def setDO(self, index: int, value: int) -> None:
        """Set digital output (gripper, fixture, signal lamp...)."""

    def setSpeed(self, linear_mm_s: float, joint_deg_s: float = -1) -> None:
        """Set max speed cho move tiếp theo. -1 = giữ nguyên."""

    # ─── Misc (RoboDK-specific, optional) ───
    def Parent(self) -> Any:
        """RoboDK parent frame. HSE/SimRobot backend trả None."""
