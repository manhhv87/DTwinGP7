"""
script_api.py
─────────────
ScriptProgramAPI: read/write façade exposed tới user Python script
(embedded script editor) qua biến `p`. Tách lớp khỏi GP7AppQt để hạn chế
surface mà script có thể đụng vào app internals.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .program_model import Instruction

if TYPE_CHECKING:
    from .gp7_app_qt import GP7AppQt


class ScriptProgramAPI:
    """Read/write façade exposed tới user Python script qua biến `p`.

    Cho phép script append instruction vào job hiện tại + đọc target library.
    """

    def __init__(self, app: "GP7AppQt") -> None:
        self._app = app

    @property
    def targets(self) -> dict:
        """Read-only view của target library."""
        return dict(self._app._targets)

    @property
    def active_job(self) -> str:
        return self._app._active_job

    def add_movej(self, joints: list[float]) -> None:
        """MoveJ với joints (6 deg)."""
        if len(joints) != 6:
            raise ValueError(f"joints phải 6 phần tử, got {len(joints)}")
        self._app._program.append(
            Instruction(type="MoveJ", joints=[float(q) for q in joints]))

    def add_movel(self, tcp_pose: list[float]) -> None:
        """MoveL với TCP pose [X,Y,Z mm, Rx,Ry,Rz deg] (WORLD frame)."""
        if len(tcp_pose) != 6:
            raise ValueError(f"tcp_pose phải 6 phần tử, got {len(tcp_pose)}")
        self._app._program.append(
            Instruction(type="MoveL", tcp_pose=[float(v) for v in tcp_pose]))

    def add_movej_to(self, target_name: str) -> None:
        """MoveJ → named target."""
        if target_name not in self._app._targets:
            raise KeyError(f"Target '{target_name}' không tồn tại")
        self._app._program.append(
            Instruction(type="MoveJ", target_name=target_name))

    def add_movel_to(self, target_name: str) -> None:
        """MoveL → named target."""
        if target_name not in self._app._targets:
            raise KeyError(f"Target '{target_name}' không tồn tại")
        self._app._program.append(
            Instruction(type="MoveL", target_name=target_name))

    def add_grip(self, close: bool) -> None:
        """SetGripper. close=True → CLOSE / False → OPEN."""
        self._app._program.append(
            Instruction(type="SetGripper", gripper_close=bool(close)))

    def add_wait(self, seconds: float) -> None:
        self._app._program.append(
            Instruction(type="Wait", wait_seconds=float(seconds)))

    def add_setspeed(self, vj_pct: float, v_mm_s: float) -> None:
        self._app._program.append(Instruction(
            type="SetSpeed",
            speed_joint_pct=float(vj_pct),
            speed_linear_mm_s=float(v_mm_s)))

    def add_msg(self, text: str) -> None:
        self._app._program.append(
            Instruction(type="ShowMessage", message=str(text)[:32]))

    def add_call(self, job_name: str) -> None:
        safe = "".join(c for c in str(job_name) if c.isalnum() or c == "_")[:32].upper()
        if not safe:
            raise ValueError(f"job_name không hợp lệ: '{job_name}'")
        self._app._program.append(Instruction(type="CallJob", job_name=safe))
