"""
script_api.py
─────────────
ScriptProgramAPI: read/write facade exposed to the user Python script
(embedded script editor) via the `p` variable. Decouples the layer from
GP7AppQt to limit the surface the script can touch of app internals.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .program_model import Instruction

if TYPE_CHECKING:
    from .gp7_app_qt import GP7AppQt


class ScriptProgramAPI:
    """Read/write facade exposed to the user Python script via the `p` variable.

    Allows the script to append instructions to the current job and read the target library.
    """

    def __init__(self, app: "GP7AppQt") -> None:
        self._app = app

    def _append(self, ins: Instruction) -> None:
        """Append one instruction, refusing while playback is running — the play
        loop iterates the (possibly CALL JOB) instruction list live, so a concurrent
        append would corrupt that iteration."""
        if self._app._playback_running():
            raise RuntimeError(
                "Cannot modify the program while it is running — stop playback first")
        self._app._program.append(ins)

    @property
    def targets(self) -> dict:
        """Read-only view of the target library."""
        return dict(self._app._targets)

    @property
    def active_job(self) -> str:
        return self._app._active_job

    def add_movej(self, joints: list[float]) -> None:
        """MoveJ with joints (6 deg)."""
        if len(joints) != 6:
            raise ValueError(f"joints must have 6 elements, got {len(joints)}")
        self._append(Instruction(type="MoveJ", joints=[float(q) for q in joints]))

    def add_movel(self, tcp_pose: list[float]) -> None:
        """MoveL with TCP pose [X,Y,Z mm, Rx,Ry,Rz deg] (WORLD frame)."""
        if len(tcp_pose) != 6:
            raise ValueError(f"tcp_pose must have 6 elements, got {len(tcp_pose)}")
        self._append(Instruction(type="MoveL", tcp_pose=[float(v) for v in tcp_pose]))

    def add_movej_to(self, target_name: str) -> None:
        """MoveJ → named target."""
        if target_name not in self._app._targets:
            raise KeyError(f"Target '{target_name}' does not exist")
        self._append(Instruction(type="MoveJ", target_name=target_name))

    def add_movel_to(self, target_name: str) -> None:
        """MoveL → named target."""
        if target_name not in self._app._targets:
            raise KeyError(f"Target '{target_name}' does not exist")
        self._append(Instruction(type="MoveL", target_name=target_name))

    def add_grip(self, close: bool) -> None:
        """SetGripper. close=True → CLOSE / False → OPEN."""
        self._append(Instruction(type="SetGripper", gripper_close=bool(close)))

    def add_wait(self, seconds: float) -> None:
        self._append(Instruction(type="Wait", wait_seconds=float(seconds)))

    def add_setspeed(self, vj_pct: float, v_mm_s: float) -> None:
        self._append(Instruction(
            type="SetSpeed",
            speed_joint_pct=float(vj_pct),
            speed_linear_mm_s=float(v_mm_s)))

    def add_msg(self, text: str) -> None:
        self._append(Instruction(type="ShowMessage", message=str(text)[:32]))

    def add_call(self, job_name: str) -> None:
        safe = "".join(c for c in str(job_name) if c.isalnum() or c == "_")[:32].upper()
        if not safe:
            raise ValueError(f"job_name is invalid: '{job_name}'")
        self._append(Instruction(type="CallJob", job_name=safe))
