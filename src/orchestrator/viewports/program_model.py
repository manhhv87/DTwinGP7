"""
program_model.py
────────────────
Instruction dataclass dùng chung giữa các GUI app (gp7_app.py Open3D legacy +
gp7_app_qt.py PyQt6+VTK). Tách khỏi gp7_app.py để Qt app không phải import
ngược vào Open3D module chỉ để lấy 1 dataclass.

Schema = RoboDK-style program: list[Instruction] đã có sẵn JSON serialization
v3 với backward-compat cho legacy program files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Instruction:
    """1 dòng program. Type quyết định trường nào dùng.

    Motion:
      MoveJ        → joints (6 deg) — MOVJ
      MoveL        → tcp_pose (X,Y,Z mm + Rx,Ry,Rz deg WORLD) — MOVL
      MoveC        → tcp_pose_mid + tcp_pose (mid + end) — MOVC

    I/O + timing:
      SetGripper   → gripper_close (True=Close / False=Open) — DOUT
      Wait         → wait_seconds — TIMER
      WaitIO       → io_index, io_state (True=ON), io_timeout_s (0=∞) — WAIT IN#

    Modal state (áp vào MOVJ/MOVL/MOVC kế tiếp):
      SetSpeed     → speed_joint_pct (VJ=) + speed_linear_mm_s (V=)
      SetRounding  → rounding_pl (0..8, PL=)
      SetTool      → tool_no (TL=)
      SetRefFrame  → ref_frame_no (UF#=)

    Operator:
      ShowMessage  → message (≤ 32 ASCII) — MSG
    """

    type: str
    # Motion (inline pose, used khi target_name == "")
    joints: list[float] = field(default_factory=list)
    tcp_pose: list[float] = field(default_factory=list)
    tcp_pose_mid: list[float] = field(default_factory=list)
    # Target library reference. Khi non-empty + type ∈ {MoveJ,MoveL}, motion
    # dereferences via app._targets[target_name] → tái sử dụng pose chung
    # giữa nhiều instructions (RoboDK-style).
    target_name: str = ""
    # Gripper / timing
    gripper_close: bool = False
    wait_seconds: float = 0.0
    # WaitIO
    io_index: int = 1
    io_state: bool = True
    io_timeout_s: float = 0.0           # 0 = block forever
    # SetDO — generic digital output (general-purpose, thay cho gripper-specific).
    do_index: int = 1
    do_state: bool = True               # True = ON
    # Modal state
    speed_joint_pct: float = 10.0
    speed_linear_mm_s: float = 100.0
    rounding_pl: int = 0
    tool_no: int = 0
    ref_frame_no: int = 0
    # Operator
    message: str = ""
    # Sub-program call (CALL JOB:job_name)
    job_name: str = ""
    # Simulation event — checkpoint/trigger không export ra INFORM (chỉ log).
    # event_name = identifier ngắn (e.g. "CHECKPOINT_1"), event_payload = info chi tiết.
    event_name: str = ""
    event_payload: str = ""

    def describe(self, modal: dict | None = None) -> str:
        """Render dòng theo cú pháp INFORM III (MOVJ/MOVL/MOVC/DOUT/WAIT/TIMER/
        MSG/CALL) — giống pendant Yaskawa thật.

        Tốc độ/PL/TOOL/UF là MODAL trong editor (set qua SET* riêng → áp vào
        move kế tiếp; export .JBI fold vào từng MOV line). Khi `modal` được
        truyền (từ program list, tích luỹ theo thứ tự lệnh), move hiển thị ĐẦY
        ĐỦ tag inline: `MOVJ <pos> VJ=10.00 PL=0`. Không có modal (vd status
        toast) → move chỉ hiện mnemonic + position.

        modal keys: vj(float %), v(float mm/s), pl(int|None), tl(int|None),
        uf(int|None).
        """
        t = self.type

        def _move_tail(is_joint: bool) -> str:
            if not modal:
                return ""
            parts: list[str] = []
            if is_joint:
                parts.append(f"VJ={float(modal.get('vj', 0.0)):.2f}")
            else:
                parts.append(f"V={float(modal.get('v', 0.0)):.1f}")
            if modal.get("pl") is not None:
                parts.append(f"PL={int(modal['pl'])}")
            if modal.get("tl") is not None:
                parts.append(f"TL={int(modal['tl'])}")
            if modal.get("uf") is not None:
                parts.append(f"UF#({int(modal['uf'])})")
            return "  " + " ".join(parts)

        if t == "MoveJ":
            pos = (self.target_name if self.target_name
                   else "[" + " ".join(f"{q:+.1f}" for q in self.joints) + "]")
            return f"MOVJ  {pos}{_move_tail(True)}"
        if t == "MoveL":
            if self.target_name:
                pos = self.target_name
            else:
                p = self.tcp_pose
                pos = (f"P(X{p[0]:+.0f} Y{p[1]:+.0f} Z{p[2]:+.0f} "
                       f"Rx{p[3]:+.0f} Ry{p[4]:+.0f} Rz{p[5]:+.0f})")
            return f"MOVL  {pos}{_move_tail(False)}"
        if t == "MoveC":
            m = self.tcp_pose_mid; e = self.tcp_pose
            return (f"MOVC  MID(X{m[0]:+.0f} Y{m[1]:+.0f} Z{m[2]:+.0f}) "
                    f"END(X{e[0]:+.0f} Y{e[1]:+.0f} Z{e[2]:+.0f}){_move_tail(False)}")
        if t == "SetGripper":
            # Legacy gripper → DOUT bit 1 cố định (xem inform_codegen).
            return f"DOUT  OT#(1) {'ON' if self.gripper_close else 'OFF'}"
        if t == "SetDO":
            return f"DOUT  OT#({self.do_index}) {'ON' if self.do_state else 'OFF'}"
        if t == "Wait":
            return f"TIMER  T={self.wait_seconds:.2f}"
        if t == "WaitIO":
            tout = f" T={self.io_timeout_s:.2f}" if self.io_timeout_s > 0 else ""
            return (f"WAIT  IN#({self.io_index})="
                    f"{'ON' if self.io_state else 'OFF'}{tout}")
        if t == "SetSpeed":
            return (f"SET SPEED  VJ={self.speed_joint_pct:.2f} "
                    f"V={self.speed_linear_mm_s:.1f}")
        if t == "SetRounding":
            return f"SET PL={self.rounding_pl}"
        if t == "SetTool":
            return f"SET TOOL  TL#({self.tool_no})"
        if t == "SetRefFrame":
            return f"SET UFRAME  UF#({self.ref_frame_no})"
        if t == "ShowMessage":
            return f'MSG  "{self.message[:32]}"'
        if t == "CallJob":
            return f"CALL  JOB:{self.job_name}"
        if t == "SimEvent":
            pl = f" ({self.event_payload[:24]})" if self.event_payload else ""
            return f"'⚑ SIMEVENT {self.event_name}{pl}"
        return f"?{t}"

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"type": self.type}
        t = self.type
        if t == "MoveJ":
            if self.target_name:
                d["target_name"] = self.target_name
            else:
                d["joints"] = list(self.joints)
        elif t == "MoveL":
            if self.target_name:
                d["target_name"] = self.target_name
            else:
                d["tcp_pose"] = list(self.tcp_pose)
        elif t == "MoveC":
            d["tcp_pose_mid"] = list(self.tcp_pose_mid)
            d["tcp_pose"] = list(self.tcp_pose)
        elif t == "SetGripper":
            d["close"] = bool(self.gripper_close)
        elif t == "SetDO":
            d["do_index"] = int(self.do_index)
            d["do_state"] = bool(self.do_state)
        elif t == "Wait":
            d["seconds"] = float(self.wait_seconds)
        elif t == "WaitIO":
            d["io_index"] = int(self.io_index)
            d["io_state"] = bool(self.io_state)
            d["io_timeout_s"] = float(self.io_timeout_s)
        elif t == "SetSpeed":
            d["speed_joint_pct"] = float(self.speed_joint_pct)
            d["speed_linear_mm_s"] = float(self.speed_linear_mm_s)
        elif t == "SetRounding":
            d["rounding_pl"] = int(self.rounding_pl)
        elif t == "SetTool":
            d["tool_no"] = int(self.tool_no)
        elif t == "SetRefFrame":
            d["ref_frame_no"] = int(self.ref_frame_no)
        elif t == "ShowMessage":
            d["message"] = str(self.message)
        elif t == "CallJob":
            d["job_name"] = str(self.job_name)
        elif t == "SimEvent":
            d["event_name"] = str(self.event_name)
            d["event_payload"] = str(self.event_payload)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Instruction":
        t = d["type"]
        if t == "MoveJ":
            if "target_name" in d:
                return cls(type=t, target_name=str(d["target_name"]))
            return cls(type=t, joints=list(d["joints"]))
        if t == "MoveL":
            if "target_name" in d:
                return cls(type=t, target_name=str(d["target_name"]))
            return cls(type=t, tcp_pose=list(d["tcp_pose"]))
        if t == "MoveC":
            return cls(type=t,
                       tcp_pose_mid=list(d["tcp_pose_mid"]),
                       tcp_pose=list(d["tcp_pose"]))
        if t == "SetGripper":
            return cls(type=t, gripper_close=bool(d["close"]))
        if t == "SetDO":
            return cls(type=t,
                       do_index=int(d.get("do_index", 1)),
                       do_state=bool(d.get("do_state", True)))
        if t == "Wait":
            return cls(type=t, wait_seconds=float(d["seconds"]))
        if t == "WaitIO":
            return cls(type=t,
                       io_index=int(d.get("io_index", 1)),
                       io_state=bool(d.get("io_state", True)),
                       io_timeout_s=float(d.get("io_timeout_s", 0.0)))
        if t == "SetSpeed":
            return cls(type=t,
                       speed_joint_pct=float(d.get("speed_joint_pct", 10.0)),
                       speed_linear_mm_s=float(d.get("speed_linear_mm_s", 100.0)))
        if t == "SetRounding":
            return cls(type=t, rounding_pl=int(d.get("rounding_pl", 0)))
        if t == "SetTool":
            return cls(type=t, tool_no=int(d.get("tool_no", 0)))
        if t == "SetRefFrame":
            return cls(type=t, ref_frame_no=int(d.get("ref_frame_no", 0)))
        if t == "ShowMessage":
            return cls(type=t, message=str(d.get("message", "")))
        if t == "CallJob":
            return cls(type=t, job_name=str(d.get("job_name", "")))
        if t == "SimEvent":
            return cls(type=t,
                       event_name=str(d.get("event_name", "")),
                       event_payload=str(d.get("event_payload", "")))
        raise ValueError(f"Unknown instruction type: {t}")
