"""
program_model.py
────────────────
Instruction dataclass shared between GUI apps (gp7_app.py Open3D legacy +
gp7_app_qt.py PyQt6+VTK). Decoupled from gp7_app.py so the Qt app does not
need to import the Open3D module just to obtain one dataclass.

Schema = RoboDK-style program: list[Instruction] with JSON serialization
v3 and backward-compat for legacy program files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Instruction:
    """One program line. The type field determines which fields are used.

    Motion:
      MoveJ        → joints (6 deg) — MOVJ
      MoveL        → tcp_pose (X,Y,Z mm + Rx,Ry,Rz deg WORLD) — MOVL
      MoveC        → tcp_pose_mid + tcp_pose (mid + end) — MOVC

    I/O + timing:
      SetGripper   → gripper_close (True=Close / False=Open) — DOUT
      Wait         → wait_seconds — TIMER
      WaitIO       → io_index, io_state (True=ON), io_timeout_s (0=∞) — WAIT IN#

    Modal state (applied to the next MOVJ/MOVL/MOVC):
      SetSpeed     → speed_joint_pct (VJ=) + speed_linear_mm_s (V=)
      SetRounding  → rounding_pl (0..8, PL=)
      SetTool      → tool_no (TL=)
      SetRefFrame  → ref_frame_no (UF#=)

    Operator:
      ShowMessage  → message (≤ 32 ASCII) — MSG

    Flow / variables (INFORM logic — Tier B1 flat):
      Label        → label_name — *LABEL
      Jump         → label_name + optional cond (cond_op="" ⇒ unconditional) —
                     JUMP *LABEL [IF <cond>]
      SetVar       → var_name (B###/I###), var_op (SET/ADD/SUB/MUL/DIV/INC/DEC),
                     var_arg (operand token; ignored for INC/DEC) — SET/ADD/...

    Structured (INFORM logic — Tier B2): IfThen/ElseIf/Else/EndIf (cond on
    IfThen/ElseIf) + While/EndWhile (cond on While). cond = cond_lhs cond_op
    cond_rhs (op ∈ =,<>,>,<,>=,<=; operands = B###/I### | literal | IN#(n)).
    """

    type: str
    # Motion (inline pose, used when target_name == "")
    joints: list[float] = field(default_factory=list)
    tcp_pose: list[float] = field(default_factory=list)
    tcp_pose_mid: list[float] = field(default_factory=list)
    # Target library reference. When non-empty + type ∈ {MoveJ,MoveL}, motion
    # dereferences via app._targets[target_name] → reuses a shared pose
    # across multiple instructions (RoboDK-style).
    target_name: str = ""
    # Gripper / timing
    gripper_close: bool = False
    wait_seconds: float = 0.0
    # WaitIO
    io_index: int = 1
    io_state: bool = True
    io_timeout_s: float = 0.0           # 0 = block forever
    # SetDO — generic digital output (general-purpose, replacing gripper-specific).
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
    # Simulation event — checkpoint/trigger not exported to INFORM (log only).
    # event_name = short identifier (e.g. "CHECKPOINT_1"), event_payload = detail info.
    event_name: str = ""
    event_payload: str = ""
    # Flow / variables (INFORM logic). label_name used for Label + Jump.
    label_name: str = ""
    var_name: str = ""                  # B### / I###
    var_op: str = ""                    # SET/ADD/SUB/MUL/DIV/INC/DEC
    var_arg: str = ""                   # operand token (literal | var); "" for INC/DEC
    # Condition (Jump IF / IfThen / ElseIf / While). cond_op="" ⇒ unconditional.
    cond_lhs: str = ""
    cond_op: str = ""                   # =, <>, >, <, >=, <=
    cond_rhs: str = ""

    def describe(self, modal: dict | None = None) -> str:
        """Render the line using INFORM III syntax (MOVJ/MOVL/MOVC/DOUT/WAIT/TIMER/
        MSG/CALL) — matching a real Yaskawa pendant.

        Speed/PL/TOOL/UF are MODAL in the editor (set via dedicated SET* instructions
        → applied to the next move; .JBI export folds them into each MOV line).
        When `modal` is provided (from program list, accumulated in instruction order),
        the move displays FULL inline tags: `MOVJ <pos> VJ=10.00 PL=0`. Without modal
        (e.g. status toast) → move shows only mnemonic + position.

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
            # Legacy gripper → DOUT bit 1 fixed (see inform_codegen).
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
        # ── Flow / variables (INFORM logic) ──
        if t == "Label":
            return f"*{self.label_name}"
        if t == "Jump":
            tail = f"  IF {self._cond_str()}" if self.cond_op else ""
            return f"JUMP  *{self.label_name}{tail}"
        if t == "SetVar":
            op = self.var_op.upper()
            if op in ("INC", "DEC"):
                return f"{op}  {self.var_name}"
            return f"{op}  {self.var_name} {self.var_arg}"
        if t == "IfThen":
            return f"IFTHEN  {self._cond_str()}"
        if t == "ElseIf":
            return f"ELSEIF  {self._cond_str()}"
        if t == "Else":
            return "ELSE"
        if t == "EndIf":
            return "ENDIF"
        if t == "While":
            return f"WHILE  {self._cond_str()}"
        if t == "EndWhile":
            return "ENDWHILE"
        return f"?{t}"

    def _cond_str(self) -> str:
        """Render an INFORM condition 'lhs op rhs' (e.g. 'B000>5'). Empty if not set."""
        if not self.cond_op:
            return ""
        return f"{self.cond_lhs}{self.cond_op}{self.cond_rhs}"

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
        elif t == "Label":
            d["label_name"] = str(self.label_name)
        elif t == "Jump":
            d["label_name"] = str(self.label_name)
            if self.cond_op:
                d["cond_lhs"] = str(self.cond_lhs)
                d["cond_op"] = str(self.cond_op)
                d["cond_rhs"] = str(self.cond_rhs)
        elif t == "SetVar":
            d["var_name"] = str(self.var_name)
            d["var_op"] = str(self.var_op)
            if self.var_op.upper() not in ("INC", "DEC"):
                d["var_arg"] = str(self.var_arg)
        elif t in ("IfThen", "ElseIf", "While"):
            d["cond_lhs"] = str(self.cond_lhs)
            d["cond_op"] = str(self.cond_op)
            d["cond_rhs"] = str(self.cond_rhs)
        # Else / EndIf / EndWhile: only "type" is needed.
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
        if t == "Label":
            return cls(type=t, label_name=str(d.get("label_name", "")))
        if t == "Jump":
            return cls(type=t,
                       label_name=str(d.get("label_name", "")),
                       cond_lhs=str(d.get("cond_lhs", "")),
                       cond_op=str(d.get("cond_op", "")),
                       cond_rhs=str(d.get("cond_rhs", "")))
        if t == "SetVar":
            return cls(type=t,
                       var_name=str(d.get("var_name", "")),
                       var_op=str(d.get("var_op", "SET")),
                       var_arg=str(d.get("var_arg", "")))
        if t in ("IfThen", "ElseIf", "While"):
            return cls(type=t,
                       cond_lhs=str(d.get("cond_lhs", "")),
                       cond_op=str(d.get("cond_op", "")),
                       cond_rhs=str(d.get("cond_rhs", "")))
        if t in ("Else", "EndIf", "EndWhile"):
            return cls(type=t)
        raise ValueError(f"Unknown instruction type: {t}")
