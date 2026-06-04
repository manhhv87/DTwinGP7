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
    # Compound condition (ANDEXP/OREXP). When cond_terms is non-empty it takes
    # precedence over the single cond_lhs/op/rhs. Each term = (lhs, op, rhs);
    # cond_join ∈ {"AND","OR"} combines them. INFORM: 'a<>1 ANDEXP b<>2'.
    cond_terms: list = field(default_factory=list)
    cond_join: str = ""                 # "AND" / "OR" / "" (single term)
    # Whether the condition used the INFORM "expression" keyword form
    # (IFTHENEXP/ELSEIFEXP/WHILEEXP). Preserved so re-synthesis reproduces the
    # original keyword even for variable conditions (not just I/O).
    cond_exp: bool = False
    # ── Extended INFORM (LG production jobs) ──
    # Clear: zero `clear_count` consecutive vars from var_name (-1 = ALL).
    clear_count: int = 0
    # DIN/DOUT group I/O: DIN Bxxx IG#(n) / SOUT#(n); DOUT OG#(n) Bxxx.
    io_group: int = 0                   # group index (IG#/OG#/SOUT#)
    io_group_kind: str = ""             # "IG" / "OG" / "SOUT"
    # SET Ixxx EXPRESS <expr> — arithmetic expression string (e.g. "5 * B005").
    var_expr: str = ""
    # Variable speed: V=Ixxx / VJ=Ixxx — speed comes from a variable at runtime.
    speed_var: str = ""
    # P[Bxxx] indirect addressing — position index taken from this variable.
    pos_index_var: str = ""

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

        def _spd(is_joint: bool) -> str:
            # Variable speed (V=Ixxx) overrides the modal numeric tail.
            if self.speed_var:
                return f"  {'VJ' if is_joint else 'V'}={self.speed_var}"
            return _move_tail(is_joint)

        if t == "MoveJ":
            if self.pos_index_var:
                pos = f"P[{self.pos_index_var}]"
            elif self.target_name:
                pos = self.target_name
            else:
                pos = "[" + " ".join(f"{q:+.1f}" for q in self.joints) + "]"
            return f"MOVJ  {pos}{_spd(True)}"
        if t == "MoveL":
            if self.pos_index_var:
                pos = f"P[{self.pos_index_var}]"
            elif self.target_name:
                pos = self.target_name
            else:
                p = self.tcp_pose
                pos = (f"P(X{p[0]:+.0f} Y{p[1]:+.0f} Z{p[2]:+.0f} "
                       f"Rx{p[3]:+.0f} Ry{p[4]:+.0f} Rz{p[5]:+.0f})")
            return f"MOVL  {pos}{_spd(False)}"
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
            if self.var_expr:
                return f"SET  {self.var_name} EXPRESS {self.var_expr}"
            return f"{op}  {self.var_name} {self.var_arg}"
        if t == "PulseDO":
            return f"PULSE  OT#({self.do_index})"
        if t == "ClearStack":
            return "CLEAR  STACK"
        if t == "ClearVar":
            n = "ALL" if self.clear_count < 0 else self.clear_count
            return f"CLEAR  {self.var_name} {n}"
        if t == "ReadGroupIn":
            return f"DIN  {self.var_name} {self.io_group_kind}#({self.io_group})"
        if t == "WriteGroupOut":
            return f"DOUT  OG#({self.io_group}) {self.var_name}"
        exp = "EXP" if self.cond_exp else ""     # IFTHENEXP/ELSEIFEXP/WHILEEXP
        if t == "IfThen":
            return f"IFTHEN{exp}  {self._cond_str()}"
        if t == "ElseIf":
            return f"ELSEIF{exp}  {self._cond_str()}"
        if t == "Else":
            return "ELSE"
        if t == "EndIf":
            return "ENDIF"
        if t == "While":
            return f"WHILE{exp}  {self._cond_str()}"
        if t == "EndWhile":
            return "ENDWHILE"
        return f"?{t}"

    def _cond_str(self) -> str:
        """Render an INFORM condition. Compound (cond_terms) → 'a<>1 ANDEXP b<>2';
        single → 'lhs op rhs' (e.g. 'B000>5'). Empty if not set."""
        if self.cond_terms:
            joiner = f" {self.cond_join or 'AND'}EXP "
            return joiner.join(f"{l}{o}{r}" for l, o, r in self.cond_terms)
        if not self.cond_op:
            return ""
        return f"{self.cond_lhs}{self.cond_op}{self.cond_rhs}"

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"type": self.type}
        t = self.type
        if t == "MoveJ":
            if self.pos_index_var:
                d["pos_index_var"] = self.pos_index_var
            elif self.target_name:
                d["target_name"] = self.target_name
            else:
                d["joints"] = list(self.joints)
            if self.speed_var:
                d["speed_var"] = self.speed_var
        elif t == "MoveL":
            if self.pos_index_var:
                d["pos_index_var"] = self.pos_index_var
            elif self.target_name:
                d["target_name"] = self.target_name
            else:
                d["tcp_pose"] = list(self.tcp_pose)
            if self.speed_var:
                d["speed_var"] = self.speed_var
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
            self._cond_to_dict(d)
        elif t == "SetVar":
            d["var_name"] = str(self.var_name)
            d["var_op"] = str(self.var_op)
            if self.var_expr:
                d["var_expr"] = str(self.var_expr)
            elif self.var_op.upper() not in ("INC", "DEC"):
                d["var_arg"] = str(self.var_arg)
        elif t in ("IfThen", "ElseIf", "While"):
            self._cond_to_dict(d)
        elif t == "PulseDO":
            d["do_index"] = int(self.do_index)
        elif t == "ClearVar":
            d["var_name"] = str(self.var_name)
            d["clear_count"] = int(self.clear_count)
        elif t in ("ReadGroupIn", "WriteGroupOut"):
            d["var_name"] = str(self.var_name)
            d["io_group"] = int(self.io_group)
            d["io_group_kind"] = str(self.io_group_kind)
        # Else / EndIf / EndWhile / ClearStack: only "type" is needed.
        return d

    def _cond_to_dict(self, d: dict) -> None:
        """Serialize condition fields (single or compound) into d."""
        if self.cond_terms:
            d["cond_terms"] = [list(t) for t in self.cond_terms]
            d["cond_join"] = str(self.cond_join or "AND")
        elif self.cond_op:
            d["cond_lhs"] = str(self.cond_lhs)
            d["cond_op"] = str(self.cond_op)
            d["cond_rhs"] = str(self.cond_rhs)
        if self.cond_exp:
            d["cond_exp"] = True

    @classmethod
    def from_dict(cls, d: dict) -> "Instruction":
        t = d["type"]
        if t == "MoveJ":
            sv = str(d.get("speed_var", ""))
            if "pos_index_var" in d:
                return cls(type=t, pos_index_var=str(d["pos_index_var"]),
                           speed_var=sv)
            if "target_name" in d:
                return cls(type=t, target_name=str(d["target_name"]), speed_var=sv)
            return cls(type=t, joints=list(d["joints"]), speed_var=sv)
        if t == "MoveL":
            sv = str(d.get("speed_var", ""))
            if "pos_index_var" in d:
                return cls(type=t, pos_index_var=str(d["pos_index_var"]),
                           speed_var=sv)
            if "target_name" in d:
                return cls(type=t, target_name=str(d["target_name"]), speed_var=sv)
            return cls(type=t, tcp_pose=list(d["tcp_pose"]), speed_var=sv)
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
            ins = cls(type=t, label_name=str(d.get("label_name", "")))
            cls._cond_from_dict(ins, d)
            return ins
        if t == "SetVar":
            return cls(type=t,
                       var_name=str(d.get("var_name", "")),
                       var_op=str(d.get("var_op", "SET")),
                       var_arg=str(d.get("var_arg", "")),
                       var_expr=str(d.get("var_expr", "")))
        if t in ("IfThen", "ElseIf", "While"):
            ins = cls(type=t)
            cls._cond_from_dict(ins, d)
            return ins
        if t == "PulseDO":
            return cls(type=t, do_index=int(d.get("do_index", 1)))
        if t == "ClearVar":
            return cls(type=t, var_name=str(d.get("var_name", "")),
                       clear_count=int(d.get("clear_count", 0)))
        if t in ("ReadGroupIn", "WriteGroupOut"):
            return cls(type=t, var_name=str(d.get("var_name", "")),
                       io_group=int(d.get("io_group", 0)),
                       io_group_kind=str(d.get("io_group_kind", "")))
        if t in ("Else", "EndIf", "EndWhile", "ClearStack"):
            return cls(type=t)
        raise ValueError(f"Unknown instruction type: {t}")

    @staticmethod
    def _cond_from_dict(ins: "Instruction", d: dict) -> None:
        """Load condition fields (single or compound) from d into ins."""
        if "cond_terms" in d:
            ins.cond_terms = [tuple(t) for t in d["cond_terms"]]
            ins.cond_join = str(d.get("cond_join", "AND"))
            if ins.cond_terms:               # mirror first term for back-compat
                ins.cond_lhs, ins.cond_op, ins.cond_rhs = ins.cond_terms[0]
        else:
            ins.cond_lhs = str(d.get("cond_lhs", ""))
            ins.cond_op = str(d.get("cond_op", ""))
            ins.cond_rhs = str(d.get("cond_rhs", ""))
        ins.cond_exp = bool(d.get("cond_exp", False))
