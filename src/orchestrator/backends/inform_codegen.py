"""
inform_codegen.py
─────────────────
Generate INFORM job files (.JBI) for the Yaskawa YRC1000.

INFORM is Yaskawa's proprietary robot programming language. Each .JBI file is a
program the robot executes upon receiving JOB_SELECT + START via HSE.

Format reference:
  - Yaskawa "INFORM Language Manual" (RE-CKI-A464)
  - C-variables (C00000-C99999): position constants
  - POSTYPE PULSE: raw encoder pulses (unit 0.0001° per axis, ratio-dependent)
  - Main instructions: MOVJ (joint), MOVL (linear), DOUT (digital output), TIMER

This module is PURE TEXT — no IO. Test via snapshot strings. Upload via FTP
in `motoman_hse.py`.

Workflow:
  builder = InformJobBuilder("PICKPLACE")
  builder.add_position("p0", joints_deg=[0,0,0,0,0,0])
  builder.add_position("p1", joints_deg=[10,-5,20,0,30,-15])
  builder.movj("p0", speed_pct=10.0)
  builder.movj("p1", speed_pct=10.0)
  builder.dout(1, True)        # gripper close
  builder.timer(0.3)
  builder.movj("p0", speed_pct=10.0)
  builder.dout(1, False)
  text = builder.render()       # str → write .JBI then FTP upload
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from .hse_protocol import GP7_PULSE_PER_DEG

# Max 999 C-variables/job but keep threshold low for safety.
MAX_POSITIONS_PER_JOB = 100

# Speed limits (Yaskawa convention) — clamp here to prevent user passing 100%.
MAX_SPEED_PCT_DEFAULT = 30.0     # joint speed %
MAX_LINEAR_MM_S = 250.0          # cartesian mm/s

PosType = Literal["PULSE", "BASE", "ROBOT", "USER"]


# ───── Internal instruction dataclasses ─────


@dataclass
class _Position:
    """Position constant declared in //POS.

    letter = 'C' (global C-variable) or 'P' (job position variable). index is the
    numeric id; the declaration token is always 5-digit ({letter}{index:05d}),
    while the //INST reference is 3-digit for P-vars and 5-digit for C-vars
    (Yaskawa teach-pendant convention).
    """

    name: str
    joints_pulse: list[int]
    letter: str = "C"
    index: int = 0
    tool_no: int = 0
    user_frame: int = 0


@dataclass
class _Instruction:
    """One INFORM instruction line in the //INST section."""

    text: str


# ───── Builder ─────


class InformJobBuilder:
    """Build an INFORM .JBI from python-level commands.

    Args:
        name: Job name (≤ 32 ASCII chars, no spaces).
        pos_type: POSTYPE — PULSE (default) for joint pulses.
        pulse_per_deg: Pulse/degree ratio per axis. Default GP7.
        max_speed_pct: Joint speed cap for safety. Default 30%.
        group: GROUP1 RB1 (single robot). Change for multi-robot setups.
        folder_name: Optional ///FOLDERNAME (controller folder). "" = omit line.
    """

    def __init__(
        self,
        name: str,
        pos_type: PosType = "PULSE",
        pulse_per_deg: tuple[float, ...] = GP7_PULSE_PER_DEG,
        max_speed_pct: float = MAX_SPEED_PCT_DEFAULT,
        group: str = "RB1",
        emit_axis_count: int = 0,
        folder_name: str = "",
    ) -> None:
        """Args:
            emit_axis_count: Number of axis values emitted in the C-var line.
                0 (default) = len(pulse_per_deg), i.e. 6 for GP7. Some YRC1000
                firmware/parameter configs require 8 (pad axes 7-8 = 0); set
                emit_axis_count=8 in that case. Verify with a dry-upload on the
                controller before production.
        """
        self._validate_name(name)
        self.name = name
        self.pos_type = pos_type
        self.pulse_per_deg = pulse_per_deg
        self.max_speed_pct = max_speed_pct
        self.group = group
        self.folder_name = folder_name
        self.emit_axis_count = (int(emit_axis_count) if emit_axis_count > 0
                                 else len(pulse_per_deg))
        if self.emit_axis_count < len(pulse_per_deg):
            raise ValueError(
                f"emit_axis_count ({self.emit_axis_count}) must be ≥ number of joints "
                f"({len(pulse_per_deg)})")
        self._positions: list[_Position] = []
        self._pos_index: dict[str, int] = {}      # name → index in _positions
        self._instructions: list[_Instruction] = []

    @staticmethod
    def _validate_name(name: str) -> None:
        if not name or len(name) > 32:
            raise ValueError(f"Job name must be 1-32 characters, got '{name}'")
        if not name.replace("_", "").isalnum():
            raise ValueError(f"Job name must be alphanumeric/_, got '{name}'")
        # INFORM convention: must start with a letter (digit-start is unstable on
        # many YRC1000 firmware versions — JOB_SELECT may fail silently).
        if not name[0].isalpha():
            raise ValueError(
                f"Job name must start with a letter, got '{name}'")

    @staticmethod
    def _validate_label(name: str) -> None:
        """Validate a *LABEL name. INFORM labels allow purely-numeric names
        (e.g. *1, *12) as well as alphanumeric — looser than job names."""
        if not name or len(name) > 32:
            raise ValueError(f"Label must be 1-32 characters, got '{name}'")
        if not name.replace("_", "").isalnum():
            raise ValueError(f"Label must be alphanumeric/_, got '{name}'")

    # ─── Positions ───
    def add_position(
        self, name: str, joints_deg: list[float],
        pos_token: str | None = None,
    ) -> "InformJobBuilder":
        """Add a position from joint angles (degrees).

        Args:
            name: Logical name used to reference this position in movj/movl/movc.
            joints_deg: Joint angles (len = number of axes).
            pos_token: Optional explicit INFORM token, e.g. 'P00001' or 'C00003',
                to preserve the original variable kind + index on round-trip. If
                None, a sequential C-variable (C00000, C00001, …) is assigned.
        """
        if name in self._pos_index:
            raise ValueError(f"Position '{name}' already exists")
        if len(joints_deg) != len(self.pulse_per_deg):
            raise ValueError(
                f"Joints must have {len(self.pulse_per_deg)} elements, "
                f"got {len(joints_deg)}"
            )
        if len(self._positions) >= MAX_POSITIONS_PER_JOB:
            raise ValueError(
                f"Reached limit of {MAX_POSITIONS_PER_JOB} positions/job. "
                f"Split large jobs into multiple files."
            )
        if pos_token is not None:
            m = re.match(r"^([CP])(\d{1,5})$", pos_token.strip(), re.IGNORECASE)
            if not m:
                raise ValueError(
                    f"Invalid pos_token '{pos_token}' (expected C##### or P###)")
            letter, index = m.group(1).upper(), int(m.group(2))
        else:
            letter, index = "C", len(self._positions)
        pulses = [int(round(d * r)) for d, r in zip(joints_deg, self.pulse_per_deg)]
        self._pos_index[name] = len(self._positions)
        self._positions.append(_Position(
            name=name, joints_pulse=pulses, letter=letter, index=index))
        return self

    def _pos_decl_token(self, pos: "_Position") -> str:
        """//POS declaration token — always 5-digit (e.g. 'P00001', 'C00003')."""
        return f"{pos.letter}{pos.index:05d}"

    def _resolve_cvar(self, position_name: str) -> str:
        """Map logical position name → //INST reference token. P-vars use 3-digit
        (P001), C-vars use 5-digit (C00000) — Yaskawa convention."""
        if position_name not in self._pos_index:
            raise KeyError(f"Position '{position_name}' not added — call add_position first")
        pos = self._positions[self._pos_index[position_name]]
        width = 3 if pos.letter == "P" else 5
        return f"{pos.letter}{pos.index:0{width}d}"

    # ─── Motion instructions ───
    @staticmethod
    def _motion_modifiers(
        pl: int | None,
        tool_no: int | None,
        user_frame: int | None,
    ) -> str:
        """Helper: build " PL=n TL=n UF#=n" tail. Yaskawa convention."""
        parts: list[str] = []
        if pl is not None:
            if not (0 <= int(pl) <= 8):
                raise ValueError(f"PL must be 0..8, got {pl}")
            parts.append(f"PL={int(pl)}")
        if tool_no is not None:
            parts.append(f"TL={int(tool_no)}")
        if user_frame is not None:
            parts.append(f"UF#({int(user_frame)})")
        return (" " + " ".join(parts)) if parts else ""

    def movj(self, position_name: str, speed_pct: float | None = None,
             tool_no: int | None = None,
             pl: int | None = None,
             user_frame: int | None = None,
             speed_var: str = "") -> "InformJobBuilder":
        """Joint move to position (high speed, path not controlled).

        Args:
            tool_no: TOOL coordinate (TL=). None = do not emit modifier.
            pl: Position level / rounding 0..8 (PL=). None = do not emit.
            user_frame: User frame index (UF#). None = do not emit.
            speed_var: If set, emit VJ=<var> (e.g. VJ=I000) instead of a number.
        """
        cvar = self._resolve_cvar(position_name)
        if speed_var:
            vj = f"VJ={self._validate_var(speed_var)}"
        else:
            vj = f"VJ={self._clamp_joint_speed(speed_pct):.2f}"
        inst = f"MOVJ {cvar} {vj}"
        inst += self._motion_modifiers(pl, tool_no, user_frame)
        self._instructions.append(_Instruction(inst))
        return self

    def movl(
        self,
        position_name: str,
        speed_mm_s: float = 100.0,
        tool_no: int | None = None,
        pl: int | None = None,
        user_frame: int | None = None,
        speed_var: str = "",
    ) -> "InformJobBuilder":
        """Linear (Cartesian) move. speed_var → emit V=<var> (e.g. V=I003)."""
        cvar = self._resolve_cvar(position_name)
        if speed_var:
            v = f"V={self._validate_var(speed_var)}"
        else:
            v = f"V={max(1.0, min(float(speed_mm_s), MAX_LINEAR_MM_S)):.1f}"
        inst = f"MOVL {cvar} {v}"
        inst += self._motion_modifiers(pl, tool_no, user_frame)
        self._instructions.append(_Instruction(inst))
        return self

    def movc(
        self,
        position_name: str,
        speed_mm_s: float = 100.0,
        tool_no: int | None = None,
        pl: int | None = None,
        user_frame: int | None = None,
    ) -> "InformJobBuilder":
        """Circular move (one waypoint on the arc).

        INFORM MOVC requires ≥3 consecutive MOVC waypoints to define a circular arc
        (start = previous MOVC or pose, mid = MOVC2, end = MOVC3).
        Caller must emit the correct number of MOVC instructions.
        """
        cvar = self._resolve_cvar(position_name)
        v = max(1.0, min(float(speed_mm_s), MAX_LINEAR_MM_S))
        inst = f"MOVC {cvar} V={v:.1f}"
        inst += self._motion_modifiers(pl, tool_no, user_frame)
        self._instructions.append(_Instruction(inst))
        return self

    # ─── I/O + timing ───
    def dout(self, output_index: int, on: bool) -> "InformJobBuilder":
        """Digital output ON/OFF — controls gripper / signal lamp."""
        if not (1 <= output_index <= 1024):
            raise ValueError(f"DOUT index out of range 1..1024: {output_index}")
        state = "ON" if on else "OFF"
        self._instructions.append(_Instruction(f"DOUT OT#({output_index}) {state}"))
        return self

    def wait_in(
        self,
        input_index: int,
        on: bool = True,
        timeout_s: float = 0.0,
    ) -> "InformJobBuilder":
        """WAIT IN#(n)=ON/OFF [T=...]. timeout_s=0 → block indefinitely."""
        if not (1 <= input_index <= 1024):
            raise ValueError(f"WAIT IN index out of range 1..1024: {input_index}")
        state = "ON" if on else "OFF"
        inst = f"WAIT IN#({input_index})={state}"
        if timeout_s > 0:
            if timeout_s > 600:
                raise ValueError(f"WAIT timeout out of range 0-600s: {timeout_s}")
            inst += f" T={timeout_s:.2f}"
        self._instructions.append(_Instruction(inst))
        return self

    def timer(self, seconds: float) -> "InformJobBuilder":
        """Pause for `seconds` seconds (gripper settle time, etc)."""
        if seconds < 0 or seconds > 600:
            raise ValueError(f"TIMER out of range 0-600s: {seconds}")
        # 3 decimals — matches Yaskawa teach-pendant output (e.g. 'TIMER T=1.000').
        self._instructions.append(_Instruction(f"TIMER T={seconds:.3f}"))
        return self

    def msg(self, text: str) -> "InformJobBuilder":
        """MSG "string" — display message on the teach pendant (≤ 32 ASCII)."""
        # INFORM MSG supports ≤ 32 characters, ASCII printable, no embedded quotes.
        clean = "".join(c for c in text if 0x20 <= ord(c) < 0x7F and c != '"')
        clean = clean[:32]
        self._instructions.append(_Instruction(f'MSG "{clean}"'))
        return self

    def call_job(self, job_name: str) -> "InformJobBuilder":
        """CALL JOB:job_name — invoke a sub-program. Real job names may contain
        '-' (e.g. SPEED-1), so this is looser than _validate_name: 1-32 chars,
        alphanumeric plus _ and -."""
        jn = (job_name or "").strip()
        if not jn or len(jn) > 32 or not jn.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid CALL JOB name: '{job_name}'")
        self._instructions.append(_Instruction(f"CALL JOB:{jn}"))
        return self

    def comment(self, text: str) -> "InformJobBuilder":
        """Add a comment 'NOP // <text>' for debugging."""
        safe = text.replace("\r", "").replace("\n", " ")[:80]
        self._instructions.append(_Instruction(f"'{safe}"))    # ' = INFORM comment
        return self

    # ─── Flow control + variables (INFORM logic) ───
    _RE_VARNAME = re.compile(r"^[BI]\d{1,3}$")

    @classmethod
    def _validate_var(cls, name: str) -> str:
        name = name.strip().upper()
        if not cls._RE_VARNAME.match(name):
            raise ValueError(f"Invalid variable name: '{name}' (expected B###/I###)")
        return name

    @staticmethod
    def _fmt_cond(lhs: str, op: str, rhs: str) -> str:
        if op not in ("=", "<>", ">", "<", ">=", "<="):
            raise ValueError(f"Unsupported condition operator: '{op}'")
        return f"{lhs.strip()}{op}{rhs.strip()}"

    @staticmethod
    def _is_io_cond(cond: tuple[str, str, str]) -> bool:
        """True if the condition tests an I/O signal — Yaskawa requires the
        expression form (IFTHENEXP/ELSEIFEXP/WHILEEXP) for IN#/OT#/ON/OFF, vs the
        plain form for variable/numeric comparisons."""
        toks = (cond[0].strip().upper(), cond[2].strip().upper())
        return any(
            t in ("ON", "OFF") or t.startswith(("IN#", "OT#", "OG#", "IG#"))
            for t in toks)

    @classmethod
    def _cond_keyword(cls, base: str, cond: tuple[str, str, str],
                      force_exp: bool = False) -> str:
        """'IFTHEN'/'ELSEIF'/'WHILE' → append 'EXP' for I/O conditions, or when
        force_exp is set (preserves the original keyword for variable conditions)."""
        return base + ("EXP" if (force_exp or cls._is_io_cond(cond)) else "")

    def label(self, name: str) -> "InformJobBuilder":
        """*LABEL — jump target. Allows numeric labels (*1) and alphanumeric/_."""
        self._validate_label(name)
        self._instructions.append(_Instruction(f"*{name}"))
        return self

    def jump(self, label: str, cond: tuple[str, str, str] | None = None,
             terms: list | None = None, join: str = "AND",
             ) -> "InformJobBuilder":
        """JUMP *LABEL [IF <cond>]. cond = (lhs, op, rhs) or None (unconditional);
        terms = compound condition (combined with ANDEXP/OREXP)."""
        self._validate_label(label)
        text = f"JUMP *{label}"
        if terms:
            joiner = f" {join or 'AND'}EXP "
            text += " IF " + joiner.join(self._fmt_cond(*t) for t in terms)
        elif cond is not None:
            text += f" IF {self._fmt_cond(*cond)}"
        self._instructions.append(_Instruction(text))
        return self

    def set_var(self, name: str, op: str, arg: str | int = "",
                ) -> "InformJobBuilder":
        """SET/ADD/SUB/MUL/DIV Bxxx arg | INC/DEC Bxxx."""
        name = self._validate_var(name)
        op = op.upper()
        if op in ("INC", "DEC"):
            self._instructions.append(_Instruction(f"{op} {name}"))
        elif op in ("SET", "ADD", "SUB", "MUL", "DIV"):
            self._instructions.append(_Instruction(f"{op} {name} {arg}"))
        else:
            raise ValueError(f"Unsupported variable assignment operator: '{op}'")
        return self

    @classmethod
    def _cond_line(
        cls, base: str, cond: tuple[str, str, str],
        terms: list | None = None, join: str = "AND", force_exp: bool = False,
    ) -> str:
        """Build a flow-condition line. Compound (terms + AND/OR) → joined with
        ANDEXP/OREXP. The EXP form (I/O conditions, or force_exp) gets a trailing
        space, matching Yaskawa teach-pendant output."""
        if terms:
            io = force_exp or any(cls._is_io_cond(t) for t in terms)
            kw = base + ("EXP" if io else "")
            joiner = f" {join or 'AND'}EXP "
            body = joiner.join(cls._fmt_cond(*t) for t in terms)
            return f"{kw} {body}" + (" " if kw.endswith("EXP") else "")
        kw = cls._cond_keyword(base, cond, force_exp)
        line = f"{kw} {cls._fmt_cond(*cond)}"
        return line + " " if kw.endswith("EXP") else line

    def if_then(self, cond: tuple[str, str, str], terms: list | None = None,
                join: str = "AND", exp: bool = False) -> "InformJobBuilder":
        self._instructions.append(
            _Instruction(self._cond_line("IFTHEN", cond, terms, join, exp)))
        return self

    def else_if(self, cond: tuple[str, str, str], terms: list | None = None,
                join: str = "AND", exp: bool = False) -> "InformJobBuilder":
        self._instructions.append(
            _Instruction(self._cond_line("ELSEIF", cond, terms, join, exp)))
        return self

    def else_(self) -> "InformJobBuilder":
        self._instructions.append(_Instruction("ELSE"))
        return self

    def end_if(self) -> "InformJobBuilder":
        self._instructions.append(_Instruction("ENDIF"))
        return self

    def while_(self, cond: tuple[str, str, str], terms: list | None = None,
               join: str = "AND", exp: bool = False) -> "InformJobBuilder":
        self._instructions.append(
            _Instruction(self._cond_line("WHILE", cond, terms, join, exp)))
        return self

    def end_while(self) -> "InformJobBuilder":
        self._instructions.append(_Instruction("ENDWHILE"))
        return self

    # ─── Extended LG instructions ───
    def set_express(self, name: str, expr: str) -> "InformJobBuilder":
        """SET Ixxx EXPRESS <expr> — assign an arithmetic expression."""
        name = self._validate_var(name)
        self._instructions.append(_Instruction(f"SET {name} EXPRESS {expr}"))
        return self

    def pulse(self, output_index: int) -> "InformJobBuilder":
        """PULSE OT#(n) — momentary output pulse."""
        self._instructions.append(_Instruction(f"PULSE OT#({int(output_index)})"))
        return self

    def clear_stack(self) -> "InformJobBuilder":
        self._instructions.append(_Instruction("CLEAR STACK"))
        return self

    def clear_var(self, name: str, count: int) -> "InformJobBuilder":
        """CLEAR Ixxx n | CLEAR Ixxx ALL (count < 0 → ALL)."""
        name = self._validate_var(name)
        n = "ALL" if count < 0 else int(count)
        self._instructions.append(_Instruction(f"CLEAR {name} {n}"))
        return self

    def din(self, name: str, group_kind: str, group_index: int,
            ) -> "InformJobBuilder":
        """DIN Bxxx IG#(n) | DIN Bxxx SOUT#(n) — read input/status group → var."""
        name = self._validate_var(name)
        self._instructions.append(
            _Instruction(f"DIN {name} {group_kind.upper()}#({int(group_index)})"))
        return self

    def dout_group(self, group_index: int, name: str) -> "InformJobBuilder":
        """DOUT OG#(n) Bxxx — write a variable to an output group."""
        name = self._validate_var(name)
        self._instructions.append(
            _Instruction(f"DOUT OG#({int(group_index)}) {name}"))
        return self

    def movj_indirect(self, index_var: str, speed_var: str = "",
                      speed_pct: float | None = None) -> "InformJobBuilder":
        """MOVJ P[Bxxx] [VJ=Ixxx | VJ=n] — indirect position, optional var speed."""
        index_var = self._validate_var(index_var)
        if speed_var:
            vj = f"VJ={self._validate_var(speed_var)}"
        else:
            vj = f"VJ={self._clamp_joint_speed(speed_pct):.2f}"
        self._instructions.append(_Instruction(f"MOVJ P[{index_var}] {vj}"))
        return self

    def movl_indirect(self, index_var: str, speed_var: str = "",
                      speed_mm_s: float = 100.0) -> "InformJobBuilder":
        """MOVL P[Bxxx] [V=Ixxx | V=n] — indirect position, optional var speed."""
        index_var = self._validate_var(index_var)
        if speed_var:
            v = f"V={self._validate_var(speed_var)}"
        else:
            v = f"V={max(1.0, min(float(speed_mm_s), MAX_LINEAR_MM_S)):.1f}"
        self._instructions.append(_Instruction(f"MOVL P[{index_var}] {v}"))
        return self

    def _clamp_joint_speed(self, speed_pct: float | None) -> float:
        if speed_pct is None:
            return self.max_speed_pct
        return max(0.01, min(float(speed_pct), self.max_speed_pct))

    # ─── Render ───
    def render(self, date_str: str = "2026/01/01 00:00") -> str:
        """Generate full INFORM .JBI text. CRLF line endings (Yaskawa convention).

        Logic-only jobs (no MOVJ/MOVL — e.g. a master/speed-calc job) are allowed:
        they render an empty //POS section (NPOS 0,0,0,0,0,0)."""
        if not self._instructions:
            raise ValueError("Job must have at least 1 instruction")

        lines: list[str] = []
        lines.append("/JOB")
        lines.append(f"//NAME {self.name}")
        if self.folder_name:
            lines.append(f"///FOLDERNAME {self.folder_name}")
        lines.append("//POS")
        # NPOS counts positions per slot. Slot 0 = C-variables; slot 3 holds the
        # P-variable count (matches real Yaskawa teach-pendant jobs, e.g.
        # '///NPOS 0,0,0,5,0,0' for 5 P-vars). Other slots (BP/EX/ST/EXP) = 0 for
        # a single 6-axis robot.
        n_cvar = sum(1 for p in self._positions if p.letter == "C")
        n_pvar = sum(1 for p in self._positions if p.letter == "P")
        lines.append(f"///NPOS {n_cvar},0,0,{n_pvar},0,0")
        if self._positions:
            lines.append(f"///TOOL {self._positions[0].tool_no}")
            lines.append(f"///POSTYPE {self.pos_type}")
            # First section (PULSE/BASE/...) matches pos_type.
            lines.append(f"///{self.pos_type}")
            for p in self._positions:
                token = self._pos_decl_token(p)
                # Pad with 0 for axes 7-8 if emit_axis_count > joints (GP7: 6→8).
                padded = list(p.joints_pulse) + [0] * (
                    self.emit_axis_count - len(p.joints_pulse))
                values = ",".join(str(v) for v in padded)
                lines.append(f"{token}={values}")

        lines.append("//INST")
        lines.append(f"///DATE {date_str}")
        lines.append("///ATTR SC,RW")
        lines.append(f"///GROUP1 {self.group}")
        lines.append("NOP")
        # Indent IF/WHILE block bodies with '\t ' per nesting level (matches
        # Yaskawa teach-pendant formatting). Opener/closer/mid-keywords sit at
        # the block's own level; the body between them is indented one deeper.
        depth = 0
        for inst in self._instructions:
            head = inst.text.split(None, 1)[0].upper() if inst.text else ""
            opens = head in ("IFTHEN", "IFTHENEXP", "WHILE", "WHILEEXP")
            closes = head in ("ENDIF", "ENDWHILE")
            mid = head in ("ELSE", "ELSEIF", "ELSEIFEXP")
            level = depth - 1 if (closes or mid) else depth
            lines.append("\t " * max(0, level) + inst.text)
            if opens:
                depth += 1
            elif closes:
                depth = max(0, depth - 1)
        lines.append("END")

        return "\r\n".join(lines) + "\r\n"


# ───── Ultra-fast P-variable template (M3++) ─────


def gen_pvar_template_job(
    name: str,
    num_positions: int,
    motion_kinds: list[str] | None = None,
    gripper_do_index: int = 1,
    gripper_close_at: int | None = None,
    gripper_open_at: int | None = None,
    gripper_delay_s: float = 0.3,
    max_speed_pct: float = MAX_SPEED_PCT_DEFAULT,
) -> str:
    """Generate an INFORM template using runtime-mutable P-variables.

    Template is uploaded once; subsequent trials only need WRITE_POS_VAR + START.
    Very fast (~50ms/trial overhead) because there is no FTP roundtrip.

    P-variables are NOT declared in the //POS section — they are runtime vars
    set via HSE WRITE_POS_VAR. The job references P000, P001, ... in //INST.

    Args:
        name: Job name (≤ 32 ASCII).
        num_positions: Number of P-variables the job moves to (= number of waypoints).
        motion_kinds: List of "movj"/"movl" per position. Default all movj.
        gripper_close_at: Index at which to insert DOUT ON before that MOVJ. None = skip.
        gripper_open_at: Index at which to insert DOUT OFF before that MOVJ.
        gripper_delay_s: TIMER seconds after each DOUT (gripper settle).
        max_speed_pct: Speed cap for VJ.

    Returns:
        INFORM .JBI text using P-variables.
    """
    if num_positions < 1 or num_positions > 128:
        raise ValueError(f"num_positions 1-128, got {num_positions}")
    if motion_kinds is None:
        motion_kinds = ["movj"] * num_positions
    if len(motion_kinds) != num_positions:
        raise ValueError(
            f"motion_kinds must have {num_positions} entries, got {len(motion_kinds)}"
        )

    vj = float(max_speed_pct)
    lines: list[str] = []
    lines.append("/JOB")
    lines.append(f"//NAME {name}")
    lines.append("//POS")
    lines.append("///NPOS 0,0,0,0,0,0")
    lines.append("///TOOL 0")
    lines.append("///POSTYPE PULSE")
    lines.append("///PULSE")
    lines.append("//INST")
    lines.append("///DATE 2026/01/01 00:00")
    lines.append("///ATTR SC,RW")
    lines.append("///GROUP1 RB1")
    lines.append("NOP")
    lines.append("'P-var template - runtime set via HSE WRITE_POS_VAR")

    for i in range(num_positions):
        if gripper_close_at is not None and i == gripper_close_at:
            lines.append(f"DOUT OT#({gripper_do_index}) ON")
            lines.append(f"TIMER T={gripper_delay_s:.2f}")
        if gripper_open_at is not None and i == gripper_open_at:
            lines.append(f"DOUT OT#({gripper_do_index}) OFF")
            lines.append(f"TIMER T={gripper_delay_s:.2f}")

        kind = motion_kinds[i].upper()
        if kind == "MOVJ":
            lines.append(f"MOVJ P{i:03d} VJ={vj:.2f}")
        elif kind == "MOVL":
            lines.append(f"MOVL P{i:03d} V=80.0")
        else:
            raise ValueError(f"motion_kinds[{i}]='{motion_kinds[i]}' not supported")

    lines.append("END")
    return "\r\n".join(lines) + "\r\n"


# ───── High-level convenience ─────


def gen_pick_place_job(
    name: str,
    home_deg: list[float],
    approach_deg: list[float],
    grasp_deg: list[float],
    transfer_deg: list[float],
    place_deg: list[float],
    gripper_do_index: int = 1,
    gripper_delay_s: float = 0.3,
    speed_pct: float = 10.0,
    max_speed_pct: float = MAX_SPEED_PCT_DEFAULT,
) -> str:
    """Generate an INFORM .JBI for a complete pick-and-place cycle.

    Workflow: home → approach → grasp (close + lift) → transfer → place
    (lower + open) → home. Speed is always ≤ max_speed_pct.
    """
    builder = (
        InformJobBuilder(name=name, max_speed_pct=max_speed_pct)
        .add_position("home", home_deg)
        .add_position("approach", approach_deg)
        .add_position("grasp", grasp_deg)
        .add_position("transfer", transfer_deg)
        .add_position("place", place_deg)
        .comment("Pick-and-place auto-generated by Orchestrator")
        .movj("home", speed_pct=speed_pct)
        .movj("approach", speed_pct=speed_pct)
        .movj("grasp", speed_pct=speed_pct)
        .dout(gripper_do_index, True)             # close gripper
        .timer(gripper_delay_s)
        .movj("approach", speed_pct=speed_pct)    # lift up
        .movj("transfer", speed_pct=speed_pct)
        .movj("place", speed_pct=speed_pct)
        .dout(gripper_do_index, False)            # open gripper
        .timer(gripper_delay_s)
        .movj("transfer", speed_pct=speed_pct)    # lift up
        .movj("home", speed_pct=speed_pct)
    )
    return builder.render()
