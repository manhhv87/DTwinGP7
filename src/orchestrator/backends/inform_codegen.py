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
    """C-variable position constant."""

    name: str
    joints_pulse: list[int]
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
    """

    def __init__(
        self,
        name: str,
        pos_type: PosType = "PULSE",
        pulse_per_deg: tuple[float, ...] = GP7_PULSE_PER_DEG,
        max_speed_pct: float = MAX_SPEED_PCT_DEFAULT,
        group: str = "RB1",
        emit_axis_count: int = 0,
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

    # ─── Positions ───
    def add_position(self, name: str, joints_deg: list[float]) -> "InformJobBuilder":
        """Add a C-variable position from joint angles (degrees)."""
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
        pulses = [int(round(d * r)) for d, r in zip(joints_deg, self.pulse_per_deg)]
        self._pos_index[name] = len(self._positions)
        self._positions.append(_Position(name=name, joints_pulse=pulses))
        return self

    def _resolve_cvar(self, position_name: str) -> str:
        """Map logical position name → C-variable token (e.g. 'p0' → 'C00000')."""
        if position_name not in self._pos_index:
            raise KeyError(f"Position '{position_name}' not added — call add_position first")
        return f"C{self._pos_index[position_name]:05d}"

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
             user_frame: int | None = None) -> "InformJobBuilder":
        """Joint move to position (high speed, path not controlled).

        Args:
            tool_no: TOOL coordinate (TL=). None = do not emit modifier.
            pl: Position level / rounding 0..8 (PL=). None = do not emit.
            user_frame: User frame index (UF#). None = do not emit.
        """
        cvar = self._resolve_cvar(position_name)
        vj = self._clamp_joint_speed(speed_pct)
        inst = f"MOVJ {cvar} VJ={vj:.2f}"
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
    ) -> "InformJobBuilder":
        """Linear (Cartesian) move."""
        cvar = self._resolve_cvar(position_name)
        v = max(1.0, min(float(speed_mm_s), MAX_LINEAR_MM_S))
        inst = f"MOVL {cvar} V={v:.1f}"
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
        self._instructions.append(_Instruction(f"TIMER T={seconds:.2f}"))
        return self

    def msg(self, text: str) -> "InformJobBuilder":
        """MSG "string" — display message on the teach pendant (≤ 32 ASCII)."""
        # INFORM MSG supports ≤ 32 characters, ASCII printable, no embedded quotes.
        clean = "".join(c for c in text if 0x20 <= ord(c) < 0x7F and c != '"')
        clean = clean[:32]
        self._instructions.append(_Instruction(f'MSG "{clean}"'))
        return self

    def call_job(self, job_name: str) -> "InformJobBuilder":
        """CALL JOB:job_name — invoke sub-program. job_name validated same as
        _validate_name (≤32 ASCII alphanumeric/_)."""
        self._validate_name(job_name)
        self._instructions.append(_Instruction(f"CALL JOB:{job_name}"))
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

    def label(self, name: str) -> "InformJobBuilder":
        """*LABEL — jump target. Name follows job convention (start letter, ≤32 alnum/_)."""
        self._validate_name(name)
        self._instructions.append(_Instruction(f"*{name}"))
        return self

    def jump(self, label: str, cond: tuple[str, str, str] | None = None,
             ) -> "InformJobBuilder":
        """JUMP *LABEL [IF <cond>]. cond = (lhs, op, rhs) or None (unconditional)."""
        self._validate_name(label)
        text = f"JUMP *{label}"
        if cond is not None:
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

    def if_then(self, cond: tuple[str, str, str]) -> "InformJobBuilder":
        self._instructions.append(_Instruction(f"IFTHEN {self._fmt_cond(*cond)}"))
        return self

    def else_if(self, cond: tuple[str, str, str]) -> "InformJobBuilder":
        self._instructions.append(_Instruction(f"ELSEIF {self._fmt_cond(*cond)}"))
        return self

    def else_(self) -> "InformJobBuilder":
        self._instructions.append(_Instruction("ELSE"))
        return self

    def end_if(self) -> "InformJobBuilder":
        self._instructions.append(_Instruction("ENDIF"))
        return self

    def while_(self, cond: tuple[str, str, str]) -> "InformJobBuilder":
        self._instructions.append(_Instruction(f"WHILE {self._fmt_cond(*cond)}"))
        return self

    def end_while(self) -> "InformJobBuilder":
        self._instructions.append(_Instruction("ENDWHILE"))
        return self

    def _clamp_joint_speed(self, speed_pct: float | None) -> float:
        if speed_pct is None:
            return self.max_speed_pct
        return max(0.01, min(float(speed_pct), self.max_speed_pct))

    # ─── Render ───
    def render(self, date_str: str = "2026/01/01 00:00") -> str:
        """Generate full INFORM .JBI text. CRLF line endings (Yaskawa convention)."""
        if not self._positions:
            raise ValueError("Job must have at least 1 position")
        if not self._instructions:
            raise ValueError("Job must have at least 1 instruction")

        lines: list[str] = []
        lines.append("/JOB")
        lines.append(f"//NAME {self.name}")
        lines.append("//POS")
        # NPOS = number of positions per group (BP/EX/ST/EXP = 0 for single 6-axis robot).
        lines.append(f"///NPOS {len(self._positions)},0,0,0,0,0")
        lines.append(f"///TOOL {self._positions[0].tool_no}")
        lines.append(f"///POSTYPE {self.pos_type}")
        # First section (PULSE/BASE/...) matches pos_type.
        lines.append(f"///{self.pos_type}")
        for p in self._positions:
            cvar = f"C{self._pos_index[p.name]:05d}"
            # Pad with 0 for axes 7-8 if emit_axis_count > number of joints (GP7: 6→8).
            padded = list(p.joints_pulse) + [0] * (
                self.emit_axis_count - len(p.joints_pulse))
            values = ",".join(str(v) for v in padded)
            lines.append(f"{cvar}={values}")

        lines.append("//INST")
        lines.append(f"///DATE {date_str}")
        lines.append("///ATTR SC,RW")
        lines.append(f"///GROUP1 {self.group}")
        lines.append("NOP")
        for inst in self._instructions:
            lines.append(inst.text)
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
