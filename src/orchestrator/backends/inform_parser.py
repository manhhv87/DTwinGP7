"""
inform_parser.py
────────────────
Parser INFORM (.JBI) → intermediate structure (ParsedJob) for GP7 / YRC1000.

This is the REVERSE of `inform_codegen.py`: reads a .JBI text file → extracts
positions (C-variables, pulse → degrees) + instruction list (MOVJ/MOVL/MOVC/DOUT/
TIMER/WAIT/MSG/CALL). The app layer (mixin_program_io) converts ParsedJob →
list[Instruction] for the editor (MOVL is FK'd back to Cartesian pose at that
layer because it needs model/tool frame — this parser stays PURE TEXT, no
numpy/IO/kinematics).

Round-trip: parse_jbi(builder.render()) faithfully reconstructs joints (pulses are
ints so reverse division has no drift) + motion kind + modifiers (VJ/V/PL/TL/UF).

Scope (matches the subset codegen supports):
  MOVJ, MOVL, MOVC, DOUT OT#(), TIMER T=, WAIT IN#()=ON/OFF [T=], MSG "...",
  CALL JOB:. Instructions outside the subset (JUMP/IF/SET/variables...) → logged
  to `warnings` and skipped (no raise) so the rest of the file still loads.

P-variable (P000...) is NOT declared in //POS (set at runtime via HSE) → no joint
data in the file ⇒ motions referencing P### are warned and skipped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .hse_protocol import GP7_PULSE_PER_DEG


@dataclass
class ParsedInstr:
    """One parsed instruction. `kind` determines which fields are meaningful.

    kind ∈ {movj, movl, movc, dout, timer, wait_in, msg, call}.
    """

    kind: str
    # Motion (movj/movl/movc)
    joints_deg: list[float] | None = None
    vj_pct: float | None = None
    v_mm_s: float | None = None
    pl: int | None = None
    tool_no: int | None = None
    user_frame: int | None = None
    # dout
    do_index: int = 1
    do_on: bool = True
    # timer
    seconds: float = 0.0
    # wait_in
    in_index: int = 1
    in_on: bool = True
    in_timeout_s: float = 0.0
    # msg / call
    text: str = ""
    # flow / variables (kind ∈ label/jump/setvar/ifthen/elseif/else/endif/
    # while/endwhile)
    label_name: str = ""
    var_name: str = ""
    var_op: str = ""
    var_arg: str = ""
    cond_lhs: str = ""
    cond_op: str = ""
    cond_rhs: str = ""


@dataclass
class ParsedJob:
    name: str
    instructions: list[ParsedInstr] = field(default_factory=list)
    pos_type: str = "PULSE"
    warnings: list[str] = field(default_factory=list)


# ───── Regex grammar (matches inform_codegen output) ─────
# All tokens uppercase, space-separated. Tolerant: allows extra spaces, optional
# modifiers in any order (PL/TL/UF). Position token = C##### or P###.

_RE_NAME = re.compile(r"^//NAME\s+(\S.*?)\s*$")
_RE_POSTYPE = re.compile(r"^///POSTYPE\s+(\S+)")
_RE_CVAR = re.compile(r"^(C\d{1,5})\s*=\s*(.+)$")
_RE_POS_TOKEN = re.compile(r"^(C\d{1,5}|P\d{1,3})$")

_RE_VJ = re.compile(r"\bVJ=([-\d.]+)")
_RE_V = re.compile(r"\bV=([-\d.]+)")
_RE_PL = re.compile(r"\bPL=(\d+)")
_RE_TL = re.compile(r"\bTL=(\d+)")
_RE_UF = re.compile(r"\bUF#\((\d+)\)")

_RE_DOUT = re.compile(r"^DOUT\s+OT#\((\d+)\)\s+(ON|OFF)\b", re.IGNORECASE)
_RE_TIMER = re.compile(r"^TIMER\s+T=([-\d.]+)", re.IGNORECASE)
_RE_WAIT = re.compile(
    r"^WAIT\s+IN#\((\d+)\)\s*=\s*(ON|OFF)\b(?:\s+T=([-\d.]+))?", re.IGNORECASE)
_RE_MSG = re.compile(r'^MSG\s+"(.*)"\s*$', re.IGNORECASE)
_RE_CALL = re.compile(r"^CALL\s+JOB:(\S+)", re.IGNORECASE)

# Flow control + variables
_RE_LABEL = re.compile(r"^\*(\w+)$")
_RE_JUMP = re.compile(r"^JUMP\s+\*(\w+)(?:\s+IF\s+(.+))?$", re.IGNORECASE)
_RE_SETVAR = re.compile(
    r"^(SET|ADD|SUB|MUL|DIV)\s+([BI]\d{1,3})\s+(\S+)$", re.IGNORECASE)
_RE_INCDEC = re.compile(r"^(INC|DEC)\s+([BI]\d{1,3})$", re.IGNORECASE)
_RE_IFTHEN = re.compile(r"^IFTHEN\s+(.+)$", re.IGNORECASE)
_RE_ELSEIF = re.compile(r"^ELSEIF\s+(.+)$", re.IGNORECASE)
_RE_WHILE = re.compile(r"^WHILE\s+(.+)$", re.IGNORECASE)
# Condition: lhs op rhs (long ops '<>','>=','<=' take priority over '=','>','<').
_RE_COND = re.compile(r"^\s*(\S+?)\s*(<>|>=|<=|=|>|<)\s*(\S+?)\s*$")


def _parse_cond(expr: str, warnings: list[str]) -> tuple[str, str, str] | None:
    m = _RE_COND.match(expr.strip())
    if not m:
        warnings.append(f"Could not parse condition: {expr!r}")
        return None
    return (m.group(1), m.group(2), m.group(3))


def _pulses_to_deg(
    pulses: list[int], pulse_per_deg: tuple[float, ...],
) -> list[float]:
    """Reverse-divide pulses → degrees. Takes only the first len(pulse_per_deg)
    axes (discards pad axes 7-8 = 0 if the file emits 8 values)."""
    n = len(pulse_per_deg)
    return [pulses[i] / pulse_per_deg[i] for i in range(min(n, len(pulses)))]


def parse_jbi(
    text: str,
    pulse_per_deg: tuple[float, ...] = GP7_PULSE_PER_DEG,
) -> ParsedJob:
    """Parse INFORM .JBI text → ParsedJob.

    Args:
        text: Content of the .JBI file (CRLF or LF both accepted).
        pulse_per_deg: Pulse/deg ratio per axis (default GP7) used to convert
            C-var pulses → degrees.

    Returns:
        ParsedJob(name, instructions, pos_type, warnings). Does not raise on
        unknown instructions — collects them in warnings.

    Raises:
        ValueError: if not a valid INFORM file (missing /JOB) or missing //INST.
    """
    # Normalise line endings + strip trailing whitespace from each line.
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [ln.rstrip() for ln in raw_lines]

    if not any(ln.strip() == "/JOB" for ln in lines):
        raise ValueError("Not a valid INFORM .JBI file (missing '/JOB' header)")

    name = ""
    pos_type = "PULSE"
    positions: dict[str, list[float]] = {}     # token (C#####) → joints_deg
    warnings: list[str] = []

    # ── Pass 1: header + //POS section ──
    in_pos = False
    in_inst = False
    inst_lines: list[str] = []
    for ln in lines:
        s = ln.strip()
        if s == "//POS":
            in_pos, in_inst = True, False
            continue
        if s == "//INST":
            in_pos, in_inst = False, True
            continue
        m = _RE_NAME.match(s)
        if m and not name:
            name = m.group(1)[:32]
            continue
        if in_pos:
            mp = _RE_POSTYPE.match(s)
            if mp:
                pos_type = mp.group(1).upper()
                continue
            mc = _RE_CVAR.match(s)
            if mc:
                token = mc.group(1)
                try:
                    vals = [int(round(float(x)))
                            for x in mc.group(2).split(",") if x.strip() != ""]
                except ValueError:
                    warnings.append(f"Could not parse C-var: {s!r}")
                    continue
                positions[token] = _pulses_to_deg(vals, pulse_per_deg)
            continue
        if in_inst:
            inst_lines.append(s)

    if not in_inst and not inst_lines:
        raise ValueError("INFORM file is missing the //INST section")

    # ── Pass 2: //INST body (between NOP and END) ──
    instructions: list[ParsedInstr] = []
    started = False
    for s in inst_lines:
        if not s:
            continue
        if s.startswith("///"):                 # ///DATE ///ATTR ///GROUP1 …
            continue
        if s == "NOP":
            started = True
            continue
        if s == "END":
            break
        if not started:
            # Before NOP (no instruction expected here) — skip safely.
            continue
        if s.startswith("'"):                    # comment INFORM → non-exec, drop
            continue

        instr = _parse_inst_line(s, positions, pulse_per_deg, warnings)
        if instr is not None:
            instructions.append(instr)

    return ParsedJob(
        name=name or "IMPORTED",
        instructions=instructions,
        pos_type=pos_type,
        warnings=warnings,
    )


def _parse_motion_modifiers(s: str, instr: ParsedInstr) -> None:
    """Extract VJ/V/PL/TL/UF from the tail of a motion line into instr."""
    m = _RE_VJ.search(s)
    if m:
        instr.vj_pct = float(m.group(1))
    m = _RE_V.search(s)
    if m:
        instr.v_mm_s = float(m.group(1))
    m = _RE_PL.search(s)
    if m:
        instr.pl = int(m.group(1))
    m = _RE_TL.search(s)
    if m:
        instr.tool_no = int(m.group(1))
    m = _RE_UF.search(s)
    if m:
        instr.user_frame = int(m.group(1))


def _parse_inst_line(
    s: str,
    positions: dict[str, list[float]],
    pulse_per_deg: tuple[float, ...],
    warnings: list[str],
) -> ParsedInstr | None:
    """Parse one //INST line → ParsedInstr (None if skipped)."""
    head = s.split(None, 1)[0].upper()

    if head in ("MOVJ", "MOVL", "MOVC"):
        parts = s.split()
        if len(parts) < 2:
            warnings.append(f"Motion instruction missing position token: {s!r}")
            return None
        token = parts[1]
        if not _RE_POS_TOKEN.match(token):
            warnings.append(f"Unrecognised position token: {token!r} in {s!r}")
            return None
        if token not in positions:
            warnings.append(
                f"{head} references '{token}' not found in //POS "
                f"(P-var runtime?) — skipping line")
            return None
        instr = ParsedInstr(kind=head.lower(),
                            joints_deg=list(positions[token]))
        _parse_motion_modifiers(s, instr)
        return instr

    m = _RE_DOUT.match(s)
    if m:
        return ParsedInstr(kind="dout", do_index=int(m.group(1)),
                           do_on=m.group(2).upper() == "ON")
    m = _RE_TIMER.match(s)
    if m:
        return ParsedInstr(kind="timer", seconds=float(m.group(1)))
    m = _RE_WAIT.match(s)
    if m:
        return ParsedInstr(
            kind="wait_in", in_index=int(m.group(1)),
            in_on=m.group(2).upper() == "ON",
            in_timeout_s=float(m.group(3)) if m.group(3) else 0.0)
    m = _RE_MSG.match(s)
    if m:
        return ParsedInstr(kind="msg", text=m.group(1)[:32])
    m = _RE_CALL.match(s)
    if m:
        return ParsedInstr(kind="call", text=m.group(1)[:32])

    # ── Flow control + variables ──
    m = _RE_LABEL.match(s)
    if m:
        return ParsedInstr(kind="label", label_name=m.group(1))
    m = _RE_JUMP.match(s)
    if m:
        instr = ParsedInstr(kind="jump", label_name=m.group(1))
        if m.group(2):
            cond = _parse_cond(m.group(2), warnings)
            if cond:
                instr.cond_lhs, instr.cond_op, instr.cond_rhs = cond
        return instr
    m = _RE_SETVAR.match(s)
    if m:
        return ParsedInstr(kind="setvar", var_name=m.group(2).upper(),
                           var_op=m.group(1).upper(), var_arg=m.group(3))
    m = _RE_INCDEC.match(s)
    if m:
        return ParsedInstr(kind="setvar", var_name=m.group(2).upper(),
                           var_op=m.group(1).upper())
    m = _RE_IFTHEN.match(s)
    if m:
        cond = _parse_cond(m.group(1), warnings)
        if not cond:
            return None
        return ParsedInstr(kind="ifthen", cond_lhs=cond[0],
                           cond_op=cond[1], cond_rhs=cond[2])
    m = _RE_ELSEIF.match(s)
    if m:
        cond = _parse_cond(m.group(1), warnings)
        if not cond:
            return None
        return ParsedInstr(kind="elseif", cond_lhs=cond[0],
                           cond_op=cond[1], cond_rhs=cond[2])
    m = _RE_WHILE.match(s)
    if m:
        cond = _parse_cond(m.group(1), warnings)
        if not cond:
            return None
        return ParsedInstr(kind="while", cond_lhs=cond[0],
                           cond_op=cond[1], cond_rhs=cond[2])
    if head == "ELSE":
        return ParsedInstr(kind="else")
    if head == "ENDIF":
        return ParsedInstr(kind="endif")
    if head == "ENDWHILE":
        return ParsedInstr(kind="endwhile")

    warnings.append(f"Instruction outside supported subset — skipping: {s!r}")
    return None
