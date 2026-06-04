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
    # Canonical position key (e.g. 'C0', 'P1') of the //POS var this motion
    # references — same key for motions reusing the same taught point, so the
    # importer can map them to one shared target.
    pos_key: str = ""
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
    # Compound condition (ANDEXP/OREXP): list of (lhs,op,rhs) + join "AND"/"OR".
    cond_terms: list = field(default_factory=list)
    cond_join: str = ""
    cond_exp: bool = False          # original used IFTHENEXP/ELSEIFEXP/WHILEEXP
    # Extended (LG): CLEAR / group I/O / EXPRESS / variable speed / P[Bxxx].
    clear_count: int = 0
    io_group: int = 0
    io_group_kind: str = ""
    var_expr: str = ""
    speed_var: str = ""             # V=Ixxx / VJ=Ixxx variable speed
    pos_index_var: str = ""         # P[Bxxx] indirect addressing


@dataclass
class ParsedJob:
    name: str
    instructions: list[ParsedInstr] = field(default_factory=list)
    pos_type: str = "PULSE"
    folder_name: str = ""          # ///FOLDERNAME (controller folder; "" if absent)
    warnings: list[str] = field(default_factory=list)
    # Full //POS table: canonical key ('P10','C0',…) → joints_deg. Lets the
    # importer resolve P[Bxxx] indirect motions (index known only at runtime).
    positions: dict = field(default_factory=dict)


# ───── Regex grammar (matches inform_codegen output) ─────
# All tokens uppercase, space-separated. Tolerant: allows extra spaces, optional
# modifiers in any order (PL/TL/UF). Position token = C##### or P###.

_RE_NAME = re.compile(r"^//NAME\s+(\S.*?)\s*$")
_RE_FOLDER = re.compile(r"^///FOLDERNAME\s+(\S.*?)\s*$")
_RE_POSTYPE = re.compile(r"^///POSTYPE\s+(\S+)")
# //POS declaration line: C-var OR P-var (real teach-pendant jobs declare P-vars
# in //POS too, e.g. "P00001=..."). Captures letter, index, values.
_RE_POSVAR = re.compile(r"^([CP])(\d{1,5})\s*=\s*(.+)$", re.IGNORECASE)
# Motion position token in //INST (e.g. "C00000", "P001"). Captures letter + index.
_RE_POS_TOKEN = re.compile(r"^([CP])(\d{1,5})$", re.IGNORECASE)
# Indirect position token: P[Bxxx] — index taken from a variable at runtime.
_RE_POS_INDIRECT = re.compile(r"^P\[([BI]\d{1,3})\]$", re.IGNORECASE)


def _norm_pos_key(letter: str, index: int) -> str:
    """Canonical position key independent of zero-padding so the //INST token
    ('P001') matches its //POS declaration ('P00001'). Both → 'P1'."""
    return f"{letter.upper()}{index}"

# Speeds accept a numeric value OR a variable (V=I003, VJ=I000).
_RE_VJ = re.compile(r"\bVJ=([-\d.]+|[BI]\d{1,3})", re.IGNORECASE)
_RE_V = re.compile(r"\bV=([-\d.]+|[BI]\d{1,3})", re.IGNORECASE)
_RE_PL = re.compile(r"\bPL=(\d+)")
_RE_TL = re.compile(r"\bTL=(\d+)")
_RE_UF = re.compile(r"\bUF#\((\d+)\)")

_RE_DOUT = re.compile(r"^DOUT\s+OT#\((\d+)\)\s+(ON|OFF)\b", re.IGNORECASE)
_RE_TIMER = re.compile(r"^TIMER\s+T=([-\d.]+)", re.IGNORECASE)
_RE_WAIT = re.compile(
    r"^WAIT\s+IN#\((\d+)\)\s*=\s*(ON|OFF)\b(?:\s+T=([-\d.]+))?", re.IGNORECASE)
_RE_MSG = re.compile(r'^MSG\s+"(.*)"\s*$', re.IGNORECASE)
_RE_CALL = re.compile(r"^CALL\s+JOB:(\S+)", re.IGNORECASE)
# Extended LG instructions.
_RE_PULSE = re.compile(r"^PULSE\s+OT#\((\d+)\)", re.IGNORECASE)
# CLEAR STACK | CLEAR Ixxx ALL | CLEAR Ixxx <n>
_RE_CLEAR_STACK = re.compile(r"^CLEAR\s+STACK\b", re.IGNORECASE)
_RE_CLEAR_VAR = re.compile(
    r"^CLEAR\s+([BI]\d{1,3})\s+(ALL|\d+)\b", re.IGNORECASE)
# DIN Bxxx IG#(n) | DIN Bxxx SOUT#(n)  — read input/status group into a variable.
_RE_DIN = re.compile(
    r"^DIN\s+([BI]\d{1,3})\s+(IG|SOUT)#\((\d+)\)", re.IGNORECASE)
# DOUT OG#(n) Bxxx — write a variable to an output group.
_RE_DOUT_OG = re.compile(
    r"^DOUT\s+OG#\((\d+)\)\s+([BI]\d{1,3})", re.IGNORECASE)
# SET Ixxx EXPRESS <expr>  — assign an arithmetic expression.
_RE_SET_EXPRESS = re.compile(
    r"^SET\s+([BI]\d{1,3})\s+EXPRESS\s+(.+)$", re.IGNORECASE)

# Flow control + variables
_RE_LABEL = re.compile(r"^\*(\w+)$")
_RE_JUMP = re.compile(r"^JUMP\s+\*(\w+)(?:\s+IF\s+(.+))?$", re.IGNORECASE)
_RE_SETVAR = re.compile(
    r"^(SET|ADD|SUB|MUL|DIV)\s+([BI]\d{1,3})\s+(\S+)$", re.IGNORECASE)
_RE_INCDEC = re.compile(r"^(INC|DEC)\s+([BI]\d{1,3})$", re.IGNORECASE)
# IFTHEN / IFTHENEXP (expression form) — both open an IF block. Same for ELSEIF.
# Group 1 captures 'EXP' (or empty) so we can preserve the original keyword form.
_RE_IFTHEN = re.compile(r"^IFTHEN(EXP)?\s+(.+)$", re.IGNORECASE)
_RE_ELSEIF = re.compile(r"^ELSEIF(EXP)?\s+(.+)$", re.IGNORECASE)
_RE_WHILE = re.compile(r"^WHILE(EXP)?\s+(.+)$", re.IGNORECASE)
# Condition: lhs op rhs (long ops '<>','>=','<=' take priority over '=','>','<').
_RE_COND = re.compile(r"^\s*(\S+?)\s*(<>|>=|<=|=|>|<)\s*(\S+?)\s*$")


_RE_ANDOR = re.compile(r"\s+(ANDEXP|OREXP|AND|OR)\s+", re.IGNORECASE)


def _parse_cond(expr: str, warnings: list[str]) -> tuple[str, str, str] | None:
    m = _RE_COND.match(expr.strip())
    if not m:
        warnings.append(f"Could not parse condition: {expr!r}")
        return None
    # Keep operands verbatim (incl. ON/OFF) — eval_condition handles ON/OFF, and
    # export reproduces the original IFTHENEXP IN#(n)=ON form (no lossy 1/0).
    return (m.group(1).strip(), m.group(2), m.group(3).strip())


def _set_cond(instr: "ParsedInstr", expr: str, warnings: list[str]) -> bool:
    """Parse a (possibly compound) condition into `instr`. Supports
    'a<>1 ANDEXP b<>2' / OREXP. Returns False (and warns) if unparseable.

    Single term → cond_lhs/op/rhs. Multiple terms → cond_terms + cond_join, and
    the first term is also mirrored into cond_lhs/op/rhs for back-compat."""
    parts = _RE_ANDOR.split(expr.strip())
    # parts = [term, joiner, term, joiner, ...]; joiners at odd indices.
    terms_str = parts[0::2]
    joiners = [j.upper().replace("EXP", "") for j in parts[1::2]]
    if joiners and len(set(joiners)) > 1:
        warnings.append(f"Mixed AND/OR not supported: {expr!r}")
        return False
    terms = []
    for t in terms_str:
        c = _parse_cond(t, warnings)
        if c is None:
            return False
        terms.append(c)
    if not terms:
        return False
    instr.cond_lhs, instr.cond_op, instr.cond_rhs = terms[0]
    if len(terms) > 1:
        instr.cond_terms = terms
        instr.cond_join = joiners[0] if joiners else "AND"
    return True


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
    folder_name = ""
    pos_type = "PULSE"
    # Canonical key (e.g. 'C0', 'P1' via _norm_pos_key) → joints_deg. Padding-
    # independent so //INST tokens resolve to their //POS declaration.
    positions: dict[str, list[float]] = {}
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
        mf = _RE_FOLDER.match(s)
        if mf and not folder_name:
            folder_name = mf.group(1)[:32]
            continue
        if in_pos:
            mp = _RE_POSTYPE.match(s)
            if mp:
                pos_type = mp.group(1).upper()
                continue
            mc = _RE_POSVAR.match(s)
            if mc:
                letter, idx = mc.group(1), int(mc.group(2))
                try:
                    vals = [int(round(float(x)))
                            for x in mc.group(3).split(",") if x.strip() != ""]
                except ValueError:
                    warnings.append(f"Could not parse position var: {s!r}")
                    continue
                positions[_norm_pos_key(letter, idx)] = _pulses_to_deg(
                    vals, pulse_per_deg)
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
        if s == "COMM" or s.startswith("COMM ") or s.startswith("COMM\t"):
            continue                             # COMM = commented-out line, drop

        instr = _parse_inst_line(s, positions, pulse_per_deg, warnings)
        if instr is not None:
            instructions.append(instr)

    return ParsedJob(
        name=name or "IMPORTED",
        instructions=instructions,
        pos_type=pos_type,
        folder_name=folder_name,
        warnings=warnings,
        positions=positions,
    )


def _parse_motion_modifiers(s: str, instr: ParsedInstr) -> None:
    """Extract VJ/V/PL/TL/UF from the tail of a motion line into instr.
    Speeds may be a number OR a variable (V=I003 → instr.speed_var='I003')."""
    m = _RE_VJ.search(s)
    if m:
        tok = m.group(1)
        if tok[0] in "BIbi":
            instr.speed_var = tok.upper()
        else:
            instr.vj_pct = float(tok)
    m = _RE_V.search(s)
    if m:
        tok = m.group(1)
        if tok[0] in "BIbi":
            instr.speed_var = tok.upper()
        else:
            instr.v_mm_s = float(tok)
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
        # Indirect: P[Bxxx] — index resolved from a variable at playback.
        mi = _RE_POS_INDIRECT.match(token)
        if mi:
            instr = ParsedInstr(kind=head.lower(),
                                pos_index_var=mi.group(1).upper())
            _parse_motion_modifiers(s, instr)
            return instr
        mt = _RE_POS_TOKEN.match(token)
        if not mt:
            warnings.append(f"Unrecognised position token: {token!r} in {s!r}")
            return None
        key = _norm_pos_key(mt.group(1), int(mt.group(2)))
        if key not in positions:
            warnings.append(
                f"{head} references '{token}' not found in //POS "
                f"(P-var runtime?) — skipping line")
            return None
        instr = ParsedInstr(kind=head.lower(),
                            joints_deg=list(positions[key]),
                            pos_key=key)
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

    # ── Extended LG instructions ──
    m = _RE_PULSE.match(s)
    if m:
        # PULSE OT#(n) — momentary output pulse. Modelled as a SetDO (ON).
        return ParsedInstr(kind="pulse", do_index=int(m.group(1)))
    if _RE_CLEAR_STACK.match(s):
        return ParsedInstr(kind="clear_stack")
    m = _RE_CLEAR_VAR.match(s)
    if m:
        cnt = -1 if m.group(2).upper() == "ALL" else int(m.group(2))
        return ParsedInstr(kind="clear", var_name=m.group(1).upper(),
                           clear_count=cnt)
    m = _RE_DIN.match(s)
    if m:
        return ParsedInstr(kind="din", var_name=m.group(1).upper(),
                           io_group_kind=m.group(2).upper(),
                           io_group=int(m.group(3)))
    m = _RE_DOUT_OG.match(s)
    if m:
        return ParsedInstr(kind="dout_group", io_group=int(m.group(1)),
                           var_name=m.group(2).upper(), io_group_kind="OG")
    m = _RE_SET_EXPRESS.match(s)
    if m:
        return ParsedInstr(kind="setvar", var_name=m.group(1).upper(),
                           var_op="SET", var_expr=m.group(2).strip())

    # ── Flow control + variables ──
    m = _RE_LABEL.match(s)
    if m:
        return ParsedInstr(kind="label", label_name=m.group(1))
    m = _RE_JUMP.match(s)
    if m:
        instr = ParsedInstr(kind="jump", label_name=m.group(1))
        if m.group(2):
            _set_cond(instr, m.group(2), warnings)
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
        instr = ParsedInstr(kind="ifthen", cond_exp=bool(m.group(1)))
        if not _set_cond(instr, m.group(2), warnings):
            return None
        return instr
    m = _RE_ELSEIF.match(s)
    if m:
        instr = ParsedInstr(kind="elseif", cond_exp=bool(m.group(1)))
        if not _set_cond(instr, m.group(2), warnings):
            return None
        return instr
    m = _RE_WHILE.match(s)
    if m:
        instr = ParsedInstr(kind="while", cond_exp=bool(m.group(1)))
        if not _set_cond(instr, m.group(2), warnings):
            return None
        return instr
    if head == "ELSE":
        return ParsedInstr(kind="else")
    if head == "ENDIF":
        return ParsedInstr(kind="endif")
    if head == "ENDWHILE":
        return ParsedInstr(kind="endwhile")

    warnings.append(f"Instruction outside supported subset — skipping: {s!r}")
    return None
