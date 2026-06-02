"""
program_logic.py
────────────────
INFORM logic core (flow control + variables) — PURE, NO Qt/OpenGL/IK. Decoupled
from the playback mixin for 100% headless unit-testing.

Provides for the interpreter (`mixin_program_playback._play_program_loop`):
  • VarStore        — B variables (byte 0–255 wrap) / I (integer).
  • eval_condition  — evaluate 'lhs op rhs' (operand = variable | literal | IN#(n)).
  • apply_setvar    — SET/ADD/SUB/MUL/DIV/INC/DEC.
  • resolve_labels  — map *LABEL → index (raise on duplicate).
  • build_block_map — match IFTHEN/ELSEIF/ELSE/ENDIF + WHILE/ENDWHILE (raise on mismatch).
  • validate_program— collect errors (labels/blocks/variables) for UI guard (NO raise).
  • next_pc         — next PC for ALL control-flow instructions (pure, testable).

Constant `CONTROL_FLOW` = set of types that the interpreter routes through `next_pc`.
"""
from __future__ import annotations

import re
from typing import Callable

from .program_model import Instruction

CONTROL_FLOW = frozenset({
    "Label", "Jump", "IfThen", "ElseIf", "Else", "EndIf", "While", "EndWhile",
})

_RE_VAR = re.compile(r"^[BI]\d{1,3}$")
_RE_IN = re.compile(r"^IN#\((\d+)\)$", re.IGNORECASE)
_CMP = {
    "=": lambda a, b: a == b,
    "<>": lambda a, b: a != b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
}

IoReader = Callable[[int], bool]


def _no_io(_idx: int) -> bool:
    return False


# ───── Variables ─────


class VarStore:
    """INFORM B/I variable store. B wraps 0–255 (byte), I is a plain integer."""

    def __init__(self) -> None:
        self._v: dict[str, int] = {}

    @staticmethod
    def validate(name: str) -> str:
        name = name.strip().upper()
        if not _RE_VAR.match(name):
            raise ValueError(
                f"Invalid variable name: '{name}' (expected B### or I###, e.g. B000)")
        return name

    def get(self, name: str) -> int:
        return self._v.get(self.validate(name), 0)

    def set(self, name: str, value: int) -> None:
        name = self.validate(name)
        v = int(value)
        if name[0] == "B":
            v &= 0xFF                       # byte wrap 0–255
        self._v[name] = v

    def as_dict(self) -> dict[str, int]:
        return dict(self._v)


def is_valid_operand(tok: str) -> bool:
    """Is the token a valid operand? = B/I variable | IN#(n) | literal int.

    Used by validate_program to block Play/Export for programs with bad operands
    (e.g. 'B0000' 4 digits, 'FOO', '??') — prevents malformed .JBI syntax from
    slipping through.
    """
    up = tok.strip().upper()
    if not up:
        return False
    if _RE_VAR.match(up) or _RE_IN.match(up):
        return True
    try:
        int(up)
        return True
    except ValueError:
        return False


def resolve_operand(tok: str, store: VarStore, io_reader: IoReader) -> int:
    """Token → int value. Token = B/I variable | IN#(n) | literal int."""
    tok = tok.strip()
    up = tok.upper()
    if _RE_VAR.match(up):
        return store.get(up)
    m = _RE_IN.match(up)
    if m:
        return 1 if io_reader(int(m.group(1))) else 0
    try:
        return int(tok)
    except ValueError as e:
        raise ValueError(f"Invalid operand: '{tok}'") from e


def eval_condition(
    lhs: str, op: str, rhs: str,
    store: VarStore, io_reader: IoReader = _no_io,
) -> bool:
    cmp = _CMP.get(op)
    if cmp is None:
        raise ValueError(f"Unsupported comparison operator: '{op}'")
    a = resolve_operand(lhs, store, io_reader)
    b = resolve_operand(rhs, store, io_reader)
    return bool(cmp(a, b))


def apply_setvar(
    name: str, op: str, arg: str,
    store: VarStore, io_reader: IoReader = _no_io,
) -> None:
    op = op.upper()
    cur = store.get(name)
    if op == "INC":
        store.set(name, cur + 1); return
    if op == "DEC":
        store.set(name, cur - 1); return
    operand = resolve_operand(arg, store, io_reader)
    if op == "SET":
        res = operand
    elif op == "ADD":
        res = cur + operand
    elif op == "SUB":
        res = cur - operand
    elif op == "MUL":
        res = cur * operand
    elif op == "DIV":
        res = cur // operand if operand != 0 else cur     # guard div-by-zero
    else:
        raise ValueError(f"Unsupported variable assignment operator: '{op}'")
    store.set(name, res)


# ───── Labels & structured blocks ─────


def resolve_labels(program: list[Instruction]) -> dict[str, int]:
    m: dict[str, int] = {}
    for i, ins in enumerate(program):
        if ins.type == "Label":
            if ins.label_name in m:
                raise ValueError(f"Duplicate label: *{ins.label_name}")
            m[ins.label_name] = i
    return m


def build_block_map(program: list[Instruction]) -> dict[int, dict]:
    """Pre-pass matching IF/WHILE blocks → bm[index] = {jnext,jend|jstart}.

    Raises ValueError if blocks are mismatched (ELSEIF/ELSE outside IF,
    extra ENDIF/ENDWHILE, unclosed block).
    """
    bm: dict[int, dict] = {}
    stack: list[dict] = []
    for i, ins in enumerate(program):
        t = ins.type
        if t == "IfThen":
            stack.append({"kind": "if", "clauses": [i]})
        elif t in ("ElseIf", "Else"):
            if not stack or stack[-1]["kind"] != "if":
                raise ValueError(f"{t.upper()} not inside IFTHEN (line {i+1})")
            stack[-1]["clauses"].append(i)
        elif t == "EndIf":
            if not stack or stack[-1]["kind"] != "if":
                raise ValueError(f"Extra ENDIF (line {i+1})")
            clauses = stack.pop()["clauses"]
            for k, ci in enumerate(clauses):
                nxt = clauses[k + 1] if k + 1 < len(clauses) else i
                bm.setdefault(ci, {})
                bm[ci]["jnext"] = nxt
                bm[ci]["jend"] = i
            bm.setdefault(i, {})
        elif t == "While":
            stack.append({"kind": "while", "start": i})
        elif t == "EndWhile":
            if not stack or stack[-1]["kind"] != "while":
                raise ValueError(f"Extra ENDWHILE (line {i+1})")
            s = stack.pop()["start"]
            bm.setdefault(s, {})["jend"] = i
            bm.setdefault(i, {})["jstart"] = s
    if stack:
        kinds = ", ".join(b["kind"].upper() for b in stack)
        raise ValueError(f"Unclosed block(s): {kinds}")
    return bm


def validate_program(program: list[Instruction]) -> list[str]:
    """Collect all static errors (NO raise) — used to block Play/Export.

    Checks: duplicate labels, JUMP to undefined label, balanced IF/WHILE blocks,
    valid variable names, and valid condition operators.
    """
    errors: list[str] = []
    try:
        labels = resolve_labels(program)
    except ValueError as e:
        errors.append(str(e)); labels = {}
    try:
        build_block_map(program)
    except ValueError as e:
        errors.append(str(e))
    for i, ins in enumerate(program):
        t = ins.type
        if t == "Jump" and ins.label_name not in labels:
            errors.append(f"Line {i+1}: JUMP to undefined label *{ins.label_name}")
        if t == "SetVar":
            try:
                VarStore.validate(ins.var_name)
            except ValueError as e:
                errors.append(f"Line {i+1}: {e}")
            op = ins.var_op.upper()
            if op not in ("SET", "ADD", "SUB", "MUL", "DIV", "INC", "DEC"):
                errors.append(f"Line {i+1}: invalid assignment operator '{ins.var_op}'")
            elif op not in ("INC", "DEC") and not is_valid_operand(ins.var_arg):
                errors.append(
                    f"Line {i+1}: invalid operand '{ins.var_arg}' "
                    f"(expected B###/I###, IN#(n), or integer)")
        if t in ("Jump", "IfThen", "ElseIf", "While"):
            # IfThen/ElseIf/While REQUIRE a condition; Jump without op = unconditional.
            if not ins.cond_op:
                if t != "Jump":
                    errors.append(f"Line {i+1}: {t.upper()} missing condition")
            elif ins.cond_op not in _CMP:
                errors.append(
                    f"Line {i+1}: unsupported condition operator '{ins.cond_op}'")
            else:
                for side, val in (("left", ins.cond_lhs), ("right", ins.cond_rhs)):
                    if not is_valid_operand(val):
                        errors.append(
                            f"Line {i+1}: invalid {side} operand '{val}' "
                            f"(expected B###/I###, IN#(n), or integer)")
    return errors


def next_pc(
    program: list[Instruction],
    pc: int,
    store: VarStore,
    io_reader: IoReader,
    block_map: dict[int, dict],
    label_map: dict[str, int],
    state: dict[int, bool],
) -> int:
    """Next PC for the control-flow instruction at `pc`. PURE (except updating `state`
    for IF — used to track which branch has already executed, key = ENDIF index)."""
    ins = program[pc]
    t = ins.type

    if t == "Label":
        return pc + 1

    if t == "Jump":
        take = True
        if ins.cond_op:
            take = eval_condition(ins.cond_lhs, ins.cond_op, ins.cond_rhs,
                                  store, io_reader)
        if take:
            if ins.label_name not in label_map:
                raise ValueError(f"JUMP to undefined label: *{ins.label_name}")
            return label_map[ins.label_name]
        return pc + 1

    if t == "IfThen":
        endif = block_map[pc]["jend"]
        state[endif] = False                       # reset each time IF is entered
        if eval_condition(ins.cond_lhs, ins.cond_op, ins.cond_rhs,
                          store, io_reader):
            state[endif] = True
            return pc + 1
        return block_map[pc]["jnext"]

    if t == "ElseIf":
        endif = block_map[pc]["jend"]
        if state.get(endif):                       # a branch already ran → skip to ENDIF
            return endif
        if eval_condition(ins.cond_lhs, ins.cond_op, ins.cond_rhs,
                          store, io_reader):
            state[endif] = True
            return pc + 1
        return block_map[pc]["jnext"]

    if t == "Else":
        endif = block_map[pc]["jend"]
        if state.get(endif):
            return endif
        return pc + 1

    if t == "EndIf":
        return pc + 1

    if t == "While":
        if eval_condition(ins.cond_lhs, ins.cond_op, ins.cond_rhs,
                          store, io_reader):
            return pc + 1
        return block_map[pc]["jend"] + 1           # exit loop (past ENDWHILE)

    if t == "EndWhile":
        return block_map[pc]["jstart"]             # loop back to WHILE for re-evaluation

    return pc + 1
