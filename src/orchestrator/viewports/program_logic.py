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


def _trunc_div(a: int, b: int) -> int:
    """Integer division truncating toward zero — matches Yaskawa controller
    semantics. Python's `//` floors (-7 // 2 == -4); the controller gives -3."""
    if b == 0:
        return a                                  # caller guards div-by-zero
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


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
        # Canonicalize to zero-padded 3-digit form: INFORM treats B5 / B05 / B005
        # as aliases for the same register, so the store must key them identically
        # (otherwise SET B5 then IF B005 read different slots, and CLEAR — which
        # always pads to :03d — would miss an un-padded var).
        return f"{name[0]}{int(name[1:]):03d}"

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
    """Is the token a valid operand? = B/I variable | IN#(n) | ON/OFF | literal int.

    Used by validate_program to block Play/Export for programs with bad operands
    (e.g. 'B0000' 4 digits, 'FOO', '??') — prevents malformed .JBI syntax from
    slipping through.
    """
    up = tok.strip().upper()
    if not up:
        return False
    if up in ("ON", "OFF") or _RE_VAR.match(up) or _RE_IN.match(up):
        return True
    try:
        int(up)
        return True
    except ValueError:
        return False


def resolve_operand(tok: str, store: VarStore, io_reader: IoReader) -> int:
    """Token → int value. Token = B/I variable | IN#(n) | ON/OFF | literal int."""
    tok = tok.strip()
    up = tok.upper()
    if up == "ON":
        return 1
    if up == "OFF":
        return 0
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


def eval_instr_condition(ins, store: VarStore, io_reader: IoReader = _no_io) -> bool:
    """Evaluate an Instruction's condition — compound (cond_terms + AND/OR) or
    single (cond_lhs/op/rhs). Used by the interpreter for Jump/If/ElseIf/While."""
    terms = getattr(ins, "cond_terms", None)
    if terms:
        results = [eval_condition(l, o, r, store, io_reader) for l, o, r in terms]
        if (getattr(ins, "cond_join", "AND") or "AND").upper() == "OR":
            return any(results)
        return all(results)
    return eval_condition(ins.cond_lhs, ins.cond_op, ins.cond_rhs,
                          store, io_reader)


# Arithmetic expression eval (SET Ixxx EXPRESS <expr>) — safe, no eval().
_RE_TOKENIZE = re.compile(r"\s*([BI]\d{1,3}|IN#\(\d+\)|\d+|[-+*/()])", re.IGNORECASE)


def eval_express(expr: str, store: VarStore, io_reader: IoReader = _no_io) -> int:
    """Evaluate an INFORM EXPRESS arithmetic expression (e.g. '5 * B005').

    Supports + - * / and parentheses over integer literals / B,I variables /
    IN#(n). Integer arithmetic (// for divide, like the controller). Safe
    recursive-descent parser — does NOT use eval()."""
    pos = 0
    tokens: list[str] = []
    s = expr.strip()
    while pos < len(s):
        m = _RE_TOKENIZE.match(s, pos)
        if not m:
            raise ValueError(f"Invalid EXPRESS token in {expr!r} at {s[pos:]!r}")
        tokens.append(m.group(1))
        pos = m.end()
    idx = [0]

    def peek():
        return tokens[idx[0]] if idx[0] < len(tokens) else None

    def nxt():
        t = tokens[idx[0]]; idx[0] += 1; return t

    def factor() -> int:
        t = peek()
        if t == "(":
            nxt(); v = expr_add();
            if peek() != ")":
                raise ValueError(f"Unbalanced parens in {expr!r}")
            nxt(); return v
        if t == "-":
            nxt(); return -factor()
        if t == "+":
            nxt(); return factor()
        if t is None:
            raise ValueError(f"Unexpected end of EXPRESS {expr!r}")
        nxt()
        return resolve_operand(t, store, io_reader)

    def term() -> int:
        v = factor()
        while peek() in ("*", "/"):
            op = nxt(); r = factor()
            v = v * r if op == "*" else (_trunc_div(v, r) if r != 0 else v)
        return v

    def expr_add() -> int:
        v = term()
        while peek() in ("+", "-"):
            op = nxt(); r = term()
            v = v + r if op == "+" else v - r
        return v

    result = expr_add()
    if idx[0] != len(tokens):
        raise ValueError(f"Trailing tokens in EXPRESS {expr!r}")
    return int(result)


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
        res = _trunc_div(cur, operand) if operand != 0 else cur  # guard div-by-zero
    else:
        raise ValueError(f"Unsupported variable assignment operator: '{op}'")
    store.set(name, res)


def apply_setvar_instr(ins, store: VarStore, io_reader: IoReader = _no_io) -> None:
    """Apply a SetVar Instruction. Handles `SET Ixxx EXPRESS <expr>` (var_expr)
    as well as plain SET/ADD/.../INC/DEC (var_arg)."""
    if getattr(ins, "var_expr", ""):
        store.set(ins.var_name, eval_express(ins.var_expr, store, io_reader))
        return
    apply_setvar(ins.var_name, ins.var_op, ins.var_arg, store, io_reader)


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
            if getattr(ins, "var_expr", ""):
                pass                                # EXPRESS validated at runtime
            elif op not in ("SET", "ADD", "SUB", "MUL", "DIV", "INC", "DEC"):
                errors.append(f"Line {i+1}: invalid assignment operator '{ins.var_op}'")
            elif op not in ("INC", "DEC") and not is_valid_operand(ins.var_arg):
                errors.append(
                    f"Line {i+1}: invalid operand '{ins.var_arg}' "
                    f"(expected B###/I###, IN#(n), or integer)")
        if t in ("Jump", "IfThen", "ElseIf", "While"):
            terms = getattr(ins, "cond_terms", None)
            if terms:                               # compound condition
                for (cl, co, cr) in terms:
                    if co not in _CMP:
                        errors.append(
                            f"Line {i+1}: unsupported condition operator '{co}'")
                    for side, val in (("left", cl), ("right", cr)):
                        if not is_valid_operand(val):
                            errors.append(
                                f"Line {i+1}: invalid {side} operand '{val}'")
            elif not ins.cond_op:
                # IfThen/ElseIf/While REQUIRE a condition; Jump w/o op = unconditional.
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
        if ins.cond_op or getattr(ins, "cond_terms", None):
            take = eval_instr_condition(ins, store, io_reader)
        if take:
            if ins.label_name not in label_map:
                raise ValueError(f"JUMP to undefined label: *{ins.label_name}")
            return label_map[ins.label_name]
        return pc + 1

    if t == "IfThen":
        endif = block_map[pc]["jend"]
        state[endif] = False                       # reset each time IF is entered
        if eval_instr_condition(ins, store, io_reader):
            state[endif] = True
            return pc + 1
        return block_map[pc]["jnext"]

    if t == "ElseIf":
        endif = block_map[pc]["jend"]
        if state.get(endif):                       # a branch already ran → skip to ENDIF
            return endif
        if eval_instr_condition(ins, store, io_reader):
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
        if eval_instr_condition(ins, store, io_reader):
            return pc + 1
        return block_map[pc]["jend"] + 1           # exit loop (past ENDWHILE)

    if t == "EndWhile":
        return block_map[pc]["jstart"]             # loop back to WHILE for re-evaluation

    return pc + 1
