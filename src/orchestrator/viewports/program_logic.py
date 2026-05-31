"""
program_logic.py
────────────────
Lõi logic INFORM (flow control + biến) — PURE, KHÔNG Qt/OpenGL/IK. Tách khỏi
playback mixin để unit-test 100% headless.

Cung cấp cho interpreter (`mixin_program_playback._play_program_loop`):
  • VarStore        — biến B (byte 0–255 wrap) / I (integer).
  • eval_condition  — so sánh 'lhs op rhs' (toán hạng = biến | literal | IN#(n)).
  • apply_setvar    — SET/ADD/SUB/MUL/DIV/INC/DEC.
  • resolve_labels  — map *LABEL → index (raise nếu trùng).
  • build_block_map — match IFTHEN/ELSEIF/ELSE/ENDIF + WHILE/ENDWHILE (raise lệch).
  • validate_program— gom lỗi (nhãn/khối/biến) cho UI guard (KHÔNG raise).
  • next_pc         — PC kế tiếp cho MỌI lệnh control-flow (thuần, testable).

Hằng `CONTROL_FLOW` = tập type mà interpreter route qua `next_pc`.
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
    """Kho biến INFORM B/I. B wrap 0–255 (byte), I là integer thường."""

    def __init__(self) -> None:
        self._v: dict[str, int] = {}

    @staticmethod
    def validate(name: str) -> str:
        name = name.strip().upper()
        if not _RE_VAR.match(name):
            raise ValueError(
                f"Tên biến không hợp lệ: '{name}' (cần B### hoặc I###, vd B000)")
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


def resolve_operand(tok: str, store: VarStore, io_reader: IoReader) -> int:
    """Token → giá trị int. Token = biến B/I | IN#(n) | literal int."""
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
        raise ValueError(f"Toán hạng không hợp lệ: '{tok}'") from e


def eval_condition(
    lhs: str, op: str, rhs: str,
    store: VarStore, io_reader: IoReader = _no_io,
) -> bool:
    cmp = _CMP.get(op)
    if cmp is None:
        raise ValueError(f"Toán tử so sánh không hỗ trợ: '{op}'")
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
        raise ValueError(f"Phép gán biến không hỗ trợ: '{op}'")
    store.set(name, res)


# ───── Labels & structured blocks ─────


def resolve_labels(program: list[Instruction]) -> dict[str, int]:
    m: dict[str, int] = {}
    for i, ins in enumerate(program):
        if ins.type == "Label":
            if ins.label_name in m:
                raise ValueError(f"Nhãn trùng: *{ins.label_name}")
            m[ins.label_name] = i
    return m


def build_block_map(program: list[Instruction]) -> dict[int, dict]:
    """Pre-pass khớp khối IF/WHILE → bm[index] = {jnext,jend|jstart}.

    Raise ValueError nếu khối lệch (ELSEIF/ELSE ngoài IF, ENDIF/ENDWHILE thừa,
    khối chưa đóng).
    """
    bm: dict[int, dict] = {}
    stack: list[dict] = []
    for i, ins in enumerate(program):
        t = ins.type
        if t == "IfThen":
            stack.append({"kind": "if", "clauses": [i]})
        elif t in ("ElseIf", "Else"):
            if not stack or stack[-1]["kind"] != "if":
                raise ValueError(f"{t.upper()} không nằm trong IFTHEN (dòng {i+1})")
            stack[-1]["clauses"].append(i)
        elif t == "EndIf":
            if not stack or stack[-1]["kind"] != "if":
                raise ValueError(f"ENDIF thừa (dòng {i+1})")
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
                raise ValueError(f"ENDWHILE thừa (dòng {i+1})")
            s = stack.pop()["start"]
            bm.setdefault(s, {})["jend"] = i
            bm.setdefault(i, {})["jstart"] = s
    if stack:
        kinds = ", ".join(b["kind"].upper() for b in stack)
        raise ValueError(f"Khối chưa đóng: {kinds}")
    return bm


def validate_program(program: list[Instruction]) -> list[str]:
    """Gom mọi lỗi tĩnh (KHÔNG raise) — dùng chặn Play/Export.

    Kiểm: nhãn trùng, JUMP tới nhãn không tồn tại, khối IF/WHILE cân bằng, tên
    biến + toán tử điều kiện hợp lệ.
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
            errors.append(f"Dòng {i+1}: JUMP tới nhãn không tồn tại *{ins.label_name}")
        if t == "SetVar":
            try:
                VarStore.validate(ins.var_name)
            except ValueError as e:
                errors.append(f"Dòng {i+1}: {e}")
            if ins.var_op.upper() not in (
                    "SET", "ADD", "SUB", "MUL", "DIV", "INC", "DEC"):
                errors.append(f"Dòng {i+1}: phép gán không hợp lệ '{ins.var_op}'")
        if (t in ("Jump", "IfThen", "ElseIf", "While") and ins.cond_op
                and ins.cond_op not in _CMP):
            errors.append(f"Dòng {i+1}: toán tử điều kiện không hỗ trợ '{ins.cond_op}'")
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
    """PC kế tiếp cho lệnh control-flow tại `pc`. THUẦN (trừ cập nhật `state` cho
    IF — dùng để biết nhánh nào đã chạy, key = index ENDIF tương ứng)."""
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
                raise ValueError(f"JUMP tới nhãn không tồn tại: *{ins.label_name}")
            return label_map[ins.label_name]
        return pc + 1

    if t == "IfThen":
        endif = block_map[pc]["jend"]
        state[endif] = False                       # reset mỗi lần vào IF
        if eval_condition(ins.cond_lhs, ins.cond_op, ins.cond_rhs,
                          store, io_reader):
            state[endif] = True
            return pc + 1
        return block_map[pc]["jnext"]

    if t == "ElseIf":
        endif = block_map[pc]["jend"]
        if state.get(endif):                       # đã có nhánh chạy → bỏ tới ENDIF
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
        return block_map[pc]["jend"] + 1           # thoát loop (sau ENDWHILE)

    if t == "EndWhile":
        return block_map[pc]["jstart"]             # quay lại WHILE đánh giá lại

    return pc + 1
