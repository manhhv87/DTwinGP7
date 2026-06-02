"""
test_program_logic.py
─────────────────────
Lõi logic INFORM (biến + flow control) — PURE, headless (không Qt/OpenGL).

Test VarStore, eval_condition, apply_setvar, resolve_labels, build_block_map,
validate_program, và **interpreter** (qua next_pc) cho JUMP/IFTHEN/WHILE — kể cả
guard chặn loop vô hạn. Cũng kiểm JSON round-trip cho các Instruction type mới.
"""
from __future__ import annotations

import pytest

from src.orchestrator.viewports.program_logic import (
    CONTROL_FLOW,
    VarStore,
    apply_setvar,
    build_block_map,
    eval_condition,
    next_pc,
    resolve_labels,
    validate_program,
)
from src.orchestrator.viewports.program_model import Instruction


# ───── Helper: interpreter thuần (mirror _play_program_loop, không Qt) ─────


def run(program, io=None, guard_max=100_000):
    """Chạy program → trả (store, moves, steps). io: dict[int,bool] cho IN#."""
    io = io or {}
    label_map = resolve_labels(program)
    block_map = build_block_map(program)
    store = VarStore()
    state: dict[int, bool] = {}
    io_reader = lambda i: bool(io.get(i, False))   # noqa: E731
    pc = 0
    moves = 0
    steps = 0
    while pc < len(program):
        steps += 1
        if steps > guard_max:
            raise RuntimeError("loop guard")
        ins = program[pc]
        t = ins.type
        if t == "SetVar":
            apply_setvar(ins.var_name, ins.var_op, ins.var_arg, store, io_reader)
            pc += 1
            continue
        if t in CONTROL_FLOW:
            pc = next_pc(program, pc, store, io_reader, block_map, label_map, state)
            continue
        if t == "MoveJ":
            moves += 1
        pc += 1
    return store, moves, steps


# ───── VarStore ─────


class TestVarStore:
    def test_default_zero(self):
        assert VarStore().get("B000") == 0

    def test_byte_wraps(self):
        s = VarStore(); s.set("B000", 257)
        assert s.get("B000") == 1                  # 257 & 0xFF
        s.set("B001", -1)
        assert s.get("B001") == 255

    def test_integer_no_wrap(self):
        s = VarStore(); s.set("I000", 1000)
        assert s.get("I000") == 1000

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Invalid variable name"):
            VarStore().get("X9")
        with pytest.raises(ValueError):
            VarStore().set("B9999", 1)             # 4 digits > 3

    def test_case_insensitive(self):
        s = VarStore(); s.set("b000", 5)
        assert s.get("B000") == 5


# ───── Conditions ─────


class TestCondition:
    @pytest.mark.parametrize("op,a,b,exp", [
        ("=", 5, 5, True), ("=", 5, 4, False),
        ("<>", 5, 4, True), ("<>", 5, 5, False),
        (">", 5, 4, True), ("<", 4, 5, True),
        (">=", 5, 5, True), ("<=", 5, 6, True),
        (">=", 4, 5, False), ("<=", 6, 5, False),
    ])
    def test_operators(self, op, a, b, exp):
        s = VarStore(); s.set("B000", a)
        assert eval_condition("B000", op, str(b), s) is exp

    def test_var_vs_var(self):
        s = VarStore(); s.set("B000", 3); s.set("B001", 7)
        assert eval_condition("B000", "<", "B001", s) is True

    def test_input_operand(self):
        s = VarStore()
        io = lambda i: i == 1                       # noqa: E731
        assert eval_condition("IN#(1)", "=", "1", s, io) is True
        assert eval_condition("IN#(2)", "=", "1", s, io) is False

    def test_bad_operator_raises(self):
        with pytest.raises(ValueError, match="Unsupported comparison operator"):
            eval_condition("B000", "==", "1", VarStore())


# ───── SetVar ops ─────


class TestSetVar:
    def test_all_ops(self):
        s = VarStore()
        apply_setvar("I000", "SET", "10", s); assert s.get("I000") == 10
        apply_setvar("I000", "ADD", "5", s); assert s.get("I000") == 15
        apply_setvar("I000", "SUB", "3", s); assert s.get("I000") == 12
        apply_setvar("I000", "MUL", "2", s); assert s.get("I000") == 24
        apply_setvar("I000", "DIV", "5", s); assert s.get("I000") == 4   # floor
        apply_setvar("I000", "INC", "", s); assert s.get("I000") == 5
        apply_setvar("I000", "DEC", "", s); assert s.get("I000") == 4

    def test_div_by_zero_guarded(self):
        s = VarStore(); s.set("I000", 9)
        apply_setvar("I000", "DIV", "0", s)        # no crash
        assert s.get("I000") == 9                  # unchanged

    def test_set_from_var(self):
        s = VarStore(); s.set("B001", 42)
        apply_setvar("B000", "SET", "B001", s)
        assert s.get("B000") == 42


# ───── Labels & validation ─────


class TestLabels:
    def test_resolve(self):
        prog = [Instruction(type="Label", label_name="A"),
                Instruction(type="MoveJ"),
                Instruction(type="Label", label_name="B")]
        assert resolve_labels(prog) == {"A": 0, "B": 2}

    def test_duplicate_raises(self):
        prog = [Instruction(type="Label", label_name="A"),
                Instruction(type="Label", label_name="A")]
        with pytest.raises(ValueError, match="Duplicate label"):
            resolve_labels(prog)

    def test_validate_missing_jump_target(self):
        errs = validate_program([Instruction(type="Jump", label_name="NOPE")])
        assert errs and "NOPE" in errs[0]

    def test_validate_bad_varname(self):
        errs = validate_program(
            [Instruction(type="SetVar", var_name="ZZ", var_op="SET", var_arg="1")])
        assert errs

    def test_validate_clean(self):
        prog = [Instruction(type="Label", label_name="L"),
                Instruction(type="Jump", label_name="L",
                            cond_lhs="B000", cond_op="<", cond_rhs="3")]
        assert validate_program(prog) == []

    def test_validate_bad_cond_operand(self):
        # Toán hạng rác trong điều kiện phải bị chặn (không lọt ra .JBI / Play).
        for ins in (
            Instruction(type="IfThen", cond_lhs="B0000", cond_op=">",
                        cond_rhs="5"),                       # B#### 4 chữ số
            Instruction(type="While", cond_lhs="FOO", cond_op="=",
                        cond_rhs="1"),                       # không phải var/lit
            Instruction(type="Jump", label_name="L", cond_lhs="I000",
                        cond_op="=", cond_rhs="??"),         # rhs rác
        ):
            prog = [Instruction(type="Label", label_name="L"), ins]
            assert validate_program(prog), f"phải báo lỗi: {ins.type}"

    def test_validate_bad_setvar_operand(self):
        errs = validate_program(
            [Instruction(type="SetVar", var_name="I000", var_op="ADD",
                         var_arg="xyz")])
        assert errs

    def test_validate_struct_missing_cond(self):
        # IfThen/ElseIf/While bắt buộc có điều kiện.
        for t in ("IfThen", "While"):
            assert validate_program([Instruction(type=t)]), f"{t} thiếu cond"

    def test_validate_accepts_in_and_var_operands(self):
        prog = [
            Instruction(type="While", cond_lhs="IN#(1)", cond_op="=",
                        cond_rhs="1"),
            Instruction(type="SetVar", var_name="I000", var_op="ADD",
                        var_arg="B001"),
            Instruction(type="EndWhile"),
        ]
        assert validate_program(prog) == []


# ───── Interpreter — flat JUMP ─────


class TestInterpreterFlat:
    def test_count_loop(self):
        prog = [
            Instruction(type="SetVar", var_name="B000", var_op="SET", var_arg="0"),
            Instruction(type="Label", label_name="LOOP"),
            Instruction(type="SetVar", var_name="B000", var_op="INC"),
            Instruction(type="MoveJ"),
            Instruction(type="Jump", label_name="LOOP",
                        cond_lhs="B000", cond_op="<", cond_rhs="3"),
        ]
        store, moves, _ = run(prog)
        assert store.get("B000") == 3 and moves == 3

    def test_unconditional_jump_forward(self):
        prog = [
            Instruction(type="Jump", label_name="SKIP"),
            Instruction(type="MoveJ"),                 # skipped
            Instruction(type="Label", label_name="SKIP"),
            Instruction(type="MoveJ"),
        ]
        _, moves, _ = run(prog)
        assert moves == 1

    def test_infinite_loop_guard(self):
        prog = [
            Instruction(type="Label", label_name="L"),
            Instruction(type="Jump", label_name="L"),   # vô điều kiện → vô hạn
        ]
        with pytest.raises(RuntimeError, match="loop guard"):
            run(prog, guard_max=5000)

    def test_linear_program_unaffected(self):
        """Không có lệnh logic ⇒ interpreter chạy tuyến tính như for-loop cũ."""
        prog = [Instruction(type="MoveJ") for _ in range(4)]
        _, moves, steps = run(prog)
        assert moves == 4 and steps == 4


# ───── Interpreter — structured IF/WHILE ─────


class TestInterpreterStructured:
    def test_while_count(self):
        prog = [
            Instruction(type="SetVar", var_name="I000", var_op="SET", var_arg="0"),
            Instruction(type="While", cond_lhs="I000", cond_op="<", cond_rhs="5"),
            Instruction(type="MoveJ"),
            Instruction(type="SetVar", var_name="I000", var_op="INC"),
            Instruction(type="EndWhile"),
        ]
        store, moves, _ = run(prog)
        assert store.get("I000") == 5 and moves == 5

    def test_ifthen_elseif_else(self):
        prog = [
            Instruction(type="SetVar", var_name="I000", var_op="SET", var_arg="0"),
            Instruction(type="While", cond_lhs="I000", cond_op="<", cond_rhs="5"),
            Instruction(type="IfThen", cond_lhs="I000", cond_op=">=", cond_rhs="2"),
            Instruction(type="SetVar", var_name="B001", var_op="INC"),   # I>=2 → 3×
            Instruction(type="Else"),
            Instruction(type="SetVar", var_name="B002", var_op="INC"),   # I<2 → 2×
            Instruction(type="EndIf"),
            Instruction(type="SetVar", var_name="I000", var_op="INC"),
            Instruction(type="EndWhile"),
        ]
        store, _, _ = run(prog)
        assert store.get("B001") == 3 and store.get("B002") == 2

    def test_nested_if_in_while(self):
        prog = [
            Instruction(type="SetVar", var_name="I000", var_op="SET", var_arg="0"),
            Instruction(type="SetVar", var_name="B003", var_op="SET", var_arg="0"),
            Instruction(type="While", cond_lhs="I000", cond_op="<", cond_rhs="4"),
            Instruction(type="IfThen", cond_lhs="I000", cond_op="=", cond_rhs="2"),
            Instruction(type="SetVar", var_name="B003", var_op="SET", var_arg="99"),
            Instruction(type="EndIf"),
            Instruction(type="SetVar", var_name="I000", var_op="INC"),
            Instruction(type="EndWhile"),
        ]
        store, _, _ = run(prog)
        assert store.get("B003") == 99

    def test_unbalanced_block_raises(self):
        with pytest.raises(ValueError, match="Unclosed block"):
            build_block_map([Instruction(type="IfThen", cond_lhs="B000",
                                         cond_op="=", cond_rhs="1")])
        with pytest.raises(ValueError, match="Extra ENDIF"):
            build_block_map([Instruction(type="EndIf")])


# ───── JSON round-trip cho type mới ─────


class TestJsonRoundTrip:
    @pytest.mark.parametrize("ins", [
        Instruction(type="Label", label_name="LOOP"),
        Instruction(type="Jump", label_name="LOOP",
                    cond_lhs="B000", cond_op="<", cond_rhs="3"),
        Instruction(type="Jump", label_name="END"),
        Instruction(type="SetVar", var_name="B000", var_op="SET", var_arg="5"),
        Instruction(type="SetVar", var_name="I001", var_op="INC"),
        Instruction(type="IfThen", cond_lhs="B000", cond_op=">=", cond_rhs="2"),
        Instruction(type="ElseIf", cond_lhs="B000", cond_op="=", cond_rhs="0"),
        Instruction(type="Else"),
        Instruction(type="EndIf"),
        Instruction(type="While", cond_lhs="B001", cond_op="<>", cond_rhs="0"),
        Instruction(type="EndWhile"),
    ])
    def test_roundtrip(self, ins):
        d = ins.to_dict()
        assert Instruction.from_dict(d).to_dict() == d


# ───── App adapter: Instruction list → .JBI → Instruction list ─────


class TestAppAdapterRoundTrip:
    """Khóa chặt 2 chuỗi elif (_export_job_to_path + _jbi_to_instructions) —
    bắt typo field-name. Stub host headless (không Qt) cho chương trình
    MoveJ + logic (không MoveL nên không cần model/IK)."""

    def _stub(self):
        import types
        from src.orchestrator.viewports.mixin_program_io import ProgramIOMixin
        s = types.SimpleNamespace(
            _pp_max_speed_pct=30.0, _pp_default_vj=10.0, _pp_default_v_mms=100.0,
            _targets={}, _model=None, _tool_frames=[("flange", None)], _tool_idx=0)
        s.export = ProgramIOMixin._export_job_to_path.__get__(s)
        s.imp = ProgramIOMixin._jbi_to_instructions.__get__(s)
        return s

    def test_structured_pipeline(self, tmp_path):
        from src.orchestrator.backends.inform_parser import parse_jbi
        prog = [
            Instruction(type="SetVar", var_name="I000", var_op="SET", var_arg="0"),
            Instruction(type="While", cond_lhs="I000", cond_op="<", cond_rhs="3"),
            Instruction(type="IfThen", cond_lhs="I000", cond_op="=", cond_rhs="1"),
            Instruction(type="ShowMessage", message="ONE"),
            Instruction(type="ElseIf", cond_lhs="I000", cond_op=">", cond_rhs="1"),
            Instruction(type="MoveJ", joints=[0, 0, 0, 0, 0, 0]),
            Instruction(type="Else"),
            Instruction(type="ShowMessage", message="ZERO"),
            Instruction(type="EndIf"),
            Instruction(type="SetVar", var_name="I000", var_op="INC"),
            Instruction(type="EndWhile"),
        ]
        s = self._stub()
        path = tmp_path / "STRUCT.JBI"
        s.export(prog, "STRUCTJOB", path)
        parsed = parse_jbi(path.read_text(encoding="utf-8"))
        assert parsed.warnings == []
        back = s.imp(parsed)
        # Bỏ SetSpeed importer tự dựng cho MoveJ → so phần còn lại.
        back_types = [i.type for i in back if i.type != "SetSpeed"]
        assert back_types == [i.type for i in prog]
        # Khối parse lại phải cân bằng (build_block_map không raise).
        build_block_map(back)
