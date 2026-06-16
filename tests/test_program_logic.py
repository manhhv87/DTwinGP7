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
    apply_setvar_instr,
    build_block_map,
    eval_condition,
    eval_express,
    eval_instr_condition,
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


# ───── Extended LG features (EXPRESS, compound conditions, ON/OFF) ─────


class TestExpress:
    def test_basic_arithmetic(self):
        s = VarStore(); s.set("B005", 20)
        assert eval_express("5 * B005", s) == 100
        assert eval_express("10 * B005", s) == 200
        assert eval_express("B005 + 5", s) == 25
        assert eval_express("B005 - 8", s) == 12
        assert eval_express("100 / B005", s) == 5          # integer divide

    def test_precedence_and_parens(self):
        s = VarStore()
        assert eval_express("2 + 3 * 4", s) == 14
        assert eval_express("(2 + 3) * 4", s) == 20
        assert eval_express("-5 + 10", s) == 5

    def test_div_by_zero_guard(self):
        s = VarStore(); s.set("B000", 0)
        assert eval_express("10 / B000", s) == 10          # guarded → unchanged lhs

    def test_apply_setvar_instr_express(self):
        s = VarStore(); s.set("B005", 7)
        ins = Instruction(type="SetVar", var_name="I000", var_op="SET",
                          var_expr="5 * B005")
        apply_setvar_instr(ins, s)
        assert s.get("I000") == 35


class TestCompoundCondition:
    def test_andexp(self):
        s = VarStore(); s.set("I010", 5); s.set("B010", 5)
        ins = Instruction(type="IfThen",
                          cond_terms=[("I010", "<>", "11"), ("B010", "<>", "12")],
                          cond_join="AND")
        assert eval_instr_condition(ins, s) is True
        s.set("I010", 11)                                  # first term now false
        assert eval_instr_condition(ins, s) is False

    def test_orexp(self):
        s = VarStore(); s.set("B000", 1); s.set("B001", 99)
        ins = Instruction(type="IfThen",
                          cond_terms=[("B000", "=", "5"), ("B001", "=", "99")],
                          cond_join="OR")
        assert eval_instr_condition(ins, s) is True        # 2nd term true

    def test_in_on_off_condition(self):
        s = VarStore()
        ins = Instruction(type="IfThen", cond_lhs="IN#(8)", cond_op="=",
                          cond_rhs="ON")
        assert eval_instr_condition(ins, s, lambda i: i == 8) is True
        assert eval_instr_condition(ins, s, lambda i: False) is False

    def test_compound_condition_uses_exp_keyword(self):
        # Regression: a variable-only compound must emit the EXP keyword (matches the
        # ANDEXP joiner) — IFTHEN/WHILE + ANDEXP is invalid INFORM.
        for typ, kw in (("IfThen", "IFTHENEXP"), ("While", "WHILEEXP")):
            ins = Instruction(type=typ, cond_exp=False,
                              cond_terms=[("I010", "<>", "11"), ("B010", "<>", "12")],
                              cond_join="AND")
            line = ins.describe()
            assert line.startswith(kw + " ")
            assert " ANDEXP " in line

    def test_from_dict_missing_optional_keys_no_keyerror(self):
        # Regression: Wait without "seconds" / SetGripper without "close" must not
        # raise KeyError (which aborted the whole project load).
        assert Instruction.from_dict({"type": "Wait"}).wait_seconds == 0.0
        assert Instruction.from_dict({"type": "SetGripper"}).gripper_close is True


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

    def test_validate_jump_into_if_block_rejected(self):
        # JUMP from OUTSIDE into a label inside an IF block is illegal INFORM.
        prog = [
            Instruction(type="Jump", label_name="INNER"),       # outside the IF
            Instruction(type="IfThen", cond_lhs="B000", cond_op="=", cond_rhs="1"),
            Instruction(type="Label", label_name="INNER"),      # inside the IF
            Instruction(type="EndIf"),
        ]
        errs = validate_program(prog)
        assert any("structured" in e and "INNER" in e for e in errs), errs

    def test_validate_jump_outward_from_block_allowed(self):
        # Jumping OUT of a block (to an enclosing/top-level label) is fine.
        prog = [
            Instruction(type="Label", label_name="DONE"),
            Instruction(type="IfThen", cond_lhs="B000", cond_op="=", cond_rhs="1"),
            Instruction(type="Jump", label_name="DONE"),        # inside → out
            Instruction(type="EndIf"),
        ]
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
        from src.orchestrator.viewports.mixin_job_target import JobTargetMixin
        s.export = ProgramIOMixin._export_job_to_path.__get__(s)
        s.imp = ProgramIOMixin._jbi_to_instructions.__get__(s)
        s._safe_target_name = JobTargetMixin._safe_target_name
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
        targets: dict = {}
        back = s.imp(parsed, targets)
        # Bỏ SetSpeed importer tự dựng cho MoveJ → so phần còn lại.
        back_types = [i.type for i in back if i.type != "SetSpeed"]
        assert back_types == [i.type for i in prog]
        # Khối parse lại phải cân bằng (build_block_map không raise).
        build_block_map(back)

    def test_pvar_job_shared_targets(self, tmp_path):
        """Import .JBI có P-var dùng lại → MoveJ/MoveL tham chiếu shared target
        (giữ joints chính xác, điểm dùng lại → 1 target)."""
        from src.orchestrator.backends.inform_parser import parse_jbi
        text = (
            "/JOB\r\n//NAME PP\r\n//POS\r\n///NPOS 0,0,0,2,0,0\r\n"
            "///TOOL 0\r\n///POSTYPE PULSE\r\n///PULSE\r\n"
            "P00001=55242,26922,11446,1601,-60437,64378\r\n"
            "P00002=55242,35623,-7881,1830,-43902,63991\r\n"
            "//INST\r\n///DATE 2026/01/01 00:00\r\n///ATTR SC,RW\r\n"
            "///GROUP1 RB1\r\nNOP\r\n"
            "MOVJ P001 VJ=20.00\r\nMOVL P002 V=80.0\r\nMOVL P001 V=80.0\r\nEND\r\n")
        parsed = parse_jbi(text)
        s = self._stub()
        targets: dict = {}
        back = s.imp(parsed, targets)
        moves = [i for i in back if i.type in ("MoveJ", "MoveL")]
        # 3 motions, all via target_name (no inline joints/pose).
        assert [i.type for i in moves] == ["MoveJ", "MoveL", "MoveL"]
        assert all(i.target_name and not i.joints and not i.tcp_pose
                   for i in moves)
        # P001 reused → 1st and 3rd share one target; only 2 unique targets.
        assert moves[0].target_name == moves[2].target_name
        assert len(targets) == 2
        # Exact joints preserved (no FK/IK drift).
        assert targets[moves[0].target_name]["joints"][0] == \
            parsed.instructions[0].joints_deg[0]

    def test_dout_gripper_index_preserved_for_sim(self):
        """DOUT OT#(n) → SetDO do_index=n must be preserved so the sim can
        actuate the gripper on the configured index (regression: pick-place jobs
        that grip via DOUT must grab objects in sim, not just log)."""
        from src.orchestrator.backends.inform_parser import parse_jbi
        text = (
            "/JOB\r\n//NAME G\r\n//POS\r\n///NPOS 0,0,0,1,0,0\r\n"
            "///TOOL 0\r\n///POSTYPE PULSE\r\n///PULSE\r\nP00001=0,0,0,0,0,0\r\n"
            "//INST\r\n///DATE 2026/01/01 00:00\r\n///ATTR SC,RW\r\n"
            "///GROUP1 RB1\r\nNOP\r\n"
            "MOVJ P001 VJ=20.00\r\nDOUT OT#(10) ON\r\nDOUT OT#(10) OFF\r\nEND\r\n")
        parsed = parse_jbi(text)
        s = self._stub()
        back = s.imp(parsed, {})
        douts = [i for i in back if i.type == "SetDO"]
        assert [d.do_index for d in douts] == [10, 10]
        assert [d.do_state for d in douts] == [True, False]


class TestCallJobResolution:
    """CALL JOB lookup for master → N sub-jobs (incl. nested + //NAME mismatch)."""

    def _stub(self, jobs):
        import types
        from src.orchestrator.viewports.mixin_program_playback import (
            ProgramPlaybackMixin,
        )
        from src.orchestrator.viewports.mixin_job_target import JobTargetMixin
        s = types.SimpleNamespace(_jobs=jobs, _sim_speed_mult=1.0, executed=[])
        s._safe_job_name = JobTargetMixin._safe_job_name
        # Capture executions instead of animating.
        s._signals = types.SimpleNamespace(
            status=types.SimpleNamespace(emit=lambda *a, **k: None))

        def fake_exec_job(prog, store, sim_io, guard, call_stack, job_label=""):
            s.executed.append(job_label)
        s._exec_job = fake_exec_job
        s._exec_call_job = ProgramPlaybackMixin._exec_call_job.__get__(s)
        return s

    def test_call_resolves_by_call_name(self):
        jobs = {"MASTER": [], "SUB_A": [], "SUB_B": []}
        s = self._stub(jobs)
        s._exec_call_job("SUB_A", None, {}, [0], [])
        s._exec_call_job("SUB_B", None, {}, [0], [])
        assert s.executed == ["SUB_A", "SUB_B"]

    def test_call_missing_subjob_skips(self):
        s = self._stub({"MASTER": []})
        s._exec_call_job("NOPE", None, {}, [0], [])
        assert s.executed == []                      # skipped, no crash

    def test_call_recursion_blocked(self):
        s = self._stub({"A": []})
        s._exec_call_job("A", None, {}, [0], ["A"])  # already on stack
        assert s.executed == []


# ───── Verbatim re-export (import .JBI → unchanged → byte-exact) ─────


class TestVerbatimReexport:
    """An imported .JBI job re-exports byte-for-byte when UNCHANGED, but is
    re-synthesised from the model once edited (the two user workflows)."""

    def _stub(self):
        import types
        from src.orchestrator.viewports.mixin_program_io import ProgramIOMixin
        from src.orchestrator.viewports.mixin_job_target import JobTargetMixin
        s = types.SimpleNamespace(
            _pp_max_speed_pct=100.0, _pp_default_vj=20.0, _pp_default_v_mms=80.0,
            _targets={}, _model=None, _tool_frames=[("flange", None)],
            _tool_idx=0, _jbi_raw={}, _jbi_positions={})
        s._safe_target_name = JobTargetMixin._safe_target_name
        s._safe_job_name = JobTargetMixin._safe_job_name
        s._job_signature = ProgramIOMixin._job_signature
        s.export = ProgramIOMixin._export_job_to_path.__get__(s)
        s.imp = ProgramIOMixin._jbi_to_instructions.__get__(s)
        return s

    _JOB = (
        "/JOB\r\n//NAME J\r\n//POS\r\n///NPOS 0,0,0,1,0,0\r\n///TOOL 0\r\n"
        "///POSTYPE PULSE\r\n///PULSE\r\nP00001=55242,26922,11446,1601,-60437,64378\r\n"
        "//INST\r\n///DATE 2024/08/21 18:20\r\n///ATTR SC,RW\r\n///GROUP1 RB1\r\n"
        "NOP\r\n'a comment line\r\nIFTHENEXP B000=1\r\n\t MOVJ P001 VJ=I000\r\n"
        "ENDIF\r\nEND\r\n")

    def test_unedited_exports_verbatim(self, tmp_path):
        from src.orchestrator.backends.inform_parser import parse_jbi
        s = self._stub()
        pj = parse_jbi(self._JOB)
        prog = s.imp(pj, s._targets)
        s._jbi_raw["J"] = {"text": self._JOB, "name": "J",
                           "sig": s._job_signature(prog, s._targets)}
        out = tmp_path / "J.JBI"
        s.export(prog, "J", out, raw_key="J")
        # Byte-for-byte identical — comment, IFTHENEXP, VJ=I000, indent all kept.
        # (read_bytes: read_text would normalize CRLF and mask the comparison.)
        assert out.read_bytes() == self._JOB.encode("utf-8")

    def test_edited_resynthesises(self, tmp_path):
        from src.orchestrator.backends.inform_parser import parse_jbi
        s = self._stub()
        pj = parse_jbi(self._JOB)
        prog = s.imp(pj, s._targets)
        s._jbi_raw["J"] = {"text": self._JOB, "name": "J",
                           "sig": s._job_signature(prog, s._targets)}
        prog.append(Instruction(type="Wait", wait_seconds=2.0))   # edit
        out = tmp_path / "J.JBI"
        s.export(prog, "J", out, raw_key="J")
        txt = out.read_text(encoding="utf-8")
        assert txt != self._JOB                  # not verbatim
        assert "TIMER T=2.000" in txt            # synthesised edit present
        assert "'a comment line" not in txt      # comments dropped on synthesis

    def test_reteaching_target_invalidates_verbatim(self, tmp_path):
        """Re-teaching a referenced target (joints change, instruction list
        UNCHANGED) must NOT take the verbatim path — otherwise the old pulses
        upload to the real robot (bug #7)."""
        from src.orchestrator.backends.inform_parser import parse_jbi
        s = self._stub()
        pj = parse_jbi(self._JOB)
        prog = s.imp(pj, s._targets)
        s._jbi_raw["J"] = {"text": self._JOB, "name": "J",
                           "sig": s._job_signature(prog, s._targets)}
        assert s._targets, "import should have created a referenced target"
        name = next(iter(s._targets))
        # Simulate a re-teach after a fixture shift: change the target joints only.
        s._targets[name]["joints"] = [d + 5.0 for d in s._targets[name]["joints"]]
        out = tmp_path / "J.JBI"
        s.export(prog, "J", out, raw_key="J")
        txt = out.read_text(encoding="utf-8")
        assert txt != self._JOB                  # re-synthesised with NEW pulses
        assert "P00001=55242,26922" not in txt   # the STALE pulses must be gone


# ───── Full instruction coverage (every type round-trips + exports) ─────


class TestAllTypesCoverage:
    """Build a program containing one of every instruction type and verify
    dict round-trip + describe for all, and export → re-parse with no warnings."""

    ALL_TYPES = [
        "MoveJ", "MoveL", "MoveC", "SetGripper", "SetDO", "Wait", "WaitIO",
        "SetSpeed", "SetRounding", "SetTool", "SetRefFrame", "ShowMessage",
        "CallJob", "SimEvent", "Label", "Jump", "SetVar", "IfThen", "ElseIf",
        "Else", "EndIf", "While", "EndWhile", "PulseDO", "ClearStack",
        "ClearVar", "ReadGroupIn", "WriteGroupOut",
    ]

    @staticmethod
    def _mk(t):
        d = {"type": t}
        if t == "MoveJ":
            d["joints"] = [0.0] * 6
        elif t in ("MoveL",):
            d["tcp_pose"] = [0.0] * 6
        elif t == "MoveC":
            d["tcp_pose"] = [0.0] * 6; d["tcp_pose_mid"] = [0.0] * 6
        elif t == "SetVar":
            d["var_name"] = "I000"; d["var_op"] = "SET"; d["var_arg"] = "1"
        elif t in ("Label", "Jump"):
            d["label_name"] = "L1"
        elif t in ("IfThen", "ElseIf", "While"):
            d["cond_lhs"] = "B000"; d["cond_op"] = "="; d["cond_rhs"] = "1"
        elif t == "ClearVar":
            d["var_name"] = "I010"; d["clear_count"] = 2
        elif t in ("ReadGroupIn", "WriteGroupOut"):
            d["var_name"] = "B005"; d["io_group"] = 2
            d["io_group_kind"] = "IG" if t == "ReadGroupIn" else "OG"
        elif t == "CallJob":
            d["job_name"] = "SUB"
        elif t == "PulseDO":
            d["do_index"] = 6
        return Instruction(**d)

    def test_every_type_describes_and_roundtrips(self):
        for t in self.ALL_TYPES:
            ins = self._mk(t)
            assert not ins.describe().startswith("?"), f"{t}: describe fallback"
            rt = Instruction.from_dict(ins.to_dict())
            assert rt.to_dict() == ins.to_dict(), f"{t}: dict round-trip mismatch"

    def test_logic_program_exports_and_reparses_clean(self, tmp_path):
        import types
        from src.orchestrator.backends.inform_parser import parse_jbi
        from src.orchestrator.viewports.mixin_program_io import ProgramIOMixin
        from src.orchestrator.viewports.mixin_job_target import JobTargetMixin

        prog = [
            Instruction(type="Label", label_name="TOP"),
            Instruction(type="SetVar", var_name="I000", var_op="SET", var_arg="0"),
            Instruction(type="SetVar", var_name="I001", var_op="SET",
                        var_expr="5 * B005"),
            Instruction(type="ClearStack"),
            Instruction(type="ClearVar", var_name="I010", clear_count=2),
            Instruction(type="ReadGroupIn", var_name="B005", io_group=2,
                        io_group_kind="IG"),
            Instruction(type="WriteGroupOut", var_name="B005", io_group=2,
                        io_group_kind="OG"),
            Instruction(type="PulseDO", do_index=6),
            Instruction(type="IfThen", cond_lhs="IN#(8)", cond_op="=",
                        cond_rhs="ON", cond_exp=True),
            Instruction(type="ShowMessage", message="HI"),
            Instruction(type="ElseIf", cond_lhs="B000", cond_op=">", cond_rhs="5"),
            Instruction(type="WaitIO", io_index=3, io_state=True),
            Instruction(type="Else"),
            Instruction(type="SetDO", do_index=1, do_state=True),
            Instruction(type="EndIf"),
            Instruction(type="While",
                        cond_terms=[("I000", "<", "3"), ("B010", "<>", "12")],
                        cond_join="AND"),
            Instruction(type="SetVar", var_name="I000", var_op="INC"),
            Instruction(type="Wait", wait_seconds=1.0),
            Instruction(type="EndWhile"),
            Instruction(type="CallJob", job_name="SUB"),
            Instruction(type="Jump", label_name="TOP", cond_lhs="I001",
                        cond_op="<>", cond_rhs="9"),
        ]
        s = types.SimpleNamespace(
            _model=None, _tool_frames=[("f", None)], _tool_idx=0, _targets={},
            _jbi_raw={}, _jbi_positions={}, _pp_max_speed_pct=100.0,
            _pp_default_vj=20.0, _pp_default_v_mms=80.0)
        s._safe_target_name = JobTargetMixin._safe_target_name
        s._safe_job_name = JobTargetMixin._safe_job_name
        s._job_signature = ProgramIOMixin._job_signature
        exp = ProgramIOMixin._export_job_to_path.__get__(s)
        out = tmp_path / "ALL.JBI"
        exp(prog, "ALLTEST", out)
        txt = out.read_bytes().decode("utf-8")
        # Spot-check the tricky lines, then confirm a clean re-parse.
        assert "IFTHENEXP IN#(8)=ON " in txt          # I/O cond → EXP + trailing space
        assert "WHILEEXP I000<3 ANDEXP B010<>12" in txt  # compound → EXP keyword
        assert "SET I001 EXPRESS 5 * B005" in txt     # arithmetic expression
        assert "CLEAR STACK" in txt and "CLEAR I010 2" in txt
        assert "DIN B005 IG#(2)" in txt and "DOUT OG#(2) B005" in txt
        assert "PULSE OT#(6)" in txt
        assert parse_jbi(txt).warnings == []          # round-trip is lossless


# ───── Condition edit helpers (compound parse + EXP toggle, no Qt) ─────


class TestConditionEditHelpers:
    """_parse_cond_text + _apply_cond power the Program-panel condition editor:
    compound ANDEXP/OREXP entry and the IFTHEN↔IFTHENEXP toggle."""

    def _fns(self):
        from src.orchestrator.viewports.gp7_app_qt import GP7AppQt
        return GP7AppQt._parse_cond_text, GP7AppQt._apply_cond

    def test_parse_single_and_compound(self):
        parse, _ = self._fns()
        assert parse("B000>5") == ([("B000", ">", "5")], "")
        assert parse("IN#(8)=ON") == ([("IN#(8)", "=", "ON")], "")
        assert parse("I010<>11 ANDEXP B010<>12") == (
            [("I010", "<>", "11"), ("B010", "<>", "12")], "AND")
        assert parse("B000=1 OREXP B001=2") == (
            [("B000", "=", "1"), ("B001", "=", "2")], "OR")

    def test_parse_rejects_bad(self):
        parse, _ = self._fns()
        assert parse("garbage") is None
        assert parse("a AND b ANDEXP c") is None      # mixed/incomplete terms

    def test_apply_single_with_exp_toggle(self):
        _, apply = self._fns()
        ins = Instruction(type="IfThen")
        apply(None, ins, {"terms": [("I010", "=", "12")], "join": "AND", "exp": True})
        assert (ins.cond_lhs, ins.cond_op, ins.cond_rhs) == ("I010", "=", "12")
        assert ins.cond_terms == [] and ins.cond_exp is True

    def test_apply_compound_then_back_to_single_clears_terms(self):
        _, apply = self._fns()
        ins = Instruction(type="While")
        apply(None, ins, {"terms": [("I000", "<", "3"), ("B010", "<>", "12")],
                          "join": "AND", "exp": False})
        assert ins.cond_terms == [("I000", "<", "3"), ("B010", "<>", "12")]
        assert ins.cond_join == "AND"
        apply(None, ins, {"terms": [("B000", ">", "5")], "join": "AND", "exp": False})
        assert ins.cond_terms == []                    # compound terms cleared
        assert (ins.cond_lhs, ins.cond_op, ins.cond_rhs) == ("B000", ">", "5")

    def test_apply_unconditional(self):
        _, apply = self._fns()
        ins = Instruction(type="Jump", label_name="L",
                          cond_lhs="B000", cond_op="=", cond_rhs="1")
        apply(None, ins, None)
        assert (ins.cond_lhs, ins.cond_op, ins.cond_rhs) == ("", "", "")
        assert ins.cond_terms == [] and ins.cond_exp is False


# ───── Replace instruction type (default instances are valid) ─────


class TestReplaceInstruction:
    """The Replace button swaps a step's TYPE in place via _default_instruction;
    every offered type must produce a valid, describable instance."""

    def _mk_fn(self):
        import types
        from src.orchestrator.viewports.gp7_app_qt import GP7AppQt
        s = types.SimpleNamespace(_joints=[0.0] * 6)
        s._current_tool_world = lambda: None
        return GP7AppQt._default_instruction.__get__(s), GP7AppQt._REPLACE_TYPES

    def test_all_replace_types_valid(self):
        from src.orchestrator.viewports.program_model import Instruction
        mk, replace_types = self._mk_fn()
        for _label, t, exp in replace_types:
            ins = mk(t, exp)
            assert ins.type == t
            assert not ins.describe().startswith("?"), f"{t}: describe fallback"
            # dict round-trip must survive too (save/load safe)
            assert Instruction.from_dict(ins.to_dict()).type == t
            # EXP flavour sets cond_exp on flow-control conditions
            if exp:
                assert ins.cond_exp is True, f"{t}: exp flag not set"

    def test_replace_exp_variant_present(self):
        """The Replace picker must offer the EXP keyword variants (IFTHENEXP
        etc.) — they were missing originally."""
        _mk, replace_types = self._mk_fn()
        exp_types = {t for _l, t, e in replace_types if e}
        assert exp_types == {"IfThen", "ElseIf", "While"}

    def test_replace_keeps_position_and_validity(self):
        from src.orchestrator.viewports.program_model import Instruction
        from src.orchestrator.viewports.program_logic import validate_program
        mk, _ = self._mk_fn()
        prog = [Instruction(type="Label", label_name="L"),
                Instruction(type="MoveJ", joints=[0.0] * 6),
                Instruction(type="Jump", label_name="L")]
        prog[1] = mk("Wait")                      # replace middle step
        assert [i.type for i in prog] == ["Label", "Wait", "Jump"]
        assert validate_program(prog) == []       # still a valid program


class TestCondExpDisplay:
    """describe() must distinguish IFTHEN from IFTHENEXP (and ELSEIF/WHILE) so a
    Replace IFTHEN→IFTHENEXP is visible in the program list (regression)."""

    def test_ifthen_vs_ifthenexp_display(self):
        plain = Instruction(type="IfThen", cond_lhs="B000", cond_op="=",
                            cond_rhs="1", cond_exp=False)
        exp = Instruction(type="IfThen", cond_lhs="B000", cond_op="=",
                          cond_rhs="1", cond_exp=True)
        assert plain.describe().startswith("IFTHEN ")
        assert exp.describe().startswith("IFTHENEXP ")
        assert plain.describe() != exp.describe()

    def test_while_elseif_exp_display(self):
        w = Instruction(type="While", cond_lhs="IN#(8)", cond_op="=",
                        cond_rhs="ON", cond_exp=True)
        e = Instruction(type="ElseIf", cond_lhs="B000", cond_op=">",
                        cond_rhs="5", cond_exp=True)
        assert w.describe().startswith("WHILEEXP ")
        assert e.describe().startswith("ELSEIFEXP ")


class TestJointTurnLabels:
    """§3.3 turn numbers (Yaskawa/RoboDK convention): turn 0 = principal value in
    (-180°,180°]; ±1 = one extra ±360° winding (only on axes with range > 360°)."""

    def _G(self):
        from src.orchestrator.viewports.gp7_app_qt import GP7AppQt
        return GP7AppQt

    def test_principal_values_are_turn_zero(self):
        G = self._G()
        assert G._solution_turns([0, 179, -180, -60, 120, 43]) == [0, 0, 0, 0, 0, 0]

    def test_plus_minus_full_turns(self):
        G = self._G()
        # 300 = -60 + 360 → +1 ; -240 = 120 - 360 → -1 ; 181 = -179 + 360 → +1
        assert G._solution_turns([0, 0, 0, 0, 0, 300])[5] == 1
        assert G._solution_turns([0, 0, 0, 0, 0, -240])[5] == -1
        assert G._solution_turns([0, 0, 0, 0, 0, 181])[5] == 1

    def test_label_lists_only_nonzero_turns(self):
        G = self._G()
        assert G._turn_label([0, 0, 0, 0, 0, 1]) == "J6+1"
        assert G._turn_label([0, 0, 0, -1, 0, 1]) == "J4-1 J6+1"
        assert G._turn_label([0, 0, 0, 0, 0, 0]) == "turn 0"


class TestAlignToReferenceFrame:
    """Align button (RoboDK-style): keep TCP position, snap tool ORIENTATION to be
    axis-aligned with the reference frame, solve IK, animate. Pure-math parts here
    (signed-perm group + nearest-rotation snap) need no Qt/OpenGL."""

    def _import(self):
        from src.orchestrator.viewports.gp7_app_qt import GP7AppQt
        return GP7AppQt

    def test_signed_perm_group_is_24_proper_rotations(self):
        import numpy as np
        mats = self._import()._signed_perm_rotations()
        assert len(mats) == 24
        for m in mats:
            assert abs(np.linalg.det(m) - 1.0) < 1e-9
            assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)

    def test_snap_tilted_orientation_back_to_frame(self):
        import math, numpy as np, types
        GP7AppQt = self._import()
        s = types.SimpleNamespace(
            _signed_perm_rotations=GP7AppQt._signed_perm_rotations)
        th = math.radians(15)
        R_cur = np.array([[math.cos(th), -math.sin(th), 0],
                          [math.sin(th),  math.cos(th), 0],
                          [0, 0, 1]])
        R_al = GP7AppQt._align_target_rotation.__get__(s)(R_cur, np.eye(3))
        assert np.allclose(R_al, np.eye(3), atol=1e-9)

    def test_exact_axis_pose_is_left_unchanged(self):
        import numpy as np, types
        GP7AppQt = self._import()
        s = types.SimpleNamespace(
            _signed_perm_rotations=GP7AppQt._signed_perm_rotations)
        R_cur = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)  # 90° about X
        R_al = GP7AppQt._align_target_rotation.__get__(s)(R_cur, np.eye(3))
        assert np.allclose(R_al, R_cur, atol=1e-9)

    def _stub(self, R_cur, ik_result=(0.0,) * 6):
        import numpy as np, types
        from src.orchestrator.viewports.gp7_app_qt import GP7AppQt
        T_cur = np.eye(4)
        T_cur[:3, :3] = R_cur
        T_cur[:3, 3] = [400.0, 0.0, 300.0]
        captured = {}
        s = types.SimpleNamespace(
            _model=object(), _joints=[0, 0, 0, 0, 0, 0],
            _base_xyz=(0.0, 0.0, 0.0),
            _ref_frames=[("Base", np.eye(4))], _ref_idx=0,
            _ALIGN_MIN_ANGLE_DEG=GP7AppQt._ALIGN_MIN_ANGLE_DEG,
            _signed_perm_rotations=GP7AppQt._signed_perm_rotations,
            _align_target_rotation=GP7AppQt._align_target_rotation,
            _current_tool_world=lambda: T_cur,
            _solve_cartesian=lambda T, seeded=True: (
                captured.setdefault("target", T), list(ik_result))[1],
            _act_live_jog=None,
            _start_animation=lambda j, label: (
                captured.setdefault("joints", list(j)), True)[1],
            _set_status=lambda msg, level="ok": captured.setdefault("status", (msg, level)),
        )
        # bind methods that call self
        s._align_target_rotation = GP7AppQt._align_target_rotation.__get__(s)
        return s, captured

    def _run(self, stub):
        import time
        from src.orchestrator.viewports.gp7_app_qt import GP7AppQt
        GP7AppQt._on_align.__get__(stub)()
        time.sleep(0.05)

    def test_tilted_tool_triggers_ik_and_keeps_position(self):
        import math, numpy as np
        th = math.radians(20)
        R_cur = np.array([[math.cos(th), -math.sin(th), 0],
                          [math.sin(th),  math.cos(th), 0],
                          [0, 0, 1]])
        stub, cap = self._stub(R_cur, ik_result=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        self._run(stub)
        # _solve_cartesian returns radians; _on_align animates in degrees
        assert cap["joints"] == [math.degrees(q) for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)]
        # IK target keeps the original TCP position
        assert np.allclose(cap["target"][:3, 3], [400.0, 0.0, 300.0])
        # ...and uses the snapped (identity) orientation
        assert np.allclose(cap["target"][:3, :3], np.eye(3), atol=1e-9)

    def test_already_aligned_does_not_move(self):
        import numpy as np
        stub, cap = self._stub(np.eye(3))
        self._run(stub)
        assert "joints" not in cap
        assert "already aligned" in cap["status"][0]

    def test_no_ik_solution_warns(self):
        import math, numpy as np
        th = math.radians(20)
        R_cur = np.array([[math.cos(th), -math.sin(th), 0],
                          [math.sin(th),  math.cos(th), 0],
                          [0, 0, 1]])
        stub, cap = self._stub(R_cur)
        # force IK failure
        stub._solve_cartesian = lambda T, seeded=True: None
        self._run(stub)
        assert "joints" not in cap
        assert cap["status"][1] == "warn"


class TestSetSpeedVEffective:
    """V (linear mm/s) of a SetSpeed only matters when a MOVL/MOVC follows it.
    Guards the GUI honesty fix: V is hidden/disabled (not silently reset) when the
    SetSpeed governs joint moves only, since MOVJ carries no V= tag in .JBI."""

    def _fn(self, program):
        import types
        from src.orchestrator.viewports.gp7_app_qt import GP7AppQt
        stub = types.SimpleNamespace(_program=program)
        return GP7AppQt._setspeed_v_effective.__get__(stub)

    def test_movj_only_after_setspeed_is_not_effective(self):
        prog = [Instruction(type="SetSpeed", speed_joint_pct=20.0,
                            speed_linear_mm_s=100.0),
                Instruction(type="MoveJ", target_name="P6")]
        assert self._fn(prog)(0) is False

    def test_movl_after_setspeed_is_effective(self):
        prog = [Instruction(type="SetSpeed", speed_joint_pct=20.0,
                            speed_linear_mm_s=150.0),
                Instruction(type="MoveJ", target_name="A"),
                Instruction(type="MoveL", target_name="B")]
        assert self._fn(prog)(0) is True

    def test_movc_after_setspeed_is_effective(self):
        prog = [Instruction(type="SetSpeed"),
                Instruction(type="MoveC")]
        assert self._fn(prog)(0) is True

    def test_next_setspeed_stops_the_scan(self):
        # The MOVL belongs to the SECOND SetSpeed, not the first.
        prog = [Instruction(type="SetSpeed"),
                Instruction(type="MoveJ", target_name="A"),
                Instruction(type="SetSpeed"),
                Instruction(type="MoveL", target_name="B")]
        assert self._fn(prog)(0) is False
        assert self._fn(prog)(2) is True

    def test_trailing_setspeed_not_effective(self):
        prog = [Instruction(type="MoveL", target_name="A"),
                Instruction(type="SetSpeed")]
        assert self._fn(prog)(1) is False


class TestPostProcessorJsonRoundTrip:
    """Post-processor settings (max VJ cap + default VJ/V) must persist through a
    Save -> Load .json round-trip and be reflected in the dirty-check signature.
    Save/Load are exercised headlessly via a stub host carrying the same attrs."""

    def _host(self):
        import json, types
        from src.orchestrator.viewports.mixin_program_io import ProgramIOMixin
        from src.orchestrator.viewports.gp7_app_qt import GP7AppQt
        h = types.SimpleNamespace(
            _active_job="MAIN",
            _jobs={"MAIN": [Instruction(type="MoveJ", target_name="P1")]},
            _targets={"P1": {"joints": [0, 0, 0, 0, 0, 0],
                             "tcp_pose": [0, 0, 0, 0, 0, 0]}},
            _job_folders={}, _jbi_raw={}, _jbi_positions={},
            _pp_max_speed_pct=30.0, _pp_default_vj=10.0, _pp_default_v_mms=100.0,
            _saved_signature="",
            _set_status=lambda *a, **k: None,
            _refresh_job_combo=lambda: None,
            _refresh_program_list=lambda: None,
            _refresh_target_list=lambda: None,
        )
        h._project_signature = GP7AppQt._project_signature.__get__(h)
        h._on_prog_save_dlg = ProgramIOMixin._on_prog_save_dlg.__get__(h)
        h._load_program_file = ProgramIOMixin._load_program_file.__get__(h)
        return h

    def test_pp_settings_survive_save_load(self, tmp_path, monkeypatch):
        from src.orchestrator.viewports import mixin_program_io as mio
        h = self._host()
        h._pp_max_speed_pct = 25.0
        h._pp_default_vj = 15.0
        h._pp_default_v_mms = 180.0
        out = tmp_path / "proj.json"
        monkeypatch.setattr(mio.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (str(out), "")))
        h._on_prog_save_dlg()
        assert out.exists()
        # Fresh host at defaults → load must overwrite with saved values.
        h2 = self._host()
        assert h2._load_program_file(str(out)) is True
        assert h2._pp_max_speed_pct == 25.0
        assert h2._pp_default_vj == 15.0
        assert h2._pp_default_v_mms == 180.0

    def test_pp_change_alters_signature(self):
        h = self._host()
        sig0 = h._project_signature()
        h._pp_default_v_mms = 222.0
        assert h._project_signature() != sig0

    def test_missing_pp_block_keeps_defaults(self, tmp_path):
        # Older v2 project (no post_processor) → current values unchanged.
        import json
        f = tmp_path / "old.json"
        f.write_text(json.dumps({"version": 2, "targets": {}, "program": []}),
                     encoding="utf-8")
        h = self._host()
        h._pp_default_vj = 12.0
        assert h._load_program_file(str(f)) is True
        assert h._pp_default_vj == 12.0


class TestFullProjectJsonRoundTrip:
    """Save -> Load .json restores EVERY project component: all jobs (with all 28
    instruction types + their fields), targets (+jbi_token), job_folders, jbi_raw,
    jbi_positions, post_processor, active_job — and the loaded project is 'clean'
    (signature == saved → no spurious unsaved-changes prompt). Run headlessly via a
    stub host that binds the real save/load/signature methods."""

    def _instructions(self):
        """One instruction of every serializable type (must stay == 28)."""
        return [
            Instruction(type="MoveJ", target_name="P1"),
            Instruction(type="MoveL", tcp_pose=[1, 2, 3, 4, 5, 6], speed_var="I003"),
            Instruction(type="MoveC", tcp_pose_mid=[1, 1, 1, 0, 0, 0],
                        tcp_pose=[2, 2, 2, 0, 0, 0]),
            Instruction(type="SetSpeed", speed_joint_pct=22.0, speed_linear_mm_s=133.0),
            Instruction(type="SetRounding", rounding_pl=3),
            Instruction(type="SetTool", tool_no=2),
            Instruction(type="SetRefFrame", ref_frame_no=4),
            Instruction(type="SetGripper", gripper_close=True),
            Instruction(type="SetDO", do_index=10, do_state=False),
            Instruction(type="PulseDO", do_index=5),
            Instruction(type="Wait", wait_seconds=0.3),
            Instruction(type="WaitIO", io_index=7, io_state=True, io_timeout_s=2.0),
            Instruction(type="ShowMessage", message="hello"),
            Instruction(type="CallJob", job_name="SUB"),
            Instruction(type="SimEvent", event_name="cp", event_payload="x"),
            Instruction(type="Label", label_name="LOOP"),
            Instruction(type="Jump", label_name="LOOP",
                        cond_lhs="B000", cond_op=">", cond_rhs="5"),
            Instruction(type="SetVar", var_name="B000", var_op="ADD", var_arg="1"),
            Instruction(type="ClearVar", var_name="I010", clear_count=3),
            Instruction(type="ReadGroupIn", var_name="B005", io_group=1,
                        io_group_kind="IG"),
            Instruction(type="WriteGroupOut", var_name="B005", io_group=2,
                        io_group_kind="OG"),
            Instruction(type="IfThen", cond_terms=[("B1", "<>", "1"),
                                                   ("B2", "<>", "2")],
                        cond_join="AND", cond_exp=True),
            Instruction(type="ElseIf", cond_lhs="B1", cond_op="=", cond_rhs="0"),
            Instruction(type="Else"),
            Instruction(type="While", cond_lhs="B3", cond_op="<", cond_rhs="9",
                        cond_exp=True),
            Instruction(type="EndWhile"),
            Instruction(type="EndIf"),
            Instruction(type="ClearStack"),
        ]

    def _host(self, **over):
        import types
        from src.orchestrator.viewports.mixin_program_io import ProgramIOMixin
        from src.orchestrator.viewports.gp7_app_qt import GP7AppQt
        h = types.SimpleNamespace(
            _set_status=lambda *a, **k: None,
            _refresh_job_combo=lambda: None,
            _refresh_program_list=lambda: None,
            _refresh_target_list=lambda: None,
            _saved_signature="",
            _pp_max_speed_pct=30.0, _pp_default_vj=10.0, _pp_default_v_mms=100.0,
            _job_folders={}, _jbi_raw={}, _jbi_positions={},
            _active_job="MAIN", _jobs={"MAIN": []}, _targets={})
        for k, v in over.items():
            setattr(h, k, v)
        h._project_signature = GP7AppQt._project_signature.__get__(h)
        h._on_prog_save_dlg = ProgramIOMixin._on_prog_save_dlg.__get__(h)
        h._load_program_file = ProgramIOMixin._load_program_file.__get__(h)
        return h

    def test_covers_all_28_instruction_types(self):
        # Guard: if a new type is added to the model, extend _instructions() too.
        assert len({i.type for i in self._instructions()}) == 28

    def test_full_project_survives_save_load(self, tmp_path, monkeypatch):
        from src.orchestrator.viewports import mixin_program_io as mio
        ins = self._instructions()
        src = self._host(
            _active_job="SUB",
            _jobs={"MAIN": ins,
                   "SUB": [Instruction(type="MoveJ", joints=[0, 0, 0, 0, 0, 0])]},
            _targets={"P1": {"joints": [-38, 16, 13, 1, -66, 167],
                             "tcp_pose": [10, 20, 30, 0, 90, 0],
                             "jbi_token": "P00001"}},
            _job_folders={"MAIN": "FOLDER_A", "SUB": "FOLDER_B"},
            _jbi_raw={"SUB": "NOP END"},
            _jbi_positions={"P10": [1, 2, 3, 4, 5, 6]},
            _pp_max_speed_pct=25.0, _pp_default_vj=15.0, _pp_default_v_mms=180.0)
        out = tmp_path / "full.json"
        monkeypatch.setattr(mio.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (str(out), "")))
        src._on_prog_save_dlg()
        assert out.exists()

        dst = self._host()
        assert dst._load_program_file(str(out)) is True

        def jd(h):
            return {n: [i.to_dict() for i in p] for n, p in h._jobs.items()}
        assert jd(dst) == jd(src)                 # every job + every instruction field
        assert dst._active_job == src._active_job
        assert dst._targets == src._targets       # incl. jbi_token
        assert dst._job_folders == src._job_folders
        assert dst._jbi_raw == src._jbi_raw
        assert dst._jbi_positions == src._jbi_positions
        assert (dst._pp_max_speed_pct, dst._pp_default_vj,
                dst._pp_default_v_mms) == (25.0, 15.0, 180.0)
        # Loaded project must be clean → no false "unsaved changes" prompt.
        assert dst._project_signature() == dst._saved_signature


class TestRunOnRobotCollectsSubJobs:
    """Run on Robot must upload the active job + ALL sub-jobs reachable via
    CALL JOB (so the controller resolves them), and flag CALL targets missing
    from the project. Tests the pure collection/naming logic headlessly."""

    def _host(self, jobs, active, jbi_raw=None):
        import types
        from src.orchestrator.viewports.gp7_app_qt import GP7AppQt
        h = types.SimpleNamespace(_jobs=jobs, _active_job=active,
                                  _jbi_raw=jbi_raw or {})
        h._safe_job_name = GP7AppQt._safe_job_name           # staticmethod
        h._resolve_job_key = GP7AppQt._resolve_job_key.__get__(h)
        h._collect_run_jobs = GP7AppQt._collect_run_jobs.__get__(h)
        h._robot_job_name = GP7AppQt._robot_job_name.__get__(h)
        return h

    def _call(self, name):
        return Instruction(type="CallJob", job_name=name)

    def test_collects_nested_subjobs_bfs(self):
        jobs = {
            "MAIN": [self._call("SUB1"), self._call("SUB2")],
            "SUB1": [self._call("SUB3")],
            "SUB2": [], "SUB3": [],
        }
        order, missing = self._host(jobs, "MAIN")._collect_run_jobs()
        assert order[0] == "MAIN"                 # main job first (JOB_SELECT target)
        assert set(order) == {"MAIN", "SUB1", "SUB2", "SUB3"}
        assert missing == []

    def test_flags_missing_call_target(self):
        jobs = {"MAIN": [self._call("PP1")]}      # PP1 not in project
        order, missing = self._host(jobs, "MAIN")._collect_run_jobs()
        assert order == ["MAIN"]
        assert missing == ["PP1"]

    def test_resolves_subjob_by_sanitized_key(self):
        # Reproduces the REAL importer keying: `CALL JOB:SPEED-1` is stored under
        # the sanitized key "SPEED1" (dash stripped). _collect_run_jobs must
        # resolve the raw CALL literal to that key — NOT report it missing.
        jobs = {"MAIN": [self._call("SPEED-1"), self._call("POS-DATA")],
                "SPEED1": [], "POSDATA": []}      # keys as the importer makes them
        order, missing = self._host(jobs, "MAIN")._collect_run_jobs()
        assert set(order) == {"MAIN", "SPEED1", "POSDATA"}
        assert missing == []                      # was ["SPEED-1", "POS-DATA"] before fix

    def test_truly_missing_dashed_target_still_flagged(self):
        jobs = {"MAIN": [self._call("NO-SUCH")]}  # neither raw nor sanitized exists
        order, missing = self._host(jobs, "MAIN")._collect_run_jobs()
        assert order == ["MAIN"]
        assert missing == ["NO-SUCH"]             # raw literal reported to the user

    def test_cycle_does_not_loop_forever(self):
        jobs = {"MAIN": [self._call("SUB")], "SUB": [self._call("MAIN")]}
        order, missing = self._host(jobs, "MAIN")._collect_run_jobs()
        assert order == ["MAIN", "SUB"]           # deduped, terminates
        assert missing == []

    def test_robot_job_name_prefers_original_with_dash(self):
        # Imported job keeps its //NAME (dash preserved) so CALL JOB:SPEED-1 resolves.
        h = self._host({"SPEED-1": []}, "SPEED-1",
                       jbi_raw={"SPEED-1": {"name": "SPEED-1"}})
        assert h._robot_job_name("SPEED-1") == "SPEED-1"

    def test_robot_job_name_falls_back_to_sanitized(self):
        h = self._host({"WELD_A": []}, "WELD_A")
        assert h._robot_job_name("WELD_A") == "WELD_A"
