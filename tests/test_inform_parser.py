"""
test_inform_parser.py
─────────────────────
Verify parser INFORM .JBI (chiều ngược của inform_codegen).

PURE TEXT — không cần YRC1000/app/Qt. Trọng tâm: round-trip codegen→parse giữ
nguyên joints (pulse int → deg không drift) + motion kind + modifiers, và xử lý
an toàn lệnh ngoài subset / P-var / padding 8-axis.
"""
from __future__ import annotations

import pytest

from src.orchestrator.backends.hse_protocol import GP7_PULSE_PER_DEG
from src.orchestrator.backends.inform_codegen import (
    InformJobBuilder,
    gen_pick_place_job,
    gen_pvar_template_job,
)
from src.orchestrator.backends.inform_parser import parse_jbi


class TestRoundTrip:
    def test_minimal_movj(self):
        text = (InformJobBuilder(name="MIN", max_speed_pct=10.0)
                .add_position("home", [0] * 6).movj("home")).render()
        pj = parse_jbi(text)
        assert pj.name == "MIN"
        assert pj.pos_type == "PULSE"
        assert len(pj.instructions) == 1
        i = pj.instructions[0]
        assert i.kind == "movj"
        assert i.joints_deg == [0.0] * 6
        assert i.vj_pct == 10.0
        assert pj.warnings == []

    def test_joints_preserved_exactly(self):
        """Pulse là int → chia ngược về deg phải khớp (codegen round() → /ratio)."""
        joints = [10.0, -5.0, 20.0, 0.0, 30.0, -15.0]
        text = (InformJobBuilder(name="J").add_position("p", joints)
                .movj("p", speed_pct=15.0)).render()
        pj = parse_jbi(text)
        got = pj.instructions[0].joints_deg
        # Pulse round-trip: |deg - deg'| < 1 pulse / ratio
        for a, b, r in zip(joints, got, GP7_PULSE_PER_DEG):
            assert abs(a - b) < 1.0 / r

    def test_pulse_reexport_is_stable(self):
        """deg→pulse→deg→pulse phải cho pulse y hệt (no drift qua nhiều vòng)."""
        joints = [12.3, -7.7, 18.4, 5.1, -22.9, 44.4]
        t1 = (InformJobBuilder(name="J").add_position("p", joints)
              .movj("p")).render()
        deg2 = parse_jbi(t1).instructions[0].joints_deg
        t2 = (InformJobBuilder(name="J").add_position("p", deg2)
              .movj("p")).render()
        # C-var line (pulses) phải identical giữa 2 lần render.
        line1 = [ln for ln in t1.splitlines() if ln.startswith("C00000=")][0]
        line2 = [ln for ln in t2.splitlines() if ln.startswith("C00000=")][0]
        assert line1 == line2

    def test_modifiers_parsed(self):
        text = (InformJobBuilder(name="J").add_position("p", [0] * 6)
                .movj("p", speed_pct=12.0, pl=3, tool_no=2, user_frame=5)).render()
        i = parse_jbi(text).instructions[0]
        assert i.vj_pct == 12.0 and i.pl == 3 and i.tool_no == 2
        assert i.user_frame == 5

    def test_movl_speed(self):
        text = (InformJobBuilder(name="J").add_position("p", [0] * 6)
                .movl("p", speed_mm_s=80.0)).render()
        i = parse_jbi(text).instructions[0]
        assert i.kind == "movl" and i.v_mm_s == 80.0

    def test_io_and_timing(self):
        text = (InformJobBuilder(name="J").add_position("p", [0] * 6).movj("p")
                .dout(2, True).timer(0.5)
                .wait_in(3, on=False, timeout_s=7.0)
                .msg("HI THERE").call_job("SUB")).render()
        kinds = [i.kind for i in parse_jbi(text).instructions]
        assert kinds == ["movj", "dout", "timer", "wait_in", "msg", "call"]
        ins = {i.kind: i for i in parse_jbi(text).instructions}
        assert ins["dout"].do_index == 2 and ins["dout"].do_on is True
        assert ins["timer"].seconds == 0.5
        assert ins["wait_in"].in_index == 3 and ins["wait_in"].in_on is False
        assert ins["wait_in"].in_timeout_s == 7.0
        assert ins["msg"].text == "HI THERE"
        assert ins["call"].text == "SUB"

    def test_full_pick_place_roundtrip(self):
        text = gen_pick_place_job(
            name="PICK", home_deg=[0] * 6, approach_deg=[10, -5, 20, 0, 30, -15],
            grasp_deg=[10, -10, 25, 0, 30, -15],
            transfer_deg=[20, -5, 20, 0, 30, -15],
            place_deg=[20, -10, 25, 0, 30, -15],
            gripper_do_index=2, gripper_delay_s=0.5, speed_pct=15.0)
        pj = parse_jbi(text)
        assert pj.name == "PICK"
        # 8 MOVJ + 2 DOUT + 2 TIMER trong template pick-place.
        kinds = [i.kind for i in pj.instructions]
        assert kinds.count("movj") == 8
        assert kinds.count("dout") == 2
        assert kinds.count("timer") == 2
        assert pj.warnings == []


class TestRobustness:
    def test_8axis_padding_truncated(self):
        """File emit 8 giá trị (pad axis 7-8) → parser chỉ lấy 6 axis đầu."""
        text = (InformJobBuilder(name="J", emit_axis_count=8)
                .add_position("p", [10, 0, 0, 0, 0, 0]).movj("p")).render()
        i = parse_jbi(text).instructions[0]
        assert len(i.joints_deg) == 6
        assert abs(i.joints_deg[0] - 10.0) < 0.01

    def test_pvar_template_skipped_with_warning(self):
        """P-var (runtime) không có //POS → motion bị bỏ qua + warning."""
        text = gen_pvar_template_job("PV", num_positions=2)
        pj = parse_jbi(text)
        assert all(i.kind != "movj" for i in pj.instructions if i.joints_deg)
        assert any("P-var" in w or "không có trong //POS" in w
                   for w in pj.warnings)

    def test_pvar_declared_in_pos_resolved(self):
        """P-var KHAI BÁO trong //POS (job teach-pendant thật) phải resolve được,
        kể cả khi //POS dùng 5 chữ số (P00001) còn //INST dùng 3 chữ số (P001).
        Regression: trước đây parser bỏ //POS P-var → mất hết motion."""
        text = (
            "/JOB\r\n//NAME PP1\r\n//POS\r\n///NPOS 0,0,0,2,0,0\r\n"
            "///TOOL 0\r\n///POSTYPE PULSE\r\n///PULSE\r\n"
            "P00001=55242,26922,11446,1601,-60437,64378\r\n"
            "P00002=55242,35623,-7881,1830,-43902,63991\r\n"
            "//INST\r\n///DATE 2025/12/05 13:12\r\n///ATTR SC,RW\r\n"
            "///GROUP1 RB1\r\nNOP\r\n"
            "MOVJ P001 VJ=20.00 PL=0\r\nMOVL P002 V=80.0 PL=0\r\nEND\r\n")
        pj = parse_jbi(text)
        assert [i.kind for i in pj.instructions] == ["movj", "movl"]
        assert pj.warnings == []
        assert len(pj.instructions[0].joints_deg) == 6
        # P001 (//INST) resolves to P00001 (//POS) — padding-independent.
        # 55242 pulses / GP7 axis-1 ratio ≈ 41.2°.
        assert abs(pj.instructions[0].joints_deg[0] - 41.2) < 1.0
        # pos_key exposes the canonical key for shared-target mapping.
        assert pj.instructions[0].pos_key == "P1"
        assert pj.instructions[1].pos_key == "P2"

    def test_pos_key_shared_for_reused_point(self):
        """Motions reusing the same //POS var get the SAME pos_key (so the
        importer can map them to one shared target)."""
        text = (
            "/JOB\r\n//NAME R\r\n//POS\r\n///NPOS 0,0,0,2,0,0\r\n"
            "///TOOL 0\r\n///POSTYPE PULSE\r\n///PULSE\r\n"
            "P00001=1,2,3,4,5,6\r\nP00002=7,8,9,10,11,12\r\n"
            "//INST\r\n///DATE 2026/01/01 00:00\r\n///ATTR SC,RW\r\n"
            "///GROUP1 RB1\r\nNOP\r\n"
            "MOVJ P001 VJ=20.00\r\nMOVL P002 V=80.0\r\nMOVL P001 V=80.0\r\nEND\r\n")
        pj = parse_jbi(text)
        keys = [i.pos_key for i in pj.instructions if i.kind in ("movj", "movl")]
        assert keys == ["P1", "P2", "P1"]      # 1st and 3rd reuse the same point

    def test_ifthenexp_parsed_as_ifthen(self):
        """IFTHENEXP (expression form) + IN#(n)=ON condition must parse, keeping
        the IF block balanced (regression: IFTHENEXP was skipped → orphan ENDIF)."""
        text = (
            "/JOB\r\n//NAME M\r\n//POS\r\n///NPOS 0,0,0,1,0,0\r\n///TOOL 0\r\n"
            "///POSTYPE PULSE\r\n///PULSE\r\nP00006=0,0,0,0,0,0\r\n//INST\r\n"
            "///DATE 2026/01/01 00:00\r\n///ATTR SC,RW\r\n///GROUP1 RB1\r\nNOP\r\n"
            "*RESET\r\nMOVJ P006 VJ=20.00\r\nIFTHENEXP IN#(8)=ON\r\n"
            "\tCALL JOB:PP1\r\nENDIF\r\nJUMP *RESET\r\nEND\r\n")
        pj = parse_jbi(text)
        kinds = [i.kind for i in pj.instructions]
        assert "ifthen" in kinds and "endif" in kinds
        assert pj.warnings == []
        ift = next(i for i in pj.instructions if i.kind == "ifthen")
        # ON/OFF kept verbatim so export reproduces 'IFTHENEXP IN#(8)=ON'.
        assert (ift.cond_lhs, ift.cond_op, ift.cond_rhs) == ("IN#(8)", "=", "ON")
        assert ift.cond_exp is True              # remembers the EXP keyword form

    def test_cond_exp_flag_distinguishes_ifthen_vs_ifthenexp(self):
        """IFTHENEXP on a VARIABLE condition keeps cond_exp=True; plain IFTHEN
        keeps False — so re-synthesis reproduces the original keyword."""
        text = (
            "/JOB\r\n//NAME J\r\n//POS\r\n///NPOS 0,0,0,0,0,0\r\n//INST\r\n"
            "///DATE 2026/01/01 00:00\r\n///ATTR SC,RW\r\n///GROUP1 RB1\r\nNOP\r\n"
            "IFTHENEXP I010=12\r\nSET B000 1\r\nENDIF\r\n"
            "IFTHEN B000=1\r\nSET B001 1\r\nENDIF\r\nEND\r\n")
        ifs = [i for i in parse_jbi(text).instructions if i.kind == "ifthen"]
        assert ifs[0].cond_exp is True           # IFTHENEXP I010=12
        assert ifs[1].cond_exp is False          # IFTHEN B000=1

    def test_on_off_operands_kept_verbatim(self):
        text = (
            "/JOB\r\n//NAME M\r\n//POS\r\n///NPOS 1,0,0,0,0,0\r\n///TOOL 0\r\n"
            "///POSTYPE PULSE\r\n///PULSE\r\nC00000=0,0,0,0,0,0\r\n//INST\r\n"
            "///DATE 2026/01/01 00:00\r\n///ATTR SC,RW\r\n///GROUP1 RB1\r\nNOP\r\n"
            "MOVJ C00000 VJ=10.00\r\nWHILEEXP IN#(3)=OFF\r\nENDWHILE\r\nEND\r\n")
        pj = parse_jbi(text)
        wh = next(i for i in pj.instructions if i.kind == "while")
        assert (wh.cond_lhs, wh.cond_op, wh.cond_rhs) == ("IN#(3)", "=", "OFF")

    def test_foldername_parsed(self):
        """///FOLDERNAME (controller folder metadata) được giữ vào ParsedJob."""
        text = (
            "/JOB\r\n//NAME PP1\r\n///FOLDERNAME FOLDER001\r\n//POS\r\n"
            "///NPOS 1,0,0,0,0,0\r\n///TOOL 0\r\n///POSTYPE PULSE\r\n///PULSE\r\n"
            "C00000=0,0,0,0,0,0\r\n//INST\r\n///DATE 2026/01/01 00:00\r\n"
            "///ATTR SC,RW\r\n///GROUP1 RB1\r\nNOP\r\nMOVJ C00000 VJ=10.00\r\nEND\r\n")
        pj = parse_jbi(text)
        assert pj.folder_name == "FOLDER001"

    def test_foldername_absent_is_empty(self):
        text = (
            "/JOB\r\n//NAME J\r\n//POS\r\n///NPOS 1,0,0,0,0,0\r\n///TOOL 0\r\n"
            "///POSTYPE PULSE\r\n///PULSE\r\nC00000=0,0,0,0,0,0\r\n//INST\r\n"
            "///DATE 2026/01/01 00:00\r\n///ATTR SC,RW\r\n///GROUP1 RB1\r\n"
            "NOP\r\nMOVJ C00000 VJ=10.00\r\nEND\r\n")
        assert parse_jbi(text).folder_name == ""

    def test_lg_production_jobs_parse_clean(self):
        """The real LG job set (4 files: extended INFORM — P[B], ANDEXP, EXPRESS,
        DIN/CLEAR/PULSE/COMM, var speeds) must parse with ZERO warnings."""
        import os
        base = os.path.join(os.path.dirname(__file__), "..", "examples", "LG")
        if not os.path.isdir(base):
            pytest.skip("examples/LG not present")
        for fn in ("MATER.JBI", "HOME.JBI", "SPEED-1.JBI", "POS-DATA.JBI"):
            p = os.path.join(base, fn)
            pj = parse_jbi(open(p, encoding="utf-8").read())
            assert pj.warnings == [], f"{fn}: {pj.warnings}"

    def test_compound_condition_andexp(self):
        text = (
            "/JOB\r\n//NAME J\r\n//POS\r\n///NPOS 0,0,0,0,0,0\r\n//INST\r\n"
            "///DATE 2026/01/01 00:00\r\n///ATTR SC,RW\r\n///GROUP1 RB1\r\nNOP\r\n"
            "IFTHENEXP I010<>11 ANDEXP B010<>12\r\nSET B000 1\r\nENDIF\r\nEND\r\n")
        pj = parse_jbi(text)
        ift = next(i for i in pj.instructions if i.kind == "ifthen")
        assert ift.cond_terms == [("I010", "<>", "11"), ("B010", "<>", "12")]
        assert ift.cond_join == "AND"

    def test_set_express_parsed(self):
        text = (
            "/JOB\r\n//NAME J\r\n//POS\r\n///NPOS 0,0,0,0,0,0\r\n//INST\r\n"
            "///DATE 2026/01/01 00:00\r\n///ATTR SC,RW\r\n///GROUP1 RB1\r\nNOP\r\n"
            "SET I000  EXPRESS 5 * B005\r\nEND\r\n")
        sv = next(i for i in parse_jbi(text).instructions if i.kind == "setvar")
        assert sv.var_name == "I000" and sv.var_expr == "5 * B005"

    def test_indirect_pos_and_var_speed(self):
        text = (
            "/JOB\r\n//NAME J\r\n//POS\r\n///NPOS 0,0,0,1,0,0\r\n///TOOL 0\r\n"
            "///POSTYPE PULSE\r\n///PULSE\r\nP00010=1,2,3,4,5,6\r\n//INST\r\n"
            "///DATE 2026/01/01 00:00\r\n///ATTR SC,RW\r\n///GROUP1 RB1\r\nNOP\r\n"
            "MOVJ P[B010] VJ=I002\r\nEND\r\n")
        pj = parse_jbi(text)
        mv = next(i for i in pj.instructions if i.kind == "movj")
        assert mv.pos_index_var == "B010"
        assert mv.speed_var == "I002"
        assert "P10" in pj.positions          # //POS table carried for indirect

    def test_comm_clear_pulse_din_parse(self):
        text = (
            "/JOB\r\n//NAME J\r\n//POS\r\n///NPOS 0,0,0,0,0,0\r\n//INST\r\n"
            "///DATE 2026/01/01 00:00\r\n///ATTR SC,RW\r\n///GROUP1 RB1\r\nNOP\r\n"
            "CLEAR STACK\r\nCLEAR I010 2\r\nPULSE OT#(6)\r\n"
            "DIN B005 IG#(2)\r\nDOUT OG#(2) B005\r\n"
            "COMM IFTHENEXP IN#(6)=OFF\r\nCOMM \t MOVJ P010 VJ=15.00\r\n"
            "COMM ENDIF\r\nEND\r\n")
        pj = parse_jbi(text)
        kinds = [i.kind for i in pj.instructions]
        assert kinds == ["clear_stack", "clear", "pulse", "din", "dout_group"]
        assert pj.warnings == []               # COMM lines dropped silently

    def test_unknown_command_warns_not_raises(self):
        # MOVS (spline) + ARCON (welding) nằm ngoài subset hỗ trợ → warning.
        text = (
            "/JOB\r\n//NAME J\r\n//POS\r\n///NPOS 1,0,0,0,0,0\r\n"
            "///TOOL 0\r\n///POSTYPE PULSE\r\n///PULSE\r\nC00000=0,0,0,0,0,0\r\n"
            "//INST\r\n///DATE 2026/01/01 00:00\r\n///ATTR SC,RW\r\n"
            "///GROUP1 RB1\r\nNOP\r\nMOVJ C00000 VJ=10.00\r\n"
            "MOVS C00000 V=100.0\r\nARCON ASF#(1)\r\nEND\r\n")
        pj = parse_jbi(text)
        assert [i.kind for i in pj.instructions] == ["movj"]
        assert len(pj.warnings) == 2          # MOVS + ARCON skipped

    def test_comment_lines_dropped(self):
        text = (InformJobBuilder(name="J").add_position("p", [0] * 6)
                .comment("hello").movj("p")).render()
        pj = parse_jbi(text)
        assert [i.kind for i in pj.instructions] == ["movj"]

    def test_lf_line_endings_ok(self):
        text = (InformJobBuilder(name="J").add_position("p", [0] * 6)
                .movj("p")).render().replace("\r\n", "\n")
        pj = parse_jbi(text)
        assert len(pj.instructions) == 1

    def test_not_inform_raises(self):
        with pytest.raises(ValueError, match="/JOB"):
            parse_jbi("just some random text\nnot a job")


class TestLogicRoundTrip:
    def test_label_jump_setvar(self):
        b = InformJobBuilder(name="J")
        b.add_position("p", [0] * 6)
        b.set_var("B000", "SET", 0).label("LOOP").set_var("B000", "INC")
        b.movj("p").jump("LOOP", cond=("B000", "<", "3"))
        pj = parse_jbi(b.render())
        kinds = [i.kind for i in pj.instructions]
        assert kinds == ["setvar", "label", "setvar", "movj", "jump"]
        jmp = pj.instructions[-1]
        assert jmp.label_name == "LOOP"
        assert (jmp.cond_lhs, jmp.cond_op, jmp.cond_rhs) == ("B000", "<", "3")
        sv = pj.instructions[0]
        assert sv.var_op == "SET" and sv.var_name == "B000" and sv.var_arg == "0"
        assert pj.warnings == []

    def test_unconditional_jump(self):
        b = InformJobBuilder(name="J")
        b.add_position("p", [0] * 6).movj("p").jump("DONE")
        jmp = [i for i in parse_jbi(b.render()).instructions if i.kind == "jump"][0]
        assert jmp.label_name == "DONE" and jmp.cond_op == ""

    def test_structured_if_while(self):
        b = InformJobBuilder(name="J")
        b.add_position("p", [0] * 6).movj("p")
        b.if_then(("B000", "=", "1")).msg("HIT").else_if(("B000", ">", "1"))
        b.msg("BIG").else_().msg("MISS").end_if()
        b.while_(("B001", "<>", "0")).set_var("B001", "DEC").end_while()
        kinds = [i.kind for i in parse_jbi(b.render()).instructions]
        assert kinds == ["movj", "ifthen", "msg", "elseif", "msg", "else",
                         "msg", "endif", "while", "setvar", "endwhile"]

    @pytest.mark.parametrize("op", ["=", "<>", ">", "<", ">=", "<="])
    def test_all_condition_ops_parse(self, op):
        b = InformJobBuilder(name="J")
        b.add_position("p", [0] * 6).movj("p").jump("L", cond=("B000", op, "5"))
        b.label("L")
        jmp = [i for i in parse_jbi(b.render()).instructions if i.kind == "jump"][0]
        assert jmp.cond_op == op


class TestExampleJobsPipeline:
    """End-to-end gate: every bundled example .JBI parses, validates, describes,
    dict-round-trips, and (for non-inline-MOVL jobs) exports + re-parses clean."""

    def test_all_examples_full_pipeline(self):
        import glob, os, tempfile, types
        from pathlib import Path
        from src.orchestrator.viewports.mixin_program_io import ProgramIOMixin
        from src.orchestrator.viewports.mixin_job_target import JobTargetMixin
        from src.orchestrator.viewports.program_logic import (
            validate_program, build_block_map, resolve_labels)
        from src.orchestrator.viewports.program_model import Instruction

        base = os.path.join(os.path.dirname(__file__), "..", "examples")
        files = sorted(glob.glob(os.path.join(base, "**", "*.JBI"), recursive=True))
        if not files:
            pytest.skip("no example .JBI files")
        d = Path(tempfile.mkdtemp())
        for f in files:
            s = types.SimpleNamespace(
                _model=None, _tool_frames=[("f", None)], _tool_idx=0, _targets={},
                _jbi_raw={}, _jbi_positions={}, _pp_max_speed_pct=100.0,
                _pp_default_vj=20.0, _pp_default_v_mms=80.0)
            s._safe_target_name = JobTargetMixin._safe_target_name
            s._safe_job_name = JobTargetMixin._safe_job_name
            s._job_signature = ProgramIOMixin._job_signature
            imp = ProgramIOMixin._jbi_to_instructions.__get__(s)
            exp = ProgramIOMixin._export_job_to_path.__get__(s)
            pj = parse_jbi(Path(f).read_text(encoding="utf-8"))
            assert pj.warnings == [], f"{f}: parse warnings {pj.warnings}"
            targets = {}
            prog = imp(pj, targets)
            s._targets = targets
            assert validate_program(prog) == [], f"{f}: validate errors"
            build_block_map(prog); resolve_labels(prog)        # raises if unbalanced
            for ins in prog:
                assert not ins.describe().startswith("?"), f"{f}: {ins.type} desc"
                assert Instruction.from_dict(ins.to_dict()).to_dict() == ins.to_dict()
            # Export + re-parse for jobs without inline (cartesian) MOVL.
            if not any(i.type == "MoveL" and not i.target_name
                       and not i.pos_index_var for i in prog):
                op = d / os.path.basename(f)
                exp(prog, "T" + (s._safe_job_name(pj.name) or "J"), op)
                assert parse_jbi(op.read_text(encoding="utf-8")).warnings == [], \
                    f"{f}: re-parse warnings after export"
