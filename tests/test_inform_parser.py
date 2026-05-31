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
