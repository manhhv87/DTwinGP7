"""
test_inform_codegen.py
──────────────────────
Verify INFORM .JBI codegen sinh đúng format Yaskawa.

KHÔNG cần YRC1000 — pure text comparison + snapshot matching.
"""
from __future__ import annotations

import pytest

from src.orchestrator.backends.hse_protocol import GP7_PULSE_PER_DEG
from src.orchestrator.backends.inform_codegen import (
    MAX_POSITIONS_PER_JOB,
    InformJobBuilder,
    gen_pick_place_job,
)


# ─────────────────────────────────────────────────────────────────────────
# Builder validation
# ─────────────────────────────────────────────────────────────────────────


class TestValidation:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="1-32 characters"):
            InformJobBuilder(name="")

    def test_long_name_raises(self):
        with pytest.raises(ValueError, match="1-32 characters"):
            InformJobBuilder(name="x" * 50)

    def test_name_with_special_chars_raises(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            InformJobBuilder(name="bad name!")

    def test_name_with_underscore_allowed(self):
        InformJobBuilder(name="PICK_PLACE_01")          # OK

    def test_non_ascii_name_rejected(self):
        # str.isalnum() accepts Vietnamese letters, but the .JBI uploads
        # ASCII-strict → must reject early with a clear error, not crash on FTP.
        with pytest.raises(ValueError, match="ASCII"):
            InformJobBuilder(name="Gắp")

    def test_non_ascii_comment_and_msg_transliterated(self):
        """Vietnamese comment/SimEvent/MSG must transliterate to ASCII so the
        whole job stays uploadable (upload_job encodes ascii-strict)."""
        b = InformJobBuilder(name="J")
        b.add_position("p", [0] * 6)
        b.comment("Đóng kẹp rồi nhấc vật")
        b.msg("Gắp vật thành công")
        b.movj("p")
        text = b.render()
        text.encode("ascii", "strict")                  # must not raise
        assert "'Dong kep roi nhac vat" in text
        assert 'MSG "Gap vat thanh cong"' in text

    def test_non_ascii_call_job_rejected(self):
        b = InformJobBuilder(name="J")
        with pytest.raises(ValueError, match="ASCII"):
            b.call_job("Gắp")

    def test_duplicate_position_raises(self):
        b = InformJobBuilder(name="J")
        b.add_position("p0", [0] * 6)
        with pytest.raises(ValueError, match="already exists"):
            b.add_position("p0", [1] * 6)

    def test_joint_count_mismatch_raises(self):
        b = InformJobBuilder(name="J")
        with pytest.raises(ValueError, match="must have 6 elements"):
            b.add_position("p0", [0, 0, 0])

    def test_movj_unknown_position_raises(self):
        b = InformJobBuilder(name="J")
        with pytest.raises(KeyError, match="not added"):
            b.movj("ghost")

    def test_too_many_positions_raises(self):
        b = InformJobBuilder(name="J")
        for i in range(MAX_POSITIONS_PER_JOB):
            b.add_position(f"p{i}", [0] * 6)
        with pytest.raises(ValueError, match="Reached limit"):
            b.add_position(f"p{MAX_POSITIONS_PER_JOB}", [0] * 6)

    def test_dout_index_out_of_range_raises(self):
        b = InformJobBuilder(name="J")
        b.add_position("p", [0] * 6)
        with pytest.raises(ValueError, match="DOUT index"):
            b.dout(0, True)
        with pytest.raises(ValueError, match="DOUT index"):
            b.dout(2000, True)

    def test_render_empty_raises(self):
        # A job must have at least 1 instruction. Positions are now OPTIONAL
        # (logic-only jobs like a master/speed-calc job have an empty //POS).
        b = InformJobBuilder(name="J")
        with pytest.raises(ValueError, match="at least 1 instruction"):
            b.render()

    def test_logic_only_job_empty_pos(self):
        """A job with instructions but no positions renders an empty //POS."""
        b = InformJobBuilder(name="MASTER")
        b.set_var("B000", "SET", "1")
        b.call_job("SUB")
        text = b.render()
        assert "///NPOS 0,0,0,0,0,0" in text
        assert "///POSTYPE" not in text         # no position section body
        assert "SET B000 1" in text and "CALL JOB:SUB" in text


# ─────────────────────────────────────────────────────────────────────────
# Pulse conversion
# ─────────────────────────────────────────────────────────────────────────


class TestPulseConversion:
    def test_degrees_converted_with_ratio(self):
        b = InformJobBuilder(name="J", pulse_per_deg=GP7_PULSE_PER_DEG)
        b.add_position("p", [10.0, -5.0, 20.0, 10.0, 5.0, 5.0])
        b.movj("p")
        text = b.render()
        # S axis 10° × 1241.212 = 12412 pulse
        assert "P00000=12412," in text
        # L axis -5° × 1517.037 = -7585 pulse
        assert ",-7585," in text

    def test_zero_joints_emits_zeros(self):
        b = InformJobBuilder(name="J")
        b.add_position("home", [0] * 6)
        b.movj("home")
        text = b.render()
        assert "P00000=0,0,0,0,0,0" in text


# ─────────────────────────────────────────────────────────────────────────
# Speed clamping
# ─────────────────────────────────────────────────────────────────────────


class TestSpeedClamp:
    def test_movj_default_uses_max_speed(self):
        b = InformJobBuilder(name="J", max_speed_pct=25.0)
        b.add_position("p", [0] * 6)
        b.movj("p")
        assert "VJ=25.00" in b.render()

    def test_movj_caps_at_max_speed(self):
        b = InformJobBuilder(name="J", max_speed_pct=20.0)
        b.add_position("p", [0] * 6)
        b.movj("p", speed_pct=80.0)
        assert "VJ=20.00" in b.render()           # capped

    def test_movj_below_max_kept(self):
        b = InformJobBuilder(name="J", max_speed_pct=50.0)
        b.add_position("p", [0] * 6)
        b.movj("p", speed_pct=10.0)
        assert "VJ=10.00" in b.render()

    def test_movl_uses_linear_speed_mm_s(self):
        b = InformJobBuilder(name="J")
        b.add_position("p", [0] * 6)
        b.movl("p", speed_mm_s=80.0)
        assert "MOVL P000 V=80.0" in b.render()


# ─────────────────────────────────────────────────────────────────────────
# P-variable positions (preserve original .JBI var kind + index on round-trip)
# ─────────────────────────────────────────────────────────────────────────


class TestPVarPositions:
    def test_pos_token_preserves_pvar_kind_and_index(self):
        """add_position(pos_token='P1') → //POS decl 'P00001' (5-digit),
        //INST ref 'P001' (3-digit), NPOS P-var slot (index 3)."""
        b = InformJobBuilder(name="J")
        b.add_position("a", [0] * 6, pos_token="P1")
        b.add_position("b", [0] * 6, pos_token="P2")
        b.movj("a"); b.movl("b")
        text = b.render()
        assert "///NPOS 0,0,0,2,0,0" in text     # 2 P-vars in slot 3
        assert "P00001=0,0,0,0,0,0" in text      # 5-digit declaration
        assert "P00002=0,0,0,0,0,0" in text
        assert "MOVJ P001 " in text              # 3-digit reference
        assert "MOVL P002 " in text

    def test_pos_token_cvar_still_default_5digit(self):
        b = InformJobBuilder(name="J")
        b.add_position("a", [0] * 6, pos_token="C3")
        b.movj("a")
        text = b.render()
        assert "///NPOS 1,0,0,0,0,0" in text      # C-var slot 0
        assert "C00003=0,0,0,0,0,0" in text
        assert "MOVJ C00003 " in text

    def test_pos_token_invalid_raises(self):
        b = InformJobBuilder(name="J")
        with pytest.raises(ValueError, match="Invalid pos_token"):
            b.add_position("a", [0] * 6, pos_token="X9")

    def test_foldername_emitted_after_name(self):
        b = InformJobBuilder(name="PP1", folder_name="FOLDER001")
        b.add_position("a", [0] * 6, pos_token="P1")
        b.movj("a")
        text = b.render()
        assert "//NAME PP1\r\n///FOLDERNAME FOLDER001\r\n//POS" in text

    def test_foldername_omitted_when_empty(self):
        b = InformJobBuilder(name="J")
        b.add_position("a", [0] * 6)
        b.movj("a")
        assert "FOLDERNAME" not in b.render()

    def test_io_condition_emits_exp_keyword(self):
        """I/O conditions (IN#/ON/OFF) must use the IFTHENEXP/WHILEEXP form;
        variable conditions stay plain IFTHEN/WHILE (Yaskawa convention)."""
        b = InformJobBuilder(name="J")
        b.add_position("a", [0] * 6, pos_token="P1")
        b.label("RESET")
        b.movj("a")
        b.if_then(("IN#(8)", "=", "ON"))
        b.call_job("PP1")
        b.end_if()
        b.jump("RESET")
        text = b.render()
        assert "IFTHENEXP IN#(8)=ON" in text       # I/O → EXP form, ON kept
        assert "IFTHEN IN#(8)" not in text          # not the plain form

    def test_variable_condition_stays_plain(self):
        b = InformJobBuilder(name="J")
        b.add_position("a", [0] * 6)
        b.movj("a")
        b.while_(("B000", "<", "3"))
        b.end_while()
        text = b.render()
        assert "WHILE B000<3" in text
        assert "WHILEEXP" not in text

    def test_block_body_indented(self):
        """IF/WHILE bodies indent with '\\t ' per level; EXP cond has trailing
        space — matches Yaskawa teach-pendant formatting (byte-exact round-trip)."""
        b = InformJobBuilder(name="J")
        b.add_position("a", [0] * 6, pos_token="P1")
        b.if_then(("IN#(8)", "=", "ON"))
        b.call_job("PP1")
        b.end_if()
        b.movj("a")
        text = b.render()
        assert "IFTHENEXP IN#(8)=ON \r\n" in text     # EXP keeps trailing space
        assert "\t CALL JOB:PP1\r\n" in text          # body indented one level
        assert "\r\nENDIF\r\n" in text                # closer back at level 0

    def test_nested_block_indent_levels(self):
        b = InformJobBuilder(name="J")
        b.add_position("a", [0] * 6)
        b.movj("a")
        b.if_then(("B000", "=", "1"))
        b.while_(("B001", "<", "3"))
        b.set_var("B001", "INC")
        b.end_while()
        b.end_if()
        text = b.render()
        assert "\t WHILE B001<3\r\n" in text          # while nested 1 level in IF
        assert "\t \t INC B001\r\n" in text           # while body nested 2 levels

    def test_default_no_token_sequential_pvar(self):
        """No pos_token → job-local P-variables (controller-accepted format).
        A C-var declared in //POS makes the YRC1000 reject the job on save
        ('451 Error closing file [5130]'), so the default must be P-vars."""
        b = InformJobBuilder(name="J")
        b.add_position("a", [0] * 6)
        b.add_position("b", [0] * 6)
        b.movj("a"); b.movj("b")
        text = b.render()
        assert "///NPOS 0,0,0,2,0,0" in text          # 2 P-vars in slot 3
        assert "P00000=" in text and "P00001=" in text  # 5-digit declaration
        assert "MOVJ P000 " in text and "MOVJ P001 " in text  # 3-digit reference
        assert "C0000" not in text                    # no global C-var in //POS

    def test_default_pvar_index_avoids_explicit_token_collision(self):
        """A new (no-token) point added after an explicit P token gets the next
        FREE P index — never collides with the imported token's index."""
        b = InformJobBuilder(name="J")
        b.add_position("imp", [0] * 6, pos_token="P5")   # imported → index 5
        b.add_position("new", [0] * 6)                    # default → index 6
        b.movj("imp"); b.movj("new")
        text = b.render()
        assert "P00005=" in text and "P00006=" in text
        assert "MOVJ P005 " in text and "MOVJ P006 " in text


# ─────────────────────────────────────────────────────────────────────────
# Snapshot test — full job text byte-exact
# ─────────────────────────────────────────────────────────────────────────


class TestSnapshot:
    def test_minimal_job_snapshot(self):
        """1 vị trí, 1 lệnh MOVJ — verify cấu trúc đầy đủ."""
        b = (
            InformJobBuilder(name="MIN", max_speed_pct=10.0)
            .add_position("home", [0] * 6)
            .movj("home")
        )
        actual = b.render(date_str="2026/05/20 22:00")
        expected = (
            "/JOB\r\n"
            "//NAME MIN\r\n"
            "//POS\r\n"
            "///NPOS 0,0,0,1,0,0\r\n"
            "///TOOL 0\r\n"
            "///POSTYPE PULSE\r\n"
            "///PULSE\r\n"
            "P00000=0,0,0,0,0,0\r\n"
            "//INST\r\n"
            "///DATE 2026/05/20 22:00\r\n"
            "///ATTR SC,RW\r\n"
            "///GROUP1 RB1\r\n"
            "NOP\r\n"
            "MOVJ P000 VJ=10.00\r\n"
            "END\r\n"
        )
        assert actual == expected, f"\nGOT:\n{actual}\nEXPECTED:\n{expected}"

    def test_full_pick_place_job_structure(self):
        text = gen_pick_place_job(
            name="PICKTEST",
            home_deg=[0] * 6,
            approach_deg=[10, -5, 20, 0, 30, -15],
            grasp_deg=[10, -10, 25, 0, 30, -15],
            transfer_deg=[20, -5, 20, 0, 30, -15],
            place_deg=[20, -10, 25, 0, 30, -15],
            gripper_do_index=2,
            gripper_delay_s=0.5,
            speed_pct=15.0,
            max_speed_pct=30.0,
        )
        # Spot check key structural elements
        assert text.startswith("/JOB\r\n//NAME PICKTEST\r\n")
        assert "///NPOS 0,0,0,5,0,0" in text       # 5 P-vars (slot 3)
        assert "DOUT OT#(2) ON" in text            # gripper close
        assert "TIMER T=0.500" in text
        assert "DOUT OT#(2) OFF" in text
        assert "VJ=15.00" in text                   # speed_pct = 15 ≤ max 30
        assert text.endswith("END\r\n")
        # Sequence sanity: open gripper phải sau khi đã transfer + place
        idx_close = text.index("DOUT OT#(2) ON")
        idx_open = text.index("DOUT OT#(2) OFF")
        assert idx_close < idx_open


# ─────────────────────────────────────────────────────────────────────────
# Line endings (Yaskawa requires CRLF)
# ─────────────────────────────────────────────────────────────────────────


class TestLineEndings:
    def test_render_uses_crlf(self):
        b = InformJobBuilder(name="J")
        b.add_position("p", [0] * 6)
        b.movj("p")
        text = b.render()
        # Mỗi dòng phải kết thúc bằng \r\n
        assert "\r\n" in text
        # Không có \n đơn lẻ (không có \r đứng trước)
        lines = text.split("\r\n")
        for line in lines:
            assert "\n" not in line, f"Line has stray \\n: {line!r}"


# ─────────────────────────────────────────────────────────────────────────
# Flow control + variables (INFORM logic)
# ─────────────────────────────────────────────────────────────────────────


class TestLogicCodegen:
    def test_label_jump_setvar(self):
        b = InformJobBuilder(name="J")
        b.add_position("p", [0] * 6)
        b.set_var("B000", "SET", 0).label("LOOP").set_var("B000", "INC")
        b.movj("p").jump("LOOP", cond=("B000", "<", "3"))
        text = b.render()
        assert "SET B000 0" in text
        assert "*LOOP" in text
        assert "INC B000" in text
        assert "JUMP *LOOP IF B000<3" in text

    def test_unconditional_jump(self):
        b = InformJobBuilder(name="J")
        b.add_position("p", [0] * 6).movj("p").jump("DONE")
        assert "JUMP *DONE\r\n" in b.render()

    def test_structured_if_while(self):
        b = InformJobBuilder(name="J")
        b.add_position("p", [0] * 6).movj("p")
        b.if_then(("B000", "=", "1")).msg("HIT").else_().msg("MISS").end_if()
        b.while_(("B001", "<>", "0")).set_var("B001", "DEC").end_while()
        text = b.render()
        for tok in ["IFTHEN B000=1", "ELSE", "ENDIF",
                    "WHILE B001<>0", "DEC B001", "ENDWHILE"]:
            assert tok in text, tok

    def test_invalid_varname_raises(self):
        b = InformJobBuilder(name="J")
        with pytest.raises(ValueError, match="Invalid variable name"):
            b.set_var("XX", "SET", 1)

    def test_invalid_condition_op_raises(self):
        b = InformJobBuilder(name="J")
        with pytest.raises(ValueError, match="Unsupported condition operator"):
            b.if_then(("B000", "==", "1"))


class TestExtendedLGCodegen:
    def test_numeric_label_allowed(self):
        """INFORM labels may be numeric (*1) — unlike job names."""
        b = InformJobBuilder(name="J")
        b.label("1").set_var("B000", "INC").jump("1")
        text = b.render()
        assert "*1\r\n" in text and "JUMP *1\r\n" in text

    def test_call_job_allows_dash(self):
        b = InformJobBuilder(name="J")
        b.call_job("SPEED-1")
        assert "CALL JOB:SPEED-1" in b.render()

    def test_set_express(self):
        b = InformJobBuilder(name="J")
        b.set_express("I000", "5 * B005")
        assert "SET I000 EXPRESS 5 * B005" in b.render()

    def test_compound_condition_andexp(self):
        b = InformJobBuilder(name="J")
        b.if_then(("", "", ""),
                  terms=[("I010", "<>", "11"), ("B010", "<>", "12")], join="AND")
        b.end_if()
        # Compound keyword MUST be the EXP form to match the ANDEXP joiner — real
        # TP exports (examples/LG/HOME.JBI:38) emit IFTHENEXP for variable compounds.
        out = b.render()
        assert "IFTHENEXP I010<>11 ANDEXP B010<>12" in out
        assert "IFTHEN I010<>11 ANDEXP" not in out      # not the invalid plain form

    def test_exp_flag_forces_ifthenexp_on_variable_cond(self):
        """exp=True emits IFTHENEXP even for a variable condition (preserves the
        original keyword on re-synthesis); exp=False stays plain IFTHEN."""
        b = InformJobBuilder(name="J")
        b.if_then(("I010", "=", "12"), exp=True).end_if()
        b.if_then(("B000", "=", "1"), exp=False).end_if()
        text = b.render()
        assert "IFTHENEXP I010=12 " in text       # forced EXP + trailing space
        assert "IFTHEN B000=1\r\n" in text         # plain form

    def test_indirect_motion_and_var_speed(self):
        b = InformJobBuilder(name="J")
        b.movj_indirect("B010", speed_var="I002")
        b.movl_indirect("B011", speed_var="I003")
        text = b.render()
        assert "MOVJ P[B010] VJ=I002" in text
        assert "MOVL P[B011] V=I003" in text

    def test_extended_io_instructions(self):
        b = InformJobBuilder(name="J")
        b.clear_stack().clear_var("I010", 2).pulse(6)
        b.din("B005", "IG", 2).dout_group(2, "B005")
        text = b.render()
        for tok in ["CLEAR STACK", "CLEAR I010 2", "PULSE OT#(6)",
                    "DIN B005 IG#(2)", "DOUT OG#(2) B005"]:
            assert tok in text, tok
