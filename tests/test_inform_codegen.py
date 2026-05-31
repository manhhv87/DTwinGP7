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
        with pytest.raises(ValueError, match="1-32 ký tự"):
            InformJobBuilder(name="")

    def test_long_name_raises(self):
        with pytest.raises(ValueError, match="1-32 ký tự"):
            InformJobBuilder(name="x" * 50)

    def test_name_with_special_chars_raises(self):
        with pytest.raises(ValueError, match="chỉ chữ"):
            InformJobBuilder(name="bad name!")

    def test_name_with_underscore_allowed(self):
        InformJobBuilder(name="PICK_PLACE_01")          # OK

    def test_duplicate_position_raises(self):
        b = InformJobBuilder(name="J")
        b.add_position("p0", [0] * 6)
        with pytest.raises(ValueError, match="đã tồn tại"):
            b.add_position("p0", [1] * 6)

    def test_joint_count_mismatch_raises(self):
        b = InformJobBuilder(name="J")
        with pytest.raises(ValueError, match="phải 6"):
            b.add_position("p0", [0, 0, 0])

    def test_movj_unknown_position_raises(self):
        b = InformJobBuilder(name="J")
        with pytest.raises(KeyError, match="chưa add"):
            b.movj("ghost")

    def test_too_many_positions_raises(self):
        b = InformJobBuilder(name="J")
        for i in range(MAX_POSITIONS_PER_JOB):
            b.add_position(f"p{i}", [0] * 6)
        with pytest.raises(ValueError, match="giới hạn"):
            b.add_position(f"p{MAX_POSITIONS_PER_JOB}", [0] * 6)

    def test_dout_index_out_of_range_raises(self):
        b = InformJobBuilder(name="J")
        b.add_position("p", [0] * 6)
        with pytest.raises(ValueError, match="DOUT index"):
            b.dout(0, True)
        with pytest.raises(ValueError, match="DOUT index"):
            b.dout(2000, True)

    def test_render_empty_raises(self):
        b = InformJobBuilder(name="J")
        with pytest.raises(ValueError, match="ít nhất 1 position"):
            b.render()
        b.add_position("p", [0] * 6)
        with pytest.raises(ValueError, match="ít nhất 1 instruction"):
            b.render()


# ─────────────────────────────────────────────────────────────────────────
# Pulse conversion
# ─────────────────────────────────────────────────────────────────────────


class TestPulseConversion:
    def test_degrees_converted_with_ratio(self):
        b = InformJobBuilder(name="J", pulse_per_deg=GP7_PULSE_PER_DEG)
        b.add_position("p", [10.0, -5.0, 20.0, 10.0, 5.0, 5.0])
        b.movj("p")
        text = b.render()
        # S axis 10° × 1341.4 = 13414 pulse
        assert "C00000=13414," in text
        # L axis -5° × 1341.4 = -6707 pulse
        assert ",-6707," in text

    def test_zero_joints_emits_zeros(self):
        b = InformJobBuilder(name="J")
        b.add_position("home", [0] * 6)
        b.movj("home")
        text = b.render()
        assert "C00000=0,0,0,0,0,0" in text


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
        assert "MOVL C00000 V=80.0" in b.render()


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
            "///NPOS 1,0,0,0,0,0\r\n"
            "///TOOL 0\r\n"
            "///POSTYPE PULSE\r\n"
            "///PULSE\r\n"
            "C00000=0,0,0,0,0,0\r\n"
            "//INST\r\n"
            "///DATE 2026/05/20 22:00\r\n"
            "///ATTR SC,RW\r\n"
            "///GROUP1 RB1\r\n"
            "NOP\r\n"
            "MOVJ C00000 VJ=10.00\r\n"
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
        assert "///NPOS 5,0,0,0,0,0" in text       # 5 positions
        assert "DOUT OT#(2) ON" in text            # gripper close
        assert "TIMER T=0.50" in text
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
        with pytest.raises(ValueError, match="biến"):
            b.set_var("XX", "SET", 1)

    def test_invalid_condition_op_raises(self):
        b = InformJobBuilder(name="J")
        with pytest.raises(ValueError, match="điều kiện"):
            b.if_then(("B000", "==", "1"))
