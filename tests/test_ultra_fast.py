"""
test_ultra_fast.py
──────────────────
Verify ultra-fast P-variable mode (M3++):
  - Template upload chỉ 1 lần đầu (signature đầu trial)
  - Trial sau cùng signature: chỉ WRITE_POS_VAR + JOB_START (0 FTP)
  - WRITE_POS_VAR byte format correct
  - gen_pvar_template_job sinh INFORM với P-references
"""
from __future__ import annotations

import struct
from unittest.mock import MagicMock

import pytest

from src.orchestrator.backends.hse_protocol import (
    ACK_RESPONSE,
    GP7_PULSE_PER_DEG,
    HSE_HEADER_SIZE,
    HSE_IDENTIFIER,
    HSE_RESERVE1,
    HSE_RESERVE2,
    Command,
)
from src.orchestrator.backends.inform_codegen import gen_pvar_template_job
from src.orchestrator.backends.motoman_hse import MotomanHSEBackend


def _build_response(status=0, payload=b"", request_id=0):
    header = struct.pack(
        "<4s H H B B B B I 8s B B B B H H",
        HSE_IDENTIFIER, HSE_HEADER_SIZE, len(payload),
        HSE_RESERVE1, 0x01, ACK_RESPONSE, request_id, 0,
        HSE_RESERVE2, 0x01, status, 0, 0, 0, 0,
    )
    return header + payload


@pytest.fixture
def backend_with_mocks(monkeypatch):
    backend = MotomanHSEBackend(ip="192.168.1.100", timeout_s=0.5)
    backend.start_confirm_timeout_s = 0.0    # skip Running-assert wait in unit tests
    mock_sock = MagicMock()
    backend._sock = mock_sock

    # FTP mock
    uploaded = {"count": 0, "data": b""}
    mock_ftp_cls = MagicMock()
    mock_ftp_inst = MagicMock()
    mock_ftp_cls.return_value = mock_ftp_inst

    def capture_stor(cmd, stream):
        uploaded["count"] += 1
        uploaded["data"] = stream.read()
        uploaded["cmd"] = cmd

    mock_ftp_inst.storbinary.side_effect = capture_stor
    import ftplib
    monkeypatch.setattr(ftplib, "FTP", mock_ftp_cls)
    return backend, mock_sock, uploaded


# ─────────────────────────────────────────────────────────────────────────
# gen_pvar_template_job
# ─────────────────────────────────────────────────────────────────────────


class TestTemplateGen:
    def test_template_uses_p_variables(self):
        text = gen_pvar_template_job("TEST", num_positions=3)
        assert "MOVJ P000" in text
        assert "MOVJ P001" in text
        assert "MOVJ P002" in text
        # KHÔNG có C-variable declarations trong //POS
        assert "C00000" not in text

    def test_template_empty_pos_section(self):
        text = gen_pvar_template_job("TEST", num_positions=2)
        assert "///NPOS 0,0,0,0,0,0" in text          # P-vars runtime, NPOS=0

    def test_template_with_gripper(self):
        text = gen_pvar_template_job(
            "PICK", num_positions=4,
            gripper_close_at=1, gripper_open_at=3,
        )
        # DOUT ON xuất hiện TRƯỚC MOVJ P001
        idx_p1 = text.index("MOVJ P001")
        idx_close = text.index("DOUT OT#(1) ON")
        assert idx_close < idx_p1

        idx_p3 = text.index("MOVJ P003")
        idx_open = text.index("DOUT OT#(1) OFF")
        assert idx_open < idx_p3

    def test_template_motion_kinds_mixed(self):
        text = gen_pvar_template_job(
            "MIX", num_positions=3,
            motion_kinds=["movj", "movl", "movj"],
        )
        assert "MOVJ P000" in text
        assert "MOVL P001" in text
        assert "MOVJ P002" in text

    def test_template_invalid_num_pos_raises(self):
        with pytest.raises(ValueError):
            gen_pvar_template_job("X", num_positions=0)
        with pytest.raises(ValueError):
            gen_pvar_template_job("X", num_positions=200)

    def test_motion_kinds_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="motion_kinds"):
            gen_pvar_template_job("X", num_positions=3, motion_kinds=["movj"])


# ─────────────────────────────────────────────────────────────────────────
# write_position_var byte format
# ─────────────────────────────────────────────────────────────────────────


class TestWritePositionVar:
    def test_write_pos_var_command_byte(self, backend_with_mocks):
        backend, mock_sock, _ = backend_with_mocks
        mock_sock.recvfrom.return_value = (_build_response(), ("x", 10040))
        backend.write_position_var(0, [10.0, 0, 0, 0, 0, 0])
        packet, _ = mock_sock.sendto.call_args[0]
        cmd = struct.unpack("<H", packet[24:26])[0]
        assert cmd == Command.WRITE_POS_VAR == 0x7F

    def test_write_pos_var_instance_is_p_index(self, backend_with_mocks):
        backend, mock_sock, _ = backend_with_mocks
        mock_sock.recvfrom.return_value = (_build_response(), ("x", 10040))
        backend.write_position_var(5, [0] * 6)
        packet, _ = mock_sock.sendto.call_args[0]
        instance = struct.unpack("<H", packet[26:28])[0]
        assert instance == 5

    def test_write_pos_var_payload_pulses_correct(self, backend_with_mocks):
        backend, mock_sock, _ = backend_with_mocks
        mock_sock.recvfrom.return_value = (_build_response(), ("x", 10040))
        backend.write_position_var(0, [10.0, -5.0, 20.0, 0, 0, 0])
        packet, _ = mock_sock.sendto.call_args[0]
        payload = packet[HSE_HEADER_SIZE:]
        # Skip 5x uint32 header → 6x int32 pulses
        pulses = struct.unpack("<6i", payload[20:44])
        expected = [int(round(d * r)) for d, r in zip(
            [10.0, -5.0, 20.0, 0, 0, 0], GP7_PULSE_PER_DEG,
        )]
        assert list(pulses) == expected

    def test_write_pos_var_invalid_index_raises(self, backend_with_mocks):
        backend, _, _ = backend_with_mocks
        with pytest.raises(ValueError, match="0-127"):
            backend.write_position_var(200, [0] * 6)

    def test_write_pos_var_wrong_joint_count_raises(self, backend_with_mocks):
        backend, _, _ = backend_with_mocks
        with pytest.raises(ValueError, match="6 joints"):
            backend.write_position_var(0, [0, 0, 0])


# ─────────────────────────────────────────────────────────────────────────
# Ultra-fast batch end-to-end
# ─────────────────────────────────────────────────────────────────────────


class TestUltraFastBatch:
    def test_first_batch_uploads_template(self, backend_with_mocks):
        backend, mock_sock, uploaded = backend_with_mocks
        backend.enable_ultra_fast(True)

        # 3 WRITE_POS_VAR + JOB_SELECT + START + READ_STATUS idle + READ_ALARM none
        mock_sock.recvfrom.side_effect = [
            (_build_response(), ("x", 10040)) for _ in range(5)
        ] + [(_build_response(payload=b"\x00"), ("x", 10040)),
             (_build_response(payload=b"\x00"), ("x", 10040))]

        with backend.batch():
            backend.MoveJ([0, 0, 0, 0, 0, 0])
            backend.MoveJ([10, 0, 0, 0, 0, 0])
            backend.MoveJ([0, 0, 0, 0, 0, 0])

        # FTP upload đúng 1 lần (template)
        assert uploaded["count"] == 1
        # Template chứa MOVJ P000, P001, P002
        assert b"MOVJ P000" in uploaded["data"]
        assert b"MOVJ P002" in uploaded["data"]

    def test_second_batch_same_signature_skips_upload(self, backend_with_mocks):
        backend, mock_sock, uploaded = backend_with_mocks
        backend.enable_ultra_fast(True)

        # Mỗi batch: 3 WRITE_POS_VAR + JOB_SELECT + START + READ_STATUS idle = 6 packets
        per_batch_responses = [
            (_build_response(), ("x", 10040)) for _ in range(5)
        ] + [(_build_response(payload=b"\x00"), ("x", 10040))]
        mock_sock.recvfrom.side_effect = per_batch_responses * 3      # 3 batches worth

        # Batch 1
        with backend.batch():
            backend.MoveJ([0] * 6)
            backend.MoveJ([10] * 6)
            backend.MoveJ([20] * 6)
        first_upload_count = uploaded["count"]

        # Batch 2 — cùng signature (3 MOVJ)
        with backend.batch():
            backend.MoveJ([0] * 6)
            backend.MoveJ([15] * 6)
            backend.MoveJ([25] * 6)

        # FTP upload KHÔNG tăng (template reused)
        assert uploaded["count"] == first_upload_count == 1

    def test_signature_change_triggers_new_upload(self, backend_with_mocks):
        backend, mock_sock, uploaded = backend_with_mocks
        backend.enable_ultra_fast(True)

        # Đủ response cho 2 batches
        responses = []
        for _ in range(20):
            responses.append((_build_response(), ("x", 10040)))
            responses.append((_build_response(payload=b"\x00"), ("x", 10040)))
        mock_sock.recvfrom.side_effect = responses

        # Batch 1: 2 MOVJ
        with backend.batch():
            backend.MoveJ([0] * 6)
            backend.MoveJ([10] * 6)
        upload_1 = uploaded["count"]

        # Batch 2: 3 MOVJ (signature khác)
        with backend.batch():
            backend.MoveJ([0] * 6)
            backend.MoveJ([10] * 6)
            backend.MoveJ([20] * 6)

        # Upload count tăng vì signature mới
        assert uploaded["count"] > upload_1

    def test_dout_in_template_at_correct_position(self, backend_with_mocks):
        backend, mock_sock, uploaded = backend_with_mocks
        backend.enable_ultra_fast(True)
        responses = [(_build_response(), ("x", 10040)) for _ in range(10)]
        responses.append((_build_response(payload=b"\x00"), ("x", 10040)))
        mock_sock.recvfrom.side_effect = responses

        with backend.batch():
            backend.MoveJ([0] * 6)           # P000
            backend.setDO(1, 1)              # → DOUT ON trước P001
            backend.timer(0.3)
            backend.MoveJ([10] * 6)          # P001

        data = uploaded["data"]
        # DOUT ON phải xuất hiện trước MOVJ P001
        idx_p001 = data.index(b"MOVJ P001")
        idx_dout = data.index(b"DOUT OT#(1) ON")
        assert idx_dout < idx_p001

    def test_disable_ultra_fast_falls_back_to_standard_batch(
        self, backend_with_mocks,
    ):
        backend, mock_sock, uploaded = backend_with_mocks
        # KHÔNG enable_ultra_fast → batch dùng đường InformJobBuilder thường
        # (khai vị trí trong //POS). Ultra-fast thì set P-var lúc runtime,
        # //POS rỗng (NPOS 0,0,0,0,0,0).
        responses = [(_build_response(), ("x", 10040)) for _ in range(3)]
        responses.append((_build_response(payload=b"\x00"), ("x", 10040)))
        mock_sock.recvfrom.side_effect = responses

        with backend.batch():
            backend.MoveJ([0] * 6)
            backend.MoveJ([10] * 6)

        # Standard batch khai P-variable trong //POS (đúng format controller
        # nhận — KHÔNG dùng global C-var, vốn bị YRC1000 từ chối lúc lưu 451).
        data = uploaded["data"]
        assert b"///NPOS 0,0,0,2,0,0" in data             # 2 P-vars khai ở slot 3
        assert b"P00000=" in data                          # khai vị trí trong //POS
        assert b"MOVJ P000" in data
        assert b"C00000=" not in data                     # KHÔNG có global C-var

    def test_ultra_fast_nested_batch_raises(self, backend_with_mocks):
        backend, _, _ = backend_with_mocks
        backend.enable_ultra_fast(True)
        with backend.batch():
            with pytest.raises(RuntimeError, match="Nested"):
                with backend.batch():
                    pass

    def test_ultra_fast_writes_pvar_for_each_motion(self, backend_with_mocks):
        backend, mock_sock, _ = backend_with_mocks
        backend.enable_ultra_fast(True)
        responses = [(_build_response(), ("x", 10040)) for _ in range(10)]
        responses.append((_build_response(payload=b"\x00"), ("x", 10040)))
        mock_sock.recvfrom.side_effect = responses

        with backend.batch():
            backend.MoveJ([10] * 6)
            backend.MoveJ([20] * 6)

        # Count WRITE_POS_VAR calls trong sendto (command byte = 0x7F)
        write_pvar_count = 0
        for call in mock_sock.sendto.call_args_list:
            packet = call.args[0]
            cmd = struct.unpack("<H", packet[24:26])[0]
            if cmd == 0x7F:                              # WRITE_POS_VAR
                write_pvar_count += 1
        assert write_pvar_count == 2                     # 2 MoveJ → 2 P-vars
