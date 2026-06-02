"""
test_hse_protocol.py
────────────────────
Verify HSE packet codec sinh byte sequence khớp spec Yaskawa.

KHÔNG cần YRC1000 — purely byte-level testing.
"""
from __future__ import annotations

import struct

import pytest

from src.orchestrator.backends.hse_protocol import (
    ACK_RESPONSE,
    GP7_PULSE_PER_DEG,
    HSE_HEADER_SIZE,
    HSE_IDENTIFIER,
    HSE_RESERVE1,
    HSE_RESERVE2,
    Command,
    HSEDecodeError,
    HSEResponse,
    HSERequest,
    HSEResponseError,
    Service,
    parse_position_response,
    pulse_to_deg,
)


# ─────────────────────────────────────────────────────────────────────────
# Request encoding
# ─────────────────────────────────────────────────────────────────────────


class TestHSERequest:
    def test_encode_size_is_32_plus_payload(self):
        req = HSERequest(command=Command.READ_STATUS)
        encoded = req.encode()
        assert len(encoded) == HSE_HEADER_SIZE      # 32 bytes, no payload

    def test_encode_with_payload(self):
        req = HSERequest(command=Command.WRITE_IO, payload=b"\x01\x00\x00\x00")
        encoded = req.encode()
        assert len(encoded) == HSE_HEADER_SIZE + 4

    def test_encode_identifier_yerc(self):
        req = HSERequest(command=Command.READ_STATUS)
        encoded = req.encode()
        assert encoded[0:4] == HSE_IDENTIFIER == b"YERC"

    def test_encode_header_size_field(self):
        req = HSERequest(command=Command.READ_STATUS)
        encoded = req.encode()
        # bytes [4-5] = header size little-endian
        header_size = struct.unpack("<H", encoded[4:6])[0]
        assert header_size == HSE_HEADER_SIZE

    def test_encode_payload_size_field(self):
        payload = b"\xAB\xCD\xEF\x12"
        req = HSERequest(command=Command.WRITE_IO, payload=payload)
        encoded = req.encode()
        payload_size = struct.unpack("<H", encoded[6:8])[0]
        assert payload_size == len(payload)

    def test_encode_reserve_bytes(self):
        req = HSERequest(command=Command.READ_STATUS, request_id=42)
        encoded = req.encode()
        assert encoded[8] == HSE_RESERVE1        # 0x03
        assert encoded[16:24] == HSE_RESERVE2    # b"99999999"

    def test_encode_ack_is_zero_for_request(self):
        req = HSERequest(command=Command.READ_STATUS, request_id=42)
        encoded = req.encode()
        assert encoded[10] == 0x00               # request

    def test_encode_request_id(self):
        req = HSERequest(command=Command.READ_STATUS, request_id=0xA5)
        encoded = req.encode()
        assert encoded[11] == 0xA5

    def test_encode_command_field_little_endian(self):
        req = HSERequest(command=0x1234, instance=0xAABB)
        encoded = req.encode()
        # bytes [24-25] = command LE → 0x34 0x12
        assert encoded[24:26] == b"\x34\x12"
        # bytes [26-27] = instance LE → 0xBB 0xAA
        assert encoded[26:28] == b"\xBB\xAA"

    def test_encode_service_byte(self):
        req = HSERequest(
            command=Command.WRITE_IO, service=Service.SET_ATTRIBUTE_SINGLE
        )
        encoded = req.encode()
        assert encoded[29] == Service.SET_ATTRIBUTE_SINGLE == 0x10

    def test_encode_rejects_out_of_range_request_id(self):
        with pytest.raises(ValueError, match="request_id"):
            HSERequest(command=Command.READ_STATUS, request_id=300).encode()

    def test_read_status_packet_full_bytes(self):
        """Snapshot test: encode lại cho 1 case cụ thể và check toàn bộ 32 byte."""
        req = HSERequest(
            command=Command.READ_STATUS,
            instance=1,
            attribute=0,
            service=Service.GET_ATTRIBUTE_ALL,
            request_id=0,
        )
        encoded = req.encode()
        expected = (
            b"YERC"               # identifier
            b"\x20\x00"           # header size = 32 (LE)
            b"\x00\x00"           # payload size = 0
            b"\x03"               # reserve1
            b"\x01"               # division = robot
            b"\x00"               # ack = request
            b"\x00"               # request_id
            b"\x00\x00\x00\x00"   # block_no = 0
            b"99999999"           # reserve2
            b"\x72\x00"           # command 0x72 READ_STATUS (LE)
            b"\x01\x00"           # instance 1 (LE)
            b"\x00"               # attribute 0
            b"\x01"               # service GET_ATTRIBUTE_ALL
            b"\x00\x00"           # padding
        )
        assert encoded == expected, f"\nGOT:      {encoded.hex()}\nEXPECTED: {expected.hex()}"


# ─────────────────────────────────────────────────────────────────────────
# Response decoding
# ─────────────────────────────────────────────────────────────────────────


def _build_response_bytes(
    status: int = 0,
    payload: bytes = b"",
    request_id: int = 0,
    added_status: int = 0,
) -> bytes:
    """Helper: build a synthetic response packet."""
    header = struct.pack(
        "<4s H H B B B B I 8s B B B B H H",
        HSE_IDENTIFIER,
        HSE_HEADER_SIZE,
        len(payload),
        HSE_RESERVE1,
        0x01,                  # division
        ACK_RESPONSE,          # ack = response
        request_id,
        0,                     # block_no
        HSE_RESERVE2,
        0x01,                  # service echo
        status,                # status
        0 if added_status == 0 else 2,  # added_status_size
        0,                     # padding
        added_status,          # added_status
        0,                     # padding
    )
    return header + payload


class TestHSEResponse:
    def test_decode_ok_status(self):
        data = _build_response_bytes(status=0, payload=b"\x01\x02\x03")
        resp = HSEResponse.decode(data)
        assert resp.status == 0
        assert resp.payload == b"\x01\x02\x03"

    def test_decode_too_short_raises(self):
        with pytest.raises(HSEDecodeError, match="too short"):
            HSEResponse.decode(b"\x00" * 10)

    def test_decode_wrong_identifier_raises(self):
        data = _build_response_bytes()
        data = b"XXXX" + data[4:]
        with pytest.raises(HSEDecodeError, match="Invalid identifier"):
            HSEResponse.decode(data)

    def test_decode_payload_truncated_raises(self):
        # Header claims 10-byte payload but only 5 bytes follow
        data = _build_response_bytes(payload=b"\x00" * 10)
        truncated = data[: HSE_HEADER_SIZE + 5]
        with pytest.raises(HSEDecodeError, match="Payload size mismatch"):
            HSEResponse.decode(truncated)

    def test_raise_for_status_with_error(self):
        data = _build_response_bytes(status=0x1F, added_status=0x2301)
        resp = HSEResponse.decode(data)
        with pytest.raises(HSEResponseError) as exc:
            resp.raise_for_status()
        assert exc.value.status == 0x1F
        assert exc.value.added_status == 0x2301

    def test_raise_for_status_ok_no_raise(self):
        data = _build_response_bytes(status=0)
        resp = HSEResponse.decode(data)
        resp.raise_for_status()                # không raise


# ─────────────────────────────────────────────────────────────────────────
# Position parse + pulse→degree
# ─────────────────────────────────────────────────────────────────────────


class TestPositionParse:
    def test_parse_position_extracts_6_joints(self):
        # data_type=0x10 (pulse), 5 header uint32 + 8 axis int32
        payload = struct.pack(
            "<5I 8i",
            0x10,   # data_type pulse
            0, 0, 0, 0,
            13414,  # S = 10° × 1341.4 pulse/deg
            -13414, # L = -10°
            26828,  # U = 20°
            10000,  # R = 10°
            5000,   # B = 5°
            3120,   # T = 5°
            0, 0,   # axis 7-8 not used
        )
        parsed = parse_position_response(payload)
        assert parsed["data_type"] == 0x10
        assert parsed["joints_raw"] == [13414, -13414, 26828, 10000, 5000, 3120]

    def test_parse_position_too_short_raises(self):
        with pytest.raises(HSEDecodeError):
            parse_position_response(b"\x00" * 20)

    def test_pulse_to_deg_roundtrip_within_ratio(self):
        deg_in = [10.0, -10.0, 20.0, 10.0, 5.0, 5.0]
        pulses = [int(d * r) for d, r in zip(deg_in, GP7_PULSE_PER_DEG)]
        deg_out = pulse_to_deg(pulses)
        for actual, expected in zip(deg_out, deg_in):
            assert abs(actual - expected) < 0.01    # < 0.01° error

    def test_pulse_to_deg_wrong_axis_count_raises(self):
        with pytest.raises(ValueError, match="wrong robot model"):
            pulse_to_deg([0, 0, 0])                # 3 ≠ 6


# ─────────────────────────────────────────────────────────────────────────
# Cartesian position encoding (Option 1 — YRC IK)
# ─────────────────────────────────────────────────────────────────────────


from src.orchestrator.backends.hse_protocol import (   # noqa: E402
    POSITION_UNIT_SCALE,
    ROTATION_UNIT_SCALE,
    DataType,
    encode_cartesian_position,
)


class TestCartesianEncoding:
    def test_payload_size_is_52_bytes(self):
        """5 uint32 header (20B) + 8 int32 axis (32B) = 52B."""
        payload = encode_cartesian_position(0, 0, 0, 0, 0, 0)
        assert len(payload) == 52

    def test_default_data_type_is_base(self):
        payload = encode_cartesian_position(0, 0, 0, 0, 0, 0)
        data_type = struct.unpack("<I", payload[0:4])[0]
        assert data_type == DataType.BASE == 16

    def test_position_units_micrometers(self):
        """X=500mm → field = 500 × 1000 = 500000 (μm)."""
        payload = encode_cartesian_position(500.0, 0, 0, 0, 0, 0)
        x_field = struct.unpack("<i", payload[20:24])[0]
        assert x_field == 500_000

    def test_rotation_units_0_0001_deg(self):
        """Rx=30° → field = 30 × 10000 = 300000."""
        payload = encode_cartesian_position(0, 0, 0, 30.0, 0, 0)
        rx_field = struct.unpack("<i", payload[32:36])[0]
        assert rx_field == 300_000

    def test_negative_position(self):
        payload = encode_cartesian_position(-200.5, 0, 0, 0, 0, 0)
        x_field = struct.unpack("<i", payload[20:24])[0]
        assert x_field == -200_500

    def test_tool_no_field(self):
        payload = encode_cartesian_position(0, 0, 0, 0, 0, 0, tool_no=1)
        tool = struct.unpack("<I", payload[8:12])[0]
        assert tool == 1

    def test_form_field(self):
        payload = encode_cartesian_position(0, 0, 0, 0, 0, 0, form=0b00000111)
        form = struct.unpack("<I", payload[4:8])[0]
        assert form == 7

    def test_user_frame_field(self):
        payload = encode_cartesian_position(0, 0, 0, 0, 0, 0, user_frame=2)
        uf = struct.unpack("<I", payload[12:16])[0]
        assert uf == 2

    def test_reserved_axis_7_8_are_zero(self):
        payload = encode_cartesian_position(100, 200, 300, 10, 20, 30)
        ax7 = struct.unpack("<i", payload[44:48])[0]
        ax8 = struct.unpack("<i", payload[48:52])[0]
        assert ax7 == 0 and ax8 == 0

    def test_invalid_tool_no_raises(self):
        with pytest.raises(ValueError, match="tool_no"):
            encode_cartesian_position(0, 0, 0, 0, 0, 0, tool_no=64)

    def test_full_pose_round_trip(self):
        """Encode → manually decode → values match."""
        payload = encode_cartesian_position(
            x_mm=500.5, y_mm=-200.3, z_mm=600.1,
            rx_deg=10.5, ry_deg=-20.5, rz_deg=30.0,
            tool_no=1, form=0,
        )
        fields = struct.unpack("<5I 8i", payload)
        # fields[0]=data_type, [1]=form, [2]=tool, [3]=uf, [4]=ext_form
        # [5]=x, [6]=y, [7]=z, [8]=rx, [9]=ry, [10]=rz, [11,12]=reserved
        assert fields[0] == 16
        assert fields[2] == 1
        assert fields[5] == int(round(500.5 * POSITION_UNIT_SCALE))
        assert fields[6] == int(round(-200.3 * POSITION_UNIT_SCALE))
        assert fields[7] == int(round(600.1 * POSITION_UNIT_SCALE))
        assert fields[8] == int(round(10.5 * ROTATION_UNIT_SCALE))
