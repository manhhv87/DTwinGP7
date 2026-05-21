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
        with pytest.raises(HSEDecodeError, match="quá ngắn"):
            HSEResponse.decode(b"\x00" * 10)

    def test_decode_wrong_identifier_raises(self):
        data = _build_response_bytes()
        data = b"XXXX" + data[4:]
        with pytest.raises(HSEDecodeError, match="Identifier"):
            HSEResponse.decode(data)

    def test_decode_payload_truncated_raises(self):
        # Header claims 10-byte payload but only 5 bytes follow
        data = _build_response_bytes(payload=b"\x00" * 10)
        truncated = data[: HSE_HEADER_SIZE + 5]
        with pytest.raises(HSEDecodeError, match="size lệch"):
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
        with pytest.raises(ValueError, match="sai robot model"):
            pulse_to_deg([0, 0, 0])                # 3 ≠ 6
