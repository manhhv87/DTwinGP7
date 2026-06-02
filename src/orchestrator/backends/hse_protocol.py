"""
hse_protocol.py
───────────────
Codec for Yaskawa High-Speed Ethernet Server protocol.

Spec reference:
  - Yaskawa "High-Speed Ethernet Server Function" manual (HW1485553-1)
  - Port UDP 10040 for robot commands, UDP 10041 for file load/save
  - Each packet: 32-byte sub-header + variable payload (≤ ~478 bytes)

This module is PURE — encode/decode byte sequences only. Socket I/O is in `motoman_hse.py`.
Testable with known byte sequences; no physical YRC1000 required.

Conventions:
  - All multi-byte integers = LITTLE ENDIAN (Yaskawa default for YRC1000)
  - Identifier "YERC" ASCII = 0x59 0x45 0x52 0x43
  - Reserve2 = "99999999" ASCII (yes, 8 '9' characters)

Service codes:
  0x01 = Get_Attribute_All
  0x02 = Set_Attribute_All
  0x0E = Get_Attribute_Single
  0x10 = Set_Attribute_Single
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum

# ───── Constants ─────
HSE_IDENTIFIER = b"YERC"
HSE_HEADER_SIZE = 0x0020          # 32 bytes
HSE_RESERVE1 = 0x03               # always 0x03
HSE_RESERVE2 = b"99999999"        # 8 ASCII '9' characters

DIVISION_ROBOT = 0x01             # robot command service
DIVISION_FILE = 0x02              # file command service

ACK_REQUEST = 0x00                # client → server
ACK_RESPONSE = 0x01               # server → client


class Service(IntEnum):
    """HSE service code (byte 29 in request sub-header)."""

    GET_ATTRIBUTE_ALL = 0x01
    SET_ATTRIBUTE_ALL = 0x02
    GET_ATTRIBUTE_SINGLE = 0x0E
    SET_ATTRIBUTE_SINGLE = 0x10


class Command(IntEnum):
    """Commonly used HSE command codes (bytes 24-25, little-endian).

    Abbreviated list — the manual has ~30 commands. Add as needed.
    """

    READ_ALARM = 0x70
    READ_STATUS = 0x72
    READ_JOB_INFO = 0x73
    READ_POSITION = 0x75           # joint angles + cartesian
    READ_POSITION_ERROR = 0x76
    READ_TORQUE = 0x77
    READ_IO = 0x78                 # network/general I/O
    WRITE_IO = 0x78                # same command, service 0x10 = write
    READ_REGISTER = 0x79
    WRITE_REGISTER = 0x79
    READ_BYTE_VAR = 0x7A
    READ_INT_VAR = 0x7B
    READ_DINT_VAR = 0x7C
    READ_REAL_VAR = 0x7D
    READ_STRING_VAR = 0x7E
    JOB_SELECT = 0x87
    HOLD_SERVO = 0x83              # servo on/off
    CYCLE = 0x84                   # cycle mode (1=step, 2=cycle, 3=auto)
    START = 0x86                   # start job (after JOB_SELECT)
    # P-variable read/write (M3++ ultra-fast mode)
    READ_POS_VAR = 0x7F            # read P-variable (instance = P-index)
    WRITE_POS_VAR = 0x7F            # same command, service SET_ATTRIBUTE_ALL


class DataType(IntEnum):
    """Position variable data type field for WRITE_POSITION_VAR / READ_POSITION.

    Per Yaskawa HSE spec HW1485553. Determines how the 8 axis values are interpreted:
      - PULSE: joint angles in encoder pulse units (requires conversion via pulse_per_deg)
      - BASE: Cartesian pose in robot BASE frame (X,Y,Z,Rx,Ry,Rz)
      - ROBOT: same as BASE for non-track robots (single arm GP7)
      - USER01-63: Cartesian in user-defined frame
      - TOOL: Cartesian in current tool frame (rarely used)
    """

    PULSE = 0
    BASE = 16                       # Robot base coordinate (recommended for pick-place)
    ROBOT = 17                      # = BASE for GP7 (no track axis)
    USER01 = 18                     # USER01 frame; +N for USER02..USER63 (max 80)
    USER63 = 80
    TOOL = 65                       # tool frame


# ───── Yaskawa Cartesian field unit conventions ─────
# Position field value = mm × POSITION_UNIT_SCALE (i.e., μm precision).
POSITION_UNIT_SCALE = 1000.0
# Rotation field value = degrees × ROTATION_UNIT_SCALE (0.0001° precision).
ROTATION_UNIT_SCALE = 10000.0


def encode_cartesian_position(
    x_mm: float, y_mm: float, z_mm: float,
    rx_deg: float, ry_deg: float, rz_deg: float,
    tool_no: int = 0,
    user_frame: int = 0,
    form: int = 0,
    data_type: int = DataType.BASE,
) -> bytes:
    """Encode Cartesian pose → 52-byte payload for WRITE_POSITION_VAR (cmd 0x7F).

    Layout matches the joint variant but axis values are Cartesian instead of pulse:
        [0-3]    data_type (BASE=16 for robot base frame)
        [4-7]    form bits (IK config branch — bit 0=back, 1=elbow, 2=flip, ...)
        [8-11]   tool_no (0-63, gripper TCP)
        [12-15]  user_frame (0 if BASE, 1-63 if USER)
        [16-19]  extended_form (0 for GP7 — no track axis)
        [20-23]  X position (int32, unit 0.001mm = μm)
        [24-27]  Y position
        [28-31]  Z position
        [32-35]  Rx rotation (int32, unit 0.0001° degree)
        [36-39]  Ry rotation
        [40-43]  Rz rotation
        [44-47]  Reserved (0)
        [48-51]  Reserved (0)

    Rotation convention: Yaskawa uses XYZ-fixed (= ZYX-intrinsic Tait-Bryan).
    Caller must convert pose matrix → (rx, ry, rz) using this convention — see
    `frame_convert.matrix_to_rpy_yaskawa()`.

    Args:
        x_mm, y_mm, z_mm: Position in the frame specified by `data_type`.
        rx_deg, ry_deg, rz_deg: Rotation in degrees (XYZ-fixed convention).
        tool_no: 0-63 — gripper TCP configured on the teach pendant.
        user_frame: 0 for BASE, 1-63 for USER01-USER63.
        form: IK config bits (0 = front + elbow up + no flip — default pick-place).
        data_type: 16 (BASE), 17 (ROBOT), 18-80 (USER), 65 (TOOL).

    Returns:
        52-byte payload ready to send via HSE.

    Raises:
        ValueError: Any field is outside int32 range after scaling.
    """
    if not (0 <= tool_no <= 63):
        raise ValueError(f"tool_no must be 0-63, got {tool_no}")
    if not (0 <= user_frame <= 63):
        raise ValueError(f"user_frame must be 0-63, got {user_frame}")
    if not (0 <= form <= 0xFF):
        raise ValueError(f"form bits must be 0-255 (8 bits), got {form}")

    # Scale to Yaskawa integer units
    x_i = int(round(x_mm * POSITION_UNIT_SCALE))
    y_i = int(round(y_mm * POSITION_UNIT_SCALE))
    z_i = int(round(z_mm * POSITION_UNIT_SCALE))
    rx_i = int(round(rx_deg * ROTATION_UNIT_SCALE))
    ry_i = int(round(ry_deg * ROTATION_UNIT_SCALE))
    rz_i = int(round(rz_deg * ROTATION_UNIT_SCALE))

    # int32 range check (~2.1 billion → ±2147km position, ±214748° rotation)
    INT32_MAX, INT32_MIN = 2_147_483_647, -2_147_483_648
    for name, v in (("X", x_i), ("Y", y_i), ("Z", z_i),
                    ("Rx", rx_i), ("Ry", ry_i), ("Rz", rz_i)):
        if not (INT32_MIN <= v <= INT32_MAX):
            raise ValueError(f"{name} value after scaling is outside int32 range: {v}")

    return struct.pack(
        "<5I 8i",
        int(data_type),     # [0-3] data_type
        int(form),          # [4-7] form
        int(tool_no),       # [8-11] tool_no
        int(user_frame),    # [12-15] user_frame
        0,                  # [16-19] extended_form
        x_i, y_i, z_i,      # [20-31] X, Y, Z (μm)
        rx_i, ry_i, rz_i,   # [32-43] Rx, Ry, Rz (0.0001°)
        0, 0,               # [44-51] reserved
    )


# ───── Errors ─────


class HSEError(Exception):
    """Base error for HSE codec / IO."""


class HSEDecodeError(HSEError):
    """Bytes cannot be decoded (wrong size or identifier)."""


class HSEResponseError(HSEError):
    """Server returned status != 0 (error from controller).

    Status code reference: manual section "Added status". Common:
      0x08 = invalid sub command
      0x0E = invalid data
      0x1F = robot not ready
      0x23 = key switch not in remote mode
    """

    def __init__(self, status: int, added_status: int = 0) -> None:
        super().__init__(
            f"HSE response status=0x{status:02X} added_status=0x{added_status:04X}"
        )
        self.status = status
        self.added_status = added_status


# ───── Request packet ─────


@dataclass
class HSERequest:
    """A single HSE request packet (client → server).

    Encodes to a 32-byte sub-header + payload. request_id pairs request/response
    when multiple commands are sent concurrently.
    """

    command: int                              # Command enum value
    instance: int = 0                         # data instance (e.g. robot 1 = 1)
    attribute: int = 0                        # attribute selector (0 = all)
    service: int = Service.GET_ATTRIBUTE_ALL  # GET or SET
    payload: bytes = b""                      # data payload (for SET)
    request_id: int = 0                       # 0-255, paired with response
    division: int = DIVISION_ROBOT
    block_no: int = 0

    def encode(self) -> bytes:
        """Pack into bytes for sending over UDP."""
        if not (0 <= self.request_id <= 0xFF):
            raise ValueError(f"request_id must be 0-255, got {self.request_id}")
        if not (0 <= self.command <= 0xFFFF):
            raise ValueError(f"command out of uint16 range: 0x{self.command:X}")

        payload_size = len(self.payload)
        if payload_size > 0xFFFF:
            raise ValueError(f"payload too large: {payload_size} > 65535 bytes")

        # Sub-header 32 bytes (little-endian per Yaskawa spec)
        header = struct.pack(
            "<4s H H B B B B I 8s H H B B H",
            HSE_IDENTIFIER,       # [0-3]   "YERC"
            HSE_HEADER_SIZE,      # [4-5]   header size = 0x0020
            payload_size,         # [6-7]   payload size
            HSE_RESERVE1,         # [8]     reserve1 = 0x03
            self.division,        # [9]     processing division
            ACK_REQUEST,          # [10]    ack = 0 (request)
            self.request_id,      # [11]    request ID
            self.block_no,        # [12-15] block number (uint32)
            HSE_RESERVE2,         # [16-23] "99999999"
            self.command,         # [24-25] command number
            self.instance,        # [26-27] instance
            self.attribute,       # [28]    attribute
            self.service,         # [29]    service
            0x0000,               # [30-31] padding
        )
        assert len(header) == HSE_HEADER_SIZE, (
            f"header encode error: {len(header)} != {HSE_HEADER_SIZE}"
        )
        return header + self.payload


# ───── Response packet ─────


@dataclass
class HSEResponse:
    """A single HSE response packet (server → client).

    Sub-header layout differs from request (bytes 24-29):
      [24]    Service (echoed from request)
      [25]    Status (0 = OK, non-zero = error)
      [26]    Added status size (number of added_status bytes)
      [27]    Padding
      [28-29] Added status (uint16 LE)
      [30-31] Padding
    """

    command_echo: int = 0          # if server echoes, for debug
    instance_echo: int = 0
    service: int = 0
    status: int = 0
    added_status_size: int = 0
    added_status: int = 0
    request_id: int = 0
    division: int = 0
    payload: bytes = field(default_factory=bytes)

    @classmethod
    def decode(cls, data: bytes) -> "HSEResponse":
        """Parse bytes from socket.recv → HSEResponse. Raises HSEDecodeError on failure."""
        if len(data) < HSE_HEADER_SIZE:
            raise HSEDecodeError(
                f"Response too short: {len(data)} < {HSE_HEADER_SIZE} bytes"
            )

        (
            ident, header_size, payload_size, reserve1, division, ack, request_id,
            block_no, reserve2, service, status, added_size, _pad1,
            added_status, _pad2,
        ) = struct.unpack("<4s H H B B B B I 8s B B B B H H", data[:HSE_HEADER_SIZE])

        if ident != HSE_IDENTIFIER:
            raise HSEDecodeError(f"Invalid identifier: {ident!r}, expected YERC")
        if header_size != HSE_HEADER_SIZE:
            raise HSEDecodeError(f"Invalid header size: {header_size}")
        if ack != ACK_RESPONSE:
            raise HSEDecodeError(f"Invalid ACK byte: {ack} (expected 0x01 for response)")

        payload = data[HSE_HEADER_SIZE:HSE_HEADER_SIZE + payload_size]
        if len(payload) != payload_size:
            raise HSEDecodeError(
                f"Payload size mismatch: declared {payload_size}, got {len(payload)}"
            )

        return cls(
            service=service,
            status=status,
            added_status_size=added_size,
            added_status=added_status,
            request_id=request_id,
            division=division,
            payload=payload,
        )

    def raise_for_status(self) -> None:
        """Raise HSEResponseError if status != 0."""
        if self.status != 0:
            raise HSEResponseError(self.status, self.added_status)


# ───── Helpers parse joint position payload ─────


def parse_position_response(payload: bytes) -> dict[str, list[float]]:
    """Parse READ_POSITION payload (Command 0x75) into a dict.

    Per spec, the READ_POSITION payload returns at minimum:
      [0-3]    Data type (0x10 = pulse, 0x11 = base coord, ...)
      [4-7]    Figure / form
      [8-11]   Tool no
      [12-15]  User coord no
      [16-19]  Extended figure
      [20-23]  Axis 1 (S) — int32 LE, unit: pulse or 0.0001° depending on data type
      [24-27]  Axis 2 (L)
      [28-31]  Axis 3 (U)
      [32-35]  Axis 4 (R)
      [36-39]  Axis 5 (B)
      [40-43]  Axis 6 (T)
      [44-47]  Axis 7 (if present — GP7 has only 6 axes, set to 0)
      [48-51]  Axis 8

    Returns:
        dict {data_type, joints_raw, ...}. Caller converts pulse → degrees
        using the GP7 pulse/degree ratio (see `pulse_to_deg()`).
    """
    if len(payload) < 52:
        raise HSEDecodeError(f"Position payload too short: {len(payload)} < 52 bytes")

    fields = struct.unpack("<5I 8i", payload[:52])
    data_type = fields[0]
    joints_raw = list(fields[5:11])           # 6 axes for GP7
    return {
        "data_type": data_type,
        "form": fields[1],
        "tool_no": fields[2],
        "user_frame": fields[3],
        "extended_form": fields[4],
        "joints_raw": joints_raw,
    }


# Pulse/degree conversion ratio for Yaskawa GP7 (datasheet).
# Each axis has a different encoder ratio — these are the default values
# (Yaskawa GP7-A00 controller pack). Verify against the real controller via
# parameter file (PULSE_PER_DEG in SF#xxx).
GP7_PULSE_PER_DEG: tuple[float, ...] = (
    1341.4,  # S axis
    1341.4,  # L axis
    1341.4,  # U axis
    1000.0,  # R axis
    1000.0,  # B axis
    624.0,   # T axis
)


def pulse_to_deg(joints_pulse: list[int],
                 ratio: tuple[float, ...] = GP7_PULSE_PER_DEG) -> list[float]:
    """Convert joint pulse (raw from HSE) to degrees."""
    if len(joints_pulse) != len(ratio):
        raise ValueError(
            f"Joint count {len(joints_pulse)} != ratio count {len(ratio)} — wrong robot model?"
        )
    return [float(p) / r for p, r in zip(joints_pulse, ratio)]
