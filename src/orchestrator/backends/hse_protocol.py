"""
hse_protocol.py
───────────────
Codec cho Yaskawa High-Speed Ethernet Server protocol.

Tham chiếu spec:
  - Yaskawa "High-Speed Ethernet Server Function" manual (HW1485553-1)
  - Port UDP 10040 cho robot commands, UDP 10041 cho file load/save
  - Mỗi packet: 32-byte sub-header + variable payload (≤ ~478 byte)

Module này PURE — chỉ encode/decode byte sequence. Socket I/O ở `motoman_hse.py`.
Test được bằng known byte sequences, không cần YRC1000 thật.

Conventions:
  - Tất cả integer multi-byte = LITTLE ENDIAN (Yaskawa default cho YRC1000)
  - Identifier "YERC" ASCII = 0x59 0x45 0x52 0x43
  - Reserve2 = "99999999" ASCII (yes, 8 ký tự '9')

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
HSE_RESERVE1 = 0x03               # luôn 0x03
HSE_RESERVE2 = b"99999999"        # 8 ký tự ASCII '9'

DIVISION_ROBOT = 0x01             # robot command service
DIVISION_FILE = 0x02              # file command service

ACK_REQUEST = 0x00                # client → server
ACK_RESPONSE = 0x01               # server → client


class Service(IntEnum):
    """HSE service code (byte 29 trong request sub-header)."""

    GET_ATTRIBUTE_ALL = 0x01
    SET_ATTRIBUTE_ALL = 0x02
    GET_ATTRIBUTE_SINGLE = 0x0E
    SET_ATTRIBUTE_SINGLE = 0x10


class Command(IntEnum):
    """HSE command code thường dùng (byte 24-25, little-endian).

    Danh sách rút gọn — manual có ~30 commands. Thêm khi cần.
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
    """Position variable data type field cho WRITE_POSITION_VAR / READ_POSITION.

    Per Yaskawa HSE spec HW1485553. Quyết định cách interpret 8 axis values:
      - PULSE: joint angles in encoder pulse units (cần convert qua pulse_per_deg)
      - BASE: Cartesian pose trong robot BASE frame (X,Y,Z,Rx,Ry,Rz)
      - ROBOT: giống BASE cho non-track robot (single arm GP7)
      - USER01-63: Cartesian trong user-defined frame
      - TOOL: Cartesian trong current tool frame (hiếm dùng)
    """

    PULSE = 0
    BASE = 16                       # Robot base coordinate (recommended cho pick-place)
    ROBOT = 17                      # = BASE cho GP7 (no track axis)
    USER01 = 18                     # USER01 frame; +N cho USER02..USER63 (max 80)
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
    """Encode Cartesian pose → 52-byte payload cho WRITE_POSITION_VAR (cmd 0x7F).

    Layout giống joint variant nhưng axis values là Cartesian thay vì pulse:
        [0-3]    data_type (BASE=16 cho robot base frame)
        [4-7]    form bits (IK config branch — bit 0=back, 1=elbow, 2=flip, ...)
        [8-11]   tool_no (0-63, gripper TCP)
        [12-15]  user_frame (0 nếu BASE, 1-63 nếu USER)
        [16-19]  extended_form (0 cho GP7 — không có track)
        [20-23]  X position (int32, đơn vị 0.001mm = μm)
        [24-27]  Y position
        [28-31]  Z position
        [32-35]  Rx rotation (int32, đơn vị 0.0001° degree)
        [36-39]  Ry rotation
        [40-43]  Rz rotation
        [44-47]  Reserved (0)
        [48-51]  Reserved (0)

    Convention rotation: Yaskawa dùng XYZ-fixed (= ZYX-intrinsic Tait-Bryan).
    Caller phải convert pose matrix → (rx, ry, rz) đúng convention này — xem
    `frame_convert.matrix_to_rpy_yaskawa()`.

    Args:
        x_mm, y_mm, z_mm: Position trong frame chỉ định bởi `data_type`.
        rx_deg, ry_deg, rz_deg: Rotation degrees (XYZ-fixed convention).
        tool_no: 0-63 — gripper TCP đã setup trên teach pendant.
        user_frame: 0 cho BASE, 1-63 cho USER01-USER63.
        form: IK config bits (0 = front + elbow up + no flip — default pick-place).
        data_type: 16 (BASE), 17 (ROBOT), 18-80 (USER), 65 (TOOL).

    Returns:
        52-byte payload sẵn sàng gửi qua HSE.

    Raises:
        ValueError: Bất kỳ field nào ngoài int32 range sau khi scale.
    """
    if not (0 <= tool_no <= 63):
        raise ValueError(f"tool_no phải 0-63, got {tool_no}")
    if not (0 <= user_frame <= 63):
        raise ValueError(f"user_frame phải 0-63, got {user_frame}")
    if not (0 <= form <= 0xFF):
        raise ValueError(f"form bits phải 0-255 (8 bit), got {form}")

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
            raise ValueError(f"{name} value sau scale ngoài int32 range: {v}")

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
    """Base lỗi cho HSE codec / IO."""


class HSEDecodeError(HSEError):
    """Bytes không decode được (kích thước/identifier sai)."""


class HSEResponseError(HSEError):
    """Server trả status != 0 (lỗi từ controller).

    Status code tham khảo manual mục "Added status". Common:
      0x08 = sub command sai
      0x0E = data sai
      0x1F = robot không sẵn sàng
      0x23 = key switch không ở chế độ remote
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
    """Một HSE request packet (client → server).

    Encode ra 32-byte sub-header + payload. request_id để pair request/response
    khi gửi đồng thời nhiều command.
    """

    command: int                              # Command enum value
    instance: int = 0                         # data instance (vd robot 1 = 1)
    attribute: int = 0                        # attribute selector (0 = all)
    service: int = Service.GET_ATTRIBUTE_ALL  # GET hay SET
    payload: bytes = b""                      # data payload (cho SET)
    request_id: int = 0                       # 0-255, pair với response
    division: int = DIVISION_ROBOT
    block_no: int = 0

    def encode(self) -> bytes:
        """Pack thành bytes gửi qua UDP."""
        if not (0 <= self.request_id <= 0xFF):
            raise ValueError(f"request_id must be 0-255, got {self.request_id}")
        if not (0 <= self.command <= 0xFFFF):
            raise ValueError(f"command out of uint16 range: 0x{self.command:X}")

        payload_size = len(self.payload)
        if payload_size > 0xFFFF:
            raise ValueError(f"payload quá lớn: {payload_size} > 65535 bytes")

        # Sub-header 32 bytes (little-endian theo spec Yaskawa)
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
            f"header encode lỗi: {len(header)} != {HSE_HEADER_SIZE}"
        )
        return header + self.payload


# ───── Response packet ─────


@dataclass
class HSEResponse:
    """Một HSE response packet (server → client).

    Layout sub-header khác request (byte 24-29):
      [24]    Service (echo từ request)
      [25]    Status (0 = OK, non-zero = error)
      [26]    Added status size (số byte added_status)
      [27]    Padding
      [28-29] Added status (uint16 LE)
      [30-31] Padding
    """

    command_echo: int = 0          # nếu server echo, để debug
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
        """Parse bytes từ socket.recv → HSEResponse. Raise HSEDecodeError nếu sai."""
        if len(data) < HSE_HEADER_SIZE:
            raise HSEDecodeError(
                f"Response quá ngắn: {len(data)} < {HSE_HEADER_SIZE} bytes"
            )

        (
            ident, header_size, payload_size, reserve1, division, ack, request_id,
            block_no, reserve2, service, status, added_size, _pad1,
            added_status, _pad2,
        ) = struct.unpack("<4s H H B B B B I 8s B B B B H H", data[:HSE_HEADER_SIZE])

        if ident != HSE_IDENTIFIER:
            raise HSEDecodeError(f"Identifier sai: {ident!r}, mong YERC")
        if header_size != HSE_HEADER_SIZE:
            raise HSEDecodeError(f"Header size sai: {header_size}")
        if ack != ACK_RESPONSE:
            raise HSEDecodeError(f"ACK byte sai: {ack} (mong 0x01 cho response)")

        payload = data[HSE_HEADER_SIZE:HSE_HEADER_SIZE + payload_size]
        if len(payload) != payload_size:
            raise HSEDecodeError(
                f"Payload size lệch: declared {payload_size}, got {len(payload)}"
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
        """Raise HSEResponseError nếu status != 0."""
        if self.status != 0:
            raise HSEResponseError(self.status, self.added_status)


# ───── Helpers parse joint position payload ─────


def parse_position_response(payload: bytes) -> dict[str, list[float]]:
    """Parse payload READ_POSITION (Command 0x75) thành dict.

    Theo spec, payload READ_POSITION trả về tối thiểu:
      [0-3]    Data type (0x10 = pulse, 0x11 = base coord, ...)
      [4-7]    Figure / form
      [8-11]   Tool no
      [12-15]  User coord no
      [16-19]  Extended figure
      [20-23]  Axis 1 (S) — int32 LE, đơn vị pulse hoặc 0.0001° tuỳ data type
      [24-27]  Axis 2 (L)
      [28-31]  Axis 3 (U)
      [32-35]  Axis 4 (R)
      [36-39]  Axis 5 (B)
      [40-43]  Axis 6 (T)
      [44-47]  Axis 7 (nếu có — GP7 chỉ 6 axis, để 0)
      [48-51]  Axis 8

    Returns:
        dict {data_type, joints_raw, ...}. Caller convert pulse → degrees
        bằng pulse/degree ratio của GP7 (xem `pulse_to_deg()`).
    """
    if len(payload) < 52:
        raise HSEDecodeError(f"Position payload quá ngắn: {len(payload)} < 52 bytes")

    fields = struct.unpack("<5I 8i", payload[:52])
    data_type = fields[0]
    joints_raw = list(fields[5:11])           # 6 axis cho GP7
    return {
        "data_type": data_type,
        "form": fields[1],
        "tool_no": fields[2],
        "user_frame": fields[3],
        "extended_form": fields[4],
        "joints_raw": joints_raw,
    }


# Pulse/degree conversion ratio cho Yaskawa GP7 (datasheet).
# Mỗi axis có encoder ratio khác nhau — đây là giá trị mặc định
# (Yaskawa GP7-A00 controller pack). Verify lại trên controller thật qua
# parameter file (PULSE_PER_DEG trong SF#xxx).
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
    """Đổi joint pulse (raw từ HSE) sang degrees."""
    if len(joints_pulse) != len(ratio):
        raise ValueError(
            f"Số joint {len(joints_pulse)} != ratio {len(ratio)} — sai robot model?"
        )
    return [float(p) / r for p, r in zip(joints_pulse, ratio)]
