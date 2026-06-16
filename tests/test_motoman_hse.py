"""
test_motoman_hse.py
───────────────────
Verify MotomanHSEBackend round-trip request/response qua mock UDP socket.

KHÔNG cần YRC1000 — mock socket trả response giả lập.
"""
from __future__ import annotations

import socket
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
    HSEResponseError,
)
from src.orchestrator.backends.motoman_hse import (
    HSE_PORT_ROBOT,
    NETWORK_IO_BASE,
    MotionHaltedError,
    MotomanHSEBackend,
    RobotAlarmError,
)


def _build_response(
    status: int = 0,
    payload: bytes = b"",
    request_id: int = 0,
) -> bytes:
    """Build a synthetic HSE response packet."""
    header = struct.pack(
        "<4s H H B B B B I 8s B B B B H H",
        HSE_IDENTIFIER, HSE_HEADER_SIZE, len(payload),
        HSE_RESERVE1, 0x01, ACK_RESPONSE, request_id, 0,
        HSE_RESERVE2,
        0x01, status, 0, 0, 0, 0,
    )
    return header + payload


def _build_position_response(joints_deg: list[float]) -> bytes:
    """Build response for READ_POSITION with given joints in degrees."""
    pulses = [int(d * r) for d, r in zip(joints_deg, GP7_PULSE_PER_DEG)]
    payload = struct.pack(
        "<5I 8i",
        0x10, 0, 0, 0, 0,          # data_type=pulse + form + tool + frame + ext
        *pulses, 0, 0,             # 6 joints + 2 unused axis
    )
    return _build_response(payload=payload, request_id=1)


@pytest.fixture
def backend_with_mock_socket(monkeypatch):
    """MotomanHSEBackend with socket.sendto/recvfrom monkey-patched."""
    backend = MotomanHSEBackend(ip="192.168.1.100", port=HSE_PORT_ROBOT, timeout_s=0.5)
    # Skip the start-confirm (Running-bit assert) phase in unit tests — the mock
    # never asserts Running, so it would just burn the grace window. Real hardware
    # keeps the default. _wait_idle still issues its completion READ_STATUS +
    # READ_ALARM, so move-test response sequences include a final alarm poll.
    backend.start_confirm_timeout_s = 0.0

    # Inject mock socket vào backend
    mock_sock = MagicMock(spec=socket.socket)
    backend._sock = mock_sock                   # bypass connect()
    return backend, mock_sock


# ─────────────────────────────────────────────────────────────────────────
# Lifecycle
# ─────────────────────────────────────────────────────────────────────────


class TestLifecycle:
    def test_connect_opens_udp_socket(self, monkeypatch):
        created_sockets = []

        def fake_socket(family, type_):
            assert family == socket.AF_INET
            assert type_ == socket.SOCK_DGRAM
            mock = MagicMock()                  # plain mock — spec strict ẩn settimeout
            created_sockets.append(mock)
            return mock

        monkeypatch.setattr(socket, "socket", fake_socket)
        backend = MotomanHSEBackend(ip="192.168.1.100")
        backend.connect()
        assert len(created_sockets) == 1
        created_sockets[0].settimeout.assert_called_once_with(backend.timeout_s)

    def test_connect_idempotent(self, monkeypatch):
        backend = MotomanHSEBackend(ip="192.168.1.100")
        backend._sock = MagicMock()             # đã connect
        backend.connect()                        # gọi lần 2 — no error
        # Không sinh socket mới

    def test_disconnect_closes_socket(self, backend_with_mock_socket):
        backend, mock_sock = backend_with_mock_socket
        backend.disconnect()
        mock_sock.close.assert_called_once()
        assert backend._sock is None


# ─────────────────────────────────────────────────────────────────────────
# Joints() — read joint position
# ─────────────────────────────────────────────────────────────────────────


class TestJoints:
    def test_joints_returns_degrees(self, backend_with_mock_socket):
        backend, mock_sock = backend_with_mock_socket
        joints_truth = [10.0, -5.0, 20.0, 0.0, 30.0, -15.0]
        mock_sock.recvfrom.return_value = (
            _build_position_response(joints_truth), ("192.168.1.100", HSE_PORT_ROBOT),
        )
        result = backend.Joints()
        assert len(result) == 6
        for actual, expected in zip(result, joints_truth):
            assert abs(actual - expected) < 0.1     # < 0.1° conversion error

    def test_joints_sends_read_position_command(self, backend_with_mock_socket):
        backend, mock_sock = backend_with_mock_socket
        mock_sock.recvfrom.return_value = (
            _build_position_response([0.0] * 6), ("192.168.1.100", HSE_PORT_ROBOT),
        )
        backend.Joints()
        # sendto call gửi packet — kiểm tra command byte
        packet, addr = mock_sock.sendto.call_args[0]
        assert addr == ("192.168.1.100", HSE_PORT_ROBOT)
        cmd_field = struct.unpack("<H", packet[24:26])[0]
        assert cmd_field == Command.READ_POSITION == 0x75

    def test_joints_raises_on_controller_error(self, backend_with_mock_socket):
        backend, mock_sock = backend_with_mock_socket
        mock_sock.recvfrom.return_value = (
            _build_response(status=0x1F),       # robot not ready
            ("192.168.1.100", HSE_PORT_ROBOT),
        )
        with pytest.raises(HSEResponseError) as exc:
            backend.Joints()
        assert exc.value.status == 0x1F

    def test_joints_raises_on_timeout(self, backend_with_mock_socket):
        backend, mock_sock = backend_with_mock_socket
        mock_sock.recvfrom.side_effect = socket.timeout()
        with pytest.raises(TimeoutError, match="timeout"):
            backend.Joints()


# ─────────────────────────────────────────────────────────────────────────
# setDO — gripper control via network I/O
# ─────────────────────────────────────────────────────────────────────────


class TestSetDO:
    def test_setdo_close_sends_write_io_command(self, backend_with_mock_socket):
        backend, mock_sock = backend_with_mock_socket
        mock_sock.recvfrom.return_value = (_build_response(), ("192.168.1.100", 10040))

        backend.setDO(1, 1)                      # gripper index 1, close

        packet, _ = mock_sock.sendto.call_args[0]
        cmd_field = struct.unpack("<H", packet[24:26])[0]
        instance_field = struct.unpack("<H", packet[26:28])[0]
        service_byte = packet[29]
        payload = packet[HSE_HEADER_SIZE:]

        assert cmd_field == Command.WRITE_IO == 0x78
        assert instance_field == NETWORK_IO_BASE      # index=1 → bit 27010
        assert service_byte == 0x10                   # SET_ATTRIBUTE_SINGLE
        assert struct.unpack("<I", payload)[0] == 1   # value = 1

    def test_setdo_open_writes_zero(self, backend_with_mock_socket):
        backend, mock_sock = backend_with_mock_socket
        mock_sock.recvfrom.return_value = (_build_response(), ("192.168.1.100", 10040))
        backend.setDO(1, 0)
        packet, _ = mock_sock.sendto.call_args[0]
        payload = packet[HSE_HEADER_SIZE:]
        assert struct.unpack("<I", payload)[0] == 0

    def test_setdo_index_offsets_bit_address(self, backend_with_mock_socket):
        backend, mock_sock = backend_with_mock_socket
        mock_sock.recvfrom.return_value = (_build_response(), ("192.168.1.100", 10040))
        backend.setDO(3, 1)
        packet, _ = mock_sock.sendto.call_args[0]
        instance_field = struct.unpack("<H", packet[26:28])[0]
        # index=3 → bit 27010 + 2 = 27012
        assert instance_field == NETWORK_IO_BASE + 2


# ─────────────────────────────────────────────────────────────────────────
# Stub motion + IK
# ─────────────────────────────────────────────────────────────────────────


class TestStubsAndPoseRejection:
    def test_solveik_raises_not_implemented(self, backend_with_mock_socket):
        backend, _ = backend_with_mock_socket
        with pytest.raises(NotImplementedError):
            backend.SolveIK(None)

    def test_movej_test_returns_zero_ok(self, backend_with_mock_socket):
        backend, _ = backend_with_mock_socket
        assert backend.MoveJ_Test([0] * 6, None) == 0

    def test_movej_with_4x4_pose_routes_to_cartesian_path(
        self, backend_with_mock_socket, monkeypatch,
    ):
        """4x4 pose → MoveJ Cartesian path (YRC IK). Verify _move_pose called."""
        import numpy as np
        backend, _ = backend_with_mock_socket
        called: dict[str, object] = {}
        monkeypatch.setattr(
            backend, "_move_pose",
            lambda T, kind: called.update(T=T, kind=kind),
        )
        T_target = np.eye(4)
        T_target[:3, 3] = [500.0, 200.0, 600.0]
        backend.MoveJ(T_target)
        assert called["kind"] == "movj"
        np.testing.assert_array_equal(called["T"], T_target)

    def test_movel_with_4x4_pose_routes_to_cartesian_path(
        self, backend_with_mock_socket, monkeypatch,
    ):
        import numpy as np
        backend, _ = backend_with_mock_socket
        called: dict[str, object] = {}
        monkeypatch.setattr(
            backend, "_move_pose",
            lambda T, kind: called.update(T=T, kind=kind),
        )
        backend.MoveL(np.eye(4))
        assert called["kind"] == "movl"

    def test_is_pose_4x4_detects_numpy_and_list(self, backend_with_mock_socket):
        import numpy as np
        backend, _ = backend_with_mock_socket
        assert backend._is_pose_4x4(np.eye(4)) is True
        assert backend._is_pose_4x4([[1, 0, 0, 0]] * 4) is True
        assert backend._is_pose_4x4([0, 0, 0, 0, 0, 0]) is False           # joints
        assert backend._is_pose_4x4("not a pose") is False

    def test_supports_cartesian_pose_class_flag(self):
        from src.orchestrator.backends.motoman_hse import MotomanHSEBackend
        assert MotomanHSEBackend.supports_cartesian_pose is True


# ─────────────────────────────────────────────────────────────────────────
# MoveJ wire-up: INFORM gen + FTP upload + JOB_SELECT + START + wait_idle
# ─────────────────────────────────────────────────────────────────────────


class TestMoveJEndToEnd:
    def test_movej_uploads_job_selects_starts_waits(
        self, backend_with_mock_socket, monkeypatch
    ):
        backend, mock_sock = backend_with_mock_socket

        # FTP mock — capture upload contents
        from unittest.mock import MagicMock
        uploaded = {}
        mock_ftp_cls = MagicMock()
        mock_ftp_inst = MagicMock()
        mock_ftp_cls.return_value = mock_ftp_inst

        def capture_stor(cmd, stream):
            uploaded["cmd"] = cmd
            uploaded["data"] = stream.read()
        mock_ftp_inst.storbinary.side_effect = capture_stor

        import ftplib
        monkeypatch.setattr(ftplib, "FTP", mock_ftp_cls)

        # Socket: trả ACK cho mọi command + status payload với bit Running tắt
        # cho READ_STATUS poll.
        responses = [
            _build_response(),                       # JOB_SELECT
            _build_response(),                       # START
            _build_response(payload=b"\x00"),        # READ_STATUS — idle (bit 1 = 0)
            _build_response(payload=b"\x00"),        # READ_ALARM — none (len<8 → (0,0))
        ]
        mock_sock.recvfrom.side_effect = [
            (r, ("192.168.1.100", 10040)) for r in responses
        ]

        backend.MoveJ([10.0, -5.0, 20.0, 0.0, 15.0, -10.0])

        # FTP đã được gọi đúng
        mock_ftp_cls.assert_called_once()
        mock_ftp_inst.login.assert_called_once()
        mock_ftp_inst.cwd.assert_called_once_with("/JOB")
        assert "STOR " in uploaded["cmd"]
        assert b"/JOB" in uploaded["data"]              # INFORM header
        assert b"MOVJ" in uploaded["data"]              # motion instruction
        # HSE sent 4 packets: JOB_SELECT, START, READ_STATUS, READ_ALARM
        assert mock_sock.sendto.call_count == 4

    def test_movej_wait_idle_times_out(self, backend_with_mock_socket, monkeypatch):
        backend, mock_sock = backend_with_mock_socket
        backend.wait_completion_timeout_s = 0.3       # short timeout cho test

        from unittest.mock import MagicMock
        mock_ftp_cls = MagicMock()
        mock_ftp_inst = MagicMock()
        mock_ftp_cls.return_value = mock_ftp_inst
        import ftplib
        monkeypatch.setattr(ftplib, "FTP", mock_ftp_cls)

        # READ_STATUS luôn trả bit Running=ON → wait_idle timeout.
        # Data1 bit3 (0x08) = "Running/operating" per Yaskawa HSE spec
        # (bit1/0x02 = "1-Cycle" mode, không phải trạng thái chạy).
        def respond(*_a, **_k):
            return _build_response(payload=b"\x08"), ("192.168.1.100", 10040)
        mock_sock.recvfrom.side_effect = respond

        with pytest.raises(TimeoutError, match="did not complete"):
            backend.MoveJ([0] * 6)


# ─────────────────────────────────────────────────────────────────────────
# JOB_SELECT payload format
# ─────────────────────────────────────────────────────────────────────────


class TestBatchMode:
    """Batch mode: gom MoveJ/MoveL/setDO/timer vào 1 INFORM job."""

    @pytest.fixture
    def backend_with_ftp_mock(self, backend_with_mock_socket, monkeypatch):
        backend, mock_sock = backend_with_mock_socket
        from unittest.mock import MagicMock
        uploaded = {}
        mock_ftp_cls = MagicMock()
        mock_ftp_inst = MagicMock()
        mock_ftp_cls.return_value = mock_ftp_inst

        def capture_stor(cmd, stream):
            uploaded["cmd"] = cmd
            uploaded["data"] = stream.read()

        mock_ftp_inst.storbinary.side_effect = capture_stor
        import ftplib
        monkeypatch.setattr(ftplib, "FTP", mock_ftp_cls)

        # Socket: JOB_SELECT + START + READ_STATUS idle + READ_ALARM none
        responses = [
            _build_response(),                               # JOB_SELECT
            _build_response(),                               # START
            _build_response(payload=b"\x00"),                # READ_STATUS idle
            _build_response(payload=b"\x00"),                # READ_ALARM none
        ]
        mock_sock.recvfrom.side_effect = [
            (r, ("x", 10040)) for r in responses
        ]
        return backend, mock_sock, mock_ftp_inst, uploaded

    def test_batch_collects_into_single_upload(self, backend_with_ftp_mock):
        backend, mock_sock, mock_ftp, uploaded = backend_with_ftp_mock

        with backend.batch("PICKTRIAL"):
            backend.MoveJ([10, 0, 0, 0, 0, 0])
            backend.MoveL([20, 0, 0, 0, 0, 0])
            backend.setDO(1, 1)
            backend.timer(0.3)
            backend.MoveJ([30, 0, 0, 0, 0, 0])

        # FTP called đúng 1 lần — không phải 3 lần (cho 3 motion)
        mock_ftp.storbinary.assert_called_once()
        # HSE socket: JOB_SELECT + START + READ_STATUS + READ_ALARM (4 packets) —
        # KHÔNG có WRITE_IO/MoveJ packet riêng vì batch
        assert mock_sock.sendto.call_count == 4

        # Verify nội dung INFORM bao gồm tất cả instructions
        data = uploaded["data"]
        assert b"MOVJ" in data
        assert b"MOVL" in data
        assert b"DOUT OT#(1) ON" in data
        assert b"TIMER T=0.300" in data

    def test_batch_outside_falls_back_to_single_shot(self, backend_with_mock_socket, monkeypatch):
        """Ngoài batch: mỗi MoveJ tự upload độc lập như trước (preserved behavior)."""
        backend, mock_sock = backend_with_mock_socket
        from unittest.mock import MagicMock
        mock_ftp_cls = MagicMock()
        mock_ftp_cls.return_value = MagicMock()
        import ftplib
        monkeypatch.setattr(ftplib, "FTP", mock_ftp_cls)

        mock_sock.recvfrom.side_effect = [
            (_build_response(), ("x", 10040)),               # JOB_SELECT
            (_build_response(), ("x", 10040)),               # START
            (_build_response(payload=b"\x00"), ("x", 10040)),  # READ_STATUS idle
            (_build_response(payload=b"\x00"), ("x", 10040)),  # READ_ALARM none
        ]
        backend.MoveJ([10, 0, 0, 0, 0, 0])
        # Non-batch path: FTP called 1 lần cho MoveJ này
        mock_ftp_cls.return_value.storbinary.assert_called_once()

    def test_batch_nested_raises(self, backend_with_ftp_mock):
        backend, *_ = backend_with_ftp_mock
        with backend.batch():
            with pytest.raises(RuntimeError, match="Nested"):
                with backend.batch():
                    pass

    def test_batch_empty_skips_upload(self, backend_with_mock_socket, monkeypatch):
        backend, mock_sock = backend_with_mock_socket
        from unittest.mock import MagicMock
        mock_ftp_cls = MagicMock()
        mock_ftp_cls.return_value = MagicMock()
        import ftplib
        monkeypatch.setattr(ftplib, "FTP", mock_ftp_cls)
        # Không setup socket response — sẽ raise nếu HSE bị gọi

        with backend.batch():
            pass                                              # không add gì

        mock_ftp_cls.return_value.storbinary.assert_not_called()
        mock_sock.sendto.assert_not_called()

    def test_batch_setdo_uses_dout_not_write_io(self, backend_with_ftp_mock):
        """Trong batch, setDO append DOUT vào INFORM, không gọi WRITE_IO HSE."""
        backend, mock_sock, mock_ftp, uploaded = backend_with_ftp_mock

        with backend.batch():
            backend.MoveJ([0] * 6)
            backend.setDO(1, 1)

        # Sock có JOB_SELECT + START + READ_STATUS + READ_ALARM (4 packets), không
        # có WRITE_IO packet (vì DOUT đi qua INFORM)
        assert mock_sock.sendto.call_count == 4
        data = uploaded["data"]
        assert b"DOUT OT#(1) ON" in data

    def test_batch_timer_uses_inform_not_sleep(self, backend_with_ftp_mock):
        """timer() trong batch → INFORM TIMER, không time.sleep."""
        import time as _time
        backend, *_, uploaded = backend_with_ftp_mock

        t_start = _time.time()
        with backend.batch():
            backend.MoveJ([0] * 6)
            backend.timer(2.0)                                # nếu sleep thật → mất 2s
        elapsed = _time.time() - t_start
        assert elapsed < 1.0, f"timer() trong batch sleep thật ({elapsed:.2f}s)"
        assert b"TIMER T=2.000" in uploaded["data"]

    def test_batch_exception_drops_builder_no_upload(self, backend_with_mock_socket, monkeypatch):
        backend, mock_sock = backend_with_mock_socket
        from unittest.mock import MagicMock
        mock_ftp_cls = MagicMock()
        mock_ftp_cls.return_value = MagicMock()
        import ftplib
        monkeypatch.setattr(ftplib, "FTP", mock_ftp_cls)

        with pytest.raises(ValueError):
            with backend.batch():
                backend.MoveJ([0] * 6)
                raise ValueError("user code error")

        # Sau exception: builder cleared, không upload
        assert backend._batch_builder is None
        mock_ftp_cls.return_value.storbinary.assert_not_called()
        # Vẫn dùng được sau khi exception (state đã reset)
        assert backend._batch_builder is None

    def test_batch_halted_skips_dispatch(self, backend_with_mock_socket, monkeypatch):
        """Review #6: a Stop() that lands mid-batch latches _halted → the batch
        __exit__ must NOT upload/JOB_SELECT/START (no command after halt)."""
        backend, mock_sock = backend_with_mock_socket
        from unittest.mock import MagicMock
        mock_ftp_cls = MagicMock(); mock_ftp_cls.return_value = MagicMock()
        import ftplib
        monkeypatch.setattr(ftplib, "FTP", mock_ftp_cls)

        with backend.batch():
            backend.MoveJ([10, 0, 0, 0, 0, 0])
            backend._halted = True                      # simulate Stop() mid-batch
        # No FTP upload and no HSE dispatch packets (JOB_SELECT/START) were sent.
        mock_ftp_cls.return_value.storbinary.assert_not_called()
        assert mock_sock.sendto.call_count == 0
        assert backend._batch_builder is None


class TestReadAlarm:
    def test_read_alarm_zero_no_alarm(self, backend_with_mock_socket):
        backend, mock_sock = backend_with_mock_socket
        # Payload: code=0, sub_code=0 (8 bytes)
        payload = struct.pack("<II", 0, 0)
        mock_sock.recvfrom.return_value = (
            _build_response(payload=payload), ("x", 10040),
        )
        code, sub = backend.read_alarm()
        assert (code, sub) == (0, 0)

    def test_read_alarm_returns_code_sub(self, backend_with_mock_socket):
        backend, mock_sock = backend_with_mock_socket
        payload = struct.pack("<II", 2010, 1)        # emergency stop, sub 1
        mock_sock.recvfrom.return_value = (
            _build_response(payload=payload), ("x", 10040),
        )
        assert backend.read_alarm() == (2010, 1)

    def test_read_alarm_sends_command_70(self, backend_with_mock_socket):
        backend, mock_sock = backend_with_mock_socket
        payload = struct.pack("<II", 0, 0)
        mock_sock.recvfrom.return_value = (
            _build_response(payload=payload), ("x", 10040),
        )
        backend.read_alarm()
        packet, _ = mock_sock.sendto.call_args[0]
        cmd_field = struct.unpack("<H", packet[24:26])[0]
        assert cmd_field == 0x70                     # READ_ALARM

    def test_read_alarm_short_payload_returns_zero(self, backend_with_mock_socket):
        backend, mock_sock = backend_with_mock_socket
        mock_sock.recvfrom.return_value = (
            _build_response(payload=b"\x00\x00"),    # only 2 bytes
            ("x", 10040),
        )
        assert backend.read_alarm() == (0, 0)


class TestJobSelectPayload:
    def test_job_select_pads_name_to_32_bytes(self, backend_with_mock_socket):
        backend, mock_sock = backend_with_mock_socket
        mock_sock.recvfrom.return_value = (_build_response(), ("x", 10040))
        backend.job_select("PICKTEST")
        packet, _ = mock_sock.sendto.call_args[0]
        payload = packet[HSE_HEADER_SIZE:]
        assert len(payload) == 32 + 4              # 32-byte name + 4-byte line
        # First 8 chars = "PICKTEST", rest null-padded
        assert payload[:8] == b"PICKTEST"
        assert payload[8:32] == b"\x00" * 24
        # Line number (last 4 bytes LE) = 0
        assert struct.unpack("<I", payload[32:])[0] == 0

    def test_job_select_strips_jbi_suffix(self, backend_with_mock_socket):
        backend, mock_sock = backend_with_mock_socket
        mock_sock.recvfrom.return_value = (_build_response(), ("x", 10040))
        backend.job_select("PICKTEST.JBI")
        packet, _ = mock_sock.sendto.call_args[0]
        # Name section bắt đầu với "PICKTEST", không có ".JBI"
        assert packet[HSE_HEADER_SIZE:HSE_HEADER_SIZE + 8] == b"PICKTEST"


# ─────────────────────────────────────────────────────────────────────────
# Home joints config
# ─────────────────────────────────────────────────────────────────────────


class TestHomeJoints:
    def test_jointshome_default_zeros(self):
        backend = MotomanHSEBackend(ip="x")
        assert backend.JointsHome() == [0.0] * 6

    def test_set_home_joints(self):
        backend = MotomanHSEBackend(ip="x")
        backend.set_home_joints([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert backend.JointsHome() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_set_home_joints_wrong_count_raises(self):
        backend = MotomanHSEBackend(ip="x")
        with pytest.raises(ValueError):
            backend.set_home_joints([1.0, 2.0, 3.0])


# ─────────────────────────────────────────────────────────────────────────
# Direct real-time motion (MOVE_PULSE 0x8B) + servo on
# ─────────────────────────────────────────────────────────────────────────


class TestDirectMove:
    def test_move_pulse_encodes_88_byte_payload(self, backend_with_mock_socket):
        backend, mock_sock = backend_with_mock_socket
        mock_sock.recvfrom.return_value = (
            _build_response(), ("192.168.1.100", 10040))
        backend.move_pulse([1000, -2000, 3000, 4000, -5000, 6000],
                           speed_pct=5.0, tool_no=1)
        packet, _ = mock_sock.sendto.call_args[0]
        cmd = struct.unpack("<H", packet[24:26])[0]
        inst = struct.unpack("<H", packet[26:28])[0]
        service = packet[29]
        payload = packet[HSE_HEADER_SIZE:]
        assert cmd == 0x8B                      # MOVE_PULSE
        assert inst == 1                        # joint absolute
        assert service == 0x02                  # SET_ATTRIBUTE_ALL
        assert len(payload) == 88
        robot, station, sclass, speed = struct.unpack("<4I", payload[:16])
        pulses = struct.unpack("<7i", payload[16:44])
        reserved, tool = struct.unpack("<2I", payload[44:52])
        assert robot == 1 and station == 0
        assert sclass == 0                      # PERCENT
        assert speed == 500                     # 5.00 % in 0.01% units
        assert pulses == (1000, -2000, 3000, 4000, -5000, 6000, 0)
        assert reserved == 0 and tool == 1

    def test_move_joints_converts_deg_to_pulse(self, backend_with_mock_socket):
        backend, mock_sock = backend_with_mock_socket
        mock_sock.recvfrom.return_value = (
            _build_response(), ("192.168.1.100", 10040))
        backend.move_joints([10.0, 0, 0, 0, 0, 0], speed_pct=5.0)
        packet, _ = mock_sock.sendto.call_args[0]
        payload = packet[HSE_HEADER_SIZE:]
        s_pulse = struct.unpack("<i", payload[16:20])[0]
        assert s_pulse == round(10.0 * 1241.212)   # 12412

    def test_servo_on_sends_power_type_2_switch_1(self, backend_with_mock_socket):
        backend, mock_sock = backend_with_mock_socket
        mock_sock.recvfrom.return_value = (
            _build_response(), ("192.168.1.100", 10040))
        backend.servo_on()
        packet, _ = mock_sock.sendto.call_args[0]
        cmd = struct.unpack("<H", packet[24:26])[0]
        inst = struct.unpack("<H", packet[26:28])[0]
        value = struct.unpack("<I", packet[HSE_HEADER_SIZE:])[0]
        assert cmd == 0x83 and inst == 2 and value == 1

    def test_move_pulse_rejects_wrong_axis_count(self, backend_with_mock_socket):
        backend, _ = backend_with_mock_socket
        with pytest.raises(ValueError):
            backend.move_pulse([1, 2, 3], speed_pct=5.0)      # only 3 axes

    def test_move_pulse_rejects_nonpositive_speed(self, backend_with_mock_socket):
        backend, _ = backend_with_mock_socket
        with pytest.raises(ValueError):
            backend.move_pulse([0, 0, 0, 0, 0, 0], speed_pct=0.0)

    def test_move_joints_rejects_wrong_count(self, backend_with_mock_socket):
        backend, _ = backend_with_mock_socket
        with pytest.raises(ValueError):
            backend.move_joints([10.0, 0, 0], speed_pct=5.0)   # only 3 joints

    def test_move_joints_rejects_out_of_limit(self, backend_with_mock_socket):
        # Defense-in-depth: an out-of-limit absolute target is refused (S limit ±170).
        backend, _ = backend_with_mock_socket
        with pytest.raises(ValueError, match="outside GP7 limit"):
            backend.move_joints([200.0, 0, 0, 0, 0, 0], speed_pct=5.0)

    def test_move_pulse_caps_speed_at_max_speed_pct(self, backend_with_mock_socket):
        # Direct-path speed is capped to the configured max even if the caller asks more.
        backend, mock_sock = backend_with_mock_socket
        backend.max_speed_pct = 10.0
        mock_sock.recvfrom.return_value = (_build_response(), ("192.168.1.100", 10040))
        backend.move_pulse([0, 0, 0, 0, 0, 0], speed_pct=80.0)   # request 80%
        packet, _ = mock_sock.sendto.call_args[0]
        speed = struct.unpack("<I", packet[HSE_HEADER_SIZE + 12:HSE_HEADER_SIZE + 16])[0]
        assert speed == 1000                    # capped to 10.00% (not 8000)


class TestUploadOverwrite:
    """YRC1000 FTP refuses STOR over an existing job ('503 Can't overwrite
    JOB-file') → upload_job must DELE first."""

    def _ftp(self, monkeypatch):
        import ftplib
        from unittest.mock import MagicMock
        ftp = MagicMock()
        monkeypatch.setattr(ftplib, "FTP", MagicMock(return_value=ftp))
        return ftp

    def test_delete_before_stor(self, monkeypatch):
        from src.orchestrator.backends.motoman_hse import MotomanHSEBackend
        order = []
        ftp = self._ftp(monkeypatch)
        ftp.delete.side_effect = lambda n: order.append(("delete", n))
        ftp.storbinary.side_effect = lambda cmd, s: order.append(("stor", cmd))
        MotomanHSEBackend(ip="x").upload_job("//NAME T\nNOP\nEND\n", "PROG")
        assert [k for k, _ in order] == ["delete", "stor"]       # delete first
        assert order[0][1] == "PROG.JBI"

    def test_missing_file_550_on_delete_is_tolerated(self, monkeypatch):
        import ftplib
        from src.orchestrator.backends.motoman_hse import MotomanHSEBackend
        stored = {}
        ftp = self._ftp(monkeypatch)
        ftp.delete.side_effect = ftplib.error_perm("550 File not found")
        ftp.storbinary.side_effect = lambda cmd, s: stored.setdefault("cmd", cmd)
        MotomanHSEBackend(ip="x").upload_job("X", "PROG")         # must NOT raise
        assert stored["cmd"] == "STOR PROG.JBI"

    def test_active_job_overwrite_refused_gives_hint(self, monkeypatch):
        import ftplib
        from src.orchestrator.backends.motoman_hse import MotomanHSEBackend
        ftp = self._ftp(monkeypatch)
        ftp.delete.side_effect = ftplib.error_perm(
            "503 Bad sequence of commands. Can't overwrite JOB-file.")
        with pytest.raises(RuntimeError, match="teach pendant"):
            MotomanHSEBackend(ip="x").upload_job("X", "PROG")

    def test_stor_451_close_error_gives_actionable_message(self, monkeypatch):
        """451 'Error closing file' is FTP 4xx (error_temp): the controller took
        the bytes but refused to SAVE the job (account/mode/lock). Must surface a
        clear, actionable message — NOT a JobUploadError rename prompt."""
        import ftplib
        from src.orchestrator.backends.motoman_hse import (
            JobUploadError, MotomanHSEBackend)
        ftp = self._ftp(monkeypatch)
        ftp.delete.side_effect = ftplib.error_perm("550 File not found")   # no file yet
        ftp.storbinary.side_effect = ftplib.error_temp(
            "451 Error closing file.--[5130]--")
        with pytest.raises(RuntimeError, match="refused to SAVE") as ei:
            MotomanHSEBackend(ip="x").upload_job("X", "PROG")
        assert not isinstance(ei.value, JobUploadError)   # not the rename path
        assert "WRITE permission" in str(ei.value)

    def test_stor_425_data_connection_not_mislabeled_as_save_refusal(self, monkeypatch):
        """A genuine transient 4xx (425 data-connection) must NOT be reported as a
        controller save-refusal — that would send the operator down the wrong path."""
        import ftplib
        from src.orchestrator.backends.motoman_hse import MotomanHSEBackend
        ftp = self._ftp(monkeypatch)
        ftp.delete.side_effect = ftplib.error_perm("550 File not found")
        ftp.storbinary.side_effect = ftplib.error_temp("425 Can't open data connection")
        with pytest.raises(RuntimeError, match="network") as ei:
            MotomanHSEBackend(ip="x").upload_job("X", "PROG")
        assert "refused to SAVE" not in str(ei.value)


# ─────────────────────────────────────────────────────────────────────────
# Motion-safety: halt latch + alarm-aware completion (bug-hunt HIGH cluster)
# ─────────────────────────────────────────────────────────────────────────


class TestMotionSafety:
    def test_wait_idle_raises_on_alarm(self, backend_with_mock_socket):
        """An alarm-aborted job must NOT be reported as success: when Running
        clears, _wait_idle checks READ_ALARM and raises if nonzero (#3)."""
        backend, mock_sock = backend_with_mock_socket
        alarm = struct.pack("<I", 2010) + struct.pack("<I", 3)   # code 2010, sub 3
        mock_sock.recvfrom.side_effect = [
            (_build_response(payload=b"\x00"), ("x", 10040)),     # READ_STATUS idle
            (_build_response(payload=alarm), ("x", 10040)),       # READ_ALARM → 2010/3
        ]
        with pytest.raises(RobotAlarmError) as ei:
            backend._wait_idle()
        assert ei.value.code == 2010 and ei.value.sub_code == 3

    def test_wait_idle_ok_when_no_alarm(self, backend_with_mock_socket):
        backend, mock_sock = backend_with_mock_socket
        mock_sock.recvfrom.side_effect = [
            (_build_response(payload=b"\x00"), ("x", 10040)),     # READ_STATUS idle
            (_build_response(payload=b"\x00"), ("x", 10040)),     # READ_ALARM none
        ]
        backend._wait_idle()                                     # must not raise

    def test_job_start_refused_when_halted(self, backend_with_mock_socket):
        """Stop() latched _halted → job_start refuses to START (no motion). (#4/#17)"""
        backend, mock_sock = backend_with_mock_socket
        backend._halted = True
        with pytest.raises(MotionHaltedError):
            backend.job_start()
        mock_sock.sendto.assert_not_called()                    # nothing was sent

    def test_move_pulse_refused_when_halted(self, backend_with_mock_socket):
        """Direct MOVE path is gated on the halt latch too (live-jog can't stream
        after a Stop). (#17)"""
        backend, mock_sock = backend_with_mock_socket
        backend._halted = True
        with pytest.raises(MotionHaltedError):
            backend.move_pulse([0, 0, 0, 0, 0, 0], speed_pct=5.0)
        mock_sock.sendto.assert_not_called()

    def test_servo_on_clears_halt_latch(self, backend_with_mock_socket):
        backend, mock_sock = backend_with_mock_socket
        backend._halted = True
        mock_sock.recvfrom.return_value = (_build_response(), ("x", 10040))
        backend.servo_on()
        assert backend._halted is False                         # resume cleared it

    def test_wait_idle_raises_if_job_never_starts(self, backend_with_mock_socket):
        """If Running never asserts within start_confirm_timeout_s, the job did NOT
        start → raise, never fall through to a false 'completed' (regression guard
        for the start-confirm fix)."""
        backend, mock_sock = backend_with_mock_socket
        backend.start_confirm_timeout_s = 0.2          # short confirm window
        backend.wait_completion_timeout_s = 5.0
        # READ_STATUS always idle (Running never asserts), READ_ALARM none.
        mock_sock.recvfrom.return_value = (_build_response(payload=b"\x00"), ("x", 10040))
        with pytest.raises(TimeoutError, match="did not start"):
            backend._wait_idle()

    def test_wait_idle_confirms_running_then_completes(self, backend_with_mock_socket):
        """With start-confirm ON: Running asserts, then clears → success (no false
        early return inside start latency)."""
        backend, mock_sock = backend_with_mock_socket
        backend.start_confirm_timeout_s = 1.0
        mock_sock.recvfrom.side_effect = [
            (_build_response(payload=b"\x08"), ("x", 10040)),   # READ_STATUS running
            (_build_response(payload=b"\x00"), ("x", 10040)),   # READ_STATUS idle
            (_build_response(payload=b"\x00"), ("x", 10040)),   # READ_ALARM none
        ]
        backend._wait_idle()                            # must not raise
