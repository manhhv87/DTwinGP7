"""
motoman_hse.py
──────────────
MotomanHSEBackend: nói chuyện thẳng với YRC1000 qua UDP HSE protocol.

KHÔNG cần RoboDK driver (chỉ cần RoboDK Free để visualize), KHÔNG cần MotoCom32
DLL, KHÔNG cần ROS 2 / MotoPlus flash. Chỉ cần:
  - YRC1000 đã enable "High-Speed Ethernet Server function" (Maintenance mode)
  - PC cùng subnet với YRC1000

Tính năng đã làm:
  - Connect / status / disconnect (M1)
  - Joints() đọc joint angles real-time → mirror lên digital twin (M1)
  - setDO(index, value) ghi network I/O (M2)

Còn thiếu (M3, follow-up):
  - MoveJ/MoveL: cần generator INFORM (.JBI) → FTP upload → JOB_SELECT + START.
    Khá phức tạp, viết riêng trong inform_codegen.py + sftp uploader.
    Stub raise NotImplementedError với hướng dẫn.

Lưu ý an toàn:
  - Setup chỉ nên test với robot trong REMOTE mode (key switch trên TP)
  - Default safety: setSpeed gọi với max_speed_percent từ config
  - Stop() gửi command HOLD_SERVO để dừng khẩn cấp (cũng có thể dùng E-stop vật lý)
"""
from __future__ import annotations

import contextlib
import ftplib
import io
import logging
import socket
import struct
import threading
import time
from typing import Any, Iterator

from .hse_protocol import (
    Command,
    HSEDecodeError,
    HSEResponse,
    HSERequest,
    Service,
    parse_position_response,
    pulse_to_deg,
)
from .inform_codegen import InformJobBuilder, gen_pvar_template_job

logger = logging.getLogger(__name__)

# Default UDP port theo Yaskawa HSE Server spec
HSE_PORT_ROBOT = 10040
HSE_PORT_FILE = 10041

# Network I/O range cho writable bits (YRC1000):
# 27010-27020 = general purpose network I/O (CIO ladder accessible).
# Mục đích cho gripper: bind 1 bit ở đây tới Y-output vật lý qua CIO ladder
# trên controller. Document trong setup guide.
NETWORK_IO_BASE = 27010


class MotomanHSEBackend:
    """Backend YRC1000 qua UDP HSE protocol.

    Duck-type khớp RoboDK Item interface (Joints, MoveJ, setDO, ...) để
    Orchestrator gọi nguyên văn.

    Args:
        ip: IP của YRC1000.
        port: UDP port (default 10040 cho robot command).
        timeout_s: Socket timeout cho mỗi request.
        request_id_seed: Giá trị bắt đầu của request_id (auto-increment).
    """

    def __init__(
        self,
        ip: str,
        port: int = HSE_PORT_ROBOT,
        timeout_s: float = 2.0,
        request_id_seed: int = 0,
        ftp_user: str = "",
        ftp_pass: str = "",
        ftp_job_dir: str = "/MPRAM1/JBI",
        max_speed_pct: float = 20.0,
        job_name_prefix: str = "DTWIN",
        wait_completion_timeout_s: float = 30.0,
        reach_envelope: Any = None,
    ) -> None:
        self.ip = ip
        self.port = port
        self.timeout_s = timeout_s
        self._sock: socket.socket | None = None
        self._request_id = request_id_seed % 256
        self._lock = threading.Lock()       # request_id ++ phải atomic
        # Cache cho JointsHome — set qua config từ orchestrator
        self._home_joints: list[float] = [0.0] * 6
        # FTP credentials cho job upload (YRC1000 mặc định "" / "")
        self.ftp_user = ftp_user
        self.ftp_pass = ftp_pass
        self.ftp_job_dir = ftp_job_dir
        # Safety + naming
        self.max_speed_pct = max_speed_pct
        self.job_name_prefix = job_name_prefix
        self._job_counter = 0
        self.wait_completion_timeout_s = wait_completion_timeout_s
        # Optional kinematic envelope cho MoveJ_Test client-side khi không có RoboDK item.
        self.reach_envelope = reach_envelope
        # Batch state: None = non-batch (mỗi MoveJ = 1 INFORM upload). Active builder
        # khi đang trong `with backend.batch():` — motion + IO calls gom vào đây.
        self._batch_builder: InformJobBuilder | None = None
        self._batch_pos_counter = 0
        self._batch_name = ""

        # Ultra-fast P-var mode state (M3++)
        self._ultra_fast_mode = False
        self._pvar_template_name = ""        # tên template đã upload
        self._pvar_template_signature = ""   # signature trial structure ("MMMDMM" etc.)
        # None = không trong batch. List = đang trong batch context (append ops).
        # Record format: ("movj", joints), ("movl", joints), ("dout", value_int),
        # ("timer", seconds_float) — collected during batch context.
        self._pvar_batch_buffer: list[tuple[str, Any]] | None = None

    # ─── Lifecycle ───
    def connect(self) -> None:
        """Mở UDP socket. KHÔNG verify với robot — gọi Joints() để verify."""
        if self._sock is not None:
            logger.debug("HSE socket already open")
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(self.timeout_s)
        self._sock = sock
        logger.info("HSE UDP socket mở (%s:%d, timeout %.1fs)", self.ip, self.port, self.timeout_s)

    def disconnect(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
            logger.info("HSE UDP socket đóng")

    def __del__(self) -> None:
        try:
            self.disconnect()
        except Exception:                       # noqa: BLE001
            pass

    # ─── Helpers ───
    def _next_request_id(self) -> int:
        with self._lock:
            self._request_id = (self._request_id + 1) % 256
            return self._request_id

    def _send_request(
        self,
        command: int,
        instance: int = 1,
        attribute: int = 0,
        service: int = Service.GET_ATTRIBUTE_ALL,
        payload: bytes = b"",
    ) -> HSEResponse:
        """Gửi request, chờ response. Raise nếu timeout / decode lỗi / status != 0."""
        if self._sock is None:
            raise RuntimeError("HSE socket chưa connect — gọi connect() trước")

        request_id = self._next_request_id()
        req = HSERequest(
            command=command, instance=instance, attribute=attribute,
            service=service, payload=payload, request_id=request_id,
        )
        packet = req.encode()
        self._sock.sendto(packet, (self.ip, self.port))

        try:
            raw, _ = self._sock.recvfrom(2048)
        except socket.timeout as e:
            raise TimeoutError(
                f"HSE request 0x{command:02X} timeout {self.timeout_s}s — "
                f"YRC1000 không phản hồi. Check ping + HSE Server function."
            ) from e

        try:
            resp = HSEResponse.decode(raw)
        except HSEDecodeError:
            logger.error("HSE response decode lỗi, raw bytes: %s", raw.hex())
            raise

        if resp.request_id != request_id:
            logger.warning(
                "HSE request_id mismatch: sent=%d got=%d", request_id, resp.request_id
            )

        resp.raise_for_status()
        return resp

    # ─── RoboDK Item-like interface ───
    def Valid(self) -> bool:
        """Heartbeat: thử ReadStatus. True nếu socket+controller alive."""
        try:
            self._send_request(Command.READ_STATUS, instance=1)
            return True
        except Exception as e:                  # noqa: BLE001
            logger.warning("HSE Valid() probe fail: %s", e)
            return False

    def Joints(self) -> list[float]:
        """Đọc joint angles hiện tại từ YRC1000 (degrees).

        Dùng READ_POSITION (0x75) với data_type pulse, rồi convert sang độ.
        """
        resp = self._send_request(
            Command.READ_POSITION, instance=1,
            service=Service.GET_ATTRIBUTE_ALL,
        )
        parsed = parse_position_response(resp.payload)
        return pulse_to_deg(parsed["joints_raw"])

    def JointsHome(self) -> list[float]:
        """Home joints — set qua config từ Orchestrator (cell_layout.yaml)."""
        return list(self._home_joints)

    def set_home_joints(self, joints_deg: list[float]) -> None:
        """Setter cho home — Orchestrator gọi sau init."""
        if len(joints_deg) != 6:
            raise ValueError(f"Home phải 6 joints, nhận {len(joints_deg)}")
        self._home_joints = list(joints_deg)

    def Parent(self) -> Any:
        """HSE không có concept frame parent (như RoboDK item)."""
        return None

    # ─── Job upload + execute (M3) ───
    def upload_job(self, job_text: str, job_name: str) -> None:
        """Upload INFORM .JBI text lên YRC1000 qua FTP.

        YRC1000 mặc định chạy FTP server trên port 21 (anonymous). File lưu
        vào `ftp_job_dir` (mặc định /MPRAM1/JBI/) để JOB_SELECT thấy được.

        Args:
            job_text: Nội dung INFORM (đã render từ InformJobBuilder).
            job_name: Tên file (không cần .JBI suffix — tự thêm).
        """
        filename = job_name if job_name.upper().endswith(".JBI") else f"{job_name}.JBI"
        # Yaskawa INFORM dùng Shift-JIS encoding nhưng ASCII-only an toàn cho cả 2.
        data = job_text.encode("ascii", errors="strict")

        logger.info("FTP upload '%s' (%d bytes) lên %s%s",
                    filename, len(data), self.ip, self.ftp_job_dir)
        ftp = ftplib.FTP()
        try:
            ftp.connect(self.ip, 21, timeout=self.timeout_s)
            ftp.login(self.ftp_user, self.ftp_pass)
            ftp.cwd(self.ftp_job_dir)
            ftp.storbinary(f"STOR {filename}", io.BytesIO(data))
        finally:
            try:
                ftp.quit()
            except Exception:                       # noqa: BLE001
                pass

    def job_select(self, job_name: str) -> None:
        """JOB_SELECT (0x87): chọn job đã upload làm current. Phải gọi trước START."""
        name = job_name.upper().removesuffix(".JBI").encode("ascii")
        # Payload: 32-byte job name (null-padded) + 4-byte line number (uint32 LE, 0=top)
        payload = name.ljust(32, b"\x00") + struct.pack("<I", 0)
        self._send_request(
            Command.JOB_SELECT, instance=1, attribute=1,
            service=Service.SET_ATTRIBUTE_ALL, payload=payload,
        )
        logger.debug("JOB_SELECT '%s'", job_name)

    def job_start(self) -> None:
        """START (0x86): execute current job. Robot phải đã servo on + REMOTE mode."""
        self._send_request(
            Command.START, instance=1, attribute=1,
            service=Service.SET_ATTRIBUTE_SINGLE,
            payload=struct.pack("<I", 1),       # 1 = start
        )
        logger.debug("JOB_START")

    def read_status_running(self) -> bool:
        """READ_STATUS (0x72): trả về True nếu robot đang chạy job."""
        resp = self._send_request(
            Command.READ_STATUS, instance=1,
            service=Service.GET_ATTRIBUTE_ALL,
        )
        # Status payload byte 0 chứa nhiều flag — bit 1 ("Running") là cái ta cần
        # theo Yaskawa HSE spec. Nếu payload < 1 byte → coi như idle.
        if not resp.payload:
            return False
        return bool(resp.payload[0] & 0x02)

    # ─── P-variable write (M3++ ultra-fast mode) ───
    def write_position_var(
        self, p_index: int, joints_deg: list[float],
        tool_no: int = 0, user_frame: int = 0,
    ) -> None:
        """Ghi P-variable (PULSE form) qua HSE — KHÔNG cần FTP upload.

        P-variables (P000-P127) là position variables runtime-mutable. Workflow
        ultra-fast: upload template INFORM dùng P-vars 1 lần, mỗi trial chỉ
        WRITE_VAR (~10ms/call) + JOB_START → 0 FTP overhead/trial.

        Args:
            p_index: 0-127 — P-variable index.
            joints_deg: 6 joint angles (degrees). Convert sang pulse qua
                GP7_PULSE_PER_DEG ratio.
            tool_no: Tool number (mặc định 0).
            user_frame: User frame number (mặc định 0).
        """
        if not (0 <= p_index <= 127):
            raise ValueError(f"P-variable index 0-127, got {p_index}")
        if len(joints_deg) != 6:
            raise ValueError(f"Cần 6 joints, got {len(joints_deg)}")

        # Convert deg → pulse (cùng ratio như INFORM C-variable)
        from .hse_protocol import GP7_PULSE_PER_DEG
        pulses = [int(round(d * r)) for d, r in zip(joints_deg, GP7_PULSE_PER_DEG)]

        # Payload format theo WRITE_POSITION_VAR spec:
        #   [0-3]   data_type (0 = PULSE)
        #   [4-7]   form/figure (0 default)
        #   [8-11]  tool_no
        #   [12-15] user_frame
        #   [16-19] extended_form (0)
        #   [20-43] 6 × int32 joint pulses
        #   [44-51] 2 × int32 axis 7-8 (unused for GP7)
        payload = struct.pack(
            "<5I 8i",
            0,                  # PULSE
            0,                  # form
            tool_no,
            user_frame,
            0,                  # extended_form
            *pulses,            # 6 joints
            0, 0,               # axis 7, 8 (unused)
        )
        self._send_request(
            Command.WRITE_POS_VAR, instance=p_index, attribute=0,
            service=Service.SET_ATTRIBUTE_ALL, payload=payload,
        )
        logger.debug("HSE WRITE_POS_VAR P%03d ← %s deg", p_index, joints_deg)

    def read_alarm(self) -> tuple[int, int]:
        """READ_ALARM (0x70): đọc alarm code hiện tại + sub-code.

        Returns:
            (code, sub_code). (0, 0) = không có alarm.
            Use `alarm_codes.decode_alarm(code)` để dịch sang tên + recovery hint.
        """
        resp = self._send_request(
            Command.READ_ALARM, instance=1,
            service=Service.GET_ATTRIBUTE_ALL,
        )
        # Alarm payload theo spec: [0-3] code (uint32 LE), [4-7] sub_code, [8-11] type, ...
        if len(resp.payload) < 8:
            return (0, 0)
        code = struct.unpack("<I", resp.payload[0:4])[0]
        sub_code = struct.unpack("<I", resp.payload[4:8])[0]
        return (int(code), int(sub_code))

    def _wait_idle(self, timeout_s: float | None = None) -> None:
        """Poll READ_STATUS đến khi running flag tắt hoặc timeout."""
        deadline = time.monotonic() + (timeout_s or self.wait_completion_timeout_s)
        while time.monotonic() < deadline:
            try:
                if not self.read_status_running():
                    return
            except Exception as e:                  # noqa: BLE001
                logger.warning("READ_STATUS lỗi khi poll: %s", e)
            time.sleep(0.1)
        raise TimeoutError(
            f"Job không kết thúc trong {timeout_s or self.wait_completion_timeout_s}s — "
            f"check teach pendant / alarm."
        )

    def _next_job_name(self) -> str:
        self._job_counter += 1
        return f"{self.job_name_prefix}{self._job_counter:04d}"

    # ─── Ultra-fast P-var mode (M3++) ───
    def enable_ultra_fast(self, enabled: bool = True) -> None:
        """Bật/tắt ultra-fast mode.

        Khi True, `batch()` sẽ dùng P-variables thay vì C-variables:
          - Lần đầu: upload INFORM template + WRITE_POS_VAR + START
          - Lần sau (template signature khớp): chỉ WRITE_POS_VAR + START
            (0 FTP roundtrip → ~50ms/trial overhead)

        Template signature = string đặc trưng trial structure ("MMDMM" = 5
        movj instructions với DOUT ở giữa). Khi signature thay đổi (vd trial
        không đối xứng) → upload template mới.
        """
        self._ultra_fast_mode = bool(enabled)
        logger.info("Ultra-fast P-var mode: %s", "ON" if enabled else "OFF")

    def _build_trial_signature(self) -> tuple[str, list[str], int | None, int | None]:
        """Phân tích buffer batch → signature + motion_kinds + gripper indices.

        Returns: (signature, motion_kinds, close_at, open_at)
            signature: "MMMDMM..." (M=movj, L=movl, D=dout) cho compare template
            motion_kinds: list movj/movl theo thứ tự P-variables
            close_at, open_at: P-index để insert DOUT ON/OFF
        """
        signature = ""
        motion_kinds: list[str] = []
        close_at: int | None = None
        open_at: int | None = None
        pos_idx = 0
        for op, value in self._pvar_batch_buffer:
            if op == "movj":
                signature += "M"
                motion_kinds.append("movj")
                pos_idx += 1
            elif op == "movl":
                signature += "L"
                motion_kinds.append("movl")
                pos_idx += 1
            elif op == "dout":
                signature += "D" if value else "d"
                # Index của motion KẾ tiếp (chưa add)
                if value and close_at is None:
                    close_at = pos_idx
                elif not value and open_at is None:
                    open_at = pos_idx
            elif op == "timer":
                pass                                 # timer luôn theo dout, không tách
        return signature, motion_kinds, close_at, open_at

    def _execute_ultra_fast_batch(self) -> None:
        """Render buffer → P-var template + WRITE_POS_VARs + JOB_START."""
        if not self._pvar_batch_buffer:
            return

        signature, motion_kinds, close_at, open_at = self._build_trial_signature()
        num_pos = sum(1 for op, _ in self._pvar_batch_buffer if op in ("movj", "movl"))

        # Upload template chỉ khi signature thay đổi (lần đầu hoặc structure khác).
        if (signature != self._pvar_template_signature
                or not self._pvar_template_name):
            template_name = f"{self.job_name_prefix}TPL"
            text = gen_pvar_template_job(
                name=template_name,
                num_positions=num_pos,
                motion_kinds=motion_kinds,
                gripper_close_at=close_at,
                gripper_open_at=open_at,
                max_speed_pct=self.max_speed_pct,
            )
            self.upload_job(text, template_name)
            self._pvar_template_name = template_name
            self._pvar_template_signature = signature
            logger.info(
                "Ultra-fast template uploaded: '%s' (signature=%s, %d positions)",
                template_name, signature, num_pos,
            )

        # WRITE_POS_VAR cho từng joint waypoint
        p_idx = 0
        for op, value in self._pvar_batch_buffer:
            if op in ("movj", "movl"):
                # value là joints list[float] degrees
                self.write_position_var(p_idx, value)        # type: ignore[arg-type]
                p_idx += 1
            # dout + timer đã được encode trong template, skip ở đây

        self.job_select(self._pvar_template_name)
        self.job_start()
        self._wait_idle()

    # ─── Batch mode (M3 optimization) ───
    @contextlib.contextmanager
    def batch(self, job_name: str | None = None) -> Iterator[None]:
        """Batch context: gom MoveJ/MoveL/setDO/timer vào 1 INFORM job.

        Trong batch:
          - Mọi MoveJ/MoveL/setDO/timer APPEND vào builder thay vì gửi ngay
          - Khi exit context → render INFORM → FTP upload → JOB_SELECT → START → wait_idle
          - Reduce overhead từ N FTP/JOB_START xuống **1 lần / trial**

        Non-batch (default): mỗi MoveJ là 1 INFORM job độc lập (~200-300ms overhead).
        Batch: cả pick-and-place trial ~200ms overhead total — speedup 5-10x.

        Usage:
            with backend.batch():
                backend.MoveJ(approach_joints)
                backend.MoveL(grasp_joints)
                backend.setDO(1, 1)
                backend.timer(0.3)
                ...
            # Auto-execute batch khi exit context

        Raises:
            RuntimeError: Nested batch.
        """
        if self._batch_builder is not None or self._pvar_batch_buffer is not None:
            raise RuntimeError("Nested batch không hỗ trợ")

        if self._ultra_fast_mode:
            # ─── Ultra-fast path: buffer ops, execute via P-vars ───
            self._pvar_batch_buffer = []
            try:
                yield
            except Exception:
                self._pvar_batch_buffer = None
                raise
            try:
                self._execute_ultra_fast_batch()
            finally:
                self._pvar_batch_buffer = None
            return

        # ─── Standard batch path (M3): single INFORM upload + execute ───
        name = job_name or self._next_job_name()
        self._batch_name = name
        self._batch_builder = InformJobBuilder(
            name=name, max_speed_pct=self.max_speed_pct,
        )
        self._batch_pos_counter = 0
        self._batch_builder.comment(f"Auto-gen batch #{self._job_counter}")

        try:
            yield
        except Exception:
            # Khi exception trong batch → drop builder, KHÔNG upload (an toàn)
            self._batch_builder = None
            raise

        # Snapshot + clear state TRƯỚC khi upload (an toàn nếu upload trigger
        # re-entry vô tình)
        builder = self._batch_builder
        self._batch_builder = None
        # Nếu batch không có instruction nào ngoài comment → skip upload
        if not builder._positions:
            logger.debug("Batch '%s' empty (no motion), skip upload", name)
            return

        text = builder.render()
        self.upload_job(text, name)
        self.job_select(name)
        self.job_start()
        self._wait_idle()

    def _batch_append_motion(self, target: Any, kind: str) -> None:
        """Helper: queue MoveJ/MoveL vào batch builder."""
        joints = self._to_joint_list(target)
        pos_name = f"p{self._batch_pos_counter}"
        self._batch_pos_counter += 1
        self._batch_builder.add_position(pos_name, joints)
        if kind == "movj":
            self._batch_builder.movj(pos_name)
        else:
            self._batch_builder.movl(pos_name)

    # ─── Motion (M3 — branched theo batch state) ───
    def MoveJ(self, target: Any) -> None:
        """Joint move tới target (joints list 6 phần tử).

        Non-batch: 1 INFORM upload + JOB_START + wait (200-300ms overhead).
        Batch (M3): append instruction vào batch builder (immediate return).
        Ultra-fast batch (M3++): append vào pvar buffer.
        """
        if self._pvar_batch_buffer is not None:
            self._pvar_batch_buffer.append(("movj", self._to_joint_list(target)))
            return
        if self._batch_builder is not None:
            self._batch_append_motion(target, "movj")
            return

        joints = self._to_joint_list(target)
        job_name = self._next_job_name()
        job_text = (
            InformJobBuilder(name=job_name, max_speed_pct=self.max_speed_pct)
            .add_position("target", joints)
            .comment(f"Auto-gen MoveJ #{self._job_counter}")
            .movj("target")
            .render()
        )
        self.upload_job(job_text, job_name)
        self.job_select(job_name)
        self.job_start()
        self._wait_idle()

    def MoveL(self, target: Any) -> None:
        """Linear move — tương tự MoveJ nhưng dùng MOVL."""
        if self._pvar_batch_buffer is not None:
            self._pvar_batch_buffer.append(("movl", self._to_joint_list(target)))
            return
        if self._batch_builder is not None:
            self._batch_append_motion(target, "movl")
            return

        joints = self._to_joint_list(target)
        job_name = self._next_job_name()
        job_text = (
            InformJobBuilder(name=job_name, max_speed_pct=self.max_speed_pct)
            .add_position("target", joints)
            .comment(f"Auto-gen MoveL #{self._job_counter}")
            .movl("target", speed_mm_s=80.0)
            .render()
        )
        self.upload_job(job_text, job_name)
        self.job_select(job_name)
        self.job_start()
        self._wait_idle()

    def timer(self, seconds: float) -> None:
        """Pause `seconds`.

        Non-batch: time.sleep — script thread chờ.
        Batch (M3): append INFORM TIMER — robot tự chờ trong khi execute job.
        Ultra-fast (M3++): TIMER baked vào template (sau DOUT), buffer chỉ track.
        """
        if self._pvar_batch_buffer is not None:
            self._pvar_batch_buffer.append(("timer", float(seconds)))
            return
        if self._batch_builder is not None:
            self._batch_builder.timer(seconds)
        else:
            time.sleep(seconds)

    @staticmethod
    def _to_joint_list(target: Any) -> list[float]:
        """Convert MoveJ target sang list[float] 6 phần tử (degrees)."""
        if isinstance(target, (list, tuple)) and len(target) >= 6:
            return [float(j) for j in target[:6]]
        raise NotImplementedError(
            "HSE backend MoveJ chỉ nhận joints (list 6 phần tử) hiện tại. "
            "Pose 4x4 cần IK solver client-side (ikfast/GP7 DH) — phase sau."
        )

    def MoveJ_Test(self, j_start: Any, target: Any, *args: Any) -> int:
        """Reachability check qua sphere envelope (nếu provided), không call YRC1000.

        RoboDK convention: 0 = OK, < 0 = unreachable.

        Nếu `reach_envelope` được set khi construct, dùng nó. Nếu không, permissive
        return 0 — caller phải chịu trách nhiệm (vd via DigitalTwinMirror delegate
        tới RoboDK robot item).
        """
        if self.reach_envelope is None:
            return 0
        return 0 if self.reach_envelope.can_reach(target) else -1

    def SolveIK(self, pose: Any, joints_approx: Any = None) -> Any:
        """HSE không có IK solver. Phải tính client-side (ikfast / pyrobot)."""
        raise NotImplementedError(
            "IK qua HSE: phải tính client-side. Xem ikfast / Yaskawa GP7 DH."
        )

    # ─── I/O ───
    def setDO(self, index: int, value: int) -> None:
        """Ghi digital output để điều khiển gripper.

        Non-batch: WRITE_IO command qua HSE (set network I/O bit qua CIO ladder).
        Batch: append DOUT instruction vào INFORM job (set Y-output trực tiếp
        khi job execute đến đó).

        Mapping non-batch: index → bit NETWORK_IO_BASE + (index-1). CIO ladder
        trên YRC1000 phải route bit này tới Y-output vật lý.
        """
        if self._pvar_batch_buffer is not None:
            self._pvar_batch_buffer.append(("dout", int(bool(value))))
            return
        if self._batch_builder is not None:
            self._batch_builder.dout(index, bool(value))
            return

        bit_addr = NETWORK_IO_BASE + (index - 1) * 1     # 1 bit/index
        # WRITE_IO command 0x78, service SET_ATTRIBUTE_SINGLE (0x10)
        # Payload: 4-byte uint32 = value (0 hoặc 1)
        payload = struct.pack("<I", 1 if value else 0)
        self._send_request(
            Command.WRITE_IO, instance=bit_addr, attribute=1,
            service=Service.SET_ATTRIBUTE_SINGLE, payload=payload,
        )
        logger.debug("HSE setDO(index=%d → bit %d) = %d", index, bit_addr, value)

    def setSpeed(self, linear_mm_s: float, joint_deg_s: float = -1) -> None:
        """No-op — HSE không control speed trực tiếp. Tốc độ set trong INFORM job."""
        logger.debug(
            "setSpeed(linear=%.1f, joint=%.1f) - HSE no-op, set trong INFORM",
            linear_mm_s, joint_deg_s,
        )

    def Stop(self) -> None:
        """Emergency stop: tắt servo trên YRC1000.

        HOLD_SERVO (0x83) với value=0 → servo off. Robot dừng hoàn toàn.
        Để chạy lại phải bật servo + reset alarm trên TP.
        """
        try:
            import struct as _struct
            self._send_request(
                Command.HOLD_SERVO, instance=2, attribute=1,
                service=Service.SET_ATTRIBUTE_SINGLE,
                payload=_struct.pack("<I", 0),    # servo off
            )
            logger.warning("HSE Stop(): servo OFF gửi tới YRC1000")
        except Exception as e:                    # noqa: BLE001
            logger.error("HSE Stop() lỗi: %s — dùng E-stop vật lý!", e)
