"""
motoman_hse.py
──────────────
MotomanHSEBackend: communicates directly with YRC1000 via UDP HSE protocol.

No RoboDK driver required (RoboDK Free is only needed for visualization), no MotoCom32
DLL, no ROS 2 / MotoPlus flash. Only requires:
  - YRC1000 with "High-Speed Ethernet Server function" enabled (Maintenance mode)
  - PC on the same subnet as YRC1000

Implemented features:
  - Connect / status / disconnect (M1)
  - Joints() reads joint angles real-time → mirrors to digital twin (M1)
  - setDO(index, value) writes network I/O (M2)

Pending (M3, follow-up):
  - MoveJ/MoveL: requires INFORM (.JBI) generator → FTP upload → JOB_SELECT + START.
    Non-trivial; implemented separately in inform_codegen.py + sftp uploader.
    Stub raises NotImplementedError with instructions.

Safety notes:
  - Setup should only be tested with robot in REMOTE mode (key switch on TP)
  - Default safety: setSpeed called with max_speed_percent from config
  - Stop() sends HOLD_SERVO command for emergency stop (physical E-stop also available)
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
    DataType,
    HSEDecodeError,
    HSEResponse,
    HSERequest,
    Service,
    encode_cartesian_position,
    parse_position_response,
    pulse_to_deg,
)
from .inform_codegen import InformJobBuilder, gen_pvar_template_job

logger = logging.getLogger(__name__)

# Default UDP port per Yaskawa HSE Server spec
HSE_PORT_ROBOT = 10040
HSE_PORT_FILE = 10041

# Max stale datagrams to drain while looking for a matching request_id before
# giving up and using the last response (prevents an infinite loop if the peer
# never echoes the expected id).
_MAX_DRAIN = 8


class JobUploadError(RuntimeError):
    """FTP upload of a .JBI was refused because the job cannot be overwritten on the
    controller (it is the currently-SELECTED job, or write-protected). Carries the
    job name so the GUI can offer to rename it. `job_name` excludes the .JBI suffix."""

    def __init__(self, message: str, job_name: str = "") -> None:
        super().__init__(message)
        self.job_name = job_name


class MotionHaltedError(RuntimeError):
    """A motion command (START / direct MOVE) was refused because Stop() latched
    the halt. Cleared by servo_on(). The robot was NOT commanded to move."""


class RobotAlarmError(RuntimeError):
    """A job/move did not complete cleanly — the controller raised an alarm. Carries
    (code, sub_code) so the caller can decode it instead of reporting false success."""

    def __init__(self, code: int, sub_code: int) -> None:
        super().__init__(f"Robot alarm {code}/{sub_code} during motion")
        self.code = int(code)
        self.sub_code = int(sub_code)

# Network I/O range for writable bits (YRC1000):
# 27010-27020 = general purpose network I/O (CIO ladder accessible).
# Purpose for gripper: bind 1 bit here to a physical Y-output via CIO ladder
# on the controller. Documented in setup guide.
NETWORK_IO_BASE = 27010


class MotomanHSEBackend:
    """YRC1000 backend via UDP HSE protocol.

    Duck-typed to match RoboDK Item interface (Joints, MoveJ, setDO, ...)
    so the Orchestrator can call it unchanged.

    Args:
        ip: IP address of the YRC1000.
        port: UDP port (default 10040 for robot commands).
        timeout_s: Socket timeout per request.
        request_id_seed: Starting value of request_id (auto-incremented).
    """

    # Class-level marker: backend supports 4x4 pose input for MoveJ/MoveL
    # (YRC1000 computes IK internally). DigitalTwinMirror checks this flag to
    # distinguish from MagicMock / generic backends — avoids false-positive duck-type detection.
    supports_cartesian_pose: bool = True

    def __init__(
        self,
        ip: str,
        port: int = HSE_PORT_ROBOT,
        timeout_s: float = 2.0,
        request_id_seed: int = 0,
        ftp_user: str = "",
        ftp_pass: str = "",
        ftp_job_dir: str = "/JOB",
        max_speed_pct: float = 20.0,
        job_name_prefix: str = "DTWIN",
        wait_completion_timeout_s: float = 30.0,
        reach_envelope: Any = None,
        tool_no: int = 1,
    ) -> None:
        self.ip = ip
        self.port = port
        self.timeout_s = timeout_s
        self._sock: socket.socket | None = None
        self._request_id = request_id_seed % 256
        self._lock = threading.Lock()       # request_id ++ must be atomic
        # Cache for JointsHome — set via config from orchestrator
        self._home_joints: list[float] = [0.0] * 6
        # FTP credentials for job upload (YRC1000 defaults to "" / "")
        self.ftp_user = ftp_user
        self.ftp_pass = ftp_pass
        self.ftp_job_dir = ftp_job_dir
        # Safety + naming
        self.max_speed_pct = max_speed_pct
        self.job_name_prefix = job_name_prefix
        self._job_counter = 0
        self.wait_completion_timeout_s = wait_completion_timeout_s
        # Bounded window for _wait_idle to confirm a started job asserted its
        # Running bit before polling for completion (job_start() returns on the
        # START ack, BEFORE the controller's start latency). Must EXCEED the
        # worst-case START→Running latency; generous because Phase 1 breaks as soon
        # as Running is seen. 0 disables Phase 1 (unit tests only).
        self.start_confirm_timeout_s = 3.0
        # Optional kinematic envelope for MoveJ_Test client-side when no RoboDK item.
        self.reach_envelope = reach_envelope
        # TOOL coordinate number — must match TOOL01 configured on teach pendant
        # (TCP offset of gripper). Default = 1 (TOOL01).
        self.tool_no = int(tool_no)
        # Batch state: None = non-batch (each MoveJ = 1 INFORM upload). Active builder
        # while inside `with backend.batch():` — motion + IO calls accumulated here.
        self._batch_builder: InformJobBuilder | None = None
        self._batch_pos_counter = 0
        self._batch_name = ""
        # Halt latch: Stop() sets it, servo_on() clears it. A batch dispatch
        # (upload+JOB_SELECT+START) checks it so a Stop() that lands mid-batch
        # cannot issue a START after the halt. _halt_lock makes the
        # check-then-START atomic w.r.t. Stop() (the long FTP upload runs OUTSIDE
        # the lock so Stop() is never blocked behind it).
        self._halted = False
        self._halt_lock = threading.Lock()

        # Ultra-fast P-var mode state (M3++)
        self._ultra_fast_mode = False
        self._pvar_template_name = ""        # name of the uploaded template
        self._pvar_template_signature = ""   # trial structure signature ("MMMDMM" etc.)
        # None = not in batch. List = inside batch context (append ops).
        # Record format: ("movj", joints), ("movl", joints), ("dout", value_int),
        # ("timer", seconds_float) — collected during batch context.
        self._pvar_batch_buffer: list[tuple[str, Any]] | None = None

    # ─── Lifecycle ───
    def connect(self) -> None:
        """Open UDP socket. Does NOT verify with robot — call Joints() to verify."""
        if self._sock is not None:
            logger.debug("HSE socket already open")
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(self.timeout_s)
        self._sock = sock
        logger.info("HSE UDP socket opened (%s:%d, timeout %.1fs)", self.ip, self.port, self.timeout_s)

    def disconnect(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
            logger.info("HSE UDP socket closed")

    def __del__(self) -> None:
        try:
            self.disconnect()
        except Exception:                       # noqa: BLE001
            pass

    # ─── Helpers ───
    def _send_request(
        self,
        command: int,
        instance: int = 1,
        attribute: int = 0,
        service: int = Service.GET_ATTRIBUTE_ALL,
        payload: bytes = b"",
    ) -> HSEResponse:
        """Send request, wait for response. Raises on timeout / decode error / status != 0.

        Thread-safe: the entire sendto + recvfrom is wrapped in `self._lock` so
        two concurrent threads cannot swap responses. UDP does not guarantee
        ordering, so previously: thread A sends req_id=5, thread B sends req_id=6,
        response B arrives first → A receives response B with id mismatch.
        """
        if self._sock is None:
            raise RuntimeError("HSE socket not connected — call connect() first")

        with self._lock:
            # Atomic: id increment + send + recv. UDP delivery may still be out-of-order
            # with multi-packet bursts, but since 1 request → 1 response and the lock
            # only releases after receiving, no other thread can interleave.
            self._request_id = (self._request_id + 1) % 256
            request_id = self._request_id
            req = HSERequest(
                command=command, instance=instance, attribute=attribute,
                service=service, payload=payload, request_id=request_id,
            )
            packet = req.encode()
            self._sock.sendto(packet, (self.ip, self.port))

            try:
                raw, _ = self._sock.recvfrom(2048)
            except socket.timeout as e:
                # Flush any datagram that arrives LATE for this now-abandoned
                # request, so a stale reply can't be mis-read as the answer to
                # the NEXT request (UDP keeps it buffered otherwise).
                self._drain_socket()
                raise TimeoutError(
                    f"HSE request 0x{command:02X} timeout {self.timeout_s}s — "
                    f"YRC1000 not responding. Check ping + HSE Server function."
                ) from e

        try:
            resp = HSEResponse.decode(raw)
        except HSEDecodeError:
            logger.error("HSE response decode error, raw bytes: %s", raw.hex())
            raise

        # NOTE: request_id is WARNED on but NOT enforced — the YRC1000 does not
        # reliably echo it, so discarding mismatches would drop valid replies. The
        # _drain_socket() on the timeout path is the actual stale-datagram guard.
        if resp.request_id != request_id:
            logger.warning(
                "HSE request_id mismatch: sent=%d got=%d", request_id, resp.request_id
            )

        resp.raise_for_status()
        return resp

    def _drain_socket(self) -> None:
        """Discard any datagrams currently buffered on the socket — e.g. a late
        reply to a request that already timed out. Non-blocking, bounded by
        _MAX_DRAIN so a flood can't stall us. Called only on the timeout path so
        the happy path is unchanged."""
        if self._sock is None:
            return
        try:
            self._sock.setblocking(False)
            for _ in range(_MAX_DRAIN):
                try:
                    self._sock.recvfrom(2048)
                except (BlockingIOError, socket.timeout, OSError):
                    break
        finally:
            self._sock.setblocking(True)
            self._sock.settimeout(self.timeout_s)

    # ─── RoboDK Item-like interface ───
    def Valid(self) -> bool:
        """Heartbeat: attempts ReadStatus. True if socket + controller alive."""
        try:
            self._send_request(Command.READ_STATUS, instance=1)
            return True
        except Exception as e:                  # noqa: BLE001
            logger.warning("HSE Valid() probe fail: %s", e)
            return False

    def Joints(self) -> list[float]:
        """Read current joint angles from YRC1000 (degrees).

        Uses READ_POSITION (0x75) with data_type pulse, then converts to degrees.
        """
        resp = self._send_request(
            Command.READ_POSITION, instance=1,
            service=Service.GET_ATTRIBUTE_ALL,
        )
        parsed = parse_position_response(resp.payload)
        return pulse_to_deg(parsed["joints_raw"])

    def JointsHome(self) -> list[float]:
        """Home joints — set via config from Orchestrator (cell_layout.yaml)."""
        return list(self._home_joints)

    def set_home_joints(self, joints_deg: list[float]) -> None:
        """Setter for home — called by Orchestrator after init."""
        if len(joints_deg) != 6:
            raise ValueError(f"Home requires 6 joints, got {len(joints_deg)}")
        self._home_joints = list(joints_deg)

    def Parent(self) -> Any:
        """HSE has no concept of a parent frame (unlike RoboDK items)."""
        return None

    # ─── Job upload + execute (M3) ───
    def upload_job(self, job_text: str, job_name: str) -> None:
        """Upload INFORM .JBI text to YRC1000 via FTP.

        YRC1000 runs an FTP server on port 21 (anonymous) by default. File is
        stored in `ftp_job_dir` (default /JOB on YRC1000) so JOB_SELECT can find it.

        Args:
            job_text: INFORM content (rendered from InformJobBuilder).
            job_name: File name (.JBI suffix optional — added automatically).
        """
        filename = job_name if job_name.upper().endswith(".JBI") else f"{job_name}.JBI"
        # Yaskawa INFORM uses Shift-JIS encoding but ASCII-only is safe for both.
        # Normalise line endings to CRLF: the YRC1000 job parser expects CRLF
        # record terminators, but the Run-on-Robot path reads the rendered job via
        # Path.read_text() (universal newlines), which COLLAPSES the file's CRLF to
        # LF. An LF-only job transfers OK yet the controller refuses to CLOSE/save
        # it -> "451 Error closing file.--[5130]--". Collapse-then-expand is
        # idempotent for already-CRLF text (the one-off / batch builder paths).
        job_text = (job_text.replace("\r\n", "\n").replace("\r", "\n")
                    .replace("\n", "\r\n"))
        data = job_text.encode("ascii", errors="strict")

        logger.info("FTP upload '%s' (%d bytes) to %s%s",
                    filename, len(data), self.ip, self.ftp_job_dir)
        ftp = ftplib.FTP()
        try:
            ftp.connect(self.ip, 21, timeout=self.timeout_s)
            ftp.login(self.ftp_user, self.ftp_pass)
            ftp.cwd(self.ftp_job_dir)
            # YRC1000 FTP refuses to STOR over an existing file ("503 Bad sequence of
            # commands. Can't overwrite JOB-file") — delete it first. A 550 (no such
            # file) is expected/fine. The controller will NOT delete the currently
            # SELECTED job, so if delete is refused, give a clear teach-pendant hint.
            _job = filename[:-4] if filename.upper().endswith(".JBI") else filename
            try:
                ftp.delete(filename)
            except ftplib.error_perm as e:
                if not str(e).strip().startswith("550"):
                    raise JobUploadError(
                        f"Cannot overwrite job '{_job}' on the controller "
                        f"({str(e).strip()}). It is likely the CURRENTLY-SELECTED job "
                        f"— rename the job in the app, or deselect/CANCEL it on the "
                        f"teach pendant, then Run on Robot again.", _job) from e
            try:
                ftp.storbinary(f"STOR {filename}", io.BytesIO(data))
            except ftplib.error_perm as e:
                raise JobUploadError(
                    f"Upload of job '{_job}' refused ({str(e).strip()}). If it is the "
                    f"active job, rename it in the app or deselect it on the teach "
                    f"pendant; also check it is not write-protected.", _job) from e
            except ftplib.error_temp as e:
                # ftplib raises error_temp for EVERY 4xx STOR reply — distinguish:
                msg = str(e).strip()
                if msg.startswith(("451", "450", "452")):
                    # "451 Error closing file.--[5130]--": the controller ACCEPTED the
                    # byte transfer but REFUSED to SAVE/close the job. NOT a job-
                    # content problem — a byte-exact valid teach-pendant export (e.g.
                    # PP1.JBI) is rejected the same way → controller-side: FTP account
                    # lacks job-WRITE permission, or operation mode / security level /
                    # an edit-lock prohibits saving. NOT an overwrite → no rename dialog.
                    raise RuntimeError(
                        f"Controller refused to SAVE job '{_job}' ({msg}). The .JBI "
                        f"transferred OK but could not be written. Check on the "
                        f"controller: (1) the FTP account has job WRITE permission "
                        f"(not read-only), (2) job editing is allowed in the current "
                        f"operation mode / security level, (3) the job is not open or "
                        f"edit-locked on the teach pendant.") from e
                # 425/426/other 4xx = data-connection / transfer failure (NOT a save
                # refusal) — report honestly so the operator checks the network, not
                # job permissions.
                raise RuntimeError(
                    f"FTP transfer of job '{_job}' failed ({msg}) — data-connection / "
                    f"network error, not a controller save-refusal. Check the network "
                    f"link to the controller and retry.") from e
        finally:
            try:
                ftp.quit()
            except Exception:                       # noqa: BLE001
                pass
            try:
                ftp.close()                         # always release the socket even
            except Exception:                       # if quit() failed (broken conn)
                pass                                # noqa: BLE001

    def job_select(self, job_name: str) -> None:
        """JOB_SELECT (0x87): selects the uploaded job as current. Must be called before START.

        Per the Yaskawa HSE spec (Job Selection and Start): command 0x87, instance 1
        (= set executing job), Service 0x02 (Set_Attribute_All), Attribute FIXED 0x00.
        We previously sent attribute=1, which the controller rejected with status 0x1F
        / added_status 0xA002 ("Attribute error") — so JOB_SELECT never took effect.
        """
        name = job_name.upper().removesuffix(".JBI").encode("ascii")
        # Payload: 32-byte job name (null-padded) + 4-byte line number (uint32 LE, 0=top)
        payload = name.ljust(32, b"\x00") + struct.pack("<I", 0)
        self._send_request(
            Command.JOB_SELECT, instance=1, attribute=0,
            service=Service.SET_ATTRIBUTE_ALL, payload=payload,
        )
        logger.debug("JOB_SELECT '%s'", job_name)

    def job_start(self) -> None:
        """START (0x86): execute current job. Robot must have servo on + REMOTE mode.

        Refuses to START while halted (Stop() latched _halted) — the central gate
        for every job-based motion path (MoveJ/MoveL/_move_pose/batch/ultra-fast).
        Cleared by servo_on()."""
        if self._halted:
            raise MotionHaltedError("START refused — motion halted (call servo_on to resume)")
        self._send_request(
            Command.START, instance=1, attribute=1,
            service=Service.SET_ATTRIBUTE_SINGLE,
            payload=struct.pack("<I", 1),       # 1 = start
        )
        logger.debug("JOB_START")

    def servo_on(self) -> None:
        """Turn servo power ON via HOLD_SERVO (0x83), instance 2 (servo-ON control),
        data 1 = ON (mirror of Stop()'s 2 = OFF). Required before START in REMOTE
        mode (the mode key switch on the TP must be set to REMOTE separately).

        SAFETY/VERIFY: value/instance per Yaskawa HSE spec, not yet confirmed on
        hardware — if START still fails after this, capture the controller error.
        """
        self._send_request(
            Command.HOLD_SERVO, instance=2, attribute=1,
            service=Service.SET_ATTRIBUTE_SINGLE,
            payload=struct.pack("<I", 1),       # 1 = servo ON
        )
        self._halted = False                    # resuming → clear the halt latch
        logger.info("HSE servo ON sent to YRC1000")

    # ─── Direct real-time motion (RoboDK-style, NO job upload) ───
    # move_type instances (per Yaskawa HSE / fs100): 1 = joint absolute,
    # 2 = linear absolute, 3 = linear incremental. speed_class 0 = percent.
    def move_pulse(self, pulses: list[int], speed_pct: float = 5.0,
                   move_type: int = 1, tool_no: int | None = None) -> None:
        """Direct MOVE to ABSOLUTE joint PULSE target via HSE (0x8B) — no FTP/job.
        This is how online jogging works (cf. RoboDK MotomanHSE driver).

        Requires servo ON + REMOTE mode. speed_pct is % of max joint speed.
        SAFETY: real motion — keep speed low and a hand on the E-stop. Payload
        layout follows the FS100 HSE reference (88 bytes).
        """
        if self._halted:
            # Gate the direct (no-job) MOVE path on the halt latch too, matching
            # job_start — otherwise live-jog could keep streaming after a Stop().
            raise MotionHaltedError("MOVE refused — motion halted (call servo_on to resume)")
        if tool_no is None:
            tool_no = self.tool_no
        # SAFETY: reject malformed input — a short pulse list would pad missing
        # axes to encoder-0 and drive them there (big unexpected move).
        if len(pulses) not in (6, 7):
            raise ValueError(
                f"move_pulse needs 6 (or 7 w/ ext axis) pulses, got {len(pulses)}")
        if speed_pct <= 0.0:
            raise ValueError(f"move_pulse speed_pct must be > 0, got {speed_pct}")
        # SAFETY: cap to the configured max even on the direct (no-job) path — the
        # INFORM codegen path already applies max_speed_pct, the direct path did not.
        speed_pct = min(float(speed_pct), float(self.max_speed_pct))
        p = list(pulses)[:7] + [0] * (7 - len(pulses))      # pad to 7 axes
        speed = int(max(1, min(10000, round(speed_pct * 100))))  # 0.01% units
        payload = (
            struct.pack("<I", 1)            # robot_no
            + struct.pack("<I", 0)          # station_no
            + struct.pack("<I", 0)          # speed_class = PERCENT
            + struct.pack("<I", speed)      # speed (0.01 %)
            + struct.pack("<7i", *p)        # 7 axis pulses
            + struct.pack("<I", 0)          # reserved
            + struct.pack("<I", int(tool_no))   # tool_no
            + bytes(36)                     # padding → 88 bytes total
        )
        self._send_request(
            Command.MOVE_PULSE, instance=int(move_type), attribute=1,
            service=Service.SET_ATTRIBUTE_ALL, payload=payload,
        )
        logger.debug("HSE MOVE_PULSE type=%d speed=%.1f%% pulses=%s",
                     move_type, speed_pct, p[:6])

    def move_joints(self, joints_deg: list[float], speed_pct: float = 5.0) -> None:
        """Direct MOVE to absolute joint angles (deg) — converts to pulses with
        GP7_PULSE_PER_DEG and calls move_pulse. For real-time joint jog."""
        from .hse_protocol import GP7_JOINT_LIMITS_DEG, GP7_PULSE_PER_DEG
        if len(joints_deg) != len(GP7_PULSE_PER_DEG):
            raise ValueError(
                f"move_joints needs {len(GP7_PULSE_PER_DEG)} angles, got {len(joints_deg)}")
        # SAFETY (defense-in-depth): refuse an out-of-limit absolute target — don't
        # rely solely on GUI slider bounds + the controller's soft-limit alarm.
        for i, (d, (lo, hi)) in enumerate(zip(joints_deg, GP7_JOINT_LIMITS_DEG)):
            if not (lo - 1e-6 <= float(d) <= hi + 1e-6):
                raise ValueError(
                    f"move_joints: axis {i} = {float(d):.2f}° outside GP7 limit "
                    f"[{lo:.0f}, {hi:.0f}]")
        pulses = [int(round(d * r)) for d, r in zip(joints_deg, GP7_PULSE_PER_DEG)]
        self.move_pulse(pulses, speed_pct=speed_pct, move_type=1)

    def read_status_running(self) -> bool:
        """READ_STATUS (0x72): returns True if robot is executing a job."""
        resp = self._send_request(
            Command.READ_STATUS, instance=1,
            service=Service.GET_ATTRIBUTE_ALL,
        )
        # Status Data 1 (low byte) flag layout per Yaskawa HSE spec:
        #   bit0=Step  bit1=1-Cycle  bit2=Continuous  bit3=Running(operating)
        #   bit4=SpeedLimited  bit5=Teach  bit6=Play  bit7=Remote
        # "Running" is bit3 (0x08); the previous 0x02 read the 1-Cycle MODE flag,
        # so _wait_idle could mis-detect completion. If payload empty → idle.
        # NOTE: verify against the real controller — see Stop()'s caveat.
        if not resp.payload:
            return False
        return bool(resp.payload[0] & 0x08)

    def read_controller_status(self) -> dict:
        """READ_STATUS (0x72) decoded into operation mode / cycle / servo / alarm.

        Data1 bits: 0 step, 1 1-cycle, 2 auto, 3 running, 4 safeguard, 5 teach,
        6 play, 7 remote.  Data2 bits: 1 hold(PP), 2 hold(ext), 3 hold(cmd),
        4 alarm, 5 error, 6 servo on.  (Verified vs Yaskawa HSE "Controller Status
        Reading" tables.) In REMOTE the controller sets BOTH the play and remote
        bits, so `mode` checks `remote` first.
        """
        resp = self._send_request(
            Command.READ_STATUS, instance=1, service=Service.GET_ATTRIBUTE_ALL)
        p = resp.payload or b""
        d1 = p[0] if len(p) >= 1 else 0
        d2 = p[4] if len(p) >= 5 else 0
        remote, play, teach = bool(d1 & 0x80), bool(d1 & 0x40), bool(d1 & 0x20)
        return {
            "data1": d1, "data2": d2,
            "mode": "REMOTE" if remote else ("PLAY" if play else ("TEACH" if teach else "?")),
            "remote": remote, "play": play, "teach": teach,
            "auto": bool(d1 & 0x04), "running": bool(d1 & 0x08),
            "servo": bool(d2 & 0x40), "alarm": bool(d2 & 0x10),
            "error": bool(d2 & 0x20), "hold": bool(d2 & 0x0E),
        }

    # ─── P-variable write (M3++ ultra-fast mode) ───
    def write_position_var(
        self, p_index: int, joints_deg: list[float],
        tool_no: int = 0, user_frame: int = 0,
    ) -> None:
        """Write P-variable (PULSE form) via HSE — no FTP upload required.

        P-variables (P000-P127) are runtime-mutable position variables. Ultra-fast
        workflow: upload INFORM template using P-vars once; each trial only calls
        WRITE_VAR (~10ms/call) + JOB_START → zero FTP overhead per trial.

        Args:
            p_index: 0-127 — P-variable index.
            joints_deg: 6 joint angles (degrees). Converted to pulses via
                GP7_PULSE_PER_DEG ratio.
            tool_no: Tool number (default 0).
            user_frame: User frame number (default 0).
        """
        if not (0 <= p_index <= 127):
            raise ValueError(f"P-variable index 0-127, got {p_index}")
        if len(joints_deg) != 6:
            raise ValueError(f"Expected 6 joints, got {len(joints_deg)}")
        # Validate tool/frame like encode_cartesian_position does — an out-of-range
        # value would pack a wrong TCP/frame for a real move (or struct.error if <0).
        if not (0 <= tool_no <= 63):
            raise ValueError(f"tool_no must be 0-63, got {tool_no}")
        if not (0 <= user_frame <= 63):
            raise ValueError(f"user_frame must be 0-63, got {user_frame}")

        # Convert deg → pulse (same ratio as INFORM C-variable)
        from .hse_protocol import GP7_PULSE_PER_DEG
        pulses = [int(round(d * r)) for d, r in zip(joints_deg, GP7_PULSE_PER_DEG)]

        # Payload format per WRITE_POSITION_VAR spec:
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

    def write_pose_var(
        self,
        p_index: int,
        T_base: Any,                           # 4x4 numpy in robot BASE frame
        tool_no: int = 0,
        user_frame: int = 0,
        form: int = 0,
        data_type: int = DataType.BASE,
    ) -> None:
        """Write P-variable as Cartesian pose (BASE coordinate) via HSE.

        This path lets YRC1000 compute IK internally (controller's own kinematics)
        instead of on the PC, avoiding DH parameter discrepancies.

        Args:
            p_index: 0-127 — P-variable index.
            T_base: 4x4 pose in robot BASE frame (mm). Orchestrator calls
                `world_to_robot_base()` to convert from world frame first.
            tool_no: TCP tool number (1 = gripper TOOL01 configured on TP).
            user_frame: 0 for BASE coordinate.
            form: IK config branch (0 = front + elbow up + no flip default).
            data_type: BASE/ROBOT/USER/TOOL (default BASE).
        """
        from src.orchestrator.frame_convert import matrix_to_xyzrpy_yaskawa

        if not (0 <= p_index <= 127):
            raise ValueError(f"P-variable index 0-127, got {p_index}")

        x, y, z, rx, ry, rz = matrix_to_xyzrpy_yaskawa(T_base)
        payload = encode_cartesian_position(
            x_mm=x, y_mm=y, z_mm=z,
            rx_deg=rx, ry_deg=ry, rz_deg=rz,
            tool_no=tool_no, user_frame=user_frame,
            form=form, data_type=int(data_type),
        )
        self._send_request(
            Command.WRITE_POS_VAR, instance=p_index, attribute=0,
            service=Service.SET_ATTRIBUTE_ALL, payload=payload,
        )
        logger.debug(
            "HSE WRITE_POS_VAR P%03d Cartesian: xyz=(%.1f, %.1f, %.1f)mm "
            "rpy=(%.1f, %.1f, %.1f)° tool=%d form=%d",
            p_index, x, y, z, rx, ry, rz, tool_no, form,
        )

    def read_alarm(self) -> tuple[int, int]:
        """READ_ALARM (0x70): reads current alarm code + sub-code.

        Returns:
            (code, sub_code). (0, 0) = no alarm.
            Use `alarm_codes.decode_alarm(code)` to translate to name + recovery hint.
        """
        resp = self._send_request(
            Command.READ_ALARM, instance=1,
            service=Service.GET_ATTRIBUTE_ALL,
        )
        # Alarm payload per spec: [0-3] code (uint32 LE), [4-7] sub_code, [8-11] type, ...
        if len(resp.payload) < 8:
            return (0, 0)
        code = struct.unpack("<I", resp.payload[0:4])[0]
        sub_code = struct.unpack("<I", resp.payload[4:8])[0]
        return (int(code), int(sub_code))

    def _raise_if_alarm(self) -> None:
        """Raise RobotAlarmError if the controller currently reports an alarm.
        Best-effort: a READ_ALARM transport error is logged, not raised."""
        try:
            code, sub = self.read_alarm()
        except Exception as e:                      # noqa: BLE001
            logger.warning("READ_ALARM failed during completion check: %s", e)
            return
        if code != 0:
            raise RobotAlarmError(code, sub)

    def _wait_idle(self, timeout_s: float | None = None) -> None:
        """Wait for a started job to complete, then verify it was not alarm-aborted.

        job_start() returns on the START ack — BEFORE the controller asserts its
        Running bit — so a status poll landing inside the start latency reads idle
        and we would return WHILE THE ROBOT IS STILL MOVING.

        Guarantee: we NEVER report completion before having CONFIRMED the job ran.
        Phase 1 polls (up to start_confirm_timeout_s) until Running asserts; if it
        never asserts within that window the job did not start (wrong mode / servo
        off / alarm) → raise rather than fall through and falsely report success.
        Phase 2 then waits for Running to CLEAR and checks the alarm state — an
        alarm-aborted job must NOT be reported as success (the caller would
        otherwise drive the gripper / next move blind). start_confirm_timeout_s
        MUST exceed the controller's worst-case START→Running latency; the default
        is generous and Phase 1 breaks as soon as Running is seen. Setting it to 0
        skips Phase 1 (unit tests only — no real start latency to cover)."""
        total = timeout_s or self.wait_completion_timeout_s
        deadline = time.monotonic() + total
        # Phase 1 — confirm the job actually STARTED (Running asserted).
        confirm = min(self.start_confirm_timeout_s, total)
        if confirm > 0:
            confirm_deadline = time.monotonic() + confirm
            running_seen = False
            while time.monotonic() < confirm_deadline:
                try:
                    if self.read_status_running():
                        running_seen = True
                        break                        # started — go wait for it to finish
                except Exception as e:              # noqa: BLE001
                    logger.warning("READ_STATUS error while confirming start: %s", e)
                time.sleep(0.02)
            if not running_seen:
                # Running never asserted within the start window: the job did not
                # start. Do NOT fall through to Phase 2 (which would read idle and
                # falsely report success while the robot may still be about to move).
                self._raise_if_alarm()
                raise TimeoutError(
                    f"Job did not start (Running not asserted within {confirm:.1f}s) "
                    f"— check REMOTE mode / servo ON / teach-pendant alarm.")
        # Phase 2 — wait for completion, then verify no alarm aborted the job.
        while time.monotonic() < deadline:
            try:
                if not self.read_status_running():
                    self._raise_if_alarm()
                    return
            except RobotAlarmError:
                raise
            except Exception as e:                  # noqa: BLE001
                logger.warning("READ_STATUS error while polling: %s", e)
            time.sleep(0.1)
        self._raise_if_alarm()                       # surface an alarm behind a timeout
        raise TimeoutError(
            f"Job did not complete within {total}s — check teach pendant / alarm."
        )

    def _next_job_name(self) -> str:
        self._job_counter += 1
        return f"{self.job_name_prefix}{self._job_counter:04d}"

    # ─── Ultra-fast P-var mode (M3++) ───
    def enable_ultra_fast(self, enabled: bool = True) -> None:
        """Enable/disable ultra-fast mode.

        When True, `batch()` uses P-variables instead of C-variables:
          - First run: upload INFORM template + WRITE_POS_VAR + START
          - Subsequent runs (matching template signature): only WRITE_POS_VAR + START
            (0 FTP roundtrips → ~50ms/trial overhead)

        Template signature = string characterizing trial structure ("MMDMM" = 5
        movj instructions with DOUT in the middle). When signature changes (e.g.
        asymmetric trial) → upload new template.
        """
        self._ultra_fast_mode = bool(enabled)
        logger.info("Ultra-fast P-var mode: %s", "ON" if enabled else "OFF")

    def _build_trial_signature(self) -> tuple[str, list[str], int | None, int | None]:
        """Analyze batch buffer → signature + motion_kinds + gripper indices.

        Returns: (signature, motion_kinds, close_at, open_at)
            signature: "MMMDMM..." (M=movj, L=movl, D=dout) for template comparison
            motion_kinds: list of movj/movl in P-variable order
            close_at, open_at: P-index at which to insert DOUT ON/OFF
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
                # Index of the NEXT motion (not yet added)
                if value and close_at is None:
                    close_at = pos_idx
                elif not value and open_at is None:
                    open_at = pos_idx
            elif op == "timer":
                pass                                 # timer always follows dout, not separated
        return signature, motion_kinds, close_at, open_at

    def _execute_ultra_fast_batch(self) -> None:
        """Render buffer → P-var template + WRITE_POS_VARs + JOB_START."""
        if not self._pvar_batch_buffer:
            return
        # SAFETY: skip dispatch if a Stop() halted motion during the batch.
        if self._halted:
            logger.warning("Ultra-fast batch dropped: motion halted (Stop) during batch")
            return

        signature, motion_kinds, close_at, open_at = self._build_trial_signature()
        num_pos = sum(1 for op, _ in self._pvar_batch_buffer if op in ("movj", "movl"))

        # Upload template only when signature changes (first run or different structure).
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

        # WRITE_POS_VAR for each joint waypoint
        p_idx = 0
        for op, value in self._pvar_batch_buffer:
            if op in ("movj", "movl"):
                # value is joints list[float] degrees
                self.write_position_var(p_idx, value)        # type: ignore[arg-type]
                p_idx += 1
            # dout + timer are already encoded in the template, skip here

        self.job_select(self._pvar_template_name)
        with self._halt_lock:                    # atomic halt re-check vs Stop()
            if self._halted:
                logger.warning("Ultra-fast batch dropped: motion halted before START")
                return
            self.job_start()
        self._wait_idle()

    # ─── Batch mode (M3 optimization) ───
    @contextlib.contextmanager
    def batch(self, job_name: str | None = None) -> Iterator[None]:
        """Batch context: accumulates MoveJ/MoveL/setDO/timer into one INFORM job.

        Inside batch:
          - All MoveJ/MoveL/setDO/timer calls APPEND to the builder instead of sending immediately
          - On context exit → render INFORM → FTP upload → JOB_SELECT → START → wait_idle
          - Reduces overhead from N FTP/JOB_START calls to **1 per trial**

        Non-batch (default): each MoveJ is an independent INFORM job (~200-300ms overhead).
        Batch: entire pick-and-place trial ~200ms overhead total — 5-10x speedup.

        Usage:
            with backend.batch():
                backend.MoveJ(approach_joints)
                backend.MoveL(grasp_joints)
                backend.setDO(1, 1)
                backend.timer(0.3)
                ...
            # Batch auto-executes on context exit

        Raises:
            RuntimeError: Nested batch.
        """
        if self._batch_builder is not None or self._pvar_batch_buffer is not None:
            raise RuntimeError("Nested batch not supported")

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
            # Exception inside batch → drop builder, do NOT upload (safe)
            self._batch_builder = None
            raise

        # Snapshot + clear state BEFORE uploading (safe if upload accidentally
        # triggers re-entry)
        builder = self._batch_builder
        self._batch_builder = None
        # If batch has no instructions besides a comment → skip upload
        if not builder._positions:
            logger.debug("Batch '%s' empty (no motion), skip upload", name)
            return
        # SAFETY: a Stop() that landed mid-batch latched _halted — do NOT dispatch a
        # START after a halt (servo is off; this prevents the documented
        # "no commands after hold" invariant from being violated by a racing batch).
        if self._halted:
            logger.warning("Batch '%s' dropped: motion halted (Stop) during batch", name)
            return

        text = builder.render()
        self.upload_job(text, name)              # slow (FTP) — intentionally OUTSIDE the lock
        self.job_select(name)
        # Re-check the halt latch atomically with the START: a Stop() during the
        # (long) upload above latches _halted under _halt_lock, so either we see
        # it here and skip, or job_start runs and a subsequent Stop() servo-OFFs.
        with self._halt_lock:
            if self._halted:
                logger.warning("Batch '%s' dropped: motion halted (Stop) before START", name)
                return
            self.job_start()
        self._wait_idle()

    def _batch_append_motion(self, target: Any, kind: str) -> None:
        """Helper: queue MoveJ/MoveL into the batch builder."""
        joints = self._to_joint_list(target)
        pos_name = f"p{self._batch_pos_counter}"
        self._batch_pos_counter += 1
        self._batch_builder.add_position(pos_name, joints)
        if kind == "movj":
            self._batch_builder.movj(pos_name)
        else:
            self._batch_builder.movl(pos_name)

    # ─── Motion (M3 — branched theo batch state) ───
    def _is_pose_4x4(self, target: Any) -> bool:
        """Detect whether target is a 4x4 pose (numpy/list-of-lists)."""
        try:
            arr = target if hasattr(target, "shape") else None
            if arr is not None and getattr(arr, "shape", None) == (4, 4):
                return True
            # list of lists 4x4
            if (isinstance(target, (list, tuple)) and len(target) == 4
                    and all(hasattr(r, "__len__") and len(r) == 4 for r in target)):
                return True
        except Exception:                                # noqa: BLE001
            pass
        return False

    def MoveJ(self, target: Any) -> None:
        """Joint move to target.

        Polymorphic — auto-detects target type:
          - list/tuple of 6 floats → joint angles (degrees) → P-variable PULSE
          - 4x4 matrix → Cartesian pose in BASE frame → P-variable BASE,
            YRC1000 computes IK internally

        Non-batch: 1 INFORM upload + JOB_START + wait (200-300ms overhead).
        Batch (M3): appends instruction to batch builder (immediate return).
        Ultra-fast batch (M3++): appends to pvar buffer.
        """
        # Cartesian path (4x4 numpy/list-of-lists)
        if self._is_pose_4x4(target):
            return self._move_pose(target, kind="movj")

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
        """Linear move — polymorphic like MoveJ (joints or 4x4 Cartesian)."""
        if self._is_pose_4x4(target):
            return self._move_pose(target, kind="movl")

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

    def _move_pose(self, T_base: Any, kind: str) -> None:
        """Single-shot Cartesian move via P-variable BASE + small INFORM job.

        Workflow:
          1. WRITE_POS_VAR P000 ← Cartesian pose (data_type=BASE)
          2. Upload tiny INFORM job referencing P000 with MOVJ/MOVL
          3. JOB_SELECT + START + wait_idle

        Same pattern as non-batch joint MoveJ but uses P-var instead of C-var.
        In batch mode only joint targets are currently supported — Cartesian batch
        is future work (requires INFORM template with BASE-typed P-vars).
        """
        if self._batch_builder is not None or self._pvar_batch_buffer is not None:
            raise NotImplementedError(
                "Cartesian MoveJ/MoveL inside a batch context is not yet supported. "
                "Call outside batch or use joint path."
            )

        from src.orchestrator.frame_convert import matrix_to_xyzrpy_yaskawa
        tool_no = getattr(self, "tool_no", 1)               # TOOL01 default
        user_frame = 0
        form = 0                                            # default IK branch

        # 1. Write P000 with Cartesian pose
        self.write_pose_var(p_index=0, T_base=T_base,
                            tool_no=tool_no, user_frame=user_frame,
                            form=form, data_type=DataType.BASE)

        # 2. Upload INFORM job that references P000 with MOVJ/MOVL
        x, y, z, rx, ry, rz = matrix_to_xyzrpy_yaskawa(T_base)
        job_name = self._next_job_name()
        # Inline INFORM with P-variable + BASE position
        instr = ("MOVJ P000" if kind == "movj" else "MOVL P000")
        if kind == "movj":
            instr += f" VJ={self.max_speed_pct:.2f}"
        else:
            instr += " V=80.0"
        job_text = (
            "/JOB\r\n"
            f"//NAME {job_name}\r\n"
            "//POS\r\n"
            "///NPOS 0,0,0,0,0,0\r\n"
            f"///TOOL {tool_no}\r\n"
            "///POSTYPE BASE\r\n"
            "///BASE\r\n"
            "//INST\r\n"
            "///DATE 2026/01/01 00:00\r\n"
            "///ATTR SC,RW\r\n"
            "///GROUP1 RB1\r\n"
            "NOP\r\n"
            f"'Cartesian {kind} target=({x:.1f},{y:.1f},{z:.1f})mm\r\n"
            f"{instr}\r\n"
            "END\r\n"
        )
        self.upload_job(job_text, job_name)
        self.job_select(job_name)
        self.job_start()
        self._wait_idle()

    def timer(self, seconds: float) -> None:
        """Pause for `seconds`.

        Non-batch: time.sleep — script thread waits.
        Batch (M3): appends INFORM TIMER — robot waits during job execution.
        Ultra-fast (M3++): TIMER baked into template (after DOUT), buffer only tracks.
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
        """Convert MoveJ target to list[float] of 6 elements (degrees)."""
        if isinstance(target, (list, tuple)) and len(target) >= 6:
            return [float(j) for j in target[:6]]
        raise NotImplementedError(
            "HSE backend MoveJ currently only accepts joints (list of 6 elements). "
            "4x4 pose requires a client-side IK solver (ikfast/GP7 DH) — planned for a later phase."
        )

    def MoveJ_Test(self, j_start: Any, target: Any, *args: Any) -> int:
        """Reachability check via sphere envelope (if provided), does not call YRC1000.

        RoboDK convention: 0 = OK, < 0 = unreachable.

        If `reach_envelope` is set at construction, uses it. Otherwise, returns
        permissive 0 — caller is responsible (e.g. via DigitalTwinMirror delegating
        to RoboDK robot item).
        """
        if self.reach_envelope is None:
            return 0
        # A 6-element JOINT list is NOT a Cartesian point — the sphere envelope would
        # mis-read its first 3 values as XYZ mm. Be permissive for joint targets
        # (their reach is bounded by the joint-limit check); only sphere-check an
        # actual 4x4 pose or 3-vector.
        try:
            if np.asarray(target, dtype=float).shape == (6,):
                return 0
        except Exception:                               # noqa: BLE001
            pass
        return 0 if self.reach_envelope.can_reach(target) else -1

    def SolveIK(self, pose: Any, joints_approx: Any = None) -> Any:
        """HSE has no IK solver. Must be computed client-side (ikfast / pyrobot)."""
        raise NotImplementedError(
            "IK via HSE: must be computed client-side. See ikfast / Yaskawa GP7 DH."
        )

    # ─── I/O ───
    def setDO(self, index: int, value: int) -> None:
        """Write digital output to control gripper.

        Non-batch: WRITE_IO command via HSE (sets network I/O bit via CIO ladder).
        Batch: appends DOUT instruction to INFORM job (sets Y-output directly
        when job execution reaches that point).

        Non-batch mapping: index → bit NETWORK_IO_BASE + (index-1). CIO ladder
        on YRC1000 must route this bit to a physical Y-output.
        """
        if self._pvar_batch_buffer is not None:
            self._pvar_batch_buffer.append(("dout", int(bool(value))))
            return
        if self._batch_builder is not None:
            self._batch_builder.dout(index, bool(value))
            return

        bit_addr = NETWORK_IO_BASE + (index - 1) * 1     # 1 bit/index
        # WRITE_IO command 0x78, service SET_ATTRIBUTE_SINGLE (0x10)
        # Payload: 4-byte uint32 = value (0 or 1)
        payload = struct.pack("<I", 1 if value else 0)
        self._send_request(
            Command.WRITE_IO, instance=bit_addr, attribute=1,
            service=Service.SET_ATTRIBUTE_SINGLE, payload=payload,
        )
        logger.debug("HSE setDO(index=%d → bit %d) = %d", index, bit_addr, value)

    def set_io(self, bit_addr: int, value: int) -> None:
        """Write 1 network I/O bit via HSE WRITE_IO at an ABSOLUTE bit address.

        Unlike `setDO(index)` (relative to NETWORK_IO_BASE), `set_io` uses
        an absolute address — required for CC-Link bits (range ~30010+) or
        I/O modules not mapped to the general network range 27010.

        Args:
            bit_addr: Absolute bit address (e.g. 30010 for CC-Link output).
            value: 0 or 1.
        """
        payload = struct.pack("<I", 1 if value else 0)
        self._send_request(
            Command.WRITE_IO, instance=bit_addr, attribute=1,
            service=Service.SET_ATTRIBUTE_SINGLE, payload=payload,
        )
        logger.debug("HSE set_io(bit=%d) = %d", bit_addr, value)

    def read_io(self, bit_addr: int) -> int:
        """Read 1 network I/O bit via HSE READ_IO (cmd 0x78).

        Used to read sensor feedback from PLC via CC-Link bridge:
        - Gripper position sensors (X503 UnClamp, X504 Clamp)
        - Object detect sensor (X505 Carrier)

        Args:
            bit_addr: Absolute bit address. E.g. 30050 for clamp sensor (CC-Link mapped).

        Returns:
            0 or 1. -1 on error.
        """
        try:
            resp = self._send_request(
                Command.READ_IO, instance=bit_addr, attribute=1,
                service=Service.GET_ATTRIBUTE_SINGLE,
            )
            if not resp.payload:
                return -1
            # Payload: 4-byte uint32 = value (0 or 1)
            value = struct.unpack("<I", resp.payload[:4])[0]
            return int(value & 0x01)
        except Exception as e:                              # noqa: BLE001
            logger.warning("read_io(%d) error: %s", bit_addr, e)
            return -1

    def setSpeed(self, linear_mm_s: float, joint_deg_s: float = -1) -> None:
        """No-op — HSE does not control speed directly. Speed is set inside the INFORM job."""
        logger.debug(
            "setSpeed(linear=%.1f, joint=%.1f) - HSE no-op, set in INFORM",
            linear_mm_s, joint_deg_s,
        )

    def Stop(self) -> bool:
        """Emergency stop: turns off servo on YRC1000.

        HOLD_SERVO (0x83), instance 2 = Servo-ON control, data part 1=ON / 2=OFF
        per the Yaskawa HSE spec. The previous value=0 is NOT a valid data value
        and the controller rejects it (status != 0) → the E-stop was a silent
        no-op. Sending 2 = servo OFF. To resume, re-enable servo + reset alarm on
        the TP.

        SAFETY: this is software stop only — verify the value/instance against the
        actual controller; never rely on it in place of a physical E-stop.
        """
        with self._halt_lock:                     # atomic vs the dispatch re-check
            self._halted = True                   # latch: block any pending START
        try:
            import struct as _struct
            self._send_request(
                Command.HOLD_SERVO, instance=2, attribute=1,
                service=Service.SET_ATTRIBUTE_SINGLE,
                payload=_struct.pack("<I", 2),    # 2 = servo OFF (1 = ON)
            )
            logger.warning("HSE Stop(): servo OFF sent to YRC1000")
            return True
        except Exception as e:                    # noqa: BLE001
            logger.error("HSE Stop() error: %s — use physical E-stop!", e)
            return False     # caller may escalate; never relied on vs physical E-stop
