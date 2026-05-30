"""
digital_twin.py
───────────────
DigitalTwinMirror: façade bidirectional digital twin combining:
  - Motion backend (HSE → YRC1000 thật, hoặc SimRobot cho dev)
  - Optional viewport callback (Open3D mirror — render robot state live)
  - Mirror thread (poll real state @ 10Hz)
  - Telemetry logger (CSV state log)
  - Drift detection + auto-stop on major alarms

KIẾN TRÚC:
  Orchestrator ──> DigitalTwinMirror ───── MoveJ joints ──> Backend ──> Robot
       (duck-type API)                                                    │
                       MirrorThread <───── poll Joints (10Hz) ────────────┘
                            ↓
                       viewport_callback(joints)  ← optional, ~2Hz
                            ↓
                       TelemetryLogger → CSV (10Hz)

Viewport callback nhận joints (degrees) mỗi N tick — Open3D mirror sẽ render
state THẬT (không phải commanded). Backend yêu cầu tự handle frame conversion
(world → robot base) bằng `frame_convert.world_to_robot_base` khi gọi MoveJ
với pose 4x4.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import Any, Iterator

import numpy as np

from .backends.alarm_codes import AlarmInfo, AlarmSeverity, decode_alarm, is_recoverable

logger = logging.getLogger(__name__)


# ─── Decoupled rates ───
# Loop chạy ở telemetry_hz (cao) → mỗi tick: poll Joints + log CSV + drift check.
# Viewport setJoints throttle xuống mirror_hz (thấp) để tránh RoboDK Free nagware.
# Alarm poll theo period giây thay vì tick (decouple khỏi loop rate).
#
# Defaults được chọn để:
#   - Telemetry 10Hz: đủ resolution cho velocity analysis (post-process)
#   - Mirror 2Hz: smooth cho mắt người + an toàn RoboDK Free (~120 setJoints/phút)
#   - Alarm 2.5s: alarm state ít đổi, không cần poll thường xuyên
DEFAULT_TELEMETRY_HZ = 10.0
DEFAULT_MIRROR_HZ = 2.0
DEFAULT_ALARM_POLL_PERIOD_S = 2.5

# Drift warning: nếu lệch giữa commanded và actual ≥ ngưỡng thì log warning.
DEFAULT_DRIFT_WARN_DEG = 2.0


class DigitalTwinMirror:
    """Bidirectional digital twin façade.

    Bên trong:
      - motion + I/O → backend (HSE cho real, SimRobot cho dev)
      - actual joint state mirror → optional viewport_callback (Open3D mirror)
      - telemetry logging + drift detection + alarm handling

    Args:
        backend: Robot backend duck-typed (MotomanHSEBackend hoặc SimRobot).
        viewport_callback: Callable(joints_deg: list[float]) → None, gọi mỗi N
            tick để render robot state. None → không có viewport mirror.
        telemetry: TelemetryLogger để log state. None → skip.
        mirror_hz: Tần số gọi viewport_callback. Default 2Hz.
        telemetry_hz: Tần số poll Joints + log CSV. Default 10Hz.
        drift_warn_deg: Threshold cảnh báo lệch commanded vs actual.
    """

    def __init__(
        self,
        backend: Any,
        viewport_callback: Any = None,
        telemetry: Any = None,
        mirror_hz: float = DEFAULT_MIRROR_HZ,
        telemetry_hz: float = DEFAULT_TELEMETRY_HZ,
        drift_warn_deg: float = DEFAULT_DRIFT_WARN_DEG,
        alarm_poll_period_s: float = DEFAULT_ALARM_POLL_PERIOD_S,
        auto_stop_on_major_alarm: bool = True,
        viewport_mirror_enabled: bool = True,
        grasp_callback: Any = None,
        release_callback: Any = None,
        reset_callback: Any = None,
    ) -> None:
        self.backend = backend
        # Backward compat: nhiều site truy cập `.hse` — giữ alias để khỏi break.
        self.hse = backend
        self.viewport_callback = viewport_callback
        self.telemetry = telemetry
        # Loop chạy ở telemetry_hz (cao). Viewport throttle theo mirror_hz.
        self.telemetry_hz = max(0.5, float(telemetry_hz))
        self.mirror_hz = min(max(0.5, float(mirror_hz)), self.telemetry_hz)
        self._viewport_throttle = max(
            1, int(round(self.telemetry_hz / self.mirror_hz))
        )
        self._viewport_tick = 0
        self.drift_warn_deg = float(drift_warn_deg)
        # Min 1ms — đủ cho any reasonable use (alarm state ít đổi nhanh hơn).
        self.alarm_poll_period_s = max(0.001, float(alarm_poll_period_s))
        # 0.0 = fire ngay tick đầu (không đợi alarm_poll_period_s sau start).
        self._next_alarm_poll_t: float = 0.0
        self.auto_stop_on_major_alarm = bool(auto_stop_on_major_alarm)
        # Tắt viewport_callback → mirror loop vẫn chạy (telemetry + drift +
        # alarm), chỉ skip render. Dùng để giảm overhead khi không cần visual.
        self.viewport_mirror_enabled = bool(viewport_mirror_enabled)
        # Optional viewport-visual hooks: Orchestrator gọi attach_object/
        # detach_object/reset_scene (duck-typed, có hasattr guard) → forward sang
        # callback để app GUI gắn/thả/reset vật trong scene. None = no-op.
        self._grasp_callback = grasp_callback
        self._release_callback = release_callback
        self._reset_callback = reset_callback

        self._mirror_thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._last_commanded: list[float] | None = None
        self._last_actual: list[float] | None = None
        self._current_alarm: AlarmInfo = decode_alarm(0)
        self._auto_stopped = False
        # E-stop latch: sau Stop()/auto-stop-on-alarm → từ chối MỌI lệnh motion
        # tiếp theo (MoveJ/MoveL) cho tới khi start_mirror() clear. Chuẩn công
        # nghiệp: không gửi lệnh chuyển động sau khi đã hold/E-stop.
        self._motion_halted = threading.Event()
        self._lock = threading.Lock()

    # ────────────────────────────────────────────────────────────────────
    # Mirror lifecycle
    # ────────────────────────────────────────────────────────────────────
    def start_mirror(self) -> None:
        """Spawn daemon thread polling actual state + cập nhật twin."""
        if self._mirror_thread is not None and self._mirror_thread.is_alive():
            logger.debug("Mirror thread đã chạy")
            return
        if self.telemetry is not None:
            self.telemetry.open()
        self._stop_flag.clear()
        self._motion_halted.clear()     # (re)start → cho phép motion trở lại
        # Fire alarm poll trên tick đầu (0.0 < monotonic), giúp test ngắn pass
        # + thực tế cũng hợp lý: check alarm ngay khi start để biết state ban đầu.
        self._next_alarm_poll_t = 0.0
        self._mirror_thread = threading.Thread(
            target=self._mirror_loop, name="DigitalTwinMirror", daemon=True,
        )
        self._mirror_thread.start()
        logger.info(
            "Mirror thread started — telemetry %.1f Hz, viewport %.1f Hz "
            "(throttle 1/%d), alarm every %.1fs",
            self.telemetry_hz, self.mirror_hz, self._viewport_throttle,
            self.alarm_poll_period_s,
        )

    def stop_mirror(self, timeout_s: float = 2.0) -> None:
        """Signal stop + join thread + close telemetry."""
        self._stop_flag.set()
        if self._mirror_thread is not None:
            self._mirror_thread.join(timeout=timeout_s)
            self._mirror_thread = None
        if self.telemetry is not None:
            self.telemetry.close()
        logger.info("Mirror thread stopped")

    def _mirror_loop(self) -> None:
        """Main loop chạy ở telemetry_hz (cao). Mỗi tick:
          1. Poll Joints từ backend     (mỗi tick, ~10Hz)
          2. Log telemetry CSV          (mỗi tick, ~10Hz)
          3. Drift detection            (mỗi tick, ~10Hz)
          4. viewport_callback(joints)  (mỗi N tick, throttle xuống ~2Hz)
          5. Poll alarm                 (mỗi alarm_poll_period_s, ~0.4Hz)
          6. Flush telemetry disk       (mỗi 1s)

        Decouple cho phép telemetry resolution cao (cho analysis chính xác)
        mà không spam viewport render.
        """
        period = 1.0 / self.telemetry_hz
        last_flush = time.monotonic()

        while not self._stop_flag.is_set():
            tick_start = time.monotonic()
            try:
                actual = self.hse.Joints()                # (1) Poll real state

                # (4) Viewport throttled — chỉ gọi callback mỗi N tick
                self._viewport_tick += 1
                if self._viewport_tick >= self._viewport_throttle:
                    self._viewport_tick = 0
                    self._update_viewport(actual)

                self._check_drift(actual)                  # (3) Drift mỗi tick

                # (5) Alarm theo period giây (không phải tick) — decouple loop rate
                alarm_code = 0
                if tick_start >= self._next_alarm_poll_t:
                    self._next_alarm_poll_t = tick_start + self.alarm_poll_period_s
                    alarm_code = self._poll_alarm()

                # (2) Telemetry CSV mỗi tick — resolution cao cho analysis
                if self.telemetry is not None:
                    self.telemetry.log_state(actual, alarm=alarm_code or None)
                with self._lock:
                    self._last_actual = list(actual)
            except Exception as e:                          # noqa: BLE001
                logger.warning("Mirror tick lỗi: %s", e)

            # (6) Flush telemetry mỗi 1s thay vì mỗi tick → giảm IO disk.
            if self.telemetry is not None and (time.monotonic() - last_flush) > 1.0:
                self.telemetry.flush()
                last_flush = time.monotonic()

            elapsed = time.monotonic() - tick_start
            sleep_for = max(0.001, period - elapsed)
            self._stop_flag.wait(sleep_for)

    def _poll_alarm(self) -> int:
        """Poll alarm code từ YRC1000 + auto-respond theo severity.

        Returns: alarm code (0 = không có alarm).
        """
        if not hasattr(self.hse, "read_alarm"):
            return 0
        try:
            code, sub_code = self.hse.read_alarm()
        except Exception as e:                       # noqa: BLE001
            logger.debug("read_alarm() lỗi: %s", e)
            return 0

        info = decode_alarm(code, sub_code)
        with self._lock:
            prev = self._current_alarm
            self._current_alarm = info

        # Log alarm transitions (NEW alarm hoặc CHANGED code)
        if code != 0 and info.code != prev.code:
            logger.error(
                "ALARM %d (%s, severity=%s): %s. Recovery: %s",
                code, info.name, info.severity.name,
                info.description, info.recovery_hint,
            )
            # Auto-stop nếu MAJOR/SYSTEM và config bật + chưa stop trước đó
            if (self.auto_stop_on_major_alarm
                    and info.severity in (AlarmSeverity.MAJOR, AlarmSeverity.SYSTEM)
                    and not self._auto_stopped):
                logger.error("Auto-stop triggered bởi alarm %d", code)
                try:
                    self.Stop()          # latch motion-halt + servo-off backend
                    self._auto_stopped = True
                except Exception as e:               # noqa: BLE001
                    logger.error("Auto-stop Stop() lỗi: %s", e)
        elif code == 0 and prev.code != 0:
            # Alarm đã được clear (manual reset trên TP)
            logger.info("Alarm %d (%s) đã clear", prev.code, prev.name)
            self._auto_stopped = False

        return code

    def current_alarm(self) -> AlarmInfo:
        """Snapshot alarm hiện tại (cho Orchestrator hỏi giữa các trial)."""
        with self._lock:
            return self._current_alarm

    def is_alarm_active(self) -> bool:
        """True nếu đang có alarm chưa clear."""
        return self.current_alarm().code != 0

    def _update_viewport(self, joints: list[float]) -> None:
        """Gọi viewport_callback với joints (degrees) — render robot state.

        Skip nếu `viewport_mirror_enabled=False` hoặc callback chưa set.
        """
        if self.viewport_callback is None or not self.viewport_mirror_enabled:
            return
        try:
            self.viewport_callback(joints)
        except Exception as e:                           # noqa: BLE001
            logger.debug("viewport_callback lỗi: %s", e)

    def _check_drift(self, actual: list[float]) -> None:
        """So sánh actual vs last commanded → warn nếu drift quá lớn."""
        with self._lock:
            commanded = self._last_commanded
        if commanded is None or len(commanded) != len(actual):
            return
        deltas = [abs(a - c) for a, c in zip(actual, commanded)]
        max_drift = max(deltas)
        if max_drift > self.drift_warn_deg:
            logger.warning(
                "Drift cao: max %.2f° (per-axis: %s°). "
                "Commanded=%s, actual=%s",
                max_drift,
                [f"{d:.2f}" for d in deltas],
                [f"{c:.1f}" for c in commanded],
                [f"{a:.1f}" for a in actual],
            )

    # ────────────────────────────────────────────────────────────────────
    # Forward calls đến backend
    # ────────────────────────────────────────────────────────────────────
    def Valid(self) -> bool:
        return self.hse.Valid()

    def Joints(self) -> list[float]:
        return self.hse.Joints()

    def JointsHome(self) -> Any:
        return self.hse.JointsHome()

    def setSpeed(self, linear_mm_s: float, joint_deg_s: float = -1) -> None:
        self.hse.setSpeed(linear_mm_s, joint_deg_s)

    def setDO(self, index: int, value: int) -> None:
        self.hse.setDO(index, value)

    def set_io(self, bit_addr: int, value: int) -> None:
        """Forward set_io (absolute bit) tới backend — cho CC-Link bits."""
        if hasattr(self.backend, "set_io") and callable(self.backend.set_io):
            self.backend.set_io(bit_addr, value)

    def read_io(self, bit_addr: int) -> int:
        """Forward read_io tới backend (CC-Link sensor feedback)."""
        if hasattr(self.backend, "read_io") and callable(self.backend.read_io):
            return self.backend.read_io(bit_addr)
        return -1

    def Stop(self) -> None:
        """Emergency stop: latch motion-halt RỒI servo-off backend.

        Set `_motion_halted` TRƯỚC khi forward → mọi MoveJ/MoveL sau đó (kể cả
        đang giữa một chu trình Orchestrator) bị từ chối ngay, không gửi thêm
        lệnh tới robot. No-op forward nếu backend không support (SimRobot).
        """
        self._motion_halted.set()
        if hasattr(self.backend, "Stop") and callable(self.backend.Stop):
            self.backend.Stop()

    # ────────────────────────────────────────────────────────────────────
    # Batch + timer — forward to HSE backend (nếu support)
    # ────────────────────────────────────────────────────────────────────
    def batch(self, job_name: str | None = None) -> Any:
        """Context manager: gom motion + IO vào 1 INFORM job (M3 optimization).

        Forward tới HSE backend nếu support, ngược lại trả nullcontext (no-op)
        để orchestrator code dùng được với mọi backend không cần check.
        """
        if hasattr(self.hse, "batch") and callable(self.hse.batch):
            return self.hse.batch(job_name)
        return contextlib.nullcontext()

    def timer(self, seconds: float) -> None:
        """Pause — INFORM TIMER trong batch mode, time.sleep ngoài batch."""
        if hasattr(self.hse, "timer") and callable(self.hse.timer):
            self.hse.timer(seconds)
        else:
            time.sleep(seconds)

    # ────────────────────────────────────────────────────────────────────
    # Reachability (predictive safety nếu Orchestrator enable; else assume OK)
    # ────────────────────────────────────────────────────────────────────
    def MoveJ_Test(self, j_start: Any, target: Any, *args: Any) -> int:
        """No-op reachability check — kinematic check thực sự nằm ở
        Orchestrator predictive safety (UC2). Trả 0 = assume reachable.
        """
        return 0

    # ────────────────────────────────────────────────────────────────────
    # Motion — Orchestrator đã giải IK (client DLS) trước khi gọi đây
    # Hoặc pass-through 4x4 pose nếu backend support YRC IK Cartesian
    # ────────────────────────────────────────────────────────────────────
    def _backend_supports_cartesian(self) -> bool:
        """True nếu backend có thể nhận 4x4 pose (HSE BASE Cartesian).

        Check class-level flag `supports_cartesian_pose=True` thay vì
        hasattr — MagicMock auto-create attr nên hasattr không phân biệt được.
        """
        return getattr(type(self.backend), "supports_cartesian_pose", False) is True

    def _check_not_halted(self) -> None:
        """Raise nếu đã Stop()/auto-stop — chặn motion sau E-stop/hold."""
        if self._motion_halted.is_set():
            raise RuntimeError(
                "Digital Twin halted (Stop/alarm) — motion command refused")

    def MoveJ(self, target: Any) -> None:
        self._check_not_halted()
        # Cartesian pass-through cho HSE backend khi target là 4x4 pose
        if (isinstance(target, np.ndarray) and target.shape == (4, 4)
                and self._backend_supports_cartesian()):
            with self._lock:
                self._last_commanded = None         # joints không biết trước
            self.backend.MoveJ(target)
            return
        joints = self._target_to_joints(target)
        with self._lock:
            self._last_commanded = list(joints)
        self.backend.MoveJ(joints)

    def MoveL(self, target: Any) -> None:
        self._check_not_halted()
        if (isinstance(target, np.ndarray) and target.shape == (4, 4)
                and self._backend_supports_cartesian()):
            with self._lock:
                self._last_commanded = None
            self.backend.MoveL(target)
            return
        joints = self._target_to_joints(target)
        with self._lock:
            self._last_commanded = list(joints)
        self.backend.MoveL(joints)

    # ── Viewport-visual hooks (Orchestrator gọi nếu robot hỗ trợ) ────────
    def attach_object(self, class_name: Any = None) -> None:
        """Orchestrator báo gripper đã GẮP vật → forward để app gắn vật vào
        gripper trong viewport (visual). No-op nếu không set callback."""
        if self._grasp_callback is not None:
            try:
                self._grasp_callback(class_name)
            except Exception as e:                              # noqa: BLE001
                logger.debug("grasp_callback lỗi: %s", e)

    def detach_object(self) -> None:
        """Orchestrator báo gripper đã THẢ vật → forward để app thả vật."""
        if self._release_callback is not None:
            try:
                self._release_callback()
            except Exception as e:                              # noqa: BLE001
                logger.debug("release_callback lỗi: %s", e)

    def reset_scene(self) -> None:
        """Orchestrator reset đầu mỗi trial → forward để app đưa vật về chỗ cũ."""
        if self._reset_callback is not None:
            try:
                self._reset_callback()
            except Exception as e:                              # noqa: BLE001
                logger.debug("reset_callback lỗi: %s", e)

    def _target_to_joints(self, target: Any) -> list[float]:
        """Convert target joint list → list[float] of 6 floats.

        Pose 4x4 KHÔNG được handle ở đây — caller (MoveJ/MoveL) đã pass-through
        4x4 pose tới backend nếu backend support Cartesian. Orchestrator giải
        client-side DLS IK trước khi gọi với joints.
        """
        if isinstance(target, (list, tuple)) and len(target) >= 6:
            try:
                return [float(j) for j in target[:6]]
            except (TypeError, ValueError) as e:
                raise ValueError(f"target joint values không phải float: {e}")
        raise ValueError(
            f"MoveJ target type không hỗ trợ: {type(target)}. "
            f"Truyền joint list[float] hoặc 4x4 numpy pose (cần "
            f"backend.supports_cartesian_pose=True)."
        )

    @staticmethod
    def _mat_to_joint_list(sol: Any) -> list[float]:
        """Convert IK solution iterable → list[float]."""
        if sol is None:
            raise RuntimeError("IK trả None — pose ngoài tầm với")
        if isinstance(sol, (list, tuple)):
            if len(sol) < 6:
                raise RuntimeError(f"IK trả {len(sol)} phần tử, kỳ vọng ≥6")
            return [float(j) for j in sol[:6]]
        arr = np.asarray(sol).flatten()
        if arr.size < 6:
            raise RuntimeError(f"IK numpy có {arr.size} phần tử")
        return [float(j) for j in arr[:6]]
