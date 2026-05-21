"""
digital_twin.py
───────────────
DigitalTwinMirror: façade bidirectional digital twin combining:
  - HSE backend (motion + I/O thật → YRC1000)
  - RoboDK robot item (kinematic helper + visualization mirror)
  - Mirror thread (poll real state → cập nhật RoboDK item liên tục)
  - Telemetry logger (CSV state log)

KIẾN TRÚC:
                                          ┌─ MoveJ joints ─> HSE ──> YRC1000
  Orchestrator ──> DigitalTwinMirror ─────┤                            │
       (RoboDK Item-like API)             ├─ IK + reach ──> RoboDK Item│
                                          │   (PC-side)                │
                                          │                            ↓
                                          └─ MirrorThread <── poll Joints (10Hz)
                                                  ↓
                                          setJoints(actual) → RoboDK viewport
                                                  ↓
                                          TelemetryLogger → CSV

Đây là implementation "true digital twin" — RoboDK viewport phản ánh state
THẬT của robot, không chỉ commanded state.
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

    Duck-types RoboDK Item interface để Orchestrator gọi nguyên văn. Bên trong:
      - motion + I/O → HSE backend
      - IK + reachability → RoboDK robot item (kinematic helper)
      - actual joint state mirror → RoboDK robot item visualize

    Args:
        hse_backend: MotomanHSEBackend đã connect.
        robodk_robot_item: RoboDK Item của robot trong cell (cho IK + visualize).
            Có thể None → IK disable + visualize disable (chỉ HSE bare-metal).
        telemetry: TelemetryLogger để log state. None → skip.
        mirror_hz: Tần số poll Joints() từ YRC1000. Default 10Hz.
        drift_warn_deg: Threshold cảnh báo lệch commanded vs actual.
    """

    def __init__(
        self,
        hse_backend: Any,
        robodk_robot_item: Any = None,
        telemetry: Any = None,
        mirror_hz: float = DEFAULT_MIRROR_HZ,
        telemetry_hz: float = DEFAULT_TELEMETRY_HZ,
        drift_warn_deg: float = DEFAULT_DRIFT_WARN_DEG,
        alarm_poll_period_s: float = DEFAULT_ALARM_POLL_PERIOD_S,
        auto_stop_on_major_alarm: bool = True,
        viewport_mirror_enabled: bool = True,
    ) -> None:
        self.hse = hse_backend
        self.rdk_item = robodk_robot_item
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
        # Tắt setJoints lên RoboDK item → 0 API call vào RoboDK Free (an toàn
        # nagware popup). Telemetry CSV vẫn ghi state thật, visualize offline.
        self.viewport_mirror_enabled = bool(viewport_mirror_enabled)

        self._mirror_thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._last_commanded: list[float] | None = None
        self._last_actual: list[float] | None = None
        self._current_alarm: AlarmInfo = decode_alarm(0)
        self._auto_stopped = False
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
          1. Poll Joints từ HSE        (mỗi tick, ~10Hz)
          2. Log telemetry CSV          (mỗi tick, ~10Hz)
          3. Drift detection            (mỗi tick, ~10Hz)
          4. setJoints lên RoboDK       (mỗi N tick, throttle xuống ~2Hz)
          5. Poll alarm                 (mỗi alarm_poll_period_s, ~0.4Hz)
          6. Flush telemetry disk       (mỗi 1s)

        Decouple cho phép telemetry resolution cao (cho analysis chính xác)
        mà không spam RoboDK setJoints (tránh nagware popup Free license).
        """
        period = 1.0 / self.telemetry_hz
        last_flush = time.monotonic()

        while not self._stop_flag.is_set():
            tick_start = time.monotonic()
            try:
                actual = self.hse.Joints()                # (1) Poll real state

                # (4) Viewport throttled — KHÔNG mỗi tick để tránh spam RoboDK
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
                    self.hse.Stop()
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
        """Cập nhật vị trí joints của RoboDK item → viewport mirror robot thật.

        Skip nếu `viewport_mirror_enabled=False` — tránh API call vào RoboDK
        khi dùng Free license (nagware popup sau ~30 actions).
        """
        if self.rdk_item is None or not self.viewport_mirror_enabled:
            return
        try:
            self.rdk_item.setJoints(joints)
        except Exception as e:                           # noqa: BLE001
            logger.debug("setJoints lỗi (viewport mirror): %s", e)

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
    # Duck-type RoboDK Item — forward calls
    # ────────────────────────────────────────────────────────────────────
    def Valid(self) -> bool:
        return self.hse.Valid()

    def Joints(self) -> list[float]:
        return self.hse.Joints()

    def JointsHome(self) -> Any:
        return self.hse.JointsHome()

    def Parent(self) -> Any:
        """Forward tới RoboDK item để Orchestrator tính world→parent transform."""
        if self.rdk_item is not None and hasattr(self.rdk_item, "Parent"):
            try:
                return self.rdk_item.Parent()
            except Exception:                            # noqa: BLE001
                return None
        return None

    def setSpeed(self, linear_mm_s: float, joint_deg_s: float = -1) -> None:
        self.hse.setSpeed(linear_mm_s, joint_deg_s)

    def setDO(self, index: int, value: int) -> None:
        self.hse.setDO(index, value)

    def Stop(self) -> None:
        """Emergency stop."""
        self.hse.Stop()

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
    # Kinematic delegation (RoboDK Item dùng để tính IK + reachability)
    # ────────────────────────────────────────────────────────────────────
    def SolveIK(self, pose: Any, joints_approx: Any = None) -> Any:
        if self.rdk_item is None:
            raise NotImplementedError(
                "DigitalTwinMirror cần robodk_robot_item để IK. "
                "Truyền RoboDK item khi construct hoặc giải IK client-side."
            )
        if joints_approx is not None:
            return self.rdk_item.SolveIK(pose, joints_approx)
        return self.rdk_item.SolveIK(pose)

    def MoveJ_Test(self, j_start: Any, target: Any, *args: Any) -> int:
        if self.rdk_item is None:
            return 0    # no kinematic check available, assume reachable
        return self.rdk_item.MoveJ_Test(j_start, target, *args)

    # ────────────────────────────────────────────────────────────────────
    # Motion — convert pose → joints (via RoboDK IK) → HSE backend
    # ────────────────────────────────────────────────────────────────────
    def MoveJ(self, target: Any) -> None:
        joints = self._target_to_joints(target)
        with self._lock:
            self._last_commanded = list(joints)
        self.hse.MoveJ(joints)

    def MoveL(self, target: Any) -> None:
        joints = self._target_to_joints(target)
        with self._lock:
            self._last_commanded = list(joints)
        self.hse.MoveL(joints)

    def _target_to_joints(self, target: Any) -> list[float]:
        """Convert target (joint list HOẶC pose 4x4) sang joint list[float]."""
        # Joint list
        if isinstance(target, (list, tuple)) and len(target) >= 6:
            try:
                return [float(j) for j in target[:6]]
            except (TypeError, ValueError):
                pass

        # Numpy 4x4 pose → IK
        if isinstance(target, np.ndarray) and target.shape == (4, 4):
            if self.rdk_item is None:
                raise NotImplementedError(
                    "MoveJ với pose 4x4 cần robodk_robot_item để IK."
                )
            current = self.Joints()
            sol = self.rdk_item.SolveIK(target, current)
            return self._mat_to_joint_list(sol)

        # RoboDK Mat → IK
        if hasattr(target, "Pos") and self.rdk_item is not None:
            current = self.Joints()
            sol = self.rdk_item.SolveIK(target, current)
            return self._mat_to_joint_list(sol)

        raise ValueError(f"MoveJ target type không hỗ trợ: {type(target)}")

    @staticmethod
    def _mat_to_joint_list(sol: Any) -> list[float]:
        """Convert RoboDK SolveIK return → list[float]."""
        if sol is None:
            raise RuntimeError("SolveIK trả None — pose ngoài tầm với")
        if isinstance(sol, (list, tuple)):
            if len(sol) < 6:
                raise RuntimeError(f"SolveIK trả {len(sol)} phần tử, kỳ vọng ≥6")
            return [float(j) for j in sol[:6]]
        # RoboDK Mat with .list() method
        if hasattr(sol, "list") and callable(sol.list):
            data = sol.list()
            if len(data) < 6:
                raise RuntimeError(f"SolveIK Mat.list() có {len(data)} phần tử")
            return [float(j) for j in data[:6]]
        # numpy array
        arr = np.asarray(sol).flatten()
        if arr.size < 6:
            raise RuntimeError(f"SolveIK numpy có {arr.size} phần tử")
        return [float(j) for j in arr[:6]]
