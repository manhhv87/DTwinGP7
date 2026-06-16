"""
digital_twin.py
───────────────
DigitalTwinMirror: facade bidirectional digital twin combining:
  - Motion backend (HSE → real YRC1000, or SimRobot for dev)
  - Optional viewport callback (Open3D mirror — render robot state live)
  - Mirror thread (poll real state @ 10Hz)
  - Telemetry logger (CSV state log)
  - Drift detection + auto-stop on major alarms

ARCHITECTURE:
  Orchestrator ──> DigitalTwinMirror ───── MoveJ joints ──> Backend ──> Robot
       (duck-type API)                                                    │
                       MirrorThread <───── poll Joints (10Hz) ────────────┘
                            ↓
                       viewport_callback(joints)  ← optional, ~2Hz
                            ↓
                       TelemetryLogger → CSV (10Hz)

Viewport callback receives joints (degrees) every N ticks — Open3D mirror renders
the ACTUAL state (not commanded). Backend must handle frame conversion
(world → robot base) via `frame_convert.world_to_robot_base` when calling MoveJ
with a 4x4 pose.
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
# Loop runs at telemetry_hz (high) → each tick: poll Joints + log CSV + drift check.
# Viewport setJoints throttled down to mirror_hz (low) to avoid RoboDK Free nagware.
# Alarm polled on a time period rather than per-tick (decoupled from loop rate).
#
# Defaults chosen so that:
#   - Telemetry 10Hz: sufficient resolution for velocity analysis (post-process)
#   - Mirror 2Hz: smooth visually + safe for RoboDK Free (~120 setJoints/min)
#   - Alarm 2.5s: alarm state changes rarely, no need to poll frequently
DEFAULT_TELEMETRY_HZ = 10.0
DEFAULT_MIRROR_HZ = 2.0
DEFAULT_ALARM_POLL_PERIOD_S = 2.5

# Drift warning: if deviation between commanded and actual ≥ threshold, log a warning.
DEFAULT_DRIFT_WARN_DEG = 2.0


class DigitalTwinMirror:
    """Bidirectional digital twin facade.

    Internals:
      - motion + I/O → backend (HSE for real, SimRobot for dev)
      - actual joint state mirror → optional viewport_callback (Open3D mirror)
      - telemetry logging + drift detection + alarm handling

    Args:
        backend: Duck-typed robot backend (MotomanHSEBackend or SimRobot).
        viewport_callback: Callable(joints_deg: list[float]) → None, called every N
            ticks to render robot state. None → no viewport mirror.
        telemetry: TelemetryLogger for state logging. None → skip.
        mirror_hz: Viewport callback call frequency. Default 2Hz.
        telemetry_hz: Joint poll + CSV log frequency. Default 10Hz.
        drift_warn_deg: Threshold for warning on commanded vs actual drift.
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
        # Backward compat: many sites access `.hse` — keep alias to avoid breakage.
        self.hse = backend
        self.viewport_callback = viewport_callback
        self.telemetry = telemetry
        # Loop runs at telemetry_hz (high). Viewport throttled to mirror_hz.
        self.telemetry_hz = max(0.5, float(telemetry_hz))
        self.mirror_hz = min(max(0.5, float(mirror_hz)), self.telemetry_hz)
        self._viewport_throttle = max(
            1, int(round(self.telemetry_hz / self.mirror_hz))
        )
        self._viewport_tick = 0
        self.drift_warn_deg = float(drift_warn_deg)
        # Min 1ms — sufficient for any reasonable use (alarm state changes slowly).
        self.alarm_poll_period_s = max(0.001, float(alarm_poll_period_s))
        # 0.0 = fire on the first tick (don't wait alarm_poll_period_s after start).
        self._next_alarm_poll_t: float = 0.0
        self.auto_stop_on_major_alarm = bool(auto_stop_on_major_alarm)
        # Disabling viewport_callback → mirror loop still runs (telemetry + drift +
        # alarm), only skips rendering. Use to reduce overhead when visuals not needed.
        self.viewport_mirror_enabled = bool(viewport_mirror_enabled)
        # Optional viewport-visual hooks: Orchestrator calls attach_object/
        # detach_object/reset_scene (duck-typed, with hasattr guard) → forwarded to
        # callback so the GUI app can attach/detach/reset objects in the scene. None = no-op.
        self._grasp_callback = grasp_callback
        self._release_callback = release_callback
        self._reset_callback = reset_callback

        self._mirror_thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._last_commanded: list[float] | None = None
        self._last_actual: list[float] | None = None
        self._current_alarm: AlarmInfo = decode_alarm(0)
        self._auto_stopped = False
        # E-stop latch: after Stop()/auto-stop-on-alarm → reject ALL subsequent motion
        # commands (MoveJ/MoveL) until start_mirror() clears it. Industry standard:
        # no motion commands sent after hold/E-stop.
        self._motion_halted = threading.Event()
        self._lock = threading.Lock()

    # ────────────────────────────────────────────────────────────────────
    # Mirror lifecycle
    # ────────────────────────────────────────────────────────────────────
    def start_mirror(self) -> None:
        """Spawn daemon thread polling actual state + updating the twin."""
        if self._mirror_thread is not None and self._mirror_thread.is_alive():
            logger.debug("Mirror thread already running")
            return
        if self.telemetry is not None:
            self.telemetry.open()
        self._stop_flag.clear()
        # SAFETY: do NOT silently re-enable motion if a fault is still active. Only
        # clear the motion-halt latch when there is no active alarm; under a
        # persistent fault the operator must clear it on the teach pendant first,
        # then restart the mirror. Resetting _auto_stopped on a clean restart re-arms
        # auto-stop. (Uses the last-polled alarm; the immediate first poll refreshes
        # it, so a stale-cached alarm at most costs one extra restart — fail-safe.)
        # Refresh the alarm from the controller NOW (synchronous) so the re-arm
        # decision uses the LIVE state, not a value cached from a previous session
        # — a stale code=0 could otherwise clear the halt latch for ~one tick while
        # an alarm is actually active. Only clear the latch when we CONFIRM clean.
        try:
            self._poll_alarm()
            clean = self.current_alarm().code == 0
        except Exception as e:              # noqa: BLE001
            logger.warning("start_mirror: initial alarm read failed (%s) — "
                           "motion stays HALTED until confirmed clean", e)
            clean = False
        if clean:
            self._motion_halted.clear()     # confirmed clean → allow motion again
            self._auto_stopped = False
        else:
            self._motion_halted.set()       # fail-safe: alarm active or unknown
            logger.warning(
                "start_mirror: alarm %d active (or unread) — motion stays HALTED "
                "(clear it on the teach pendant, then restart the mirror)",
                self.current_alarm().code)
        # Fire alarm poll on first tick (0.0 < monotonic), helps short tests pass
        # + practically sensible: check alarm immediately at start to know initial state.
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
        """Main loop running at telemetry_hz (high). Each tick:
          1. Poll Joints from backend   (every tick, ~10Hz)
          2. Log telemetry CSV          (every tick, ~10Hz)
          3. Drift detection            (every tick, ~10Hz)
          4. viewport_callback(joints)  (every N ticks, throttled to ~2Hz)
          5. Poll alarm                 (every alarm_poll_period_s, ~0.4Hz)
          6. Flush telemetry to disk    (every 1s)

        Decoupling allows high telemetry resolution (for accurate analysis)
        without spamming the viewport renderer.
        """
        period = 1.0 / self.telemetry_hz
        last_flush = time.monotonic()
        last_running: bool | None = None                  # read on alarm cadence

        while not self._stop_flag.is_set():
            tick_start = time.monotonic()
            try:
                actual = self.hse.Joints()                # (1) Poll real state

                # (4) Viewport throttled — only call callback every N ticks
                self._viewport_tick += 1
                if self._viewport_tick >= self._viewport_throttle:
                    self._viewport_tick = 0
                    self._update_viewport(actual)

                self._check_drift(actual)                  # (3) Drift every tick

                # (5) Alarm on time period (not per-tick) — decouple from loop rate.
                # Read the Running bit on the SAME slow cadence (cached + logged each
                # tick) so the telemetry 'running' column is populated without adding
                # an HSE round-trip per tick.
                alarm_code = 0
                if tick_start >= self._next_alarm_poll_t:
                    self._next_alarm_poll_t = tick_start + self.alarm_poll_period_s
                    alarm_code = self._poll_alarm()
                    if hasattr(self.hse, "read_status_running"):
                        try:
                            last_running = bool(self.hse.read_status_running())
                        except Exception:                  # noqa: BLE001
                            last_running = None

                # (2) Telemetry CSV every tick — high resolution for analysis
                if self.telemetry is not None:
                    self.telemetry.log_state(
                        actual, running=last_running, alarm=alarm_code or None)
                with self._lock:
                    self._last_actual = list(actual)
            except Exception as e:                          # noqa: BLE001
                logger.warning("Mirror tick error: %s", e)

            # (6) Flush telemetry every 1s instead of every tick → reduce disk IO.
            if self.telemetry is not None and (time.monotonic() - last_flush) > 1.0:
                self.telemetry.flush()
                last_flush = time.monotonic()

            elapsed = time.monotonic() - tick_start
            sleep_for = max(0.001, period - elapsed)
            self._stop_flag.wait(sleep_for)

    def _poll_alarm(self) -> int:
        """Poll alarm code from YRC1000 + auto-respond based on severity.

        Returns: alarm code (0 = no alarm).
        """
        if not hasattr(self.hse, "read_alarm"):
            return 0
        try:
            code, sub_code = self.hse.read_alarm()
        except Exception as e:                       # noqa: BLE001
            logger.debug("read_alarm() error: %s", e)
            return 0

        info = decode_alarm(code, sub_code)
        with self._lock:
            prev = self._current_alarm
            self._current_alarm = info

        # Log alarm transitions (NEW alarm or CHANGED code)
        if code != 0 and info.code != prev.code:
            logger.error(
                "ALARM %d (%s, severity=%s): %s. Recovery: %s",
                code, info.name, info.severity.name,
                info.description, info.recovery_hint,
            )
            # Auto-stop if MAJOR/SYSTEM and config enabled + not already stopped
            if (self.auto_stop_on_major_alarm
                    and info.severity in (AlarmSeverity.MAJOR, AlarmSeverity.SYSTEM)
                    and not self._auto_stopped):
                logger.error("Auto-stop triggered by alarm %d", code)
                try:
                    self.Stop()          # latch motion-halt + servo-off backend
                    self._auto_stopped = True
                except Exception as e:               # noqa: BLE001
                    logger.error("Auto-stop Stop() error: %s", e)
        elif code == 0 and prev.code != 0:
            # Alarm has been cleared (manual reset on TP)
            logger.info("Alarm %d (%s) cleared", prev.code, prev.name)
            self._auto_stopped = False

        return code

    def current_alarm(self) -> AlarmInfo:
        """Snapshot of current alarm (for Orchestrator to query between trials)."""
        with self._lock:
            return self._current_alarm

    def is_alarm_active(self) -> bool:
        """True if an alarm is currently active and uncleared."""
        return self.current_alarm().code != 0

    def _update_viewport(self, joints: list[float]) -> None:
        """Call viewport_callback with joints (degrees) — render robot state.

        Skip if `viewport_mirror_enabled=False` or callback not set.
        """
        if self.viewport_callback is None or not self.viewport_mirror_enabled:
            return
        try:
            self.viewport_callback(joints)
        except Exception as e:                           # noqa: BLE001
            logger.debug("viewport_callback error: %s", e)

    def _check_drift(self, actual: list[float]) -> None:
        """Compare actual vs last commanded → warn if drift is too large."""
        with self._lock:
            commanded = self._last_commanded
        if commanded is None or len(commanded) != len(actual):
            return
        deltas = [abs(a - c) for a, c in zip(actual, commanded)]
        max_drift = max(deltas)
        if max_drift > self.drift_warn_deg:
            logger.warning(
                "High drift: max %.2f° (per-axis: %s°). "
                "Commanded=%s, actual=%s",
                max_drift,
                [f"{d:.2f}" for d in deltas],
                [f"{c:.1f}" for c in commanded],
                [f"{a:.1f}" for a in actual],
            )

    # ────────────────────────────────────────────────────────────────────
    # Forward calls to backend
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
        """Forward set_io (absolute bit) to backend — for CC-Link bits."""
        if hasattr(self.backend, "set_io") and callable(self.backend.set_io):
            self.backend.set_io(bit_addr, value)

    def read_io(self, bit_addr: int) -> int:
        """Forward read_io to backend (CC-Link sensor feedback)."""
        if hasattr(self.backend, "read_io") and callable(self.backend.read_io):
            return self.backend.read_io(bit_addr)
        return -1

    def Stop(self) -> None:
        """Emergency stop: latch motion-halt THEN servo-off backend.

        Sets `_motion_halted` BEFORE forwarding → all subsequent MoveJ/MoveL
        (even mid-Orchestrator cycle) are rejected immediately, no further
        commands sent to robot. No-op forward if backend does not support it (SimRobot).
        """
        self._motion_halted.set()
        if hasattr(self.backend, "Stop") and callable(self.backend.Stop):
            self.backend.Stop()

    # ────────────────────────────────────────────────────────────────────
    # Batch + timer — forward to HSE backend (if supported)
    # ────────────────────────────────────────────────────────────────────
    def batch(self, job_name: str | None = None) -> Any:
        """Context manager: batch motion + IO into one INFORM job (M3 optimization).

        Forwarded to HSE backend if supported, otherwise returns nullcontext (no-op)
        so orchestrator code works with any backend without needing to check.
        """
        if hasattr(self.hse, "batch") and callable(self.hse.batch):
            return self.hse.batch(job_name)
        return contextlib.nullcontext()

    def timer(self, seconds: float) -> None:
        """Pause — INFORM TIMER in batch mode, time.sleep outside batch."""
        if hasattr(self.hse, "timer") and callable(self.hse.timer):
            self.hse.timer(seconds)
        else:
            time.sleep(seconds)

    # ────────────────────────────────────────────────────────────────────
    # Reachability (predictive safety if Orchestrator enables it; else assume OK)
    # ────────────────────────────────────────────────────────────────────
    def MoveJ_Test(self, j_start: Any, target: Any, *args: Any) -> int:
        """No-op reachability check — real kinematic check lives in
        Orchestrator predictive safety (UC2). Returns 0 = assume reachable.
        """
        return 0

    # ────────────────────────────────────────────────────────────────────
    # Motion — Orchestrator resolves IK (client DLS) before calling here
    # Or pass-through 4x4 pose if backend supports YRC IK Cartesian
    # ────────────────────────────────────────────────────────────────────
    def _backend_supports_cartesian(self) -> bool:
        """True if backend can accept a 4x4 pose (HSE BASE Cartesian).

        Checks class-level flag `supports_cartesian_pose=True` instead of
        hasattr — MagicMock auto-creates attrs so hasattr cannot distinguish.
        """
        return getattr(type(self.backend), "supports_cartesian_pose", False) is True

    def _check_not_halted(self) -> None:
        """Raise if Stop()/auto-stop has been triggered — block motion after E-stop/hold."""
        if self._motion_halted.is_set():
            raise RuntimeError(
                "Digital Twin halted (Stop/alarm) — motion command refused")

    def _note_drift_disabled(self) -> None:
        """Log once: in YRC Cartesian mode the controller does its own IK so the
        commanded joints are unknown client-side → drift detection cannot run."""
        if not getattr(self, "_drift_disabled_warned", False):
            self._drift_disabled_warned = True
            logger.warning("Drift detection disabled in YRC Cartesian mode "
                           "(commanded joints unknown client-side).")

    def MoveJ(self, target: Any) -> None:
        self._check_not_halted()
        # Cartesian pass-through for HSE backend when target is a 4x4 pose
        if (isinstance(target, np.ndarray) and target.shape == (4, 4)
                and self._backend_supports_cartesian()):
            with self._lock:
                self._last_commanded = None         # joints not known in advance
            self._note_drift_disabled()
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
            self._note_drift_disabled()
            self.backend.MoveL(target)
            return
        joints = self._target_to_joints(target)
        with self._lock:
            self._last_commanded = list(joints)
        self.backend.MoveL(joints)

    # ── Viewport-visual hooks (Orchestrator calls if robot supports them) ────────
    def attach_object(self, class_name: Any = None) -> None:
        """Orchestrator signals gripper has GRASPED an object → forward so app
        attaches the object to the gripper in the viewport (visual). No-op if callback not set."""
        if self._grasp_callback is not None:
            try:
                self._grasp_callback(class_name)
            except Exception as e:                              # noqa: BLE001
                logger.debug("grasp_callback error: %s", e)

    def detach_object(self) -> None:
        """Orchestrator signals gripper has RELEASED an object → forward so app detaches object."""
        if self._release_callback is not None:
            try:
                self._release_callback()
            except Exception as e:                              # noqa: BLE001
                logger.debug("release_callback error: %s", e)

    def reset_scene(self) -> None:
        """Orchestrator reset at start of each trial → forward so app returns objects to original position."""
        if self._reset_callback is not None:
            try:
                self._reset_callback()
            except Exception as e:                              # noqa: BLE001
                logger.debug("reset_callback error: %s", e)

    def _target_to_joints(self, target: Any) -> list[float]:
        """Convert target joint list → list[float] of 6 floats.

        4x4 pose is NOT handled here — the caller (MoveJ/MoveL) already passes
        the 4x4 pose through to backend if backend supports Cartesian. Orchestrator
        resolves client-side DLS IK before calling with joints.
        """
        if isinstance(target, (list, tuple)) and len(target) >= 6:
            try:
                return [float(j) for j in target[:6]]
            except (TypeError, ValueError) as e:
                raise ValueError(f"target joint values are not float: {e}")
        raise ValueError(
            f"MoveJ target type not supported: {type(target)}. "
            f"Pass a joint list[float] or 4x4 numpy pose (requires "
            f"backend.supports_cartesian_pose=True)."
        )

    @staticmethod
    def _mat_to_joint_list(sol: Any) -> list[float]:
        """Convert IK solution iterable → list[float]."""
        if sol is None:
            raise RuntimeError("IK returned None — pose out of reach")
        if isinstance(sol, (list, tuple)):
            if len(sol) < 6:
                raise RuntimeError(f"IK returned {len(sol)} elements, expected ≥6")
            return [float(j) for j in sol[:6]]
        arr = np.asarray(sol).flatten()
        if arr.size < 6:
            raise RuntimeError(f"IK numpy array has {arr.size} elements")
        return [float(j) for j in arr[:6]]
