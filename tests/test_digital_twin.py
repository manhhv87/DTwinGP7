"""
test_digital_twin.py
────────────────────
Verify DigitalTwinMirror:
  - Forward calls đến backend (Joints/MoveJ/setDO/Stop)
  - Mirror thread polls Joints + gọi viewport_callback(joints)
  - Telemetry CSV + alarm polling + drift detection
  - Cartesian pose pass-through cho backend.supports_cartesian_pose=True
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.orchestrator.digital_twin import DigitalTwinMirror


@pytest.fixture
def mock_hse():
    """Mock HSE backend trả về fixed joints."""
    hse = MagicMock()
    hse.Joints.return_value = [10.0, -5.0, 20.0, 0.0, 15.0, -10.0]
    hse.JointsHome.return_value = [0.0] * 6
    hse.Valid.return_value = True
    return hse


@pytest.fixture
def viewport_cb():
    """Mock viewport callback — receives joints (degrees) each mirror tick."""
    return MagicMock(name="viewport_callback")


# ─────────────────────────────────────────────────────────────────────────
# Forward to HSE backend
# ─────────────────────────────────────────────────────────────────────────


class TestForwardToHSE:
    def test_joints_forwards_to_hse(self, mock_hse):
        twin = DigitalTwinMirror(mock_hse)
        assert twin.Joints() == [10.0, -5.0, 20.0, 0.0, 15.0, -10.0]
        mock_hse.Joints.assert_called()

    def test_setdo_forwards_to_hse(self, mock_hse):
        twin = DigitalTwinMirror(mock_hse)
        twin.setDO(1, 1)
        mock_hse.setDO.assert_called_once_with(1, 1)

    def test_stop_forwards_to_hse(self, mock_hse):
        twin = DigitalTwinMirror(mock_hse)
        twin.Stop()
        mock_hse.Stop.assert_called_once()

    def test_valid_forwards_to_hse(self, mock_hse):
        twin = DigitalTwinMirror(mock_hse)
        assert twin.Valid() is True


# ─────────────────────────────────────────────────────────────────────────
# Reachability — DigitalTwinMirror.MoveJ_Test is no-op (always 0)
# Kinematic check thực sự nằm ở Orchestrator predictive safety (UC2)
# ─────────────────────────────────────────────────────────────────────────


class TestReachabilityNoOp:
    def test_movej_test_always_returns_zero(self, mock_hse):
        twin = DigitalTwinMirror(mock_hse)
        assert twin.MoveJ_Test([0] * 6, np.eye(4)) == 0


# ─────────────────────────────────────────────────────────────────────────
# MoveJ — joints list pass-through; pose 4x4 nếu backend supports Cartesian
# ─────────────────────────────────────────────────────────────────────────


class TestMoveJ:
    def test_movej_with_joints_list_calls_hse_directly(self, mock_hse):
        twin = DigitalTwinMirror(mock_hse)
        twin.MoveJ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        mock_hse.MoveJ.assert_called_once_with([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_movej_with_4x4_pose_passes_through_when_backend_supports(self):
        """Backend với supports_cartesian_pose=True → pose 4x4 pass-through."""
        class CartBackend:
            supports_cartesian_pose = True
            def __init__(self):
                self.Joints = MagicMock(return_value=[0]*6)
                self.MoveJ = MagicMock()
                self.JointsHome = MagicMock(return_value=[0]*6)
                self.Valid = MagicMock(return_value=True)
        be = CartBackend()
        twin = DigitalTwinMirror(be)
        pose = np.eye(4)
        twin.MoveJ(pose)
        be.MoveJ.assert_called_once()
        # Backend nhận pose 4x4 trực tiếp (Cartesian path)
        called = be.MoveJ.call_args[0][0]
        assert isinstance(called, np.ndarray)
        assert called.shape == (4, 4)

    def test_movej_with_pose_no_cartesian_support_raises(self, mock_hse):
        """Backend KHÔNG support Cartesian → MoveJ(pose 4x4) raise."""
        twin = DigitalTwinMirror(mock_hse)
        with pytest.raises(ValueError, match="target type not supported"):
            twin.MoveJ(np.eye(4))


# ─────────────────────────────────────────────────────────────────────────
# Mirror thread — polls Joints + setJoints viewport
# ─────────────────────────────────────────────────────────────────────────


class TestMirrorThread:
    def test_mirror_thread_polls_hse_joints(self, mock_hse, viewport_cb):
        twin = DigitalTwinMirror(mock_hse, viewport_callback=viewport_cb, telemetry_hz=50.0, mirror_hz=50.0)
        twin.start_mirror()
        time.sleep(0.2)                                 # ~10 ticks at 50Hz
        twin.stop_mirror()
        assert mock_hse.Joints.call_count >= 5

    def test_mirror_thread_calls_viewport_callback(self, mock_hse, viewport_cb):
        twin = DigitalTwinMirror(mock_hse, viewport_callback=viewport_cb, telemetry_hz=50.0, mirror_hz=50.0)
        twin.start_mirror()
        time.sleep(0.15)
        twin.stop_mirror()
        # viewport_callback được gọi với joints từ backend
        assert viewport_cb.call_count >= 3
        args = viewport_cb.call_args[0]
        assert args[0] == [10.0, -5.0, 20.0, 0.0, 15.0, -10.0]

    def test_mirror_thread_idempotent_start(self, mock_hse, viewport_cb):
        twin = DigitalTwinMirror(mock_hse, viewport_callback=viewport_cb, telemetry_hz=20.0, mirror_hz=20.0)
        twin.start_mirror()
        twin.start_mirror()                             # 2nd call: no-op
        assert twin._mirror_thread is not None
        twin.stop_mirror()

    def test_mirror_thread_survives_hse_error(self, mock_hse, viewport_cb):
        """Lỗi sporadic không kill thread — log warning + tiếp tục tick."""
        side_effects = [
            RuntimeError("transient"),
            [10.0] * 6,
            [10.0] * 6,
        ]
        mock_hse.Joints.side_effect = side_effects + [[10.0] * 6] * 100
        twin = DigitalTwinMirror(mock_hse, viewport_callback=viewport_cb, telemetry_hz=100.0, mirror_hz=100.0)
        twin.start_mirror()
        time.sleep(0.2)
        twin.stop_mirror()
        # Đã có nhiều call sau lỗi đầu — thread vẫn sống
        assert mock_hse.Joints.call_count >= 5

    def test_mirror_thread_logs_telemetry(self, mock_hse, viewport_cb):
        telemetry = MagicMock()
        twin = DigitalTwinMirror(
            mock_hse, viewport_callback=viewport_cb, telemetry=telemetry,
            telemetry_hz=50.0, mirror_hz=50.0,
        )
        twin.start_mirror()
        time.sleep(0.15)
        twin.stop_mirror()
        telemetry.open.assert_called_once()
        telemetry.close.assert_called_once()
        assert telemetry.log_state.call_count >= 3

    def test_stop_mirror_joins_thread(self, mock_hse, viewport_cb):
        twin = DigitalTwinMirror(mock_hse, viewport_callback=viewport_cb, telemetry_hz=20.0, mirror_hz=20.0)
        twin.start_mirror()
        thread_ref = twin._mirror_thread
        assert thread_ref.is_alive()
        twin.stop_mirror()
        assert not thread_ref.is_alive()


# ─────────────────────────────────────────────────────────────────────────
# Drift detection
# ─────────────────────────────────────────────────────────────────────────


class TestViewportMirrorDisable:
    """Flag viewport_mirror_enabled=False → mirror loop vẫn chạy nhưng skip callback."""

    def test_viewport_disabled_skips_callback(self, mock_hse, viewport_cb):
        twin = DigitalTwinMirror(
            mock_hse, viewport_callback=viewport_cb, telemetry_hz=50.0, mirror_hz=50.0,
            viewport_mirror_enabled=False,
        )
        twin.start_mirror()
        time.sleep(0.2)
        twin.stop_mirror()
        # mirror loop chạy (telemetry + drift) nhưng viewport_cb KHÔNG được gọi
        assert mock_hse.Joints.call_count >= 3
        viewport_cb.assert_not_called()

    def test_viewport_enabled_default_calls_callback(self, mock_hse, viewport_cb):
        twin = DigitalTwinMirror(mock_hse, viewport_callback=viewport_cb, telemetry_hz=50.0, mirror_hz=50.0)
        # Default viewport_mirror_enabled=True
        twin.start_mirror()
        time.sleep(0.2)
        twin.stop_mirror()
        assert viewport_cb.call_count >= 3

    def test_default_mirror_hz_low_enough_to_be_smooth(self):
        """Default 2Hz đủ smooth cho mắt người."""
        from src.orchestrator.digital_twin import DEFAULT_MIRROR_HZ
        assert DEFAULT_MIRROR_HZ <= 5.0


class TestDecoupledRates:
    """Telemetry rate (cao) decouple khỏi viewport rate (thấp)."""

    def test_viewport_throttled_below_telemetry(self, mock_hse, viewport_cb):
        # Telemetry 50Hz, viewport 10Hz → throttle 5:1
        twin = DigitalTwinMirror(
            mock_hse, viewport_callback=viewport_cb, telemetry_hz=50.0, mirror_hz=10.0,
        )
        twin.start_mirror()
        time.sleep(0.3)                                   # ~15 ticks @ 50Hz
        twin.stop_mirror()

        joints_calls = mock_hse.Joints.call_count
        set_calls = viewport_cb.call_count
        # Joints (telemetry) ≈ 15. setJoints (viewport) ≈ 15/5 = 3
        assert joints_calls >= 5
        assert set_calls < joints_calls                   # viewport throttled
        # Tỉ lệ throttle phải gần 5:1 (cho phép ±50% jitter)
        ratio = joints_calls / max(set_calls, 1)
        assert 3 <= ratio <= 10

    def test_telemetry_resolution_high(self, mock_hse, viewport_cb):
        """Telemetry CSV ghi mỗi tick @telemetry_hz, không bị throttle."""
        from unittest.mock import MagicMock
        telemetry = MagicMock()
        twin = DigitalTwinMirror(
            mock_hse, viewport_callback=viewport_cb, telemetry=telemetry,
            telemetry_hz=50.0, mirror_hz=2.0,             # viewport throttle 25:1
        )
        twin.start_mirror()
        time.sleep(0.3)
        twin.stop_mirror()
        # Telemetry log call_count ≈ Joints call_count (mỗi tick, không throttle)
        joints_n = mock_hse.Joints.call_count
        log_n = telemetry.log_state.call_count
        assert log_n >= joints_n - 1                       # cho phép ±1 do timing

    def test_mirror_hz_clamped_to_telemetry_hz(self, mock_hse, viewport_cb):
        """mirror_hz không vượt telemetry_hz (vô lý)."""
        twin = DigitalTwinMirror(
            mock_hse, viewport_callback=viewport_cb,
            telemetry_hz=10.0, mirror_hz=50.0,            # 50 > 10 → clamp
        )
        assert twin.mirror_hz <= twin.telemetry_hz

    def test_alarm_poll_period_independent_of_loop_rate(self, mock_hse, viewport_cb):
        """Alarm poll period là TIME-based, không tick-based → giữ rate khi loop đổi."""
        from unittest.mock import MagicMock
        mock_hse.read_alarm = MagicMock(return_value=(0, 0))

        twin = DigitalTwinMirror(
            mock_hse, viewport_callback=viewport_cb,
            telemetry_hz=100.0,                            # rất cao
            alarm_poll_period_s=0.1,                       # 10/sec
        )
        twin.start_mirror()
        time.sleep(0.35)                                   # ~3 alarm polls expected
        twin.stop_mirror()
        # ~3-4 alarm polls (so với joint polls ~35) — verify alarm dùng time-based.
        # Trừ 1 lần poll ĐỒNG BỘ lúc start_mirror (refresh alarm state, #7).
        alarm_calls = mock_hse.read_alarm.call_count - 1
        joints_calls = mock_hse.Joints.call_count
        assert alarm_calls < joints_calls / 5             # alarm thưa hơn joint nhiều


class TestAlarmAutoPoll:
    """Mirror thread poll read_alarm() định kỳ + log/stop theo severity."""

    @pytest.fixture
    def mock_hse_with_alarm(self, mock_hse):
        """HSE mock có read_alarm method."""
        mock_hse.read_alarm = MagicMock(return_value=(0, 0))    # default no alarm
        return mock_hse

    def test_alarm_polled_periodically(self, mock_hse_with_alarm, viewport_cb):
        # Telemetry 50Hz, alarm mỗi 0.1s → ~3 alarm polls trong 0.3s
        twin = DigitalTwinMirror(
            mock_hse_with_alarm, viewport_callback=viewport_cb,
            telemetry_hz=50.0, alarm_poll_period_s=0.1,
        )
        twin.start_mirror()
        time.sleep(0.3)
        twin.stop_mirror()
        assert mock_hse_with_alarm.read_alarm.call_count >= 1
        # Alarm thưa hơn joints nhiều (period-based, không spam). Trừ 1 lần poll
        # đồng bộ lúc start_mirror (#7) trước khi so tỉ lệ.
        periodic_alarms = mock_hse_with_alarm.read_alarm.call_count - 1
        assert periodic_alarms <= mock_hse_with_alarm.Joints.call_count // 3 + 1

    def test_major_alarm_triggers_auto_stop(
        self, mock_hse_with_alarm, viewport_cb, caplog
    ):
        # 2010 = EMERGENCY_STOP (MAJOR)
        mock_hse_with_alarm.read_alarm.return_value = (2010, 0)

        import logging
        caplog.set_level(logging.ERROR, logger="src.orchestrator.digital_twin")

        twin = DigitalTwinMirror(
            mock_hse_with_alarm, viewport_callback=viewport_cb, telemetry_hz=50.0, mirror_hz=50.0,
            alarm_poll_period_s=0.001, auto_stop_on_major_alarm=True,
        )
        twin.start_mirror()
        time.sleep(0.15)
        twin.stop_mirror()

        mock_hse_with_alarm.Stop.assert_called()
        assert twin.is_alarm_active()
        assert twin.current_alarm().code == 2010

    def test_minor_alarm_no_auto_stop(self, mock_hse_with_alarm, viewport_cb):
        # 1010 = REMOTE_MODE_REQUIRED (MINOR — không auto-stop)
        mock_hse_with_alarm.read_alarm.return_value = (1010, 0)
        twin = DigitalTwinMirror(
            mock_hse_with_alarm, viewport_callback=viewport_cb, telemetry_hz=50.0, mirror_hz=50.0,
            alarm_poll_period_s=0.001, auto_stop_on_major_alarm=True,
        )
        twin.start_mirror()
        time.sleep(0.15)
        twin.stop_mirror()
        mock_hse_with_alarm.Stop.assert_not_called()
        assert twin.current_alarm().code == 1010

    def test_auto_stop_disabled_respects_config(
        self, mock_hse_with_alarm, viewport_cb
    ):
        mock_hse_with_alarm.read_alarm.return_value = (2010, 0)        # MAJOR
        twin = DigitalTwinMirror(
            mock_hse_with_alarm, viewport_callback=viewport_cb, telemetry_hz=50.0, mirror_hz=50.0,
            alarm_poll_period_s=0.001, auto_stop_on_major_alarm=False,
        )
        twin.start_mirror()
        time.sleep(0.15)
        twin.stop_mirror()
        mock_hse_with_alarm.Stop.assert_not_called()

    def test_alarm_clear_resets_state(self, mock_hse_with_alarm, viewport_cb):
        # Trình tự: alarm → clear → alarm again
        alarm_sequence = [(2010, 0), (2010, 0), (0, 0), (0, 0), (2010, 0)]
        mock_hse_with_alarm.read_alarm.side_effect = alarm_sequence * 20
        twin = DigitalTwinMirror(
            mock_hse_with_alarm, viewport_callback=viewport_cb, telemetry_hz=200.0, mirror_hz=200.0,
            alarm_poll_period_s=0.001,
        )
        twin.start_mirror()
        time.sleep(0.1)                                 # ~20 tick
        twin.stop_mirror()
        # Stop() được gọi ≥ 2 lần (mỗi lần alarm xuất hiện sau clear)
        assert mock_hse_with_alarm.Stop.call_count >= 1

    def test_no_alarm_method_on_backend_noop(self, mock_hse, viewport_cb):
        """Backend không có read_alarm (vd SimRobot) → mirror thread vẫn chạy."""
        # mock_hse fixture không có read_alarm method by default
        twin = DigitalTwinMirror(
            mock_hse, viewport_callback=viewport_cb,
            telemetry_hz=50.0, alarm_poll_period_s=0.001,
        )
        twin.start_mirror()
        time.sleep(0.1)
        twin.stop_mirror()
        assert twin.current_alarm().code == 0           # mặc định = no alarm


class TestBatchAndTimerForwarding:
    def test_batch_forwarded_to_hse(self, mock_hse, viewport_cb):
        from contextlib import contextmanager
        called_with = {}

        @contextmanager
        def fake_batch(job_name=None):
            called_with["name"] = job_name
            yield

        mock_hse.batch = fake_batch
        twin = DigitalTwinMirror(mock_hse, viewport_callback=viewport_cb)
        with twin.batch("MYTRIAL"):
            pass
        assert called_with["name"] == "MYTRIAL"

    def test_batch_nullcontext_when_hse_lacks_batch(self, mock_hse, viewport_cb):
        del mock_hse.batch
        twin = DigitalTwinMirror(mock_hse, viewport_callback=viewport_cb)
        with twin.batch():                                    # không raise
            pass

    def test_timer_forwarded_to_hse(self, mock_hse, viewport_cb):
        twin = DigitalTwinMirror(mock_hse, viewport_callback=viewport_cb)
        twin.timer(0.5)
        mock_hse.timer.assert_called_once_with(0.5)

    def test_timer_falls_back_to_sleep(self, mock_hse, viewport_cb, monkeypatch):
        del mock_hse.timer
        sleeps = []
        monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
        twin = DigitalTwinMirror(mock_hse, viewport_callback=viewport_cb)
        twin.timer(0.4)
        assert sleeps == [0.4]


class TestSimBackendCompat:
    """DigitalTwinMirror phải làm việc với SimRobot (không phải HSE) — vì sim
    headless mode dùng SimRobot làm motion backend qua DigitalTwinMirror façade."""

    def test_simrobot_backend_basic_forwarding(self):
        from src.orchestrator.sim_robot import SimRobot
        sim = SimRobot(home_joints=[1, 2, 3, 4, 5, 6])
        twin = DigitalTwinMirror(sim)
        assert twin.Valid() is True
        assert twin.Joints() == [1, 2, 3, 4, 5, 6]
        assert twin.JointsHome() == [1, 2, 3, 4, 5, 6]

    def test_simrobot_stop_does_not_crash(self):
        """SimRobot.Stop() no-op — DigitalTwinMirror.Stop() phải safe."""
        from src.orchestrator.sim_robot import SimRobot
        sim = SimRobot()
        twin = DigitalTwinMirror(sim)
        twin.Stop()                                    # không raise

    def test_backend_without_stop_is_safe(self):
        """Backend không có Stop() (vd custom) → DigitalTwinMirror.Stop() no-op."""
        bare = MagicMock(spec=["Joints", "JointsHome", "Valid", "MoveJ", "setDO"])
        bare.Joints.return_value = [0] * 6
        twin = DigitalTwinMirror(bare)
        twin.Stop()                                    # không raise

    def test_simrobot_disconnect_is_safe(self):
        """SimRobot.disconnect() no-op — script cleanup không break."""
        from src.orchestrator.sim_robot import SimRobot
        sim = SimRobot()
        twin = DigitalTwinMirror(sim)
        twin.backend.disconnect()                      # no-op


class TestDriftDetection:
    def test_drift_warning_logged_when_exceeds_threshold(
        self, mock_hse, viewport_cb, caplog
    ):
        # HSE trả actual = 0, commanded sẽ là 5 → drift 5° > threshold 2°
        mock_hse.Joints.return_value = [0.0] * 6
        twin = DigitalTwinMirror(
            mock_hse, viewport_callback=viewport_cb, telemetry_hz=50.0, mirror_hz=50.0, drift_warn_deg=2.0,
        )
        # Inject commanded state thông qua MoveJ
        twin.MoveJ([5.0] * 6)

        import logging
        caplog.set_level(logging.WARNING, logger="src.orchestrator.digital_twin")
        twin.start_mirror()
        time.sleep(0.15)
        twin.stop_mirror()

        # Có ít nhất 1 warning Drift
        drift_logs = [r for r in caplog.records if "drift" in r.message]
        assert len(drift_logs) >= 1

    def test_no_drift_warning_when_within_threshold(self, mock_hse, viewport_cb, caplog):
        mock_hse.Joints.return_value = [5.5] * 6           # 0.5° lệch khỏi 5.0
        twin = DigitalTwinMirror(
            mock_hse, viewport_callback=viewport_cb, telemetry_hz=50.0, mirror_hz=50.0, drift_warn_deg=2.0,
        )
        twin.MoveJ([5.0] * 6)

        import logging
        caplog.set_level(logging.WARNING, logger="src.orchestrator.digital_twin")
        twin.start_mirror()
        time.sleep(0.15)
        twin.stop_mirror()

        drift_logs = [r for r in caplog.records if "drift" in r.message]
        assert len(drift_logs) == 0
