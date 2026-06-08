"""
test_live_jog.py
────────────────
Phase-2 streaming live jog → REAL robot: unit-test the non-GUI / non-hardware
logic of ConnectionMixin — the coalescing tap (`_stream_live_jog`), the motion
guard (`_robot_motion_active`), and the abnormal-exit slot. The HSE worker
itself (threads + sockets) is NOT exercised here; only the wiring that decides
WHAT gets streamed and WHEN, which is what correctness hinges on.
"""
from __future__ import annotations

import threading
import types

from src.orchestrator.viewports.mixin_connection import ConnectionMixin

stream = ConnectionMixin._stream_live_jog
motion_active = ConnectionMixin._robot_motion_active
worker_exit = ConnectionMixin._on_live_jog_worker_exit


class _FakeAction:
    def __init__(self, checked=True):
        self._checked = checked
    def isChecked(self):
        return self._checked
    def setChecked(self, v):
        self._checked = v
    def blockSignals(self, _):
        pass


class _FakeThread:
    def __init__(self, alive=True):
        self._alive = alive
    def is_alive(self):
        return self._alive


def _stub(checked=True, alive=True, joints=(1, 2, 3, 4, 5, 6)):
    return types.SimpleNamespace(
        _act_live_jog=_FakeAction(checked),
        _live_jog_thread=_FakeThread(alive),
        _live_jog_lock=threading.Lock(),
        _live_jog_target=None,
        _live_jog_dirty=False,
        _joints=list(joints),
        _hse_thread=None,
        _exp_running=False,
        _status=[],
    )


def test_stream_pushes_latest_when_on():
    s = _stub()
    stream(s)
    assert s._live_jog_dirty is True
    assert s._live_jog_target == [1, 2, 3, 4, 5, 6]
    # target is a COPY — later mutation of _joints must not bleed through
    s._joints[0] = 99
    assert s._live_jog_target[0] == 1


def test_stream_coalesces_to_newest():
    s = _stub()
    stream(s)                       # first jog
    s._joints = [10, 0, 0, 0, 0, 0]
    stream(s)                       # second jog before worker consumed the first
    # latest-value-wins: only the newest target is held
    assert s._live_jog_target == [10, 0, 0, 0, 0, 0]
    assert s._live_jog_dirty is True


def test_stream_noop_when_toggle_off():
    s = _stub(checked=False)
    stream(s)
    assert s._live_jog_target is None
    assert s._live_jog_dirty is False


def test_stream_noop_when_worker_dead():
    s = _stub(alive=False)
    stream(s)
    assert s._live_jog_target is None
    assert s._live_jog_dirty is False


def test_motion_active_counts_live_jog():
    s = _stub(alive=True)
    assert motion_active(s) is True          # live-jog thread alive
    s._live_jog_thread = _FakeThread(alive=False)
    assert motion_active(s) is False         # nothing running
    s._exp_running = True
    assert motion_active(s) is True          # experiment/mirror running


def test_worker_exit_unchecks_toggle():
    s = _stub(checked=True)
    s._set_status = lambda msg, level="info": s._status.append((msg, level))
    worker_exit(s, "Robot: ALARM 0x1234 — servo OFF")
    assert s._act_live_jog.isChecked() is False
    assert s._status and s._status[-1][1] == "err"


# ── enable guards auto-untick (the "toggle tự tắt" behaviour) ────────────────
toggle = ConnectionMixin._on_toggle_live_jog


def _enable_stub(**kw):
    s = _stub(checked=True)             # user just ticked it ON
    s._set_status = lambda msg, level="info": s._status.append((msg, level))
    s._hse_ip = kw.get("ip", "192.168.125.100")
    s._model = kw.get("model", object())
    s._robot_motion_active = lambda: kw.get("busy", False)
    s._live_jog_thread = None
    return s


def test_toggle_unchecks_when_no_ip():
    s = _enable_stub(ip="")            # no IP -> refuse before any dialog
    toggle(s, True)
    assert s._act_live_jog.isChecked() is False

def test_toggle_unchecks_when_no_model():
    s = _enable_stub(model=None)
    s._model = None
    toggle(s, True)
    assert s._act_live_jog.isChecked() is False

def test_toggle_unchecks_when_busy():
    s = _enable_stub(busy=True)        # Run/Mirror/experiment active
    toggle(s, True)
    assert s._act_live_jog.isChecked() is False

def test_toggle_off_signals_stop_only_when_alive():
    # Unticking with a live worker sets the stop event (worker servo-offs itself).
    s = _stub(checked=False)
    s._set_status = lambda msg, level="info": s._status.append((msg, level))
    import threading as _th
    s._live_jog_stop = _th.Event()
    s._live_jog_thread = _FakeThread(alive=True)
    toggle(s, False)
    assert s._live_jog_stop.is_set() is True


# ── teleport guard (_live_jog_max_step) ─────────────────────────────────────
max_step = ConnectionMixin._live_jog_max_step
LIMIT = ConnectionMixin._LIVE_JOG_MAX_STEP_DEG


def test_max_step_is_largest_per_joint_delta():
    a = [0, 0, 0, 0, 0, 0]
    b = [1, -2, 3, 0, -5, 4]
    assert max_step(b, a) == 5            # |−5| is the biggest per-joint move


def test_small_jog_passes_guard():
    last = [10, 20, 30, 0, 90, 0]
    target = [12, 19, 33, 0, 90, 1]      # a few deg per joint
    assert max_step(target, last) <= LIMIT


def test_teleport_blocked_by_guard():
    last = [0, 0, 0, 0, 0, 0]
    target = [0, 0, 0, 0, 0, 0]
    target[2] = LIMIT + 20               # paste/Home jump on one joint
    assert max_step(target, last) > LIMIT


def test_limit_is_sane():
    assert 5.0 <= LIMIT <= 90.0          # generous for jog, blocks teleports


# ── motion-source guards (review batch-1 fixes) ─────────────────────────────
def test_motion_active_counts_send_pose_busy():
    s = _stub(alive=False)               # no live-jog thread
    s._send_pose_busy = False
    assert motion_active(s) is False
    s._send_pose_busy = True             # discrete Send-pose in progress
    assert motion_active(s) is True


def test_validate_dt_common_blocks_when_run_on_robot_active():
    # HIGH-1: experiment/mirror validator must refuse while a Run-on-Robot job runs.
    from src.orchestrator.viewports.mixin_experiment import ExperimentMixin
    validate = ExperimentMixin._validate_dt_common
    s = types.SimpleNamespace(
        _exp_running=False,
        _hse_thread=_FakeThread(alive=True),     # Run-on-Robot worker alive
        _live_jog_thread=None,
        _model=object(), _hse_ip="192.168.125.100",
    )
    err = validate(s)
    assert err is not None and "Run-on-Robot" in err

    s._hse_thread = _FakeThread(alive=False)      # nothing running → passes
    assert validate(s) is None
