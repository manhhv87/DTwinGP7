"""The stage durations each trial row carries for the cycle-time table (E6).

The orchestrator's and the perception node's `time` is replaced by a clock that
advances only where a stage is simulated, so every expected duration is exact and
nothing depends on the platform's clocks: the class selection takes 4 ms (in the
detection window and in localize_ms), the predictive safety check 500 ms (in
plan_ms), the DTWINRUN call 9 s, and the frame reports inference 31 ms and
localization 4 ms. The backend's bracket of the job's end (last_job_timing, timed
in tests/test_job_end_timing.py) puts the end 40 ms before the call returned, with
the last read in the job 55 ms before it and the first after it 60 ms after.
"""
import csv
import queue
import time
import types
from pathlib import Path

import numpy as np
import pytest

from src.logging.logger import TrialLogger
from src.orchestrator import orchestrator as orch_mod
from src.orchestrator.orchestrator import Orchestrator
from src.perception import perception_node as pn_mod
from src.perception.camera import MockCamera
from src.perception.perception_node import PerceptionNode
from tests.test_placement_cycle import _called, _frame, _part
from tests.test_real_cycle import MotomanHSEBackend, _detection, _real_config
from tests.test_real_cycle import TestHome as _Home   # aliased: not collected twice

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("detect_window_ms", "inference_ms", "localize_ms", "plan_ms", "job_ms", "cycle_ms",
          "cycle_pc_ms", "confirm_ms", "job_end_lo_ms", "job_end_hi_ms")
TEXT = ("job_flags", "frame_capture_ms", "frame_arrival_ms", "frame_capture_src")
PICKED = _Home.ARGS
REFUSED = ((572.5, 168.2, 596.9, "inox_box"),        # 227 mm long side for a 180 mm part
           dict(frac=0.74, h_depth=42.6, rect=(227.2, 130.1), yaw_deg=-89.1))
BRACKET = "bracket"


class Clock:
    """Stands in for the time module: perf_counter() and time() read one clock."""

    def __init__(self):
        self.now = 1000.0

    def advance(self, s):
        self.now += s

    def perf_counter(self):
        return self.now

    def time(self):
        return 1.7e9 + self.now

    def __getattr__(self, name):          # monotonic, sleep: the real ones
        return getattr(time, name)


@pytest.fixture
def clock(monkeypatch):
    monkeypatch.chdir(ROOT)               # calibration_path is relative to the repo
    c = Clock()
    monkeypatch.setattr(orch_mod, "time", c)
    monkeypatch.setattr(pn_mod, "time", c)
    return c


def _orch(clock, tmp_path, job_error=None, timing=BRACKET):
    """timing: BRACKET for an end 40 ms before the call returned, a dict for that
    record as given, None for a backend that gives none."""
    config = _real_config("rgbd")
    q = queue.Queue()
    robot = MotomanHSEBackend()
    recorded_job = robot.run_dtwinrun_motion
    record = {}

    def job(**kw):
        clock.advance(9.0)
        end = clock.now - 0.040
        record["timing"] = (dict(lo=end - 0.055, hi=end + 0.060, end=end, flags=[])
                            if timing == BRACKET else timing)
        if job_error is not None:
            raise job_error
        recorded_job(**kw)

    robot.run_dtwinrun_motion = job
    if timing is not None:
        robot.last_job_timing = lambda: record.get("timing")
    orch = Orchestrator(q, config=config, robot=robot,
                        logger_obj=TrialLogger(tmp_path / "run.csv"))
    select, safety = orch._select_objects, orch._predict_safety_for_trajectory
    orch._select_objects = lambda msg: (clock.advance(0.004), select(msg))[1]
    orch._predict_safety_for_trajectory = lambda traj: (clock.advance(0.5), safety(traj))[1]
    return orch, q, config


def _put(orch, q, config, det_args, timed=True, frame=None):
    det = _detection(config, *det_args[0], **det_args[1])
    msg = {"timestamp": 0.0, "objects": [det], "fps": 20.0}
    if timed:
        msg.update(inference_ms=31.0, localize_ms=4.0)
    if frame is not None:
        msg["clock"] = frame
    orch.current_expected_class = det["class_name"]
    q.put(msg)


def _rows(tmp_path):
    with open(tmp_path / "run.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ms(row, keys=STAGES):
    return {k: (float(row[k]) if row[k] != "" else "") for k in keys}


def test_the_stages_of_a_picked_trial_tile_its_cycle(clock, tmp_path):
    orch, q, config = _orch(clock, tmp_path)
    _put(orch, q, config, PICKED)
    assert orch.run_one_cycle(trial_id=1) is True
    (row,) = _rows(tmp_path)
    assert _ms(row) == {
        "detect_window_ms": 4.0, "inference_ms": 31.0, "localize_ms": 8.0, "plan_ms": 500.0,
        "job_ms": 8960.0, "cycle_ms": 9464.0, "cycle_pc_ms": 9504.0, "confirm_ms": 40.0,
        "job_end_lo_ms": 9409.0, "job_end_hi_ms": 9524.0}
    assert row["job_flags"] == "" and row["cycle_start_perf_s"] == "1000.000000"
    # detect + plan + job tile the cycle to the robot's return exactly.
    assert 4.0 + 500.0 + 8960.0 == float(row["cycle_ms"])
    assert float(row["cycle_time_s"]) == 9.504


def test_without_a_bracket_the_job_and_cycle_ends_stay_out(clock, tmp_path):
    orch, q, config = _orch(clock, tmp_path, timing=None)
    _put(orch, q, config, PICKED)
    assert orch.run_one_cycle(trial_id=1) is True
    (row,) = _rows(tmp_path)
    assert row["job_flags"] == "untimed"
    assert [row[k] for k in ("job_ms", "cycle_ms", "confirm_ms", "job_end_lo_ms")] == [""] * 4
    assert float(row["cycle_pc_ms"]) == 9504.0 and float(row["plan_ms"]) == 500.0


def test_a_race_at_a_hold_is_flagged_and_not_timed(clock, tmp_path):
    orch, q, config = _orch(clock, tmp_path, timing=dict(lo=None, hi=None, end=None,
                                                       flags=["held", "race"]))
    _put(orch, q, config, PICKED)
    orch.run_one_cycle(trial_id=1)
    (row,) = _rows(tmp_path)
    assert row["job_flags"] == "held+race" and row["job_ms"] == "" and row["cycle_ms"] == ""


def test_the_job_is_timed_when_it_raises(clock, tmp_path):
    orch, q, config = _orch(clock, tmp_path, job_error=RuntimeError("alarm 4315"))
    _put(orch, q, config, PICKED)
    assert orch.run_one_cycle(trial_id=1) is False
    (row,) = _rows(tmp_path)
    assert row["failure_reason"].startswith("motion_error")
    assert float(row["plan_ms"]) == 500.0 and float(row["job_ms"]) == 8960.0


def test_a_refused_trial_after_a_picked_one_carries_none_of_its_stages(clock, tmp_path):
    orch, q, config = _orch(clock, tmp_path)
    _put(orch, q, config, PICKED, frame={"capture": clock.now - 0.2, "src": "arrival"})
    assert orch.run_one_cycle(trial_id=1) is True
    _put(orch, q, config, REFUSED, timed=False)
    assert orch.run_one_cycle(trial_id=2) is False
    row = _rows(tmp_path)[-1]
    assert row["failure_reason"] == "yaw_geometry_reject"
    assert float(row["detect_window_ms"]) == 4.0 and float(row["cycle_pc_ms"]) == 4.0
    assert [row[k] for k in STAGES if k not in ("detect_window_ms", "cycle_pc_ms")] == [""] * 8
    assert [row[k] for k in TEXT] == [""] * 4


def test_a_run_stopped_at_the_check_carries_no_stage_of_the_trial_before(clock, tmp_path):
    orch, q, config = _orch(clock, tmp_path)
    _put(orch, q, config, PICKED, frame={"capture": clock.now - 0.2, "src": "arrival"})
    assert orch.run_one_cycle(trial_id=1) is True
    orch.current_called_pose = _called()
    orch.current_expected_class = "inox_box"
    q.put(_frame(_part(config, 640.0, 20.0, 41.5)))     # 65 mm off its card

    def operator(trial_id, lines, last):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        orch.run_placement_check(2, operator)
    row = _rows(tmp_path)[-1]
    assert row["failure_reason"] == "aborted_at_check" and row["cycle_start_perf_s"] == ""
    assert {k: row[k] for k in STAGES + TEXT} == {k: "" for k in STAGES + TEXT}


def test_the_operator_verdict_is_left_out_of_the_cycle(clock, tmp_path):
    """cycle_time_s keeps the verdict, as in the rows of earlier runs; cycle_ms and
    cycle_pc_ms do not."""
    orch, q, config = _orch(clock, tmp_path)
    orch.human_score_hook = lambda trial_id, ok: (clock.advance(3.0), 1)[1]
    _put(orch, q, config, PICKED)
    assert orch.run_one_cycle(trial_id=1) is True
    (row,) = _rows(tmp_path)
    assert row["human_ok"] == "1"
    assert float(row["cycle_ms"]) == 9464.0 and float(row["cycle_pc_ms"]) == 9504.0
    assert float(row["cycle_time_s"]) == 12.504


def test_the_frame_capture_rides_with_the_trial(clock, tmp_path):
    orch, q, config = _orch(clock, tmp_path)
    t = clock.now
    _put(orch, q, config, PICKED, frame={"got": t - 0.21, "capture": t - 0.25,
                                         "arrival": t - 0.23, "src": "arrival"})
    assert orch.run_one_cycle(trial_id=1) is True
    (row,) = _rows(tmp_path)
    assert float(row["frame_capture_ms"]) == -250.0
    assert float(row["frame_arrival_ms"]) == -230.0 and row["frame_capture_src"] == "arrival"
    # the frame's age when the detection was taken
    assert float(row["detect_window_ms"]) - float(row["frame_capture_ms"]) == 254.0


class _Det:
    """What field_dict reads off a detection."""
    class_id, class_name, confidence, mask, bbox = 0, "inox_box", 0.9, None, (0, 0, 1, 1)
    pose_camera, pixel_uv, mask_area, height_mm = (0.0, 0.0, 0.5, 0.0), (640, 360), 100, None


def test_the_detector_and_the_extractor_are_timed_apart(clock):
    def detect(rgb):
        clock.advance(0.050)
        return [_Det()]

    def extract(det, depth):
        clock.advance(0.020)
        return dict(det)

    rgb = np.zeros((720, 1280, 3), np.uint8)
    depth = np.ones((720, 1280), np.float32)
    node = PerceptionNode(MockCamera([rgb], [depth]), types.SimpleNamespace(detect=detect),
                          queue.Queue(), extractor=types.SimpleNamespace(extract=extract))
    msg = node.process_once()
    assert len(msg["objects"]) == 1
    assert msg["inference_ms"] == pytest.approx(50.0, abs=1e-6)
    assert msg["localize_ms"] == pytest.approx(20.0, abs=1e-6)


def test_the_node_passes_on_the_frames_camera_instants(clock):
    rgb = np.zeros((720, 1280, 3), np.uint8)
    cam = MockCamera([rgb], [np.ones((720, 1280), np.float32)],
                     clock=clock.perf_counter, latency_s=0.030)
    node = PerceptionNode(cam, types.SimpleNamespace(detect=lambda rgb: []), queue.Queue())
    t0 = clock.now
    assert node.process_once()["clock"] == {"got": t0, "capture": pytest.approx(t0 - 0.030),
                                            "src": "mock"}
