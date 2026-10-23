"""The placement check before the trial, and the trial's own cycle after it.

The orchestrator is configured as 03_run_experiment configures it for --mode real
(tests/test_real_cycle.py) and the controller is a recorder. run_placement_check is
called as 03's pre-trial hook calls it once the operator has pressed ENTER, with an
operator that presses ENTER at every flagged check; then a fresh frame reaches the
queue, as after the motion gate, and run_one_cycle runs the trial. What must hold:

  * no flag, no question;
  * a flagged placement is checked again after each answer, MAX_CHECKS checks at
    most; after the last the trial runs as placed;
  * no check capture is used for the trial: the trial detects afresh, so a
    perception failure at the trial stays a failure, and a cycle with no check
    before it is the cycle of session 1;
  * the flags are the same in every depth mode;
  * every check, the detection it judged and the time land in that trial's row and
    no other; a run stopped at a check prompt writes aborted_at_check.

The operator's side (prompts, servo, stale frames) is tested in test_operator_hooks.

Run:
    pytest tests/test_placement_cycle.py -v
"""
from __future__ import annotations

import csv
import json
import math
import queue
import time

import numpy as np
import pytest

from src.logging.logger import CONTEXT_FIELDNAMES, TrialLogger
from src.orchestrator import orchestrator as orch_mod
from src.orchestrator.coord_conv import load_calibration
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.placement_check import CalledPose
from tests.test_real_cycle import (
    ROOT, MotomanHSEBackend, _detection, _plane_z, _real_config, _world_z,
)

PLACEMENT_COLUMNS = ("placement_flags", "placement_recaptures", "placement_override",
                     "placement_events", "placement_captures", "placement_check_s")


class Operator:
    """Presses ENTER at every flagged check and records the call (trial, lines,
    last). Each answer lets the next scripted frame reach the queue: the camera's
    first frame once the hand has left the cell."""

    def __init__(self, q, *frames, sleep_s=0.0):
        self.q = q
        self.frames = list(frames)
        self.sleep_s = sleep_s
        self.asked = []

    def __call__(self, trial_id, lines, last):
        self.asked.append((trial_id, list(lines), last))
        time.sleep(self.sleep_s)
        if self.frames:
            self.q.put(self.frames.pop(0))


def _called(cls="inox_box", x=575.0, y=20.0, yaw=40.0, stack_on="", carton_mm=None,
            stack_on_carton_mm=None):
    return CalledPose(pose_id="P0100", card_id="11", x_mm=x, y_mm=y, yaw_deg=yaw,
                      class_name=cls, stack_on=stack_on, carton_mm=carton_mm,
                      stack_on_carton_mm=stack_on_carton_mm)


def _part(config, x, y, top_mm, cls="inox_box", yaw_off=0.0, called_yaw=40.0,
          depth_used="rgbd", point_mm=None, rect=(188.0, 90.0)):
    """A detection of a part whose top-face centre is at (x, y), top_mm above the
    table, read at the called yaw plus yaw_off. Its commanded point lies on the
    viewing ray through that centre, point_mm above the table (default: the top),
    as each depth mode places it."""
    T = load_calibration(ROOT / config["calibration_path"])
    top = np.array([x, y, _plane_z(config, x, y) + top_mm])
    point = top
    if point_mm is not None:
        cam = T[:3, 3]
        d = top - cam
        s = (_plane_z(config, x, y) + point_mm - cam[2]) / d[2]
        point = cam + s * d
    det = _detection(config, *point, cls, depth_used=depth_used, frac=0.9,
                     h_depth=top_mm, rect=rect, yaw_deg=called_yaw - 90.0 + yaw_off)
    det["yaw_top_height_mm"] = top_mm
    return det


def _frame(*objects, path=""):
    return {"timestamp": 0.0, "objects": list(objects), "fps": 20.0, "frame_path": path}


def _orch(config, called, tmp_path, logger=True):
    robot = MotomanHSEBackend()
    q = queue.Queue()
    trial_logger = TrialLogger(tmp_path / "run.csv") if logger else None
    orch = Orchestrator(q, config=config, robot=robot, logger_obj=trial_logger)
    orch.current_expected_class = called.class_name if called else ""
    orch.current_called_pose = called
    return orch, robot, q


def _trial(orch, q, op, trial_frame, trial_id=1):
    """03's order: the check (servo OFF), then after the motion gate a fresh frame
    and the cycle. trial_frame None: no frame reaches the trial."""
    orch.run_placement_check(trial_id, op)
    if trial_frame is not None:
        q.put(trial_frame)
    return orch.run_one_cycle(trial_id=trial_id)


def _row(tmp_path, name="run.csv"):
    with (tmp_path / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))[-1]


def _captures(row):
    return json.loads(row["placement_captures"])


def test_the_new_columns_are_in_the_schema():
    for col in ("carton_mm", "stack_on_carton_mm", "top_x_mm", "top_y_mm",
                *PLACEMENT_COLUMNS):
        assert col in CONTEXT_FIELDNAMES


def test_a_clean_placement_asks_nothing_and_the_trial_detects_afresh(tmp_path):
    config = _real_config("rgbd")
    orch, robot, q = _orch(config, _called(), tmp_path)
    q.put(_frame(_part(config, 578.0, 16.0, 41.5, yaw_off=3.0), path="check1.png"))
    op = Operator(q)
    trial = _frame(_part(config, 572.0, 24.0, 41.5), path="trial.png")
    assert _trial(orch, q, op, trial) is True
    assert op.asked == [] and len(robot.sent) == 1
    # The trial acted on its own capture, not on the check's.
    assert float(robot.sent[0]["pick"][0, 3]) == pytest.approx(572.0, abs=1.0)
    row = _row(tmp_path)
    assert row["frame_path"] == "trial.png"
    assert float(row["top_x_mm"]) == pytest.approx(572.0, abs=0.5)
    assert row["placement_flags"] == "" and row["placement_events"] == "none:clear"
    assert row["placement_recaptures"] == "0" and row["placement_override"] == "0"
    (cap,) = _captures(row)
    assert cap["outcome"] == "clear" and cap["flags"] == [] and "answer_s" not in cap
    assert cap["frame"] == "check1.png"
    assert cap["top_x_mm"] == pytest.approx(578.0, abs=0.5)
    assert float(row["placement_check_s"]) < 0.5


def test_a_flag_then_a_fix_asks_once_and_checks_again(tmp_path):
    config = _real_config("rgbd")
    orch, robot, q = _orch(config, _called(stack_on="metal_box"), tmp_path)
    q.put(_frame(_part(config, 575.0, 20.0, 41.5), path="check1.png"))
    stacked = _part(config, 577.0, 22.0, 41.5 + 84.7, rect=(193.0, 130.2))
    op = Operator(q, _frame(stacked, path="check2.png"), sleep_s=0.3)
    orch.run_placement_check(2, op)
    q.put(_frame(stacked, path="trial.png"))
    t0 = time.time()
    assert orch.run_one_cycle(trial_id=2) is True
    cycle_s = time.time() - t0
    assert len(op.asked) == 1
    trial_id, lines, last = op.asked[0]
    assert trial_id == 2 and last is False and lines[0].startswith("STACK:")
    row = _row(tmp_path)
    assert row["placement_flags"] == "stack"
    assert row["placement_events"] == "stack:fixed;none:clear"
    assert row["placement_recaptures"] == "1" and row["placement_override"] == "0"
    assert [c["outcome"] for c in _captures(row)] == ["fixed", "clear"]
    # The trial ran on the part on its support: the pick stays above the metal box.
    assert float(row["h_depth_mm"]) == pytest.approx(126.2, abs=0.1)
    assert _world_z(robot.sent[0]["pick"]) - _plane_z(config, 577.0, 22.0) > 84.7
    # The operator's time is the check's; the cycle time starts after the checks.
    assert float(row["placement_check_s"]) >= 0.25
    assert float(row["cycle_time_s"]) <= cycle_s + 0.01


def test_three_flags_then_the_trial_runs_on_its_own_capture(tmp_path):
    config = _real_config("rgbd")
    orch, robot, q = _orch(config, _called(), tmp_path)
    checks = [_frame(_part(config, 575.0, 80.0 + k, 41.5), path=f"check{k + 1}.png")
              for k in range(3)]
    q.put(checks[0])
    op = Operator(q, *checks[1:])
    # Set right at the last prompt: the trial's own capture sees the part on its card.
    trial = _frame(_part(config, 575.0, 20.0, 41.5), path="trial.png")
    assert _trial(orch, q, op, trial) is True
    assert orch_mod.MAX_CHECKS == 3
    assert [last for _, _, last in op.asked] == [False, False, True]
    assert [lines[0][:31] for _, lines, _ in op.asked] == [
        "POSITION: part centre 60 mm fro", "POSITION: part centre 61 mm fro",
        "POSITION: part centre 62 mm fro"]
    row = _row(tmp_path)
    assert row["placement_events"] == "position:fixed;position:fixed;position:limit"
    assert row["placement_recaptures"] == "2" and row["placement_override"] == "1"
    assert [c["outcome"] for c in _captures(row)] == ["fixed", "fixed", "limit"]
    assert [c["frame"] for c in _captures(row)] == ["check1.png", "check2.png",
                                                    "check3.png"]
    assert row["frame_path"] == "trial.png"
    assert float(robot.sent[0]["pick"][1, 3]) == pytest.approx(20.0, abs=1.0)
    assert q.empty()


def test_the_limit_follows_the_constant(tmp_path, monkeypatch):
    monkeypatch.setattr(orch_mod, "MAX_CHECKS", 1)
    config = _real_config("rgbd")
    orch, robot, q = _orch(config, _called(), tmp_path)
    q.put(_frame(_part(config, 575.0, 90.0, 41.5)))
    op = Operator(q)
    orch.run_placement_check(1, op)
    assert [last for _, _, last in op.asked] == [True]
    assert orch._pending_placement["placement_events"] == "position:limit"


def test_the_wrong_class_is_flagged_and_the_trial_detects_afresh(tmp_path):
    config = _real_config("rgbd")
    config["detection_timeout_s"] = 0.2
    orch, robot, q = _orch(config, _called(cls="carton", carton_mm=120), tmp_path)
    inox = _part(config, 575.0, 20.0, 41.5)
    q.put(_frame(inox))
    op = Operator(q, _frame(inox), _frame(inox))
    assert _trial(orch, q, op, _frame(inox), trial_id=17) is False
    assert robot.sent == []
    assert op.asked[0][1] == ["CLASS: no carton(120) found; check the part on card 11"]
    row = _row(tmp_path)
    assert row["failure_reason"] == "detection_miss" and row["class_name"] == ""
    assert row["placement_flags"] == "class" and row["placement_override"] == "1"
    assert row["carton_mm"] == "120"


def test_a_check_capture_is_never_used_for_the_trial(tmp_path):
    config = _real_config("rgbd")
    config["detection_timeout_s"] = 0.1
    orch, robot, q = _orch(config, _called(), tmp_path)
    q.put(_frame(_part(config, 575.0, 20.0, 41.5)))
    assert _trial(orch, q, Operator(q), None) is False
    assert robot.sent == []
    row = _row(tmp_path)
    assert row["failure_reason"] == "detection_timeout"
    assert row["placement_events"] == "none:clear"


def test_a_clean_check_does_not_clear_a_perception_failure_at_the_trial(tmp_path):
    """15 Oct 2026, trial 23: a 227 mm long side for a 180 mm part, refused by the
    shape guard. The check saw a clean frame of the placement; the trial's own
    capture is the spilled one, and its refusal stands."""
    config = _real_config("rgbd")
    orch, robot, q = _orch(config, _called(), tmp_path)
    q.put(_frame(_part(config, 575.0, 20.0, 41.5)))
    spilled = _part(config, 575.0, 20.0, 42.6, yaw_off=30.0, rect=(227.2, 130.1))
    assert _trial(orch, q, Operator(q), _frame(spilled)) is False
    assert robot.sent == [] and orch.sm.history[-1].note == "yaw_geometry_reject"
    assert _row(tmp_path)["placement_events"] == "none:clear"


def test_the_flags_are_the_same_in_every_depth_mode(tmp_path):
    """Inox on the plastic box for a call on the metal box, turned 25 deg. rgbd
    commands the median depth, plane the class height on the table, fusion the
    measured top: three points of one ray, one set of flags at every check."""
    asked = {}
    for mode, used, point in (("rgbd", "rgbd", 66.0), ("plane", "plane", 40.0),
                              ("fusion", "rgbd", 71.0)):
        config = _real_config(mode)
        det = _part(config, 565.0, 30.0, 71.0, yaw_off=25.0, depth_used=used,
                    point_mm=point)
        orch, robot, q = _orch(config, _called(stack_on="metal_box"), tmp_path / mode,
                               logger=False)
        q.put(_frame(det))
        op = Operator(q, _frame(det), _frame(det))
        orch.run_placement_check(1, op)
        asked[mode] = op.asked
    assert asked["rgbd"] == asked["plane"] == asked["fusion"]
    assert [ln.split(":")[0] for ln in asked["rgbd"][0][1]] == ["STACK", "YAW"]


def test_a_cycle_with_a_called_pose_and_no_check_is_that_of_session_1(tmp_path):
    """No check before it (no hook, as at 5dcf934): run_one_cycle reads one
    detection window and acts on it. A part 65 mm off its card is picked as it
    lies, nobody is asked, nothing is logged for the check, and the robot gets the
    commands of a cycle that called no pose."""
    config = _real_config("rgbd")
    off_card = _part(config, 640.0, 20.0, 41.5)
    sent = {}
    for name, called in (("called", _called()), ("none", None)):
        orch, robot, q = _orch(config, called, tmp_path, logger=False)
        orch.trial_logger = TrialLogger(tmp_path / f"{name}.csv")
        orch.current_expected_class = "inox_box"
        orch.placement_check_hook = lambda *a: pytest.fail("a hook ran in the cycle")
        q.put(_frame(off_card))
        q.put(_frame(_part(config, 575.0, 20.0, 41.5)))
        assert orch.run_one_cycle(trial_id=1) is True
        assert q.qsize() == 1                         # one window, one frame
        row = _row(tmp_path, f"{name}.csv")
        for col in PLACEMENT_COLUMNS:
            assert row[col] == "", col
        sent[name] = robot.sent
    assert len(sent["called"]) == len(sent["none"]) == 1
    assert float(sent["called"][0]["pick"][0, 3]) == pytest.approx(640.0, abs=1.0)
    for key in ("pick", "approach", "place"):
        np.testing.assert_allclose(sent["called"][0][key], sent["none"][0][key])


def test_stopping_at_a_check_prompt_writes_aborted_at_check(tmp_path):
    """Ctrl+C at a check prompt (03 turns a closed stdin into the same) stops the
    run: the trial is written as aborted_at_check with what was judged, unscored,
    in no state of the trial before it."""
    config = _real_config("rgbd")
    asked = []

    def operator(trial_id, lines, last):
        raise KeyboardInterrupt

    orch, robot, q = _orch(config, _called(), tmp_path)
    orch.human_score_hook = lambda *a: asked.append(a) or 1
    orch.sm.fail("the trial before")
    q.put(_frame(_part(config, 640.0, 20.0, 41.5)))
    with pytest.raises(KeyboardInterrupt):
        orch.run_placement_check(1, operator)
    assert robot.sent == [] and asked == []
    row = _row(tmp_path)
    assert row["failure_reason"] == "aborted_at_check" and row["success"] == "0"
    assert row["placement_events"] == "position:abort" and row["human_ok"] == ""
    assert row["placement_flags"] == "position" and row["final_state"] == "idle"
    (cap,) = _captures(row)
    assert cap["outcome"] == "abort" and cap["top_x_mm"] == pytest.approx(640.0, abs=0.5)


def test_a_camera_without_frames_is_not_a_placement_flag(tmp_path):
    config = _real_config("rgbd")
    config["detection_timeout_s"] = 0.1
    orch, robot, q = _orch(config, _called(), tmp_path)
    op = Operator(q)
    assert _trial(orch, q, op, None) is False
    assert op.asked == []
    row = _row(tmp_path)
    assert row["failure_reason"] == "detection_timeout"
    assert row["placement_events"] == "noframe:skip"
    (cap,) = _captures(row)
    assert cap["frames"] == 0 and cap["outcome"] == "skip"


def test_a_check_with_no_frame_after_a_fix_is_not_read_as_cleared(tmp_path):
    config = _real_config("rgbd")
    config["detection_timeout_s"] = 0.2
    orch, robot, q = _orch(config, _called(), tmp_path)
    q.put(_frame(_part(config, 640.0, 20.0, 41.5)))
    orch.run_placement_check(1, Operator(q))           # the answer brings no frame
    record = orch._pending_placement
    assert record["placement_events"] == "position:fixed;noframe:skip"
    assert record["placement_recaptures"] == 1 and record["placement_override"] == 0


def test_every_check_keeps_its_detection_and_frame(tmp_path):
    """The flagged capture the operator acted on is kept beside the clear one, and
    neither is the trial's: frame, class, top centre, top height, yaw, flags,
    outcome, answer time."""
    config = _real_config("rgbd")
    orch, robot, q = _orch(config, _called(stack_on="metal_box"), tmp_path)
    q.put(_frame(_part(config, 575.0, 20.0, 41.5), path="first.png"))
    stacked = _part(config, 577.0, 22.0, 41.5 + 84.7, rect=(193.0, 130.2))
    op = Operator(q, _frame(stacked, path="second.png"), sleep_s=0.05)
    assert _trial(orch, q, op, _frame(stacked, path="trial.png"), trial_id=2) is True
    row = _row(tmp_path)
    assert row["frame_path"] == "trial.png"
    first, second = _captures(row)
    assert (first["capture"], second["capture"]) == (1, 2)
    assert first["frame"] == "first.png" and second["frame"] == "second.png"
    assert first["class"] == second["class"] == "inox_box"
    assert first["h_depth_mm"] == pytest.approx(41.5) and second["h_depth_mm"] == 126.2
    assert first["top_x_mm"] == pytest.approx(575.0, abs=0.5)
    assert second["top_y_mm"] == pytest.approx(22.0, abs=0.5)
    assert first["yaw_deg"] == pytest.approx(-50.0, abs=0.05)
    assert first["flags"] == ["stack"] and first["outcome"] == "fixed"
    assert first["answer_s"] >= 0.05 and second["outcome"] == "clear"
    assert float(row["placement_check_s"]) >= 0.05


def test_the_record_goes_with_the_next_cycle_only(tmp_path):
    config = _real_config("rgbd")
    orch, robot, q = _orch(config, _called(), tmp_path)
    q.put(_frame(_part(config, 640.0, 20.0, 41.5)))
    op = Operator(q, _frame(_part(config, 575.0, 20.0, 41.5)))
    _trial(orch, q, op, _frame(_part(config, 575.0, 20.0, 41.5)), trial_id=1)
    assert _row(tmp_path)["placement_events"] == "position:fixed;none:clear"
    assert orch._pending_placement == {}
    q.put(_frame(_part(config, 575.0, 20.0, 41.5)))
    orch.run_one_cycle(trial_id=2)                     # no check before this one
    row = _row(tmp_path)
    assert row["trial_id"] == "2"
    for col in PLACEMENT_COLUMNS:
        assert row[col] == "", col


def test_of_two_detections_of_the_called_class_the_topmost_is_judged(tmp_path):
    """A spare inox standing higher off the card and the called one on it: a cycle
    acts on the topmost, so that is the one checked."""
    config = _real_config("rgbd")
    spare = _part(config, 640.0, -120.0, 41.5)
    spare["h_median_mm"] = 60.0
    called_part = _part(config, 575.0, 20.0, 41.5)
    called_part["h_median_mm"] = 40.0
    orch, robot, q = _orch(config, _called(), tmp_path)
    q.put(_frame(called_part, spare))
    op = Operator(q, _frame(called_part, spare), _frame(called_part, spare))
    _trial(orch, q, op, _frame(called_part, spare))
    assert op.asked and op.asked[0][1][0].startswith("POSITION: part centre 154 mm")
    assert float(robot.sent[0]["pick"][0, 3]) == pytest.approx(640.0, abs=1.0)


def test_a_detection_without_depth_is_passed_over_by_the_check(tmp_path):
    """plane keeps a detection whose mask returned no depth, rgbd drops it: the
    check judges the next one, as rgbd would see the scene."""
    config = _real_config("plane")
    blind = _part(config, 640.0, -120.0, 40.0, depth_used="plane")
    blind.update(depth_valid_frac=3 / 20000, h_depth_mm=None, h_median_mm=80.0)
    seen = _part(config, 575.0, 20.0, 41.5, depth_used="plane")
    seen["h_median_mm"] = 40.0
    orch, robot, q = _orch(config, _called(), tmp_path)
    q.put(_frame(seen, blind))
    op = Operator(q)
    orch.run_placement_check(1, op)
    assert op.asked == []
    (cap,) = json.loads(orch._pending_placement["placement_captures"])
    assert cap["top_x_mm"] == pytest.approx(575.0, abs=0.5) and cap["flags"] == []


def test_the_centre_is_logged_as_the_top_face_centre(tmp_path):
    """In plane mode the commanded point of a stacked part lies on the table
    plane lifted by the class height; the logged top centre is on the part."""
    config = _real_config("plane")
    det = _part(config, 525.0, -100.0, 126.2, depth_used="plane", point_mm=40.0)
    orch, robot, q = _orch(config, _called(x=525.0, y=-100.0, stack_on="metal_box"),
                           tmp_path)
    q.put(_frame(det))
    orch.run_one_cycle(trial_id=1)
    row = _row(tmp_path)
    assert math.hypot(float(row["det_x_mm"]) - 525.0, float(row["det_y_mm"]) + 100.0) > 15
    assert float(row["top_x_mm"]) == pytest.approx(525.0, abs=0.5)
    assert float(row["top_y_mm"]) == pytest.approx(-100.0, abs=0.5)
