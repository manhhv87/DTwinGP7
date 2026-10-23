"""The placement check inside run_one_cycle, with a fake operator.

The orchestrator is configured as 03_run_experiment configures it for --mode real
(tests/test_real_cycle.py), the controller is a recorder, and the operator is a hook
that answers "recapture" or "proceed" like ask_placement_fix in 03. What must hold:

  * no flag, no question;
  * ENTER over a flag goes on with the detection that was checked, so a perception
    failure stays a failure and no further frame is read;
  * a fix captures again and goes on with the new detection;
  * recaptures stop at MAX_RECAPTURES;
  * the flags are the same in every depth mode;
  * every check, the choice and the recaptures land in the CSV.

Run:
    pytest tests/test_placement_cycle.py -v
"""
from __future__ import annotations

import csv
import math
import queue

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


class Operator:
    """Answers the placement check from a script; records every question."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.asked = []

    def __call__(self, trial_id, lines, can_recapture):
        self.asked.append((trial_id, list(lines), can_recapture))
        return self.answers.pop(0) if self.answers else "proceed"


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


def _orch(config, operator, called, frames, tmp_path, logger=True):
    robot = MotomanHSEBackend()
    q = queue.Queue()
    trial_logger = TrialLogger(tmp_path / "run.csv") if logger else None
    orch = Orchestrator(q, config=config, robot=robot, logger_obj=trial_logger)
    orch.current_expected_class = called.class_name
    orch.current_called_pose = called
    orch.placement_check_hook = operator
    for objects in frames:
        q.put({"timestamp": 0.0, "objects": objects, "fps": 20.0})
    return orch, robot, q


def _row(tmp_path):
    with (tmp_path / "run.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))[-1]


def test_the_new_columns_are_in_the_schema():
    for col in ("carton_mm", "stack_on_carton_mm", "top_x_mm", "top_y_mm",
                "placement_flags", "placement_recaptures", "placement_override",
                "placement_events"):
        assert col in CONTEXT_FIELDNAMES


def test_no_flag_no_question(tmp_path):
    config = _real_config("rgbd")
    op = Operator()
    orch, robot, q = _orch(config, op, _called(),
                           [[_part(config, 578.0, 16.0, 41.5, yaw_off=3.0)]], tmp_path)
    assert orch.run_one_cycle(trial_id=1) is True
    assert op.asked == [] and len(robot.sent) == 1
    row = _row(tmp_path)
    assert row["placement_flags"] == "" and row["placement_events"] == "none:clear"
    assert row["placement_recaptures"] == "0" and row["placement_override"] == "0"
    assert float(row["top_x_mm"]) == pytest.approx(578.0, abs=0.5)
    assert float(row["top_y_mm"]) == pytest.approx(16.0, abs=0.5)


def test_enter_over_a_flag_goes_on_with_the_checked_detection(tmp_path):
    config = _real_config("rgbd")
    op = Operator("proceed")
    off_card = _part(config, 635.0, 20.0, 41.5)
    on_card = _part(config, 575.0, 20.0, 41.5)
    orch, robot, q = _orch(config, op, _called(), [[off_card], [on_card]], tmp_path)
    assert orch.run_one_cycle(trial_id=4) is True
    assert len(op.asked) == 1
    trial_id, lines, can = op.asked[0]
    assert trial_id == 4 and can is True
    assert lines[0].startswith("POSITION: part centre 60 mm from card 11")
    pick = robot.sent[0]["pick"]
    assert float(pick[0, 3]) == pytest.approx(635.0, abs=1.0)
    assert q.qsize() == 1                     # the second frame was never read
    row = _row(tmp_path)
    assert row["placement_flags"] == "position"
    assert row["placement_override"] == "1" and row["placement_recaptures"] == "0"
    assert row["placement_events"] == "position:proceed"


def test_a_fixed_placement_is_captured_again(tmp_path):
    config = _real_config("rgbd")
    op = Operator("recapture")
    called = _called(stack_on="metal_box")
    alone = _part(config, 575.0, 20.0, 41.5)
    stacked = _part(config, 577.0, 22.0, 41.5 + 84.7, rect=(193.0, 130.2))
    orch, robot, q = _orch(config, op, called, [[alone], [stacked]], tmp_path)
    assert orch.run_one_cycle(trial_id=2) is True
    assert len(op.asked) == 1 and op.asked[0][1][0].startswith("STACK:")
    row = _row(tmp_path)
    assert row["placement_flags"] == "stack"
    assert row["placement_events"] == "stack:recapture;none:clear"
    assert row["placement_recaptures"] == "1" and row["placement_override"] == "0"
    # The trial went on with the second capture, the part on its support: the
    # pick stays above the metal box.
    assert float(row["h_depth_mm"]) == pytest.approx(126.2, abs=0.1)
    assert _world_z(robot.sent[0]["pick"]) - _plane_z(config, 577.0, 22.0) > 84.7


def test_recaptures_stop_at_the_limit(tmp_path):
    config = _real_config("rgbd")
    op = Operator("recapture", "recapture", "recapture", "recapture", "recapture")
    frames = [[_part(config, 575.0, 20.0 + 60.0 + k, 41.5)] for k in range(5)]
    orch, robot, q = _orch(config, op, _called(), frames, tmp_path)
    orch.run_one_cycle(trial_id=1)
    assert [can for _, _, can in op.asked] == [True, True, True, False]
    assert orch_mod.MAX_RECAPTURES == 3
    pick = robot.sent[0]["pick"]
    assert float(pick[1, 3]) == pytest.approx(83.0, abs=1.0)   # the fourth capture
    row = _row(tmp_path)
    assert row["placement_recaptures"] == "3" and row["placement_override"] == "0"
    assert row["placement_events"] == ";".join(["position:recapture"] * 3
                                               + ["position:limit"])
    assert q.qsize() == 1


def test_the_limit_follows_the_constant(tmp_path, monkeypatch):
    monkeypatch.setattr(orch_mod, "MAX_RECAPTURES", 1)
    config = _real_config("rgbd")
    op = Operator("recapture", "recapture")
    frames = [[_part(config, 575.0, 90.0, 41.5)], [_part(config, 575.0, 91.0, 41.5)]]
    orch, robot, q = _orch(config, op, _called(), frames, tmp_path)
    orch.run_one_cycle(trial_id=1)
    assert [can for _, _, can in op.asked] == [True, False]


def test_the_wrong_class_is_flagged_and_stays_a_miss(tmp_path):
    config = _real_config("rgbd")
    config["detection_timeout_s"] = 0.2
    op = Operator("proceed")
    called = _called(cls="carton", carton_mm=120)
    orch, robot, q = _orch(config, op, called, [[_part(config, 575.0, 20.0, 41.5)]],
                           tmp_path)
    assert orch.run_one_cycle(trial_id=17) is False
    assert robot.sent == []
    assert op.asked[0][1] == ["CLASS: no carton(120) found; check the part on card 11"]
    row = _row(tmp_path)
    assert row["failure_reason"] == "detection_miss" and row["class_name"] == ""
    assert row["placement_flags"] == "class" and row["placement_override"] == "1"
    assert row["carton_mm"] == "120"


def test_a_perception_failure_is_not_retried_into_a_success(tmp_path):
    """15 Oct 2026, trial 23: a 227 mm long side for a 180 mm part, refused by the
    shape guard. Placed 30 deg off the called yaw, it is also flagged; the operator
    sees the part where it should be (yaw aside) and goes on: the refusal stands,
    and the clean frame behind it is never read."""
    config = _real_config("rgbd")
    op = Operator("proceed")
    spilled = _part(config, 575.0, 20.0, 42.6, yaw_off=30.0, rect=(227.2, 130.1))
    clean = _part(config, 575.0, 20.0, 41.5)
    orch, robot, q = _orch(config, op, _called(), [[spilled], [clean]], tmp_path)
    assert orch.run_one_cycle(trial_id=3) is False
    assert robot.sent == [] and orch.sm.history[-1].note == "yaw_geometry_reject"
    assert q.qsize() == 1
    assert _row(tmp_path)["placement_events"] == "yaw:proceed"


def test_the_flags_are_the_same_in_every_depth_mode(tmp_path):
    """Inox on the plastic box for a call on the metal box, turned 25 deg. rgbd
    commands the median depth, plane the class height on the table, fusion the
    measured top: three points of one ray, one set of flags."""
    lines = {}
    for mode, used, point in (("rgbd", "rgbd", 66.0), ("plane", "plane", 40.0),
                              ("fusion", "rgbd", 71.0)):
        config = _real_config(mode)
        op = Operator("proceed")
        det = _part(config, 565.0, 30.0, 71.0, yaw_off=25.0, depth_used=used,
                    point_mm=point)
        orch, robot, q = _orch(config, op, _called(stack_on="metal_box"), [[det]],
                               tmp_path / mode, logger=False)
        orch.run_one_cycle(trial_id=1)
        lines[mode] = op.asked[0][1]
    assert lines["rgbd"] == lines["plane"] == lines["fusion"]
    assert [ln.split(":")[0] for ln in lines["rgbd"]] == ["STACK", "YAW"]


def test_without_the_hook_nothing_is_checked(tmp_path):
    config = _real_config("rgbd")
    orch, robot, q = _orch(config, None, _called(stack_on="carton", stack_on_carton_mm=80),
                           [[_part(config, 640.0, 20.0, 41.5)]], tmp_path)
    assert orch.run_one_cycle(trial_id=1) is True
    row = _row(tmp_path)
    assert row["placement_events"] == "" and row["placement_flags"] == ""
    assert row["stack_on_carton_mm"] == "80" and row["carton_mm"] == ""


def test_stopping_at_the_check_moves_nothing_and_logs_nothing(tmp_path):
    config = _real_config("rgbd")

    def operator(trial_id, lines, can_recapture):
        raise KeyboardInterrupt

    orch, robot, q = _orch(config, operator, _called(),
                           [[_part(config, 640.0, 20.0, 41.5)]], tmp_path)
    with pytest.raises(KeyboardInterrupt):
        orch.run_one_cycle(trial_id=1)
    assert robot.sent == []
    with (tmp_path / "run.csv").open(newline="", encoding="utf-8") as f:
        assert list(csv.DictReader(f)) == []


def test_a_camera_without_frames_is_not_a_placement_flag(tmp_path):
    config = _real_config("rgbd")
    config["detection_timeout_s"] = 0.1
    op = Operator()
    orch, robot, q = _orch(config, op, _called(), [], tmp_path)
    assert orch.run_one_cycle(trial_id=1) is False
    assert op.asked == []
    row = _row(tmp_path)
    assert row["failure_reason"] == "detection_timeout"
    assert row["placement_events"] == "none:clear"


def test_the_centre_is_logged_as_the_top_face_centre(tmp_path):
    """In plane mode the commanded point of a stacked part lies on the table
    plane lifted by the class height; the logged top centre is on the part."""
    config = _real_config("plane")
    det = _part(config, 525.0, -100.0, 126.2, depth_used="plane", point_mm=40.0)
    orch, robot, q = _orch(config, None, _called(x=525.0, y=-100.0, stack_on="metal_box"),
                           [[det]], tmp_path)
    orch.run_one_cycle(trial_id=1)
    row = _row(tmp_path)
    assert math.hypot(float(row["det_x_mm"]) - 525.0, float(row["det_y_mm"]) + 100.0) > 15
    assert float(row["top_x_mm"]) == pytest.approx(525.0, abs=0.5)
    assert float(row["top_y_mm"]) == pytest.approx(-100.0, abs=0.5)
