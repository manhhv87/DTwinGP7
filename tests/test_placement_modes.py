"""
test_placement_modes.py
───────────────────────
The placement check reads a scene the same way in every depth mode, end to end.

Scenes are ray-cast through the cell calibration (tests/test_stacked_yaw.render),
each detection goes through the real PoseExtractor in rgbd, plane and fusion, and
the trial then runs through 03's pre-trial hook as on the real cell: the placement
gate, the placement prompt, the check with the servo OFF (the operator pressing
ENTER at every prompt and leaving the scene as it is), the motion gate, and
run_one_cycle. Every line shown and every prompt asked before the servo goes on,
and the check's record, must be identical in the three modes. What follows (the
trial's own detection, the motion guards and the motion) is the system under test
and may differ.

One scene is the edge case of PoseExtractor: a mask that returns depth on fewer
than 5 pixels. rgbd drops that detection, plane and fusion keep it; the check passes
over a detection without depth (placement_check.judged_detection), so all three
show the same CLASS line.

Run:
    pytest tests/test_placement_modes.py -v
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.placement_check import MAX_CHECKS, CalledPose, called_pose_from_row
from tests.test_operator_hooks import LOG, RUNNER, Camera, Console, FakeCell, _args
from tests.test_real_cycle import MotomanHSEBackend, _real_config
from tests.test_stacked_yaw import (
    CARD_7, HEIGHTS, INOX, LARGE_CARTON, METAL, PLASTIC, WOOD, extractor, render,
)

MODES = ("rgbd", "plane", "fusion")
CODE_HEADS = {"class": "CLASS", "position": "POSITION", "stack": "STACK",
              "carton_size": "CARTON SIZE", "yaw": "YAW"}


def _called(cls, yaw, stack_on="", carton_mm=None, stack_on_carton_mm=None):
    return CalledPose(pose_id="P0700", card_id="7", x_mm=CARD_7[0], y_mm=CARD_7[1],
                      yaw_deg=yaw, class_name=cls, stack_on=stack_on,
                      carton_mm=carton_mm, stack_on_carton_mm=stack_on_carton_mm)


def _no_depth(labels, depth, label):
    """The part's mask returns depth on 3 pixels only."""
    depth = depth.copy()
    mask = labels == label
    ys, xs = np.nonzero(mask)
    depth[mask] = 0.0
    for k in range(3):
        depth[ys[k * 50], xs[k * 50]] = 1.2
    return depth


# name: (called pose, boxes as render takes them, class, the codes expected)
# render's yaw is the long side's angle in the base frame: the called yaw - 90.
SCENES = {
    "inox_as_called": (
        _called("inox_box", 40.0), ((CARD_7, -50.0, INOX, 0.0),), "inox_box", []),
    "inox_on_plastic_for_metal_turned_25": (
        _called("inox_box", 40.0, "metal_box"),
        ((CARD_7, -25.0, PLASTIC, 0.0), (CARD_7, -25.0, INOX, PLASTIC[2])),
        "inox_box", ["stack", "yaw"]),
    "metal_on_plastic_for_metal_on_inox": (
        _called("metal_box", -30.0, "inox_box"),
        ((CARD_7, -120.0, PLASTIC, 0.0), (CARD_7, -120.0, METAL, PLASTIC[2])),
        "metal_box", ["stack"]),
    "carton120_for_carton80": (
        _called("carton", 40.0, carton_mm=80), ((CARD_7, -50.0, LARGE_CARTON, 0.0),),
        "carton", ["carton_size"]),
    "inox_60_mm_off_its_card": (
        _called("inox_box", 40.0), (((CARD_7[0] + 60.0, CARD_7[1]), -50.0, INOX, 0.0),),
        "inox_box", ["position"]),
    "inox_turned_30": (
        _called("inox_box", 40.0), ((CARD_7, -20.0, INOX, 0.0),), "inox_box", ["yaw"]),
    "wood_turned_87_is_square": (
        _called("wood_box", 10.0), ((CARD_7, -80.0 + 87.0, WOOD, 0.0),), "wood_box", []),
    "inox_without_depth": (
        _called("inox_box", 40.0), ((CARD_7, -50.0, INOX, 0.0),), "inox_box", ["class"]),
}


class Operator(Console):
    """ENTER at every prompt of the placement, y at the verdict."""

    def ask(self, prompt):
        self.events.append(("ask", prompt, self.cell.status["servo"]))
        return "y" if "did the robot" in prompt else ""


def _pose_list_row(called):
    """The pose-list row and carton sizes with which 03 calls this staging."""
    row = {"pose_id": called.pose_id, "card_id": called.card_id,
           "x_mm": str(called.x_mm), "y_mm": str(called.y_mm),
           "yaw_deg": str(called.yaw_deg), "class_hint": called.class_name,
           "stack_on": called.stack_on, "condition": ""}
    sizes = {called.pose_id: (called.carton_mm, called.stack_on_carton_mm)}
    assert called_pose_from_row(row, sizes) == called
    return row, sizes


def _through_03(name, mode):
    called, boxes, cls, _ = SCENES[name]
    labels, depth = render(boxes)
    label = len(boxes)
    if name == "inox_without_depth":
        depth = _no_depth(labels, depth, label)
    det = extractor(mode).extract(
        {"mask": (labels == label).astype(np.uint8), "class_name": cls,
         "height_mm": HEIGHTS[cls], "confidence": 0.9}, depth)
    msg = {"timestamp": 0.0, "objects": [] if det is None else [det], "fps": 20.0}
    config = _real_config(mode)
    config["detection_timeout_s"] = 0.2
    cell = FakeCell(servo=True)
    op = Operator(cell)
    cell.events = op.events
    # The scene as placed, unchanged: one frame for every check and for the trial.
    cam = Camera([msg] * (MAX_CHECKS + 1), cell, op.events)
    orch = Orchestrator(cam, config=config, robot=MotomanHSEBackend())
    row, sizes = _pose_list_row(called)
    pre = RUNNER.wire_trial_hooks(orch, _args(), [row], sizes, cell, cam, LOG,
                                  ask=op.ask, say=op.say, flush=op.flush)
    orch.run_n_trials(1, pre_trial_hook=pre)
    before_motion = op.events[:op.events.index(("cell", "servo_on"))]
    shown = [e for e in before_motion if e[0] in ("say", "ask")]
    record = dict(orch._placement)
    captures = json.loads(record.pop("placement_captures"))
    for c in captures:
        c.pop("answer_s", None)
    record.pop("placement_check_s")
    return det, shown, record, captures


@pytest.mark.parametrize("name", sorted(SCENES))
def test_the_operator_is_shown_the_same_in_every_mode(name):
    seen = {mode: _through_03(name, mode) for mode in MODES}
    shown = {mode: seen[mode][1] for mode in MODES}
    records = {mode: seen[mode][2] for mode in MODES}
    captures = {mode: seen[mode][3] for mode in MODES}
    assert shown["rgbd"] == shown["plane"] == shown["fusion"]
    assert records["rgbd"] == records["plane"] == records["fusion"]
    assert captures["rgbd"] == captures["plane"] == captures["fusion"]
    expected = SCENES[name][3]
    assert records["rgbd"]["placement_flags"] == "+".join(expected)
    flag_lines = [e[1].strip() for e in shown["rgbd"]
                  if e[0] == "say" and e[1].startswith("     ")]
    assert [ln.split(":")[0] for ln in flag_lines[:len(expected)]] == [
        CODE_HEADS[c] for c in expected]
    # Left as placed, a flagged scene is checked MAX_CHECKS times, then runs.
    prompts = [e for e in shown["rgbd"] if e[0] == "ask"]
    assert len(prompts) == (1 + MAX_CHECKS if expected else 1)
    assert all(servo is False for _, _, servo in prompts)


def test_without_depth_only_rgbd_drops_the_detection():
    """The asymmetry the check has to absorb: rgbd has no detection, plane and
    fusion have one without depth; the CLASS line is the same in all three."""
    seen = {mode: _through_03("inox_without_depth", mode) for mode in MODES}
    assert seen["rgbd"][0] is None
    assert seen["plane"][0] is not None and seen["fusion"][0] is not None
    assert seen["plane"][0]["h_depth_mm"] is None
    for mode in MODES:
        said = [e[1] for e in seen[mode][1] if e[0] == "say"]
        assert said.count("     CLASS: no inox_box found; check the part on card 7") == (
            MAX_CHECKS)
