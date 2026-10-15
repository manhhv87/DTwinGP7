"""The real-cell cycle end to end, with the controller replaced by a recorder.

Earlier tests exercised one helper at a time, which is how a pick-height clamp
that never reached the robot, and an IK routine that raised on every call, went
unnoticed. These run run_one_cycle as 03_run_experiment configures it for
--mode real (same config files, calibration, preflight and predictive safety),
on detections measured on 15 Oct 2026, and check what the fixed DTWINRUN job
would have been sent.
"""
import pathlib
import queue

import numpy as np
import pytest
import yaml

from src.cell.cell_models import CellConfig
from src.orchestrator.coord_conv import load_calibration
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.preflight import check_depth_mode, check_real_mode

ROOT = pathlib.Path(__file__).resolve().parents[1]
HOME = [-86.481, -17.385, -21.274, 0.0, -86.115, 91.775]
BASE_Z = 630.0   # robot base in the world frame, unrotated (cell_layout_real.yaml)


def _real_config(depth_mode):
    """The configuration 03_run_experiment builds for --mode real."""
    config = yaml.safe_load((ROOT / "config/experiment.yaml").read_text(encoding="utf-8"))
    config.update(config.pop("real", None) or {})
    config["is_real_mode"] = True
    config["skip_reachability_check"] = False
    config["predictive_safety_enabled"] = True
    cell = CellConfig.from_yaml(ROOT / "config/cell_layout_real.yaml")
    config["home_joints_deg"] = list(cell.robot.home_joints_deg)
    calib = ROOT / config["calibration_path"]
    problems, table_z = check_real_mode(
        config, calib, cell.robot.pose.xyz_mm, cell.robot.pose.rpy_deg,
        getattr(cell.gripper, "tcp_offset_xyz_mm", None))
    dm_problems, plane = check_depth_mode(config, calib, depth_mode)
    assert not problems and not dm_problems, problems + dm_problems
    config["table_top_z_mm"] = table_z
    config["table_plane"] = dict(plane) if plane is not None else None
    config["robot_base_xyz_mm"] = tuple(cell.robot.pose.xyz_mm)
    config["robot_base_rpy_deg"] = tuple(cell.robot.pose.rpy_deg)
    config["robot_tool_offset_mm"] = 0.0
    config["use_yrc_ik"] = True
    config["inter_trial_delay_s"] = 0.0
    return config


class MotomanHSEBackend:
    """Stands in for the HSE backend under the same class name, so the real-cell
    yaw branch is taken, and records what DTWINRUN would receive."""

    __module__ = "tests.fake.motoman_hse"

    def __init__(self, start=HOME, end=HOME):
        self.joints = list(start)
        self.end = list(end)
        self.sent = []
        self.moves = []

    def Joints(self):
        return list(self.joints)

    def run_dtwinrun_motion(self, pick_T_base, approach_T_base, place_T_base,
                            job_name="DTWINRUN"):
        self.sent.append({"pick": np.asarray(pick_T_base),
                          "approach": np.asarray(approach_T_base),
                          "place": np.asarray(place_T_base)})
        self.joints = list(self.end)

    def setSpeed(self, *args, **kwargs):
        pass

    def MoveJ(self, target):
        self.moves.append(target)


def _plane_z(config, x, y):
    p = config.get("table_plane") or {}
    if not p:
        return float(config["table_top_z_mm"])
    return float(p["a"]) * x + float(p["b"]) * y + float(p["c"])


def _detection(config, x, y, z, cls, depth_used="rgbd", frac=0.9, h_depth=None,
               rect=(188.0, 90.0), yaw_deg=0.0):
    """A perception output for a part whose measured top is at world (x, y, z)."""
    T = load_calibration(ROOT / config["calibration_path"])
    p = np.linalg.inv(T) @ np.array([x, y, z, 1.0])
    return {"class_name": cls, "pose_camera": (p[0], p[1], p[2], 0.0),
            "confidence": 0.9, "depth_used": depth_used, "depth_valid_frac": frac,
            "h_depth_mm": h_depth, "yaw_base": float(np.deg2rad(yaw_deg)),
            "yaw_source": "metric_base_minAreaRect",
            "yaw_rect_long_mm": rect[0], "yaw_rect_short_mm": rect[1],
            "yaw_near_square": False, "yaw_top_height_mm": h_depth or 0.0,
            "pixel_uv": (640, 360), "mask_area": 20000}


def _cycle(depth_mode, det_args, start=HOME, end=HOME):
    config = _real_config(depth_mode)
    det = _detection(config, *det_args[0], **det_args[1])
    robot = MotomanHSEBackend(start, end)
    q = queue.Queue(maxsize=3)
    orch = Orchestrator(q, config=config, robot=robot)
    orch.current_expected_class = det["class_name"]
    q.put({"timestamp": 0.0, "objects": [det], "fps": 20.0})
    ok = orch.run_one_cycle(trial_id=1)
    return ok, orch, robot, config


def _world_z(T_base):
    return float(T_base[2, 3]) + BASE_Z


PLACE_XY = None


def _place_plane(config):
    x, y, _ = config["place_position"]
    return _plane_z(config, float(x), float(y))


class TestRgbd:
    def test_a_flat_plastic_box_is_held_35_mm_up_and_set_down_on_the_table(self):
        """15 Oct 2026, trial 2: plastic read 26.2 mm above the table top."""
        ok, orch, robot, cfg = _cycle("rgbd", ((579.9, -173.7, 620.1, "plastic_box"),
                                               dict(frac=0.999, h_depth=28.4,
                                                    rect=(183.4, 147.2), yaw_deg=-32.0)))
        assert ok is True and len(robot.sent) == 1
        p0 = _world_z(robot.sent[0]["pick"])
        p2 = _world_z(robot.sent[0]["place"])
        assert p0 == pytest.approx(cfg["table_top_z_mm"] + 35.0, abs=0.01)
        # Flat part: the TCP stays as high above the place table as it was
        # above the pick table, so the box lands on the table.
        assert p2 - _place_plane(cfg) == pytest.approx(
            p0 - _plane_z(cfg, 579.9, -173.7), abs=0.01)

    def test_an_80_mm_carton_on_the_plastic_box_is_set_down_on_the_table(self):
        """15 Oct 2026, trial 1: by height alone it read as a 120 mm carton."""
        ok, orch, robot, cfg = _cycle("rgbd", ((577.0, -149.5, 704.9, "carton"),
                                               dict(frac=0.957, h_depth=112.9,
                                                    rect=(196.8, 105.1), yaw_deg=75.9)))
        assert ok is True
        p2 = _world_z(robot.sent[0]["place"])
        assert p2 - _place_plane(cfg) == pytest.approx(80.0 + 8.0, abs=1.0)

    def test_inox_on_the_wood_box_keeps_clear_of_the_wood_and_lands_on_the_table(self):
        """15 Oct 2026, trial 31: the median read 11 mm below the top band."""
        ok, orch, robot, cfg = _cycle("rgbd", ((514.1, 30.1, 777.7, "inox_box"),
                                               dict(frac=0.451, h_depth=194.9,
                                                    rect=(189.3, 155.9), yaw_deg=56.5)))
        assert ok is True
        p0 = _world_z(robot.sent[0]["pick"])
        p2 = _world_z(robot.sent[0]["place"])
        wood_top = _plane_z(cfg, 514.1, 30.1) + 194.9 - 40.0
        assert p0 - wood_top >= 35.0 - 0.01
        assert 43.0 <= p2 - _place_plane(cfg) <= 48.0

    def test_a_collapsed_median_on_flat_inox_still_lands_on_the_table(self):
        """15 Oct 2026, trial 24: median 17.3 mm against a top band of 40.9."""
        ok, orch, robot, cfg = _cycle("rgbd", ((573.8, -40.7, 611.2, "inox_box"),
                                               dict(frac=0.222, h_depth=40.9,
                                                    rect=(191.6, 90.9), yaw_deg=38.3)))
        assert ok is True
        p0 = _world_z(robot.sent[0]["pick"])
        p2 = _world_z(robot.sent[0]["place"])
        assert p0 - cfg["table_top_z_mm"] == pytest.approx(35.9, abs=0.01)
        assert p2 - _place_plane(cfg) == pytest.approx(
            p0 - _plane_z(cfg, 573.8, -40.7), abs=0.01)

    def test_an_over_long_mask_is_refused_before_motion(self):
        """15 Oct 2026, trial 23: a 227 mm long side for a 180 mm part."""
        ok, orch, robot, cfg = _cycle("rgbd", ((572.5, 168.2, 596.9, "inox_box"),
                                               dict(frac=0.74, h_depth=42.6,
                                                    rect=(227.2, 130.1), yaw_deg=-89.1)))
        assert ok is False and robot.sent == []
        assert orch.sm.history[-1].note == "yaw_geometry_reject"


class TestPlaneAndFusion:
    def test_plane_on_a_stack_seen_by_depth_is_refused(self):
        """Trial 47's stack in plane mode: plane puts the inox on the table."""
        cfg0 = _real_config("plane")
        z = _plane_z(cfg0, 578.3, -29.1) + 40.0
        ok, orch, robot, cfg = _cycle("plane", ((578.3, -29.1, z, "inox_box"),
                                                dict(depth_used="plane", frac=0.267,
                                                     h_depth=194.8, yaw_deg=48.3)))
        assert ok is False and robot.sent == []
        assert orch.sm.history[-1].note == "depth_geometry_reject"

    def test_plane_on_a_flat_part_is_picked_from_the_plane_estimate(self):
        cfg0 = _real_config("plane")
        z = _plane_z(cfg0, 573.8, -40.7) + 40.0
        ok, orch, robot, cfg = _cycle("plane", ((573.8, -40.7, z, "inox_box"),
                                                dict(depth_used="plane", frac=0.6,
                                                     h_depth=41.0, yaw_deg=38.3)))
        assert ok is True
        p0 = _world_z(robot.sent[0]["pick"])
        assert p0 == pytest.approx(z, abs=0.01)

    def test_fusion_keeping_the_prior_on_a_flat_part(self):
        cfg0 = _real_config("fusion")
        z = _plane_z(cfg0, 575.0, 20.0) + 85.0
        ok, orch, robot, cfg = _cycle("fusion", ((575.0, 20.0, z, "metal_box"),
                                                 dict(depth_used="plane", frac=0.99,
                                                      h_depth=84.6, rect=(193.0, 139.0))))
        assert ok is True
        p2 = _world_z(robot.sent[0]["place"])
        assert p2 - _place_plane(cfg) == pytest.approx(85.0, abs=0.01)


class TestHome:
    ARGS = ((579.9, -173.7, 620.1, "plastic_box"),
            dict(frac=0.999, h_depth=28.4, rect=(183.4, 147.2)))

    def test_a_cycle_away_from_home_is_refused_before_motion(self):
        start = [q + 2.0 for q in HOME]
        ok, orch, robot, cfg = _cycle("rgbd", self.ARGS, start=start)
        assert ok is False and robot.sent == []
        assert orch.sm.history[-1].note == "not_at_home"

    def test_a_cycle_that_stops_half_way_is_a_failure_and_nothing_moves(self):
        """15 Oct 2026, trial 25: stopped by alarm 4746 above the carton."""
        stopped = [1.89, 7.54, 1.27, 0.0, -83.72, 80.54]
        ok, orch, robot, cfg = _cycle("rgbd", self.ARGS, end=stopped)
        assert ok is False and len(robot.sent) == 1
        assert "cycle incomplete" in orch.sm.history[-1].note
        assert robot.moves == []


# Heights measured with each part placed flat on 14 and 15 Oct 2026 (median of
# the top band), not the nominal ones: what each support really is.
FLAT_MM = {"plastic_box": 29.5, "wood_box": 153.1, "metal_box": 84.7,
           "inox_box": 41.5, "carton_80": 82.2, "carton_120": 125.4}

# One trial of each of the eight stacks in the pose lists, B3 of 15 Oct 2026:
# (target, support, det x, det y, det z, valid fraction, top band, rect, yaw).
STACKS = [
    ("carton", "plastic_box", 577.0, -149.5, 704.9, 0.957, 112.9, (196.8, 105.1), 75.9),
    ("inox_box", "plastic_box", 518.3, 145.0, 664.0, 0.874, 72.9, (197.4, 177.7), 11.0),
    ("wood_box", "metal_box", 510.8, 35.8, 829.7, 0.983, 238.0, (204.1, 193.1), 9.3),
    ("inox_box", "metal_box", 577.5, -38.8, 717.2, 0.492, 128.8, (193.0, 130.2), -3.2),
    ("inox_box", "carton_120", 583.3, 197.5, 758.7, 0.437, 167.1, (189.6, 99.6), -13.7),
    ("plastic_box", "wood_box", 633.1, -146.6, 774.9, 0.994, 182.6, (183.8, 167.4), -36.9),
    ("inox_box", "wood_box", 514.1, 30.1, 777.7, 0.451, 194.9, (189.3, 155.9), 56.5),
    ("metal_box", "inox_box", 527.1, -172.0, 720.0, 0.98, 128.3, (193.9, 143.1), -31.0),
]


@pytest.mark.parametrize("target,support,x,y,z,frac,h_depth,rect,yaw", STACKS)
def test_every_stack_is_set_down_neither_dropped_nor_pressed(
        target, support, x, y, z, frac, h_depth, rect, yaw):
    """The fault reported from the cell: a stacked part picked correctly was
    released at its stacked height and fell. At release its bottom must sit
    between the table and a small drop, never pressed into the table."""
    ok, orch, robot, cfg = _cycle("rgbd", ((x, y, z, target),
                                           dict(frac=frac, h_depth=h_depth,
                                                rect=rect, yaw_deg=yaw)))
    assert ok is True and len(robot.sent) == 1
    p0 = _world_z(robot.sent[0]["pick"])
    p2 = _world_z(robot.sent[0]["place"])
    bottom_at_pick = _plane_z(cfg, x, y) + FLAT_MM[support]
    gap = (p2 - _place_plane(cfg)) - (p0 - bottom_at_pick)
    assert 0.0 <= gap <= 12.0, gap
