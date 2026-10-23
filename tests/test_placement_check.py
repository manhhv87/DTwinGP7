"""The placement check: the scene perception sees against the pose the trial called.

Unit tests of src/orchestrator/placement_check.py, then trials of session 1 of E2
(e2chuan-1, 23 Oct 2026) rebuilt from their logged rows: every kind of verified
staging deviation is flagged, and the genuine trials nearest the thresholds are not.
The rows and the calibration in config/calibration are those of that session.

Run:
    pytest tests/test_placement_check.py -v
"""
from __future__ import annotations

import csv
import json
import math
import pathlib

import numpy as np
import pytest

from src.orchestrator.placement_check import (
    CARTON_TOP_MM, FLAG_CODES, FLAT_TOP_MM, HEIGHT_TOL_ABOVE_MM, HEIGHT_TOL_BELOW_MIRROR_MM,
    HEIGHT_TOL_BELOW_MM, MAX_CHECKS, MIN_DEPTH_PIXELS, NEAR_SQUARE_RATIO,
    POSITION_TOL_MM, YAW_TOL_DEG, CalledPose, called_pose_from_row, carton_sizes_path,
    check_placement, class_label, detection_yaw_period_deg, has_depth, judged_detection,
    load_carton_sizes, lower_tolerance_mm, missing_carton_sizes, top_centre_xy,
    yaw_error_deg, yaw_period_deg,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
T_BC = np.load(ROOT / "config/calibration/T_base_camera.npy")
PLANE = json.loads((ROOT / "config/calibration/table_plane.json").read_text(encoding="utf-8"))
SIZES = {"carton": [[180, 120, 120], [180, 100, 80]], "plastic_box": [[180, 140, 30]],
         "wood_box": [[190, 190, 150]], "metal_box": [[190, 135, 85]],
         "inox_box": [[180, 85, 40]]}


def _plane_z(x, y):
    return PLANE["a"] * x + PLANE["b"] * y + PLANE["c"]


def _height_above(p):
    n_z = 1.0 / math.sqrt(PLANE["a"] ** 2 + PLANE["b"] ** 2 + 1.0)
    return (p[2] - _plane_z(p[0], p[1])) * n_z


def _detection(x, y, top_mm, *, yaw_deg=0.0, h_depth=None, cls="inox_box",
               depth_mm=None, yaw_top=None):
    """A detection whose top-face centre is at base (x, y) on the plane lifted by
    top_mm; pose_camera is the point of that ray at camera depth depth_mm (the
    commanded point of some depth mode), the top itself when None."""
    n_z = 1.0 / math.sqrt(PLANE["a"] ** 2 + PLANE["b"] ** 2 + 1.0)
    top = np.array([x, y, _plane_z(x, y) + top_mm / n_z])
    cam = np.linalg.inv(T_BC) @ np.append(top, 1.0)
    p = cam[:3] if depth_mm is None else cam[:3] * (depth_mm / cam[2])
    return {"class_name": cls, "pose_camera": (p[0], p[1], p[2], 0.0),
            "yaw_top_height_mm": top_mm if yaw_top is None else yaw_top,
            "h_depth_mm": top_mm if h_depth is None else h_depth,
            "yaw_base": math.radians(yaw_deg)}


def _called(cls="inox_box", x=575.0, y=20.0, yaw=0.0, stack_on="", carton_mm=None,
            stack_on_carton_mm=None, card="11"):
    return CalledPose(pose_id="P9999", card_id=card, x_mm=x, y_mm=y, yaw_deg=yaw,
                      class_name=cls, stack_on=stack_on, carton_mm=carton_mm,
                      stack_on_carton_mm=stack_on_carton_mm)


def _flags(called, obj, **kw):
    return [f.code for f in check_placement(called, obj, T_BC=T_BC, table_plane=PLANE,
                                            class_sizes=SIZES, **kw)]


# Pose-list yaw = measured yaw + 90: a part placed at the called yaw reads called - 90.
def _yaw_for(called_yaw, off=0.0):
    return called_yaw - 90.0 + off


class TestFlags:
    def test_a_part_placed_as_called_raises_nothing(self):
        obj = _detection(578.0, 15.0, 41.5, yaw_deg=_yaw_for(30.0, 4.0))
        assert _flags(_called(yaw=30.0), obj) == []

    def test_no_usable_detection_of_the_called_class(self):
        flags = check_placement(_called(cls="carton", carton_mm=120), None)
        assert [f.code for f in flags] == ["class"]
        assert flags[0].text == "CLASS: no carton(120) found; check the part on card 11"

    def test_a_class_miss_on_a_stacked_pose_names_the_support(self):
        """Fusion P0062 of session 1, inox on the metal box for the plastic box, was
        flagged only as class: the line now points at the support as well."""
        flags = check_placement(_called(stack_on="plastic_box", card="4"), None)
        assert flags[0].text == ("CLASS: no inox_box found; check the part on card 4 "
                                 "and the plastic_box under it")
        flags = check_placement(_called(stack_on="carton", stack_on_carton_mm=120), None)
        assert flags[0].text.endswith("and the carton(120) under it")

    def test_no_frame_at_all_is_a_camera_fault_not_a_placement(self):
        assert _flags(_called(), None, frames_seen=False) == []

    @pytest.mark.parametrize("off,flagged", [(30.0, False), (POSITION_TOL_MM + 5, True),
                                             (60.0, True)])
    def test_position(self, off, flagged):
        obj = _detection(575.0 + off, 20.0, 41.5, yaw_deg=_yaw_for(0.0))
        assert ("position" in _flags(_called(), obj)) is flagged

    def test_a_missing_support(self):
        flags = check_placement(_called(stack_on="metal_box"),
                                _detection(575.0, 20.0, 41.5, yaw_deg=_yaw_for(0.0)),
                                T_BC=T_BC, table_plane=PLANE)
        assert [f.code for f in flags] == ["stack"]
        assert "top at 42 mm" in flags[0].text and "126 mm expected" in flags[0].text

    def test_a_support_under_a_single_part(self):
        obj = _detection(575.0, 20.0, 41.5 + 29.5, yaw_deg=_yaw_for(0.0))
        flags = check_placement(_called(), obj, T_BC=T_BC, table_plane=PLANE)
        assert [f.code for f in flags] == ["stack"]
        assert "nothing is under the part" in flags[0].text

    def test_a_single_part_reading_low_names_no_support(self):
        """Nothing put under a part makes it read lower: the line asks to check the
        part, not what is under it."""
        obj = _detection(575.0, 20.0, 41.5, h_depth=41.5 - 20.0, yaw_deg=_yaw_for(0.0))
        flags = check_placement(_called(), obj, T_BC=T_BC, table_plane=PLANE)
        assert [f.text for f in flags] == [
            "STACK: top at 22 mm, 42 mm expected for inox_box alone; check the part"]

    def test_a_wrong_support(self):
        """Fusion P0016 of session 1: inox on the plastic box for the metal box."""
        obj = _detection(575.0, -40.0, 74.0, yaw_deg=_yaw_for(0.0))
        assert _flags(_called(stack_on="metal_box", y=-40.0), obj) == ["stack"]

    @pytest.mark.parametrize("cls,top,delta,flagged", [
        ("inox_box", 41.5, HEIGHT_TOL_ABOVE_MM - 0.5, False),
        ("inox_box", 41.5, HEIGHT_TOL_ABOVE_MM + 0.5, True),
        ("inox_box", 41.5, -HEIGHT_TOL_BELOW_MIRROR_MM + 0.5, False),
        ("inox_box", 41.5, -HEIGHT_TOL_BELOW_MIRROR_MM - 0.5, True),
        ("metal_box", 84.7, HEIGHT_TOL_ABOVE_MM - 0.5, False),
        ("metal_box", 84.7, HEIGHT_TOL_ABOVE_MM + 0.5, True),
        ("metal_box", 84.7, -HEIGHT_TOL_BELOW_MM + 0.5, False),
        ("metal_box", 84.7, -HEIGHT_TOL_BELOW_MM - 0.5, True),
        ("wood_box", 153.1, -HEIGHT_TOL_BELOW_MM - 0.5, True),
        ("plastic_box", 29.5, -HEIGHT_TOL_BELOW_MM + 0.5, False)])
    def test_the_height_band(self, cls, top, delta, flagged):
        """A mirror top (the inox box) may read up to 15 mm low, a matte one 8 mm."""
        obj = _detection(575.0, 20.0, top, h_depth=top + delta, cls=cls,
                         yaw_deg=_yaw_for(0.0))
        assert ("stack" in _flags(_called(cls=cls), obj)) is flagged

    def test_the_lower_tolerance_follows_the_top(self):
        assert lower_tolerance_mm("inox_box") == HEIGHT_TOL_BELOW_MIRROR_MM == 15.0
        for cls in ("carton", "plastic_box", "wood_box", "metal_box"):
            assert lower_tolerance_mm(cls) == HEIGHT_TOL_BELOW_MM == 8.0

    def test_a_support_12_mm_lower_under_a_matte_part(self):
        """The metal box on the plastic box for a call on the inox box reads 12.0 mm
        low; with one 15 mm band for every class (b9f4e52) it passed in every mode."""
        obj = _detection(575.0, 20.0, 84.7 + 29.5, cls="metal_box", yaw_deg=_yaw_for(0.0))
        flags = check_placement(_called(cls="metal_box", stack_on="inox_box"), obj,
                                T_BC=T_BC, table_plane=PLANE)
        assert [f.text for f in flags] == [
            "STACK: top at 114 mm, 126 mm expected for metal_box on inox_box; "
            "check the support"]

    def test_a_small_carton_on_a_support_for_a_large_carton_alone(self):
        """carton(80) on the plastic box for a carton(120) alone reads 13.7 mm low."""
        obj = _detection(575.0, 20.0, 82.2 + 29.5, cls="carton", yaw_deg=_yaw_for(0.0))
        flags = check_placement(_called(cls="carton", carton_mm=120), obj,
                                T_BC=T_BC, table_plane=PLANE)
        assert [f.text for f in flags] == [
            "STACK: top at 112 mm, 125 mm expected for carton(120) alone; check the part"]

    def test_the_other_carton_size_as_target(self):
        obj = _detection(575.0, 20.0, 82.2, cls="carton", yaw_deg=_yaw_for(0.0))
        flags = check_placement(_called(cls="carton", carton_mm=120), obj,
                                T_BC=T_BC, table_plane=PLANE)
        assert [f.code for f in flags] == ["carton_size"]
        assert "carton(80) alone" in flags[0].text

    def test_the_other_carton_size_stacked_on_plastic(self):
        obj = _detection(575.0, 20.0, 82.2 + 29.5, cls="carton", yaw_deg=_yaw_for(0.0))
        called = _called(cls="carton", carton_mm=120, stack_on="plastic_box")
        assert _flags(called, obj) == ["carton_size"]

    def test_the_other_carton_size_as_support(self):
        obj = _detection(575.0, 20.0, 41.5 + 125.4, yaw_deg=_yaw_for(0.0))
        called = _called(stack_on="carton", stack_on_carton_mm=80)
        flags = check_placement(called, obj, T_BC=T_BC, table_plane=PLANE)
        assert [f.code for f in flags] == ["carton_size"]
        assert "inox_box on carton(120)" in flags[0].text

    def test_either_carton_size_passes_when_none_was_called(self):
        for top in (82.2, 125.4):
            obj = _detection(575.0, 20.0, top, cls="carton", yaw_deg=_yaw_for(0.0))
            assert _flags(_called(cls="carton"), obj) == []

    @pytest.mark.parametrize("off,flagged", [(YAW_TOL_DEG - 1, False),
                                             (YAW_TOL_DEG + 1, True), (-35.0, True)])
    def test_yaw(self, off, flagged):
        obj = _detection(575.0, 20.0, 41.5, yaw_deg=_yaw_for(40.0, off))
        assert ("yaw" in _flags(_called(yaw=40.0), obj)) is flagged

    def test_a_part_turned_half_a_turn_is_as_called(self):
        obj = _detection(575.0, 20.0, 41.5, yaw_deg=_yaw_for(40.0, 180.0))
        assert _flags(_called(yaw=40.0), obj) == []

    def test_a_near_square_outline_folds_the_yaw_by_90_deg(self):
        """Without the known-size fit PoseExtractor folds the yaw of a rectangle read
        near square into (-45, 45] deg: a plastic box placed as called at 10 deg
        (measured -80 deg) reads +10 deg, which is no yaw error."""
        obj = _detection(575.0, 20.0, 29.5, cls="plastic_box", yaw_deg=10.0)
        obj.update(yaw_near_square=True, yaw_source="metric_base_minAreaRect")
        called = _called(cls="plastic_box", yaw=10.0)
        assert detection_yaw_period_deg(obj, "plastic_box", SIZES) == 90.0
        assert _flags(called, obj) == []
        obj["yaw_source"] = "known_size_fit"          # the fit folds by the class
        assert detection_yaw_period_deg(obj, "plastic_box", SIZES) == 180.0
        assert _flags(called, obj) == ["yaw"]

    def test_the_square_wood_box_repeats_every_90_deg(self):
        obj = _detection(575.0, 20.0, 153.1, cls="wood_box", yaw_deg=_yaw_for(10.0, 87.0))
        assert _flags(_called(cls="wood_box", yaw=10.0), obj) == []
        obj = _detection(575.0, 20.0, 41.5, cls="inox_box", yaw_deg=_yaw_for(10.0, 87.0))
        assert _flags(_called(yaw=10.0), obj) == ["yaw"]

    def test_several_flags_at_once(self):
        """Plane P0033 of session 1: a spare wood box labelled carton at the edge."""
        obj = _detection(900.0, 130.0, 151.8, cls="carton", yaw_deg=_yaw_for(-65.0, 66.0))
        flags = _flags(_called(cls="carton", x=625.0, y=-100.0, yaw=-65.0), obj)
        assert flags == ["position", "stack", "yaw"]
        assert set(flags) <= set(FLAG_CODES)


class TestMissingData:
    def test_no_measured_top_skips_the_height(self):
        obj = _detection(575.0, 20.0, 41.5, yaw_deg=_yaw_for(0.0))
        obj["h_depth_mm"] = None
        assert _flags(_called(stack_on="metal_box"), obj) == []

    def test_no_table_plane_skips_the_position(self):
        obj = _detection(700.0, 20.0, 41.5, yaw_deg=_yaw_for(0.0))
        flags = check_placement(_called(), obj, T_BC=T_BC, table_plane=None)
        assert flags == []

    def test_no_called_yaw_skips_the_yaw(self):
        obj = _detection(575.0, 20.0, 41.5, yaw_deg=33.0)
        called = CalledPose(class_name="inox_box", x_mm=575.0, y_mm=20.0)
        assert _flags(called, obj) == []

    def test_an_unknown_class_skips_the_height(self):
        obj = _detection(575.0, 20.0, 99.0, cls="tray", yaw_deg=_yaw_for(0.0))
        assert _flags(_called(cls="tray"), obj) == []


class TestCentre:
    @pytest.mark.parametrize("depth_mm", [None, 640.0, 700.0])
    def test_the_same_scene_gives_the_same_centre_in_every_depth_mode(self, depth_mm):
        """rgbd commands the median depth, plane the class height on the table,
        fusion either: points of one ray. The centre is cut on the measured top."""
        obj = _detection(560.0, -60.0, 41.5 + 84.7, depth_mm=depth_mm)
        cx, cy = top_centre_xy(obj, T_BC, PLANE)
        assert (cx, cy) == pytest.approx((560.0, -60.0), abs=1e-6)

    def test_without_a_measured_top_the_fit_plane_is_used(self):
        obj = _detection(560.0, -60.0, 41.5, depth_mm=650.0)
        obj["h_depth_mm"] = None
        assert top_centre_xy(obj, T_BC, PLANE) == pytest.approx((560.0, -60.0), abs=1e-6)

    @pytest.mark.parametrize("bad_h", [2.0, 400.0])
    def test_a_measured_top_outside_5_to_350_mm_falls_back_to_the_fit_plane(self, bad_h):
        """The window PoseExtractor accepts a measured top in; outside it the ray
        is cut on yaw_top_height_mm."""
        obj = _detection(560.0, -60.0, 41.5, depth_mm=650.0)
        obj["h_depth_mm"] = bad_h
        assert top_centre_xy(obj, T_BC, PLANE) == pytest.approx((560.0, -60.0), abs=1e-6)

    def test_the_flags_do_not_depend_on_the_depth_mode(self):
        called = _called(stack_on="metal_box", x=525.0, y=80.0, yaw=-40.0)
        flags = []
        for depth_mm in (None, 600.0, 650.0, 700.0):
            obj = _detection(560.0, 85.0, 41.5, depth_mm=depth_mm,
                             yaw_deg=_yaw_for(-40.0, 25.0))
            flags.append([f.text for f in check_placement(
                called, obj, T_BC=T_BC, table_plane=PLANE, class_sizes=SIZES)])
        assert all(f == flags[0] for f in flags) and len(flags[0]) == 2


class TestYawHelpers:
    def test_offset_and_wrap(self):
        assert yaw_error_deg(-60.0, 30.0) == pytest.approx(0.0)
        assert yaw_error_deg(-60.0, 30.0 + 180.0) == pytest.approx(0.0)
        assert yaw_error_deg(10.0, 30.0) == pytest.approx(70.0)
        assert yaw_error_deg(10.0, 30.0, 90.0) == pytest.approx(-20.0)

    def test_period_from_the_footprint(self):
        assert yaw_period_deg("wood_box", SIZES) == 90.0
        assert yaw_period_deg("carton", SIZES) == 180.0
        assert yaw_period_deg("unknown", SIZES) == 180.0


class TestJudgedDetection:
    """rgbd drops a detection whose mask returns fewer than MIN_DEPTH_PIXELS valid
    depth pixels; plane and fusion keep it. The check passes over such detections,
    so it judges the same one in every mode."""

    def test_min_depth_pixels_is_the_rule_of_masked_depth(self):
        from src.perception.postprocess import masked_depth
        mask = np.ones((10, 10), np.uint8)
        depth = np.zeros((10, 10), np.float32)
        depth.flat[:MIN_DEPTH_PIXELS - 1] = 0.6
        assert masked_depth(depth, mask) is None
        depth.flat[:MIN_DEPTH_PIXELS] = 0.6
        assert masked_depth(depth, mask) == pytest.approx(0.6)

    def test_has_depth(self):
        assert not has_depth({"depth_valid_frac": 4 / 15000, "mask_area": 15000})
        assert has_depth({"depth_valid_frac": MIN_DEPTH_PIXELS / 15000, "mask_area": 15000})
        assert has_depth({"depth_valid_frac": 0.206, "mask_area": 15245})  # session 1 low
        assert has_depth({"class_name": "inox_box"})            # not known: kept

    def test_the_first_detection_with_depth_is_judged(self):
        blind = {"class_name": "inox_box", "depth_valid_frac": 0.0, "mask_area": 15000}
        seen = {"class_name": "inox_box", "depth_valid_frac": 0.4, "mask_area": 15000}
        other = {"class_name": "inox_box", "depth_valid_frac": 0.9, "mask_area": 15000}
        assert judged_detection([blind, seen, other]) is seen
        assert judged_detection([seen, blind]) is seen
        assert judged_detection([blind]) is None
        assert judged_detection([]) is None and judged_detection(None) is None


class TestConstants:
    def test_near_square_ratio_is_the_perception_rule(self):
        from src.perception.depth_modes import NEAR_SQUARE_RATIO as PERCEPTION_RATIO
        assert NEAR_SQUARE_RATIO == PERCEPTION_RATIO

    def test_heights_are_those_measured_flat(self):
        """14 and 15 Oct 2026 (tests/test_real_cycle.FLAT_MM)."""
        from tests.test_real_cycle import FLAT_MM
        assert FLAT_TOP_MM == {k: v for k, v in FLAT_MM.items() if not k.startswith("carton")}
        assert CARTON_TOP_MM == {80: FLAT_MM["carton_80"], 120: FLAT_MM["carton_120"]}

    def test_checks_are_limited(self):
        assert MAX_CHECKS == 3


# ── session 1 rebuilt from its logged rows ─────────────────────────────────────

POSES = {r["pose_id"]: r for r in csv.DictReader(
    (ROOT / "config/pose_lists/std_v2.csv").open(encoding="utf-8"))}


def _logged(cls, x, y, z, yaw_deg, frac, part_h, h_depth):
    """The detection fields the check reads, from one logged row of session 1.

    pose_camera only needs to lie on the viewing ray, which the logged commanded
    point does in every mode; yaw_top_height_mm, not logged, follows
    PoseExtractor's rule (measured top when depth is trusted, else part height).
    """
    p = np.linalg.inv(T_BC) @ np.array([x, y, z, 1.0])
    trusted = h_depth is not None and frac >= 0.3 and 5.0 <= h_depth <= 350.0
    return {"class_name": cls, "pose_camera": (p[0], p[1], p[2], 0.0),
            "yaw_top_height_mm": h_depth if trusted else part_h,
            "h_depth_mm": h_depth, "yaw_base": math.radians(yaw_deg)}


# arm, pose, class, det x, y, z, yaw, valid fraction, part height, top band; flags
SESSION1 = [
    # verified staging deviations
    ("fusion", "P0016", "inox_box", 575.7, -38.8, 665.9, -5.7, 0.413, 40.0, 74.0, ["stack"]),
    ("plane", "P0021", "inox_box", 577.0, 194.4, 631.5, -8.0, 0.73, 40.0, 41.5, ["stack"]),
    ("rgbd", "P0036", "metal_box", 526.1, -168.4, 677.0, -61.7, 0.998, 85.0, 85.4, ["stack"]),
    ("rgbd", "P0062", "inox_box", 523.2, -107.7, 594.6, 82.1, 0.227, 40.0, 1.3, ["stack"]),
    ("plane", "P0001", "carton", 578.2, -159.0, 712.1, 82.0, 0.949, 120.0, 124.0, ["stack"]),
    ("plane", "P0057", "carton", 578.6, -52.7, 711.9, 17.2, 0.98, 120.0, 125.4, ["stack"]),
    ("rgbd", "P0097", "carton", 643.4, 0.1, 717.0, 55.7, 0.994, 120.0, 124.7, ["stack"]),
    ("rgbd", "P0002", "plastic_box", 575.3, -169.9, 622.0, -15.7, 0.999, 30.0, 29.7,
     ["position"]),
    ("rgbd", "P0044", "metal_box", 526.2, 42.9, 677.6, -81.7, 0.997, 85.0, 85.8, ["position"]),
    ("plane", "P0033", "carton", 915.5, 138.2, 713.4, -88.9, 0.998, 120.0, 151.8,
     ["position", "stack", "yaw"]),
    ("fusion", "P0001", "carton", 562.2, -167.4, 747.9, 13.8, 0.91, 120.0, 155.9, ["yaw"]),
    ("fusion", "P0034", "plastic_box", 639.5, 70.9, 622.0, -32.5, 0.999, 30.0, 29.7, ["yaw"]),
    ("rgbd", "P0064", "inox_box", 651.1, -42.4, 633.3, 64.9, 0.436, 40.0, 41.6, ["yaw"]),
    # genuine trials nearest a threshold
    ("plane", "P0079", "inox_box", 586.7, 50.8, 631.8, 83.5, 0.681, 40.0, 40.9, []),
    ("rgbd", "P0047", "inox_box", 567.4, -49.4, 780.3, 34.3, 0.258, 40.0, 191.9, []),
    ("plane", "P0047", "inox_box", 536.5, -58.4, 631.7, 32.4, 0.282, 40.0, 193.9, []),
    ("rgbd", "P0207", "inox_box", 576.5, -165.2, 592.0, 85.0, 0.461, 40.0, 31.6, []),
    ("rgbd", "P0207", "inox_box", 573.4, -161.9, 594.0, 85.7, 0.306, 40.0, None, []),
    ("plane", "P0086", "inox_box", 528.5, -43.0, 631.6, 22.1, 0.206, 40.0, 31.9, []),
    ("plane", "P0005", "inox_box", 522.9, 131.8, 631.3, -23.1, 0.951, 40.0, 40.6, []),
    ("fusion", "P0006", "inox_box", 512.7, 146.1, 664.8, 8.2, 0.952, 40.0, 73.6, []),
    ("fusion", "P0075", "wood_box", 623.9, 58.0, 742.0, 39.2, 0.999, 150.0, 153.4, []),
    ("rgbd", "P0087", "inox_box", 517.2, -107.7, 786.6, 31.5, 0.436, 40.0, 195.3, []),
    ("plane", "P0006", "inox_box", 507.0, 164.5, 631.2, 5.1, 0.954, 40.0, 74.7, []),
    # the lowest matte readings of the session, 1.6 and 1.4 mm under the wood box
    ("rgbd", "P0203", "wood_box", 542.7, -45.8, 743.2, -20.3, 0.989, 150.0, 151.5, []),
    ("fusion", "P0011", "wood_box", 521.7, 43.1, 827.8, 8.5, 0.979, 150.0, 236.4, []),
]


@pytest.mark.parametrize("arm,pose,cls,x,y,z,yaw,frac,part_h,h_depth,expected", SESSION1,
                         ids=[f"{r[0]}-{r[1]}" for r in SESSION1])
def test_session1_trial(arm, pose, cls, x, y, z, yaw, frac, part_h, h_depth, expected):
    """No carton size was called in session 1, so either size is accepted."""
    called = called_pose_from_row(POSES[pose], None)
    obj = _logged(cls, x, y, z, yaw, frac, part_h, h_depth)
    assert _flags(called, obj) == expected


def test_session1_mirror_top_read_low_on_its_support_is_flagged():
    """Plane P0062: the inox stood on the plastic box as called (frame checked), but
    its mirror top read 30.9 mm, as an inox with no support would. The check cannot
    tell the two apart and flags it; the operator goes on with one ENTER and the
    trial keeps its detection (it was refused by the shape guard)."""
    called = called_pose_from_row(POSES["P0062"], None)
    obj = _logged("inox_box", 519.1, -110.0, 631.7, 85.7, 0.358, 40.0, 30.9)
    assert _flags(called, obj) == ["stack"]


def test_session1_class_miss():
    """Rgbd P0017: an inox placed for the carton, the class lock found no carton."""
    called = called_pose_from_row(POSES["P0017"], None)
    assert _flags(called, None) == ["class"]


# ── carton sizes per pose ──────────────────────────────────────────────────────

def test_class_label():
    assert class_label("carton", 120) == "carton(120)"
    assert class_label("carton", None) == "carton"
    assert class_label("inox_box", 120) == "inox_box"


def test_called_pose_carries_the_carton_sizes():
    row = {"pose_id": "P0021", "card_id": "20", "x_mm": "575.0", "y_mm": "200.0",
           "yaw_deg": "85.0", "class_hint": "inox_box", "stack_on": "carton"}
    called = called_pose_from_row(row, {"P0021": (None, 120)})
    assert called.stack_on_carton_mm == 120 and called.carton_mm is None
    assert called.x_mm == 575.0 and called.yaw_deg == 85.0


def test_sidecar_path():
    assert carton_sizes_path("config/pose_lists/std_v2.csv").name == "std_v2_carton.csv"


def test_load_rejects_a_size_that_is_not_a_carton(tmp_path):
    p = tmp_path / "x_carton.csv"
    p.write_text("pose_id,carton_mm,stack_on_carton_mm\nP0001,100,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected one of"):
        load_carton_sizes(p)


def test_load_rejects_a_pose_listed_twice(tmp_path):
    p = tmp_path / "x_carton.csv"
    p.write_text("pose_id,carton_mm,stack_on_carton_mm\nP0001,80,\nP0001,120,\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="twice"):
        load_carton_sizes(p)


def test_missing_sizes_are_listed():
    rows = [{"pose_id": "P1", "class_hint": "carton", "stack_on": ""},
            {"pose_id": "P2", "class_hint": "inox_box", "stack_on": "carton"},
            {"pose_id": "P3", "class_hint": "inox_box", "stack_on": ""}]
    assert missing_carton_sizes(rows, {}) == ["P1", "P2"]
    assert missing_carton_sizes(rows, {"P1": (80, None), "P2": (None, 120)}) == []
