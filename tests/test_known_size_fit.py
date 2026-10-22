"""
test_known_size_fit.py
──────────────────────
The grasp pose from the class's known top size: depth_modes.fit_footprint, as
PoseExtractor.extract uses it.

21 Oct 2026, kiemtra-2110, trial 3 (pose P0216): the detector's mask of the
mirror-finish inox box (180 x 85 x 40 mm) also covered the paper position cards around
one corner of it, about 31 percent of the top's area beyond the far end and the far
side. The minimum-area rectangle of the outline turned 6.3 deg, the mask centroid
moved 10.5 mm along and 15.3 mm across the long side, and a jaw came down on the box
corner. The rectangle of the known size, fitted by IoU, replaces both.

Scenes are ray-cast through the cell calibration (test_stacked_yaw.render), and the
extractor gets the class tables of the real block of config/experiment.yaml, as
03_run_experiment builds it.
"""
from __future__ import annotations

import csv
import math
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.logging.logger import CONTEXT_FIELDNAMES, TrialLogger
from src.orchestrator.orchestrator import Orchestrator
from src.perception.depth_modes import (
    MIN_FOOTPRINT_IOU,
    fit_footprint,
    project_points,
    ray_plane_depth,
)
from src.perception.postprocess import (
    PoseExtractor,
    _metric_edge_yaw_base,
    deproject_pixel,
    mask_centroid,
)
from tests.test_stacked_yaw import (
    INOX,
    INTR,
    LARGE_CARTON,
    METAL,
    PLANE,
    PLASTIC,
    SMALL_CARTON,
    T,
    WOOD,
    axis_err_deg,
    extract,
    heights,
    render,
)

MODES = ["rgbd", "plane", "fusion"]


def base_xy(det):
    """Grasp point of a detection in the base frame (x, y), mm."""
    p = T @ np.array([*det["pose_camera"][:3], 1.0])
    return p[:2]


def xy_err(det, center):
    p = base_xy(det)
    return math.hypot(p[0] - center[0], p[1] - center[1])


def centroid_err(mask, center, height_mm):
    """How far the ray through the mask centroid meets the top plane from the centre."""
    u, v = mask_centroid(mask.astype(np.uint8))
    z = ray_plane_depth(INTR, u, v, T, PLANE, height_mm)
    p = T[:3, :3] @ deproject_pixel(INTR, u, v, z) + T[:3, 3]
    return math.hypot(p[0] - center[0], p[1] - center[1])


def corner_spill(mask, center, yaw_deg, size, semi=(60.0, 30.0), reach=15.0, angle_deg=30.0):
    """The mask and a lobe of table out of one corner of the part, the way the
    detector's masks reached onto the paper cards on 21 Oct 2026: an ellipse on the
    table with semi-axes semi (mm), centred reach mm out of the (+L, -W) corner along a
    direction angle_deg off the long side, its long axis along that direction."""
    th = math.radians(yaw_deg)
    e = np.array([math.cos(th), math.sin(th)])
    n = np.array([-e[1], e[0]])
    corner = np.asarray(center) + e * size[0] / 2.0 - n * size[1] / 2.0
    a = math.radians(angle_deg)
    d = math.cos(a) * e - math.sin(a) * n
    c = corner + reach * d
    t = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
    xy = c + np.outer(semi[0] * np.cos(t), d) + np.outer(semi[1] * np.sin(t), [-d[1], d[0]])
    px = project_points(INTR, T, np.column_stack([xy, PLANE.z_at(xy[:, 0], xy[:, 1])]))
    out = mask.astype(np.uint8).copy()
    cv2.fillPoly(out, [np.round(px * 16).astype(np.int32)], 1, shift=4)
    return out > 0


def end_spill(mask, center, yaw_deg, size, beyond=50.0, extra_width=30.0):
    """The mask and a band of table past the part's +L end, wider than the part, the
    way the inox mask of 22 Oct 2026 (kiemtra-2210 first run, trial 3) ran about 50 mm
    past the box's end over the paper cards."""
    th = math.radians(yaw_deg)
    e = np.array([math.cos(th), math.sin(th)])
    n = np.array([-e[1], e[0]])
    c = np.asarray(center) + e * (size[0] / 2.0 + beyond / 2.0 - 5.0)
    half_l, half_w = beyond / 2.0 + 5.0, (size[1] + extra_width) / 2.0
    xy = np.array([c + sl * half_l * e + sw * half_w * n
                   for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    px = project_points(INTR, T, np.column_stack([xy, PLANE.z_at(xy[:, 0], xy[:, 1])]))
    out = mask.astype(np.uint8).copy()
    cv2.fillPoly(out, [np.round(px * 16).astype(np.int32)], 1, shift=4)
    return out > 0


class TestPlateau:
    """How far the fitted rectangle slides along its long side at the same IoU."""

    def test_a_clean_mask_pins_the_rectangle(self):
        center, yaw = (575.0, 20.0), -85.0
        labels, _ = render(((center, yaw, INOX, 0.0),))
        fit = fit_footprint(labels > 0, INTR, T, PLANE, INOX[2], INOX[:2],
                            math.radians(yaw), bottom_height_mm=0.0)
        assert fit.plateau_mm < 3.0

    def test_a_wide_spill_past_the_end_lets_it_slide(self):
        center, yaw = (575.0, 20.0), -85.0
        labels, _ = render(((center, yaw, INOX, 0.0),))
        mask = end_spill(labels > 0, center, yaw, INOX)
        fit = fit_footprint(mask, INTR, T, PLANE, INOX[2], INOX[:2],
                            math.radians(yaw), bottom_height_mm=0.0)
        assert fit.plateau_mm > 10.0

    def test_the_slide_is_recorded_with_the_fit(self):
        center, yaw = (575.0, 20.0), -85.0
        labels, depth = render(((center, yaw, INOX, 0.0),))
        det = extract(labels > 0, depth, "inox_box")
        assert det["fit_plateau_mm"] < 3.0


SPILL_POSES = [pytest.param((575.0, 20.0), -82.7, id="P0216_yaw"),
               pytest.param((625.0, -40.0), 65.0, id="card_65")]


class TestCornerSpill:
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("center,yaw", SPILL_POSES)
    def test_the_grasp_stays_on_the_part(self, center, yaw, mode):
        labels, depth = render(((center, yaw, INOX, 0.0),))
        mask = corner_spill(labels > 0, center, yaw, INOX)
        # The defect, as on P0216: the rectangle turns and the centroid moves.
        rect = _metric_edge_yaw_base(mask, INTR, T, PLANE, INOX[2])
        assert axis_err_deg(rect["yaw_base"], yaw) > 3.0
        assert centroid_err(mask, center, INOX[2]) > 10.0
        det = extract(mask, depth, "inox_box", mode)
        assert det["yaw_source"] == "known_size_fit"
        assert det["fit_iou"] >= MIN_FOOTPRINT_IOU
        assert axis_err_deg(det["yaw_base"], yaw) < 1.0
        # The grasp pixel feeds every mode's position: plane and fusion through the
        # ray, rgbd through the deprojection at the median depth.
        assert xy_err(det, center) < 3.0
        # The rectangle the long-side check reads is still the outline's.
        rect = _metric_edge_yaw_base(mask, INTR, T, PLANE, det["h_depth_mm"])
        assert det["yaw_rect_long_mm"] == rect["long_mm"]
        assert det["yaw_rect_short_mm"] == rect["short_mm"]

    def test_the_median_height_is_read_at_the_grasp_pixel(self):
        center, yaw = (575.0, 20.0), -82.7
        labels, depth = render(((center, yaw, INOX, 0.0),))
        mask = corner_spill(labels > 0, center, yaw, INOX)
        det = extract(mask, depth, "inox_box")
        u, v = det["pixel_uv"]
        vals = depth[mask]
        p = T[:3, :3] @ deproject_pixel(INTR, u, v, float(np.median(vals[vals > 0]))) + T[:3, 3]
        assert det["h_median_mm"] == pytest.approx(float(PLANE.height_above(p)), abs=1e-6)

    @pytest.mark.parametrize("center,yaw", SPILL_POSES)
    def test_the_three_modes_get_the_same_pixel_and_yaw(self, center, yaw):
        labels, depth = render(((center, yaw, INOX, 0.0),))
        mask = corner_spill(labels > 0, center, yaw, INOX)
        dets = [extract(mask, depth, "inox_box", m) for m in MODES]
        for d in dets[1:]:
            for key in ("yaw_base", "pixel_uv", "fit_iou", "yaw_source"):
                assert d[key] == dets[0][key]


CLEAN = [  # class, size, footprint centre, yaw_deg; one pose near the nadir, one 192 mm out
    (cls, size, center, yaw)
    for cls, size in (("carton", LARGE_CARTON), ("carton", SMALL_CARTON),
                      ("plastic_box", PLASTIC), ("wood_box", WOOD),
                      ("metal_box", METAL), ("inox_box", INOX))
    for center, yaw in (((575.0, 20.0), 25.0), ((450.0, 0.0), -50.0))
]


class TestCleanMasks:
    @pytest.mark.parametrize("cls,size,center,yaw", CLEAN,
                             ids=[f"{c}_{s[2]:g}_{y:g}" for c, s, _, y in CLEAN])
    def test_every_class_and_size(self, cls, size, center, yaw):
        labels, depth = render(((center, yaw, size, 0.0),))
        det = extract(labels > 0, depth, cls, "plane")
        assert det["yaw_source"] == "known_size_fit"
        err = (math.degrees(det["yaw_base"]) - yaw + 90.0) % 180.0 - 90.0
        if size[0] == size[1]:
            err = (err + 45.0) % 90.0 - 45.0
        assert abs(err) < 0.5
        assert xy_err(det, center) < 2.0

    @pytest.mark.parametrize("yaw", [20.0, 60.0, -70.0, 88.0, -44.0])
    def test_the_square_wood_box_folds_modulo_90(self, yaw):
        center = (525.0, -40.0)
        labels, depth = render(((center, yaw, WOOD, 0.0),))
        det = extract(labels > 0, depth, "wood_box", "plane")
        assert det["yaw_source"] == "known_size_fit"
        assert -math.pi / 4 < det["yaw_base"] <= math.pi / 4
        assert abs((math.degrees(det["yaw_base"]) - yaw + 45.0) % 90.0 - 45.0) < 0.5
        assert xy_err(det, center) < 2.0


class TestFitFunction:
    def test_side_faces_are_read_out_with_the_bottom_plane(self):
        # The large carton 262 mm from the camera's nadir: on the top plane its
        # visible side faces widen the mask on the nadir side, and the rectangle read
        # on that plane alone slides into them (5.9 mm off here).
        center, yaw = (430.0, -120.0), 65.0
        labels, _ = render(((center, yaw, LARGE_CARTON, 0.0),))
        mask = labels > 0
        seed = math.radians(yaw + 5.0)
        plain = fit_footprint(mask, INTR, T, PLANE, 120.0, LARGE_CARTON[:2], seed)
        both = fit_footprint(mask, INTR, T, PLANE, 120.0, LARGE_CARTON[:2], seed,
                             bottom_height_mm=0.0)
        assert math.hypot(plain.center_xy[0] - center[0], plain.center_xy[1] - center[1]) > 4.0
        assert math.hypot(both.center_xy[0] - center[0], both.center_xy[1] - center[1]) < 1.0
        assert both.iou > plain.iou

    def test_a_mask_without_its_side_faces_keeps_the_plain_reading(self):
        # On 21 Oct 2026 one metal box's mask covered 0.09 of its side faces. Read on
        # the bottom plane, the top face alone loses a strip; the plain reading
        # overlaps the rectangle more and is kept.
        center, yaw = (450.0, 0.0), -50.0
        labels, depth = render(((center, yaw, METAL, 0.0),))
        top = (labels > 0) & (heights(depth) > METAL[2] - 0.5)
        fit = fit_footprint(top, INTR, T, PLANE, METAL[2], METAL[:2], math.radians(yaw),
                            bottom_height_mm=0.0)
        assert math.hypot(fit.center_xy[0] - center[0], fit.center_xy[1] - center[1]) < 2.0
        assert axis_err_deg(fit.yaw_base, yaw) < 0.5

    def test_the_pixel_is_the_centre_on_the_top_plane(self):
        center, yaw = (625.0, -160.0), 10.0
        labels, _ = render(((center, yaw, INOX, 0.0),))
        fit = fit_footprint(labels > 0, INTR, T, PLANE, 40.0, INOX[:2], math.radians(yaw))
        cx, cy = fit.center_xy
        top = np.array([[cx, cy, PLANE.z_at(cx, cy) + 40.0 / PLANE.normal[2]]])
        assert project_points(INTR, T, top)[0] == pytest.approx(fit.pixel_uv, abs=1e-6)

    @pytest.mark.parametrize("yaw", [89.8, -89.8, 0.2])
    def test_the_long_side_yaw_lies_in_minus_90_to_90(self, yaw):
        labels, _ = render((((575.0, 20.0), yaw, METAL, 0.0),))
        fit = fit_footprint(labels > 0, INTR, T, PLANE, METAL[2], METAL[:2],
                            math.radians(yaw), bottom_height_mm=0.0)
        assert -math.pi / 2 < fit.yaw_base <= math.pi / 2
        assert axis_err_deg(fit.yaw_base, yaw) < 0.5

    def test_an_empty_mask_gives_nothing(self):
        empty = np.zeros((int(INTR["height"]), int(INTR["width"])), np.uint8)
        assert fit_footprint(empty, INTR, T, PLANE, 40.0, INOX[:2], 0.0) is None

    def test_time_per_fit(self):
        # The wood box is the slowest class: a square footprint, its yaw searched over
        # the whole 90 deg, the largest grids. On its 27 real masks of 20 and 21 Oct
        # 2026 the fit took 18.9 ms at the median and 21.2 ms at most on an i7-8650U;
        # the bar is generous so that a loaded machine does not fail it.
        labels, _ = render((((525.0, -40.0), 20.0, WOOD, 0.0),))
        mask = labels > 0
        fit_footprint(mask, INTR, T, PLANE, WOOD[2], WOOD[:2], 0.0, bottom_height_mm=0.0)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            fit_footprint(mask, INTR, T, PLANE, WOOD[2], WOOD[:2], 0.0, bottom_height_mm=0.0)
            times.append(time.perf_counter() - t0)
        assert float(np.median(times)) < 0.15


class TestFallback:
    def test_a_class_without_a_size_keeps_the_centroid_and_rectangle(self):
        center, yaw = (575.0, 20.0), -82.7
        labels, depth = render(((center, yaw, INOX, 0.0),))
        mask = corner_spill(labels > 0, center, yaw, INOX).astype(np.uint8)
        ex = PoseExtractor(INTR, depth_mode="plane", T_BC_mm=T, table_plane=PLANE,
                           class_heights_mm={"inox_box": 40.0})
        det = ex.extract({"mask": mask, "class_name": "inox_box", "height_mm": 40.0}, depth)
        assert "fit_iou" not in det
        assert det["yaw_source"] == "metric_base_minAreaRect"
        assert det["pixel_uv"] == mask_centroid(mask)
        rect = _metric_edge_yaw_base(mask, INTR, T, PLANE, det["h_depth_mm"])
        assert det["yaw_base"] == rect["yaw_base"]

    def test_a_fit_below_the_bar_is_recorded_but_not_used(self):
        # Most of the part hidden: the rectangle shares less than half its union with
        # what shows.
        center, yaw = (575.0, 20.0), 0.0
        labels, depth = render(((center, yaw, METAL, 0.0),))
        mask = (labels > 0).astype(np.uint8)
        xs = np.nonzero(mask)[1]
        mask[:, :int(xs.min() + 0.6 * (xs.max() - xs.min()))] = 0
        det = extract(mask, depth, "metal_box")
        assert det["fit_iou"] < MIN_FOOTPRINT_IOU
        assert det["yaw_source"] == "metric_base_minAreaRect"
        assert det["pixel_uv"] == mask_centroid(mask)


class TestLogged:
    def test_the_column_travels_to_the_csv(self, tmp_path):
        assert "fit_iou" in CONTEXT_FIELDNAMES
        p = tmp_path / "trials.csv"
        TrialLogger(p).log_trial(trial_id=1, success=True, class_name="inox_box",
                                 extra={"fit_iou": 0.718})
        assert list(csv.DictReader(p.open(newline="", encoding="utf-8")))[0]["fit_iou"] == "0.718"

    def test_the_trial_log_takes_it_from_the_detection(self, tmp_path):
        o = Orchestrator.__new__(Orchestrator)     # only what _log_trial touches
        o.trial_logger = TrialLogger(tmp_path / "run.csv")
        o._last_frame_path = ""
        o.current_pose_id = "P0216"
        o.current_condition = o.current_stack_on = ""
        o.session_id = o.block_id = o.operator_id = ""
        o.sm = SimpleNamespace(state=SimpleNamespace(value="done"))
        o.human_score_hook = None
        o._last_support_height_mm = 0.0
        o._last_place_z_mm = None
        o._log_trial(1, True, "", time.time(), {"class_name": "inox_box", "fit_iou": 0.71794})
        o._log_trial(2, True, "", time.time(), {"class_name": "wood_box"})
        rows = list(csv.DictReader((tmp_path / "run.csv").open(newline="", encoding="utf-8")))
        assert [r["fit_iou"] for r in rows] == ["0.718", ""]
