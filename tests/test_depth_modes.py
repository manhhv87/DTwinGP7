"""
test_depth_modes.py
───────────────────
Contribution C4: the plane and fusion depth modes, and telling the two carton
sizes apart from the silhouette.

Scenes are ray-cast from the box geometry (an independent renderer, not the
projection under test) through a camera 710 mm above the table, as in the cell.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from src.logging.logger import CONTEXT_FIELDNAMES, TrialLogger
from src.orchestrator.coord_conv import camera_yaw_to_base
from src.orchestrator.preflight import TABLE_PLANE_FILE, check_depth_mode, meta_path_for
from src.perception.depth_modes import (
    TablePlane,
    box_corners,
    fit_box,
    fuse_height,
    image_yaw_to_base,
    parse_class_sizes,
    project_points,
    ray_plane_depth,
    resolve_size,
    top_face_height,
    valid_depth_fraction,
)
from src.perception.depth_modes import MIN_SIZE_IOU
from src.perception.postprocess import (
    PoseExtractor,
    deproject_pixel,
    mask_pca_yaw,
    masked_depth,
    z_from_ground_plane,
)

ROOT = Path(__file__).resolve().parent.parent

# Half the cell's resolution keeps the ray caster quick; 327.5 px is the cell's
# 655 px focal length at 1280 x 720.
INTR = {"fx": 327.5, "fy": 327.5, "ppx": 320.0, "ppy": 180.0, "width": 640, "height": 360}
SHAPE = (360, 640)
# Eye-to-hand camera looking straight down, 710 mm above the table.
T_BC = np.array([[0.0, 1.0, 0.0, 518.0],
                 [1.0, 0.0, 0.0, 10.0],
                 [0.0, 0.0, -1.0, 1645.0],
                 [0.0, 0.0, 0.0, 1.0]])
TABLE_Z = 935.0
LEVEL = TablePlane(0.0, 0.0, TABLE_Z)
AXIS_XY = (518.0, 10.0)                      # where the optical axis meets the table
CARTON = [(180.0, 120.0, 120.0), (180.0, 100.0, 80.0)]
LARGE, SMALL = CARTON
PAPER_CLASSES = ["carton", "plastic_box", "wood_box", "metal_box", "inox_box"]


def render_box(center_xy, yaw_rad, size_mm, plane_z=TABLE_Z):
    """Ray-cast one box standing on a level table: (mask, depth in metres).

    Every pixel's ray is intersected with the box's own slabs, so the scene owes
    nothing to the projection the fit uses.
    """
    h, w = SHAPE
    vv, uu = np.mgrid[0:h, 0:w]
    d_cam = np.stack([(uu - INTR["ppx"]) / INTR["fx"],
                      (vv - INTR["ppy"]) / INTR["fy"], np.ones((h, w))], axis=-1)
    R, t = T_BC[:3, :3], T_BC[:3, 3]
    r = d_cam @ R.T
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    M = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])       # base → box frame
    o = M @ (t - np.array([center_xy[0], center_xy[1], plane_z]))
    rl = r @ M.T
    rl = np.where(np.abs(rl) < 1e-9, 1e-9, rl)
    L, W, H = size_mm
    lo, hi = np.array([-L / 2, -W / 2, 0.0]), np.array([L / 2, W / 2, H])
    t1, t2 = (lo - o) / rl, (hi - o) / rl
    tmin = np.max(np.minimum(t1, t2), axis=-1)
    tmax = np.min(np.maximum(t1, t2), axis=-1)
    hit = (tmax >= tmin) & (tmin > 0)
    depth_mm = np.where(hit, tmin, (plane_z - t[2]) / r[..., 2])
    return hit.astype(np.uint8), (depth_mm / 1000.0).astype(np.float32)


def base_xyz(pose_camera):
    """Camera-frame pose (mm) of a detection → base frame."""
    return (T_BC @ np.array([pose_camera[0], pose_camera[1], pose_camera[2], 1.0]))[:3]


class TestRayPlaneDepth:
    @pytest.mark.parametrize("u,v", [(320.0, 180.0), (120.0, 300.0), (560.0, 40.0)])
    @pytest.mark.parametrize("h", [0.0, 40.0, 150.0])
    def test_matches_ground_plane_when_level(self, u, v, h):
        # On a level table the generalised form must reproduce z_from_ground_plane.
        got = ray_plane_depth(INTR, u, v, T_BC, LEVEL, h)
        want = z_from_ground_plane(INTR, u, v, T_BC, TABLE_Z + h)
        assert got == pytest.approx(want, abs=1e-9)

    @pytest.mark.parametrize("u,v", [(320.0, 180.0), (100.0, 60.0), (600.0, 330.0)])
    def test_point_lands_h_above_a_tilted_plane(self, u, v):
        plane = TablePlane(0.01, -0.006, 930.0)
        z = ray_plane_depth(INTR, u, v, T_BC, plane, 85.0)
        p = base_xyz(deproject_pixel(INTR, u, v, z))
        assert plane.height_above(p) == pytest.approx(85.0, abs=1e-6)

    def test_parallel_ray_returns_none(self):
        T = np.eye(4)
        T[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        T[:3, 3] = [700.0, 0.0, 1200.0]
        assert ray_plane_depth(INTR, 320.0, 180.0, T, LEVEL, 0.0) is None

    def test_plane_behind_the_camera_returns_none(self):
        above = TablePlane(0.0, 0.0, 1800.0)          # above a camera at z = 1645
        assert ray_plane_depth(INTR, 320.0, 180.0, T_BC, above, 0.0) is None

    @pytest.mark.parametrize("deg", [0, 25, -40, 90])
    def test_image_yaw_matches_coord_conv(self, deg):
        # coord_conv is the single source of truth for the yaw transform.
        y = math.radians(deg)
        assert image_yaw_to_base(y, T_BC) == pytest.approx(camera_yaw_to_base(y, T_BC))


class TestParseClassSizes:
    def test_single_triple_becomes_a_list(self):
        assert parse_class_sizes({"wood_box": [190, 190, 150]}) == {
            "wood_box": [(190.0, 190.0, 150.0)]}

    def test_two_candidates_kept(self):
        assert parse_class_sizes({"carton": [[180, 120, 120], [180, 100, 80]]})["carton"] == [
            (180.0, 120.0, 120.0), (180.0, 100.0, 80.0)]

    def test_none_is_empty(self):
        assert parse_class_sizes(None) == {}

    @pytest.mark.parametrize("bad", [{"x": [1, 2]}, {"x": [[1, 2, 3, 4]]}, {"x": []},
                                     {"x": [180, 100, 0]}, {"x": [-1, 2, 3]}])
    def test_bad_entries_raise(self, bad):
        with pytest.raises(ValueError):
            parse_class_sizes(bad)


class TestBoxGeometry:
    def test_corners_stand_on_the_plane(self):
        plane = TablePlane(0.008, -0.004, 930.0)
        C = box_corners((600.0, 40.0), 0.3, LARGE, plane)
        assert plane.height_above(C[:4]) == pytest.approx(0.0, abs=1e-9)
        assert plane.height_above(C[4:]) == pytest.approx(120.0, abs=1e-9)

    def test_projection_round_trip(self):
        # A point on the table projects to the pixel whose ray meets it again.
        p = np.array([[560.0, -60.0, TABLE_Z]])
        px = project_points(INTR, T_BC, p)[0]
        z = ray_plane_depth(INTR, px[0], px[1], T_BC, LEVEL, 0.0)
        assert base_xyz(deproject_pixel(INTR, px[0], px[1], z)) == pytest.approx(p[0], abs=1e-6)


POSES = [(AXIS_XY, 0.0), ((AXIS_XY[0] + 220, AXIS_XY[1] - 150), math.radians(35)),
         ((AXIS_XY[0] - 180, AXIS_XY[1] + 190), math.radians(-50)),
         ((AXIS_XY[0] + 40, AXIS_XY[1] + 260), math.radians(90))]


class TestSizeResolution:
    @pytest.mark.parametrize("center,yaw", POSES)
    @pytest.mark.parametrize("size", CARTON)
    def test_picks_the_rendered_size(self, center, yaw, size):
        mask, _ = render_box(center, yaw, size)
        fit, ious = resolve_size(mask, INTR, T_BC, LEVEL, CARTON, mask_pca_yaw(mask))
        assert fit.size == size, f"IoUs {ious}"
        assert fit.iou > 0.9
        assert math.hypot(fit.center_xy[0] - center[0], fit.center_xy[1] - center[1]) < 3.0

    def test_resolves_where_the_side_faces_make_the_mask_look_square(self):
        # 300 mm off the optical axis along its own long axis, the 120 mm end face
        # widens the silhouette until a 180 x 120 box is nearly square and the mask's
        # major axis can flip; the fit tries both hypotheses.
        center = (AXIS_XY[0], AXIS_XY[1] + 300.0)
        mask, _ = render_box(center, math.pi / 2, LARGE)
        fit, ious = resolve_size(mask, INTR, T_BC, LEVEL, CARTON, mask_pca_yaw(mask))
        assert fit.size == LARGE, f"IoUs {ious}"

    def test_fitted_centre_beats_the_centroid_ray(self):
        # The mask centroid sits between the top face and the visible side faces, so
        # the ray through it meets the top plane short of the true centre, by more
        # than the 13 to 18 mm the jaws leave around a part. The fit models the side
        # faces and does not have that bias.
        center = (AXIS_XY[0] + 40.0, AXIS_XY[1] + 260.0)
        mask, _ = render_box(center, math.pi / 2, LARGE)
        ys, xs = np.nonzero(mask)
        z = ray_plane_depth(INTR, xs.mean(), ys.mean(), T_BC, LEVEL, LARGE[2])
        p = base_xyz(deproject_pixel(INTR, xs.mean(), ys.mean(), z))
        centroid_err = math.hypot(p[0] - center[0], p[1] - center[1])
        fit = fit_box(mask, INTR, T_BC, LEVEL, LARGE, math.pi / 2)
        fit_err = math.hypot(fit.center_xy[0] - center[0], fit.center_xy[1] - center[1])
        assert fit_err < 3.0
        assert centroid_err > 8.0
        assert fit_err < centroid_err

    def test_fit_survives_a_part_cut_off_by_the_frame(self):
        # The frame cuts the mask, and its centroid with it. The modelled silhouette
        # is clipped the same way, so the fit holds; without clipping it lands 36 mm out.
        center = (AXIS_XY[0] + 300.0, AXIS_XY[1])
        mask, _ = render_box(center, 0.0, LARGE)
        assert np.nonzero(mask)[0].max() == SHAPE[0] - 1       # the part really is cut
        fit = fit_box(mask, INTR, T_BC, LEVEL, LARGE, 0.0)
        assert math.hypot(fit.center_xy[0] - center[0], fit.center_xy[1] - center[1]) < 3.0
        assert fit.iou > 0.95


class TestFusionRule:
    def test_prior_stands_when_depth_agrees(self):
        assert fuse_height(122.0, 120.0, 80.0, 0.95) == (120.0, "plane")

    def test_depth_wins_when_the_part_is_raised(self):
        assert fuse_height(270.0, 120.0, 80.0, 0.95) == (270.0, "rgbd")

    def test_depth_wins_when_the_other_carton_size_fits(self):
        assert fuse_height(80.0, 120.0, 80.0, 0.95) == (80.0, "rgbd")

    def test_dropout_keeps_the_prior(self):
        assert fuse_height(270.0, 120.0, 80.0, 0.2) == (120.0, "plane")

    def test_depth_below_the_part_keeps_the_prior(self):
        # A top face sunk into the table is impossible: that is dropout or edge bleed.
        assert fuse_height(2.0, 80.0, 80.0, 0.95) == (80.0, "plane")

    def test_no_measurement_keeps_the_prior(self):
        assert fuse_height(None, 80.0, 80.0, 0.95) == (80.0, "plane")


class TestTopFaceHeight:
    @pytest.mark.parametrize("size", CARTON)
    def test_measures_the_rendered_height(self, size):
        mask, depth = render_box((AXIS_XY[0] + 120, AXIS_XY[1] - 90), 0.3, size)
        assert top_face_height(depth, mask, INTR, T_BC, LEVEL) == pytest.approx(
            size[2], abs=1.0)

    def test_survives_a_mask_that_bleeds_onto_the_table(self):
        # A detector mask spills a little past the part's edge, onto the table. The
        # erosion keeps those pixels out, so the estimate does not follow them down.
        mask, depth = render_box(AXIS_XY, 0.0, SMALL)
        spilled = cv2.dilate(mask, np.ones((21, 21), np.uint8))
        assert top_face_height(depth, spilled, INTR, T_BC, LEVEL) == pytest.approx(
            80.0, abs=1.5)

    def test_no_depth_gives_nothing(self):
        mask, _ = render_box(AXIS_XY, 0.0, LARGE)
        assert top_face_height(np.zeros(SHAPE, np.float32), mask, INTR, T_BC, LEVEL) is None

    def test_scattered_returns_give_nothing(self):
        # A mirror finish returns depths that form no top face, and a scattered cloud
        # must not be allowed to override the plane prior.
        mask, depth = render_box(AXIS_XY, 0.0, LARGE)
        rng = np.random.default_rng(1)
        depth = depth.copy()
        depth[mask > 0] = rng.uniform(0.35, 0.71, int(np.count_nonzero(mask))).astype(np.float32)
        assert top_face_height(depth, mask, INTR, T_BC, LEVEL) is None


def _extractor(mode="plane", **kw):
    kw.setdefault("class_heights_mm", {"carton": 80.0, "wood_box": 150.0})
    kw.setdefault("class_sizes_mm", {"carton": [list(LARGE), list(SMALL)]})
    return PoseExtractor(INTR, depth_mode=mode, T_BC_mm=T_BC, table_plane=LEVEL, **kw)


class TestPoseExtractorModes:
    def test_constructor_rejects_an_unknown_mode(self):
        with pytest.raises(ValueError):
            PoseExtractor(INTR, depth_mode="lidar")

    def test_constructor_rejects_plane_without_geometry(self):
        with pytest.raises(ValueError):
            PoseExtractor(INTR, depth_mode="plane")
        with pytest.raises(ValueError):
            PoseExtractor(INTR, depth_mode="fusion", T_BC_mm=T_BC)

    def test_plane_mode_survives_total_dropout(self):
        mask, _ = render_box(AXIS_XY, 0.2, LARGE)
        blind = np.zeros(SHAPE, np.float32)
        det = _extractor("plane").extract({"mask": mask, "class_name": "carton"}, blind)
        assert det is not None and det["depth_used"] == "plane"
        assert det["depth_valid_frac"] == 0.0
        assert base_xyz(det["pose_camera"])[2] == pytest.approx(TABLE_Z + 120.0, abs=1e-6)

    def test_rgbd_mode_returns_nothing_without_depth(self):
        mask, _ = render_box(AXIS_XY, 0.2, LARGE)
        blind = np.zeros(SHAPE, np.float32)
        assert _extractor("rgbd").extract({"mask": mask, "class_name": "carton"}, blind) is None

    def test_rgbd_mode_still_reads_the_median_depth(self):
        mask, depth = render_box(AXIS_XY, 0.0, SMALL)
        det = _extractor("rgbd").extract({"mask": mask, "class_name": "carton"}, depth)
        assert det["depth_used"] == "rgbd"
        assert det["pose_camera"][2] == pytest.approx(masked_depth(depth, mask) * 1000.0, abs=1e-6)

    def test_plane_mode_needs_a_height(self):
        mask, depth = render_box(AXIS_XY, 0.0, SMALL)
        ex = PoseExtractor(INTR, depth_mode="plane", T_BC_mm=T_BC, table_plane=LEVEL)
        assert ex.extract({"mask": mask, "class_name": "unknown_part"}, depth) is None

    @pytest.mark.parametrize("size,height", [(LARGE, 120.0), (SMALL, 80.0)])
    def test_carton_size_resolved_during_extraction(self, size, height):
        mask, depth = render_box((AXIS_XY[0] + 150, AXIS_XY[1] - 80), math.radians(25), size)
        det = _extractor("plane").extract(
            {"mask": mask, "class_name": "carton", "height_mm": 80.0}, depth)
        assert det["size_mm"] == size
        assert det["height_mm"] == height
        assert det["size_iou"] > 0.9

    def test_fusion_keeps_the_prior_on_a_matte_single_layer(self):
        mask, depth = render_box(AXIS_XY, 0.1, SMALL)
        det = _extractor("fusion").extract({"mask": mask, "class_name": "carton"}, depth)
        assert det["depth_used"] == "plane"
        assert base_xyz(det["pose_camera"])[2] == pytest.approx(TABLE_Z + 80.0, abs=1e-6)

    def test_fusion_takes_depth_when_the_part_is_stacked(self):
        # The same carton resting on a 150 mm support: the plane prior is 150 mm low.
        mask, depth = render_box(AXIS_XY, 0.0, LARGE, plane_z=TABLE_Z + 150.0)
        det = _extractor("fusion").extract({"mask": mask, "class_name": "carton"}, depth)
        assert det["depth_used"] == "rgbd"
        assert det["h_depth_mm"] == pytest.approx(270.0, abs=2.0)
        assert base_xyz(det["pose_camera"])[2] == pytest.approx(TABLE_Z + 270.0, abs=2.0)
        # A lift is not a part height: 270 mm must never become the grasp-depth cap.
        assert det["height_mm"] in (80.0, 120.0)

    @staticmethod
    def _occluded_large_carton():
        """A large carton with 45% of it hidden, as a neighbouring box would hide it."""
        mask, depth = render_box(AXIS_XY, 0.0, LARGE)
        xs = np.nonzero(mask)[1]
        mask = mask.copy()
        mask[:, :int(xs.min() + 0.45 * (xs.max() - xs.min()))] = 0
        return mask, depth

    def test_occluded_part_falls_back_to_the_lowest_size(self):
        # The silhouette match means little once another part hides a piece of this one,
        # and assuming a part is taller than it is would put the jaws above it.
        mask, depth = self._occluded_large_carton()
        det = _extractor("plane").extract({"mask": mask, "class_name": "carton"}, depth)
        assert det["size_iou"] < MIN_SIZE_IOU
        assert det["height_mm"] == 80.0

    def test_fusion_recovers_the_size_the_silhouette_could_not_read(self):
        # The same occluded part: what shows of its top face still stands at 120 mm, so
        # fusion corrects the pose and the part height together.
        mask, depth = self._occluded_large_carton()
        det = _extractor("fusion").extract({"mask": mask, "class_name": "carton"}, depth)
        assert det["depth_used"] == "rgbd"
        assert det["height_mm"] == 120.0
        assert det["size_mm"] == LARGE
        assert base_xyz(det["pose_camera"])[2] == pytest.approx(TABLE_Z + 120.0, abs=2.0)

    def test_fusion_falls_back_to_plane_on_dropout(self):
        mask, depth = render_box(AXIS_XY, 0.0, LARGE, plane_z=TABLE_Z + 150.0)
        rng = np.random.default_rng(0)
        holes = (mask > 0) & (rng.random(SHAPE) < 0.8)
        depth = depth.copy()
        depth[holes] = 0.0                       # a mirror finish returns nothing
        det = _extractor("fusion").extract({"mask": mask, "class_name": "carton"}, depth)
        assert det["depth_valid_frac"] < 0.3
        assert det["depth_used"] == "plane"

    def test_fusion_ignores_depth_that_sinks_the_part(self):
        mask, depth = render_box(AXIS_XY, 0.0, SMALL)
        depth = depth.copy()
        depth[mask > 0] = 0.710                  # reads the table through the part
        det = _extractor("fusion").extract({"mask": mask, "class_name": "carton"}, depth)
        assert det["depth_used"] == "plane"

    def test_valid_depth_fraction(self):
        mask, depth = render_box(AXIS_XY, 0.0, SMALL)
        assert valid_depth_fraction(depth, mask) == 1.0
        assert valid_depth_fraction(np.zeros(SHAPE, np.float32), mask) == 0.0


class TestPerceptionNodeExtractor:
    def _node(self, extractor):
        import queue as q

        from src.perception import MockCamera, MockDetector, PerceptionNode

        camera = MockCamera(depth_frames=[np.zeros(SHAPE, np.float32)], intrinsics=dict(INTR))
        det = MockDetector.make_detection("tray", mask_hw=SHAPE, mask_box=(300, 150, 380, 210))
        return PerceptionNode(camera, MockDetector(scripted=[[det]]), q.Queue(maxsize=2),
                              extractor=extractor)

    def test_injected_extractor_is_used(self):
        ex = PoseExtractor(INTR, depth_mode="plane", T_BC_mm=T_BC, table_plane=LEVEL,
                           class_heights_mm={"tray": 25.0})
        msg = self._node(ex).process_once()
        assert msg["objects"] and msg["objects"][0]["depth_used"] == "plane"

    def test_default_is_the_rgbd_baseline(self):
        node = self._node(None)
        assert node.extractor.depth_mode == "rgbd"
        assert node.process_once()["objects"] == []      # no depth, no pose


REAL_T = np.array([[0.0, 1.0, 0.0, 518.0],
                   [1.0, 0.0, 0.0, 10.0],
                   [0.0, 0.0, -1.0, 1645.0],
                   [0.0, 0.0, 0.0, 1.0]])
SIZES = {"carton": [[180, 120, 120], [180, 100, 80]], "wood_box": [190, 190, 150]}
HEIGHTS = {"carton": 80.0, "wood_box": 150.0}


def _calib(tmp_path, plane=None, plane_T=None):
    p = tmp_path / "T_base_camera.npy"
    np.save(p, REAL_T)
    meta_path_for(p).write_text(json.dumps({"robot_base_xyz_mm": [0, 0, 630],
                                            "robot_base_rpy_deg": [0, 0, 0]}), encoding="utf-8")
    if plane is not None:
        plane = dict(plane)
        plane["T_BC"] = np.asarray(REAL_T if plane_T is None else plane_T, float).tolist()
        (tmp_path / TABLE_PLANE_FILE).write_text(json.dumps(plane), encoding="utf-8")
    return p


GOOD_PLANE = {"a": 0.004, "b": -0.002, "c": 935.0, "z_top_mm": 937.0, "tilt_deg": 0.26}


class TestPreflightDepthMode:
    def _cfg(self, **kw):
        c = {"class_heights_mm": dict(HEIGHTS), "class_sizes_mm": dict(SIZES)}
        c.update(kw)
        return c

    def test_rgbd_runs_without_a_table_plane(self, tmp_path):
        problems, plane = check_depth_mode(self._cfg(), _calib(tmp_path), "rgbd")
        assert problems == [] and plane is None

    @pytest.mark.parametrize("mode", ["plane", "fusion"])
    def test_missing_plane_refused(self, tmp_path, mode):
        problems, plane = check_depth_mode(self._cfg(), _calib(tmp_path), mode)
        assert plane is None
        assert any(TABLE_PLANE_FILE in p and p.startswith(f"depth mode {mode}") for p in problems)

    def test_plane_without_coefficients_refused(self, tmp_path):
        p = _calib(tmp_path, plane={"z_top_mm": 937.0})
        problems, plane = check_depth_mode(self._cfg(), p, "plane")
        assert plane is None and any("coefficients" in x for x in problems)

    def test_plane_from_another_calibration_refused(self, tmp_path):
        older = REAL_T.copy()
        older[0, 3] += 5.0
        p = _calib(tmp_path, plane=GOOD_PLANE, plane_T=older)
        problems, _ = check_depth_mode(self._cfg(), p, "fusion")
        assert any("different T_BC" in x for x in problems)

    def test_sizes_must_match_the_configured_heights(self, tmp_path):
        p = _calib(tmp_path, plane=GOOD_PLANE)
        cfg = self._cfg(class_heights_mm={"carton": 120.0, "wood_box": 150.0})
        problems, _ = check_depth_mode(cfg, p, "rgbd")
        assert any("lowest height" in x for x in problems)

    def test_unparsable_sizes_refused(self, tmp_path):
        p = _calib(tmp_path, plane=GOOD_PLANE)
        problems, _ = check_depth_mode(self._cfg(class_sizes_mm={"carton": [1, 2]}), p, "rgbd")
        assert any("class_sizes_mm" in x for x in problems)

    def test_unknown_mode_refused(self, tmp_path):
        problems, _ = check_depth_mode(self._cfg(), _calib(tmp_path), "stereo")
        assert any("unknown depth mode" in x for x in problems)

    @pytest.mark.parametrize("mode", ["rgbd", "plane", "fusion"])
    def test_happy_path_returns_the_plane(self, tmp_path, mode):
        p = _calib(tmp_path, plane=GOOD_PLANE)
        problems, plane = check_depth_mode(self._cfg(), p, mode)
        assert problems == []
        assert plane["a"] == pytest.approx(0.004) and plane["c"] == pytest.approx(935.0)


class TestExperimentConfigSizes:
    def test_real_block_sizes_agree_with_the_heights(self):
        cfg = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text(encoding="utf-8"))
        real = cfg["real"]
        sizes = parse_class_sizes(real["class_sizes_mm"])
        assert sorted(sizes) == sorted(PAPER_CLASSES)
        for cls, cands in sizes.items():
            assert min(c[2] for c in cands) == pytest.approx(real["class_heights_mm"][cls])
        assert sorted(c[2] for c in sizes["carton"]) == [80.0, 120.0]

    def test_only_the_carton_has_two_sizes(self):
        cfg = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text(encoding="utf-8"))
        sizes = parse_class_sizes(cfg["real"]["class_sizes_mm"])
        assert [c for c, v in sizes.items() if len(v) > 1] == ["carton"]


class TestTrialLogColumns:
    def test_depth_columns_travel_to_the_csv(self, tmp_path):
        for col in ("depth_mode", "depth_used", "depth_valid_frac", "part_height_mm",
                    "size_iou", "h_depth_mm"):
            assert col in CONTEXT_FIELDNAMES
        p = tmp_path / "trials.csv"
        TrialLogger(p, extra_context={"mode": "real", "depth_mode": "fusion"}).log_trial(
            trial_id=1, success=True, class_name="carton",
            extra={"depth_used": "plane", "depth_valid_frac": 0.42,
                   "part_height_mm": 120.0, "size_iou": 0.93, "h_depth_mm": 118.4})
        row = list(csv.DictReader(p.open(newline="", encoding="utf-8")))[0]
        assert row["depth_mode"] == "fusion" and row["depth_used"] == "plane"
        assert row["part_height_mm"] == "120.0" and row["size_iou"] == "0.93"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
