"""Real-mode preparation for the paper's parts: square-part yaw, grasp depth,
preflight checks, table-plane fit, and the real-mode experiment config."""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from src.calibration.table_plane import depth_to_points, fit_table_plane
from src.orchestrator.orchestrator import grasp_depth_offset
from src.orchestrator.preflight import (
    SIM_PLACEHOLDER_T_BC,
    TABLE_PLANE_FILE,
    check_real_mode,
    meta_path_for,
)
from src.perception.postprocess import mask_pca_yaw

ROOT = Path(__file__).resolve().parent.parent
PAPER_CLASSES = ["carton", "plastic_box", "wood_box", "metal_box", "inox_box"]


def _square(angle_deg: float, side: int = 160, size: int = 400) -> np.ndarray:
    """Square mask rotated by angle_deg in image coordinates (x right, y down)."""
    t = math.radians(angle_deg)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * side / 2
    pts = np.round(corners @ R.T + size / 2).astype(np.int32)
    m = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m


def _mod90(deg: float) -> float:
    return (deg + 45.0) % 90.0 - 45.0


class TestSquarePartYaw:
    def test_axis_aligned_square(self):
        assert abs(mask_pca_yaw(_square(0))) < 0.05

    @pytest.mark.parametrize("deg", [10, 30, -25, 40, 75])
    def test_rotated_square_follows_its_edges(self, deg):
        yaw = math.degrees(mask_pca_yaw(_square(deg)))
        assert abs(_mod90(yaw - deg)) < 3.0
        assert -45.0 < yaw <= 45.0 + 1e-6

    def test_disk_keeps_zero(self):
        yy, xx = np.ogrid[:200, :200]
        m = (((xx - 100) ** 2 + (yy - 100) ** 2) <= 40 ** 2).astype(np.uint8)
        assert mask_pca_yaw(m) == 0.0

    def test_elongated_part_still_uses_pca(self):
        m = np.zeros((400, 400), np.uint8)
        m[180:220, 50:350] = 1
        assert abs(mask_pca_yaw(m)) < 0.05


class TestGraspDepth:
    def test_unknown_height_uses_max(self):
        assert grasp_depth_offset(None, 50.0, 20.0) == 50.0

    @pytest.mark.parametrize("h,want", [(30, 10), (40, 20), (80, 50), (85, 50), (150, 50)])
    def test_paper_parts_with_real_settings(self, h, want):
        assert grasp_depth_offset(h, 50.0, 20.0) == pytest.approx(want)

    def test_never_negative(self):
        assert grasp_depth_offset(15.0, 50.0, 20.0) == 0.0

    def test_sim_defaults_keep_tall_part_depth(self):
        # Simulation defaults (margin 100, max 50): the 150 mm bottle keeps 50 mm.
        assert grasp_depth_offset(150.0, 50.0, 100.0) == 50.0


REAL_T = np.array([[0.0, 1.0, 0.0, 518.0],
                   [1.0, 0.0, 0.0, 10.0],
                   [0.0, 0.0, -1.0, 1645.0],
                   [0.0, 0.0, 0.0, 1.0]])
BASE, RPY = (0.0, 0.0, 630.0), (0.0, 0.0, 0.0)


def _calib(tmp_path, T, base=BASE, rpy=RPY, meta=True, plane_z=None, plane_T=None):
    p = tmp_path / "T_base_camera.npy"
    np.save(p, np.asarray(T, float))
    if meta:
        meta_path_for(p).write_text(json.dumps(
            {"robot_base_xyz_mm": list(base), "robot_base_rpy_deg": list(rpy)}),
            encoding="utf-8")
    if plane_z is not None:
        (tmp_path / TABLE_PLANE_FILE).write_text(json.dumps(
            {"z_top_mm": plane_z,
             "T_BC": np.asarray(T if plane_T is None else plane_T, float).tolist()}),
            encoding="utf-8")
    return p


def _cfg(**kw):
    c = {"table_top_z_mm": None, "class_heights_mm": dict.fromkeys(PAPER_CLASSES, 50.0)}
    c.update(kw)
    return c


class TestPreflight:
    def test_placeholder_refused(self, tmp_path):
        p = _calib(tmp_path, SIM_PLACEHOLDER_T_BC, plane_z=935.0)
        problems, _ = check_real_mode(_cfg(), p, BASE, RPY)
        assert any("placeholder" in s for s in problems)

    def test_missing_meta_refused(self, tmp_path):
        p = _calib(tmp_path, REAL_T, meta=False, plane_z=935.0)
        problems, _ = check_real_mode(_cfg(), p, BASE, RPY)
        assert any("_meta.json missing" in s for s in problems)

    def test_changed_base_pose_refused(self, tmp_path):
        p = _calib(tmp_path, REAL_T, plane_z=935.0)
        problems, _ = check_real_mode(_cfg(), p, (0.0, 0.0, 530.0), RPY)
        assert any("base pose differs" in s for s in problems)

    def test_missing_table_refused(self, tmp_path):
        p = _calib(tmp_path, REAL_T)
        problems, z = check_real_mode(_cfg(), p, BASE, RPY)
        assert z is None
        assert any("table height unknown" in s for s in problems)

    def test_table_from_older_calibration_refused(self, tmp_path):
        older = REAL_T.copy()
        older[0, 3] += 5.0
        p = _calib(tmp_path, REAL_T, plane_z=935.0, plane_T=older)
        problems, _ = check_real_mode(_cfg(), p, BASE, RPY)
        assert any("different T_BC" in s for s in problems)

    def test_empty_heights_refused(self, tmp_path):
        p = _calib(tmp_path, REAL_T, plane_z=935.0)
        problems, _ = check_real_mode(_cfg(class_heights_mm={}), p, BASE, RPY)
        assert any("class_heights_mm" in s for s in problems)

    def test_placeholder_place_point_refused(self, tmp_path):
        """Shipped as an invented number; parts would be released over the floor."""
        p = _calib(tmp_path, REAL_T, plane_z=935.0)
        problems, _ = check_real_mode(
            _cfg(place_position=[700.0, 120.0, 700.0]), p, BASE, RPY)
        assert any("place_position" in s for s in problems)

    def test_placeholder_tcp_refused(self, tmp_path):
        p = _calib(tmp_path, REAL_T, plane_z=935.0)
        problems, _ = check_real_mode(_cfg(), p, BASE, RPY, [0, 0, 100])
        assert any("tcp_offset_xyz_mm" in s for s in problems)

    def test_measured_place_point_and_tcp_pass(self, tmp_path):
        p = _calib(tmp_path, REAL_T, plane_z=935.0)
        problems, _ = check_real_mode(
            _cfg(place_position=[812.0, -95.0, 700.0]), p, BASE, RPY, [0.4, -1.2, 183.5])
        assert problems == []

    def test_all_good_returns_measured_table(self, tmp_path):
        p = _calib(tmp_path, REAL_T, plane_z=935.4)
        problems, z = check_real_mode(_cfg(), p, BASE, RPY)
        assert problems == []
        assert z == pytest.approx(935.4)

    def test_explicit_table_height_is_used(self, tmp_path):
        p = _calib(tmp_path, REAL_T)
        problems, z = check_real_mode(_cfg(table_top_z_mm=940.0), p, BASE, RPY)
        assert problems == []
        assert z == pytest.approx(940.0)


class TestTablePlane:
    def test_recovers_tilted_plane_among_outliers(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0.0, 1000.0, 20000)
        y = rng.uniform(-250.0, 250.0, 20000)
        z = 935.0 + 0.004 * x - 0.002 * y + rng.normal(0.0, 0.8, x.size)
        k = rng.random(x.size) < 0.2
        z[k] = rng.uniform(600.0, 1100.0, int(k.sum()))
        r = fit_table_plane(np.column_stack([x, y, z]))
        assert r["a"] == pytest.approx(0.004, abs=5e-4)
        assert r["b"] == pytest.approx(-0.002, abs=5e-4)
        assert r["c"] == pytest.approx(935.0, abs=1.0)
        assert r["tilt_deg"] == pytest.approx(
            math.degrees(math.atan(math.hypot(0.004, 0.002))), abs=0.05)
        assert r["z_top_mm"] >= r["z_mean_mm"]

    def test_depth_to_points_goes_through_the_transform(self):
        d = np.full((72, 128), 0.712, np.float32)
        intr = {"fx": 65.7, "fy": 65.7, "ppx": 64.0, "ppy": 36.0}
        P = depth_to_points(d, intr, SIM_PLACEHOLDER_T_BC, stride=1, roi=1.0)
        # placeholder: camera looks down from z = 1200, so base z = 1200 - depth
        assert np.allclose(P[:, 2], 1200.0 - 712.0)

    def test_too_few_points(self):
        with pytest.raises(ValueError):
            fit_table_plane(np.zeros((10, 3)))


class TestExperimentConfig:
    def test_real_block_covers_the_paper_parts(self):
        cfg = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text(encoding="utf-8"))
        real = cfg["real"]
        assert real["yaw_offset_deg"] == 0.0
        assert real["table_top_z_mm"] is None
        assert sorted(real["class_heights_mm"]) == sorted(PAPER_CLASSES)
        assert real["class_heights_mm"]["plastic_box"] == 30.0
        assert real["class_heights_mm"]["inox_box"] == 40.0
