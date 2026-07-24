"""
test_postprocess.py
───────────────────
Unit tests cho trích xuất pose từ mask + depth — pure numpy.

Run:
    pytest tests/test_postprocess.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from src.perception.postprocess import (
    PoseExtractor,
    deproject_pixel,
    mask_centroid,
    mask_pca_yaw,
    masked_depth,
    z_from_ground_plane,
)

INTRINSICS = {"fx": 600.0, "fy": 600.0, "ppx": 320.0, "ppy": 240.0}


def _overhead_T_BC(cam_xyz_mm=(700.0, 0.0, 1200.0)):
    """Eye-to-hand camera looking straight down: R = Rx(180) = diag(1,-1,-1)."""
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = cam_xyz_mm
    return T


class TestZFromGroundPlane:
    """C4 depth-free localisation: ray--plane intersection."""

    def test_center_pixel_distance(self):
        # Camera 1200 mm up, table plane at 500 mm → distance 700 mm = 0.7 m.
        T = _overhead_T_BC()
        z = z_from_ground_plane(INTRINSICS, 320.0, 240.0, T, plane_z_base_mm=500.0)
        assert z == pytest.approx(0.7, abs=1e-6)

    def test_deprojected_point_lies_on_plane(self):
        # Any pixel → deprojected + transformed to base must land on Z=plane.
        T = _overhead_T_BC()
        for u, v in [(320.0, 240.0), (500.0, 120.0), (100.0, 400.0)]:
            z = z_from_ground_plane(INTRINSICS, u, v, T, plane_z_base_mm=500.0)
            p_cam = np.append(deproject_pixel(INTRINSICS, u, v, z), 1.0)  # mm, homog
            p_base = (T @ p_cam)[:3]
            assert p_base[2] == pytest.approx(500.0, abs=1e-3)

    def test_center_pixel_lands_under_camera(self):
        T = _overhead_T_BC(cam_xyz_mm=(700.0, 0.0, 1200.0))
        z = z_from_ground_plane(INTRINSICS, 320.0, 240.0, T, plane_z_base_mm=500.0)
        p_cam = np.append(deproject_pixel(INTRINSICS, 320.0, 240.0, z), 1.0)
        p_base = (T @ p_cam)[:3]
        assert p_base[0] == pytest.approx(700.0, abs=1e-3)  # under the camera
        assert p_base[1] == pytest.approx(0.0, abs=1e-3)

    def test_higher_plane_gives_smaller_z(self):
        # Top of a taller box (higher plane) is closer to the overhead camera.
        T = _overhead_T_BC()
        z_low = z_from_ground_plane(INTRINSICS, 320.0, 240.0, T, 500.0)
        z_high = z_from_ground_plane(INTRINSICS, 320.0, 240.0, T, 560.0)
        assert z_high < z_low

    def test_ray_parallel_returns_none(self):
        # Camera looking horizontally → ray never meets a horizontal plane.
        T = np.eye(4)
        T[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        T[:3, 3] = [700.0, 0.0, 600.0]
        assert z_from_ground_plane(INTRINSICS, 320.0, 240.0, T, 500.0) is None


def rect_mask(hw, box):
    """Tạo mask nhị phân với 1 hình chữ nhật."""
    m = np.zeros(hw, np.uint8)
    x1, y1, x2, y2 = box
    m[y1:y2, x1:x2] = 1
    return m


class TestMaskCentroid:
    def test_centroid_of_rectangle(self):
        mask = rect_mask((480, 640), (300, 200, 340, 280))
        u, v = mask_centroid(mask)
        assert u == pytest.approx(319, abs=1)  # (300+339)/2
        assert v == pytest.approx(239, abs=1)

    def test_too_small_returns_none(self):
        mask = np.zeros((480, 640), np.uint8)
        mask[0, 0] = 1
        assert mask_centroid(mask) is None


class TestMaskedDepth:
    def test_median_ignores_zeros(self):
        depth = np.zeros((480, 640), np.float32)
        mask = rect_mask((480, 640), (100, 100, 200, 200))
        depth[mask > 0] = 0.8           # vùng vật
        depth[150, 150] = 0.0           # 1 pixel hỏng
        z = masked_depth(depth, mask, method="median")
        assert z == pytest.approx(0.8, abs=1e-6)

    def test_no_valid_depth_returns_none(self):
        depth = np.zeros((480, 640), np.float32)
        mask = rect_mask((480, 640), (100, 100, 200, 200))
        assert masked_depth(depth, mask) is None


class TestMaskPcaYaw:
    def test_horizontal_bar_yaw_zero(self):
        mask = rect_mask((480, 640), (100, 235, 540, 245))  # rộng theo x
        assert mask_pca_yaw(mask) == pytest.approx(0.0, abs=0.05)

    def test_vertical_bar_yaw_90(self):
        mask = rect_mask((480, 640), (315, 50, 325, 430))   # dài theo y
        assert abs(mask_pca_yaw(mask)) == pytest.approx(np.pi / 2, abs=0.05)

    def test_near_circular_mask_yaw_low_confidence(self):
        # Regression: near-circular mask → major axis is arbitrary/noisy → return 0.0
        # (low-confidence) instead of a yaw that flips frame-to-frame.
        yy, xx = np.ogrid[:200, :200]
        disk = ((xx - 100) ** 2 + (yy - 100) ** 2) <= 40 ** 2
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[disk] = 1
        assert mask_pca_yaw(mask) == 0.0


class TestDeprojectPixel:
    def test_principal_point_maps_to_axis(self):
        xyz = deproject_pixel(INTRINSICS, 320, 240, 1.0)
        np.testing.assert_array_almost_equal(xyz, [0, 0, 1000])  # mm

    def test_offset_pixel(self):
        # Lệch fx pixel theo u, z=1m → x = 1m = 1000mm.
        xyz = deproject_pixel(INTRINSICS, 320 + 600, 240, 1.0)
        np.testing.assert_array_almost_equal(xyz, [1000, 0, 1000])


class TestPoseExtractor:
    def test_extract_full_pose(self):
        depth = np.full((480, 640), 0.8, np.float32)
        mask = rect_mask((480, 640), (300, 220, 340, 260))
        det = {"mask": mask, "class_name": "cup"}

        extractor = PoseExtractor(INTRINSICS)
        result = extractor.extract(det, depth)

        assert result is not None
        x, y, z, yaw = result["pose_camera"]
        assert z == pytest.approx(800.0, abs=1.0)   # 0.8m → mm
        assert result["mask_area"] == 40 * 40

    def test_extract_empty_mask_returns_none(self):
        depth = np.full((480, 640), 0.8, np.float32)
        det = {"mask": np.zeros((480, 640), np.uint8)}
        assert PoseExtractor(INTRINSICS).extract(det, depth) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
