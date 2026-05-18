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
)

INTRINSICS = {"fx": 600.0, "fy": 600.0, "ppx": 320.0, "ppy": 240.0}


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
        det = {"mask": mask, "class_name": "box"}

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
