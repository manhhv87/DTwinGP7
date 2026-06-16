"""
test_coord_conv.py
──────────────────
Unit tests cho coordinate transforms — pure numpy, không cần RoboDK.

Run:
    pytest tests/test_coord_conv.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from src.orchestrator.coord_conv import (
    camera_to_base,
    load_calibration,
    make_grasp_pose,
    save_calibration,
    transform_point,
)


class TestCameraToBase:
    """Khớp kỳ vọng ở mục 8.1 tài liệu."""

    def setup_method(self):
        self.identity = np.eye(4)
        # Camera phía trên base, nhìn xuống.
        self.real = np.array([
            [1, 0, 0, 400],
            [0, -1, 0, 0],
            [0, 0, -1, 850],
            [0, 0, 0, 1],
        ], dtype=float)

    def test_origin_identity(self):
        np.testing.assert_array_almost_equal(
            camera_to_base(np.array([0, 0, 0]), self.identity), [0, 0, 0])

    def test_z_axis_identity(self):
        np.testing.assert_array_almost_equal(
            camera_to_base(np.array([0, 0, 500]), self.identity), [0, 0, 500])

    def test_real_transform_origin(self):
        np.testing.assert_array_almost_equal(
            camera_to_base(np.array([0, 0, 0]), self.real), [400, 0, 850])

    def test_depth_distance(self):
        # Điểm cách camera 500mm (z=500) → base.z = 850 - 500 = 350.
        p = camera_to_base(np.array([0, 0, 500]), self.real)
        assert p[2] == pytest.approx(350)


class TestTransformPoint:
    def test_translation(self):
        T = np.eye(4)
        T[:3, 3] = [10, 20, 30]
        np.testing.assert_array_almost_equal(
            transform_point([1, 2, 3], T), [11, 22, 33])


class TestMakeGraspPose:
    def test_translation_preserved(self):
        T = make_grasp_pose(np.array([123, -45, 67]), yaw_rad=0.0)
        np.testing.assert_array_almost_equal(T[:3, 3], [123, -45, 67])

    def test_gripper_points_down(self):
        """Trục Z của tool phải hướng xuống (-Z base) khi gắp top-down."""
        T = make_grasp_pose(np.array([0, 0, 0]), yaw_rad=0.0)
        np.testing.assert_array_almost_equal(T[:3, 2], [0, 0, -1])

    def test_yaw_rotates_about_vertical(self):
        T = make_grasp_pose(np.array([0, 0, 0]), yaw_rad=np.pi / 2)
        # Z tool vẫn hướng xuống dù xoay yaw.
        np.testing.assert_array_almost_equal(T[:3, 2], [0, 0, -1])


class TestCalibrationIO:
    def test_save_load_roundtrip(self, tmp_path):
        T = make_grasp_pose(np.array([400, 0, 850]), yaw_rad=0.3)
        path = tmp_path / "T.npy"
        save_calibration(path, T)
        np.testing.assert_array_almost_equal(load_calibration(path), T)

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_calibration(tmp_path / "nope.npy")

    def test_save_bad_shape_raises(self, tmp_path):
        with pytest.raises(ValueError, match="4x4"):
            save_calibration(tmp_path / "T.npy", np.eye(3))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_camera_yaw_to_base_rotates_by_hand_eye():
    """camera_yaw_to_base must rotate the image-frame yaw through R_BC — shared by
    the orchestrator pick path AND the GUI camera-teach path so they can't diverge
    (bug #R6-1). T_BC=+90° about Z → a camera yaw of 0 → base yaw +90°."""
    import numpy as np
    from src.orchestrator.coord_conv import camera_yaw_to_base
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert abs(camera_yaw_to_base(0.0, T) - np.pi / 2) < 1e-9
    assert abs(camera_yaw_to_base(0.3, np.eye(4)) - 0.3) < 1e-9   # identity → unchanged
