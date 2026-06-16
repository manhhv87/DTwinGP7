"""
test_hand_eye_solver.py
───────────────────────
Unit tests cho hand-eye solver eye-to-hand.

Phần solve_hand_eye cần OpenCV (cv2) — tự skip nếu chưa cài.
Test dùng dữ liệu tổng hợp: dựng T_BC đã biết, sinh các cặp pose, rồi
kiểm tra solver khôi phục đúng T_BC → xác minh quy ước eye-to-hand.

Run:
    pytest tests/test_hand_eye_solver.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration.hand_eye_solver import (
    invert_transform,
    touch_test_error,
    validate_poses,
)

cv2 = pytest.importorskip("cv2", reason="cần opencv-python cho solve_hand_eye")
from scipy.spatial.transform import Rotation  # noqa: E402

from src.calibration.hand_eye_solver import solve_hand_eye  # noqa: E402


def make_T(rpy_deg, xyz) -> np.ndarray:
    """Dựng ma trận SE(3) 4x4 từ góc RPY (độ) + translation."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    T[:3, 3] = xyz
    return T


def synthetic_dataset(T_BC: np.ndarray, T_TG: np.ndarray, n: int = 15):
    """Sinh n cặp (gripper2base, target2cam) nhất quán với T_BC, T_TG.

    Quan hệ: T_target2cam = inv(T_BC) @ T_GB @ T_TG
    """
    rng = np.random.default_rng(42)
    g2b_list, t2c_list = [], []
    for _ in range(n):
        rpy = rng.uniform(-40, 40, 3)          # rotation đa dạng
        xyz = rng.uniform(-0.3, 0.3, 3)        # translation (mét)
        T_GB = make_T(rpy, xyz)
        T_t2c = invert_transform(T_BC) @ T_GB @ T_TG
        g2b_list.append(T_GB)
        t2c_list.append(T_t2c)
    return g2b_list, t2c_list


class TestInvertTransform:
    def test_roundtrip(self):
        T = make_T([12, -34, 56], [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(invert_transform(T) @ T, np.eye(4))


class TestValidatePoses:
    def test_too_few_raises(self):
        with pytest.raises(ValueError, match="At least 10 poses"):
            validate_poses([np.eye(4)] * 5)

    def test_enough_poses_ok(self):
        poses = [make_T([i * 5, 0, 0], [0, 0, 0]) for i in range(12)]
        validate_poses(poses)  # không raise


class TestTouchTestError:
    def test_zero_error_identity(self):
        pts_cam = [np.array([10, 20, 30]), np.array([40, 50, 60])]
        pts_base = [p.copy() for p in pts_cam]
        err = touch_test_error(pts_cam, pts_base, np.eye(4))
        assert err["mean_mm"] == pytest.approx(0.0)
        assert err["max_mm"] == pytest.approx(0.0)

    def test_known_offset(self):
        T = np.eye(4)
        T[:3, 3] = [5, 0, 0]                 # lệch 5mm theo X
        pts_cam = [np.array([0, 0, 0])]
        pts_base = [np.array([0, 0, 0])]     # "đo thực tế" = gốc
        err = touch_test_error(pts_cam, pts_base, T)
        assert err["mean_mm"] == pytest.approx(5.0)


class TestSolveHandEye:
    """Xác minh solver khôi phục đúng T_BC từ dữ liệu tổng hợp."""

    def setup_method(self):
        self.T_BC = make_T([180, 0, 30], [0.4, 0.0, 0.85])   # camera trong base
        self.T_TG = make_T([5, -10, 15], [0.0, 0.0, 0.12])   # board trên gripper

    @pytest.mark.parametrize("method", ["park", "horaud", "daniilidis"])
    def test_recovers_known_calibration(self, method):
        g2b, t2c = synthetic_dataset(self.T_BC, self.T_TG, n=15)
        recovered = solve_hand_eye(g2b, t2c, method=method)
        np.testing.assert_array_almost_equal(recovered, self.T_BC, decimal=4)

    def test_tsai_fails_at_180deg_rotation(self):
        """Ghi nhận điểm kỳ dị: Tsai-Lenz KHÔNG khôi phục được T_BC khi
        camera xoay ~180° — lý do solver mặc định dùng 'park'."""
        g2b, t2c = synthetic_dataset(self.T_BC, self.T_TG, n=15)
        recovered = solve_hand_eye(g2b, t2c, method="tsai")
        assert not np.allclose(recovered, self.T_BC, atol=1e-3)

    def test_mismatched_lengths_raise(self):
        g2b, t2c = synthetic_dataset(self.T_BC, self.T_TG, n=12)
        with pytest.raises(ValueError, match="Pose count mismatch"):
            solve_hand_eye(g2b, t2c[:-1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_touch_test_error_empty_raises():
    """Empty point lists → clear ValueError, not NaN/crash (bug #19)."""
    with pytest.raises(ValueError, match="No evaluation points|empty"):
        touch_test_error([], [], np.eye(4))
