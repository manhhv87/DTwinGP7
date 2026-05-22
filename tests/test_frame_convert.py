"""
test_frame_convert.py
─────────────────────
Verify frame conversion utilities cho HSE Cartesian path.

Critical correctness tests:
  - world ↔ robot base round-trip
  - matrix ↔ RPY (Yaskawa XYZ-fixed) round-trip
  - Edge cases: identity, pure translation, pure rotation, gimbal lock
"""
from __future__ import annotations

import numpy as np
import pytest

from src.orchestrator.frame_convert import (
    matrix_to_rpy_yaskawa,
    matrix_to_xyzrpy_yaskawa,
    rpy_yaskawa_to_matrix,
    world_to_robot_base,
)


# ─────────────────────────────────────────────────────────────────────────
# RPY ↔ matrix round-trip
# ─────────────────────────────────────────────────────────────────────────


class TestRpyMatrix:
    def test_identity_rotation_round_trip(self):
        R = rpy_yaskawa_to_matrix(0, 0, 0)
        np.testing.assert_array_almost_equal(R, np.eye(3))
        rx, ry, rz = matrix_to_rpy_yaskawa(R)
        assert abs(rx) < 1e-9 and abs(ry) < 1e-9 and abs(rz) < 1e-9

    @pytest.mark.parametrize("rx,ry,rz", [
        (30, 0, 0), (0, 45, 0), (0, 0, 60),
        (10, -20, 30), (-45, 30, -60), (90, 0, 0),
    ])
    def test_rpy_round_trip(self, rx, ry, rz):
        """RPY → matrix → RPY phải match (trừ gimbal lock ±90°)."""
        R = rpy_yaskawa_to_matrix(rx, ry, rz)
        rx2, ry2, rz2 = matrix_to_rpy_yaskawa(R)
        # Should match within numerical precision
        assert abs(rx - rx2) < 1e-4
        assert abs(ry - ry2) < 1e-4
        assert abs(rz - rz2) < 1e-4

    def test_matrix_is_orthonormal(self):
        R = rpy_yaskawa_to_matrix(15, -30, 45)
        # R^T · R = I
        np.testing.assert_array_almost_equal(R.T @ R, np.eye(3), decimal=10)
        # det(R) = +1
        assert abs(np.linalg.det(R) - 1.0) < 1e-9

    def test_gimbal_lock_pitch_90(self):
        """β = 90° là gimbal lock — rx + rz degenerate, expect no crash."""
        R = rpy_yaskawa_to_matrix(30, 90, 0)
        rx, ry, rz = matrix_to_rpy_yaskawa(R)
        assert abs(ry - 90) < 1e-3
        # rx + rz combined is well-defined, but split is arbitrary


# ─────────────────────────────────────────────────────────────────────────
# World → robot base
# ─────────────────────────────────────────────────────────────────────────


class TestWorldToRobotBase:
    def test_identity_when_base_at_origin(self):
        """Robot base at world origin → T_base == T_world."""
        T_world = np.eye(4)
        T_world[:3, 3] = [500, 200, 600]
        T_base = world_to_robot_base(T_world, (0.0, 0.0, 0.0))
        np.testing.assert_array_almost_equal(T_base, T_world)

    def test_translation_only_offset(self):
        """Robot base ở Z=630, target ở world (500, 200, 1000) → base (500, 200, 370)."""
        T_world = np.eye(4)
        T_world[:3, 3] = [500, 200, 1000]
        T_base = world_to_robot_base(T_world, (0.0, 0.0, 630.0))
        np.testing.assert_array_almost_equal(T_base[:3, 3], [500, 200, 370])
        np.testing.assert_array_almost_equal(T_base[:3, :3], np.eye(3))

    def test_translation_and_rotation_offset(self):
        """Robot base xoay 90° quanh Z → vector world (1,0,0) thành (0,-1,0) trong base."""
        T_world = np.eye(4)
        T_world[:3, 3] = [1000, 0, 0]
        # Base ở origin xoay 90° quanh Z (yaw 90° in our XYZ-fixed convention)
        T_base = world_to_robot_base(T_world, (0.0, 0.0, 0.0), (0.0, 0.0, 90.0))
        # World x-axis (1,0,0) in base frame xoay -90° → (0,-1,0)
        np.testing.assert_array_almost_equal(T_base[:3, 3], [0, -1000, 0], decimal=4)

    def test_raises_on_wrong_shape(self):
        with pytest.raises(ValueError, match="4x4"):
            world_to_robot_base(np.eye(3), (0.0, 0.0, 0.0))


# ─────────────────────────────────────────────────────────────────────────
# Full pipeline: 4x4 → xyz + rpy
# ─────────────────────────────────────────────────────────────────────────


class TestMatrixToXyzRpy:
    def test_pure_translation(self):
        T = np.eye(4)
        T[:3, 3] = [100, 200, 300]
        x, y, z, rx, ry, rz = matrix_to_xyzrpy_yaskawa(T)
        assert (x, y, z) == (100, 200, 300)
        assert abs(rx) < 1e-9 and abs(ry) < 1e-9 and abs(rz) < 1e-9

    def test_translation_plus_rotation(self):
        R = rpy_yaskawa_to_matrix(10, 20, 30)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [500, -200, 800]
        x, y, z, rx, ry, rz = matrix_to_xyzrpy_yaskawa(T)
        assert abs(x - 500) < 1e-6
        assert abs(y - (-200)) < 1e-6
        assert abs(z - 800) < 1e-6
        assert abs(rx - 10) < 1e-3
        assert abs(ry - 20) < 1e-3
        assert abs(rz - 30) < 1e-3
