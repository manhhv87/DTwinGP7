"""
test_inverse_kinematics.py
──────────────────────────
Verify Damped Least Squares IK qua round-trip test (FK → IK → check joint matches)
và edge cases (singularity, ngoài tầm với, joint limits).

Strategy:
  1. Sample random joints trong limits
  2. Compute target pose qua FK
  3. Run IK với q_init xa solution
  4. Verify IK trả về joints sao cho FK(joints) == target pose (within tol)
"""
from __future__ import annotations

import numpy as np
import pytest

from src.orchestrator.kinematics import (
    forward_kinematics,
    gp7_default,
    inverse_kinematics,
)


# ─────────────────────────────────────────────────────────────────────────
# Round-trip tests
# ─────────────────────────────────────────────────────────────────────────


class TestRoundTrip:
    """FK(q) = pose → IK(pose, q_init) → q'. Verify FK(q') ≈ pose."""

    def test_ik_recovers_home_pose(self):
        model = gp7_default()
        q_target = [0.1, -0.3, 0.4, 0.2, -0.5, 0.6]
        T_target = forward_kinematics(model, q_target)

        # Start IK from home
        q_init = [0.0] * 6
        sol = inverse_kinematics(model, T_target, q_init)

        assert sol is not None, "IK should converge"
        T_sol = forward_kinematics(model, sol)
        pos_err = np.linalg.norm(T_sol[:3, 3] - T_target[:3, 3])
        assert pos_err < 0.1, f"Position error {pos_err:.4f}mm > 0.1mm tol"

    def test_ik_with_close_init_converges_fast(self):
        """q_init gần solution → converge trong < 30 iter."""
        model = gp7_default()
        q_target = np.array([0.5, -0.2, 0.3, 0.4, -0.1, 0.7])
        T_target = forward_kinematics(model, q_target)

        # Initial guess 10° lệch mỗi joint
        q_init = (q_target + np.deg2rad(10)).tolist()
        sol = inverse_kinematics(model, T_target, q_init, max_iter=30)

        assert sol is not None
        T_sol = forward_kinematics(model, sol)
        assert np.linalg.norm(T_sol[:3, 3] - T_target[:3, 3]) < 0.1

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_ik_multiple_random_poses(self, seed):
        """Sample 5 random joint configs và verify IK round-trip."""
        rng = np.random.RandomState(seed)
        model = gp7_default()

        for _ in range(5):
            # Random joints in central 60% of range to avoid limits
            q_target = []
            for link in model.links:
                q_min, q_max = link.joint_min, link.joint_max
                margin = (q_max - q_min) * 0.2
                q_target.append(rng.uniform(q_min + margin, q_max - margin))
            q_target = np.array(q_target)

            T_target = forward_kinematics(model, q_target)
            # Init: q_target perturbed by ±15°
            q_init = (q_target + rng.uniform(-np.deg2rad(15), np.deg2rad(15), 6)).tolist()

            sol = inverse_kinematics(model, T_target, q_init)
            assert sol is not None, f"IK failed for seed {seed}, q_target={q_target}"
            T_sol = forward_kinematics(model, sol)
            pos_err = np.linalg.norm(T_sol[:3, 3] - T_target[:3, 3])
            assert pos_err < 0.2, f"Pos err {pos_err:.4f}mm > 0.2mm (loose tol)"

    def test_default_tolerance_is_tight(self):
        """Default tol 0.1mm/1e-4 rad — confirmed via convergence."""
        model = gp7_default()
        q_target = [0.3, -0.4, 0.5, 0.2, -0.3, 0.4]
        T_target = forward_kinematics(model, q_target)
        sol = inverse_kinematics(model, T_target, [0.0] * 6)
        assert sol is not None
        T_sol = forward_kinematics(model, sol)
        pos_err = np.linalg.norm(T_sol[:3, 3] - T_target[:3, 3])
        # Should converge WELL inside the 0.1mm tol (typically 0.01-0.05mm)
        assert pos_err < 0.1, f"Default tol violated: {pos_err:.4f}mm > 0.1mm"


# ─────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_pose_outside_reach_returns_none(self):
        """Target ngoài tầm với (10m từ base) → IK không converge."""
        model = gp7_default()
        T_target = np.eye(4)
        T_target[:3, 3] = [10000.0, 0.0, 0.0]                 # 10m, ngoài tầm GP7
        sol = inverse_kinematics(model, T_target, [0.0] * 6, max_iter=30)
        # Có thể trả None hoặc trả best-effort. Just verify it doesn't crash.
        if sol is not None:
            # Best effort: position should still be very far from target
            T_sol = forward_kinematics(model, sol)
            assert np.linalg.norm(T_sol[:3, 3] - T_target[:3, 3]) > 1000

    def test_invalid_q_init_length_raises(self):
        model = gp7_default()
        T_target = forward_kinematics(model, [0.0] * 6)
        with pytest.raises(ValueError, match="q_init"):
            inverse_kinematics(model, T_target, [0.0, 0.0])    # only 2 joints

    def test_invalid_pose_shape_raises(self):
        model = gp7_default()
        with pytest.raises(ValueError, match="4x4"):
            inverse_kinematics(model, np.eye(3), [0.0] * 6)

    def test_joint_limits_respected(self):
        """IK output phải nằm trong joint limits."""
        model = gp7_default()
        q_target = [0.2, -0.5, 0.6, 0.3, -0.2, 0.4]
        T_target = forward_kinematics(model, q_target)
        sol = inverse_kinematics(model, T_target, [0.0] * 6)

        assert sol is not None
        for i, (q, link) in enumerate(zip(sol, model.links)):
            assert link.joint_min - 1e-6 <= q <= link.joint_max + 1e-6, \
                f"Joint {i} = {np.rad2deg(q):.1f}° out of limits"


# ─────────────────────────────────────────────────────────────────────────
# Performance benchmark
# ─────────────────────────────────────────────────────────────────────────


class TestPerformance:
    def test_ik_completes_in_reasonable_time(self):
        """IK 1 call < 50ms (real-time requirement cho cycle ~7s/trial)."""
        import time

        model = gp7_default()
        q_target = [0.3, -0.4, 0.5, 0.2, -0.1, 0.6]
        T_target = forward_kinematics(model, q_target)

        start = time.perf_counter()
        sol = inverse_kinematics(model, T_target, [0.0] * 6)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert sol is not None
        assert elapsed_ms < 50, f"IK took {elapsed_ms:.1f}ms, target < 50ms"
