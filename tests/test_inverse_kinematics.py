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

    def test_pose_error_180deg_rotation_stable(self):
        """`_pose_error` corner case: rotation exactly π giữa current và target.

        Bug cũ: diagonal extraction có thể fail khi 2 diagonal entries gần
        bằng nhau. Fix mới: column-norm extraction từ R+I luôn numerically stable.
        """
        from src.orchestrator.kinematics.inverse_kinematics import _pose_error
        # R = 180° rotation about X axis: diag = (1, -1, -1)
        T_current = np.eye(4)
        T_target = np.eye(4)
        T_target[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        err = _pose_error(T_current, T_target)
        # rot_err magnitude = π, direction = ±X axis
        rot_err = err[3:]
        assert abs(np.linalg.norm(rot_err) - np.pi) < 1e-6
        # Axis should be (±1, 0, 0)
        axis = rot_err / np.linalg.norm(rot_err)
        assert abs(abs(axis[0]) - 1.0) < 1e-6
        assert abs(axis[1]) < 1e-6
        assert abs(axis[2]) < 1e-6

    def test_pose_error_180deg_around_yz_diagonal(self):
        """Edge case khó nhất: 180° rotation around (0, 1, 1)/√2 axis →
        diag entries của R đều âm + 2 entries bằng nhau. Robust column-norm
        algorithm phải pick đúng axis."""
        from src.orchestrator.kinematics.inverse_kinematics import _pose_error
        # 180° about axis (0, 1, 1)/√2: R = 2·n·n^T - I
        n = np.array([0.0, 1.0, 1.0]) / np.sqrt(2)
        R = 2 * np.outer(n, n) - np.eye(3)
        T_current = np.eye(4)
        T_target = np.eye(4); T_target[:3, :3] = R
        err = _pose_error(T_current, T_target)
        rot_err = err[3:]
        # Magnitude = π
        assert abs(np.linalg.norm(rot_err) - np.pi) < 1e-6
        axis = rot_err / np.linalg.norm(rot_err)
        # Axis should be ±(0, 1, 1)/√2 (sign ambiguous for 180°)
        assert abs(axis[0]) < 1e-6
        assert abs(abs(axis[1]) - 1/np.sqrt(2)) < 1e-6
        assert abs(abs(axis[2]) - 1/np.sqrt(2)) < 1e-6

    def test_pose_error_batch_matches_scalar_incl_180deg(self):
        """Regression: `_pose_error_batch` must NOT collapse a real 180° rotation
        error to zero (false convergence). It must match the scalar `_pose_error`
        rotation-norm for all rotations, including θ≈π."""
        from scipy.spatial.transform import Rotation as Rot
        from src.orchestrator.kinematics.inverse_kinematics import (
            _pose_error, _pose_error_batch)
        # Explicit 180° about Z (the case that previously returned 0).
        Tt = np.eye(4)
        Tc = np.eye(4); Tc[:3, :3] = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1.0]])
        b = _pose_error_batch(Tc[None], Tt)[0]
        assert abs(np.linalg.norm(b[3:]) - np.pi) < 1e-6   # was ~0 before the fix
        # Cross-check batch vs scalar rotation-norm over many random rotations.
        rng = np.random.default_rng(0)
        for _ in range(500):
            Rc = Rot.random(random_state=rng).as_matrix()
            Rt = Rot.random(random_state=rng).as_matrix()
            Tc = np.eye(4); Tc[:3, :3] = Rc
            Tt = np.eye(4); Tt[:3, :3] = Rt
            nb = np.linalg.norm(_pose_error_batch(Tc[None], Tt)[0][3:])
            ns = np.linalg.norm(_pose_error(Tc, Tt)[3:])
            assert abs(nb - ns) < 1e-6


# ─────────────────────────────────────────────────────────────────────────
# Alternative IK algorithms (Levenberg-Marquardt, SDLS, BFGS)
# ─────────────────────────────────────────────────────────────────────────


class TestAlternativeIK:
    """Verify LM / SDLS / BFGS đều converge cho representative poses."""

    @pytest.fixture
    def model_target(self):
        from src.orchestrator.kinematics.urdf_chain import (
            gp7_urdf, forward_kinematics_urdf,
        )
        m = gp7_urdf()
        q_true = np.deg2rad([20, -10, 25, 5, 40, 10])
        T = forward_kinematics_urdf(m, q_true)
        return m, T, q_true

    def test_lm_converges_from_close_init(self, model_target):
        """Levenberg-Marquardt converge từ q_init lệch ±5°."""
        from src.orchestrator.kinematics.inverse_kinematics import (
            inverse_kinematics_lm,
        )
        from src.orchestrator.kinematics.urdf_chain import forward_kinematics_urdf
        model, T_target, q_true = model_target
        q_init = q_true + np.deg2rad(5) * np.random.RandomState(0).uniform(-1, 1, 6)
        sol = inverse_kinematics_lm(
            model, T_target, q_init.tolist(),
            tol_mm=0.5, tol_rad=1e-3, max_iter=100)
        assert sol is not None
        T_back = forward_kinematics_urdf(model, sol)
        pos_err = float(np.linalg.norm(T_back[:3, 3] - T_target[:3, 3]))
        assert pos_err < 0.5, f"LM pos err {pos_err:.4f} mm > tol"

    def test_sdls_converges_from_close_init(self, model_target):
        """SDLS converge với SVD-based selective damping."""
        from src.orchestrator.kinematics.inverse_kinematics import (
            inverse_kinematics_sdls,
        )
        from src.orchestrator.kinematics.urdf_chain import forward_kinematics_urdf
        model, T_target, q_true = model_target
        q_init = q_true + np.deg2rad(5) * np.random.RandomState(1).uniform(-1, 1, 6)
        sol = inverse_kinematics_sdls(
            model, T_target, q_init.tolist(),
            tol_mm=0.5, tol_rad=1e-3, max_iter=100)
        assert sol is not None
        T_back = forward_kinematics_urdf(model, sol)
        pos_err = float(np.linalg.norm(T_back[:3, 3] - T_target[:3, 3]))
        assert pos_err < 0.5, f"SDLS pos err {pos_err:.4f} mm > tol"

    def test_bfgs_converges_from_close_init(self, model_target):
        """BFGS (scipy L-BFGS-B với weighted cost) converge tới float precision."""
        from src.orchestrator.kinematics.inverse_kinematics import (
            inverse_kinematics_bfgs,
        )
        from src.orchestrator.kinematics.urdf_chain import forward_kinematics_urdf
        model, T_target, q_true = model_target
        q_init = q_true + np.deg2rad(5) * np.random.RandomState(2).uniform(-1, 1, 6)
        sol = inverse_kinematics_bfgs(
            model, T_target, q_init.tolist(),
            tol_mm=0.5, tol_rad=1e-3, max_iter=200)
        assert sol is not None
        T_back = forward_kinematics_urdf(model, sol)
        pos_err = float(np.linalg.norm(T_back[:3, 3] - T_target[:3, 3]))
        # BFGS thường đạt sub-micrometer với weighted cost
        assert pos_err < 0.5, f"BFGS pos err {pos_err:.4f} mm > tol"

    def test_all_methods_respect_joint_limits(self, model_target):
        """Tất cả 3 methods clip/respect joint limits."""
        from src.orchestrator.kinematics.inverse_kinematics import (
            inverse_kinematics_lm, inverse_kinematics_sdls, inverse_kinematics_bfgs,
        )
        model, T_target, q_true = model_target
        q_init = q_true.copy()
        for fn in (inverse_kinematics_lm,
                    inverse_kinematics_sdls,
                    inverse_kinematics_bfgs):
            sol = fn(model, T_target, q_init.tolist(),
                      tol_mm=0.5, tol_rad=1e-3, max_iter=100)
            assert sol is not None, f"{fn.__name__} returned None for in-init pose"
            for i, q in enumerate(sol):
                jl = model.joints[i]
                assert jl.joint_min - 1e-6 <= q <= jl.joint_max + 1e-6, \
                    f"{fn.__name__} joint {i} out of limits: {np.degrees(q):.1f}°"

    def test_unreachable_pose_returns_none(self, model_target):
        """Pose ngoài tầm → LM/SDLS/BFGS trả None (không crash)."""
        from src.orchestrator.kinematics.inverse_kinematics import (
            inverse_kinematics_lm, inverse_kinematics_sdls, inverse_kinematics_bfgs,
        )
        model, _, _ = model_target
        # 10m ngoài reach (GP7 max ~927mm)
        T_unreach = np.eye(4)
        T_unreach[:3, 3] = [10000.0, 0.0, 0.0]
        for fn in (inverse_kinematics_lm,
                    inverse_kinematics_sdls,
                    inverse_kinematics_bfgs):
            sol = fn(model, T_unreach, [0.0] * 6,
                      tol_mm=0.5, tol_rad=1e-3, max_iter=30)
            # Có thể trả None hoặc trả best-effort xa target
            if sol is not None:
                from src.orchestrator.kinematics.urdf_chain import forward_kinematics_urdf
                T_back = forward_kinematics_urdf(model, sol)
                pos_err = float(np.linalg.norm(T_back[:3, 3] - T_unreach[:3, 3]))
                # Best-effort còn cách xa target
                assert pos_err > 1000, \
                    f"{fn.__name__} accepted unreachable as solution"


# ─────────────────────────────────────────────────────────────────────────
# Performance benchmark
# ─────────────────────────────────────────────────────────────────────────


class TestTaggedConfigurations:
    """Pieper tagged IK enumerates IK solutions like RoboDK. The §3.3 'Other
    configurations' list uses include_turns=True so the COUNT matches RoboDK:
      • up to 8 postures (Front/Rear · Up/Down · Flip), singularity-separated;
      • each posture × its ±360° joint-turn variants (GP7 has wide axes: J6 ±360°,
        J4 ±190°, J3 spans 371°), which RoboDK also lists.
    Dropping turns would show roughly half the rows — the bug this guards against."""

    @pytest.fixture
    def model_pose(self):
        from src.orchestrator.kinematics.urdf_chain import (
            gp7_urdf, forward_kinematics_urdf,
        )
        m = gp7_urdf()
        T = forward_kinematics_urdf(m, np.deg2rad([20, 10, -15, 30, -40, 25]))
        return m, T

    def test_turns_enumerated_so_count_matches_robodk(self, model_pose):
        # The panel uses include_turns=True. For this GP7 pose the ±360° windings
        # of the wide axes give strictly MORE solutions than the bare postures.
        from src.orchestrator.kinematics.pieper_gp7 import (
            inverse_kinematics_pieper_gp7_tagged as tag)
        m, T = model_pose
        with_turns = tag(m, T, include_turns=True)
        no_turns = tag(m, T, include_turns=False)
        assert with_turns, "pose should be reachable"
        assert len(with_turns) > len(no_turns), (
            "joint turns must inflate the list (else we under-count vs RoboDK)")
        # turn variants share a config_id → duplicate ids are expected with turns
        ids = [c["id"] for c in with_turns]
        assert len(ids) != len(set(ids)), "expected ±360° turn variants per posture"

    def test_each_row_flags_consistent_with_id(self, model_pose):
        from src.orchestrator.kinematics.pieper_gp7 import (
            inverse_kinematics_pieper_gp7_tagged as tag)
        m, T = model_pose
        for c in tag(m, T, include_turns=True):
            assert c["id"] == ((not c["front"]) * 4
                               + (not c["elbow_up"]) * 2
                               + (not c["no_flip"]))
            assert 0 <= c["id"] <= 7

    def test_no_turns_is_one_per_posture(self, model_pose):
        # Sanity on the postures themselves: ≤ 8, one per id, singularity-separated.
        from src.orchestrator.kinematics.pieper_gp7 import (
            inverse_kinematics_pieper_gp7_tagged as tag)
        m, T = model_pose
        no_turns = tag(m, T, include_turns=False)
        ids = [c["id"] for c in no_turns]
        assert len(ids) == len(set(ids))
        assert len(no_turns) <= 8


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
