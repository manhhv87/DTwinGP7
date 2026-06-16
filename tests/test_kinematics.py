"""
test_kinematics.py
──────────────────
Verify forward kinematics + trajectory interpolation + safety checks.

KHÔNG cần RoboDK — pure numpy testing.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.orchestrator.kinematics import (
    TrajectorySample,
    check_joint_limits,
    check_self_collision_spheres,
    forward_kinematics,
    gp7_default,
    interpolate_joints,
    joint_positions,
)
from src.orchestrator.kinematics.dh_model import DHLink, RobotDHModel


# ─────────────────────────────────────────────────────────────────────────
# DH model
# ─────────────────────────────────────────────────────────────────────────


class TestDHModel:
    def test_gp7_default_has_6_links(self):
        m = gp7_default()
        assert m.num_joints() == 6
        assert m.name == "Yaskawa GP7"

    def test_gp7_joint_limits_reasonable(self):
        m = gp7_default()
        # S axis ±170° = ~±2.97 rad
        assert -3.0 <= m.links[0].joint_min < m.links[0].joint_max <= 3.0
        # T (J6) ±360°
        assert m.links[5].joint_max >= np.deg2rad(355)

    def test_base_xyz_configurable(self):
        m = gp7_default(base_xyz_mm=(0, 0, 630))
        assert m.base_xyz_mm == (0, 0, 630)


# ─────────────────────────────────────────────────────────────────────────
# Forward kinematics
# ─────────────────────────────────────────────────────────────────────────


class TestForwardKinematics:
    def test_fk_returns_4x4(self):
        m = gp7_default()
        T = forward_kinematics(m, [0] * 6)
        assert T.shape == (4, 4)
        # Bottom row identity
        np.testing.assert_array_equal(T[3], [0, 0, 0, 1])

    def test_fk_at_zero_joints_within_reach(self):
        """Zero pose: TCP nằm trong reach envelope GP7 (~927mm radius)."""
        m = gp7_default()
        T = forward_kinematics(m, [0] * 6)
        tcp_xyz = T[:3, 3]
        dist = np.linalg.norm(tcp_xyz)
        assert dist < 2000                # within reasonable workspace

    def test_fk_with_base_offset(self):
        """Base offset cộng đúng vào TCP."""
        m1 = gp7_default(base_xyz_mm=(0, 0, 0))
        m2 = gp7_default(base_xyz_mm=(100, 200, 300))
        T1 = forward_kinematics(m1, [0] * 6)
        T2 = forward_kinematics(m2, [0] * 6)
        diff = T2[:3, 3] - T1[:3, 3]
        np.testing.assert_array_almost_equal(diff, [100, 200, 300])

    def test_fk_wrong_joint_count_raises(self):
        m = gp7_default()
        with pytest.raises(ValueError, match="Expected 6"):
            forward_kinematics(m, [0, 0, 0])

    def test_fk_deterministic(self):
        """Same input → same output (sanity: no random state)."""
        m = gp7_default()
        T1 = forward_kinematics(m, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        T2 = forward_kinematics(m, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        np.testing.assert_array_equal(T1, T2)


class TestJointPositions:
    def test_joint_positions_count(self):
        m = gp7_default()
        positions = joint_positions(m, [0] * 6)
        # base + 6 joints (+ TCP nếu tool_offset > 0; default 0 → 7 điểm)
        assert len(positions) >= 7
        assert all(p.shape == (3,) for p in positions)

    def test_first_is_base_origin(self):
        m = gp7_default(base_xyz_mm=(100, 0, 0))
        positions = joint_positions(m, [0] * 6)
        np.testing.assert_array_almost_equal(positions[0], [100, 0, 0])


# ─────────────────────────────────────────────────────────────────────────
# Trajectory interpolation
# ─────────────────────────────────────────────────────────────────────────


class TestInterpolate:
    def test_two_waypoints_linear(self):
        wp = [[0] * 6, [np.pi / 4] * 6]
        samples = interpolate_joints(wp, dt=0.05, max_joint_speed_deg_s=30)
        assert len(samples) >= 5
        # First sample ≈ start
        np.testing.assert_array_almost_equal(samples[0].joints_rad, wp[0])
        # Last sample = end exact
        np.testing.assert_array_almost_equal(samples[-1].joints_rad, wp[1])

    def test_time_increases_monotonic(self):
        wp = [[0] * 6, [0.5] * 6, [1.0] * 6]
        samples = interpolate_joints(wp, dt=0.05, max_joint_speed_deg_s=30)
        times = [s.t for s in samples]
        for i in range(len(times) - 1):
            assert times[i] <= times[i + 1]

    def test_max_speed_respected(self):
        """Joint movement / duration không vượt max_joint_speed."""
        wp = [[0] * 6, [np.pi] * 6]                    # 180° move
        samples = interpolate_joints(wp, dt=0.05, max_joint_speed_deg_s=30)
        # 180° / 30°/s = 6s → samples cover ~6s
        assert samples[-1].t >= 5.5

    def test_one_waypoint_raises(self):
        with pytest.raises(ValueError, match="at least 2 waypoints"):
            interpolate_joints([[0] * 6])

    def test_zero_dt_raises(self):
        with pytest.raises(ValueError, match="dt"):
            interpolate_joints([[0] * 6, [0.1] * 6], dt=0)

    def test_nonpositive_speed_raises(self):
        # Regression: speed 0 used to OverflowError (duration = dq/0 = inf);
        # negative gave a bogus negative-time trajectory. Now a clear ValueError.
        with pytest.raises(ValueError, match="max_joint_speed"):
            interpolate_joints([[0] * 6, [0.1] * 6], dt=0.05, max_joint_speed_deg_s=0)
        with pytest.raises(ValueError, match="max_joint_speed"):
            interpolate_joints([[0] * 6, [0.1] * 6], dt=0.05, max_joint_speed_deg_s=-5)

    def test_identical_waypoints_skipped(self):
        wp = [[0] * 6, [0] * 6, [0.5] * 6]
        samples = interpolate_joints(wp, dt=0.05, max_joint_speed_deg_s=30)
        # Đoạn 0→0 bị skip, chỉ có 0→0.5 + final
        assert len(samples) >= 2


# ─────────────────────────────────────────────────────────────────────────
# Joint limit check
# ─────────────────────────────────────────────────────────────────────────


class TestJointLimits:
    def test_within_limits_no_violation(self):
        m = gp7_default()
        samples = [TrajectorySample(t=0, joints_rad=[0] * 6)]
        assert check_joint_limits(m, samples) == []

    def test_exceeds_limit_detected(self):
        m = gp7_default()
        # S axis limit ±170°. Truyền 200° = vượt
        samples = [TrajectorySample(t=0, joints_rad=[np.deg2rad(200), 0, 0, 0, 0, 0])]
        violations = check_joint_limits(m, samples)
        assert len(violations) == 1
        sample_idx, joint_idx, value = violations[0]
        assert sample_idx == 0
        assert joint_idx == 0
        assert value > np.deg2rad(170)


# ─────────────────────────────────────────────────────────────────────────
# Self-collision sphere check
# ─────────────────────────────────────────────────────────────────────────


class TestSelfCollision:
    def test_normal_pose_no_self_collision(self):
        m = gp7_default()
        samples = [TrajectorySample(t=0, joints_rad=[0] * 6)]
        # Default GP7 default joints — bình thường không self-collide
        violations = check_self_collision_spheres(m, samples)
        # Có thể có vài false positive ở base/J1 do sphere conservative
        # Verify chỉ tối thiểu — không crash, list trả về list[tuple]
        assert isinstance(violations, list)

    def test_extreme_fold_pose_may_collide(self):
        """Gấp cánh tay extreme: J2 + J3 cùng dấu lớn → có khả năng collision."""
        m = gp7_default()
        samples = [
            TrajectorySample(t=0, joints_rad=[
                0, np.deg2rad(120), np.deg2rad(-150), 0, 0, 0,
            ]),
        ]
        violations = check_self_collision_spheres(m, samples)
        # Smoke test: không raise, trả list
        assert isinstance(violations, list)


# ─────────────────────────────────────────────────────────────────────────
# Custom DH model (smoke test framework)
# ─────────────────────────────────────────────────────────────────────────


class TestCustomDH:
    def test_2dof_planar_robot(self):
        """2-DOF planar — verify FK math chứ không phải hard-code GP7."""
        links = (
            DHLink(a=100, alpha=0, d=0, theta_offset=0,
                   joint_min=-np.pi, joint_max=np.pi),
            DHLink(a=50,  alpha=0, d=0, theta_offset=0,
                   joint_min=-np.pi, joint_max=np.pi),
        )
        m = RobotDHModel(name="2DOF", links=links)
        # joints [0, 0] → end effector tại x = 100 + 50 = 150
        T = forward_kinematics(m, [0, 0])
        np.testing.assert_almost_equal(T[0, 3], 150, decimal=4)
        np.testing.assert_almost_equal(T[1, 3], 0, decimal=4)


def test_interpolate_cartesian_straight_line_and_orientation():
    """interpolate_cartesian samples a Cartesian straight line (pos lerp + axis-angle
    orientation slerp), endpoints inclusive (bug #14 predictive-MoveL helper)."""
    import numpy as np
    from src.orchestrator.kinematics import interpolate_cartesian
    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[:3, 3] = [100.0, 0.0, -50.0]                      # pure translation
    # 90° about Z between the two
    T1[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    poses = interpolate_cartesian(T0, T1, n_steps=4)
    assert len(poses) == 5                               # n_steps + 1, inclusive
    np.testing.assert_allclose(poses[0], T0, atol=1e-9)
    np.testing.assert_allclose(poses[-1][:3, 3], T1[:3, 3], atol=1e-6)
    np.testing.assert_allclose(poses[-1][:3, :3], T1[:3, :3], atol=1e-6)
    # midpoint position is exactly halfway (straight line)
    np.testing.assert_allclose(poses[2][:3, 3], [50.0, 0.0, -25.0], atol=1e-6)
    # every sampled rotation stays a valid rotation matrix (orthonormal)
    for T in poses:
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)


def test_interpolate_cartesian_180deg_axis_recovery():
    """At exactly 180° the antisymmetric axis recovery is degenerate — the symmetric
    (R+I) fallback must recover the correct axis (bug #8). 180° about X here."""
    import numpy as np
    from src.orchestrator.kinematics import interpolate_cartesian
    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[:3, :3] = np.array([[1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0],
                           [0.0, 0.0, -1.0]])              # 180° about X
    poses = interpolate_cartesian(T0, T1, n_steps=4)
    np.testing.assert_allclose(poses[-1][:3, :3], T1[:3, :3], atol=1e-6)
    # midpoint = 90° about X (NOT identity / wrong axis): R[1,1]=cos90=0
    mid = poses[2][:3, :3]
    assert abs(mid[1, 1]) < 1e-6 and abs(mid[2, 2]) < 1e-6
    for T in poses:                                         # all valid rotations
        np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-6)


def test_check_joint_limits_tolerates_tiny_overshoot():
    """A solution 1e-10 rad over a limit (IK boundary rounding) must NOT be flagged
    (bug #R4-8 — zero-tolerance vs IK's 1e-9 acceptance)."""
    import numpy as np
    from src.orchestrator.kinematics import (
        check_joint_limits, gp7_default, TrajectorySample)
    m = gp7_default()
    qmax = [lk.joint_max for lk in m.links]
    over = list(qmax); over[0] += 1e-10                  # just over J1 max
    samples = [TrajectorySample(t=0.0, joints_rad=over)]
    assert check_joint_limits(m, samples) == []          # within 1e-9 tolerance
    way_over = list(qmax); way_over[0] += 0.05           # 0.05 rad over → real violation
    assert check_joint_limits(m, [TrajectorySample(t=0.0, joints_rad=way_over)])
