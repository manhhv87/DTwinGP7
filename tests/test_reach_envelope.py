"""
test_reach_envelope.py
──────────────────────
Verify sphere reachability model client-side.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.orchestrator.backends.reach_envelope import ReachEnvelope


class TestGP7Default:
    def test_default_uses_927_reach(self):
        env = ReachEnvelope.gp7_default()
        assert env.reach_max_mm == 927.0
        assert env.reach_min_mm == 150.0

    def test_custom_base(self):
        env = ReachEnvelope.gp7_default(base_xyz_mm=(0, 0, 630))
        assert env.base_xyz_mm == (0, 0, 630)


class TestCanReach:
    @pytest.fixture
    def env(self):
        return ReachEnvelope(
            base_xyz_mm=(0, 0, 630),
            reach_max_mm=927.0,
            reach_min_mm=150.0,
        )

    def test_within_envelope_ok(self, env):
        # 700mm horizontal từ base = √(700² + 0²) khoảng cách
        target = np.array([700, 0, 500])
        # dist from base = √(700² + 0² + 130²) ≈ 712 → OK (150..927)
        assert env.can_reach(target) is True

    def test_outside_envelope_far_rejected(self, env):
        target = np.array([2000, 0, 0])
        assert env.can_reach(target) is False

    def test_inside_min_dead_zone_rejected(self, env):
        target = np.array([50, 0, 630])           # 50mm từ base — dead zone
        assert env.can_reach(target) is False

    def test_at_max_boundary_ok(self, env):
        # Tại exactly 927mm — inclusive
        target = np.array([927, 0, 630])
        assert env.can_reach(target) is True

    def test_4x4_pose_extracted(self, env):
        T = np.eye(4)
        T[:3, 3] = [700, 0, 500]
        assert env.can_reach(T) is True

    def test_list_extracted(self, env):
        assert env.can_reach([700, 0, 500]) is True

    def test_invalid_target_permissive(self, env):
        # Không extract được → coi như reachable (an toàn cho test code)
        assert env.can_reach("nonsense") is True


class TestDistanceFromBase:
    def test_distance_correct(self):
        env = ReachEnvelope(base_xyz_mm=(0, 0, 0), reach_max_mm=1000, reach_min_mm=0)
        d = env.distance_from_base([3, 4, 0])         # pythagoras
        assert d == pytest.approx(5.0)

    def test_distance_none_for_invalid(self):
        env = ReachEnvelope(base_xyz_mm=(0, 0, 0), reach_max_mm=1000, reach_min_mm=0)
        assert env.distance_from_base("xxx") is None


class TestIntegrationWithHSEBackend:
    """ReachEnvelope wire vào MotomanHSEBackend.MoveJ_Test."""

    def test_movej_test_uses_envelope_when_provided(self):
        from src.orchestrator.backends.motoman_hse import MotomanHSEBackend

        env = ReachEnvelope(base_xyz_mm=(0, 0, 0), reach_max_mm=500, reach_min_mm=100)
        backend = MotomanHSEBackend(ip="x", reach_envelope=env)

        T_in = np.eye(4)
        T_in[:3, 3] = [300, 0, 0]
        assert backend.MoveJ_Test(None, T_in) == 0

        T_out = np.eye(4)
        T_out[:3, 3] = [1000, 0, 0]
        assert backend.MoveJ_Test(None, T_out) == -1

    def test_movej_test_permissive_without_envelope(self):
        from src.orchestrator.backends.motoman_hse import MotomanHSEBackend

        backend = MotomanHSEBackend(ip="x")    # no envelope
        T = np.eye(4)
        T[:3, 3] = [99999, 0, 0]
        assert backend.MoveJ_Test(None, T) == 0    # permissive
