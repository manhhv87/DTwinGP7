"""
test_sim_robot.py
─────────────────
Unit test cho SimRobot — robot giả lập headless (reach model + grasp injection).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.orchestrator.sim_robot import SimRobot


def _pose(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


class TestReachModel:
    def setup_method(self):
        self.robot = SimRobot(base_xyz=(0.0, 0.0, 630.0),
                              reach_max_mm=927.0, reach_min_mm=150.0)

    def test_in_reach_returns_zero(self):
        # (700,0,500): dist từ (0,0,630) ≈ 705mm → trong tầm.
        assert self.robot.MoveJ_Test(None, _pose(700, 0, 500)) == 0

    def test_out_of_reach_returns_negative(self):
        # (1500,0,500): dist ≈ 1577mm > 927 → ngoài tầm.
        assert self.robot.MoveJ_Test(None, _pose(1500, 0, 500)) < 0

    def test_too_close_returns_negative(self):
        # Ngay tại base → dist 0 < reach_min → ngoài tầm.
        assert self.robot.MoveJ_Test(None, _pose(0, 0, 630)) < 0

    def test_solveik_none_when_unreachable(self):
        assert self.robot.SolveIK(_pose(1500, 0, 500)) is None

    def test_solveik_joints_when_reachable(self):
        joints = self.robot.SolveIK(_pose(700, 0, 500))
        assert joints is not None
        assert len(joints) == 6
        assert all(isinstance(j, float) for j in joints)


class TestGraspInjection:
    def test_no_fail_when_rate_zero(self):
        robot = SimRobot(grasp_fail_rate=0.0)
        robot.setDO(1, 1)            # đóng gripper → bật cờ check
        robot.MoveL([0, 0, 0, 0, 0, 0])   # LIFT — không được raise
        # Không exception → pass

    def test_always_fail_when_rate_one(self):
        robot = SimRobot(grasp_fail_rate=1.0, seed=0)
        robot.setDO(1, 1)            # đóng gripper
        with pytest.raises(RuntimeError, match="grasp_slip"):
            robot.MoveL([0, 0, 0, 0, 0, 0])   # LIFT — phải raise

    def test_slip_checked_only_once(self):
        # Sau khi raise/skip 1 lần, các move sau không check nữa.
        robot = SimRobot(grasp_fail_rate=0.0)
        robot.setDO(1, 1)
        robot.MoveL([0] * 6)         # check (rate 0 → no raise), reset cờ
        # Move tiếp theo: cờ đã tắt → an toàn
        robot.MoveL([0] * 6)

    def test_deterministic_with_seed(self):
        r1 = SimRobot(grasp_fail_rate=0.5, seed=123)
        r2 = SimRobot(grasp_fail_rate=0.5, seed=123)
        # Cùng seed → cùng chuỗi quyết định slip.
        results = []
        for r in (r1, r2):
            outcomes = []
            for _ in range(10):
                r.setDO(1, 1)
                try:
                    r.MoveL([0] * 6)
                    outcomes.append(True)
                except RuntimeError:
                    outcomes.append(False)
            results.append(outcomes)
        assert results[0] == results[1]


class TestInterface:
    def test_joints_roundtrip(self):
        robot = SimRobot(home_joints=[10, 20, 30, 40, 50, 60])
        assert robot.JointsHome() == [10, 20, 30, 40, 50, 60]
        robot.MoveJ([1, 2, 3, 4, 5, 6])
        assert robot.Joints() == [1, 2, 3, 4, 5, 6]

    def test_parent_none(self):
        # Parent=None → Orchestrator dùng identity world→base.
        assert SimRobot().Parent() is None

    def test_valid_true(self):
        assert SimRobot().Valid() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
