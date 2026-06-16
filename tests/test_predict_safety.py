"""
test_predict_safety.py
──────────────────────
Verify UC2 — Orchestrator._predict_safety check joint limit + self-collision
trên trajectory đầy đủ.
"""
from __future__ import annotations

import queue
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.orchestrator.coord_conv import save_calibration
from src.orchestrator.orchestrator import Orchestrator


@pytest.fixture
def calibration_file(tmp_path):
    path = tmp_path / "T_base_camera.npy"
    save_calibration(path, np.eye(4))
    return path


@pytest.fixture
def orchestrator_with_predict(calibration_file):
    return Orchestrator(
        queue.Queue(maxsize=3),
        config={
            "calibration_path": str(calibration_file),
            "predictive_safety_enabled": True,
        },
        robot=MagicMock(),
    )


@pytest.fixture
def orchestrator_no_predict(calibration_file):
    return Orchestrator(
        queue.Queue(maxsize=3),
        config={
            "calibration_path": str(calibration_file),
            "predictive_safety_enabled": False,
        },
        robot=MagicMock(),
    )


class TestPredictSafety:
    def test_disabled_returns_none(self, orchestrator_no_predict):
        # Mọi trajectory đều pass khi disabled
        bad_traj = [[0] * 6, [np.deg2rad(500)] * 6]      # vượt limit
        assert orchestrator_no_predict._predict_safety(bad_traj) is None

    def test_safe_trajectory_returns_none(self, orchestrator_with_predict):
        safe_traj = [
            [0, np.deg2rad(30), np.deg2rad(-60), 0, np.deg2rad(60), 0],
            [np.deg2rad(10), np.deg2rad(40), np.deg2rad(-50), 0,
             np.deg2rad(50), np.deg2rad(10)],
        ]
        assert orchestrator_with_predict._predict_safety(safe_traj) is None

    def test_joint_limit_violation_detected(self, orchestrator_with_predict):
        # S axis limit ±170°. Target 200° vượt.
        bad_traj = [
            [0] * 6,
            [np.deg2rad(200), 0, 0, 0, 0, 0],
        ]
        reason = orchestrator_with_predict._predict_safety(bad_traj)
        assert reason is not None
        assert "joint_limit" in reason
        assert "J1" in reason

    def test_too_few_waypoints_returns_none(self, orchestrator_with_predict):
        # 1 waypoint — không đủ để predict
        assert orchestrator_with_predict._predict_safety([[0] * 6]) is None
        assert orchestrator_with_predict._predict_safety([]) is None

    def test_speed_param_respected(self, calibration_file):
        # Đường đi tốc độ cao → interpolation ít sample hơn → cũng OK
        orch = Orchestrator(
            queue.Queue(maxsize=3),
            config={
                "calibration_path": str(calibration_file),
                "predictive_safety_enabled": True,
                "predict_max_speed_deg_s": 120.0,
            },
            robot=MagicMock(),
        )
        traj = [[0] * 6, [np.deg2rad(30)] * 6]
        result = orch._predict_safety(traj)
        assert result is None                            # safe, fast speed


class TestYrcToolFrameConsistency:
    """The yrc/Cartesian path must retract the TCP target to the FLANGE using the
    SAME app tool offset the client path uses (so neither relies on the controller
    TOOL config). _build_backend uses TOOL00 to match."""

    def _orch(self, calibration_file, tool_offset):
        return Orchestrator(
            queue.Queue(maxsize=3),
            config={
                "calibration_path": str(calibration_file),
                "use_yrc_ik": True,
                "robot_tool_offset_mm": tool_offset,
                "robot_base_xyz_mm": (0.0, 0.0, 0.0),
                "robot_base_rpy_deg": (0.0, 0.0, 0.0),
            },
            robot=MagicMock(),
        )

    @staticmethod
    def _topdown(x, y, z):
        T = np.eye(4)
        T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])  # tool Z = -world Z
        T[:3, 3] = [x, y, z]
        return T

    def test_yrc_retracts_tcp_to_flange(self, calibration_file):
        orch = self._orch(calibration_file, tool_offset=100.0)
        kind, T_base = orch._solve_ik_routed(self._topdown(400.0, 0.0, 300.0))
        assert kind == "YRC_POSE"
        # tool Z points down → flange is 100mm ABOVE the TCP target (Z 300→400).
        assert T_base[2, 3] == pytest.approx(400.0, abs=1e-6)
        assert T_base[0, 3] == pytest.approx(400.0, abs=1e-6)

    def test_yrc_no_offset_sends_tcp_unchanged(self, calibration_file):
        orch = self._orch(calibration_file, tool_offset=0.0)
        _, T_base = orch._solve_ik_routed(self._topdown(400.0, 0.0, 300.0))
        assert T_base[2, 3] == pytest.approx(300.0, abs=1e-6)   # no retract


class TestSelectObjectsMalformed:
    def test_malformed_pose_camera_skipped_not_crash(self, orchestrator_with_predict):
        """A detection with a short/missing pose_camera must be SKIPPED, not crash
        the whole trial loop (bug #26 — PLAN reads pose_camera[3])."""
        orch = orchestrator_with_predict
        det = {"objects": [
            {"pose_camera": [0.1, 0.2, 0.5, 1.0], "class_name": "good"},
            {"pose_camera": [0.1, 0.2], "class_name": "short"},     # len < 4
            {"class_name": "missing"},                              # no pose_camera
        ]}
        objs = orch._select_objects(det)
        assert [o["class_name"] for o in objs] == ["good"]


class TestPredictTrajectoryFailSafe:
    def test_ik_fail_rejects_not_fail_open(self, orchestrator_with_predict):
        """A pose the predictor can't IK-solve must be REJECTED (reason string),
        NOT silently skipped (the old fail-OPEN 'return None' = proceed). #1/#14."""
        orch = orchestrator_with_predict
        orch._current_joints = [0.0] * 6
        orch._solve_ik_client = lambda T, seed_deg=None: None   # IK always fails
        T = np.eye(4); T[:3, 3] = [400.0, 0.0, 300.0]
        reason = orch._predict_safety_for_trajectory([(T, "movj")])
        assert reason is not None and "ik_unreachable" in reason

    def test_predict_seeds_from_running_prev_joints(self, orchestrator_with_predict):
        """The prediction loop threads the RUNNING prev joints into _solve_ik_client
        (seed chaining like execution), not always self._current_joints. #2."""
        orch = orchestrator_with_predict
        orch._current_joints = [0.0] * 6
        seeds_seen = []

        def fake_ik(T, seed_deg=None):
            seeds_seen.append(seed_deg)
            return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]          # constant solution
        orch._solve_ik_client = fake_ik
        T0 = np.eye(4); T0[:3, 3] = [400.0, 0.0, 300.0]
        T1 = np.eye(4); T1[:3, 3] = [410.0, 0.0, 300.0]
        orch._predict_safety_for_trajectory([(T0, "movj"), (T1, "movj")])
        # First seed = current ([0]*6); second seed = the prev solution (chained).
        assert seeds_seen[0] == [0.0] * 6
        assert seeds_seen[-1] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
