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
