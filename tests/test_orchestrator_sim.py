"""
test_orchestrator_sim.py
────────────────────────
Integration test cho Orchestrator với robot giả lập (MagicMock).

Không cần RoboDK: robot được inject sẵn, _to_robodk_pose bị monkeypatch
thành passthrough numpy → toàn bộ chu trình run_one_cycle chạy được.

Run:
    pytest tests/test_orchestrator_sim.py -v
"""
from __future__ import annotations

import queue
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.orchestrator.coord_conv import save_calibration
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.state_machine import PickState


def make_detection_msg(objects):
    """Tạo message detection theo định dạng perception queue."""
    return {"timestamp": 0.0, "objects": objects, "fps": 20.0}


def make_object(class_name="bottle", xyz=(400, 0, 100), yaw=0.0):
    """Tạo một object detection tối thiểu cho orchestrator."""
    return {
        "class_name": class_name,
        "pose_camera": (xyz[0], xyz[1], xyz[2], yaw),
        "confidence": 0.95,
    }


@pytest.fixture
def calibration_file(tmp_path):
    """File calibration identity → pose_base = pose_camera."""
    path = tmp_path / "T_base_camera.npy"
    save_calibration(path, np.eye(4))
    return path


@pytest.fixture
def mock_robot():
    """Robot giả: mọi chuyển động no-op, MoveJ_Test luôn báo với tới được."""
    robot = MagicMock(name="MockRobot")
    robot.MoveJ_Test.return_value = 0
    return robot


@pytest.fixture
def orchestrator(calibration_file, mock_robot, monkeypatch):
    """Orchestrator sẵn sàng — _to_robodk_pose passthrough numpy."""
    monkeypatch.setattr(
        Orchestrator, "_to_robodk_pose", lambda self, T: np.asarray(T)
    )
    det_queue: queue.Queue = queue.Queue(maxsize=3)
    config = {
        "calibration_path": str(calibration_file),
        "inter_trial_delay_s": 0.0,
        "gripper_delay_s": 0.0,
        "skip_reachability_check": False,    # test verify hành vi reach check
    }
    return Orchestrator(det_queue, config=config, robot=mock_robot)


class TestSelectObjects:
    def test_transforms_to_base_frame(self, orchestrator):
        msg = make_detection_msg([make_object(xyz=(100, 200, 300))])
        objs = orchestrator._select_objects(msg)
        # Calibration identity → pose_base == pose_camera xyz.
        np.testing.assert_array_almost_equal(objs[0]["pose_base"], [100, 200, 300])

    def test_sorts_top_object_first(self, orchestrator):
        msg = make_detection_msg([
            make_object("bottle", xyz=(400, 0, 50)),
            make_object("cup", xyz=(400, 0, 250)),
        ])
        objs = orchestrator._select_objects(msg)
        assert objs[0]["class_name"] == "cup"  # Z cao hơn → gắp trước


class TestObjectMirror:
    """Real mode: detection update RoboDK item pose để twin khớp camera view."""

    def test_object_pose_mirrored_to_robodk_item(
        self, calibration_file, mock_robot, monkeypatch
    ):
        # Setup orchestrator với robodk_objects (mock item)
        monkeypatch.setattr(
            Orchestrator, "_to_robodk_pose", lambda self, T: np.asarray(T)
        )
        mock_item = MagicMock(name="trayItem")
        orch = Orchestrator(
            queue.Queue(maxsize=3),
            config={"calibration_path": str(calibration_file),
                    "inter_trial_delay_s": 0, "gripper_delay_s": 0},
            robot=mock_robot,
            robodk_objects={"tray": mock_item},
        )
        msg = make_detection_msg([
            make_object("tray", xyz=(500, 200, 600), yaw=0.5),
        ])
        orch._select_objects(msg)
        # setPose được gọi với pose 4x4 chứa xyz và yaw
        mock_item.setPose.assert_called_once()
        pose_arg = mock_item.setPose.call_args[0][0]
        np.testing.assert_array_almost_equal(pose_arg[:3, 3], [500, 200, 600])
        # Top-left 2x2 phản ánh rotation Z theo yaw
        assert abs(pose_arg[0, 0] - np.cos(0.5)) < 1e-6
        assert abs(pose_arg[1, 0] - np.sin(0.5)) < 1e-6

    def test_no_robodk_objects_skips_mirror(self, orchestrator):
        """Không có robodk_objects → _select_objects vẫn chạy."""
        msg = make_detection_msg([make_object("tray", xyz=(100, 0, 50))])
        objs = orchestrator._select_objects(msg)
        assert len(objs) == 1                       # no crash

    def test_unknown_class_name_no_setpose(
        self, calibration_file, mock_robot, monkeypatch
    ):
        monkeypatch.setattr(
            Orchestrator, "_to_robodk_pose", lambda self, T: np.asarray(T)
        )
        mock_item = MagicMock()
        orch = Orchestrator(
            queue.Queue(maxsize=3),
            config={"calibration_path": str(calibration_file)},
            robot=mock_robot,
            robodk_objects={"tray": mock_item},
        )
        # Detection của 'cup' không có trong robodk_objects
        msg = make_detection_msg([make_object("cup", xyz=(100, 0, 50))])
        orch._select_objects(msg)
        mock_item.setPose.assert_not_called()       # tray item KHÔNG bị đổi pose

    def test_attached_object_not_updated(self, calibration_file, mock_robot, monkeypatch):
        """Khi object đang được attach vào gripper → không update pose."""
        monkeypatch.setattr(
            Orchestrator, "_to_robodk_pose", lambda self, T: np.asarray(T)
        )
        mock_item = MagicMock()
        orch = Orchestrator(
            queue.Queue(maxsize=3),
            config={"calibration_path": str(calibration_file)},
            robot=mock_robot,
            robodk_objects={"tray": mock_item},
        )
        orch._attached_obj = mock_item                   # đã gắp
        msg = make_detection_msg([make_object("tray", xyz=(100, 0, 50))])
        orch._select_objects(msg)
        mock_item.setPose.assert_not_called()


class TestRunOneCycle:
    def test_successful_pick(self, orchestrator):
        orchestrator.queue.put(make_detection_msg([make_object()]))
        ok = orchestrator.run_one_cycle(trial_id=1)
        assert ok is True
        assert orchestrator.stats["successful"] == 1
        assert orchestrator.sm.state == PickState.DONE

    def test_no_objects_fails_gracefully(self, orchestrator):
        orchestrator.queue.put(make_detection_msg([]))
        ok = orchestrator.run_one_cycle(trial_id=1)
        assert ok is False
        assert orchestrator.stats["attempted"] == 0

    def test_empty_queue_times_out(self, orchestrator):
        orchestrator.config["detection_timeout_s"] = 0.1
        ok = orchestrator.run_one_cycle(trial_id=1)
        assert ok is False

    def test_unreachable_object_skipped(self, orchestrator, mock_robot):
        mock_robot.MoveJ_Test.return_value = -1  # ngoài tầm với
        orchestrator.queue.put(make_detection_msg([make_object()]))
        ok = orchestrator.run_one_cycle(trial_id=1)
        assert ok is False
        assert orchestrator.stats["successful"] == 0


class TestRunNTrials:
    def test_runs_all_trials(self, orchestrator):
        for _ in range(3):
            orchestrator.queue.put(make_detection_msg([make_object()]))
        stats = orchestrator.run_n_trials(3)
        assert stats["attempted"] == 3
        assert stats["successful"] == 3
        assert stats["success_rate"] == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
