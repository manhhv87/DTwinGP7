"""
test_orchestrator_sim.py
────────────────────────
Integration test cho Orchestrator với robot giả lập (MagicMock).

Mock robot duck-types interface (Joints, MoveJ, MoveL, MoveJ_Test, setDO,
setSpeed). Tất cả pose ở world-frame numpy 4x4 — backend tự lo frame conversion.

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
def orchestrator(calibration_file, mock_robot):
    """Orchestrator sẵn sàng. Robot mock duck-types backend interface.

    Robot base ở (0,0,630) mm (pedestal — khớp cell_layout_real.yaml) để
    reach envelope (927 mm) cover được toàn bộ workspace bàn (xyz mặc định
    place_position=(700,120,700)).

    Dùng use_yrc_ik=True → MoveJ nhận pose 4x4 trực tiếp (controller-side IK
    convention), không cần client DLS converge — phù hợp test với MagicMock.
    """
    det_queue: queue.Queue = queue.Queue(maxsize=3)
    config = {
        "calibration_path": str(calibration_file),
        "inter_trial_delay_s": 0.0,
        "gripper_delay_s": 0.0,
        "skip_reachability_check": False,    # test verify hành vi reach check
        "use_yrc_ik": True,                  # MoveJ pose passthrough
        "robot_base_xyz_mm": (0.0, 0.0, 630.0),  # khớp pedestal cell layout
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


class TestGripperCcLink:
    """CC-Link path: 2 solenoid (clamp/unclamp) + 3 sensors (X503/X504/X505)."""

    def _make_orch(self, calibration_file, mock_robot, **cc_overrides):
        cc = {
            "clamp_bit": 30010, "unclamp_bit": 30011,
            "clamp_sensor_bit": 30050, "unclamp_sensor_bit": 30051,
            "detect_bit": 30052,
            "wait_sensor_timeout_s": 0.1,
            "wait_sensor_poll_s": 0.01,
            "require_detect_on_close": True,
            **cc_overrides,
        }
        # MagicMock has set_io/read_io auto-created — read_io returns 1 by default
        mock_robot.read_io = MagicMock(return_value=1)
        mock_robot.set_io = MagicMock()
        return Orchestrator(
            queue.Queue(maxsize=3),
            config={"calibration_path": str(calibration_file),
                    "gripper_cc_link": cc, "gripper_delay_s": 0.0,
                    "inter_trial_delay_s": 0.0},
            robot=mock_robot,
        )

    def test_close_sets_clamp_bit_after_unclamp_off(self, calibration_file, mock_robot):
        """An toàn cylinder: tắt UnClamp trước → bật Clamp."""
        orch = self._make_orch(calibration_file, mock_robot)
        orch._gripper(close=True, obj_class=None)
        # Order: set_io(unclamp_bit, 0) trước set_io(clamp_bit, 1)
        calls = mock_robot.set_io.call_args_list
        assert calls[0].args == (30011, 0)              # unclamp OFF
        assert calls[1].args == (30010, 1)              # clamp ON

    def test_open_sets_unclamp_bit_after_clamp_off(self, calibration_file, mock_robot):
        orch = self._make_orch(calibration_file, mock_robot)
        orch._gripper(close=False, obj_class=None)
        calls = mock_robot.set_io.call_args_list
        assert calls[0].args == (30010, 0)              # clamp OFF
        assert calls[1].args == (30011, 1)              # unclamp ON

    def test_grasp_fail_when_detect_sensor_off(self, calibration_file, mock_robot):
        """X505 detect OFF khi close → raise grasp_failed."""
        orch = self._make_orch(calibration_file, mock_robot)
        # read_io trả 1 cho position sensor (clamp_sensor), 0 cho detect
        def read_io_mock(bit):
            return 0 if bit == 30052 else 1
        mock_robot.read_io.side_effect = read_io_mock
        with pytest.raises(RuntimeError, match="grasp_failed"):
            orch._gripper(close=True, obj_class=None)

    def test_sensor_timeout_raises(self, calibration_file, mock_robot):
        """Position sensor không ON trong timeout → raise gripper_timeout."""
        orch = self._make_orch(calibration_file, mock_robot)
        mock_robot.read_io.return_value = 0              # never ON
        with pytest.raises(RuntimeError, match="gripper_timeout"):
            orch._gripper(close=True, obj_class=None)

    def test_require_detect_false_allows_no_object(self, calibration_file, mock_robot):
        """require_detect_on_close=False → skip detect verify."""
        orch = self._make_orch(calibration_file, mock_robot, require_detect_on_close=False)
        mock_robot.read_io.side_effect = lambda bit: 0 if bit == 30052 else 1
        orch._gripper(close=True, obj_class=None)        # no raise


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

    def test_unreachable_object_skipped(self, orchestrator):
        """Reach envelope của GP7 ~927mm từ base (0,0,630). Object ở
        (4000, 0, 0) cách base sqrt(4000² + 630²) ≈ 4050 mm — chắc chắn ngoài."""
        orchestrator.queue.put(make_detection_msg([make_object(xyz=(4000, 0, 0))]))
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
