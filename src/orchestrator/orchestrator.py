"""
orchestrator.py
───────────────
Orchestrator: kết nối Perception ↔ RoboDK, điều khiển chu trình pick-and-place.

Vai trò:
  - Nhận detection (camera frame) từ perception queue
  - Chuyển sang base frame qua ma trận hand-eye
  - Kiểm tra reachability + collision bằng digital twin (RoboDK) TRƯỚC khi gắp
    → đây là "lớp an toàn dựa trên digital twin", đóng góp C2 của paper
  - Thực thi pick-and-place qua RoboDK API (sim hoặc robot thật)
  - Ghi log mọi trial qua TrialLogger

RoboDK được lazy-import → module này import được khi chỉ chạy unit test logic.
"""
from __future__ import annotations

import logging
import queue
import time
from typing import Any

import numpy as np

from .coord_conv import camera_to_base, load_calibration, make_grasp_pose
from .state_machine import PickPlaceStateMachine, PickState

logger = logging.getLogger(__name__)


# Giá trị mặc định cho config — override qua dict truyền vào.
_DEFAULT_CONFIG: dict[str, Any] = {
    "robot_name": "Yaskawa GP7",
    "calibration_path": "config/calibration/T_base_camera.npy",
    "place_position": [400.0, 300.0, 50.0],
    "approach_height_mm": 60.0,
    "gripper_do_index": 1,
    "gripper_delay_s": 0.3,
    "inter_trial_delay_s": 1.0,
    "speed_joint_deg_s": 60.0,
    "speed_linear_mm_s": 80.0,
    "yaw_offset_deg": 0.0,
    "detection_timeout_s": 2.0,
}


class Orchestrator:
    """Điều phối chu trình pick-and-place dựa trên RoboDK digital twin.

    Sử dụng:
        orch = Orchestrator(perception_queue, config, robot=mock_robot)
        stats = orch.run_n_trials(50)

    Args:
        perception_queue: Queue chứa message detection từ PerceptionNode.
        config: Dict cấu hình (xem _DEFAULT_CONFIG).
        robot: (Tùy chọn) Inject sẵn một robot item — dùng cho test với mock.
            Nếu None → tự kết nối RoboDK và tìm robot.
        logger_obj: (Tùy chọn) TrialLogger để ghi kết quả.
    """

    def __init__(
        self,
        perception_queue: queue.Queue,
        config: dict[str, Any] | None = None,
        robot: Any = None,
        logger_obj: Any = None,
    ) -> None:
        self.queue = perception_queue
        self.config = {**_DEFAULT_CONFIG, **(config or {})}
        self.trial_logger = logger_obj

        self._rdk = None
        self.robot = robot
        if self.robot is None:
            self._connect_robodk()

        # Tải ma trận hand-eye calibration.
        self.T_BC = load_calibration(self.config["calibration_path"])
        logger.info("Loaded hand-eye T_BC, translation=%s mm", self.T_BC[:3, 3].round(1))

        self.sm = PickPlaceStateMachine()
        self.stats = {"attempted": 0, "successful": 0, "failed": 0}

    # ────────────────────────────────────────────────────────────
    # Kết nối
    # ────────────────────────────────────────────────────────────

    def _connect_robodk(self) -> None:
        """Kết nối RoboDK và lấy robot item (lazy import)."""
        from robodk.robolink import ITEM_TYPE_ROBOT, Robolink

        self._rdk = Robolink()
        self.robot = self._rdk.Item(self.config["robot_name"], ITEM_TYPE_ROBOT)
        if not self.robot.Valid():
            raise RuntimeError(
                f"Robot '{self.config['robot_name']}' không có trong RoboDK station. "
                f"Chạy scripts/build_station.py trước."
            )
        logger.info("Connected to robot '%s'", self.config["robot_name"])

    # ────────────────────────────────────────────────────────────
    # Helpers chuyển toạ độ + điều khiển
    # ────────────────────────────────────────────────────────────

    def _select_objects(self, det_msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Chuyển detections sang base frame, sắp xếp vật trên-cùng trước.

        Trả về list rỗng nếu không có vật.
        """
        objects = det_msg.get("objects", [])
        for o in objects:
            xyz_cam = np.array(o["pose_camera"][:3], dtype=float)
            o["pose_base"] = camera_to_base(xyz_cam, self.T_BC)
        # Vật có Z cao nhất (gần camera / nằm trên cùng) được gắp trước.
        objects.sort(key=lambda o: o["pose_base"][2], reverse=True)
        return objects

    def _is_reachable(self, target_T: np.ndarray) -> bool:
        """Kiểm tra robot có với tới pose `target_T` (numpy 4x4) không.

        Dùng digital twin: RoboDK.MoveJ_Test trả về < 0 nếu không tới được
        hoặc va chạm. Đây là lớp an toàn chạy TRƯỚC mỗi lần gắp vật lý.
        """
        try:
            target_pose = self._to_robodk_pose(target_T)
            result = self.robot.MoveJ_Test(self.robot.Joints(), target_pose)
            return result == 0
        except Exception as e:  # noqa: BLE001
            logger.warning("MoveJ_Test lỗi: %s", e)
            return False

    def _gripper(self, close: bool) -> None:
        """Đóng/mở gripper qua Digital Output, chờ ổn định."""
        self.robot.setDO(self.config["gripper_do_index"], 1 if close else 0)
        time.sleep(self.config["gripper_delay_s"])

    def _to_robodk_pose(self, T: np.ndarray):
        """Chuyển numpy 4x4 → RoboDK Mat (lazy import)."""
        from robodk.robomath import Mat

        return Mat(T.tolist())

    # ────────────────────────────────────────────────────────────
    # Chu trình chính
    # ────────────────────────────────────────────────────────────

    def run_one_cycle(self, trial_id: int = -1) -> bool:
        """Thực thi một chu trình pick-and-place.

        Returns:
            True nếu gắp-thả thành công, False nếu thất bại/không có vật.
        """
        self.sm.reset()
        t_start = time.time()

        # ─── DETECT ───
        self.sm.transition_to(PickState.DETECT)
        try:
            det_msg = self.queue.get(timeout=self.config["detection_timeout_s"])
        except queue.Empty:
            logger.warning("Trial %d: không nhận được detection", trial_id)
            self._log_trial(trial_id, False, "detection_timeout", t_start, None)
            return False

        objects = self._select_objects(det_msg)
        if not objects:
            logger.info("Trial %d: không phát hiện vật nào", trial_id)
            self.sm.transition_to(PickState.IDLE, "no objects")
            self._log_trial(trial_id, False, "detection_miss", t_start, None)
            return False

        # ─── PLAN: thử từng vật tới khi có vật với tới được ───
        self.sm.transition_to(PickState.PLAN)
        for obj in objects:
            xyz_base = obj["pose_base"]
            yaw = obj["pose_camera"][3]
            grasp_T = make_grasp_pose(xyz_base, yaw, self.config["yaw_offset_deg"])

            if not self._is_reachable(grasp_T):
                logger.info(
                    "Trial %d: vật '%s' không với tới được, bỏ qua",
                    trial_id, obj.get("class_name", "?"),
                )
                continue

            # ─── Thực thi pick-and-place cho vật này ───
            self.stats["attempted"] += 1
            ok = self._execute_pick_place(grasp_T, obj, trial_id)
            self._log_trial(
                trial_id, ok,
                "" if ok else self.sm.history[-1].note,
                t_start, obj,
            )
            return ok

        # Không vật nào với tới được.
        logger.info("Trial %d: mọi vật đều ngoài tầm với", trial_id)
        self.sm.fail("unreachable")
        self._log_trial(trial_id, False, "unreachable", t_start, objects[0])
        return False

    def _execute_pick_place(
        self, grasp_T: np.ndarray, obj: dict[str, Any], trial_id: int
    ) -> bool:
        """Chạy chuỗi chuyển động gắp-thả. Trả về True nếu thành công.

        Mọi pose tính bằng numpy 4x4; chỉ chuyển sang RoboDK Mat ngay trước
        khi gọi robot → logic này test được bằng mock không cần RoboDK.
        """
        dz = self.config["approach_height_mm"]

        # Pose approach/lift = grasp dịch lên theo Z base.
        lift_T = grasp_T.copy()
        lift_T[2, 3] += dz
        place_T = make_grasp_pose(np.array(self.config["place_position"]), 0.0)
        place_lift_T = place_T.copy()
        place_lift_T[2, 3] += dz

        try:
            self._set_speed()

            self.sm.transition_to(PickState.APPROACH)
            self.robot.MoveJ(self._to_robodk_pose(lift_T))

            self.sm.transition_to(PickState.GRASP)
            self.robot.MoveL(self._to_robodk_pose(grasp_T))
            self._gripper(close=True)

            self.sm.transition_to(PickState.LIFT)
            self.robot.MoveL(self._to_robodk_pose(lift_T))

            self.sm.transition_to(PickState.TRANSFER)
            self.robot.MoveJ(self._to_robodk_pose(place_lift_T))

            self.sm.transition_to(PickState.PLACE)
            self.robot.MoveL(self._to_robodk_pose(place_T))
            self._gripper(close=False)

            self.sm.transition_to(PickState.RETREAT)
            self.robot.MoveL(self._to_robodk_pose(place_lift_T))

            self.sm.transition_to(PickState.DONE)
            self.stats["successful"] += 1
            logger.info(
                "Trial %d: gắp-thả '%s' THÀNH CÔNG | stats=%s",
                trial_id, obj.get("class_name", "?"), self.stats,
            )
            return True

        except Exception as e:  # noqa: BLE001
            self.stats["failed"] += 1
            self.sm.fail(f"motion_error: {e}")
            logger.error("Trial %d: pick thất bại — %s", trial_id, e)
            self._return_home()
            return False

    def _set_speed(self) -> None:
        """Áp tốc độ joint/linear từ config."""
        try:
            self.robot.setSpeed(
                self.config["speed_linear_mm_s"],
                self.config["speed_joint_deg_s"],
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("setSpeed bỏ qua: %s", e)

    def _return_home(self) -> None:
        """Đưa robot về home sau lỗi (best-effort)."""
        try:
            self.robot.MoveJ(self.robot.JointsHome())
        except Exception as e:  # noqa: BLE001
            logger.warning("Không về home được: %s", e)

    def run_n_trials(self, n: int) -> dict[str, Any]:
        """Chạy `n` trial liên tiếp, trả về thống kê tổng hợp."""
        logger.info("Bắt đầu %d trials", n)
        for i in range(n):
            logger.info("─── Trial %d/%d ───", i + 1, n)
            self.run_one_cycle(trial_id=i + 1)
            time.sleep(self.config["inter_trial_delay_s"])

        attempted = max(self.stats["attempted"], 1)
        rate = self.stats["successful"] / attempted
        result = {**self.stats, "success_rate": rate}
        logger.info("Hoàn tất. %s", result)
        return result

    # ────────────────────────────────────────────────────────────
    # Logging
    # ────────────────────────────────────────────────────────────

    def _log_trial(
        self,
        trial_id: int,
        success: bool,
        failure_reason: str,
        t_start: float,
        obj: dict[str, Any] | None,
    ) -> None:
        """Ghi 1 dòng kết quả trial nếu có TrialLogger."""
        if self.trial_logger is None:
            return
        self.trial_logger.log_trial(
            trial_id=trial_id,
            success=success,
            class_name=obj.get("class_name", "") if obj else "",
            cycle_time_s=time.time() - t_start,
            failure_reason=failure_reason,
            final_state=self.sm.state.value,
        )
