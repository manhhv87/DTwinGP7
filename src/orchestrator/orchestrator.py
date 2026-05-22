"""
orchestrator.py
───────────────
Orchestrator: kết nối Perception ↔ robot backend, điều khiển chu trình pick-and-place.

Vai trò:
  - Nhận detection (camera frame) từ perception queue
  - Chuyển sang base frame qua ma trận hand-eye
  - Kiểm tra reachability bằng backend (SimRobot reach envelope hoặc predictive
    safety pure-Python) → đóng góp C2 của paper
  - Thực thi pick-and-place qua robot backend (SimRobot sim hoặc DigitalTwin
    + HSE real)
  - Ghi log mọi trial qua TrialLogger

Backend cần inject sẵn (`robot=` arg). Module này KHÔNG còn phụ thuộc RoboDK —
SolveIK đi qua URDF chain pure-Python (verified match RoboDK 0.00mm) hoặc YRC1000
controller-side (Cartesian path).
"""
from __future__ import annotations

import contextlib
import logging
import queue
import time
from typing import Any

import numpy as np

from .coord_conv import camera_to_base, load_calibration, make_grasp_pose
from .state_machine import PickPlaceStateMachine, PickState

logger = logging.getLogger(__name__)


# Giá trị mặc định cho config — override qua dict truyền vào.
# place_position phải khớp PlaceZone trong cell_layout.yaml (mm, base frame).
_DEFAULT_CONFIG: dict[str, Any] = {
    "robot_name": "Yaskawa GP7",
    "calibration_path": "config/calibration/T_base_camera.npy",
    "place_position": [700.0, 120.0, 700.0],
    # Lift cao trên grasp/place để robot tiếp cận thẳng đứng (MoveL xuống),
    # arm không quét ngang workspace.
    "approach_height_mm": 200.0,
    # Offset trừ vào Z grasp pose để fingertip vào giữa thân object thay vì
    # đỉnh (postprocess.deproject trả về tọa độ TOP). Adaptive theo height.
    "grasp_depth_offset_mm": 50.0,
    # Hard clamp: fingertip TCP không bao giờ thấp hơn (table_top + safety),
    # tránh xuyên bàn dù offset adaptive sai.
    "table_top_z_mm": 500.0,
    "table_safety_margin_mm": 100.0,
    # Bù lệch yaw giữa PCA major axis trên mask và trục mở gripper. Yaw=0:
    # gripper jaws spread cùng hướng PCA major axis (= longest object dim).
    # Yaw=90: jaws spread vuông góc PCA (= shortest object dim) — dùng cho
    # khay Galaxy S23: PCA major = 180mm (chiều dài), jaws phải spread theo
    # 100mm (chiều rộng) → cần yaw_offset 90°.
    "yaw_offset_deg": 90.0,
    # Về home sau mỗi success → trial kế APPROACH từ trên cao xuống (không
    # "đi ngang" từ place_lift đến lift mới). Tốn ~2 API call/trial nhưng
    # cần cho video demo trông tự nhiên.
    "return_home_after_success": True,
    # ─── Gripper config ───
    # 2 chế độ:
    #   1. Single-bit toggle (default, sim + legacy): `gripper_do_index` 1 bit
    #   2. Dual-solenoid + sensors (CC-Link real): set `gripper_cc_link` dict
    #      trong config experiment.yaml hoặc CLI. Khi có → orchestrator dùng
    #      pattern double-acting với feedback sensors thay vì blind delay.
    "gripper_do_index": 1,
    "gripper_delay_s": 0.5,                 # fallback blind delay (no sensor)
    "gripper_cc_link": None,                # opt-in: set dict cho CC-Link path
    "inter_trial_delay_s": 1.0,
    "speed_joint_deg_s": 60.0,
    "speed_linear_mm_s": 80.0,
    "detection_timeout_s": 2.0,
    # Sim mặc định SKIP reachability check (MoveJ tự raise nếu out-of-reach).
    # Real mode nên đổi thành False để bật C2 (predictive safety layer).
    "skip_reachability_check": True,
    "is_real_mode": False,
    # UC2 — Pure-Python predictive simulation TRƯỚC khi gửi MoveJ tới robot.
    # Verify joint limit + self-collision đầy đủ trajectory. Khi True:
    # pre-check ~50ms/trial, reject trial nếu predicted unsafe.
    "predictive_safety_enabled": False,
    # Max joint speed (deg/s) dùng cho predict interpolation. Phải khớp tốc độ
    # thực tế của robot để predict đúng motion sequence.
    "predict_max_speed_deg_s": 30.0,
    # Khi True: dùng numerical IK client-side (URDF chain DLS pure-Python).
    # Default cho mọi backend non-YRC.
    "use_client_ik": False,
    # Base pose của robot trong world frame (mm + radian). Dùng để init URDF
    # model với base offset đúng cho client IK.
    "robot_base_xyz_mm": (0.0, 0.0, 0.0),
    "robot_base_rpy_deg": (0.0, 0.0, 0.0),
    # Tool TCP offset (mm) theo Z của flange. Phải khớp gripper TCP trong
    # cell config để IK trả đúng joints cho fingertip pose.
    "robot_tool_offset_mm": 0.0,
    # Khi True: gửi pose Cartesian thẳng tới YRC1000 qua HSE BASE position
    # variable → controller tự IK. Recommended cho HSE real mode. Override
    # use_client_ik.
    "use_yrc_ik": False,
}


class Orchestrator:
    """Điều phối chu trình pick-and-place qua robot backend (SimRobot hoặc HSE).

    Sử dụng:
        orch = Orchestrator(perception_queue, config, robot=sim_robot)
        stats = orch.run_n_trials(50)

    Args:
        perception_queue: Queue chứa message detection từ PerceptionNode.
        config: Dict cấu hình (xem _DEFAULT_CONFIG).
        robot: Backend duck-typed (SimRobot hoặc DigitalTwinMirror). BẮT BUỘC
            inject — không còn auto-connect tới RoboDK.
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

        if robot is None:
            raise ValueError(
                "Orchestrator yêu cầu robot backend (SimRobot hoặc DigitalTwinMirror) "
                "qua arg `robot=`. RoboDK auto-connect đã bị loại bỏ."
            )
        self.robot = robot

        self.T_BC = load_calibration(self.config["calibration_path"])
        logger.info("Loaded hand-eye T_BC, translation=%s mm", self.T_BC[:3, 3].round(1))

        self.sm = PickPlaceStateMachine()
        self.stats = {"attempted": 0, "successful": 0, "failed": 0}
        self._current_joints: list[float] | None = None       # cache cho SolveIK ref

        self._set_speed()

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
        """Kiểm tra pose `target_T` (numpy 4x4) trong reach envelope GP7.

        Dùng sphere envelope client-side (`backends.reach_envelope`) — pure
        Python, không phụ thuộc backend. SimRobot có MoveJ_Test riêng (sphere
        envelope nội bộ) nhưng DigitalTwinMirror.MoveJ_Test giờ là no-op → cần
        check client-side ở orchestrator level.

        Đây là layer pre-filter NHẸ (0 IK call). Predictive safety C2 (joint
        limit + self-collision toàn trajectory) chạy sau trong
        `_execute_pick_place` nếu `predictive_safety_enabled=True`.
        """
        try:
            from .backends.reach_envelope import ReachEnvelope
        except ImportError:
            return True

        base_xyz = tuple(self.config.get("robot_base_xyz_mm", (0.0, 0.0, 0.0)))
        if not hasattr(self, "_reach_env_cached"):
            self._reach_env_cached = ReachEnvelope.gp7_default(base_xyz_mm=base_xyz)

        target_xyz = np.asarray(target_T)[:3, 3]
        if not self._reach_env_cached.can_reach(target_xyz):
            logger.info(
                "Reach envelope fail tại world %s (base=%s)",
                target_xyz.round(1).tolist(), list(base_xyz),
            )
            return False
        return True

    def _gripper(self, close: bool, obj_class: str | None = None) -> None:
        """Đóng/mở gripper. 2 paths tùy config:

        **Path A — CC-Link double-acting + feedback** (real, có PLC + sensors):
        Khi `config["gripper_cc_link"]` set:
          1. Mutually exclusive: tắt solenoid kia trước rồi bật solenoid này
             (an toàn cylinder — không drive 2 chiều cùng lúc)
          2. Wait sensor position bit confirm cylinder đã đến vị trí (clamp/unclamp)
          3. (Close only) verify detect sensor X505 ON → vật trong gripper
             → fail "grasp_failed" nếu OFF (config require_detect_on_close=True)

        **Path B — Single-bit blind delay** (sim, legacy, no PLC feedback):
        Fallback `setDO(gripper_do_index, ...)` + `gripper_delay_s` cố định.

        Sim mode: kèm attach/detach object item vào gripper tool qua
        `setParentStatic` để object visually di chuyển theo gripper.
        """
        cc = self.config.get("gripper_cc_link")
        if cc and hasattr(self.robot, "set_io") and hasattr(self.robot, "read_io"):
            self._gripper_cc_link(close, obj_class, cc)
        else:
            self._gripper_simple(close, obj_class)

    def _gripper_simple(self, close: bool, obj_class: str | None) -> None:
        """Path B: 1-bit toggle + blind delay (legacy / sim path)."""
        self.robot.setDO(self.config["gripper_do_index"], 1 if close else 0)
        # Visual attach/detach (sim viewport) — gọi backend nếu support
        self._notify_viewport_grasp(close, obj_class)
        self._robot_timer(self.config["gripper_delay_s"])

    def _gripper_cc_link(
        self, close: bool, obj_class: str | None, cc: dict,
    ) -> None:
        """Path A: double-acting solenoid + sensor feedback qua CC-Link."""
        if close:
            # Sequence an toàn: tắt UnClamp trước, BẬT Clamp
            self.robot.set_io(cc["unclamp_bit"], 0)
            self.robot.set_io(cc["clamp_bit"], 1)
            sensor_bit = cc["clamp_sensor_bit"]
            action_name = "Clamp"
        else:
            self.robot.set_io(cc["clamp_bit"], 0)
            self.robot.set_io(cc["unclamp_bit"], 1)
            sensor_bit = cc["unclamp_sensor_bit"]
            action_name = "UnClamp"

        # Wait sensor confirm cylinder đã đến vị trí (KHÔNG dùng blind delay)
        if not self._wait_sensor_on(
            sensor_bit,
            timeout_s=float(cc.get("wait_sensor_timeout_s", 2.0)),
            poll_s=float(cc.get("wait_sensor_poll_s", 0.05)),
        ):
            raise RuntimeError(
                f"gripper_timeout: {action_name} sensor (bit {sensor_bit}) "
                f"không ON sau {cc.get('wait_sensor_timeout_s', 2.0)}s"
            )

        # Verify grasp khi close (require detect sensor X505)
        if close and cc.get("require_detect_on_close", True):
            detect = self.robot.read_io(cc["detect_bit"])
            if detect != 1:
                raise RuntimeError(
                    f"grasp_failed: detect sensor (bit {cc['detect_bit']}) "
                    f"OFF — vật không trong gripper"
                )

        # Visual object attach/detach (sim viewport)
        self._notify_viewport_grasp(close, obj_class)

    def _notify_viewport_grasp(self, close: bool, obj_class: str | None) -> None:
        """Notify backend viewport (nếu support) khi gripper đóng/mở.

        Viewport (Open3D SimRobot/Mirror) dùng signal này để attach/detach
        object mesh khỏi tool. Backend không có → no-op.
        """
        if close and obj_class and hasattr(self.robot, "attach_object"):
            try:
                self.robot.attach_object(obj_class)
            except Exception as e:  # noqa: BLE001
                logger.debug("viewport attach_object lỗi: %s", e)
        elif not close and hasattr(self.robot, "detach_object"):
            try:
                self.robot.detach_object()
            except Exception as e:  # noqa: BLE001
                logger.debug("viewport detach_object lỗi: %s", e)

    def _wait_sensor_on(
        self, bit_addr: int, timeout_s: float, poll_s: float,
    ) -> bool:
        """Poll sensor bit cho tới khi == 1 hoặc timeout. Returns True nếu ON."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            val = self.robot.read_io(bit_addr)
            if val == 1:
                return True
            time.sleep(poll_s)
        return False

    def _robot_timer(self, seconds: float) -> None:
        """Sleep với batch awareness — INFORM TIMER trong batch, time.sleep ngoài."""
        if hasattr(self.robot, "timer") and callable(self.robot.timer):
            self.robot.timer(seconds)
        else:
            time.sleep(seconds)

    def _robot_batch_ctx(self) -> Any:
        """Context manager batch nếu backend support (HSE), nullcontext otherwise.

        Cho phép `_execute_pick_place` gom 5-7 motion call thành 1 INFORM upload —
        giảm overhead từ ~1-2s/trial xuống ~200ms/trial (HSE backend M3).
        """
        if hasattr(self.robot, "batch") and callable(self.robot.batch):
            return self.robot.batch()
        return contextlib.nullcontext()

    def _predict_safety(self, joint_waypoints_rad: list[list[float]]) -> str | None:
        """UC2 — Pre-execute trajectory prediction.

        Run pure-Python FK trên trajectory đầy đủ → check joint limit +
        self-collision toàn bộ path. Tốn ~50ms/trial nhưng catch được unsafe
        path mà single-point MoveJ_Test miss.

        Args:
            joint_waypoints_rad: List 6-tuple joint configs (radian).

        Returns:
            None nếu safe. String mô tả vi phạm nếu unsafe.
        """
        if not self.config.get("predictive_safety_enabled", False):
            return None

        try:
            from src.orchestrator.kinematics import (
                check_joint_limits, check_self_collision_spheres,
                gp7_default, interpolate_joints,
            )
        except ImportError as e:                    # noqa: BLE001
            logger.debug("Kinematics module không import được: %s", e)
            return None

        if len(joint_waypoints_rad) < 2:
            return None                              # Không đủ waypoint để predict

        model = gp7_default()
        samples = interpolate_joints(
            joint_waypoints_rad, dt=0.05,
            max_joint_speed_deg_s=self.config["predict_max_speed_deg_s"],
        )

        # Joint limits
        violations = check_joint_limits(model, samples)
        if violations:
            s_idx, j_idx, val = violations[0]
            return (f"predicted_joint_limit: J{j_idx+1}={np.rad2deg(val):.1f}° "
                    f"@ sample {s_idx}/{len(samples)}")

        # Self-collision (sphere-based, conservative)
        collisions = check_self_collision_spheres(model, samples)
        if collisions:
            s_idx, i, j, dist = collisions[0]
            return (f"predicted_self_collision: joint {i} vs {j} dist={dist:.0f}mm "
                    f"@ sample {s_idx}/{len(samples)}")

        return None

    def _predict_safety_for_trajectory(
        self, target_T_world_list: list[np.ndarray]
    ) -> str | None:
        """Build joint trajectory từ list pose world + check predictive safety.

        Solve IK client-side cho từng pose (URDF DLS), prepend joints hiện tại,
        rồi gọi `_predict_safety`. Short-circuit return None nếu predictive
        safety disabled hoặc IK fail (để main flow tự handle).
        """
        if not self.config.get("predictive_safety_enabled", False):
            return None
        if not target_T_world_list:
            return None

        # Build joint waypoints (radian) — current first, then each target via IK
        waypoints_rad: list[list[float]] = []
        if self._current_joints is not None:
            waypoints_rad.append([np.deg2rad(q) for q in self._current_joints])
        for T in target_T_world_list:
            joints_deg = self._solve_ik_client(T)
            if joints_deg is None:
                # IK fail — không predict được, để main flow raise tự nhiên
                logger.debug("Predictive safety skip: IK fail tại world %s",
                             T[:3, 3].round(1).tolist())
                return None
            waypoints_rad.append([np.deg2rad(q) for q in joints_deg])

        return self._predict_safety(waypoints_rad)

    @staticmethod
    def _joints_to_list(joints) -> list[float] | None:
        """Convert iterable joints → flat list of 6 floats."""
        if joints is None:
            return None
        try:
            data = list(joints)
        except TypeError:
            return None
        if not data:
            return None
        if isinstance(data[0], (int, float)) and len(data) >= 6:
            try:
                return [float(j) for j in data[:6]]
            except (TypeError, ValueError):
                return None
        return None

    def _normalize_target_joints(self, target_joints):
        """Wrap mỗi joint của target ±360° để gần `_current_joints` nhất.

        MoveJ nội suy LINEAR trong joint space. Nếu home_joints[J4]=-180 và
        current J4=+180 (cùng orientation thực tế), MoveJ sẽ quay -360° → "chong
        chóng". Wrap target về phía gần current pick đường ngắn nhất.

        Yaskawa GP7 joint ranges lớn (R/T tới ±360) cho phép nhiều biểu diễn
        cùng một orientation — normalize bắt buộc để tránh full-rotation.
        """
        if self._current_joints is None or target_joints is None:
            return target_joints
        result = list(target_joints)
        n = min(len(result), len(self._current_joints))
        for i in range(n):
            cur = float(self._current_joints[i])
            tgt = float(result[i])
            while tgt - cur > 180.0:
                tgt -= 360.0
            while tgt - cur < -180.0:
                tgt += 360.0
            result[i] = tgt
        return result

    def _solve_ik_client(self, target_T_world: np.ndarray) -> list[float] | None:
        """Numerical IK client-side qua kinematics module (DLS pure-Python).

        URDF chain verified match RoboDK SolveFK 0.00mm — không cần RoboDK fallback.

        Args:
            target_T_world: 4x4 pose trong WORLD frame (orchestrator native).

        Returns:
            Joints (degrees) gần `_current_joints`, hoặc None nếu không converge.
        """
        try:
            from .kinematics import inverse_kinematics_seeded
            from .kinematics.urdf_chain import gp7_urdf
        except ImportError:
            return None

        if not hasattr(self, "_dh_model_cached"):
            # URDF chain — verified match RoboDK SolveFK (0.00mm diff)
            self._dh_model_cached = gp7_urdf(
                base_xyz_mm=tuple(self.config.get("robot_base_xyz_mm", (0.0, 0.0, 0.0))),
                tool_offset_mm=float(self.config.get("robot_tool_offset_mm", 0.0)),
            )
        model = self._dh_model_cached

        q_init = (
            [np.deg2rad(q) for q in self._current_joints]
            if self._current_joints is not None
            else [0.0] * 6
        )
        # Multi-seed: thử q_init trước (nghiệm gần, mượt); fail thì retry từ seed
        # đa dạng → ~100% hội tụ.
        sol_rad = inverse_kinematics_seeded(model, target_T_world, q_init)
        if sol_rad is None:
            return None
        return [float(np.rad2deg(q)) for q in sol_rad]

    def _world_to_robot_base(self, T_world: np.ndarray) -> np.ndarray:
        """Convert pose 4x4 từ world → robot BASE frame (cho HSE Cartesian).

        Wrap frame_convert.world_to_robot_base() với config-driven base pose.
        """
        from .frame_convert import world_to_robot_base
        base_xyz = tuple(self.config.get("robot_base_xyz_mm", (0.0, 0.0, 0.0)))
        base_rpy = tuple(self.config.get("robot_base_rpy_deg", (0.0, 0.0, 0.0)))
        return world_to_robot_base(T_world, base_xyz, base_rpy)

    def _solve_ik_routed(self, target_T_world: np.ndarray):
        """Pick IK source: YRC controller (Cartesian) hoặc client-side DLS.

        Priority:
          1. use_yrc_ik=True → gửi pose Cartesian thẳng cho backend (YRC tự IK).
             Trả về ("YRC_POSE", T_base).
          2. Default → numerical DLS client-side. Trả về (joint_list, None).

        Returns:
            ("YRC_POSE", T_base): caller gọi MoveJ(T_base) Cartesian
            (joint_list, None): caller gọi MoveJ(joint_list)
            (None, None): IK fail
        """
        # Path 1: YRC tự IK — gửi pose Cartesian thẳng
        if self.config.get("use_yrc_ik", False):
            T_base = self._world_to_robot_base(target_T_world)
            return ("YRC_POSE", T_base)

        # Path 2: client-side numerical IK (default — URDF chain match RoboDK 0.00mm)
        joint_list = self._solve_ik_client(target_T_world)
        if joint_list is None:
            logger.warning("Client DLS IK fail tại world %s",
                           target_T_world[:3, 3].round(1).tolist())
            return (None, None)
        return (joint_list, None)

    def _move_j_via_ik(self, target_T: np.ndarray) -> None:
        """MoveJ qua IK route (YRC controller hoặc client DLS)."""
        result, pose = self._solve_ik_routed(target_T)

        # YRC IK path — pass pose 4x4 trực tiếp, controller tự IK
        if result == "YRC_POSE":
            logger.debug("MoveJ Cartesian (YRC IK): target_base=%s",
                         pose[:3, 3].round(1).tolist())
            self.robot.MoveJ(pose)
            self._current_joints = None
            return

        joint_list = result
        if joint_list is None:
            raise RuntimeError(
                f"IK fail at world {target_T[:3, 3].round(1).tolist()} (no fallback)"
            )
        joint_list = self._normalize_target_joints(joint_list)
        logger.debug("MoveJ joints=%s (target world=%s)",
                     [round(j, 1) for j in joint_list],
                     target_T[:3, 3].round(1).tolist())
        self.robot.MoveJ(joint_list)
        self._current_joints = joint_list

    def _move_l_via_ik(self, target_T: np.ndarray) -> None:
        """MoveL qua IK route — same pattern as MoveJ."""
        result, pose = self._solve_ik_routed(target_T)

        if result == "YRC_POSE":
            self.robot.MoveL(pose)
            self._current_joints = None
            return

        joint_list = result
        if joint_list is None:
            raise RuntimeError(
                f"IK fail at world {target_T[:3, 3].round(1).tolist()} (no fallback)"
            )
        joint_list = self._normalize_target_joints(joint_list)
        self.robot.MoveL(joint_list)
        self._current_joints = joint_list

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

        # Reset viewport scene về template ban đầu (nếu backend support) —
        # tránh object "trôi" dần qua nhiều trial.
        if hasattr(self.robot, "reset_scene"):
            try:
                self.robot.reset_scene()
            except Exception as e:  # noqa: BLE001
                logger.debug("reset_scene lỗi: %s", e)

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
        dz = self.config["approach_height_mm"]

        # Clamp cứng cho fingertip TCP: không bao giờ xuống dưới table_top + safety.
        # Tránh xuyên bàn 100% cho mọi vật, kể cả khi adaptive offset chọn sai.
        table_top = float(self.config["table_top_z_mm"])
        safety_margin = float(self.config["table_safety_margin_mm"])
        min_grasp_z = table_top + safety_margin
        max_offset = float(self.config["grasp_depth_offset_mm"])

        for obj in objects:
            xyz_base = np.array(obj["pose_base"], dtype=float).copy()
            # pose_base[2] là Z của TOP object (camera nhìn xuống → depth là top).
            # Adaptive offset: clip theo chiều cao để fingertip vào giữa thân.
            obj_height = obj.get("height_mm")
            if obj_height and obj_height > 0:
                effective_offset = min(max_offset,
                                       max(safety_margin, obj_height - safety_margin))
            else:
                effective_offset = max_offset
            target_z = xyz_base[2] - effective_offset
            # HARD CLAMP: dù offset thế nào, TCP không bao giờ dưới min_grasp_z.
            clamped_z = max(target_z, min_grasp_z)
            if clamped_z > target_z:
                logger.info(
                    "Grasp Z clamped: target=%.0f → %.0f (table_safety, "
                    "fingertip giữ %dmm trên bàn)",
                    target_z, clamped_z, int(safety_margin),
                )
            xyz_base[2] = clamped_z
            yaw = obj["pose_camera"][3]
            grasp_T = make_grasp_pose(xyz_base, yaw, self.config["yaw_offset_deg"])
            lift_T = grasp_T.copy()
            lift_T[2, 3] += dz

            # Place pose: X,Y từ config; Z = grasp Z (cùng tool-object offset
            # → vật đặt lại ở cùng độ cao bàn). yaw + yaw_offset giống grasp →
            # gripper giữ orientation, attached object không xoay khi transfer.
            place_xyz = np.array(self.config["place_position"], dtype=float).copy()
            place_xyz[2] = xyz_base[2]
            place_T = make_grasp_pose(place_xyz, yaw, self.config["yaw_offset_deg"])
            place_lift_T = place_T.copy()
            place_lift_T[2, 3] += dz

            # Kiểm tra reachability cho TOÀN BỘ trajectory (contribution C2).
            # Skip trong sim để tiết kiệm API budget (4 calls/trial × N trials).
            if not self.config.get("skip_reachability_check", False):
                unreachable = [
                    name for name, T in (
                        ("approach", lift_T),
                        ("grasp", grasp_T),
                        ("place_lift", place_lift_T),
                        ("place", place_T),
                    )
                    if not self._is_reachable(T)
                ]
                if unreachable:
                    logger.info(
                        "Trial %d: vật '%s' không với tới được tại %s, bỏ qua",
                        trial_id, obj.get("class_name", "?"), unreachable,
                    )
                    continue

            # ─── Thực thi pick-and-place cho vật này ───
            self.stats["attempted"] += 1
            ok = self._execute_pick_place(
                grasp_T, lift_T, place_T, place_lift_T, obj, trial_id,
            )
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
        self,
        grasp_T: np.ndarray,
        lift_T: np.ndarray,
        place_T: np.ndarray,
        place_lift_T: np.ndarray,
        obj: dict[str, Any],
        trial_id: int,
    ) -> bool:
        """Chạy chuỗi chuyển động gắp-thả. Trả về True nếu thành công.

        Nhận 4 pose đã được PLAN tính sẵn. Pose ở WORLD frame numpy 4x4;
        backend (SimRobot / DigitalTwinMirror) tự xử lý frame conversion.
        """
        try:
            # Cache current joints chỉ lần đầu (trial 1) — các trial sau dùng
            # self._current_joints đã được _move_*_via_ik cập nhật ở trial trước.
            if self._current_joints is None:
                try:
                    self._current_joints = self._joints_to_list(self.robot.Joints())
                except Exception:  # noqa: BLE001
                    pass

            # UC2 — Predictive safety: solve IK cho 4 waypoint, build joint
            # trajectory (gồm current → lift → grasp → lift → place_lift →
            # place → place_lift), verify joint limit + self-collision pure-
            # Python TRƯỚC khi gửi MoveJ. Reject sớm trial unsafe → tránh
            # gửi alarm-prone command lên controller thật.
            reason = self._predict_safety_for_trajectory(
                [lift_T, grasp_T, lift_T, place_lift_T, place_T, place_lift_T]
            )
            if reason is not None:
                logger.warning("Trial %d: predictive safety reject — %s",
                               trial_id, reason)
                self.stats["failed"] += 1
                self.sm.fail(reason)
                return False

            # Batch context: HSE backend gom toàn bộ motion + IO vào 1 INFORM job
            # → 1 FTP upload + 1 JOB_START/trial thay vì 5-7 lần. Backends khác
            # (SimRobot) → nullcontext, code chạy nguyên văn.
            with self._robot_batch_ctx():
                self.sm.transition_to(PickState.APPROACH)
                self._move_j_via_ik(lift_T)
                # Cache joints sau APPROACH để LIFT reuse — cùng pose lift_T, không
                # cần SolveIK lại (-1 API call/trial).
                approach_joints = self._current_joints

                self.sm.transition_to(PickState.GRASP)
                self._move_l_via_ik(grasp_T)
                self._gripper(close=True, obj_class=obj.get("class_name"))

                self.sm.transition_to(PickState.LIFT)
                if approach_joints is not None:
                    self.robot.MoveL(approach_joints)
                    self._current_joints = approach_joints
                else:
                    self._move_l_via_ik(lift_T)

                self.sm.transition_to(PickState.TRANSFER)
                self._move_j_via_ik(place_lift_T)
                # Cache joints sau TRANSFER để RETREAT reuse (-1 API call/trial).
                transfer_joints = self._current_joints

                self.sm.transition_to(PickState.PLACE)
                self._move_l_via_ik(place_T)
                self._gripper(close=False)

                self.sm.transition_to(PickState.RETREAT)
                if transfer_joints is not None:
                    self.robot.MoveL(transfer_joints)
                    self._current_joints = transfer_joints
                else:
                    self._move_l_via_ik(place_lift_T)

            self.sm.transition_to(PickState.DONE)
            self.stats["successful"] += 1
            logger.info(
                "Trial %d: gắp-thả '%s' THÀNH CÔNG | stats=%s",
                trial_id, obj.get("class_name", "?"), self.stats,
            )
            if self.config["return_home_after_success"]:
                self._return_home()
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
        """Đưa robot về home — normalize joints để tránh quay chong chóng.

        Ưu tiên `config['home_joints_deg']` (truyền từ cell layout) thay vì
        `robot.JointsHome()` — vì JointsHome() đọc từ file .robot library
        (mặc định Yaskawa GP7 = [0,0,0,0,0,0]), KHÔNG phải pose home user đặt
        trong cell_layout.yaml. Dùng sai source → MoveJ về 0 mà current đang
        ở -180 → quay 180° trên J4/J6 = "chong chóng".
        """
        try:
            home_list = None
            cfg_home = self.config.get("home_joints_deg")
            if cfg_home:
                home_list = [float(j) for j in cfg_home]
            else:
                home = self.robot.JointsHome()
                home_list = self._joints_to_list(home)
            if home_list:
                home_list = self._normalize_target_joints(home_list)
                self.robot.MoveJ(home_list)
                self._current_joints = home_list
            else:
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
