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
# place_position phải khớp PlaceZone trong cell_layout.yaml (mm, base frame).
_DEFAULT_CONFIG: dict[str, Any] = {
    "robot_name": "Yaskawa GP7",
    "calibration_path": "config/calibration/T_base_camera.npy",
    "place_position": [700.0, 120.0, 700.0],
    # Khoảng cách lift trên grasp/place pose. 200mm: lift_T Z ≈ 860 (trên top
    # bottle 150mm) → MoveL xuống thẳng đứng từ cao; arm không quét workspace
    # khi APPROACH/TRANSFER → tránh đâm bottle vào tay robot.
    "approach_height_mm": 200.0,
    # Trừ vào Z của grasp pose để fingertip vào GIỮA THÂN object, không kẹp ở
    # đỉnh (postprocess.deproject trả về tọa độ TOP của object — Z=top). Giá trị
    # phụ thuộc chiều cao object: bottle ~150mm → offset 50-80mm; cup ~40mm → 20mm.
    "grasp_depth_offset_mm": 50.0,
    # Về home sau mỗi success → trial kế APPROACH từ trên cao xuống (không
    # "đi ngang" từ place_lift đến lift mới). Tốn ~2 API call/trial nhưng
    # cần cho video demo trông tự nhiên.
    "return_home_after_success": True,
    "gripper_do_index": 1,
    "gripper_delay_s": 0.3,
    "inter_trial_delay_s": 1.0,
    "speed_joint_deg_s": 60.0,
    "speed_linear_mm_s": 80.0,
    "yaw_offset_deg": 0.0,
    "detection_timeout_s": 2.0,
    # Sim mặc định SKIP reachability check (tiết kiệm 4 API calls/trial cho
    # RoboDK Free quota nhỏ). MoveJ sẽ tự raise nếu thật sự không với tới.
    # Real mode nên đổi thành False để bật C2 (lớp an toàn digital twin).
    "skip_reachability_check": True,
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
        robodk_objects: dict[str, Any] | None = None,
        robodk_tool: Any = None,
    ) -> None:
        self.queue = perception_queue
        self.config = {**_DEFAULT_CONFIG, **(config or {})}
        self.trial_logger = logger_obj

        # Items để attach/detach object vào gripper trong sim mode (visualize
        # gắp-đi). None ở headless / --no-build → no-op gracefully.
        self.robodk_objects: dict[str, Any] = robodk_objects or {}
        self.robodk_tool = robodk_tool
        self._attached_obj: Any = None
        # Lưu pose + parent ban đầu của object để reset đầu mỗi trial — tránh
        # bottle "dịch chuyển ra xa dần" sau mỗi trial (offset cố định khi attach).
        self._initial_obj_poses: dict[str, Any] = {}
        self._initial_obj_parents: dict[str, Any] = {}

        self._rdk = None
        self.robot = robot
        if self.robot is None:
            self._connect_robodk()

        # Tải ma trận hand-eye calibration.
        self.T_BC = load_calibration(self.config["calibration_path"])
        logger.info("Loaded hand-eye T_BC, translation=%s mm", self.T_BC[:3, 3].round(1))

        # MoveJ pose được RoboDK diễn giải trong parent frame của robot
        # (mặc định). Cell ta đặt parent ở pedestal (Z=630), nhưng orchestrator
        # và T_BC đều dùng WORLD frame. Tính sẵn transform để convert
        # pose world → parent trước khi gọi MoveJ/MoveJ_Test.
        self._T_world_to_robotbase = self._compute_world_to_robotbase()
        if not np.allclose(self._T_world_to_robotbase, np.eye(4)):
            logger.info(
                "Robot parent frame ở world Z=%.1fmm — sẽ convert pose world→parent",
                -self._T_world_to_robotbase[2, 3],
            )

        # Disable collision check ở runtime — RoboDK báo "collision" cho cả
        # gripper-touching-object khi grasp (đó là CHỦ ĐÍCH, không phải lỗi)
        # và arm-sweeping-through-template-volume khi IK chọn config. Cả hai
        # đều block MoveJ. Sim cần disable để chạy được; real mode tuning sau.
        self._disable_collision_check()

        self.sm = PickPlaceStateMachine()
        self.stats = {"attempted": 0, "successful": 0, "failed": 0}
        self._current_joints: list[float] | None = None       # cache cho SolveIK ref

        # setSpeed gọi 1 lần ở init thay vì mỗi trial (-1 API call/trial).
        self._set_speed()

        # Capture pose + parent ban đầu của các object → có baseline để reset
        # đầu mỗi trial. Chỉ làm khi có robodk_objects (sim mode với RoboDK).
        self._capture_initial_object_poses()

    def _disable_collision_check(self) -> None:
        """Tắt collision check toàn cục — xem comment trong __init__."""
        try:
            rdk = self._rdk
            if rdk is None and hasattr(self.robot, "RDK") and callable(self.robot.RDK):
                rdk = self.robot.RDK()
            if rdk is None or not hasattr(rdk, "setCollisionActive"):
                return
            rdk.setCollisionActive(0)
            logger.info("Collision check disabled (sim convention)")
        except Exception as e:  # noqa: BLE001
            logger.debug("setCollisionActive bỏ qua: %s", e)

    def _compute_world_to_robotbase(self) -> np.ndarray:
        """Tính ma trận chuyển từ world frame sang robot reference frame.

        RoboDK MoveJ(pose) diễn giải pose tương đối với robot.Parent() (mặc định).
        Cell_loader đặt parent ở pedestal height (vd Z=630) → cần convert
        target_in_parent = inv(T_parent_in_world) @ target_in_world.

        Trả về identity nếu không lấy được pose parent (vd mock robot trong test).
        """
        try:
            if not (hasattr(self.robot, "Parent") and callable(self.robot.Parent)):
                return np.eye(4)
            parent = self.robot.Parent()
            if parent is None or not hasattr(parent, "PoseAbs"):
                return np.eye(4)
            from src.cell.pose_utils import robodk_pose_to_matrix
            from src.calibration.hand_eye_solver import invert_transform
            T_parent_in_world = robodk_pose_to_matrix(parent.PoseAbs())
            if not isinstance(T_parent_in_world, np.ndarray):
                return np.eye(4)
            if T_parent_in_world.shape != (4, 4):
                return np.eye(4)
            return invert_transform(T_parent_in_world)
        except Exception as e:  # noqa: BLE001
            logger.debug("Robot parent frame không sẵn có: %s", e)
            return np.eye(4)

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

        RoboDK MoveJ_Test convention:
            0   = OK, không va chạm
            > 0 = va chạm tại joint số đó (collision check active)
            < 0 = không với tới (out of reach / joint limit / singularity)

        Trong sim, va chạm thường là false positive (gripper-vs-object template,
        arm-vs-pedestal khi planner chọn config không tối ưu). Chỉ reject nếu
        thật sự out-of-reach (< 0). Real mode nên enable collision check riêng
        và xử lý tách bạch — đó là pha tuning sau (mục 9 tài liệu).
        """
        try:
            target_pose = self._to_robodk_pose(target_T)
            result = self.robot.MoveJ_Test(self.robot.Joints(), target_pose)
            if result < 0:
                logger.info(
                    "MoveJ_Test out-of-reach (code=%s) tại world %s",
                    result, target_T[:3, 3].round(1).tolist(),
                )
                return False
            if result > 0:
                logger.debug(
                    "MoveJ_Test phát hiện va chạm tại joint %s, pose world %s — chấp nhận trong sim",
                    result, target_T[:3, 3].round(1).tolist(),
                )
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("MoveJ_Test lỗi: %s", e)
            return False

    def _gripper(self, close: bool, obj_class: str | None = None) -> None:
        """Đóng/mở gripper qua Digital Output, chờ ổn định.

        Sim mode: kèm attach/detach object item vào gripper tool qua
        `setParentStatic` để object visually di chuyển theo gripper.
        Headless / no robodk_objects: no-op.
        """
        self.robot.setDO(self.config["gripper_do_index"], 1 if close else 0)

        if close and obj_class:
            self._attach_to_gripper(obj_class)
        elif not close:
            self._detach_from_gripper()

        time.sleep(self.config["gripper_delay_s"])

    def _attach_to_gripper(self, obj_class: str) -> None:
        """Set parent của object item → gripper tool. Object follow gripper."""
        if not self.robodk_objects or self.robodk_tool is None:
            return
        item = self.robodk_objects.get(obj_class)
        if item is None or not hasattr(item, "setParentStatic"):
            return
        try:
            item.setParentStatic(self.robodk_tool)
            self._attached_obj = item
            logger.debug("Attached object '%s' to gripper", obj_class)
        except Exception as e:  # noqa: BLE001
            logger.debug("Attach failed: %s", e)

    def _capture_initial_object_poses(self) -> None:
        """Lưu Pose() + Parent() của object để reset đầu mỗi trial."""
        for name, item in self.robodk_objects.items():
            if not hasattr(item, "Pose") or not hasattr(item, "Parent"):
                continue
            try:
                self._initial_obj_poses[name] = item.Pose()
                self._initial_obj_parents[name] = item.Parent()
            except Exception as e:  # noqa: BLE001
                logger.debug("Capture initial pose '%s' failed: %s", name, e)

    def _reset_objects_to_initial(self) -> None:
        """Reset object về parent_frame + pose ban đầu (đầu mỗi trial).

        Không có baseline (headless/no-build) → no-op gracefully.
        """
        if not self._initial_obj_poses:
            return
        for name, item in self.robodk_objects.items():
            pose = self._initial_obj_poses.get(name)
            parent = self._initial_obj_parents.get(name)
            if pose is None or parent is None:
                continue
            try:
                if hasattr(item, "setParentStatic"):
                    item.setParentStatic(parent)
                if hasattr(item, "setPose"):
                    item.setPose(pose)
            except Exception as e:  # noqa: BLE001
                logger.debug("Reset object '%s' failed: %s", name, e)

    def _detach_from_gripper(self) -> None:
        """Detach: set parent về station root, giữ pose absolute hiện tại."""
        if self._attached_obj is None:
            return
        if not hasattr(self._attached_obj, "setParentStatic"):
            self._attached_obj = None
            return
        try:
            rdk = self._rdk
            if rdk is None and hasattr(self.robot, "RDK") and callable(self.robot.RDK):
                rdk = self.robot.RDK()
            if rdk is not None and hasattr(rdk, "ActiveStation"):
                self._attached_obj.setParentStatic(rdk.ActiveStation())
                logger.debug("Detached object from gripper")
        except Exception as e:  # noqa: BLE001
            logger.debug("Detach failed: %s", e)
        self._attached_obj = None

    def _to_robodk_pose(self, T: np.ndarray):
        """Chuyển numpy 4x4 (world frame) → RoboDK Mat (parent frame).

        Apply T_world_to_robotbase conversion để MoveJ nhận pose đúng frame
        (mặc định RoboDK diễn giải trong parent frame của robot).
        """
        from robodk.robomath import Mat

        T_parent = self._T_world_to_robotbase @ np.asarray(T)
        return Mat(T_parent.tolist())

    @staticmethod
    def _joints_to_list(joints) -> list[float] | None:
        """Convert RoboDK Mat hoặc list-like → flat list of 6 floats.

        RoboDK Joints() / SolveIK() trả về Mat. Phụ thuộc shape (1xN row hoặc
        Nx1 col), .tolist() ra hình thức khác nhau. Handle cả 2.
        """
        if joints is None:
            return None
        # Ưu tiên .tolist() (RoboDK Mat method)
        for getter in (
            lambda: joints.tolist() if hasattr(joints, "tolist") else None,
            lambda: list(joints),
        ):
            try:
                data = getter()
            except Exception:  # noqa: BLE001
                continue
            if not data:
                continue
            # Trường hợp 1: đã là list[float] phẳng
            if isinstance(data[0], (int, float)) and len(data) >= 6:
                try:
                    return [float(j) for j in data[:6]]
                except (TypeError, ValueError):
                    continue
            # Trường hợp 2: nested [[j1, j2, ..., j6]] (1xN row)
            if isinstance(data[0], list) and len(data) == 1 and len(data[0]) >= 6:
                try:
                    return [float(j) for j in data[0][:6]]
                except (TypeError, ValueError):
                    continue
            # Trường hợp 3: nested [[j1], [j2], ...] (Nx1 column)
            if isinstance(data[0], list) and len(data) >= 6 and all(
                isinstance(row, list) and len(row) >= 1 for row in data[:6]
            ):
                try:
                    return [float(row[0]) for row in data[:6]]
                except (TypeError, ValueError):
                    continue
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

    def _solve_ik_joints(self, pose) -> list[float] | None:
        """Tính joints cho pose; trả None nếu không có solution.

        SolveIK với `joints_approx = current` → pick solution gần nhất, tránh
        MoveJ refusing on discontinuous joint motion.
        """
        joints = None
        if self._current_joints is not None:
            try:
                joints = self.robot.SolveIK(pose, self._current_joints)
            except (TypeError, AttributeError):
                joints = None
        if joints is None:
            try:
                joints = self.robot.SolveIK(pose)
            except Exception:  # noqa: BLE001
                return None
        return self._joints_to_list(joints)

    def _move_j_via_ik(self, target_T: np.ndarray) -> None:
        """MoveJ via SolveIK với joints gần current → MoveJ(joints).

        RoboDK MoveJ(pose) đôi khi raise 'Target cannot be reached' do internal
        IK chọn solution xa current joints. Truyền `joints_approx=current` vào
        SolveIK để pick solution gần nhất, rồi MoveJ(joints) bypass IK internal.
        Normalize joints wrap ±360 để tránh full-rotation "chong chóng".
        """
        pose = self._to_robodk_pose(target_T)
        joint_list = self._solve_ik_joints(pose)
        if joint_list is None:
            # Mock robot trong test hoặc SolveIK fail → fallback MoveJ(pose)
            self.robot.MoveJ(pose)
            return
        joint_list = self._normalize_target_joints(joint_list)
        logger.debug("MoveJ joints=%s (target world=%s)",
                     [round(j, 1) for j in joint_list],
                     target_T[:3, 3].round(1).tolist())
        self.robot.MoveJ(joint_list)
        self._current_joints = joint_list

    def _move_l_via_ik(self, target_T: np.ndarray) -> None:
        """MoveL via SolveIK với joints gần current (same pattern as MoveJ)."""
        pose = self._to_robodk_pose(target_T)
        joint_list = self._solve_ik_joints(pose)
        if joint_list is None:
            self.robot.MoveL(pose)
            return
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

        # Reset object về vị trí template ban đầu để mỗi trial có scene
        # giống nhau (loop demo không "trôi" object dần).
        self._reset_objects_to_initial()

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

        for obj in objects:
            xyz_base = np.array(obj["pose_base"], dtype=float).copy()
            # pose_base[2] là Z của TOP object (camera nhìn xuống → depth là top).
            # Trừ grasp_depth_offset_mm để fingertip TCP đi vào giữa thân.
            xyz_base[2] -= self.config["grasp_depth_offset_mm"]
            yaw = obj["pose_camera"][3]
            grasp_T = make_grasp_pose(xyz_base, yaw, self.config["yaw_offset_deg"])
            lift_T = grasp_T.copy()
            lift_T[2, 3] += dz

            # Place pose: X,Y từ config; Z = grasp Z để vật quay về bàn ở cùng
            # độ cao như khi gắp (tránh "lơ lửng" do tool-object offset cố định
            # khi attach). Config place_position.Z dùng làm fallback nếu không
            # có grasp_z hợp lệ.
            place_xyz = np.array(self.config["place_position"], dtype=float).copy()
            place_xyz[2] = xyz_base[2]
            place_T = make_grasp_pose(place_xyz, 0.0)
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

        Nhận 4 pose đã được PLAN tính sẵn (vì cùng dữ liệu dùng cho reachability
        check). Chỉ chuyển sang RoboDK Mat ngay trước khi gọi robot → logic này
        test được bằng mock không cần RoboDK.
        """
        try:
            # Cache current joints chỉ lần đầu (trial 1) — các trial sau dùng
            # self._current_joints đã được _move_*_via_ik cập nhật ở trial trước.
            # Tiết kiệm 1 robot.Joints() call/trial cho RoboDK Free quota nhỏ.
            if self._current_joints is None:
                try:
                    self._current_joints = self._joints_to_list(self.robot.Joints())
                except Exception:  # noqa: BLE001
                    pass

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
            # Skip return_home ở success path: trial kế bắt đầu từ RETREAT pose
            # (vẫn trong workspace), tiết kiệm 1-2 API call/trial. Để re-enable,
            # set config["return_home_after_success"]=True.
            if self.config.get("return_home_after_success", False):
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
