"""
orchestrator.py
───────────────
Orchestrator: bridges Perception ↔ robot backend, controls the pick-and-place cycle.

Responsibilities:
  - Receive detections (camera frame) from the perception queue
  - Transform to base frame via hand-eye matrix
  - Check reachability via backend (SimRobot reach envelope or predictive
    safety pure-Python) → contribution C2 of the paper
  - Execute pick-and-place via robot backend (SimRobot sim or DigitalTwin
    + HSE real)
  - Log every trial via TrialLogger

Backend must be injected via the `robot=` arg. This module NO longer depends on RoboDK —
SolveIK goes through a pure-Python URDF chain (verified match RoboDK 0.00mm) or YRC1000
controller-side (Cartesian path).
"""
from __future__ import annotations

import contextlib
import logging
import queue
import time
from typing import Any

import numpy as np

from .coord_conv import (
    camera_to_base, camera_yaw_to_base, load_calibration, make_grasp_pose)
from .state_machine import PickPlaceStateMachine, PickState

logger = logging.getLogger(__name__)


# Default config values — override via dict passed in.
# place_position must match PlaceZone in cell_layout.yaml (mm, base frame).
_DEFAULT_CONFIG: dict[str, Any] = {
    "robot_name": "Yaskawa GP7",
    "calibration_path": "config/calibration/T_base_camera.npy",
    "place_position": [700.0, 120.0, 700.0],
    # Lift height above grasp/place so the robot approaches vertically (MoveL down),
    # arm does not sweep across workspace.
    "approach_height_mm": 200.0,
    # Offset subtracted from grasp pose Z so the fingertip enters the object body
    # rather than the top (postprocess.deproject returns TOP coordinate). Adaptive
    # to object height.
    "grasp_depth_offset_mm": 50.0,
    # Hard clamp: fingertip TCP never goes below (table_top + safety),
    # prevents table penetration even if adaptive offset is wrong.
    "table_top_z_mm": 500.0,
    "table_safety_margin_mm": 100.0,
    # Yaw offset between PCA major axis on mask and gripper opening axis. Yaw=0:
    # gripper jaws spread along PCA major axis (= longest object dim).
    # Yaw=90: jaws spread perpendicular to PCA (= shortest object dim) — used for
    # Galaxy S23 tray: PCA major = 180mm (length), jaws must spread along
    # 100mm (width) → requires yaw_offset 90°.
    "yaw_offset_deg": 90.0,
    # Return home after each success → next trial APPROACHes top-down (no
    # lateral move from place_lift to new lift). Costs ~2 API calls/trial but
    # needed for the demo video to look natural.
    "return_home_after_success": True,
    # ─── Gripper config ───
    # 2 modes:
    #   1. Single-bit toggle (default, sim + legacy): `gripper_do_index` 1 bit
    #   2. Dual-solenoid + sensors (CC-Link real): set `gripper_cc_link` dict
    #      in experiment.yaml config or CLI. When present → orchestrator uses
    #      double-acting pattern with feedback sensors instead of blind delay.
    "gripper_do_index": 1,
    "gripper_delay_s": 0.5,                 # fallback blind delay (no sensor)
    "gripper_cc_link": None,                # opt-in: set dict for CC-Link path
    "inter_trial_delay_s": 1.0,
    "speed_joint_deg_s": 60.0,
    "speed_linear_mm_s": 80.0,
    "detection_timeout_s": 2.0,
    # Sim defaults to SKIP reachability check (MoveJ raises if out-of-reach).
    # Real mode should set False to enable C2 (predictive safety layer).
    "skip_reachability_check": True,
    "is_real_mode": False,
    # UC2 — Pure-Python predictive simulation BEFORE sending MoveJ to robot.
    # Verifies joint limits + self-collision for the full trajectory. When True:
    # pre-check ~50ms/trial, rejects trial if predicted unsafe.
    "predictive_safety_enabled": False,
    # Max joint speed (deg/s) used for predict interpolation. Must match actual
    # robot speed to predict the correct motion sequence.
    "predict_max_speed_deg_s": 30.0,
    # When True: use client-side IK (Pieper analytical nearest-branch, DLS
    # fallback). Default for all non-YRC backends.
    "use_client_ik": False,
    # Robot base pose in world frame (mm + degrees). Used to init the URDF
    # model with the correct base offset for client IK.
    "robot_base_xyz_mm": (0.0, 0.0, 0.0),
    "robot_base_rpy_deg": (0.0, 0.0, 0.0),
    # Tool TCP offset (mm) along flange Z. Must match gripper TCP in
    # cell config so IK returns correct joints for fingertip pose.
    "robot_tool_offset_mm": 0.0,
    # When True: send Cartesian pose directly to YRC1000 via HSE BASE position
    # variable → controller handles IK. Recommended for HSE real mode. Overrides
    # use_client_ik.
    "use_yrc_ik": False,
}


def grasp_depth_offset(obj_height_mm: float | None, max_offset_mm: float,
                       clearance_mm: float) -> float:
    """How far below the part's top face the jaw tips go (mm).

    Never deeper than the part itself minus ``clearance_mm``, so the tips stay
    above whatever the part rests on: the table, or another part in a stacked
    scene, which the table-level floor alone cannot protect. Without a known
    height, fall back to ``max_offset_mm`` and rely on that floor.

    The previous formula, min(max_offset, max(clearance, height - clearance)),
    returned max_offset for every height whenever clearance >= max_offset (the
    defaults are 100 and 50), so the grasp depth never adapted to the part.
    """
    if obj_height_mm is None or obj_height_mm <= 0:
        return float(max_offset_mm)
    return float(max(0.0, min(max_offset_mm, obj_height_mm - clearance_mm)))


class Orchestrator:
    """Orchestrates the pick-and-place cycle via robot backend (SimRobot or HSE).

    Usage:
        orch = Orchestrator(perception_queue, config, robot=sim_robot)
        stats = orch.run_n_trials(50)

    Args:
        perception_queue: Queue carrying detection messages from PerceptionNode.
        config: Configuration dict (see _DEFAULT_CONFIG).
        robot: Duck-typed backend (SimRobot or DigitalTwinMirror). REQUIRED
            inject — auto-connect to RoboDK has been removed.
        logger_obj: (Optional) TrialLogger for recording results.
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
                "Orchestrator requires a robot backend (SimRobot or DigitalTwinMirror) "
                "via the `robot=` arg. RoboDK auto-connect has been removed."
            )
        self.robot = robot

        self.T_BC = load_calibration(self.config["calibration_path"])
        logger.info("Loaded hand-eye T_BC, translation=%s mm", self.T_BC[:3, 3].round(1))

        self.sm = PickPlaceStateMachine()
        self.stats = {"attempted": 0, "successful": 0, "failed": 0}
        self._current_joints: list[float] | None = None       # cache for SolveIK ref
        # C3 failure context: pre-drawn pose-list id (set per trial by the
        # experiment script) and the RGB frame saved by PerceptionNode at DETECT.
        self.current_pose_id: str = ""
        # Set per trial by the runner's pre-trial hook, and constant for a run.
        self.current_condition: str = ""
        self.current_stack_on: str = ""
        # B3_EXPECTED_YAW_CONTEXT_V2
        # Ground-truth placement yaw from the pose list. Safety reference only.
        self.current_expected_yaw_deg: float | None = None
        # Pose-list target class. When set, detections of every other class
        # are ignored. This is essential for STACKED trials where both the
        # target and support object may be detected.
        self.current_expected_class: str = ""
        self.session_id: str = ""
        self.block_id: str = ""
        self.operator_id: str = ""
        self._last_frame_path: str = ""
        # Optional callable(trial_id, machine_success) -> 1 | 0 | "", asked once per
        # trial that actually moved the robot. The runner sets it under
        # --confirm-each-trial; its answer goes to the `human_ok` column.
        self.human_score_hook: Any = None

        self._set_speed()

        self._last_support_height_mm: float = 0.0
        self._last_place_z_mm: float | None = None

    # ────────────────────────────────────────────────────────────
    # Coordinate transform + control helpers
    # ────────────────────────────────────────────────────────────

    def _select_objects(self, det_msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Transform detections to base frame, sorted topmost object first.

        Returns an empty list if no objects are present.
        """
        # Work on shallow copies (with pose_base added) and a new list — do NOT
        # mutate or re-sort the producer-owned detection dicts/list in place.
        objects = []
        for o in det_msg.get("objects", []):
            # Validate the detection shape HERE so a malformed pose_camera skips
            # this one object instead of crashing the whole trial loop later (PLAN
            # reads pose_camera[3] for yaw — IndexError would abort everything).
            pc = o.get("pose_camera")
            if not isinstance(pc, (list, tuple, np.ndarray)) or len(pc) < 4:
                logger.warning("Skipping detection with malformed pose_camera: %r", pc)
                continue

            # POSE-LIST CLASS LOCK
            #
            # STACKED scenes contain at least two real objects. Never choose
            # whichever detection merely happens to have the highest Z.
            expected_cls = str(
                getattr(self, "current_expected_class", "") or ""
            ).strip().lower()

            detected_cls = str(
                o.get("class_name", "") or ""
            ).strip().lower()

            if expected_cls and detected_cls != expected_cls:
                continue

            xyz_cam = np.array(pc[:3], dtype=float)

            # Transform the in-plane grasp YAW from the camera image frame into the
            # base frame with the SAME hand-eye rotation as the position. The PCA yaw
            # is a pixel-frame angle (x-right, y-down); carrying it straight into
            # rotz() about base Z leaves it off by the camera mounting rotation +
            # image handedness, mis-aligning the gripper with the object long axis.
            # Prefer metric BASE yaw produced by PoseExtractor:
            # mask outline -> top plane -> BASE XY -> minAreaRect edge.
            metric_yaw_base = o.get("yaw_base")
            if metric_yaw_base is not None and np.isfinite(float(metric_yaw_base)):
                yaw_base = float(metric_yaw_base)
            else:
                yaw_base = camera_yaw_to_base(float(pc[3]), self.T_BC)

            logger.info(
                "VISION YAW class=%s source=%s pca_cam=%+.2fdeg "
                "edge_base=%+.2fdeg rect=%.1fx%.1fmm square=%s",
                o.get("class_name", "?"),
                o.get("yaw_source", "legacy_pca"),
                float(np.rad2deg(float(pc[3]))),
                float(np.rad2deg(yaw_base)),
                float(o.get("yaw_rect_long_mm", float("nan"))),
                float(o.get("yaw_rect_short_mm", float("nan"))),
                o.get("yaw_near_square", False),
            )
            objects.append({**o, "pose_base": camera_to_base(xyz_cam, self.T_BC),
                            "yaw_base": yaw_base})
        # Object with highest Z (closest to camera / on top) is grasped first.
        # With current_expected_class set, all entries here are already the
        # requested pose-list class.
        objects.sort(key=lambda o: o["pose_base"][2], reverse=True)

        expected_cls = str(
            getattr(self, "current_expected_class", "") or ""
        ).strip().lower()

        if expected_cls:
            if objects:
                logger.info(
                    "POSE-LIST CLASS LOCK: expected=%s selected=%s",
                    expected_cls,
                    objects[0].get("class_name", "?"),
                )
            else:
                seen = [
                    str(x.get("class_name", "?"))
                    for x in det_msg.get("objects", [])
                ]
                logger.warning(
                    "POSE-LIST CLASS LOCK: expected=%s NOT FOUND; seen=%s "
                    "-> motion refused for this trial",
                    expected_cls,
                    seen,
                )

        return objects

    def _is_reachable(self, target_T: np.ndarray) -> bool:
        """Check whether `target_T` (numpy 4x4) lies within the GP7 reach envelope.

        Uses a client-side sphere envelope (`backends.reach_envelope`) — pure
        Python, no backend dependency. SimRobot has its own MoveJ_Test (internal
        sphere envelope) but DigitalTwinMirror.MoveJ_Test is now a no-op → need
        client-side check at orchestrator level.

        This is a LIGHTWEIGHT pre-filter layer (0 IK calls). Predictive safety C2
        (joint limits + full-trajectory self-collision) runs later in
        `_execute_pick_place` when `predictive_safety_enabled=True`.
        """
        try:
            from .backends.reach_envelope import ReachEnvelope
        except ImportError:
            return True

        base_xyz = tuple(self.config.get("robot_base_xyz_mm", (0.0, 0.0, 0.0)))
        if not hasattr(self, "_reach_env_cached"):
            self._reach_env_cached = ReachEnvelope.gp7_default(base_xyz_mm=base_xyz)

        target_xyz = np.asarray(target_T)[:3, 3]
        # The envelope is about the FLANGE/wrist; target_T is the TCP. Back off the
        # tool offset along tool Z so a long tool doesn't skew the gate by ~its length
        # at the reach boundary (consistent with the IK flange retract).
        tool_off = float(self.config.get("robot_tool_offset_mm", 0.0))
        if abs(tool_off) > 1e-6:
            target_xyz = target_xyz - tool_off * np.asarray(target_T)[:3, 2]
        if not self._reach_env_cached.can_reach(target_xyz):
            logger.info(
                "Reach envelope fail at world %s (base=%s)",
                target_xyz.round(1).tolist(), list(base_xyz),
            )
            return False
        return True

    def _gripper(self, close: bool, obj_class: str | None = None) -> None:
        """Close/open gripper. 2 paths depending on config:

        **Path A — CC-Link double-acting + feedback** (real, with PLC + sensors):
        When `config["gripper_cc_link"]` is set:
          1. Mutually exclusive: de-energize the other solenoid first, then
             energize this one (cylinder safety — do not drive both directions at once)
          2. Wait for sensor position bit confirming cylinder has reached position
             (clamp/unclamp)
          3. (Close only) verify detect sensor X505 ON → object in gripper
             → fail "grasp_failed" if OFF (config require_detect_on_close=True)

        **Path B — Single-bit blind delay** (sim, legacy, no PLC feedback):
        Fallback `setDO(gripper_do_index, ...)` + fixed `gripper_delay_s`.

        Sim mode: includes attach/detach of the object item to the gripper tool via
        `setParentStatic` so the object visually follows the gripper.
        """
        cc = self.config.get("gripper_cc_link")
        if cc and hasattr(self.robot, "set_io") and hasattr(self.robot, "read_io"):
            self._gripper_cc_link(close, obj_class, cc)
        else:
            self._gripper_simple(close, obj_class)

    def _gripper_simple(self, close: bool, obj_class: str | None) -> None:
        """Path B: 1-bit toggle + blind delay (legacy / sim path)."""
        self.robot.setDO(self.config["gripper_do_index"], 1 if close else 0)
        # Visual attach/detach (sim viewport) — call backend if supported
        self._notify_viewport_grasp(close, obj_class)
        self._robot_timer(self.config["gripper_delay_s"])

    def _gripper_cc_link(
        self, close: bool, obj_class: str | None, cc: dict,
    ) -> None:
        """Path A: double-acting solenoid + sensor feedback via CC-Link."""
        if close:
            # Safe sequence: de-energize UnClamp first, then energize Clamp
            self.robot.set_io(cc["unclamp_bit"], 0)
            self.robot.set_io(cc["clamp_bit"], 1)
            sensor_bit = cc["clamp_sensor_bit"]
            action_name = "Clamp"
        else:
            self.robot.set_io(cc["clamp_bit"], 0)
            self.robot.set_io(cc["unclamp_bit"], 1)
            sensor_bit = cc["unclamp_sensor_bit"]
            action_name = "UnClamp"

        # Wait for sensor to confirm cylinder has reached position (NO blind delay)
        if not self._wait_sensor_on(
            sensor_bit,
            timeout_s=float(cc.get("wait_sensor_timeout_s", 2.0)),
            poll_s=float(cc.get("wait_sensor_poll_s", 0.05)),
        ):
            raise RuntimeError(
                f"gripper_timeout: {action_name} sensor (bit {sensor_bit}) "
                f"did not turn ON after {cc.get('wait_sensor_timeout_s', 2.0)}s"
            )

        # Verify grasp on close (require detect sensor X505)
        if close and cc.get("require_detect_on_close", True):
            detect = self.robot.read_io(cc["detect_bit"])
            if detect != 1:
                raise RuntimeError(
                    f"grasp_failed: detect sensor (bit {cc['detect_bit']}) "
                    f"OFF — object not in gripper"
                )

        # Visual object attach/detach (sim viewport)
        self._notify_viewport_grasp(close, obj_class)

    def _notify_viewport_grasp(self, close: bool, obj_class: str | None) -> None:
        """Notify backend viewport (if supported) when gripper closes/opens.

        Viewport (Open3D SimRobot/Mirror) uses this signal to attach/detach
        the object mesh from the tool. Backends without this → no-op.
        """
        if close and obj_class and hasattr(self.robot, "attach_object"):
            try:
                self.robot.attach_object(obj_class)
            except Exception as e:  # noqa: BLE001
                logger.debug("viewport attach_object error: %s", e)
        elif not close and hasattr(self.robot, "detach_object"):
            try:
                self.robot.detach_object()
            except Exception as e:  # noqa: BLE001
                logger.debug("viewport detach_object error: %s", e)

    def _wait_sensor_on(
        self, bit_addr: int, timeout_s: float, poll_s: float,
    ) -> bool:
        """Poll sensor bit until == 1 or timeout. Returns True if ON."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            val = self.robot.read_io(bit_addr)
            if val == 1:
                return True
            time.sleep(poll_s)
        return False

    def _robot_timer(self, seconds: float) -> None:
        """Sleep with batch awareness — INFORM TIMER inside batch, time.sleep outside."""
        if hasattr(self.robot, "timer") and callable(self.robot.timer):
            self.robot.timer(seconds)
        else:
            time.sleep(seconds)

    def _robot_batch_ctx(self) -> Any:
        """Batch context manager if backend supports it (HSE), nullcontext otherwise.

        Allows `_execute_pick_place` to consolidate 5-7 motion calls into 1 INFORM
        upload — reduces overhead from ~1-2s/trial to ~200ms/trial (HSE backend M3).

        IMPORTANT: with `use_yrc_ik`, motion is Cartesian pose (4x4 → P-var BASE,
        YRC handles IK). HSE backend does NOT yet support Cartesian in batch/ultra-fast
        (`_move_pose` raises NotImplementedError) → batch would cause EVERY Cartesian
        MoveJ/MoveL to raise, robot won't move. Therefore do NOT batch when yrc — each
        move runs single-shot (correct, just costs a few extra INFORM uploads/trial).
        Batching is only enabled for the joint-list path (client IK).
        """
        if self.config.get("use_yrc_ik", False):
            return contextlib.nullcontext()
        if hasattr(self.robot, "batch") and callable(self.robot.batch):
            return self.robot.batch()
        return contextlib.nullcontext()

    def _predict_safety(self, joint_waypoints_rad: list[list[float]]) -> str | None:
        """UC2 — Pre-execute trajectory prediction.

        Runs pure-Python FK on the full trajectory → checks joint limits +
        self-collision along the entire path. Costs ~50ms/trial but catches
        unsafe paths that single-point MoveJ_Test misses.

        Args:
            joint_waypoints_rad: List of 6-tuple joint configs (radians).

        Returns:
            None if safe. String describing the violation if unsafe.
        """
        if not self.config.get("predictive_safety_enabled", False):
            return None

        try:
            from src.orchestrator.kinematics import (
                check_joint_limits, check_self_collision_spheres,
                interpolate_joints,
            )
            from src.orchestrator.kinematics.urdf_chain import gp7_urdf
        except ImportError as e:                    # noqa: BLE001
            logger.debug("Kinematics module could not be imported: %s", e)
            return None

        if len(joint_waypoints_rad) < 2:
            return None                              # Not enough waypoints to predict

        # Use the SAME URDF chain as IK + actual motion. The old gp7_default() DH
        # SKELETON's joint origins do NOT match the real robot (verified URDF=RoboDK
        # 0.00mm), so its self-collision geometry was wrong — false negatives in the
        # dangerous direction. Reuse the IK cache when present.
        model = getattr(self, "_dh_model_cached", None)
        if model is None:
            base_rpy_deg = self.config.get("robot_base_rpy_deg", (0.0, 0.0, 0.0))
            model = gp7_urdf(
                base_xyz_mm=tuple(self.config.get("robot_base_xyz_mm", (0.0, 0.0, 0.0))),
                base_rpy_rad=tuple(float(np.deg2rad(d)) for d in base_rpy_deg),
                tool_offset_mm=float(self.config.get("robot_tool_offset_mm", 0.0)))
            self._dh_model_cached = model
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
        self, segments: list[tuple[np.ndarray, str]]
    ) -> str | None:
        """Build a joint trajectory from (pose, move-kind) segments and check safety.

        `segments` = [(T_world 4x4, kind), …] where kind is 'movj' or 'movl' — the
        move type EXECUTION uses for that leg. A MoveL runs as a CARTESIAN straight
        line, which traces a different joint path than joint-linear interpolation, so
        MoveL legs are sampled in Cartesian space (interpolate_cartesian → IK each
        sample). MoveJ legs use the endpoint (joint-linear, matching execution).

        Solves IK client-side (URDF DLS) and applies the SAME ±360° wrap execution
        does, against a RUNNING previous-joint state, so the validated trajectory is
        the one actually commanded. Returns None if disabled / IK fails (let the main
        flow handle it) / safe; a reason string if unsafe.
        """
        if not self.config.get("predictive_safety_enabled", False):
            return None
        if not segments:
            return None
        from src.orchestrator.kinematics import interpolate_cartesian

        waypoints_rad: list[list[float]] = []
        prev_deg: list[float] | None = None
        prev_pose: np.ndarray | None = None
        if self._current_joints is not None:
            prev_deg = [float(q) for q in self._current_joints]
            waypoints_rad.append([np.deg2rad(q) for q in prev_deg])
        for T, kind in segments:
            # MoveL → sample the straight line (drop the start pose, already a
            # waypoint); MoveJ / first leg / unknown-start → endpoint only.
            if kind == "movl" and prev_pose is not None:
                poses = interpolate_cartesian(prev_pose, T, n_steps=8)[1:]
            else:
                poses = [np.asarray(T, dtype=float)]
            for T_s in poses:
                # Seed from the RUNNING prev joints (chained, like execution) so the
                # predicted IK branch matches the commanded one.
                joints_deg = self._solve_ik_client(T_s, seed_deg=prev_deg)
                if joints_deg is None:
                    # Fail-SAFE: a pose the predictor cannot solve is treated as
                    # UNSAFE (a mid-MoveL pose that fails IK = the controller's
                    # straight-line MoveL would alarm). Reject the trial instead of
                    # silently skipping ALL predictive validation (the old fail-OPEN
                    # 'return None' which the caller reads as 'proceed').
                    pos = np.asarray(T_s)[:3, 3].round(1).tolist()
                    logger.warning("Predictive safety: IK unreachable at world %s", pos)
                    return f"predicted_ik_unreachable @ {pos}"
                joints_deg = self._wrap_joints_toward(joints_deg, prev_deg)
                prev_deg = joints_deg
                waypoints_rad.append([np.deg2rad(q) for q in joints_deg])
            prev_pose = np.asarray(T, dtype=float)

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

    @staticmethod
    def _wrap_joints_toward(target_joints, prev_joints):
        """Wrap each target joint ±360° to be closest to `prev_joints` (shortest
        path). Shared by execution (`_normalize_target_joints`) and predictive
        safety so both reason about the SAME commanded joints."""
        if prev_joints is None or target_joints is None:
            return target_joints
        result = list(target_joints)
        n = min(len(result), len(prev_joints))
        for i in range(n):
            cur = float(prev_joints[i])
            tgt = float(result[i])
            while tgt - cur > 180.0:
                tgt -= 360.0
            while tgt - cur < -180.0:
                tgt += 360.0
            result[i] = tgt
        return result

    def _normalize_target_joints(self, target_joints):
        """Wrap each target joint ±360° to be closest to `_current_joints`.

        MoveJ interpolates LINEARLY in joint space. If home_joints[J4]=-180 and
        current J4=+180 (same physical orientation), MoveJ would rotate -360° →
        "spin". Wrap the target toward current via the shortest path.

        Yaskawa GP7 large joint ranges (R/T up to ±360) allow multiple
        representations of the same orientation — normalization is mandatory to
        avoid full-rotation artifacts.
        """
        return self._wrap_joints_toward(target_joints, self._current_joints)

    def _solve_ik_client(self, target_T_world: np.ndarray,
                         seed_deg: list[float] | None = None) -> list[float] | None:
        """Client-side IK: **Pieper analytical** (nearest-branch) → DLS fallback.

        Pieper exact (~1e-10mm, deterministic, ~50µs) + selects the branch nearest
        to `_current_joints` for continuous motion without wrist spin — this is the
        same solver used by the app GUI (Find branches / Change Config). DLS
        (`inverse_kinematics_seeded`) only serves as fallback when Pieper returns
        empty (pose fully out of reach — rare). URDF chain verified match RoboDK 0.00mm.

        Args:
            target_T_world: 4x4 pose in WORLD frame (orchestrator native).

        Returns:
            Joints (degrees) nearest to `_current_joints`, or None if unreachable.
        """
        try:
            from .kinematics import inverse_kinematics_seeded
            from .kinematics.pieper_gp7 import (
                inverse_kinematics_pieper_gp7_nearest,
            )
            from .kinematics.urdf_chain import gp7_urdf
        except ImportError:
            return None

        tool_offset = float(self.config.get("robot_tool_offset_mm", 0.0))
        if not hasattr(self, "_dh_model_cached"):
            # URDF chain — verified match RoboDK SolveFK (0.00mm diff). Pass
            # BOTH base_xyz + base_rpy to match _world_to_robot_base (YRC path) —
            # if robot base is rotated and rpy is missing, IK will solve the wrong frame.
            base_rpy_deg = self.config.get("robot_base_rpy_deg", (0.0, 0.0, 0.0))
            self._dh_model_cached = gp7_urdf(
                base_xyz_mm=tuple(self.config.get("robot_base_xyz_mm", (0.0, 0.0, 0.0))),
                base_rpy_rad=tuple(float(np.deg2rad(d)) for d in base_rpy_deg),
                tool_offset_mm=tool_offset,
            )
        model = self._dh_model_cached

        # Seed nearest-branch: explicit seed_deg (e.g. the RUNNING prev-joint state
        # threaded by predictive safety so prediction branch-chains exactly like
        # execution) wins; else current joints; else (first trial / after a YRC_POSE
        # move that sets None) HOME instead of [0]*6 — prevents Pieper from picking a
        # "near-zero" branch that differs from the actual pose → large MoveJ jump.
        seed = seed_deg if seed_deg is not None else self._current_joints
        if seed is not None:
            q_init = [np.deg2rad(q) for q in seed]
        else:
            home = self.config.get("home_joints_deg") or [0.0] * 6
            q_init = [np.deg2rad(q) for q in home]

        # Path 1: Pieper analytical → branch nearest to q_init (exact + correct branch).
        # Pieper solves for FLANGE (tool0) and does NOT automatically add tool_offset_mm.
        # When a tool offset is present, retract the target TCP to the flange (back off
        # tool_offset along the tool Z axis = column Z of the pose) BEFORE calling Pieper
        # — consistent with industrial controller convention (IK about wrist center, TCP
        # is a fixed offset). The returned solution still places TCP at the target.
        if abs(tool_offset) > 1e-6:
            pieper_target = target_T_world.copy()
            pieper_target[:3, 3] = (
                target_T_world[:3, 3] - tool_offset * target_T_world[:3, 2])
        else:
            pieper_target = target_T_world
        try:
            sol_rad = inverse_kinematics_pieper_gp7_nearest(
                model, pieper_target, q_init)
        except Exception as e:                                  # noqa: BLE001
            logger.debug("Pieper IK error (%s) → fallback DLS", e)
            sol_rad = None
        # Path 2 (fallback): DLS multi-seed — full FK (handles tool offset internally).
        if sol_rad is None:
            sol_rad = inverse_kinematics_seeded(model, target_T_world, q_init)
        if sol_rad is None:
            return None
        return [float(np.rad2deg(q)) for q in sol_rad]

    def _world_to_robot_base(self, T_world: np.ndarray) -> np.ndarray:
        """Convert 4x4 pose from world → robot BASE frame (for HSE Cartesian).

        Wraps frame_convert.world_to_robot_base() with config-driven base pose.
        """
        from .frame_convert import world_to_robot_base
        base_xyz = tuple(self.config.get("robot_base_xyz_mm", (0.0, 0.0, 0.0)))
        base_rpy = tuple(self.config.get("robot_base_rpy_deg", (0.0, 0.0, 0.0)))
        return world_to_robot_base(T_world, base_xyz, base_rpy)

    def _robot_base_to_world(self, T_base: np.ndarray) -> np.ndarray:
        """Inverse of _world_to_robot_base(), using the exact same convention."""
        from .frame_convert import world_to_robot_base

        base_xyz = tuple(
            self.config.get("robot_base_xyz_mm", (0.0, 0.0, 0.0))
        )
        base_rpy = tuple(
            self.config.get("robot_base_rpy_deg", (0.0, 0.0, 0.0))
        )

        # world_to_robot_base(I) = inv(T_world_robot)
        # therefore inverse(...) = T_world_robot.
        T_robot_world = world_to_robot_base(
            np.eye(4, dtype=float), base_xyz, base_rpy
        )
        T_world_robot = np.linalg.inv(T_robot_world)

        return T_world_robot @ np.asarray(T_base, dtype=float)


    def _table_z_base_at_xy(
        self, x_base: float, y_base: float
    ) -> float:
        """Return calibrated table Z in YRC ROBOT BASE at fixed BASE X/Y.

        table_plane is stored in the calibrated project/world frame:
            z = a*x + b*y + c

        Intersect that plane with the YRC BASE-Z line passing through
        (x_base, y_base). This remains correct if the robot base has a
        translation and avoids hard-coding the pedestal height.
        """
        plane = self.config.get("table_plane")

        if not isinstance(plane, dict):
            # SIM / unit-test fallback only.
            #
            # table_top_z_mm is expressed in the project/WORLD frame. Model
            # the simulated table as a flat WORLD plane, then let the existing
            # line/plane intersection below convert it to ROBOT BASE Z.
            #
            # REAL ROBOT MUST FAIL CLOSED: never guess the physical table plane.
            if not self.config.get("is_real_mode", False):
                table_z_world = self.config.get("table_top_z_mm")
                if table_z_world is None:
                    raise RuntimeError(
                        "SIM PLACE-Z requires table_top_z_mm when "
                        "table_plane is absent"
                    )
                try:
                    table_z_world = float(table_z_world)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Invalid SIM table_top_z_mm: {table_z_world!r}"
                    ) from exc
                if not np.isfinite(table_z_world):
                    raise RuntimeError(
                        f"Invalid SIM table_top_z_mm: {table_z_world!r}"
                    )
                plane = {
                    "a": 0.0,
                    "b": 0.0,
                    "c": table_z_world,
                }
            else:
                raise RuntimeError(
                    "PLACE-Z requires calibrated table_plane; none was provided"
                )

        try:
            a = float(plane["a"])
            b = float(plane["b"])
            c = float(plane["c"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid table_plane configuration: {plane!r}"
            ) from exc

        # Two points one millimetre apart along BASE +Z.
        T0 = np.eye(4, dtype=float)
        T1 = np.eye(4, dtype=float)

        T0[:3, 3] = [float(x_base), float(y_base), 0.0]
        T1[:3, 3] = [float(x_base), float(y_base), 1.0]

        p0_w = self._robot_base_to_world(T0)[:3, 3]
        p1_w = self._robot_base_to_world(T1)[:3, 3]

        d = p1_w - p0_w

        den = float(
            d[2] - a * d[0] - b * d[1]
        )

        if not np.isfinite(den) or abs(den) < 1e-9:
            raise RuntimeError(
                "Robot BASE-Z is nearly parallel to calibrated table plane"
            )

        z_base = float(
            (
                a * p0_w[0]
                + b * p0_w[1]
                + c
                - p0_w[2]
            )
            / den
        )

        if not np.isfinite(z_base):
            raise RuntimeError(
                f"Invalid calibrated table Z at BASE "
                f"({x_base:.1f}, {y_base:.1f})"
            )

        return z_base


    def _detected_target_height_mm(
        self, obj: dict[str, Any]
    ) -> tuple[float, float, float, float]:
        """Measure target-part height from RGB-D top Z and the table plane.

        For a normal trial:
            H_target = Z_top - Z_table

        For STACKED:
            H_target = Z_top - Z_table - H_support

        Returns:
            target_height_mm,
            support_height_mm,
            detected_top_z_base,
            table_z_at_pick_base
        """
        pose_world = np.asarray(
            obj.get("pose_base"), dtype=float
        )

        if pose_world.size < 3:
            raise RuntimeError(
                f"Detection has invalid pose_base: {obj.get('pose_base')!r}"
            )

        # Detection pose_base is calibrated project/world XYZ.
        # Convert the detected TOP point to the actual YRC ROBOT BASE.
        T_top_world = np.eye(4, dtype=float)
        T_top_world[:3, 3] = pose_world[:3]

        top_base = self._world_to_robot_base(T_top_world)[:3, 3]

        table_pick_z = self._table_z_base_at_xy(
            float(top_base[0]),
            float(top_base[1]),
        )

        stack_on = str(
            getattr(self, "current_stack_on", "") or ""
        ).strip().lower()

        support_h_mm = 0.0

        if stack_on:
            heights = self.config.get("class_heights_mm") or {}

            if stack_on not in heights:
                raise RuntimeError(
                    f"STACKED trial has no configured support height for "
                    f"{stack_on!r}"
                )

            support_h_mm = float(heights[stack_on])

            if (
                not np.isfinite(support_h_mm)
                or support_h_mm <= 0.0
            ):
                raise RuntimeError(
                    f"Invalid support height for {stack_on!r}: "
                    f"{support_h_mm}"
                )

        total_height_mm = float(
            top_base[2] - table_pick_z
        )

        target_height_mm = float(
            total_height_mm - support_h_mm
        )

        if (
            not np.isfinite(target_height_mm)
            or target_height_mm <= 0.0
            or target_height_mm > 350.0
        ):
            raise RuntimeError(
                "Unsafe/invalid detected target height: "
                f"top={top_base[2]:.1f}, "
                f"table={table_pick_z:.1f}, "
                f"support={support_h_mm:.1f} "
                f"-> target_h={target_height_mm:.1f} mm"
            )

        return (
            target_height_mm,
            support_h_mm,
            float(top_base[2]),
            float(table_pick_z),
        )


    def _build_dynamic_p2_base(
        self,
        p0_base: np.ndarray,
        obj: dict[str, Any],
        *,
        emit_log: bool = False,
    ) -> tuple[np.ndarray, float]:
        """Build safe DTWINRUN P002 in YRC ROBOT BASE.

        P2.Z is reconstructed from:
            calibrated table Z at PLACE
            + detected target-object height
            + actual TCP-to-object-top offset used at PICK.

        It therefore does NOT inherit the support object's height from P0.
        """

        # SIM_DYNAMIC_P2_FALLBACK_V2
        # Measured dynamic PLACE-Z is a physical-cell feature. Mock/sim
        # detections are not guaranteed to share the calibrated table
        # geometry, so deriving target height from synthetic top-Z is invalid.
        #
        # SIM keeps the configured legacy place_position in WORLD/project
        # coordinates, then converts that pose to ROBOT BASE because this
        # helper's return contract is a BASE-frame pose.
        #
        # REAL mode does NOT enter this branch and continues through the
        # calibrated, fail-closed measured-height PLACE-Z logic below.
        if not self.config.get("is_real_mode", False):
            place_xyz_world = np.asarray(
                self.config["place_position"], dtype=float
            ).copy()
            if place_xyz_world.shape != (3,):
                raise ValueError(
                    "place_position must contain exactly X,Y,Z"
                )
            if not np.all(np.isfinite(place_xyz_world)):
                raise RuntimeError(
                    f"Invalid SIM place_position: {place_xyz_world.tolist()}"
                )

            place_yaw_deg = float(
                self.config.get("place_yaw_deg", 5.3)
            )
            place_world = make_grasp_pose(
                place_xyz_world,
                yaw_rad=np.deg2rad(place_yaw_deg),
                yaw_offset_deg=0.0,
            )
            return self._world_to_robot_base(place_world), place_yaw_deg

        tool_offset = float(
            self.config.get("robot_tool_offset_mm", 0.0)
        )

        if abs(tool_offset) > 1e-6:
            raise RuntimeError(
                "Dynamic P2 fixed-job path requires "
                "robot_tool_offset_mm=0.0"
            )

        p0_base = np.asarray(p0_base, dtype=float)

        if p0_base.shape != (4, 4):
            raise ValueError(
                f"p0_base must be 4x4, got {p0_base.shape}"
            )

        place_xyz = np.asarray(
            self.config["place_position"], dtype=float
        ).copy()

        if place_xyz.shape != (3,):
            raise ValueError(
                "place_position must contain exactly X,Y,Z"
            )

        (
            target_h_mm,
            support_h_mm,
            detected_top_z_base,
            table_pick_z_base,
        ) = self._detected_target_height_mm(obj)

        # Preserve the exact TCP-to-object-top relation selected at PICK.
        #
        # Normal objects usually give ~0 mm.
        # Plastic currently includes the configured +10 mm class bias.
        tcp_top_offset_mm = float(
            p0_base[2, 3] - detected_top_z_base
        )

        table_place_z_base = self._table_z_base_at_xy(
            float(place_xyz[0]),
            float(place_xyz[1]),
        )

        p2_z = float(
            table_place_z_base
            + target_h_mm
            + tcp_top_offset_mm
        )

        if not np.isfinite(p2_z):
            raise RuntimeError(
                f"Invalid computed P2.Z: {p2_z}"
            )

        # ENVIRONMENT SAFETY GATE:
        # P2 TCP must never be below the same jaw-tip/table floor used
        # during PICK planning.
        safety_margin_mm = float(
            self.config.get("table_safety_margin_mm", 20.0)
        )

        min_safe_p2_z = float(
            table_place_z_base + safety_margin_mm
        )

        if p2_z < min_safe_p2_z:
            raise RuntimeError(
                "UNSAFE P2 rejected before JOB_START: "
                f"P2.Z={p2_z:.1f} < "
                f"table_place.Z({table_place_z_base:.1f}) + "
                f"safety({safety_margin_mm:.1f}) = "
                f"{min_safe_p2_z:.1f} mm"
            )

        place_xyz[2] = p2_z

        self._last_support_height_mm = support_h_mm
        self._last_place_z_mm = p2_z

        place_yaw_deg = float(
            self.config.get("place_yaw_deg", 5.3)
        )

        p2_base = make_grasp_pose(
            place_xyz,
            yaw_rad=np.deg2rad(place_yaw_deg),
            yaw_offset_deg=0.0,
        )

        if emit_log:
            logger.info(
                "PLACE-Z DETECTED: class=%s stack_on=%s | "
                "topZ=%.1f tablePickZ=%.1f supportH=%.1f "
                "-> targetH=%.1f | tcpTopOff=%+.1f | "
                "tablePlaceZ=%.1f -> P2.Z=%.1f",
                obj.get("class_name", "?"),
                str(getattr(self, "current_stack_on", "") or "-"),
                detected_top_z_base,
                table_pick_z_base,
                support_h_mm,
                target_h_mm,
                tcp_top_offset_mm,
                table_place_z_base,
                p2_z,
            )

        return p2_base, place_yaw_deg


    def _solve_ik_routed(self, target_T_world: np.ndarray):
        """Select IK source: YRC controller (Cartesian) or client-side (Pieper/DLS).

        Priority:
          1. use_yrc_ik=True → send Cartesian pose directly to backend (YRC handles IK).
             Returns ("YRC_POSE", T_base).
          2. Default → client-side IK (Pieper analytical, DLS fallback).
             Returns (joint_list, None).

        Returns:
            ("YRC_POSE", T_base): caller calls MoveJ(T_base) Cartesian
            (joint_list, None): caller calls MoveJ(joint_list)
            (None, None): IK fail
        """
        # Path 1: YRC handles IK — send Cartesian pose directly.
        # TOOL-FRAME CONSISTENCY: the app owns the tool offset (robot_tool_offset_mm).
        # Retract the TCP target back to the FLANGE along tool Z (identical to the
        # client path above) and command the FLANGE pose with TOOL00 (the HSE backend
        # for the experiment is created with tool_no=0). So BOTH IK paths use the SAME
        # app tool offset, and neither depends on the controller's TOOL01 matching.
        if self.config.get("use_yrc_ik", False):
            tool_offset = float(self.config.get("robot_tool_offset_mm", 0.0))
            flange_T = target_T_world
            if abs(tool_offset) > 1e-6:
                flange_T = target_T_world.copy()
                flange_T[:3, 3] = (
                    target_T_world[:3, 3] - tool_offset * target_T_world[:3, 2])
            T_base = self._world_to_robot_base(flange_T)
            return ("YRC_POSE", T_base)

        # Path 2: client-side numerical IK (default — URDF chain match RoboDK 0.00mm)
        joint_list = self._solve_ik_client(target_T_world)
        if joint_list is None:
            logger.warning("Client IK (Pieper/DLS) fail at world %s",
                           target_T_world[:3, 3].round(1).tolist())
            return (None, None)
        return (joint_list, None)

    def _move_j_via_ik(self, target_T: np.ndarray) -> None:
        """MoveJ via IK route (YRC controller or client DLS)."""
        result, pose = self._solve_ik_routed(target_T)

        # YRC IK path — pass 4x4 pose directly, controller handles IK
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
        """MoveL via IK route — same pattern as MoveJ."""
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
    # Main cycle
    # ────────────────────────────────────────────────────────────

    def run_one_cycle(self, trial_id: int = -1) -> bool:
        """Execute one pick-and-place cycle.

        Returns:
            True if grasp-and-place succeeded, False if failed or no objects detected.
        """
        self.sm.reset()
        self._last_support_height_mm = 0.0
        self._last_place_z_mm = None
        t_start = time.time()
        # Every cycle is one ATTEMPT. Counting here (not after reachability) so
        # detection-miss / timeout / unreachable failures are included in the
        # denominator — otherwise the success rate is inflated and the trial counts
        # under-report (the GUI/thesis metrics read these).
        self.stats["attempted"] += 1

        # Reset viewport scene to the initial template (if backend supports it) —
        # prevents objects from drifting across multiple trials.
        if hasattr(self.robot, "reset_scene"):
            try:
                self.robot.reset_scene()
            except Exception as e:  # noqa: BLE001
                logger.debug("reset_scene error: %s", e)

        # ─── DETECT ───
        self.sm.transition_to(PickState.DETECT)
        self._last_frame_path = ""

        # IMPORTANT: detection_timeout_s is a WINDOW for finding a usable
        # object, not merely a timeout for receiving one queue message.
        #
        # The old code consumed exactly one perception message and returned
        # detection_miss immediately when that frame contained no usable pose.
        # Reflective Inox can intermittently lose RGB-D depth for a frame, so a
        # single bad frame caused false misses in ~0.1 s even though the config
        # allowed 2 s.  Keep consuming fresh frames until a usable object is
        # available or the whole detection window expires.
        detect_timeout = float(self.config["detection_timeout_s"])
        detect_deadline = time.monotonic() + detect_timeout
        objects: list[dict[str, Any]] = []
        received_frames = 0
        empty_frames = 0

        while True:
            remaining = detect_deadline - time.monotonic()
            if remaining <= 0.0:
                break

            try:
                det_msg = self.queue.get(timeout=remaining)
            except queue.Empty:
                break

            received_frames += 1
            frame_path = det_msg.get("frame_path", "")
            if frame_path:
                self._last_frame_path = frame_path

            objects = self._select_objects(det_msg)
            if objects:
                if empty_frames:
                    logger.info(
                        "Trial %d: detection recovered after %d empty frame(s) "
                        "within %.2fs window",
                        trial_id,
                        empty_frames,
                        detect_timeout,
                    )
                break

            empty_frames += 1
            # Keep logs useful without flooding them at camera/inference rate.
            if empty_frames == 1 or empty_frames % 5 == 0:
                logger.info(
                    "Trial %d: no usable object in frame %d; retrying "
                    "(%.2fs remaining)",
                    trial_id,
                    received_frames,
                    max(0.0, detect_deadline - time.monotonic()),
                )

        if not objects:
            self.stats["failed"] += 1
            if received_frames == 0:
                logger.warning(
                    "Trial %d: no perception message received within %.2fs",
                    trial_id,
                    detect_timeout,
                )
                self._log_trial(
                    trial_id, False, "detection_timeout", t_start, None
                )
            else:
                logger.info(
                    "Trial %d: no usable object after %d frame(s) in %.2fs",
                    trial_id,
                    received_frames,
                    detect_timeout,
                )
                self.sm.transition_to(PickState.IDLE, "no objects")
                self._log_trial(
                    trial_id, False, "detection_miss", t_start, None
                )
            return False

        # ─── PLAN: try each object until one is reachable ───
        self.sm.transition_to(PickState.PLAN)
        dz = self.config["approach_height_mm"]

        # Hard clamp for fingertip TCP: never go below table_top + safety.
        # Guarantees no table penetration regardless of adaptive offset errors.
        table_top = float(self.config["table_top_z_mm"])
        safety_margin = float(self.config["table_safety_margin_mm"])
        min_grasp_z = table_top + safety_margin
        max_offset = float(self.config["grasp_depth_offset_mm"])

        for obj in objects:
            xyz_base = np.array(obj["pose_base"], dtype=float).copy()
            # pose_base[2] is Z of the object TOP (camera looks down → depth is top).
            # Adaptive offset: clipped to object height so fingertip enters the body.
            obj_height = obj.get("height_mm")
            effective_offset = grasp_depth_offset(obj_height, max_offset, safety_margin)
            target_z = xyz_base[2] - effective_offset
            # HARD CLAMP: regardless of offset, TCP never goes below min_grasp_z.
            clamped_z = max(target_z, min_grasp_z)
            if clamped_z > target_z:
                logger.info(
                    "Grasp Z clamped: target=%.0f → %.0f (table_safety, "
                    "fingertip held %dmm above table)",
                    target_z, clamped_z, int(safety_margin),
                )
            # PLASTIC_PICK_Z_GUARD_V1
            # Restore the previous Plastic-only pick protection.
            #
            # This does NOT change RGB-D measurements or reject a detection.
            # It only raises the final P0 sent to the robot if Plastic would
            # otherwise be picked too close to the table.
            cls_name = str(obj.get("class_name", "")).lower()

            if cls_name == "plastic_box":
                class_heights = self.config.get("class_heights_mm", {}) or {}
                known_h = float(class_heights.get("plastic_box", 30.0))

                insertion_mm = float(
                    self.config.get("low_object_grasp_insertion_mm", 0.0)
                )
                insertion_mm = max(0.0, min(insertion_mm, known_h))

                # Old low-object guard:
                #   P0 >= table_top + Plastic height - insertion
                plastic_floor_z = table_top + known_h - insertion_mm

                if clamped_z < plastic_floor_z:
                    logger.warning(
                        "PLASTIC PICK Z GUARD: P0 %.1f -> %.1f mm "
                        "(height=%.1f, insertion=%.1f)",
                        clamped_z,
                        plastic_floor_z,
                        known_h,
                        insertion_mm,
                    )
                    clamped_z = plastic_floor_z

                # Old Plastic class-specific bias: +10 mm by config.
                class_z_bias = self.config.get(
                    "class_pick_z_bias_mm", {}
                ) or {}
                z_bias_mm = float(class_z_bias.get("plastic_box", 10.0))

                if z_bias_mm != 0.0:
                    logger.info(
                        "PLASTIC CLASS Z BIAS: P0 Z %.1f -> %.1f mm (%+.1f mm)",
                        clamped_z,
                        clamped_z + z_bias_mm,
                        z_bias_mm,
                    )
                    clamped_z += z_bias_mm

            # B3_INOX_ACTUAL_P0_MOTION_SAFETY_V3
            # Authoritative REAL-motion interlock on the ACTUAL P0 that will be
            # converted to YRC BASE and written to P000.
            #
            # IMPORTANT:
            # - Do NOT repair/replace RGB-D.
            # - Do NOT clamp Inox upward to a nominal height.
            # - If measured P0 is too low, record FAIL and dispatch NO MOTION.
            #
            # The previous V2 gate checked obj["h_depth_mm"]. Live B3 showed
            # h_depth_mm can disagree with the centroid/deprojected Z that actually
            # becomes P0. Therefore safety must inspect `clamped_z` itself.
            cls_name_p0 = str(obj.get("class_name", "")).strip().lower()

            if (
                self.config.get("is_real_mode", False)
                and cls_name_p0 == "inox_box"
            ):
                stack_key_p0 = str(
                    getattr(self, "current_stack_on", "") or ""
                ).strip().lower()

                support_safety_mm = {
                    "": 0.0,
                    "-": 0.0,
                    "none": 0.0,
                    "plastic": 30.0,
                    "plastic_box": 30.0,
                    "inox": 40.0,
                    "inox_box": 40.0,
                    "metal": 85.0,
                    "metal_box": 85.0,
                    "carton": 120.0,
                    "carton_box": 120.0,
                    "wood": 150.0,
                    "wood_box": 150.0,
                }

                support_h_p0 = support_safety_mm.get(stack_key_p0)
                reject_reason_p0 = None

                try:
                    actual_p0_world_z = float(clamped_z)
                    table_top_world_z = float(table_top)
                    finite_p0 = bool(
                        np.isfinite(actual_p0_world_z)
                        and np.isfinite(table_top_world_z)
                    )
                except (TypeError, ValueError):
                    actual_p0_world_z = float("nan")
                    table_top_world_z = float("nan")
                    finite_p0 = False

                if support_h_p0 is None:
                    reject_reason_p0 = (
                        f"unknown_stack_support={stack_key_p0!r}"
                    )
                    min_p0_world_z = float("nan")
                    clearance_over_support = float("nan")
                elif not finite_p0:
                    reject_reason_p0 = "actual_p0_or_table_nonfinite"
                    min_p0_world_z = float("nan")
                    clearance_over_support = float("nan")
                else:
                    min_inox_clearance_mm = 38.0
                    min_p0_world_z = (
                        table_top_world_z
                        + float(support_h_p0)
                        + min_inox_clearance_mm
                    )
                    clearance_over_support = (
                        actual_p0_world_z
                        - table_top_world_z
                        - float(support_h_p0)
                    )

                    if actual_p0_world_z < min_p0_world_z:
                        reject_reason_p0 = (
                            f"clearance={clearance_over_support:.1f}"
                            f"<38.0mm"
                        )

                if reject_reason_p0 is not None:
                    logger.warning(
                        "Trial %d: INOX ACTUAL P0 unsafe for motion: "
                        "P0worldZ=%s tableTop=%.1f stack_on=%s "
                        "supportSafety=%s clearance=%s minP0worldZ=%s "
                        "reason=%s -> trial FAIL, NO MOTION",
                        trial_id,
                        (
                            f"{actual_p0_world_z:.1f}"
                            if np.isfinite(actual_p0_world_z)
                            else "?"
                        ),
                        (
                            table_top_world_z
                            if np.isfinite(table_top_world_z)
                            else float("nan")
                        ),
                        stack_key_p0 or "-",
                        (
                            f"{float(support_h_p0):.1f}"
                            if support_h_p0 is not None
                            else "?"
                        ),
                        (
                            f"{clearance_over_support:.1f}"
                            if np.isfinite(clearance_over_support)
                            else "?"
                        ),
                        (
                            f"{min_p0_world_z:.1f}"
                            if np.isfinite(min_p0_world_z)
                            else "?"
                        ),
                        reject_reason_p0,
                    )
                    self.sm.fail("pick_z_safety_reject")
                    self.stats["failed"] += 1
                    self._log_trial(
                        trial_id,
                        False,
                        "pick_z_safety_reject",
                        t_start,
                        obj,
                    )
                    return False

            xyz_base[2] = clamped_z
            # B3_INOX_P0_MOTION_SAFETY_V2
            # REAL-motion safety only. Do NOT repair or overwrite raw RGB-D.
            #
            # For Inox, refuse motion when raw top-face height is too low.
            # Nominal Inox height is 40 mm. Allow 2 mm sensor tolerance, so
            # measured target height must be >= 38 mm.
            #
            # In stacked scenes h_depth_mm is height above the TABLE, so subtract
            # a conservative support height first. Carton is treated as 120 mm
            # because this cell has both 80-mm and 120-mm cartons.
            if (
                self.config.get("is_real_mode", False)
                and str(obj.get("class_name", "")).lower() == "inox_box"
            ):
                raw_h_depth = obj.get("h_depth_mm")

                stack_key = str(
                    getattr(self, "current_stack_on", "") or ""
                ).strip().lower()

                support_safety_mm = {
                    "": 0.0,
                    "none": 0.0,
                    "-": 0.0,
                    "plastic": 30.0,
                    "plastic_box": 30.0,
                    "inox": 40.0,
                    "inox_box": 40.0,
                    "metal": 85.0,
                    "metal_box": 85.0,
                    "carton": 120.0,
                    "carton_box": 120.0,
                    "wood": 150.0,
                    "wood_box": 150.0,
                }

                support_h = support_safety_mm.get(stack_key)
                reject_reason = None
                measured_target_h = float("nan")

                if support_h is None:
                    reject_reason = f"unknown_stack_support={stack_key!r}"
                else:
                    try:
                        raw_h_depth_f = float(raw_h_depth)
                        if not np.isfinite(raw_h_depth_f):
                            raise ValueError("nonfinite")
                        measured_target_h = raw_h_depth_f - float(support_h)
                    except (TypeError, ValueError):
                        reject_reason = "h_depth_missing_or_nonfinite"

                inox_min_safe_h = 38.0

                if (
                    reject_reason is None
                    and measured_target_h < inox_min_safe_h
                ):
                    reject_reason = (
                        f"measured_target_h={measured_target_h:.1f}"
                        f"<min_safe={inox_min_safe_h:.1f}"
                    )

                if reject_reason is not None:
                    logger.warning(
                        "Trial %d: INOX P0 unsafe for motion: "
                        "h_depth=%s stack_on=%s supportSafety=%.1f "
                        "targetH=%s minSafe=38.0 reason=%s "
                        "-> trial FAIL, NO MOTION",
                        trial_id,
                        raw_h_depth,
                        stack_key or "-",
                        float(support_h) if support_h is not None else float("nan"),
                        (
                            f"{measured_target_h:.1f}"
                            if np.isfinite(measured_target_h)
                            else "?"
                        ),
                        reject_reason,
                    )
                    self.sm.fail("pick_z_safety_reject")
                    self.stats["failed"] += 1
                    self._log_trial(
                        trial_id,
                        False,
                        "pick_z_safety_reject",
                        t_start,
                        obj,
                    )
                    return False

            # B3_INOX_POSE_YAW_REFERENCE_GUARD_V2
            # Safety-only comparison against the known B3 placement yaw.
            # Do NOT replace vision yaw with pose-list yaw.
            if (
                self.config.get("is_real_mode", False)
                and str(obj.get("class_name", "")).lower() == "inox_box"
                and bool(getattr(self, "current_pose_id", ""))
            ):
                expected_pose_yaw = getattr(
                    self, "current_expected_yaw_deg", None
                )

                vision_yaw_rad = obj.get("yaw_base")
                yaw_ref_bad = False
                yaw_ref_reason = ""
                yaw_ref_err = float("nan")
                expected_metric_edge = float("nan")
                vision_yaw_deg = float("nan")

                if expected_pose_yaw is None:
                    yaw_ref_bad = True
                    yaw_ref_reason = "pose_list_yaw_missing"
                else:
                    try:
                        vision_yaw_deg = float(
                            np.rad2deg(float(vision_yaw_rad))
                        )
                        expected_pose_yaw = float(expected_pose_yaw)

                        if not (
                            np.isfinite(vision_yaw_deg)
                            and np.isfinite(expected_pose_yaw)
                        ):
                            raise ValueError("nonfinite")

                        # Project convention: metric yaw is the LONG edge while
                        # pose-list yaw is the marked reference edge, 90 deg apart.
                        expected_metric_edge = expected_pose_yaw - 90.0

                        # Rectangle orientation is 180-deg symmetric.
                        yaw_ref_err = abs(
                            (
                                (
                                    vision_yaw_deg
                                    - expected_metric_edge
                                    + 90.0
                                )
                                % 180.0
                            )
                            - 90.0
                        )

                        if yaw_ref_err > 20.0:
                            yaw_ref_bad = True
                            yaw_ref_reason = (
                                f"yaw_error={yaw_ref_err:.1f}>20.0deg"
                            )
                    except (TypeError, ValueError):
                        yaw_ref_bad = True
                        yaw_ref_reason = "vision_or_pose_yaw_nonfinite"

                if yaw_ref_bad:
                    logger.warning(
                        "Trial %d: INOX pose-yaw safety mismatch: "
                        "poseYaw=%s expectedMetricEdge=%s visionEdge=%s "
                        "err=%s reason=%s -> trial FAIL, NO MOTION",
                        trial_id,
                        expected_pose_yaw,
                        (
                            f"{expected_metric_edge:.1f}"
                            if np.isfinite(expected_metric_edge)
                            else "?"
                        ),
                        (
                            f"{vision_yaw_deg:.1f}"
                            if np.isfinite(vision_yaw_deg)
                            else "?"
                        ),
                        (
                            f"{yaw_ref_err:.1f}"
                            if np.isfinite(yaw_ref_err)
                            else "?"
                        ),
                        yaw_ref_reason,
                    )
                    self.sm.fail("yaw_geometry_reject")
                    self.stats["failed"] += 1
                    self._log_trial(
                        trial_id,
                        False,
                        "yaw_geometry_reject",
                        t_start,
                        obj,
                    )
                    return False

            # B3_INOX_YAW_MOTION_SAFETY_V1
            # Safety-only gate for REAL Inox motion.
            #
            # This does NOT change h_depth_mm and does NOT repair/fallback depth.
            # It only prevents robot motion when the selected Inox footprint is
            # too distorted to trust the metric yaw that drives Rz/J6.
            if (
                self.config.get("is_real_mode", False)
                and str(obj.get("class_name", "")).lower() == "inox_box"
            ):
                yaw_source = str(obj.get("yaw_source", ""))
                rect_l_raw = obj.get("yaw_rect_long_mm")
                rect_w_raw = obj.get("yaw_rect_short_mm")
                near_square = bool(obj.get("yaw_near_square", False))

                bad_yaw_geometry = False
                reason_parts = []

                if yaw_source != "metric_base_minAreaRect":
                    bad_yaw_geometry = True
                    reason_parts.append(f"yaw_source={yaw_source or 'missing'}")

                try:
                    rect_l = float(rect_l_raw)
                    rect_w = float(rect_w_raw)
                    finite_rect = bool(np.isfinite(rect_l) and np.isfinite(rect_w))
                except (TypeError, ValueError):
                    rect_l = float("nan")
                    rect_w = float("nan")
                    finite_rect = False

                if not finite_rect:
                    bad_yaw_geometry = True
                    reason_parts.append("metric_rect_missing_or_nonfinite")
                else:
                    exp_l = 180.0
                    exp_w = 85.0
                    err_l = abs(rect_l - exp_l) / exp_l
                    err_w = abs(rect_w - exp_w) / exp_w

                    if near_square:
                        bad_yaw_geometry = True
                        reason_parts.append("near_square")
                    if err_l > 0.22:
                        bad_yaw_geometry = True
                        reason_parts.append(f"long_err={100.0*err_l:.1f}%")
                    if err_w > 0.35:
                        bad_yaw_geometry = True
                        reason_parts.append(f"short_err={100.0*err_w:.1f}%")

                if bad_yaw_geometry:
                    logger.warning(
                        "Trial %d: INOX yaw geometry unsafe for motion: "
                        "rect=%s x %s mm source=%s h_depth=%s reason=%s "
                        "-> trial FAIL, NO MOTION",
                        trial_id,
                        f"{rect_l:.1f}" if np.isfinite(rect_l) else "?",
                        f"{rect_w:.1f}" if np.isfinite(rect_w) else "?",
                        yaw_source or "missing",
                        obj.get("h_depth_mm"),
                        ",".join(reason_parts),
                    )
                    self.sm.fail("yaw_geometry_reject")
                    self.stats["failed"] += 1
                    self._log_trial(
                        trial_id,
                        False,
                        "yaw_geometry_reject",
                        t_start,
                        obj,
                    )
                    return False

            # Base-frame yaw (camera→base transformed in _select_objects); fall back
            # to the raw pixel-frame yaw only if a producer bypassed _select_objects.
            yaw = float(obj.get("yaw_base", obj["pose_camera"][3]))

            # -------------------------------------------------------------
            # J6-NEAREST PARALLEL-GRIPPER BRANCH
            #
            # Parallel gripper symmetry:
            #     Rz and Rz + 180 deg are physically equivalent.
            #
            # For the current top-down GP7 / FORM=4 posture:
            #     Rz ~= J1 + J6
            # therefore:
            #     J6 ~= Rz - J1
            #
            # Read actual current J6 and predict J1 from target BASE XY.
            # Choose the equivalent Rz branch requiring less J6 rotation.
            # -------------------------------------------------------------
            yaw = float(obj.get("yaw_base", obj["pose_camera"][3]))

            raw_yaw_deg = float(np.rad2deg(yaw))
            yaw_offset_deg = float(self.config["yaw_offset_deg"])
            base_cmd_deg = raw_yaw_deg + yaw_offset_deg

            motion_backend = getattr(self.robot, "backend", self.robot)
            backend_cls = type(motion_backend)

            # J6-nearest selector is ONLY for the physical YRC/HSE path.
            # Do not let SimRobot / MagicMock / generic Cartesian backends
            # enter this branch merely because an attribute is truthy.
            is_real_yrc = bool(
                self.config.get("is_real_mode", False)
                and self.config.get("use_yrc_ik", False)
                and backend_cls.__name__ == "MotomanHSEBackend"
                and backend_cls.__module__.endswith(".motoman_hse")
            )

            if is_real_yrc:

                try:
                    current_joints = [
                        float(v) for v in motion_backend.Joints()
                    ]
                except Exception as exc:
                    raise RuntimeError(
                        f"Cannot read current joints for yaw branch: {exc}"
                    ) from exc

                if len(current_joints) < 6:
                    raise RuntimeError(
                        "Cannot select J6-nearest yaw branch: "
                        f"Joints() returned {current_joints}"
                    )

                current_j1_deg = current_joints[0]
                current_j6_deg = current_joints[5]

                # xyz_base is PROJECT/WORLD. Convert target point to actual
                # YRC ROBOT BASE before estimating J1.
                T_pos = np.eye(4, dtype=float)
                T_pos[:3, 3] = xyz_base
                p_robot = self._world_to_robot_base(T_pos)[:3, 3]

                target_x = float(p_robot[0])
                target_y = float(p_robot[1])

                pred_j1_deg = float(
                    np.rad2deg(np.arctan2(target_y, target_x))
                )

                def _wrap180_deg(a: float) -> float:
                    return (a + 180.0) % 360.0 - 180.0

                candidates = (
                    base_cmd_deg,
                    base_cmd_deg + 180.0,
                )

                scored = []

                for rz_deg in candidates:
                    pred_j6_raw = rz_deg - pred_j1_deg

                    # Represent predicted J6 nearest to current physical J6.
                    pred_j6_near = (
                        current_j6_deg
                        + _wrap180_deg(
                            pred_j6_raw - current_j6_deg
                        )
                    )

                    delta_j6 = pred_j6_near - current_j6_deg

                    scored.append(
                        (
                            abs(delta_j6),
                            rz_deg,
                            pred_j6_near,
                            delta_j6,
                        )
                    )

                scored.sort(key=lambda x: x[0])

                _, selected_yaw_deg, pred_j6_deg, pred_delta_j6 = scored[0]

                alt = scored[1]

                logger.info(
                    "YAW J6-BRANCH raw=%+.2f off=%+.2f "
                    "Jnow=(%+.2f,%+.2f) predJ1=%+.2f | "
                    "RzA=%+.2f predJ6=%+.2f dJ6=%+.2f | "
                    "RzB=%+.2f predJ6=%+.2f dJ6=%+.2f | "
                    "SELECT Rz=%+.2f",
                    raw_yaw_deg,
                    yaw_offset_deg,
                    current_j1_deg,
                    current_j6_deg,
                    pred_j1_deg,
                    scored[0][1],
                    scored[0][2],
                    scored[0][3],
                    alt[1],
                    alt[2],
                    alt[3],
                    selected_yaw_deg,
                )

            else:
                # Simulation/client-IK path: preserve normal requested yaw.
                selected_yaw_deg = base_cmd_deg

            # selected_yaw_deg already includes yaw_offset.
            grasp_T = make_grasp_pose(
                xyz_base,
                np.deg2rad(selected_yaw_deg),
                0.0,
            )

            lift_T = grasp_T.copy()
            lift_T[2, 3] += dz

            # Place pose: X,Y from config; Z = grasp Z (same tool-object offset
            # → object placed at the same table height). yaw + yaw_offset same as
            # grasp → gripper maintains orientation, attached object does not rotate
            # during transfer.
            # Build the SAME P2/P3 that DTWINRUN will actually receive.
            # PLAN/safety poses are WORLD-frame; runtime P-vars are YRC BASE.
            p0_plan_base = self._world_to_robot_base(grasp_T)
            p1_plan_base = self._world_to_robot_base(lift_T)

            try:
                p2_plan_base, _ = self._build_dynamic_p2_base(
                    p0_plan_base, obj
                )
            except RuntimeError as exc:
                msg = str(exc)
                is_measurement_reject = (
                    "Detected target height inconsistent with physical geometry"
                    in msg
                    or "Unsafe/invalid detected target height" in msg
                    or "UNSAFE P2 rejected before JOB_START" in msg
                )

                if not is_measurement_reject:
                    raise

                logger.warning(
                    "Trial %d: physical depth/height geometry rejected in PLAN "
                    "(%s) -> trial FAIL, NO MOTION dispatched",
                    trial_id,
                    msg,
                )
                self.sm.fail("depth_geometry_reject")
                self.stats["failed"] += 1
                self._log_trial(
                    trial_id,
                    False,
                    "depth_geometry_reject",
                    t_start,
                    obj,
                )
                return False

            # P003 = place approach/retreat.
            #
            # Keep X/Y/orientation exactly equal to P002.
            # Raise only BASE-Z by the same amount used between P0 -> P1.
            place_approach_dz_base = float(
                p1_plan_base[2, 3] - p0_plan_base[2, 3]
            )

            if place_approach_dz_base <= 1e-6:
                raise RuntimeError(
                    "Invalid P3 approach: expected P1.Z > P0.Z in "
                    f"YRC BASE, got dz={place_approach_dz_base:.3f} mm"
                )

            p3_plan_base = p2_plan_base.copy()
            p3_plan_base[2, 3] += place_approach_dz_base

            # P002 = actual PLACE.
            place_T = self._robot_base_to_world(p2_plan_base)

            # P003 = PLACE approach/retreat.
            place_lift_T = self._robot_base_to_world(p3_plan_base)

            # Check reachability for the FULL trajectory (contribution C2).
            # Skipped in sim to save API budget (4 calls/trial × N trials).
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
                        "Trial %d: object '%s' not reachable at %s, skipping",
                        trial_id, obj.get("class_name", "?"), unreachable,
                    )
                    continue

            # ─── Execute pick-and-place for this object ───
            # ('attempted' already counted at the top of run_one_cycle.)
            ok = self._execute_pick_place(
                grasp_T, lift_T, place_T, place_lift_T, obj, trial_id,
            )
            self._log_trial(
                trial_id, ok,
                "" if ok else self.sm.history[-1].note,
                t_start, obj, ask_human=True,
            )
            return ok

        # No object was reachable.
        logger.info("Trial %d: all objects out of reach", trial_id)
        self.sm.fail("unreachable")
        self.stats["failed"] += 1
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
        """Execute the grasp-and-place motion sequence. Returns True on success.

        Receives 4 poses pre-computed by PLAN. Poses are in WORLD frame (numpy 4x4);
        the backend (SimRobot / DigitalTwinMirror) handles frame conversion.
        """
        try:
            # Cache current joints only on the first call (trial 1) — subsequent trials
            # use self._current_joints already updated by _move_*_via_ik.
            if self._current_joints is None:
                try:
                    self._current_joints = self._joints_to_list(self.robot.Joints())
                except Exception as e:  # noqa: BLE001
                    logger.warning("Could not read current joints: %s", e)
                if (self._current_joints is None
                        and self.config.get("predictive_safety_enabled", False)):
                    # Fail SAFE: without the start config the predicted trajectory
                    # would OMIT the current→lift segment (the riskiest one) and
                    # validate a TRUNCATED path. Abort rather than move blind.
                    logger.error("Trial %d: no start joints — cannot predict safety, "
                                 "aborting trial", trial_id)
                    self.stats["failed"] += 1
                    self.sm.fail("no_start_joints_for_prediction")
                    return False

            # UC2 — Predictive safety: build the joint trajectory the way EXECUTION
            # runs it — current →(MoveJ) lift →(MoveL) grasp →(MoveL) lift →(MoveJ)
            # place_lift →(MoveL) place →(MoveL) place_lift — verifying joint limits +
            # self-collision in pure Python BEFORE moving. MoveL legs are sampled in
            # Cartesian space so the validated path matches the controller's straight-
            # line execution (joint-linear would validate a DIFFERENT path).
            reason = self._predict_safety_for_trajectory(
                [
                    (lift_T, "movj"),        # -> P1
                    (grasp_T, "movl"),       # P1 -> P0
                    (lift_T, "movl"),        # P0 -> P1
                    (place_lift_T, "movj"),  # P1 -> P3
                    (place_T, "movl"),       # P3 -> P2
                    (place_lift_T, "movl"),  # P2 -> P3
                ]
            )
            if reason is not None:
                logger.warning("Trial %d: predictive safety reject — %s",
                               trial_id, reason)
                self.stats["failed"] += 1
                self.sm.fail(reason)
                return False

            # FIXED DTWINRUN MOTION-ONLY PATH
            # One pre-installed YRC job:
            #
            #   MOVJ P001
            #   MOVL P000
            #   TIMER 1.5
            #   MOVL P001
            #   MOVJ P003
            #   MOVL P002
            #   MOVL P003
            #   MOVJ P004
            #
            # P000-P003 are runtime P-vars written by the PC.
            # P004 is HOME taught on the YRC and is NEVER overwritten here.
            # No IO/gripper commands are issued in this path.
            if self.config.get("use_yrc_ik", False):
                motion_backend = getattr(self.robot, "backend", self.robot)
                if hasattr(motion_backend, "run_dtwinrun_motion"):
                    r0, p0_base = self._solve_ik_routed(grasp_T)
                    r1, p1_base = self._solve_ik_routed(lift_T)
                    if not (r0 == r1 == "YRC_POSE"):
                        raise RuntimeError(
                            "DTWINRUN requires YRC Cartesian poses for P000/P001"
                        )

                    # Build P002 with the SAME helper used by PLAN/safety.
                    p2_base, place_yaw_deg = self._build_dynamic_p2_base(
                        p0_base, obj, emit_log=True
                    )

                    logger.info(
                        "Trial %d: P2 dynamic-Z "
                        "P0.Z=%.1f -> P2=(%.1f, %.1f, %.1f) "
                        "place_yaw=%.1f deg",
                        trial_id,
                        float(p0_base[2, 3]),
                        float(p2_base[0, 3]),
                        float(p2_base[1, 3]),
                        float(p2_base[2, 3]),
                        place_yaw_deg,
                    )

                    logger.info(
                        "Trial %d: fixed JOB DTWINRUN motion-only dispatch",
                        trial_id,
                    )
                    self.sm.transition_to(PickState.APPROACH)
                    motion_backend.run_dtwinrun_motion(
                        pick_T_base=p0_base,
                        approach_T_base=p1_base,
                        place_T_base=p2_base,
                        job_name="DTWINRUN",
                    )

                    self.sm.transition_to(PickState.GRASP)
                    self.sm.transition_to(PickState.LIFT)
                    self.sm.transition_to(PickState.TRANSFER)
                    self.sm.transition_to(PickState.PLACE)
                    self.sm.transition_to(PickState.RETREAT)
                    self.sm.transition_to(PickState.DONE)
                    self._current_joints = None
                    self.stats["successful"] += 1
                    logger.info(
                        "Trial %d: DTWINRUN motion-only SUCCESS | stats=%s",
                        trial_id, self.stats,
                    )
                    return True

            # Batch context: HSE backend collects all motion + IO into 1 INFORM job
            # → 1 FTP upload + 1 JOB_START/trial instead of 5-7 round trips. Other
            # backends (SimRobot) → nullcontext, code runs as-is.
            with self._robot_batch_ctx():
                # P1 = approach / retreat point
                self.sm.transition_to(PickState.APPROACH)
                self._move_j_via_ik(lift_T)

                # P0 = pick point
                self.sm.transition_to(PickState.GRASP)
                self._move_l_via_ik(grasp_T)
                self._gripper(close=True, obj_class=obj.get("class_name"))

                # Retreat linearly to the SAME P1
                self.sm.transition_to(PickState.LIFT)
                self._move_l_via_ik(lift_T)

                # Transfer directly to fixed PLACE_POINT
                self.sm.transition_to(PickState.TRANSFER)
                self._move_j_via_ik(place_T)

                # Release at PLACE_POINT; no second place waypoint
                self.sm.transition_to(PickState.PLACE)
                self._gripper(close=False)

                # Preserve state-machine order; RETREAT has no motion in this scheme.
                self.sm.transition_to(PickState.RETREAT)

            self.sm.transition_to(PickState.DONE)
            self.stats["successful"] += 1
            logger.info(
                "Trial %d: pick-and-place '%s' SUCCESS | stats=%s",
                trial_id, obj.get("class_name", "?"), self.stats,
            )
            if self.config["return_home_after_success"]:
                self._return_home()
            return True

        except Exception as e:  # noqa: BLE001
            self.stats["failed"] += 1
            self.sm.fail(f"motion_error: {e}")
            logger.error("Trial %d: pick failed — %s", trial_id, e)
            self._return_home()
            return False

    def _set_speed(self) -> None:
        """Apply joint/linear speed from config."""
        try:
            self.robot.setSpeed(
                self.config["speed_linear_mm_s"],
                self.config["speed_joint_deg_s"],
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("setSpeed skipped: %s", e)

    def _return_home(self) -> None:
        """Move robot to home — normalize joints to avoid spin artifacts.

        Prefers `config['home_joints_deg']` (supplied from cell layout) over
        `robot.JointsHome()` — because JointsHome() reads from the .robot library
        file (Yaskawa GP7 default = [0,0,0,0,0,0]), NOT the home pose the user set
        in cell_layout.yaml. Using the wrong source → MoveJ to 0 while current is
        at -180 → 180° rotation on J4/J6 = spin artifact.
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
            logger.warning("Could not return home: %s", e)

    def run_n_trials(self, n: int, pre_trial_hook: Any = None) -> dict[str, Any]:
        """Run `n` consecutive trials, returning aggregate statistics.

        Args:
            pre_trial_hook: Optional callable(trial_index_1based) invoked before
                each cycle — the experiment script uses it to print the pose-list
                placement instruction and set `current_pose_id` (paired McNemar
                design, paper §3.7).
        """
        logger.info("Starting %d trials", n)
        for i in range(n):
            logger.info("─── Trial %d/%d ───", i + 1, n)
            if pre_trial_hook is not None:
                pre_trial_hook(i + 1)
            self.run_one_cycle(trial_id=i + 1)
            time.sleep(self.config["inter_trial_delay_s"])

        attempted = max(self.stats["attempted"], 1)
        rate = self.stats["successful"] / attempted
        result = {**self.stats, "success_rate": rate}
        logger.info("Done. %s", result)
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
        ask_human: bool = False,
    ) -> None:
        """Write one trial result row if a TrialLogger is present.

        Besides the base fields, logs the C3 failure context: detected pose in
        base frame, detector confidence, mask area, the RGB frame captured at
        DETECT, and the pre-drawn pose-list id — everything failure_miner needs
        to map a physical failure back into renderer parameters.

        `ask_human` is set on the one path where the robot actually ran the
        pick-and-place: there is something for an operator to score. On a
        detection miss or an unreachable pose nothing moved, so asking would only
        collect a keystroke that repeats what the machine already knows.
        """
        if self.trial_logger is None:
            return
        extra: dict[str, Any] = {
            "frame_path": self._last_frame_path,
            "pose_id": self.current_pose_id,
            "condition": self.current_condition,
            "stack_on": self.current_stack_on,
            "session_id": self.session_id,
            "block_id": self.block_id,
            "operator_id": self.operator_id,
            "support_height_mm": round(
                float(self._last_support_height_mm), 1
            ),
            "place_z_mm": (
                round(float(self._last_place_z_mm), 1)
                if self._last_place_z_mm is not None
                else ""
            ),
        }
        if obj:
            pose_base = obj.get("pose_base")
            if pose_base is not None:
                extra["det_x_mm"] = round(float(pose_base[0]), 1)
                extra["det_y_mm"] = round(float(pose_base[1]), 1)
                extra["det_z_mm"] = round(float(pose_base[2]), 1)
            yaw = obj.get("yaw_base")
            if yaw is None and obj.get("pose_camera") is not None:
                yaw = obj["pose_camera"][3]
            if yaw is not None:
                extra["det_yaw_deg"] = round(float(np.rad2deg(float(yaw))), 1)
            if obj.get("confidence") is not None:
                extra["confidence"] = round(float(obj["confidence"]), 3)
            if obj.get("mask_area"):
                extra["mask_area"] = int(obj["mask_area"])
            # C4 depth-mode context (perception/depth_modes.py).
            if obj.get("depth_used"):
                extra["depth_used"] = obj["depth_used"]
            if obj.get("depth_valid_frac") is not None:
                extra["depth_valid_frac"] = round(float(obj["depth_valid_frac"]), 3)
            if obj.get("height_mm") is not None:
                extra["part_height_mm"] = round(float(obj["height_mm"]), 1)
            if obj.get("size_iou") is not None:
                extra["size_iou"] = round(float(obj["size_iou"]), 3)
            if obj.get("h_depth_mm") is not None:
                extra["h_depth_mm"] = round(float(obj["h_depth_mm"]), 1)
        if ask_human and self.human_score_hook is not None:
            # Ctrl+C propagates: the operator meant to stop, and this row is lost
            # on purpose rather than being written with a made-up verdict.
            extra["human_ok"] = self.human_score_hook(trial_id, success)
        self.trial_logger.log_trial(
            trial_id=trial_id,
            success=success,
            class_name=obj.get("class_name", "") if obj else "",
            cycle_time_s=time.time() - t_start,
            failure_reason=failure_reason,
            final_state=self.sm.state.value,
            extra=extra,
        )
