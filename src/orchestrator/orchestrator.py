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

from .coord_conv import camera_to_base, load_calibration, make_grasp_pose
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

        self._set_speed()

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
            xyz_cam = np.array(pc[:3], dtype=float)
            # Transform the in-plane grasp YAW from the camera image frame into the
            # base frame with the SAME hand-eye rotation as the position. The PCA yaw
            # is a pixel-frame angle (x-right, y-down); carrying it straight into
            # rotz() about base Z leaves it off by the camera mounting rotation +
            # image handedness, mis-aligning the gripper with the object long axis.
            R_BC = np.asarray(self.T_BC, dtype=float)[:3, :3]
            yaw_cam = float(pc[3])
            d_base = R_BC @ np.array([np.cos(yaw_cam), np.sin(yaw_cam), 0.0])
            yaw_base = float(np.arctan2(d_base[1], d_base[0]))
            objects.append({**o, "pose_base": camera_to_base(xyz_cam, self.T_BC),
                            "yaw_base": yaw_base})
        # Object with highest Z (closest to camera / on top) is grasped first.
        objects.sort(key=lambda o: o["pose_base"][2], reverse=True)
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
        t_start = time.time()

        # Reset viewport scene to the initial template (if backend supports it) —
        # prevents objects from drifting across multiple trials.
        if hasattr(self.robot, "reset_scene"):
            try:
                self.robot.reset_scene()
            except Exception as e:  # noqa: BLE001
                logger.debug("reset_scene error: %s", e)

        # ─── DETECT ───
        self.sm.transition_to(PickState.DETECT)
        try:
            det_msg = self.queue.get(timeout=self.config["detection_timeout_s"])
        except queue.Empty:
            logger.warning("Trial %d: no detection received", trial_id)
            self._log_trial(trial_id, False, "detection_timeout", t_start, None)
            return False

        objects = self._select_objects(det_msg)
        if not objects:
            logger.info("Trial %d: no objects detected", trial_id)
            self.sm.transition_to(PickState.IDLE, "no objects")
            self._log_trial(trial_id, False, "detection_miss", t_start, None)
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
            if obj_height and obj_height > 0:
                effective_offset = min(max_offset,
                                       max(safety_margin, obj_height - safety_margin))
            else:
                effective_offset = max_offset
            target_z = xyz_base[2] - effective_offset
            # HARD CLAMP: regardless of offset, TCP never goes below min_grasp_z.
            clamped_z = max(target_z, min_grasp_z)
            if clamped_z > target_z:
                logger.info(
                    "Grasp Z clamped: target=%.0f → %.0f (table_safety, "
                    "fingertip held %dmm above table)",
                    target_z, clamped_z, int(safety_margin),
                )
            xyz_base[2] = clamped_z
            # Base-frame yaw (camera→base transformed in _select_objects); fall back
            # to the raw pixel-frame yaw only if a producer bypassed _select_objects.
            yaw = obj.get("yaw_base", obj["pose_camera"][3])
            grasp_T = make_grasp_pose(xyz_base, yaw, self.config["yaw_offset_deg"])
            lift_T = grasp_T.copy()
            lift_T[2, 3] += dz

            # Place pose: X,Y from config; Z = grasp Z (same tool-object offset
            # → object placed at the same table height). yaw + yaw_offset same as
            # grasp → gripper maintains orientation, attached object does not rotate
            # during transfer.
            place_xyz = np.array(self.config["place_position"], dtype=float).copy()
            place_xyz[2] = xyz_base[2]
            place_T = make_grasp_pose(place_xyz, yaw, self.config["yaw_offset_deg"])
            place_lift_T = place_T.copy()
            place_lift_T[2, 3] += dz

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

        # No object was reachable.
        logger.info("Trial %d: all objects out of reach", trial_id)
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
                [(lift_T, "movj"), (grasp_T, "movl"), (lift_T, "movl"),
                 (place_lift_T, "movj"), (place_T, "movl"), (place_lift_T, "movl")]
            )
            if reason is not None:
                logger.warning("Trial %d: predictive safety reject — %s",
                               trial_id, reason)
                self.stats["failed"] += 1
                self.sm.fail(reason)
                return False

            # Batch context: HSE backend collects all motion + IO into 1 INFORM job
            # → 1 FTP upload + 1 JOB_START/trial instead of 5-7 round trips. Other
            # backends (SimRobot) → nullcontext, code runs as-is.
            with self._robot_batch_ctx():
                self.sm.transition_to(PickState.APPROACH)
                self._move_j_via_ik(lift_T)
                # Cache joints after APPROACH for LIFT reuse — same pose lift_T, no
                # need to re-run SolveIK (-1 API call/trial).
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
                # Cache joints after TRANSFER for RETREAT reuse (-1 API call/trial).
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

    def run_n_trials(self, n: int) -> dict[str, Any]:
        """Run `n` consecutive trials, returning aggregate statistics."""
        logger.info("Starting %d trials", n)
        for i in range(n):
            logger.info("─── Trial %d/%d ───", i + 1, n)
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
    ) -> None:
        """Write one trial result row if a TrialLogger is present."""
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
