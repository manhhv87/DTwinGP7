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
import json
import logging
import math
import queue
import time
from typing import Any

import numpy as np

from .coord_conv import (
    camera_to_base, camera_yaw_to_base, load_calibration, make_grasp_pose)
from .placement_check import (
    MAX_CHECKS, check_placement, judged_detection, top_centre_xy)
from .state_machine import PickPlaceStateMachine, PickState

logger = logging.getLogger(__name__)

# What a blinded console shows for every trial that ends before any motion, in
# place of the logged reason: which guard refused, and on what numbers, differs
# between depth modes (src/utils/blind_console.py reads the "console" extra).
_REFUSED = "Trial {}: refused before motion (details in the log)"


def _rounded(value: Any, digits: int = 1) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return round(out, digits) if math.isfinite(out) else None


def _capture_record(n: int, received_frames: int, frame_path: str,
                    judged: dict[str, Any] | None, codes: list[str],
                    T_BC: Any, table_plane: Any) -> dict[str, Any]:
    """One capture of the placement check, for the placement_captures column: what
    the detection window received, the detection judged (class, top-face centre,
    top height, yaw in base degrees) and the flags it raised. None where there is
    no value. Outcome and the operator's time are added by the caller."""
    rec: dict[str, Any] = {"capture": n, "frames": received_frames,
                           "frame": frame_path or "", "class": None,
                           "top_x_mm": None, "top_y_mm": None,
                           "h_depth_mm": None, "yaw_deg": None}
    if judged is not None:
        rec["class"] = judged.get("class_name")
        centre = top_centre_xy(judged, T_BC, table_plane)
        if centre is not None:
            rec["top_x_mm"], rec["top_y_mm"] = _rounded(centre[0]), _rounded(centre[1])
        rec["h_depth_mm"] = _rounded(judged.get("h_depth_mm"))
        yaw = _rounded(judged.get("yaw_base"), 6)
        rec["yaw_deg"] = None if yaw is None else round(math.degrees(yaw), 1)
    rec["flags"] = list(codes)
    return rec


# Default config values — override via dict passed in.
# place_position must match PlaceZone in cell_layout.yaml (mm, base frame).
_DEFAULT_CONFIG: dict[str, Any] = {
    "robot_name": "Yaskawa GP7",
    "calibration_path": "config/calibration/T_base_camera.npy",
    "place_position": [700.0, 120.0, 700.0],
    # Lift height above grasp/place so the robot approaches vertically (MoveL down),
    # arm does not sweep across workspace. See config/experiment.yaml for why 100.
    "approach_height_mm": 100.0,
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
        # Pose-list target class. When set, detections of every other class
        # are ignored, which matters in stacked trials where both the target
        # and its support are detected. It is the task input, declared in the
        # paper, and the only pose-list field any decision reads; the pose id,
        # condition and stacking column above are logged and nothing more.
        self.current_expected_class: str = ""
        # Joint pose of the taught HOME (P004), read before the first
        # fixed-job cycle of the session; see _check_at_home.
        self._session_home_deg: list[float] | None = None
        self.session_id: str = ""
        self.block_id: str = ""
        self.operator_id: str = ""
        self._last_frame_path: str = ""
        # Optional callable(trial_id, machine_success) -> 1 | 0 | "", asked once per
        # trial that actually moved the robot. The runner sets it under
        # --confirm-each-trial; its answer goes to the `human_ok` column.
        self.human_score_hook: Any = None
        # Placement check (see run_placement_check). The staging the current trial
        # called, a placement_check.CalledPose set per trial by the runner; read by
        # the check and logged, never by a motion decision. The record of the checks
        # made before a trial waits in _pending_placement until that trial's cycle
        # takes it into _placement, which _log_trial writes with the trial's row.
        self.current_called_pose: Any = None
        self._placement: dict[str, Any] = {}
        self._pending_placement: dict[str, Any] = {}

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
            # The trial names the class to pick. That is the task input, and
            # the paper declares it. Only detections of that class are kept, so
            # in a stacked scene, which holds at least two real objects, a
            # support of another class is never chosen merely because it reads
            # higher. Position, height and yaw still come from perception
            # alone. A target the detector labels as another class is
            # therefore a detection miss, never a grasp of the wrong part.
            expected_cls = str(
                getattr(self, "current_expected_class", "") or ""
            ).strip().lower()

            detected_cls = str(
                o.get("class_name", "") or ""
            ).strip().lower()

            if expected_cls and detected_cls != expected_cls:
                continue

            reason = self._mask_off_part(o)
            if reason is not None:
                logger.warning("Detection %s ignored: %s", detected_cls or "?", reason)
                continue

            xyz_cam = np.array(pc[:3], dtype=float)

            # Transform the in-plane grasp YAW from the camera image frame into the
            # base frame with the SAME hand-eye rotation as the position. The PCA yaw
            # is a pixel-frame angle (x-right, y-down); carrying it straight into
            # rotz() about base Z leaves it off by the camera mounting rotation +
            # image handedness, mis-aligning the gripper with the object long axis.
            # Prefer metric BASE yaw produced by PoseExtractor: the class's known
            # top-face rectangle fitted to the mask outline on the top plane
            # (yaw_source known_size_fit), else the outline's minAreaRect edge.
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
        # The detection whose median depth stands highest above the table is
        # grasped first, confidence breaking a tie. The median is measured in
        # every mode, so the three modes choose alike; the grasp height of a
        # plane or fusion detection is the class height by construction and
        # cannot rank anything. Without depth the commanded height stands in.
        table_top = float(self.config.get("table_top_z_mm") or 0.0)

        def _rank(o):
            h = o.get("h_median_mm")
            if h is None or not np.isfinite(float(h)):
                h = float(o["pose_base"][2]) - table_top
            return (float(h), float(o.get("confidence") or 0.0))

        objects.sort(key=_rank, reverse=True)

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

        model = self._dh_model(gp7_urdf)

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

        tool_offset = float(self.config.get("robot_tool_offset_mm", 0.0))

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

        The part carried away is the detected one, so its height is the height
        of its class, not the distance from the table to the top face: that
        distance also contains whatever the part stands on. Reading the pose
        list's stacking column here would feed the experiment's answer sheet
        into the motion, so the support height is derived instead:

            H_target  = height of the detected class
            H_support = (Z_top - Z_table) - H_target

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

        total_height_mm = float(
            top_base[2] - table_pick_z
        )
        # As for the pick clearance, a command built from depth takes the
        # higher of the median and the top band, measured above the same
        # fitted plane: a median dragged onto the support would otherwise
        # hide the support and drop the part at the place.
        h_depth = obj.get("h_depth_mm")
        frac = obj.get("depth_valid_frac")
        if (str(obj.get("depth_used", "")) == "rgbd"
                and h_depth is not None and frac is not None
                and np.isfinite(float(h_depth))
                and float(frac) >= float(
                    self.config.get("min_depth_valid_frac", 0.15))):
            total_height_mm = max(total_height_mm, float(h_depth))

        # Height of the part being carried, from the class the detector
        # reported. A class with two sizes, the carton at 120 and 80 mm high,
        # is told apart by the measured top face, 180 x 120 against 180 x 100
        # mm. The top face height cannot do it once the part stands on another:
        # an 80 mm carton on the 30 mm plastic box reads about 113 mm, nearer
        # 120, and was released about 40 mm above the table. On 15 Oct 2026
        # the measured short side was 102.7 and 105.1 mm for the 80 mm carton
        # and 122.4 to 134.2 mm for the 120 mm one. Height decides only when no
        # rectangle was measured.
        cls = obj.get("class_name")
        variants = [v for v in self._class_variants_mm(cls)
                    if len(v) >= 3 and v[2] == v[2]]
        long_mm = obj.get("yaw_rect_long_mm")
        short_mm = obj.get("yaw_rect_short_mm")
        if not variants:
            known_h = self._class_min_height_mm(cls)
            if known_h is None:
                raise RuntimeError(
                    f"No configured height for detected class {cls!r}")
            target_height_mm = float(known_h)
        elif len(variants) > 1 and long_mm is not None and short_mm is not None:
            target_height_mm = float(min(variants, key=lambda v: (
                abs(float(long_mm) - max(v[0], v[1]))
                + abs(float(short_mm) - min(v[0], v[1]))))[2])
        else:
            target_height_mm = float(min(
                variants, key=lambda v: abs(total_height_mm - float(v[2])))[2])

        # Whatever the part stands on is the remainder. Less than half the
        # thinnest part in the class table cannot be a part, only the bias of
        # the measured top (a 120 mm carton read 125 mm on 15 Oct 2026), and is
        # taken as the table: subtracting it would press the part into the
        # table at the place.
        heights = [float(v[2]) for c in (self.config.get("class_sizes_mm") or {})
                   for v in self._class_variants_mm(c)
                   if len(v) >= 3 and v[2] == v[2]]
        heights += [float(h) for h in
                    (self.config.get("class_heights_mm") or {}).values() if h]
        min_support_mm = 0.5 * min(heights) if heights else 0.0
        support_h_mm = float(total_height_mm - target_height_mm)
        if support_h_mm < min_support_mm:
            support_h_mm = 0.0

        if (
            not np.isfinite(target_height_mm)
            or target_height_mm <= 0.0
            or target_height_mm > 350.0
        ):
            raise RuntimeError(
                "Unsafe/invalid target height for the detected class: "
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
                self.config.get("place_yaw_deg", 0.0)
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
        # The part leaves the gripper held exactly as it was picked: the TCP
        # stays as far above the place table as it was above whatever the part
        # stood on. A flat part therefore lands on the table whatever the
        # camera made of its top face; the earlier rule added the pick's lift
        # above the measured top, so a top read 23.6 mm low (15 Oct 2026,
        # trial 24) released the part about 20 mm up, and only in rgbd.
        hold_above_support_mm = float(
            p0_base[2, 3] - (table_pick_z_base + support_h_mm)
        )

        table_place_z_base = self._table_z_base_at_xy(
            float(place_xyz[0]),
            float(place_xyz[1]),
        )

        p2_z = float(table_place_z_base + hold_above_support_mm)
        # A stacked part's support is its measured top less the part's nominal
        # height, so the part's own height error lands in the support: placed
        # flat on 14 and 15 Oct 2026 the inox read 41.5 mm against 40 and the
        # 120 mm carton 125.4. Released at the computed height, the stacks of
        # 15 Oct would have been pressed 1.8 to 4.1 mm into the table, and a
        # 120 mm carton on the plastic box about 6.6 mm. A stacked part is
        # therefore released stack_release_margin_mm higher: a drop of a few
        # millimetres instead of a press. A flat part needs none.
        if support_h_mm > 0.0:
            p2_z += float(self.config.get("stack_release_margin_mm", 8.0))

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

        # A part lifted to exactly the clearance lands exactly on it at the
        # place; rounding in the frame transforms must not refuse it.
        if p2_z < min_safe_p2_z - 1e-6:
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
            self.config.get("place_yaw_deg", 0.0)
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
                "-> targetH=%.1f | holdAboveSupport=%+.1f | "
                "tablePlaceZ=%.1f -> P2.Z=%.1f",
                obj.get("class_name", "?"),
                str(getattr(self, "current_stack_on", "") or "-"),
                detected_top_z_base,
                table_pick_z_base,
                support_h_mm,
                target_h_mm,
                hold_above_support_mm,
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
        # TOOL-FRAME CONSISTENCY: the app keeps robot_tool_offset_mm at 0 and
        # commands the TCP pose directly; the controller holds the tool length in
        # its own TOOL frame (TOOL01 on this cell, the default of --tool-no). The
        # fixed-job path refuses to run with a non-zero app offset, so the two can
        # never both apply it.
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

    @staticmethod
    def _lower_keys(table) -> dict:
        """Config tables keyed case-insensitively.

        The detector writes its own labels and the config is written by hand;
        a capital letter must not silently disable a guard.
        """
        return {str(k).strip().lower(): v for k, v in (table or {}).items()}

    def _class_variants_mm(self, class_name) -> list[tuple[float, float, float]]:
        """Measured (long, wide, high) variants of a class, from the size table.

        A class with a height but no measured footprint has no variants here.
        That is the honest answer: there is no rectangle to compare against.
        Returning a placeholder would make every trial of that class fail the
        shape clause, which is the opposite of what missing information means.
        """
        if not class_name:
            return []
        raw = self._lower_keys(self.config.get("class_sizes_mm")).get(
            str(class_name).strip().lower())
        if not raw:
            return []
        if not isinstance(raw[0], (list, tuple)):
            raw = [raw]
        return [tuple(float(v) for v in variant) for variant in raw]

    def _class_min_height_mm(self, class_name):
        """Lowest height the detected class can stand at, or None if unknown.

        The size table answers first because it carries every variant; the
        height table is the fallback for a class that has only that entry.
        """
        if not class_name:
            return None
        heights = [float(v[2]) for v in self._class_variants_mm(class_name)
                   if len(v) >= 3 and v[2] == v[2]]
        if heights:
            return min(heights)
        h = self._lower_keys(self.config.get("class_heights_mm")).get(
            str(class_name).strip().lower())
        return float(h) if h else None

    def _long_side_matches_class(self, class_name, long_mm: float,
                                 tol_mm: float) -> bool:
        """True when the measured long side fits one variant of the class.

        Unknown class or no measured footprint: nothing to contradict, so it
        passes. A guard must not refuse a part merely because the config is
        silent about it.
        """
        variants = self._class_variants_mm(class_name)
        longs = [max(float(v[0]), float(v[1])) for v in variants
                 if len(v) >= 2 and v[0] == v[0] and v[1] == v[1]]
        if not longs:
            return True
        return min(abs(long_mm - side) for side in longs) <= tol_mm

    def _reads_its_support(self, obj, tol_mm: float) -> bool:
        """True when depth reads what the part rests on instead of its top.

        Both signs at once: the measured rectangle's short side is more than
        tol_mm off every variant, so the mask takes in a support wider than
        the part, and the measured top (h_depth_mm) lies more than 5 mm below
        the lowest variant's own height, which no part standing on the table
        or on a support can show. Unknown class, no footprint or no depth:
        nothing to contradict.
        """
        short_mm = obj.get("yaw_rect_short_mm")
        top_mm = obj.get("h_depth_mm")
        if short_mm is None or top_mm is None:
            return False
        short_mm, top_mm = float(short_mm), float(top_mm)
        if not (np.isfinite(short_mm) and np.isfinite(top_mm)):
            return False
        variants = [v for v in self._class_variants_mm(obj.get("class_name"))
                    if len(v) >= 3]
        if not variants:
            return False
        wide = all(abs(short_mm - min(float(v[0]), float(v[1]))) > tol_mm
                   for v in variants)
        low = top_mm < min(float(v[2]) for v in variants) - 5.0
        return wide and low

    def _commanded_pick_z(self, obj, table_top: float) -> float:
        """World Z the pick is commanded to, after every height rule.

        The commanded point is the controller's TCP, and the jaw tips sit
        below it. Over 38 picks of 14 Oct 2026 the forward kinematics of the
        logged joints, with the 178 mm tool of the cell layout, put the tool
        23.6 to 23.9 mm under the commanded P0, so a clearance is counted from
        the commanded point and not from the tips. Picks commanded 34.9 mm or
        more above the table ran without a collision; trial 45, commanded
        about 25 mm above it, raised collision alarm 4315 while the jaws
        closed. table_safety_margin_mm is set from that evidence.

        Rules, in order:
          1. the measured top face, less the configured grasp depth;
          2. the commanded point stays table_safety_margin_mm above the table
             and as far above what the part rests on, whose top is taken as
             the measured top less the lowest height of the detected class.
             A part thinner than the margin is lifted by the difference, and
             the same margin then holds at the place, where P2 sits at the
             part's height plus that lift above the table; thicker parts are
             not touched;
          3. the floor of _pick_floor_z, against a collapsed reading;
          4. a per-class bias from the config, never below rule 2.
        All of it reads the detection and the class table, never the pose
        list, and applies alike in every depth mode.
        """
        top_z = float(obj["pose_base"][2])
        cls_name = str(obj.get("class_name", "")).lower()
        margin = float(self.config["table_safety_margin_mm"])
        z = top_z - grasp_depth_offset(
            obj.get("height_mm"), float(self.config["grasp_depth_offset_mm"]),
            margin)

        clear_z = float(table_top) + margin
        h_min = self._class_min_height_mm(obj.get("class_name"))
        if h_min is not None:
            # When the command came from depth, the top is the higher of the
            # median and the top band: the median can collapse onto what lies
            # around the part (15 Oct 2026, trial 23: 3.0 mm against 42.6 mm),
            # and on a stacked part that would send the tool into the support.
            # Never in plane mode, whose command must stay the plane estimate.
            top_est = top_z
            h_depth = obj.get("h_depth_mm")
            frac = obj.get("depth_valid_frac")
            if (str(obj.get("depth_used", "")) == "rgbd"
                    and h_depth is not None and frac is not None
                    and np.isfinite(float(h_depth))
                    and float(frac) >= float(
                        self.config.get("min_depth_valid_frac", 0.15))):
                top_est = max(top_est, float(table_top) + float(h_depth))
            clear_z = max(clear_z, top_est - h_min + margin)
        if z < clear_z:
            logger.info(
                "PICK Z CLEARANCE (%s): P0 %.1f -> %.1f mm "
                "(TCP %.0f mm above the table or the support)",
                cls_name or "?", z, clear_z, margin,
            )
            z = clear_z

        floor_z = self._pick_floor_z(obj.get("class_name"), table_top)
        if floor_z is not None and z < floor_z:
            logger.warning("PICK Z FLOOR (%s): P0 %.1f -> %.1f mm",
                           cls_name or "?", z, floor_z)
            z = floor_z

        bias = float(self._lower_keys(
            self.config.get("class_pick_z_bias_mm")).get(cls_name, 0.0))
        if bias != 0.0:
            logger.info("PICK Z BIAS (%s): P0 Z %.1f -> %.1f mm (%+.1f mm)",
                        cls_name, z, z + bias, bias)
            z = max(z + bias, clear_z)
        return z

    def _mask_off_part(self, obj):
        """Why depth shows the mask lies on the table rather than a part, or None.

        A detection is dropped, before ranking and alike in every depth mode,
        when at least table_share_max (0.3) of its mask's pixels return depth
        within 5 mm of the table plane (table_share, measured in perception).
        That catches the two failures seen on the cell:

          a mask spilling beside a part: on 19 Oct 2026 masks over the
            camera-side end of the inox and the table next to it were picked
            three times in plane and fusion mode, 76 to 119 mm from the part's
            centre (their share reconstructed at about 0.6 to 0.8);
          a false detection on the paper position cards: on 20 Oct 2026 the
            deployed detector labelled groups of cards inox_box in nearly every
            frame, with confidence up to 0.939, flat on the table (share near
            the whole mask).

        It replaces a test on heights (top band or median-depth point below the
        class floor), which threw away genuine mirror-finish parts: a mirror
        top mostly returns no depth, and what returns is its outline band at
        table height and reflections far below the table, so the median falls
        onto the table. On the C4 labelled dataset (real D455 depth) the height
        test discarded 103 of 842 single inox with human masks and 34 of 181
        with the deployed detector's masks; this share discards 1 and 2 (an
        inox against a tall part at 0.331, two masks 40 percent off their
        part), and 0 of the other classes and of 679 stacked parts. On masks
        shifted or grown onto clear table it misses at most 0.4 percent, against
        14 to 72 percent for the height test, and 0.1 percent of flat masks.
        Genuine isolated inox reached 0.223 at most. The share is logged
        (table_share) to keep checking it on the cell.

        A detection at or over the limit is kept when what is left of its mask
        still holds the class's top. Perception fits the outline and the
        known-size rectangle on the mask less its pixels at table height
        (depth_modes.off_table_mask, logged as part_mask_area); when that
        rectangle fits (yaw_source known_size_fit, IoU at least 0.5) and passes
        the shape check (_shape_contradicts_class), the mask lies on a part and
        only spilled beside it. On 23 Oct 2026 (e2chuan-1, P0038 and P0096)
        inox masks over the box and the row of cards beside it read 0.38 to
        0.48 at table height, and the part was never picked. A piece of a part
        stays dropped, because once chosen it would be refused: on rendered
        depth the second inox mask of P0029 (23 Oct, over half the box and the
        cards) was chosen ahead of the whole part and left a rectangle 130 mm
        long. A false detection flat on the cards leaves almost nothing to fit
        and is dropped as before.
        """
        share = obj.get("table_share")
        if share is None or not np.isfinite(float(share)):
            return None
        limit = float(self.config.get("table_share_max", 0.3))
        if float(share) < limit:
            return None
        # What is left still holds the class's top: a spill beside a part.
        if (obj.get("yaw_source") == "known_size_fit"
                and self._shape_contradicts_class(obj) is None):
            return None
        return (f"on the table: {float(share):.2f} of the mask returns depth "
                f"at table height (limit {limit:.2f}) and the rest holds no "
                f"{obj.get('class_name') or 'part'} top")

    def _pick_floor_z(self, class_name, table_top: float):
        """Lowest Z the pick may be commanded to for this class, or None.

        One rule for every class. A depth reading that has collapsed is not
        followed down: the command is held at the lowest top face the detected
        class can have, less height_tolerance_mm. That is the same margin at
        which the depth plausibility clause refuses a trial, so the two cannot
        disagree about what counts as a collapsed reading.

        The floor must not touch a sound reading. Placed at the nominal top
        face it would have lifted the command on 36 of the 55 trials measured
        on 12 and 14 Oct 2026, 21 of the 24 inox among them and by up to 8.2
        mm, because for inox, metal and plastic the median over the whole mask
        sits 1.4 to 2.8 mm below the nominal height. In rgbd that replaces the
        measurement with the class prior, the very difference E2 measures. At
        this level it lifted none.

        None when the class has no known height; the table clamp still holds.
        """
        known_h = self._class_min_height_mm(class_name)
        if known_h is None:
            return None
        tol_mm = float(self.config.get("height_tolerance_mm", 20.0))
        return float(table_top) + max(0.0, known_h - tol_mm)

    def _depth_implausible(self, obj):
        """Why the depth behind the command cannot be believed, or None.

        Two tests, the ones fuse_height already uses: enough of the mask must
        return depth, and the measured top face must not lie more than
        height_tolerance_mm below the lowest variant of the detected class. A
        stacked part reads higher, never lower, so this needs no knowledge of
        what the part stands on.

        Judged only when the commanded height came from depth, that is when
        depth_used is rgbd: always in rgbd mode, and in fusion when depth
        overrode the prior. In plane the command does not come from depth. In
        fusion depth wins only after passing fuse_height's own tests, which
        with the current thresholds are at least as strict as these (valid
        fraction 0.3, the same 20 mm tolerance), so in practice these fire in
        rgbd only.
        """
        if str(obj.get("depth_used", "")) != "rgbd":
            return None
        frac = obj.get("depth_valid_frac")
        min_frac = float(self.config.get("min_depth_valid_frac", 0.15))
        if frac is not None and float(frac) < min_frac:
            return (f"depth valid fraction {float(frac):.2f} "
                    f"below {min_frac:.2f}")
        h_depth = obj.get("h_depth_mm")
        if h_depth is None or not np.isfinite(float(h_depth)):
            # The returns scatter instead of forming a top face, as on a mirror
            # finish; the median behind the command is then no better.
            return "depth does not form a top face"
        h_min = self._class_min_height_mm(obj.get("class_name"))
        tol_mm = float(self.config.get("height_tolerance_mm", 20.0))
        if (h_min is not None and h_depth is not None
                and np.isfinite(float(h_depth))
                and float(h_depth) < h_min - tol_mm):
            return (f"top face {float(h_depth):.1f} mm below the "
                    f"{h_min:.0f} mm the class can stand at")
        return None

    def _shape_contradicts_class(self, obj):
        """Why the measured top face cannot be grasped as the detected class.

        The jaws close along the long axis of the measured rectangle. A mask
        longer than the class can shift the grasp centre along that axis, by
        up to half the excess, towards where a finger comes down; that is
        refused when the long side is more than size_tolerance_mm off every
        variant. The short side alone is not judged: it does not move where a finger
        lands. Judging it on 15 Oct 2026 refused four inox picks whose masks
        were only too wide, with vision yaw 5.7 to 16.8 deg from the called
        yaw, while for inox, whose 199 mm diagonal is shorter than the 216 mm
        jaw opening, no yaw can put a finger on the part. No measurement,
        nothing to contradict.

        When the grasp pose comes from the known-size rectangle (yaw_source
        known_size_fit), the trial is also refused when that rectangle can slide
        more than fit_plateau_max_mm (10) along its long side at the same IoU
        (fit_plateau_mm): the mask then leaves the grasp centre undetermined
        along the axis the jaws close on. On 22 Oct 2026 (kiemtra-2210, first
        run, trial 3, P0216) an inox mask spilling 50 mm past the box's end
        over the cards let the rectangle slide 49 to 50 mm, its centre 31.8 mm
        off the part; the long side (231.5 mm) refused it as well. On the 76
        genuine detections of 20 to 22 Oct with the arm out of view the slide
        was at most 7 mm; on 842 cases built from real masks (spills, cuts,
        merges, wrong classes) adding this test to the long side halves the
        poses that pass with a jaw on the part, 67 to 32, and refuses no
        genuine detection.

        The short side is judged together with the measured top
        (_reads_its_support): a rectangle more than size_tolerance_mm too wide
        whose top reads below the part's own height means depth sees the
        support around a mirror top, and a command from that depth brings the
        fingers down on the support. On 28 Oct 2026 (e2chuan-2, P0118, inox on
        the plastic box) the rectangle read 139 to 141 mm wide and the top 28
        to 30 mm in rgbd and plane, and both picks ended in collision stops
        (alarm 4315); P0062 of 27 Oct read the same in rgbd and fusion. Over
        the 680 trials of e2chuan-1 and e2chuan-2 this refuses those four and
        no successful pick: wide masks of tall cartons read their tops right,
        and inox tops read low came with narrow masks.
        """
        long_mm = obj.get("yaw_rect_long_mm")
        tol_mm = float(self.config.get("size_tolerance_mm", 40.0))
        if long_mm is not None:
            if not self._long_side_matches_class(obj.get("class_name"),
                                                 float(long_mm), tol_mm):
                return (f"top face long side {float(long_mm):.0f} mm, no "
                        f"{obj.get('class_name')} variant within {tol_mm:.0f} mm")
        if self._reads_its_support(obj, tol_mm):
            return (f"top face {float(obj['yaw_rect_short_mm']):.0f} mm wide and "
                    f"{float(obj['h_depth_mm']):.0f} mm high: depth reads what the "
                    f"{obj.get('class_name')} rests on, not its top")
        slide = obj.get("fit_plateau_mm")
        limit = float(self.config.get("fit_plateau_max_mm", 10.0))
        if (obj.get("yaw_source") == "known_size_fit" and slide is not None
                and np.isfinite(float(slide)) and float(slide) > limit):
            return (f"fitted {obj.get('class_name')} rectangle slides "
                    f"{float(slide):.0f} mm along its long side at the same overlap "
                    f"(limit {limit:.0f} mm)")
        return None

    def _depth_contradicts_command(self, obj, commanded_h_mm: float):
        """Why depth refuses this commanded height, or None if it does not.

        The plane estimate stands every part on the table by its class height,
        so a part resting on another one is commanded a whole support too low
        and the tool would be driven into that support. Depth sees the real top
        face, and this is where the two are compared.

        The rule applies only where the commanded height did not come from
        depth. In rgbd the commanded Z is the median depth over the whole mask
        while h_depth_mm is the median of the top band of an eroded mask: two
        deliberately different estimators of one surface. Comparing them lets a
        ragged mask veto a sound command. Cell run of 14 Oct 2026, trial 9: one
        unstacked carton read 124.3 mm one way and 173.3 mm the other, and the
        pick succeeded. In rgbd the depth plausibility clause carries the load.

        Depth is believed down to the fraction that marks a dead mask. Below
        the fusion bar of 0.3 the top band still reads the part: flat inox at
        fractions of 0.215 to 0.394 came within 1.8 mm of its 40 mm, and a
        stacked inox read at 0.267 (15 Oct 2026, trial 47). A plane command
        on a stack with such depth would otherwise go through.

        The datum is the scalar table top, the 95th percentile of the fitted
        plane, while h_depth_mm is measured above the plane itself. Across the
        pick region that understates the height by 1.4 to 2.8 mm, so the
        effective tolerance is near 18 mm rather than 20. The error is
        one-directional and refuses slightly early, the safe direction.
        """
        if str(obj.get("depth_used", "")) == "rgbd":
            return None
        h_depth = obj.get("h_depth_mm")
        frac = obj.get("depth_valid_frac")
        if h_depth is None or frac is None:
            return None
        if float(frac) < float(self.config.get("min_depth_valid_frac", 0.15)):
            return None
        if not np.isfinite(float(h_depth)):
            return None
        tol_mm = float(self.config.get("height_tolerance_mm", 20.0))
        if commanded_h_mm < float(h_depth) - tol_mm:
            return (f"commanded top face {commanded_h_mm:.1f} mm, depth sees "
                    f"{float(h_depth):.1f} mm: the tool would go into whatever "
                    f"the part rests on")
        return None

    def _dh_model(self, gp7_urdf=None):
        """The URDF chain used by both client IK and the reached-pose check.

        Built once. base_xyz + base_rpy are included so the chain solves in the
        WORLD frame, exactly as _world_to_robot_base does for the YRC path.
        """
        if not hasattr(self, "_dh_model_cached"):
            if gp7_urdf is None:
                from .kinematics.urdf_chain import gp7_urdf
            base_rpy_deg = self.config.get("robot_base_rpy_deg", (0.0, 0.0, 0.0))
            self._dh_model_cached = gp7_urdf(
                base_xyz_mm=tuple(self.config.get("robot_base_xyz_mm", (0.0, 0.0, 0.0))),
                base_rpy_rad=tuple(float(np.deg2rad(d)) for d in base_rpy_deg),
                tool_offset_mm=float(self.config.get("robot_tool_offset_mm", 0.0)),
            )
        return self._dh_model_cached

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
        # The record of the placement checks made just before this cycle, empty when
        # none were: taken here, so it goes with this trial's row and no other.
        self._placement, self._pending_placement = self._pending_placement, {}
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
        objects, received_frames = self._detect_objects(trial_id)

        if not objects:
            self.stats["failed"] += 1
            if received_frames == 0:
                logger.warning(
                    "Trial %d: no perception message received within %.2fs",
                    trial_id,
                    float(self.config["detection_timeout_s"]),
                    extra={"console": f"Trial {trial_id}: no camera frame arrived; "
                                      "check the camera (details in the log)"},
                )
                self._log_trial(
                    trial_id, False, "detection_timeout", t_start, None
                )
            else:
                logger.info(
                    "Trial %d: no usable object after %d frame(s) in %.2fs",
                    trial_id,
                    received_frames,
                    float(self.config["detection_timeout_s"]),
                    extra={"console": _REFUSED.format(trial_id)},
                )
                self.sm.transition_to(PickState.IDLE, "no objects")
                self._log_trial(
                    trial_id, False, "detection_miss", t_start, None
                )
            return False

        return self._plan_and_execute(trial_id, objects, t_start)

    def _detect_objects(self, trial_id: int) -> tuple[list[dict[str, Any]], int]:
        """Consume perception messages until one holds a usable object, or the
        detection window ends. Returns (objects, number of messages received)."""
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

        return objects, received_frames

    def run_placement_check(self, trial_id: int, operator: Any) -> None:
        """Check the placement of the coming trial before its cycle, until it is as
        called or MAX_CHECKS checks have been made.

        03 calls this from its pre-trial hook once the operator has placed the part
        and pressed ENTER, with the servo still OFF, and runs the motion gate (servo
        ON) only after it returns; nothing here switches the servo. Each check is one
        detection window through this orchestrator's class lock (_detect_objects,
        current_expected_class set), judged by placement_check.check_placement on
        placement_check.judged_detection. A check without flags ends it, and so
        does a window that received no frame (nothing checked). Otherwise
        operator(trial_id, lines, last) shows the flag lines and returns once the
        operator has pressed ENTER: with last False the operator was asked to fix the
        placement, and it is checked again; with last True, at the MAX_CHECKS-th
        check, the operator was asked to make sure the part is placed as called, and
        the trial runs as placed.

        Why the arms stay comparable: the flags read only quantities that are
        identical in every depth mode, so every arm asks the operator the same; a
        check capture is never used for the trial; and the trial's detection is the
        fresh capture run_one_cycle takes after the motion gate, as in session 1
        (commit 5dcf934).

        Every check is logged for audit (placement_* columns, see logger.py): the
        record waits in _pending_placement and goes with the trial's row. It holds
        the flags of the first check, the checks after the first, whether the trial
        ran after the last check still flagged, every check as flags:outcome, the
        detection each check judged (class, top centre, top height, yaw, frame) with
        the operator's time on it, and the seconds from the first check to the
        return, just before the motion gate (not part of cycle_time_s). An exception
        from operator (Ctrl+C, a closed stdin) writes the trial as aborted_at_check,
        unscored, before it is raised again.
        """
        called = self.current_called_pose
        plane = self.config.get("table_plane")
        events: list[str] = []
        captures: list[dict[str, Any]] = []
        first: list[str] | None = None
        override = False
        t_check = time.time()

        def record() -> dict[str, Any]:
            return {
                "placement_flags": "+".join(first or []),
                "placement_recaptures": len(captures) - 1,
                "placement_override": int(override),
                "placement_events": ";".join(events),
                "placement_captures": json.dumps(captures, separators=(",", ":")),
                "placement_check_s": round(time.time() - t_check, 2),
            }

        while True:
            objects, received_frames = self._detect_objects(trial_id)
            judged = judged_detection(objects)
            flags = check_placement(
                called, judged, frames_seen=received_frames > 0,
                T_BC=self.T_BC, table_plane=plane,
                class_sizes=self.config.get("class_sizes_mm"))
            codes = [f.code for f in flags]
            if first is None:
                first = codes
            capture = _capture_record(len(captures) + 1, received_frames,
                                      self._last_frame_path, judged, codes,
                                      self.T_BC, plane)
            captures.append(capture)
            if received_frames == 0:
                capture["outcome"] = "skip"
                events.append("noframe:skip")
                logger.info("PLACEMENT CHECK trial %d (%s): no frame, nothing checked",
                            trial_id, getattr(called, "pose_id", ""))
                break
            if not flags:
                capture["outcome"] = "clear"
                events.append("none:clear")
                logger.info("PLACEMENT CHECK trial %d (%s): clear", trial_id,
                            getattr(called, "pose_id", ""))
                break
            last = len(captures) >= MAX_CHECKS
            label = "+".join(codes)
            logger.info("PLACEMENT CHECK trial %d (%s): %s", trial_id,
                        getattr(called, "pose_id", ""),
                        " / ".join(f.text for f in flags))
            t_ask = time.time()
            try:
                operator(trial_id, [f.text for f in flags], last)
            except BaseException:
                capture["outcome"] = "abort"
                capture["answer_s"] = round(time.time() - t_ask, 2)
                events.append(f"{label}:abort")
                self._placement = record()
                self.sm.reset()      # no cycle has begun: not the last trial's state
                logger.warning("PLACEMENT CHECK trial %d: stopped at the check; "
                               "logged as aborted_at_check", trial_id)
                try:
                    self._log_trial(trial_id, False, "aborted_at_check", t_check,
                                    judged)
                except Exception:  # noqa: BLE001 - the original exception matters
                    logger.exception("Could not log trial %d as aborted_at_check",
                                     trial_id)
                raise
            capture["answer_s"] = round(time.time() - t_ask, 2)
            if not last:
                capture["outcome"] = "fixed"
                events.append(f"{label}:fixed")
                logger.info("PLACEMENT CHECK trial %d: the operator was asked to fix "
                            "it; check %d", trial_id, len(captures) + 1)
                continue
            override = True
            capture["outcome"] = "limit"
            events.append(f"{label}:limit")
            logger.info("PLACEMENT CHECK trial %d: still flagged after %d checks; the "
                        "trial runs as placed", trial_id, len(captures))
            break
        self._pending_placement = record()

    def _plan_and_execute(
        self,
        trial_id: int,
        objects: list[dict[str, Any]],
        t_start: float,
    ) -> bool:
        """PLAN, the motion guards and the motion, for the detected objects."""
        # ─── PLAN: try each object until one is reachable ───
        self.sm.transition_to(PickState.PLAN)
        dz = self.config["approach_height_mm"]

        # Every height rule for the pick, and the cell evidence behind the
        # clearance, is in _commanded_pick_z.
        table_top = float(self.config["table_top_z_mm"])

        for obj in objects:
            xyz_base = np.array(obj["pose_base"], dtype=float).copy()
            # The commanded height is only real once it reaches xyz_base.
            # Everything after this line, the refusal clauses and the grasp
            # pose itself, reads it from there.
            xyz_base[2] = self._commanded_pick_z(obj, table_top)

            # MOTION SAFETY, one rule set for every class.
            #
            # Built only from what perception reports. Nothing here may read
            # the called position, yaw or stacking column: those are the
            # experiment's answer sheet, and a guard that consulted them would
            # refuse exactly the trials where perception was wrong, which is
            # the quantity E2 measures. The class is different. The trial names
            # the class to pick, _select_objects keeps only detections of that
            # class, and the paper declares it as the task input.
            #
            # Three refusals, judged in this order before any motion is
            # dispatched, each kept whole in its own method so that it can be
            # tested as itself:
            #   pick_z_safety_reject   the depth behind the command cannot be
            #                          believed (_depth_implausible);
            #   depth_geometry_reject  trustworthy depth puts the top face far
            #                          above the command
            #                          (_depth_contradicts_command);
            #   yaw_geometry_reject    the measured top rectangle is not the
            #                          detected class (_shape_contradicts_class).
            if self.config.get("is_real_mode", False):
                commanded_h = float(xyz_base[2]) - table_top
                for code, judge, args in (
                    ("pick_z_safety_reject", self._depth_implausible, (obj,)),
                    ("depth_geometry_reject", self._depth_contradicts_command,
                     (obj, commanded_h)),
                    ("yaw_geometry_reject", self._shape_contradicts_class,
                     (obj,)),
                ):
                    reason = judge(*args)
                    if reason is not None:
                        logger.error("Trial %d: pick refused (%s): %s",
                                     trial_id, code, reason,
                                     extra={"console": _REFUSED.format(trial_id)})
                        self.sm.fail(code)
                        self.stats["failed"] += 1
                        self._log_trial(trial_id, False, code, t_start, obj)
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
                    or "Unsafe/invalid target height" in msg
                    or "UNSAFE P2 rejected before JOB_START" in msg
                )

                if not is_measurement_reject:
                    raise

                logger.warning(
                    "Trial %d: physical depth/height geometry rejected in PLAN "
                    "(%s) -> trial FAIL, NO MOTION dispatched",
                    trial_id,
                    msg,
                    extra={"console": _REFUSED.format(trial_id)},
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
            # Keep X/Y/orientation exactly equal to P002. Raise only BASE-Z, by
            # the amount used between P0 -> P1 but never below P1, so the
            # transfer is level over a support. The backend builds the same
            # P003 with the same helper.
            place_approach_dz_base = float(
                p1_plan_base[2, 3] - p0_plan_base[2, 3]
            )

            if place_approach_dz_base <= 1e-6:
                raise RuntimeError(
                    "Invalid P3 approach: expected P1.Z > P0.Z in "
                    f"YRC BASE, got dz={place_approach_dz_base:.3f} mm"
                )

            from .frame_convert import place_approach_base
            p3_plan_base = place_approach_base(
                p0_plan_base, p1_plan_base, p2_plan_base)

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
        logger.info("Trial %d: all objects out of reach", trial_id,
                    extra={"console": _REFUSED.format(trial_id)})
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
                                 "aborting trial", trial_id,
                                 extra={"console": _REFUSED.format(trial_id)})
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
                               trial_id, reason,
                               extra={"console": _REFUSED.format(trial_id)})
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
            #
            # The listing shows the motion only. The taught job also closes and
            # opens the gripper and waits for the carrier-detect sensor, with the
            # PLC supervising actuation and sensing (confirmed on the cell, 14 Oct
            # 2026). The PC learns only whether the job finished without an
            # alarm, which is why the operator's verdict is the task outcome.
            if self.config.get("use_yrc_ik", False):
                motion_backend = getattr(self.robot, "backend", self.robot)
                if hasattr(motion_backend, "run_dtwinrun_motion"):
                    reason = self._check_at_home("before")
                    if reason is not None:
                        logger.error("Trial %d: cycle refused, %s; return the "
                                     "robot to HOME and continue",
                                     trial_id, reason,
                                     extra={"console": (
                                         f"Trial {trial_id}: robot not at HOME; return "
                                         "it to HOME and continue (details in the log)")})
                        self.stats["failed"] += 1
                        self.sm.fail("not_at_home")
                        return False
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
                    reason = self._check_at_home("after")
                    if reason is not None:
                        raise RuntimeError(f"cycle incomplete: {reason}")

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
            logger.error("Trial %d: pick failed — %s", trial_id, e,
                         extra={"console": f"Trial {trial_id}: the cycle stopped with "
                                           "an error (details in the log)"})
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

    def _check_at_home(self, when: str) -> str | None:
        """Why the robot is not at HOME, or None. Real cell only.

        Every fixed-job cycle starts and ends at the taught HOME (P004). A
        cycle that does not end there did not run to its last instruction,
        whatever the controller's RUNNING and alarm flags said: on 15 Oct 2026
        a cycle stopped by alarm 4746 half way down was otherwise logged as a
        success. A cycle must not start anywhere else either, since its first
        move goes straight to the approach point.

        HOME is taught_home_joints_deg from the config, read off the telemetry
        of 14 and 15 Oct 2026, where the robot rested within 0.006 deg of it
        in five of six runs (home_joints_deg is a simulation value). Without
        it, the pose before the first cycle is taken. That fallback is weak:
        each campaign block is its own process, and the third run of 14 Oct
        started away from HOME, 43.6 deg off at J6.
        """
        if not self.config.get("is_real_mode", False):
            return None
        try:
            q = [float(v) for v in self._joints_to_list(self.robot.Joints())[:6]]
        except Exception as exc:  # noqa: BLE001
            return f"joints unreadable {when} the cycle: {exc}"
        if len(q) < 6:
            return f"joints unreadable {when} the cycle: {q}"
        home = (self.config.get("taught_home_joints_deg")
                or getattr(self, "_session_home_deg", None))
        if home is None:
            if when == "before":
                self._session_home_deg = q
                logger.info("Session HOME taken as %s", [round(v, 2) for v in q])
            return None
        tol = float(self.config.get("home_tolerance_deg", 0.5))
        dev = max(abs(a - b) for a, b in zip(q, home))
        if dev > tol:
            return (f"robot {dev:.2f} deg from HOME {when} the cycle "
                    f"(tolerance {tol:.2f})")
        return None

    def _return_home(self) -> None:
        """Move robot to home — normalize joints to avoid spin artifacts.

        Prefers `config['home_joints_deg']` (supplied from cell layout) over
        `robot.JointsHome()` — because JointsHome() reads from the .robot library
        file (Yaskawa GP7 default = [0,0,0,0,0,0]), NOT the home pose the user set
        in cell_layout.yaml. Using the wrong source → MoveJ to 0 while current is
        at -180 → 180° rotation on J4/J6 = spin artifact.

        Not on the real cell: the fixed job ends at the taught HOME by itself,
        and after a fault the robot is left where it stopped. home_joints_deg
        there is a simulation pose far from the taught HOME, and moving on its
        own after an unexplained stop is the wrong reflex.
        """
        if self.config.get("is_real_mode", False):
            logger.warning("Real cell: robot left where it stopped; the "
                           "operator returns it to HOME", extra={"console": True})
            return
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
            for key, digits in (("yaw_rect_long_mm", 1), ("yaw_rect_short_mm", 1)):
                if obj.get(key) is not None:
                    extra[key] = round(float(obj[key]), digits)
            if obj.get("yaw_near_square") is not None:
                extra["yaw_near_square"] = int(bool(obj["yaw_near_square"]))
            if obj.get("h_depth_mm") is not None:
                extra["h_depth_mm"] = round(float(obj["h_depth_mm"]), 1)
            if obj.get("h_median_mm") is not None:
                extra["h_median_mm"] = round(float(obj["h_median_mm"]), 1)
            if obj.get("table_share") is not None:
                extra["table_share"] = round(float(obj["table_share"]), 3)
            if obj.get("part_mask_area") is not None:
                extra["part_mask_area"] = int(obj["part_mask_area"])
            if obj.get("fit_iou") is not None:
                extra["fit_iou"] = round(float(obj["fit_iou"]), 3)
            if obj.get("fit_plateau_mm") is not None:
                extra["fit_plateau_mm"] = round(float(obj["fit_plateau_mm"]), 1)
            # Centre of the top face, the same in every depth mode; the placement
            # check compares it with the card (placement_check.top_centre_xy).
            centre = top_centre_xy(obj, getattr(self, "T_BC", None),
                                   (getattr(self, "config", None) or {}).get("table_plane"))
            if centre is not None:
                extra["top_x_mm"] = round(centre[0], 1)
                extra["top_y_mm"] = round(centre[1], 1)
        called = getattr(self, "current_called_pose", None)
        if called is not None:
            extra["carton_mm"] = getattr(called, "carton_mm", None) or ""
            extra["stack_on_carton_mm"] = getattr(called, "stack_on_carton_mm", None) or ""
        extra.update(getattr(self, "_placement", None) or {})
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
