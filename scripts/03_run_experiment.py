#!/usr/bin/env python
"""
03_run_experiment.py
────────────────────
Entry point for pick-and-place experiment (see thesis section 10).

Starts perception + viewport (sim) or HSE backend (real), runs N trials
through Orchestrator, writes results to results/.

Modes:
    --mode sim   : MockCamera + MockDetector + SimRobot + Open3D viewport → L4 test
                   logic pipeline without a real D455/model.
    --mode real  : D455 + YOLO + HSE backend → real GP7, telemetry CSV @10Hz.

Usage:
    python scripts/03_run_experiment.py --mode sim --trials 50
    python scripts/03_run_experiment.py --mode real --trials 50 --lighting bright
"""
from __future__ import annotations

import argparse
import logging
import os
import queue
import shutil
import sys
import threading
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cell import CellConfig  # noqa: E402
from src.logging import TrialLogger  # noqa: E402
from src.orchestrator import Orchestrator  # noqa: E402
from src.perception import PerceptionNode  # noqa: E402
from src.utils import load_yaml, setup_logging, timestamp  # noqa: E402


def _load_pose_list(path: Path) -> list[dict[str, str]]:
    """Load a pre-drawn pose-list CSV (from scripts/22_make_pose_lists.py).

    Required column: pose_id. Optional: card_id, x_mm, y_mm, yaw_deg,
    class_hint, condition. Row order = trial order.
    """
    import csv

    if not path.exists():
        raise FileNotFoundError(f"Pose list not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "pose_id" not in rows[0]:
        raise ValueError(f"Pose list {path} must have a 'pose_id' column")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument(
        "--cell-config", default=None,
        help="Cell layout YAML. Default: cell_layout.yaml for sim, "
             "cell_layout_real.yaml for real (auto-picked by --mode).",
    )
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--mode", choices=["sim", "real"], default="sim")
    parser.add_argument(
        "--backend", choices=["hse", "sim"], default=None,
        help="Robot motion backend. Default: 'sim' when --headless or "
             "--mode sim, 'hse' when --mode real.",
    )
    # NOTE: --viewport removed. Sim non-headless ALWAYS uses O3DGuiSimRobot
    # (Filament). Real mode uses O3DGuiSimRobot as mirror viewport-only. Headless
    # = no viewport. Requires `pip install open3d` for both non-headless modes.
    parser.add_argument(
        "--hse-ip", default=None,
        help="YRC1000 IP for --backend hse. Default: read from robot_connection.ip "
             "in cell config.",
    )
    parser.add_argument(
        "--mirror-hz", type=float, default=2.0,
        help="Viewport callback rate (Hz). Default 2.0 — smooth enough visually "
             "while reducing overhead. Real-mode viewport not yet wired (telemetry-only).",
    )
    parser.add_argument(
        "--telemetry-hz", type=float, default=10.0,
        help="Backend joints poll + CSV log rate (Hz). Default 10.0 — sufficient "
             "resolution for velocity / cycle-time analysis. Decoupled from "
             "mirror-hz: viewport is automatically throttled to mirror-hz.",
    )
    parser.add_argument(
        "--no-viewport-mirror", action="store_true",
        help="Disable viewport callback in mirror thread. Telemetry CSV is still "
             "fully recorded; visualize offline via 05_analyze_telemetry.py.",
    )
    parser.add_argument(
        "--ultra-fast", action="store_true",
        help="HSE backend ultra-fast mode: upload INFORM template only once, "
             "each trial does only WRITE_POS_VAR + JOB_START via HSE (0 FTP overhead). "
             "~50ms/trial instead of ~200ms (M3 batch). Requires constant trial structure "
             "(e.g. same pick-and-place pattern).",
    )
    parser.add_argument(
        "--ik-source", choices=["yrc", "client"], default=None,
        help="IK source: yrc (YRC1000 controller-side IK — recommended for HSE real, 0 DH "
             "dependency on PC), client (numerical DLS pure-Python URDF chain, "
             "matches RoboDK 0.00mm). Default 'yrc' for --mode real, 'client' for sim.",
    )
    parser.add_argument(
        "--tool-no", type=int, default=1,
        help="TOOL coordinate number on YRC1000 (TOOL01 default). Must match "
             "TCP gripper configured on the teach pendant. See docs/HUONG_DAN_CAI_DAT.md §2.10.",
    )
    parser.add_argument("--lighting", default="", help="Lighting condition label (log)")
    parser.add_argument("--overlap", default="", help="Overlap level label (log)")
    parser.add_argument(
        "--pose-list", default=None,
        help="Pre-drawn pose-list CSV (from 22_make_pose_lists.py). Prints the "
             "card placement instruction before each trial and logs pose_id → "
             "outcomes pair pose-by-pose across configurations (McNemar, §3.7). "
             "Reuse the SAME list for every configuration under comparison.")
    parser.add_argument(
        "--depth-mode", choices=["rgbd", "plane", "fusion"], default="rgbd",
        help="C4 depth mode. rgbd: median depth under the mask (default). plane: the "
             "ray through the mask centroid meets the measured table plane lifted by "
             "the part height, no depth read. fusion: the plane estimate, replaced by "
             "the median depth when enough of the mask returns depth and disagrees "
             "with it. plane and fusion need --mode real and table_plane.json.")
    parser.add_argument(
        "--confirm-each-trial", action="store_true",
        help="Wait for the operator to press ENTER before every cycle, and drop any "
             "frames the perception node captured while they were still reaching over "
             "the table. Use this for every real run where a person places the part by "
             "hand: without it the robot starts the next cycle 1 s after the last one "
             "and detects whatever was on the table before the part was set down. Leave "
             "it off for the cycle-time run, where the operator's pace is not part of "
             "what is being measured. It also asks the operator, after every cycle that "
             "moved the robot, whether the part ended up in the right place, and writes "
             "the answer to the human_ok column — the machine's own success column "
             "cannot see a part dropped in transfer.")
    parser.add_argument(
        "--save-frames", action="store_true",
        help="Save the RGB frame of every detection and log its path per trial "
             "(failure context for the C3 loop). NOTE: the perception node runs "
             "free, so this writes one image per PROCESSED FRAME, not one per "
             "trial — a campaign produces thousands. Point --frames-dir at a disk "
             "with room; when it fills, imwrite fails, frame_path goes empty and "
             "the run carries on regardless.")
    parser.add_argument(
        "--blind-label", default=None, metavar="CODE",
        help="Run blinded: the console shows only CODE and the placement prompts, and "
             "every line naming the depth mode, the model or the outcome goes to the log "
             "file instead. The person placing the parts is also the person scoring the "
             "place tolerance by eye, so if they can see which configuration is running "
             "they are an unblinded assessor of the primary endpoint. Use "
             "tools/run_blinded_campaign.py, which allocates the codes and seals the key.")
    parser.add_argument(
        "--pose-slice", default=None, metavar="A:B",
        help="Run rows A..B-1 of the pose list (0-based, Python slice) instead of the "
             "first --trials rows. This is what makes interleaving possible: run block "
             "0:25 of configuration A, then 0:25 of B, then 25:50 of A, and so on. "
             "Without it every configuration is one long uninterrupted stretch, and any "
             "drift across the afternoon lands in the paired test looking exactly like a "
             "configuration effect.")
    parser.add_argument(
        "--session-id", default="",
        help="Free-text tag for this sitting, e.g. 2026-09-20-morning. Written to every "
             "row so a configuration effect can be told apart from an across-the-day "
             "drift effect; without it the two are indistinguishable after the fact.")
    parser.add_argument(
        "--block-id", default="",
        help="Free-text tag for this block within the sitting. Use it when configurations "
             "are interleaved in short blocks rather than run as one long stretch each.")
    parser.add_argument(
        "--operator-id", default="",
        help="Who placed the parts. Two people place differently, and that difference "
             "lands in the disagreement cells of the paired test.")
    parser.add_argument(
        "--frames-dir", default="results/frames",
        help="Where --save-frames writes. A timestamped subfolder is created under "
             "it. Relative paths resolve against the repo. Default results/frames, "
             "which is on the same disk as the code.")
    parser.add_argument(
        "--headless", action="store_true",
        help="Run WITHOUT viewport (SimRobot mock) — validate logic and generate CSV. "
             "Intended for CI/CD and large batch runs.",
    )
    parser.add_argument(
        "--minimal-build", action="store_true",
        help="Minimal Open3D viewport (omit floor, calib frame, secondary objects) — "
             "improves render FPS.",
    )
    # ─── Headless scenario tuning (applies only when --headless) ───
    parser.add_argument(
        "--grasp-fail-rate", type=float, default=0.0,
        help="(headless) Grasp slip probability [0..1] → failure mode 'grasp_slip'.",
    )
    parser.add_argument(
        "--detection-miss-rate", type=float, default=0.0,
        help="(headless) Object detection miss probability [0..1] → 'detection_miss'.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="(headless) RNG seed for reproducible results.",
    )
    return parser.parse_args()


DEFAULT_OBJECT_HEIGHTS_MM = {"tray": 25, "bottle": 150, "cup": 40, "bolt": 25}
DEFAULT_MASK_SIZE_PX = {"tray": (180, 100), "bottle": (120, 40), "cup": (80, 80), "bolt": (40, 40)}


def _stl_height_mm(stl_path):
    """Read binary STL → height along Z axis (max Z - min Z). Returns None on error.

    Used to auto-detect real object dimensions instead of hard-coding, so depth
    and grasp pose align correctly with the actual mesh file.
    """
    import struct
    try:
        with open(stl_path, "rb") as f:
            f.read(80)
            ntri = struct.unpack("<I", f.read(4))[0]
            z_min, z_max = float("inf"), float("-inf")
            for _ in range(ntri):
                f.read(12)
                for _ in range(3):
                    _, _, z = struct.unpack("<fff", f.read(12))
                    if z < z_min:
                        z_min = z
                    if z > z_max:
                        z_max = z
                f.read(2)
            return z_max - z_min if z_max > z_min else None
    except Exception:
        return None


def _auto_mock_detection_params(cell_config, intrinsics_dict, object_name="tray"):
    """Auto-compute (mask_box, depth_m, height_mm) from object pose in cell_layout.

    Ensures mock detection reflects the correct object template position in RoboDK
    rather than hard-coding a pixel center. Projects object_top from world space to
    pixel via camera intrinsics + camera world pose. Falls back to legacy values if
    config is missing.
    """
    import numpy as np

    from src.cell.pose_utils import make_homogeneous

    obj = next((o for o in cell_config.objects if o.name == object_name), None)
    if obj is None:
        return (580, 350, 700, 390), 0.49, 150

    # World pose of object base = parent_frame.pose + object.pose offset
    if obj.parent_frame:
        parent = next((f for f in cell_config.frames if f.name == obj.parent_frame), None)
        parent_xyz = np.array(parent.pose.xyz_mm, dtype=float) if parent else np.zeros(3)
    else:
        parent_xyz = np.zeros(3)
    offset_xyz = np.array(obj.pose.xyz_mm, dtype=float) if obj.pose else np.zeros(3)
    obj_base_world = parent_xyz + offset_xyz

    # Object top = base + actual height. Prefer measuring directly from STL file
    # to match the real mesh and avoid hard-coding a potentially wrong height.
    height_mm = DEFAULT_OBJECT_HEIGHTS_MM.get(object_name, 100)
    if obj.mesh:
        mesh_path = Path(obj.mesh)
        if not mesh_path.is_absolute():
            mesh_path = PROJECT_ROOT / mesh_path
        if mesh_path.exists():
            actual = _stl_height_mm(mesh_path)
            if actual is not None and actual > 1.0:
                height_mm = actual
    obj_top_world = obj_base_world.copy()
    obj_top_world[2] += height_mm

    # Camera world pose → T_BC; invert to transform world → camera frame
    cam_T = make_homogeneous(cell_config.camera.pose.xyz_mm, cell_config.camera.pose.rpy_deg)
    p_world_h = np.array([*obj_top_world, 1.0])
    p_cam_h = np.linalg.inv(cam_T) @ p_world_h
    x_cam, y_cam, z_cam = p_cam_h[:3]

    if z_cam <= 10:
        return (580, 350, 700, 390), 0.49, height_mm

    fx, fy = intrinsics_dict["fx"], intrinsics_dict["fy"]
    ppx, ppy = intrinsics_dict["ppx"], intrinsics_dict["ppy"]
    u = ppx + fx * x_cam / z_cam
    v = ppy + fy * y_cam / z_cam

    W, H = DEFAULT_MASK_SIZE_PX.get(object_name, (60, 60))
    mask_box = (int(u - W / 2), int(v - H / 2), int(u + W / 2), int(v + H / 2))
    depth_m = float(z_cam) / 1000.0
    return mask_box, depth_m, height_mm


def build_perception(mode: str, config: dict, args=None, cell_config=None):
    """Create (camera, detector) for the given run mode.

    Args:
        cell_config: If provided, mock detection is auto-generated from the object
            pose in the cell (mask_box + depth matching the real object) instead of
            hard-coding pixel values.
    """
    if mode == "real":
        from src.perception import D455Camera, ObjectDetector

        camera = D455Camera()
        model_path = Path(config.get("model_path", "models/yolov8s-seg_best.pt"))
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path      # independent of the working dir
        heights = config.get("class_heights_mm") or {}
        detector = ObjectDetector(
            model_path=str(model_path),
            conf=config.get("conf_threshold", 0.5),
            class_heights_mm=heights,
        )
        missing = [c for c in detector.class_names if c not in heights]
        if missing:
            logging.getLogger("experiment").warning(
                "No height configured for classes %s: their grasp depth falls back to "
                "grasp_depth_offset_mm and only the table floor protects them.", missing)
        return camera, detector

    # mode == "sim" (both RoboDK sim and headless): simulated detection scenario.
    import numpy as np

    from src.perception import MockCamera, MockDetector

    headless = bool(getattr(args, "headless", False))

    # Auto-compute mask_box + depth from cell_config (mock detection matches real
    # object template in RoboDK). Target = first object in cell (= tray default).
    auto_height_mm = None
    target_name = "tray"
    if cell_config is not None and not headless:
        if cell_config.objects:
            target_name = cell_config.objects[0].name
        mask_box, sim_depth_m, auto_height_mm = _auto_mock_detection_params(
            cell_config, MockCamera.DEFAULT_INTRINSICS, target_name,
        )
    else:
        sim_depth_m = config.get("sim_depth_m", 0.49)
        mask_box = (580, 350, 700, 390)

    h, w = 720, 1280
    camera = MockCamera(
        depth_frames=[np.full((h, w), sim_depth_m, np.float32)],
    )

    if not headless:
        # RoboDK sim: single fixed scenario, mask matches real object in cell.
        det = MockDetector.make_detection(target_name, mask_box=mask_box)
        if auto_height_mm is not None:
            det.height_mm = auto_height_mm   # pass real height for adaptive grasp
        scripted = [[det]]
    else:
        # Headless: generate diverse class + position scenarios for meaningful statistics.
        # miss_rate: fraction of trials with no object (tests recovery behavior).
        import random

        n = getattr(args, "trials", 50)
        miss_rate = getattr(args, "detection_miss_rate", 0.0)
        rng = random.Random(getattr(args, "seed", 42))
        classes = ["tray", "bottle", "cup", "bolt"]
        scripted = []
        for _ in range(n):
            if rng.random() < miss_rate:
                scripted.append([])                       # detection miss
                continue
            cls = rng.choice(classes)
            cu = rng.randint(450, 1050)                   # pixel u → world X span
            cv = rng.randint(330, 410)
            det_h = MockDetector.make_detection(
                cls, mask_box=(cu - 60, cv - 20, cu + 60, cv + 20),
            )
            det_h.height_mm = DEFAULT_OBJECT_HEIGHTS_MM.get(cls, 100)
            scripted.append([det_h])
    detector = MockDetector(scripted=scripted)
    return camera, detector


def build_extractor(intrinsics, detector, config: dict, depth_mode: str, table_plane,
                    calib_path):
    """PoseExtractor for the physical cell, or None (logged) when the depth mode needs a
    height that some class the detector can report does not have."""
    from src.orchestrator.coord_conv import load_calibration
    from src.perception import PoseExtractor

    log = logging.getLogger("experiment")
    heights = config.get("class_heights_mm") or {}
    if depth_mode != "rgbd":
        missing = [c for c in detector.class_names if c not in heights]
        if missing:
            log.error("PREFLIGHT: depth mode %s places each part's top face at its height, "
                      "but classes %s have none in class_heights_mm. Run refused before "
                      "connecting to the robot.", depth_mode, missing)
            return None
    T_BC = load_calibration(calib_path) if table_plane is not None else None
    return PoseExtractor(intrinsics, depth_mode=depth_mode, T_BC_mm=T_BC,
                         table_plane=table_plane, class_heights_mm=heights,
                         class_sizes_mm=config.get("class_sizes_mm"))


def main() -> int:
    args = parse_args()
    log = setup_logging("experiment", log_file=PROJECT_ROOT / "logs/experiment.log")
    log.info("=" * 60)
    log.info("Pick-and-place experiment — mode=%s, trials=%d", args.mode, args.trials)
    log.info("=" * 60)

    if args.blind_label:
        # Console keeps warnings and errors (safety must never be hidden); everything
        # routine drops to the log file, which nobody reads mid-session.
        import logging as _logging
        for h in _logging.getLogger().handlers:
            if isinstance(h, _logging.StreamHandler) and not isinstance(
                    h, _logging.FileHandler):
                h.setLevel(_logging.WARNING)
        print("=" * 60)
        print(f"  BLINDED RUN — arm {args.blind_label}")
        print("  Place the parts as prompted. Which configuration this is, and how it is")
        print("  doing, are deliberately not shown. Both are in the log file.")
        print("=" * 60, flush=True)

    if args.confirm_each_trial:
        if args.headless:
            log.error("--confirm-each-trial needs an operator; --headless pre-fills the "
                      "detection queue and would hang. Drop one of the two flags.")
            return 2
        if not sys.stdin.isatty():
            log.error("--confirm-each-trial reads ENTER from the terminal, but stdin is "
                      "not a terminal here. Run it in a real terminal, otherwise the gate "
                      "would silently never wait.")
            return 2

    # Auto-pick cell config by --mode (if user did not override). Avoids the
    # situation where --mode real accidentally uses cell_layout.yaml (robot_connection
    # disabled → commands never reach the real GP7).
    if args.cell_config is None:
        args.cell_config = (
            "config/cell_layout_real.yaml" if args.mode == "real"
            else "config/cell_layout.yaml"
        )
        log.info("Auto-pick cell-config: %s (theo --mode=%s)", args.cell_config, args.mode)

    config = load_yaml(PROJECT_ROOT / args.config)
    # Settings for the physical cell and the paper's five parts live under "real:"
    # and override the top-level (simulation) keys only in real mode, so the
    # simulated tray/bottle/cup/bolt scenario keeps its behaviour.
    real_overrides = config.pop("real", None) or {}
    if args.mode == "real":
        config.update(real_overrides)

    # Real mode: enable C2 safety layer (reach envelope + predictive safety over full trajectory).
    # Overrides the sim-friendly defaults in orchestrator _DEFAULT_CONFIG.
    if args.mode == "real":
        config["is_real_mode"] = True
        config["skip_reachability_check"] = False
        config["predictive_safety_enabled"] = True  # UC2: joint limit + self-collision over full trajectory
        log.info("Real mode: reach envelope + predictive safety C2 ENABLED "
                 "(joint limit + self-collision check over full trajectory before MoveJ)")

        # Real mode: enable CC-Link gripper path (double-acting + feedback sensors)
        # Default mapping — TODO verify from YRC1000 TP: Setup → I/O Module → CC-Link
        # Can be overridden via experiment.yaml (key "gripper_cc_link").
        if "gripper_cc_link" not in config or config["gripper_cc_link"] is None:
            config["gripper_cc_link"] = {
                "clamp_bit":            30010,   # → PLC Y502 (Clamp solenoid)
                "unclamp_bit":          30011,   # → PLC Y503 (UnClamp solenoid)
                "clamp_sensor_bit":     30050,   # ← PLC X504 (cylinder at Clamp)
                "unclamp_sensor_bit":   30051,   # ← PLC X503 (cylinder at UnClamp)
                "detect_bit":           30052,   # ← PLC X505 (Carrier Detect)
                "wait_sensor_timeout_s": 2.0,
                "wait_sensor_poll_s":    0.05,
                "require_detect_on_close": True,
            }
            log.info("Gripper CC-Link path ENABLED (default mapping — verify TP)")

    # Load cell_config early: needed for build_perception auto-mock detection
    # from object pose (used in all modes — including --no-build).
    cell_config = CellConfig.from_yaml(PROJECT_ROOT / args.cell_config)

    # Pass home_joints from cell config to Orchestrator so _return_home uses
    # the correct home defined in the cell (not JointsHome() from the .robot file).
    config["home_joints_deg"] = list(cell_config.robot.home_joints_deg)

    # ─── Preflight (real mode), BEFORE any connection to the robot ───
    table_plane = None
    calib_path = PROJECT_ROOT / config["calibration_path"]
    if args.mode == "real":
        from src.orchestrator.preflight import check_depth_mode, check_real_mode

        problems, table_z = check_real_mode(
            config, calib_path,
            cell_config.robot.pose.xyz_mm, cell_config.robot.pose.rpy_deg)
        dm_problems, table_plane = check_depth_mode(config, calib_path, args.depth_mode)
        problems += dm_problems
        if problems:
            for p in problems:
                log.error("PREFLIGHT: %s", p)
            log.error("Real-mode run refused before connecting to the robot (%d problem%s).",
                      len(problems), "" if len(problems) == 1 else "s")
            return 5
        config["table_top_z_mm"] = table_z
        log.info("PREFLIGHT ok: calibrated T_BC, base pose unchanged, table top %.1f mm, "
                 "jaw-tip floor %.1f mm, heights %s", table_z,
                 table_z + float(config.get("table_safety_margin_mm", 100.0)),
                 config.get("class_heights_mm"))
        if table_plane is not None:
            log.info("Depth mode %s; table plane z = %.5f x + %.5f y + %.1f mm (tilt %.2f deg)",
                     args.depth_mode, table_plane["a"], table_plane["b"], table_plane["c"],
                     float(table_plane.get("tilt_deg", float("nan"))))
        else:
            log.warning("Depth mode rgbd without a usable table plane: carton sizes are not "
                        "resolved per instance, and the configured carton height is used.")
    elif args.depth_mode != "rgbd":
        log.error("--depth-mode %s needs the calibrated cell (--mode real): it intersects "
                  "viewing rays with the measured table plane.", args.depth_mode)
        return 2

    # ─── Perception: camera, detector and pose extractor ───
    # Built BEFORE the robot connection, so a missing model, a camera fault or a part
    # without a height stops the run without touching the controller.
    # Pass cell_config so mock detection auto-matches the real object pose.
    camera, detector = build_perception(args.mode, config, args, cell_config)
    extractor = None
    if args.mode == "real":
        extractor = build_extractor(camera.intrinsics, detector, config, args.depth_mode,
                                    table_plane, calib_path)
        if extractor is None:
            camera.stop()
            return 5

    # ─── Resolve backend ───
    # Motion backend:
    #   --headless    → sim  (SimRobot bare-metal, no viewport)
    #   --mode sim    → sim  (SimRobot + Open3D viewport, drop-in robot=)
    #   --mode real   → hse  (MotomanHSEBackend UDP → YRC1000, telemetry-only)
    if args.backend is None:
        args.backend = "hse" if args.mode == "real" else "sim"
    if args.mode == "real" and args.backend != "hse":
        log.error("--mode real only accepts --backend hse.")
        return 4
    log.info("Backend: %s", args.backend)

    sim_robot: Any = None             # Orchestrator uses as `robot=`
    twin: Any = None                  # DigitalTwinMirror (None when headless/sim Open3D)
    real_viewer: Any = None           # O3DGuiSimRobot as viewport-only mirror for real mode

    # ─── Common: client IK config (URDF chain verified match RoboDK 0.00mm) ───
    config["robot_base_xyz_mm"] = tuple(cell_config.robot.pose.xyz_mm)
    config["robot_base_rpy_deg"] = tuple(cell_config.robot.pose.rpy_deg)
    tool_offset = 0.0
    if hasattr(cell_config, "tool") and cell_config.tool:
        tcp = getattr(cell_config.tool, "tcp_offset_mm", None)
        if tcp:
            tool_offset = float(tcp[2])
    config["robot_tool_offset_mm"] = tool_offset

    if args.headless:
        # ─── Headless: SimRobot bare-metal ───────────────────────────────
        from src.orchestrator.sim_robot import SimRobot

        sim_robot = SimRobot(
            home_joints=cell_config.robot.home_joints_deg,
            base_xyz=tuple(cell_config.robot.pose.xyz_mm),
            grasp_fail_rate=args.grasp_fail_rate,
            seed=args.seed,
        )
        config["use_client_ik"] = True
        config["skip_reachability_check"] = False
        config["inter_trial_delay_s"] = 0.0
        config["gripper_delay_s"] = 0.0
        log.info("HEADLESS mode — SimRobot (base=%s, grasp_fail=%.0f%%, miss=%.0f%%).",
                 list(cell_config.robot.pose.xyz_mm),
                 args.grasp_fail_rate * 100, args.detection_miss_rate * 100)
    elif args.backend == "sim":
        # ─── Sim non-headless + O3DGuiSimRobot (Filament) ──────────────
        # SimRobot + Open3D GUI; GUI runs on main thread, experiment runs on worker.
        from src.orchestrator.viewports.open3d_gui_sim_robot import O3DGuiSimRobot
        sim_robot = O3DGuiSimRobot(
            base_xyz=tuple(cell_config.robot.pose.xyz_mm),
            home_joints=cell_config.robot.home_joints_deg,
            cell_config=cell_config,
            project_root=PROJECT_ROOT,
            minimal_build=args.minimal_build,
            grasp_fail_rate=args.grasp_fail_rate,
            seed=args.seed,
        )
        config["use_client_ik"] = True
        config["skip_reachability_check"] = False   # SimRobot has reach envelope
        config["inter_trial_delay_s"] = 0.0
        log.info("Viewport: O3DGuiSimRobot (SimRobot motion + client DLS IK).")
    else:                                           # args.backend == "hse"
        # ─── Real mode: HSE backend + DigitalTwinMirror (telemetry-only) ──
        from src.orchestrator.digital_twin import DigitalTwinMirror
        from src.orchestrator.backends.motoman_hse import MotomanHSEBackend
        from src.orchestrator.telemetry import TelemetryLogger

        hse_ip = args.hse_ip or cell_config.robot_connection.ip
        if not hse_ip:
            log.error("--backend hse requires YRC1000 IP (via --hse-ip or cell config)")
            return 2
        backend = MotomanHSEBackend(
            ip=hse_ip,
            max_speed_pct=cell_config.robot_connection.max_speed_percent,
            tool_no=args.tool_no,
        )
        backend.set_home_joints(list(cell_config.robot.home_joints_deg))
        backend.connect()
        if not backend.Valid():
            log.error("HSE heartbeat failed — check ping %s + HSE Server function", hse_ip)
            backend.disconnect()
            return 3
        log.info("HSE backend connected: %s. Joints (sanity): %s",
                 hse_ip, backend.Joints())
        if args.ultra_fast:
            backend.enable_ultra_fast(True)
            log.info("HSE Ultra-fast P-var mode: ON (template uploaded once)")

        # IK source: yrc (controller-side, recommended) or client (DLS PC-side)
        ik_source = args.ik_source or "yrc"
        if ik_source == "yrc":
            config["use_yrc_ik"] = True
            log.info("IK source: YRC1000 controller (0 PC IK overhead). "
                     "TOOL%02d must be configured on the TP.", args.tool_no)
        else:
            config["use_client_ik"] = True
            log.info("IK source: client-side DLS (URDF chain match RoboDK 0.00mm).")

        telemetry = TelemetryLogger(
            PROJECT_ROOT / f"results/telemetry_{timestamp()}.csv"
        )

        # ─── Live Open3D mirror for real mode ──────────────────────────────
        # O3DGuiSimRobot is RENDER-ONLY (not used as robot=). HSE backend
        # polls Joints @telemetry_hz; mirror thread calls viewport_callback @mirror_hz;
        # callback posts transforms to GUI thread (thread-safe via post_to_main_thread).
        # Main thread blocks in real_viewer.run_gui() until the user closes the window.
        viewport_cb = None
        if not args.no_viewport_mirror:
            try:
                from src.orchestrator.viewports.open3d_gui_sim_robot import O3DGuiSimRobot
                real_viewer = O3DGuiSimRobot(
                    base_xyz=tuple(cell_config.robot.pose.xyz_mm),
                    home_joints=cell_config.robot.home_joints_deg,
                    cell_config=cell_config,
                    project_root=PROJECT_ROOT,
                    minimal_build=args.minimal_build,
                )
                viewport_cb = real_viewer.mirror_state
                log.info("Real-mode Open3D mirror ready (Filament GUI).")
            except Exception as e:                              # noqa: BLE001
                log.warning("Could not create Open3D mirror for real mode: %s. "
                            "Falling back to telemetry-only (replay via 07_replay_telemetry.py).",
                            e)
                viewport_cb = None

        twin = DigitalTwinMirror(
            backend=backend,
            viewport_callback=viewport_cb,
            telemetry=telemetry,
            mirror_hz=args.mirror_hz,
            telemetry_hz=args.telemetry_hz,
            drift_warn_deg=2.0,
            viewport_mirror_enabled=not args.no_viewport_mirror,
        )
        twin.start_mirror()
        sim_robot = twin                # Orchestrator receives the facade via `robot=`
        log.info(
            "DigitalTwinMirror active (HSE) — telemetry %.1fHz, mirror %.1fHz%s",
            args.telemetry_hz, args.mirror_hz,
            " (live Open3D viewport ON)" if viewport_cb else " (viewport OFF — telemetry-only)",
        )

    # ─── Perception node (camera, detector and extractor were built above) ───
    # Headless: queue large enough to pre-fill 1 scenario/trial (deterministic).
    qsize = args.trials + 1 if args.headless else 3
    det_queue: queue.Queue = queue.Queue(maxsize=qsize)
    ts = timestamp()
    if args.save_frames:
        base = Path(args.frames_dir)
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        frames_dir = str(base / ts)
        free_gb = shutil.disk_usage(base.parent if base.exists() else base.anchor).free / 2**30
        log.info("Frames → %s (%.1f GB free on that disk)", frames_dir, free_gb)
        if free_gb < 20.0:
            log.warning("Under 20 GB free where frames are written. A campaign writes "
                        "one image per processed frame; when the disk fills, saving "
                        "fails silently and the failure context for C3 is lost.")
    else:
        frames_dir = None
    perception = PerceptionNode(camera, detector, det_queue,
                                save_frames_dir=frames_dir, extractor=extractor)

    # ─── Logger ───
    label = "headless" if args.headless else args.mode
    trial_logger = TrialLogger(
        PROJECT_ROOT / f"results/experiment_{label}_{ts}.csv",
        extra_context={"lighting": args.lighting, "overlap": args.overlap,
                       "mode": label, "depth_mode": args.depth_mode},
    )

    # ─── Orchestrator ───
    orch = Orchestrator(det_queue, config=config, robot=sim_robot,
                        logger_obj=trial_logger)

    # ─── Pose list (paired McNemar design, paper §3.7) ───
    pose_rows = None
    if args.pose_list:
        pl_path = Path(args.pose_list)
        if not pl_path.is_absolute():
            pl_path = PROJECT_ROOT / args.pose_list
        pose_rows = _load_pose_list(pl_path)
        if args.pose_slice:
            try:
                lo_s, hi_s = args.pose_slice.split(":")
                lo = int(lo_s) if lo_s else 0
                hi = int(hi_s) if hi_s else len(pose_rows)
            except ValueError:
                log.error("--pose-slice must look like 25:50 (0-based, end exclusive); "
                          "got %r", args.pose_slice)
                return 2
            if not 0 <= lo < hi <= len(pose_rows):
                log.error("--pose-slice %s is outside the pose list of %d rows",
                          args.pose_slice, len(pose_rows))
                return 2
            pose_rows = pose_rows[lo:hi]
            args.trials = len(pose_rows)
            log.info("Pose slice %s → %d trial(s), pose_id %s..%s", args.pose_slice,
                     len(pose_rows), pose_rows[0].get("pose_id", "?"),
                     pose_rows[-1].get("pose_id", "?"))
        if len(pose_rows) < args.trials:
            log.warning("Pose list has %d entries < --trials %d → running %d trials",
                        len(pose_rows), args.trials, len(pose_rows))
            args.trials = len(pose_rows)

    orch.session_id = args.session_id
    orch.block_id = args.block_id
    orch.operator_id = args.operator_id

    pre_trial_hook = None
    if pose_rows is not None or args.confirm_each_trial:

        def pre_trial_hook(i: int) -> None:
            if pose_rows is not None:
                row = pose_rows[i - 1]
                orch.current_pose_id = str(row.get("pose_id", i))
                orch.current_condition = str(row.get("condition", ""))
                orch.current_stack_on = str(row.get("stack_on", ""))
                log.info(
                    "▶ Trial %d — PLACE OBJECT: card=%s  x=%s mm  y=%s mm  yaw=%s deg%s%s",
                    i, row.get("card_id", "?"), row.get("x_mm", "?"),
                    row.get("y_mm", "?"), row.get("yaw_deg", "?"),
                    f"  class={row['class_hint']}" if row.get("class_hint") else "",
                    f"  condition={row['condition']}" if row.get("condition") else "",
                )
                if row.get("stack_on"):
                    # Easy to miss in a list of otherwise identical instructions, and
                    # a missed one silently turns a stacked trial into a flat one.
                    log.info("    STACKED: put the %s down on the card FIRST, then the "
                             "%s on top of it.", row["stack_on"],
                             row.get("class_hint", "part"))
            if args.blind_label:
                row_ = pose_rows[i - 1] if pose_rows is not None else {}
                print(f"\n[{args.blind_label}] Trial {i}/{args.trials} — PLACE: "
                      f"card={row_.get('card_id', '?')}  yaw={row_.get('yaw_deg', '?')} deg"
                      f"  class={row_.get('class_hint', '?')}", flush=True)
                if row_.get("stack_on"):
                    print(f"    STACKED: {row_['stack_on']} down first, then "
                          f"{row_.get('class_hint', 'the part')} on top.", flush=True)
            if not args.confirm_each_trial:
                return
            try:
                input("   Placed it? TAKE YOUR HAND OUT of the cell, then press ENTER ")
            except EOFError:
                log.error("stdin closed during --confirm-each-trial; stopping rather than "
                          "moving the robot with nobody confirming.")
                raise KeyboardInterrupt from None
            # The perception node keeps grabbing frames while the operator is still
            # over the table, and the queue holds up to 3 of them. Without this the
            # cycle would detect the table as it was BEFORE the part was placed.
            dropped = 0
            while True:
                try:
                    det_queue.get_nowait()
                    dropped += 1
                except queue.Empty:
                    break
            if dropped:
                log.debug("Dropped %d frame(s) captured before placement finished", dropped)

    if args.confirm_each_trial:

        def human_score_hook(trial_id: int, machine_success: bool) -> int | str:
            """Ask the operator what they saw, and put it in the `human_ok` column.

            The machine's own verdict is deliberately NOT shown in the prompt, and
            the question never names the configuration, so this stays usable inside
            a blinded run. Anything other than the four accepted keys re-asks: a
            stray ENTER must not be recorded as a good grasp.
            """
            while True:
                try:
                    ans = input(f"   Trial {trial_id} — part in the right place? "
                                "[y]=yes  [n]=no  ").strip().lower()
                except EOFError:
                    log.error("stdin closed while scoring trial %d — logging it "
                              "unscored. The next trial will stop the run.", trial_id)
                    return ""
                if ans in {"y", "t", "1"}:
                    return 1
                if ans in {"n", "h", "0"}:
                    return 0
                print("   Type y or n.", flush=True)

        orch.human_score_hook = human_score_hook

    def _drive_experiment() -> None:
        """Run N trials → cleanup digital twin → log summary.

        With --viewport open3d-gui: this function runs in a WORKER thread because
        the Filament GUI holds the main thread (run_gui is blocking). Other modes:
        runs synchronously on the main thread.
        """
        if args.headless:
            # Deterministic: generate exactly N messages (1 scenario/trial) then
            # pre-fill queue in order. NO perception thread → trial i ↔ scenario i,
            # miss/reachability rates match config exactly.
            for _ in range(args.trials):
                msg = perception.process_once()
                if msg is not None:
                    det_queue.put(msg)
            stats = orch.run_n_trials(args.trials, pre_trial_hook=pre_trial_hook)
        else:
            try:
                perception.start()
                stats = orch.run_n_trials(args.trials, pre_trial_hook=pre_trial_hook)
            except KeyboardInterrupt:
                # Real mode: Ctrl+C must stop the GP7 IMMEDIATELY rather than just killing Python.
                # Call robot.Stop() before propagating.
                log.warning("Ctrl+C — emergency robot stop")
                try:
                    if args.mode == "real" and hasattr(orch.robot, "Stop"):
                        orch.robot.Stop()
                        log.info("robot.Stop() sent")
                except Exception as e:  # noqa: BLE001
                    log.error("Error stopping robot: %s", e)
                stats = {"attempted": 0, "successful": 0, "failed": 0,
                         "success_rate": 0.0, "aborted_by_user": True}
            finally:
                perception.stop()

        # Cleanup digital twin — stop mirror thread, close telemetry, close socket.
        # Active for both sim non-headless and hse mode (DigitalTwinMirror used in both).
        if twin is not None:
            try:
                twin.stop_mirror()
                if hasattr(twin.backend, "disconnect"):
                    twin.backend.disconnect()
            except Exception as e:                  # noqa: BLE001
                log.warning("Digital twin cleanup error: %s", e)

        summary = trial_logger.summarize()
        log.info("─" * 60)
        log.info("RESULTS: success_rate=%.1f%% (%d/%d)",
                 stats["success_rate"] * 100, stats["successful"], stats["attempted"])
        log.info("Failure modes: %s", summary["failure_modes"])
        log.info("CSV: %s", trial_logger.csv_path)

    def _shutdown_after_gui(viewer: Any, worker: threading.Thread) -> None:
        """Clean up after GUI closes: join worker + cleanup twin + force-exit if
        Filament still holds C-level threads."""
        # Wait for the worker to detect that the viewer closed (every anim/post where
        # _open=False returns immediately) — typically < 1s. Cap at 3s to avoid hang.
        worker.join(timeout=3.0)
        # Cleanup twin (real mode) — stop_mirror, close telemetry, close HSE socket.
        if twin is not None:
            try:
                twin.stop_mirror()
                if hasattr(twin.backend, "disconnect"):
                    twin.backend.disconnect()
            except Exception as e:                      # noqa: BLE001
                log.warning("Digital twin cleanup error: %s", e)
        if hasattr(viewer, "disconnect"):
            try:
                viewer.disconnect()                     # Application.instance.quit()
            except Exception as e:                      # noqa: BLE001
                log.debug("viewer.disconnect error: %s", e)
        # Open3D Filament holds C-level threads (renderer pool, asset loader)
        # that do not die after Application.quit() on Windows → Python hangs at
        # exit. Force-exit so the terminal prompt returns immediately.
        log.info("Cleanup done — exiting.")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    # Sim non-headless: O3DGuiSimRobot.run_gui runs on main thread, experiment on worker.
    if hasattr(sim_robot, "run_gui"):
        worker = threading.Thread(target=_drive_experiment, name="experiment",
                                  daemon=True)
        worker.start()
        try:
            sim_robot.run_gui("Running… close window to finish "
                              "(mouse: left=rotate, right=pan, scroll=zoom).")
        finally:
            _shutdown_after_gui(sim_robot, worker)
        return 0  # unreachable (os._exit) — kept for linter

    # Real mode with live Open3D mirror: viewer GUI runs on main thread,
    # experiment + DigitalTwinMirror run on worker thread.
    if real_viewer is not None and hasattr(real_viewer, "run_gui"):
        worker = threading.Thread(target=_drive_experiment, name="experiment",
                                  daemon=True)
        worker.start()
        try:
            real_viewer.run_gui("Real digital twin — running… close window to finish "
                                "(mouse: left=rotate, right=pan, scroll=zoom).")
        finally:
            _shutdown_after_gui(real_viewer, worker)
        return 0  # unreachable

    # Headless / real telemetry-only: run synchronously on main thread.
    _drive_experiment()
    return 0


if __name__ == "__main__":
    sys.exit(main())
