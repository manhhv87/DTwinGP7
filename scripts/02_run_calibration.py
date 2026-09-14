#!/usr/bin/env python
"""
02_run_calibration.py
─────────────────────
Hand-eye calibration eye-to-hand for D455 + GP7 (see section 6 of the manual),
followed by a measurement of the feeding-table plane in the same frame.

Procedure:
    1. Robot holds a ChArUco board on the end-effector
    2. Move the robot to 25-30 diverse poses (rotation ±30°) using the teach pendant
    3. At each pose: press ENTER → script reads joints from YRC1000 via HSE, computes
       T_gripper2base using URDF FK pure-Python, captures ChArUco frame
    4. Press 's' to solve → saves config/calibration/T_base_camera.npy and
       T_base_camera_meta.json (robot base pose used, board, method, intrinsics)
    5. Clear the table, move the robot out of view, press ENTER → the table plane is
       fitted from depth through the freshly solved T_BC and written to
       config/calibration/table_plane.json. Real-mode runs read the table height from
       there, so it is in the frame of T_BC by construction.

Usage:
    python scripts/02_run_calibration.py --hse-ip 192.168.1.100 --square-mm 40 --marker-mm 30
    python scripts/02_run_calibration.py --method park --hse-ip 192.168.1.100
    python scripts/02_run_calibration.py --table-only        # re-measure the table only

The board geometry MUST match the printed board. Measure a square with calipers and
pass --squares/--square-mm/--marker-mm/--dict. A wrong square size still detects
fine but scales every board translation, and with it the whole camera position,
without any error.

Why the robot base pose is recorded: FK uses the base pose of the cell layout, so
T_BC comes out in that frame, and the experiment converts grasp targets back with
the pose it reads at run time. If the layout's base pose is edited after this
calibration, every grasp shifts by the difference; the experiment's preflight
compares the two and refuses to run.

Notes:
  - YRC1000 must have High-Speed Ethernet Server function enabled (Maintenance mode)
  - Robot mode: TEACH (REMOTE not required — reads joints only, no motion commands)
  - URDF chain forward kinematics verified to match RoboDK SolveFK 0.00mm
    (see scripts/13_verify_vs_robodk.py)
  - UNITS: the entire pipeline uses mm. FK gripper2base returns mm; estimate_pose
    converts ChArUco target2cam from metres to mm. solve_hand_eye requires both
    inputs in the SAME unit → keep mm throughout; output T_BC is also mm → save
    as-is (do NOT convert to metres).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import CalibrationSession, CharucoBoardEstimator  # noqa: E402
from src.cell import CellConfig  # noqa: E402
from src.orchestrator.backends.motoman_hse import MotomanHSEBackend  # noqa: E402
from src.orchestrator.coord_conv import load_calibration, save_calibration  # noqa: E402
from src.orchestrator.kinematics.urdf_chain import (  # noqa: E402
    forward_kinematics_urdf,
    gp7_urdf,
)
from src.orchestrator.preflight import (  # noqa: E402
    TABLE_PLANE_FILE,
    is_sim_placeholder,
    meta_path_for,
)
from src.utils import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell-config", default="config/cell_layout_real.yaml",
                        help="Cell config (provides robot base pose + tool offset).")
    parser.add_argument("--hse-ip", default=None,
                        help="YRC1000 IP address. Default: robot_connection.ip from cell config.")
    # Default "park": "tsai" is incorrect for a downward-facing camera (rotates ~180°).
    parser.add_argument("--method", default="park",
                        choices=["park", "horaud", "daniilidis", "andreff", "tsai"])
    parser.add_argument("--output", default="config/calibration/T_base_camera.npy")
    parser.add_argument(
        "--bootstrap", type=int, default=0, metavar="B",
        help="Bootstrap the solve B times (paper uses 200) → per-parameter sigma "
             "for anchored DR (Eq. 1). Writes --sigma-output.")
    parser.add_argument(
        "--sigma-output", default="config/calibration/T_base_camera_sigma.json",
        help="Where to write the bootstrap sigma JSON (synthgen sampler input).")
    parser.add_argument("--squares", type=int, nargs=2, default=[7, 5], metavar=("COLS", "ROWS"),
                        help="ChArUco squares (columns rows) of the PRINTED board. Default 7 5.")
    parser.add_argument("--square-mm", type=float, default=40.0,
                        help="Measured side of one chessboard square, mm. Default 40.")
    parser.add_argument("--marker-mm", type=float, default=30.0,
                        help="Measured side of one ArUco marker, mm. Default 30.")
    parser.add_argument("--dict", default="DICT_4X4_50",
                        help="ArUco dictionary of the printed board. Default DICT_4X4_50.")
    parser.add_argument("--table-only", action="store_true",
                        help="Skip the hand-eye capture; measure the table plane with the "
                             "existing T_BC only.")
    parser.add_argument("--table-frames", type=int, default=10,
                        help="Depth frames combined (median) for the table plane. Default 10.")
    return parser.parse_args()


def _measure_table(camera, T_BC_mm: np.ndarray, calib_path: Path, n_frames: int, log) -> int:
    """Fit the table plane from depth through T_BC and write table_plane.json."""
    from src.calibration.table_plane import depth_to_points, fit_table_plane

    depths = []
    for _ in range(3 * n_frames):
        _, depth = camera.get_frame()
        if depth is not None:
            depths.append(depth)
        if len(depths) >= n_frames:
            break
    if not depths:
        log.error("No depth frame captured; table plane NOT measured.")
        return 1
    depth = np.median(np.stack(depths), axis=0)
    try:
        plane = fit_table_plane(depth_to_points(depth, camera.intrinsics, T_BC_mm))
    except ValueError as e:
        log.error("Table plane fit failed: %s", e)
        return 1
    plane["frames"] = len(depths)
    plane["date"] = datetime.now().isoformat(timespec="seconds")
    # The T_BC this plane was measured with: the preflight rejects the plane if the
    # calibration is re-solved later without re-measuring the table.
    plane["T_BC"] = np.asarray(T_BC_mm, float).tolist()
    out = Path(calib_path).with_name(TABLE_PLANE_FILE)
    out.write_text(json.dumps(plane, indent=2), encoding="utf-8")
    log.info("Table plane: top %.1f mm (mean %.1f, rms %.2f mm, tilt %.2f deg, %d/%d "
             "inliers) → %s", plane["z_top_mm"], plane["z_mean_mm"], plane["rms_mm"],
             plane["tilt_deg"], plane["n_inliers"], plane["n_points"], out)
    if plane["tilt_deg"] > 2.0:
        log.warning("Table tilts %.1f deg in the T_BC frame: check the calibration, or "
                    "whether the table really is out of level.", plane["tilt_deg"])
    return 0


def main() -> int:
    args = parse_args()
    log = setup_logging("calibration", log_file=PROJECT_ROOT / "logs/calibration.log")

    try:
        from src.perception.camera import D455Camera
        import cv2
    except ImportError as e:
        log.error("Missing dependency: %s", e)
        return 1

    cell_config = CellConfig.from_yaml(PROJECT_ROOT / args.cell_config)
    out_path = PROJECT_ROOT / args.output

    camera = D455Camera()
    try:
        intr = camera.intrinsics
        log.info("Colour intrinsics: fx=%.2f fy=%.2f ppx=%.2f ppy=%.2f model=%s coeffs=%s",
                 intr["fx"], intr["fy"], intr["ppx"], intr["ppy"],
                 intr.get("model"), intr.get("coeffs"))
        coeffs = [float(c) for c in (intr.get("coeffs") or [])]
        if any(abs(c) > 1e-9 for c in coeffs):
            log.warning("The colour stream reports non-zero distortion (%s, %s), but this "
                        "script solves with dist=0; the board poses carry that error.",
                        intr.get("model"), coeffs)

        if args.table_only:
            try:
                T_BC_mm = load_calibration(out_path)
            except (FileNotFoundError, ValueError) as e:
                log.error("%s", e)
                return 1
            if is_sim_placeholder(T_BC_mm):
                log.error("%s is still the simulation placeholder: calibrate first.", out_path)
                return 1
            return _measure_table(camera, T_BC_mm, out_path, args.table_frames, log)

        hse_ip = args.hse_ip or cell_config.robot_connection.ip
        if not hse_ip:
            log.error("YRC1000 IP required (--hse-ip or robot_connection.ip in cell config).")
            return 1

        tool_offset_mm = 0.0
        if hasattr(cell_config, "tool") and cell_config.tool:
            tcp = getattr(cell_config.tool, "tcp_offset_mm", None)
            if tcp:
                tool_offset_mm = float(tcp[2])

        base_xyz = tuple(float(v) for v in cell_config.robot.pose.xyz_mm)
        base_rpy = tuple(float(v) for v in cell_config.robot.pose.rpy_deg)
        urdf_model = gp7_urdf(base_xyz_mm=base_xyz, tool_offset_mm=tool_offset_mm)
        log.info("FK frame: robot base at %s mm, rpy %s deg (from %s). T_BC is solved in "
                 "this frame; the experiment must run with the same cell layout.",
                 list(base_xyz), list(base_rpy), args.cell_config)

        K = np.array([
            [intr["fx"], 0, intr["ppx"]],
            [0, intr["fy"], intr["ppy"]],
            [0, 0, 1],
        ])
        # D455 colour stream treated as rectified: dist=0 (see the warning above).
        dist = np.zeros(5)

        estimator = CharucoBoardEstimator(
            squares_xy=(args.squares[0], args.squares[1]),
            square_length_m=args.square_mm / 1000.0,
            marker_length_m=args.marker_mm / 1000.0,
            dictionary=args.dict,
        )
        log.info("BOARD: %dx%d squares, square %.2f mm, marker %.2f mm, %s. These MUST "
                 "match the printed board: a wrong square size scales the camera position "
                 "without any error.", args.squares[0], args.squares[1], args.square_mm,
                 args.marker_mm, args.dict)
        session = CalibrationSession(estimator)

        backend = MotomanHSEBackend(ip=hse_ip)
        backend.connect()
        if not backend.Valid():
            log.error("HSE heartbeat fail — check ping %s and HSE Server function", hse_ip)
            backend.disconnect()
            return 1
        log.info("HSE connected: %s", hse_ip)

        log.info("=" * 60)
        log.info("Hand-eye calibration — move robot with teach pendant then press ENTER to capture a pose.")
        log.info("Press 's' to solve (need >=10 poses), 'q' to abort.")
        log.info("=" * 60)

        try:
            while True:
                cmd = input(f"[pose {session.num_poses}] ENTER=capture / s=solve / q=abort: ")
                cmd = cmd.strip().lower()
                if cmd == "q":
                    log.info("Calibration aborted.")
                    return 1
                if cmd == "s":
                    break
                rgb, _ = camera.get_frame()
                if rgb is None:
                    log.warning("Could not capture frame, retrying.")
                    continue
                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

                # Read joints from YRC1000 via HSE → FK pure-Python → T_gripper2base (mm)
                joints_deg = backend.Joints()
                joints_rad = [np.deg2rad(j) for j in joints_deg]
                # FK returns mm; session + estimate_pose both use mm → do NOT convert to metres
                # (solve_hand_eye requires gripper2base & target2cam in the SAME unit).
                T_gripper2base = forward_kinematics_urdf(urdf_model, joints_rad)

                if session.capture_pose(gray, T_gripper2base, K, dist):
                    log.info("→ Captured pose #%d (joints=%s)",
                             session.num_poses,
                             [round(j, 1) for j in joints_deg])
        finally:
            backend.disconnect()

        # ─── Solve + save ───
        # All inputs in mm → solve returns T_BC in mm (same unit as input) → save as-is.
        T_BC_mm = session.solve(method=args.method)
        save_calibration(out_path, T_BC_mm)
        meta = {
            "date": datetime.now().isoformat(timespec="seconds"),
            "cell_config": args.cell_config,
            "robot_base_xyz_mm": list(base_xyz),
            "robot_base_rpy_deg": list(base_rpy),
            "tool_offset_mm": tool_offset_mm,
            "method": args.method,
            "n_poses": session.num_poses,
            "board": {"squares": list(args.squares), "square_mm": args.square_mm,
                      "marker_mm": args.marker_mm, "dictionary": args.dict},
            "intrinsics": {k: intr[k] for k in ("fx", "fy", "ppx", "ppy") if k in intr},
        }
        meta_path_for(out_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log.info("Saved T_base_camera (mm) → %s (+ %s)", out_path, meta_path_for(out_path).name)
        log.info("Camera at base frame: %s mm", T_BC_mm[:3, 3].round(2))

        # ─── Bootstrap uncertainty (anchored-DR sampling widths, paper §3.4) ───
        if args.bootstrap > 0:
            from src.calibration.uncertainty import bootstrap_hand_eye, save_sigma_json

            result = bootstrap_hand_eye(
                session.poses_gripper2base,
                session.poses_target2cam,
                method=args.method,
                n_boot=args.bootstrap,
            )
            save_sigma_json(result, PROJECT_ROOT / args.sigma_output)
            log.info(
                "Bootstrap B=%d (valid %d): sigma_t=%s mm, sigma_r=%s deg → %s",
                args.bootstrap, result["n_valid"],
                result["sigma_trans_mm"], result["sigma_rot_deg"], args.sigma_output,
            )

        # ─── Table plane in the same frame ───
        ans = input("Clear the table and move the robot out of the camera view, then press "
                    "ENTER to measure the table plane ('k' to skip): ").strip().lower()
        if ans == "k":
            log.warning("Table plane NOT measured: run --table-only before any real trial "
                        "(the experiment refuses to start without it).")
        else:
            _measure_table(camera, T_BC_mm, out_path, args.table_frames, log)

        log.info("Verify further with touch test (section 6.4).")
        return 0
    finally:
        camera.stop()


if __name__ == "__main__":
    sys.exit(main())
