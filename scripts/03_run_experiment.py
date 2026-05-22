#!/usr/bin/env python
"""
03_run_experiment.py
────────────────────
Entry point chạy thí nghiệm pick-and-place (xem mục 10 tài liệu).

Khởi động perception + viewport (sim) hoặc HSE backend (real), chạy N trial
qua Orchestrator, ghi kết quả ra results/.

Chế độ:
    --mode sim   : MockCamera + MockDetector + SimRobot + Open3D viewport → L4 test
                   logic pipeline mà không cần D455/model thật.
    --mode real  : D455 + YOLO + HSE backend → GP7 thật, telemetry CSV @10Hz.

Usage:
    python scripts/03_run_experiment.py --mode sim --trials 50
    python scripts/03_run_experiment.py --mode real --trials 50 --lighting bright
"""
from __future__ import annotations

import argparse
import os
import queue
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument(
        "--cell-config", default=None,
        help="Cell layout YAML. Mặc định: cell_layout.yaml cho sim, "
             "cell_layout_real.yaml cho real (auto-pick theo --mode).",
    )
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--mode", choices=["sim", "real"], default="sim")
    parser.add_argument(
        "--backend", choices=["hse", "sim"], default=None,
        help="Robot motion backend. Default: 'sim' khi --headless hoặc "
             "--mode sim, 'hse' khi --mode real.",
    )
    # NOTE: --viewport đã bỏ. Sim non-headless ALWAYS dùng O3DGuiSimRobot
    # (Filament). Real mode dùng O3DGuiSimRobot làm mirror viewport-only. Headless
    # = không viewport. Cần `pip install open3d` cho cả 2 mode non-headless.
    parser.add_argument(
        "--hse-ip", default=None,
        help="IP YRC1000 cho --backend hse. Default: lấy từ robot_connection.ip "
             "trong cell config.",
    )
    parser.add_argument(
        "--mirror-hz", type=float, default=2.0,
        help="Tần số viewport callback gọi (Hz). Default 2.0 — đủ smooth cho mắt "
             "+ giảm overhead. Hiện chưa wire viewport real-mode (telemetry-only).",
    )
    parser.add_argument(
        "--telemetry-hz", type=float, default=10.0,
        help="Tần số backend Joints poll + log CSV (Hz). Default 10.0 — đủ "
             "resolution cho velocity / cycle time analysis. Decouple khỏi "
             "mirror-hz: viewport throttle xuống mirror-hz tự động.",
    )
    parser.add_argument(
        "--no-viewport-mirror", action="store_true",
        help="Tắt viewport callback trong mirror thread. Telemetry CSV vẫn ghi "
             "đầy đủ, visualize offline qua 05_analyze_telemetry.py.",
    )
    parser.add_argument(
        "--ultra-fast", action="store_true",
        help="HSE backend ultra-fast mode: upload INFORM template chỉ 1 lần, "
             "mỗi trial chỉ WRITE_POS_VAR + JOB_START qua HSE (0 FTP overhead). "
             "~50ms/trial thay vì ~200ms (M3 batch). Yêu cầu trial structure "
             "không đổi (vd cùng pattern pick-and-place).",
    )
    parser.add_argument(
        "--ik-source", choices=["yrc", "client"], default=None,
        help="IK source: yrc (YRC1000 tự IK — recommended cho HSE real, 0 DH "
             "dependency phía PC), client (numerical DLS pure-Python URDF chain, "
             "match RoboDK 0.00mm). Default 'yrc' khi --mode real, 'client' khi sim.",
    )
    parser.add_argument(
        "--tool-no", type=int, default=1,
        help="TOOL coordinate number trên YRC1000 (TOOL01 default). Phải khớp "
             "TCP gripper đã setup trên teach pendant. Xem docs/SETUP_YRC_TOOL.md.",
    )
    parser.add_argument("--lighting", default="", help="Nhãn điều kiện ánh sáng (log)")
    parser.add_argument("--overlap", default="", help="Nhãn mức chồng lấn (log)")
    parser.add_argument(
        "--headless", action="store_true",
        help="Chạy KHÔNG có viewport (SimRobot mock) — validate logic + sinh CSV. "
             "Dùng cho CI/CD và batch lớn.",
    )
    parser.add_argument(
        "--minimal-build", action="store_true",
        help="Open3D viewport tối giản (bỏ floor, calib frame, objects phụ) — "
             "tăng FPS render.",
    )
    # ─── Headless scenario tuning (chỉ áp dụng khi --headless) ───
    parser.add_argument(
        "--grasp-fail-rate", type=float, default=0.0,
        help="(headless) Xác suất grasp slip [0..1] → failure mode 'grasp_slip'.",
    )
    parser.add_argument(
        "--detection-miss-rate", type=float, default=0.0,
        help="(headless) Xác suất không phát hiện vật [0..1] → 'detection_miss'.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="(headless) Seed RNG để tái lập kết quả.",
    )
    return parser.parse_args()


DEFAULT_OBJECT_HEIGHTS_MM = {"tray": 25, "bottle": 150, "cup": 40, "bolt": 25}
DEFAULT_MASK_SIZE_PX = {"tray": (180, 100), "bottle": (120, 40), "cup": (80, 80), "bolt": (40, 40)}


def _stl_height_mm(stl_path):
    """Đọc binary STL → chiều cao theo trục Z (max Z - min Z). None nếu lỗi.

    Dùng để auto-detect dimensions object thật thay vì hard-code, giúp depth
    + grasp pose khớp đúng với mesh file thực.
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
    """Auto-compute (mask_box, depth_m, height_mm) từ object pose trong cell_layout.

    Đảm bảo mock detection PHẢN ÁNH ĐÚNG vị trí object template trong RoboDK,
    không phải hard-code pixel center. Project object_top thế giới → pixel qua
    camera intrinsics + camera world pose. Fallback giá trị cũ nếu thiếu config.
    """
    import numpy as np

    from src.cell.pose_utils import make_homogeneous

    obj = next((o for o in cell_config.objects if o.name == object_name), None)
    if obj is None:
        return (580, 350, 700, 390), 0.49, 150

    # World pose của object base = parent_frame.pose + object.pose offset
    if obj.parent_frame:
        parent = next((f for f in cell_config.frames if f.name == obj.parent_frame), None)
        parent_xyz = np.array(parent.pose.xyz_mm, dtype=float) if parent else np.zeros(3)
    else:
        parent_xyz = np.zeros(3)
    offset_xyz = np.array(obj.pose.xyz_mm, dtype=float) if obj.pose else np.zeros(3)
    obj_base_world = parent_xyz + offset_xyz

    # Top object = base + chiều cao thực. Ưu tiên ĐO trực tiếp từ STL file
    # để khớp với mesh thật, tránh hard-code chiều cao có thể sai.
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

    # Camera world pose → T_BC; inverse để đổi world → camera frame
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
    """Tạo (camera, detector) theo chế độ chạy.

    Args:
        cell_config: Nếu có, mock detection sẽ tự sinh từ object pose trong cell
            (mask_box + depth khớp object thật) thay vì hard-code pixel.
    """
    if mode == "real":
        from src.perception import D455Camera, ObjectDetector

        camera = D455Camera()
        detector = ObjectDetector(
            model_path=config.get("model_path", "models/yolov8s-seg_best.pt"),
            conf=config.get("conf_threshold", 0.5),
        )
        return camera, detector

    # mode == "sim" (cả RoboDK sim lẫn headless): kịch bản detection giả lập.
    import numpy as np

    from src.perception import MockCamera, MockDetector

    headless = bool(getattr(args, "headless", False))

    # Auto-compute mask_box + depth từ cell_config (mock detection khớp object
    # template thật trong RoboDK). Target = first object in cell (= tray default).
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
        # RoboDK sim: 1 kịch bản cố định, mask khớp object thật trong cell.
        det = MockDetector.make_detection(target_name, mask_box=mask_box)
        if auto_height_mm is not None:
            det.height_mm = auto_height_mm   # truyền chiều cao thật cho adaptive grasp
        scripted = [[det]]
    else:
        # Headless: sinh nhiều kịch bản đa dạng class + position để có thống kê
        # có ý nghĩa. miss_rate: tỉ lệ trial không có vật (test recovery).
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


def main() -> int:
    args = parse_args()
    log = setup_logging("experiment", log_file=PROJECT_ROOT / "logs/experiment.log")
    log.info("=" * 60)
    log.info("Thí nghiệm pick-and-place — mode=%s, trials=%d", args.mode, args.trials)
    log.info("=" * 60)

    # Auto-pick cell config theo --mode (nếu user không override). Tránh
    # tình huống --mode real vô tình dùng cell_layout.yaml (robot_connection
    # disabled → không bao giờ gửi command tới GP7 thật).
    if args.cell_config is None:
        args.cell_config = (
            "config/cell_layout_real.yaml" if args.mode == "real"
            else "config/cell_layout.yaml"
        )
        log.info("Auto-pick cell-config: %s (theo --mode=%s)", args.cell_config, args.mode)

    config = load_yaml(PROJECT_ROOT / args.config)

    # Real mode: bật C2 safety layer (reach envelope + predictive safety full trajectory).
    # Override default sim-friendly config trong _DEFAULT_CONFIG của orchestrator.
    if args.mode == "real":
        config["is_real_mode"] = True
        config["skip_reachability_check"] = False
        config["predictive_safety_enabled"] = True  # UC2: joint limit + self-collision toàn trajectory
        log.info("Real mode: reach envelope + predictive safety C2 BẬT "
                 "(joint limit + self-collision check toàn trajectory trước MoveJ)")

        # Real mode: enable CC-Link gripper path (double-acting + feedback sensors)
        # Default mapping — TODO verify từ YRC1000 TP: Setup → I/O Module → CC-Link
        # Có thể override qua experiment.yaml (key "gripper_cc_link").
        if "gripper_cc_link" not in config or config["gripper_cc_link"] is None:
            config["gripper_cc_link"] = {
                "clamp_bit":            30010,   # → PLC Y502 (Clamp solenoid)
                "unclamp_bit":          30011,   # → PLC Y503 (UnClamp solenoid)
                "clamp_sensor_bit":     30050,   # ← PLC X504 (cylinder ở Clamp)
                "unclamp_sensor_bit":   30051,   # ← PLC X503 (cylinder ở UnClamp)
                "detect_bit":           30052,   # ← PLC X505 (Carrier Detect)
                "wait_sensor_timeout_s": 2.0,
                "wait_sensor_poll_s":    0.05,
                "require_detect_on_close": True,
            }
            log.info("Gripper CC-Link path ENABLED (default mapping — verify TP)")

    # Load cell_config sớm: cần cho build_perception auto-mock detection
    # từ object pose (mọi mode đều dùng — kể cả --no-build).
    cell_config = CellConfig.from_yaml(PROJECT_ROOT / args.cell_config)

    # Truyền home_joints từ cell config xuống Orchestrator để _return_home dùng
    # ĐÚNG home đã set trong cell (không phải JointsHome() từ file .robot).
    config["home_joints_deg"] = list(cell_config.robot.home_joints_deg)

    # ─── Resolve backend ───
    # Motion backend:
    #   --headless    → sim  (SimRobot bare-metal, không viewport)
    #   --mode sim    → sim  (SimRobot + Open3D viewport, drop-in robot=)
    #   --mode real   → hse  (MotomanHSEBackend UDP → YRC1000, telemetry-only)
    if args.backend is None:
        args.backend = "hse" if args.mode == "real" else "sim"
    if args.mode == "real" and args.backend != "hse":
        log.error("--mode real chỉ chấp nhận --backend hse.")
        return 4
    log.info("Backend: %s", args.backend)

    sim_robot: Any = None             # Orchestrator dùng làm `robot=`
    twin: Any = None                  # DigitalTwinMirror (None khi headless/sim Open3D)
    real_viewer: Any = None           # O3DGuiSimRobot làm viewport-only mirror cho real

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
        # SimRobot + Open3D GUI; GUI chạy main thread, experiment chạy worker.
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
        config["skip_reachability_check"] = False   # SimRobot có reach envelope
        config["inter_trial_delay_s"] = 0.0
        log.info("Viewport: O3DGuiSimRobot (SimRobot motion + client DLS IK).")
    else:                                           # args.backend == "hse"
        # ─── Real mode: HSE backend + DigitalTwinMirror (telemetry-only) ──
        from src.orchestrator.digital_twin import DigitalTwinMirror
        from src.orchestrator.backends.motoman_hse import MotomanHSEBackend
        from src.orchestrator.telemetry import TelemetryLogger

        hse_ip = args.hse_ip or cell_config.robot_connection.ip
        if not hse_ip:
            log.error("--backend hse cần IP YRC1000 (qua --hse-ip hoặc cell config)")
            return 2
        backend = MotomanHSEBackend(
            ip=hse_ip,
            max_speed_pct=cell_config.robot_connection.max_speed_percent,
            tool_no=args.tool_no,
        )
        backend.set_home_joints(list(cell_config.robot.home_joints_deg))
        backend.connect()
        if not backend.Valid():
            log.error("HSE heartbeat fail — kiểm tra ping %s + HSE Server function", hse_ip)
            backend.disconnect()
            return 3
        log.info("HSE backend connected: %s. Joints (sanity): %s",
                 hse_ip, backend.Joints())
        if args.ultra_fast:
            backend.enable_ultra_fast(True)
            log.info("HSE Ultra-fast P-var mode: ON (template upload chỉ 1 lần)")

        # IK source: yrc (controller-side, recommended) hoặc client (DLS PC-side)
        ik_source = args.ik_source or "yrc"
        if ik_source == "yrc":
            config["use_yrc_ik"] = True
            log.info("IK source: YRC1000 controller (0 PC IK overhead). "
                     "TOOL%02d phải setup trên TP.", args.tool_no)
        else:
            config["use_client_ik"] = True
            log.info("IK source: client-side DLS (URDF chain match RoboDK 0.00mm).")

        telemetry = TelemetryLogger(
            PROJECT_ROOT / f"results/telemetry_{timestamp()}.csv"
        )

        # ─── Live Open3D mirror cho real mode ──────────────────────────────
        # O3DGuiSimRobot làm RENDER-ONLY (không dùng làm robot=). HSE backend
        # poll Joints @telemetry_hz; mirror thread gọi viewport_callback @mirror_hz;
        # callback post transform lên GUI thread (thread-safe qua post_to_main_thread).
        # Main thread block trong real_viewer.run_gui() tới khi user đóng cửa sổ.
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
                log.info("Real-mode Open3D mirror sẵn sàng (Filament GUI).")
            except Exception as e:                              # noqa: BLE001
                log.warning("Không tạo được Open3D mirror cho real mode: %s. "
                            "Fallback telemetry-only (replay qua 07_replay_telemetry.py).",
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
        sim_robot = twin                # Orchestrator nhận façade qua `robot=`
        log.info(
            "DigitalTwinMirror active (HSE) — telemetry %.1fHz, mirror %.1fHz%s",
            args.telemetry_hz, args.mirror_hz,
            " (live Open3D viewport ON)" if viewport_cb else " (viewport OFF — telemetry-only)",
        )

    # ─── Perception ───
    # Pass cell_config để mock detection auto-khớp object pose thật.
    camera, detector = build_perception(args.mode, config, args, cell_config)
    # Headless: queue đủ lớn để pre-fill 1 scenario/trial (deterministic).
    qsize = args.trials + 1 if args.headless else 3
    det_queue: queue.Queue = queue.Queue(maxsize=qsize)
    perception = PerceptionNode(camera, detector, det_queue)

    # ─── Logger ───
    ts = timestamp()
    label = "headless" if args.headless else args.mode
    trial_logger = TrialLogger(
        PROJECT_ROOT / f"results/experiment_{label}_{ts}.csv",
        extra_context={"lighting": args.lighting, "overlap": args.overlap,
                       "mode": label},
    )

    # ─── Orchestrator ───
    orch = Orchestrator(det_queue, config=config, robot=sim_robot,
                        logger_obj=trial_logger)

    def _drive_experiment() -> None:
        """Chạy N trial → cleanup digital twin → log summary.

        Với --viewport open3d-gui: hàm này chạy ở WORKER thread vì GUI Filament
        giữ main thread (run_gui blocking). Các mode khác: chạy đồng bộ main thread.
        """
        if args.headless:
            # Deterministic: sinh đúng N message (1 scenario/trial) rồi pre-fill
            # queue theo thứ tự. KHÔNG dùng perception thread → trial i ↔ scenario i,
            # tỉ lệ miss/reachability khớp chính xác config.
            for _ in range(args.trials):
                msg = perception.process_once()
                if msg is not None:
                    det_queue.put(msg)
            stats = orch.run_n_trials(args.trials)
        else:
            try:
                perception.start()
                stats = orch.run_n_trials(args.trials)
            except KeyboardInterrupt:
                # Real mode: Ctrl+C phải dừng GP7 NGAY thay vì chỉ kill Python.
                # Gọi robot.Stop() trước khi propagate.
                log.warning("Ctrl+C — dừng robot khẩn cấp")
                try:
                    if args.mode == "real" and hasattr(orch.robot, "Stop"):
                        orch.robot.Stop()
                        log.info("robot.Stop() đã gửi")
                except Exception as e:  # noqa: BLE001
                    log.error("Lỗi khi Stop robot: %s", e)
                stats = {"attempted": 0, "successful": 0, "failed": 0,
                         "success_rate": 0.0, "aborted_by_user": True}
            finally:
                perception.stop()

        # Cleanup digital twin — stop mirror thread, close telemetry, đóng socket.
        # Active cho cả sim non-headless và hse mode (DigitalTwinMirror dùng cho cả 2).
        if twin is not None:
            try:
                twin.stop_mirror()
                if hasattr(twin.backend, "disconnect"):
                    twin.backend.disconnect()
            except Exception as e:                  # noqa: BLE001
                log.warning("Cleanup digital twin lỗi: %s", e)

        summary = trial_logger.summarize()
        log.info("─" * 60)
        log.info("KẾT QUẢ: success_rate=%.1f%% (%d/%d)",
                 stats["success_rate"] * 100, stats["successful"], stats["attempted"])
        log.info("Failure modes: %s", summary["failure_modes"])
        log.info("CSV: %s", trial_logger.csv_path)

    def _shutdown_after_gui(viewer: Any, worker: threading.Thread) -> None:
        """Dọn sau khi GUI đóng: join worker + cleanup twin + force-exit nếu
        Filament còn ngậm C-level threads."""
        # Đợi worker đến lúc nó tự nhận biết viewer đóng (mọi anim/post chỗ
        # _open=False return ngay) → thường < 1s. Cap 3s để không treo.
        worker.join(timeout=3.0)
        # Cleanup twin (real mode) — stop_mirror, close telemetry, đóng HSE socket.
        if twin is not None:
            try:
                twin.stop_mirror()
                if hasattr(twin.backend, "disconnect"):
                    twin.backend.disconnect()
            except Exception as e:                      # noqa: BLE001
                log.warning("Cleanup digital twin lỗi: %s", e)
        if hasattr(viewer, "disconnect"):
            try:
                viewer.disconnect()                     # Application.instance.quit()
            except Exception as e:                      # noqa: BLE001
                log.debug("viewer.disconnect lỗi: %s", e)
        # Open3D Filament giữ C-level threads (renderer pool, asset loader)
        # không tự chết sau Application.quit() trên Windows → Python hang ở
        # exit. Force-exit để terminal trả prompt ngay.
        log.info("Cleanup xong — exit.")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    # Sim non-headless: O3DGuiSimRobot.run_gui chạy main thread, experiment worker.
    if hasattr(sim_robot, "run_gui"):
        worker = threading.Thread(target=_drive_experiment, name="experiment",
                                  daemon=True)
        worker.start()
        try:
            sim_robot.run_gui("Đang chạy… đóng cửa sổ để kết thúc "
                              "(chuột: trái=xoay, phải=pan, lăn=zoom).")
        finally:
            _shutdown_after_gui(sim_robot, worker)
        return 0  # unreachable (os._exit) — giữ cho linter

    # Real mode với live Open3D mirror: viewer GUI chạy main thread,
    # experiment + DigitalTwinMirror chạy worker thread.
    if real_viewer is not None and hasattr(real_viewer, "run_gui"):
        worker = threading.Thread(target=_drive_experiment, name="experiment",
                                  daemon=True)
        worker.start()
        try:
            real_viewer.run_gui("Real digital twin — đang chạy… đóng cửa sổ để kết thúc "
                                "(chuột: trái=xoay, phải=pan, lăn=zoom).")
        finally:
            _shutdown_after_gui(real_viewer, worker)
        return 0  # unreachable

    # Headless / real telemetry-only: chạy đồng bộ trên main thread.
    _drive_experiment()
    return 0


if __name__ == "__main__":
    sys.exit(main())
