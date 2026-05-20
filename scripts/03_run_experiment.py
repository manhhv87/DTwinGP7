#!/usr/bin/env python
"""
03_run_experiment.py
────────────────────
Entry point chạy thí nghiệm pick-and-place (xem mục 10 tài liệu).

Dựng cell trong RoboDK, khởi động perception, chạy N trial qua Orchestrator,
ghi kết quả ra results/.

Chế độ:
    --mode sim   : dùng MockCamera + MockDetector (không cần D455/model thật),
                   nhưng vẫn chạy full pipeline + RoboDK digital twin → L4 test.
    --mode real  : dùng D455 + YOLO thật + kết nối GP7 → L5 test.

Usage:
    python scripts/03_run_experiment.py --mode sim --trials 50
    python scripts/03_run_experiment.py --mode real --trials 50 --lighting bright
"""
from __future__ import annotations

import argparse
import queue
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cell import CellConfig, CellLoader  # noqa: E402
from src.logging import TrialLogger  # noqa: E402
from src.orchestrator import Orchestrator  # noqa: E402
from src.perception import PerceptionNode  # noqa: E402
from src.utils import load_yaml, setup_logging, timestamp  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--cell-config", default="config/cell_layout.yaml")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--mode", choices=["sim", "real"], default="sim")
    parser.add_argument("--lighting", default="", help="Nhãn điều kiện ánh sáng (log)")
    parser.add_argument("--overlap", default="", help="Nhãn mức chồng lấn (log)")
    parser.add_argument(
        "--headless", action="store_true",
        help="Chạy KHÔNG cần RoboDK (SimRobot mock) — validate logic + sinh CSV, "
             "0 API call. Dùng để phát triển pipeline khi RoboDK Free hết quota.",
    )
    parser.add_argument(
        "--no-build", action="store_true",
        help="Bỏ qua dựng cell — dùng station RoboDK đã build sẵn, tiết kiệm "
             "~22 API call. Chạy build_station.py trước.",
    )
    parser.add_argument(
        "--minimal-build", action="store_true",
        help="Build cell tối giản (bỏ floor, Cam2D, calib frame, 2/3 objects) "
             "→ tiết kiệm ~10 API call để chạy thêm trial với RoboDK Free quota.",
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


DEFAULT_OBJECT_HEIGHTS_MM = {"tray": 25}
DEFAULT_MASK_SIZE_PX = {"tray": (180, 100)}


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
        # Headless: sinh N kịch bản tray với position varying (pixel u/v) để
        # thống kê success rate trong nhiều vị trí. miss_rate: tỉ lệ trial
        # không có vật (test detection_miss recovery).
        import random

        n = getattr(args, "trials", 50)
        miss_rate = getattr(args, "detection_miss_rate", 0.0)
        rng = random.Random(getattr(args, "seed", 42))
        tray_height = DEFAULT_OBJECT_HEIGHTS_MM.get("tray", 25)
        scripted = []
        for _ in range(n):
            if rng.random() < miss_rate:
                scripted.append([])                       # detection miss
                continue
            cu = rng.randint(450, 1050)                   # pixel u → world X span
            cv = rng.randint(330, 410)
            det_h = MockDetector.make_detection(
                "tray", mask_box=(cu - 90, cv - 50, cu + 90, cv + 50),
            )
            det_h.height_mm = tray_height
            scripted.append([det_h])
    detector = MockDetector(scripted=scripted)
    return camera, detector


def main() -> int:
    args = parse_args()
    log = setup_logging("experiment", log_file=PROJECT_ROOT / "logs/experiment.log")
    log.info("=" * 60)
    log.info("Thí nghiệm pick-and-place — mode=%s, trials=%d", args.mode, args.trials)
    log.info("=" * 60)

    config = load_yaml(PROJECT_ROOT / args.config)

    # Load cell_config sớm: cần cho build_perception auto-mock detection
    # từ object pose (mọi mode đều dùng — kể cả --no-build).
    cell_config = CellConfig.from_yaml(PROJECT_ROOT / args.cell_config)

    # Truyền home_joints từ cell config xuống Orchestrator để _return_home dùng
    # ĐÚNG home đã set trong cell (không phải JointsHome() từ file .robot).
    config["home_joints_deg"] = list(cell_config.robot.home_joints_deg)

    # ─── Robot + cell ───
    sim_robot = None
    robodk_objects: dict = {}
    robodk_tool = None
    if args.headless:
        # Headless: KHÔNG kết nối RoboDK, dùng SimRobot mock → 0 API call.
        from src.orchestrator.sim_robot import SimRobot

        sim_robot = SimRobot(
            home_joints=cell_config.robot.home_joints_deg,
            base_xyz=tuple(cell_config.robot.pose.xyz_mm),    # J1 world cho reach model
            grasp_fail_rate=args.grasp_fail_rate,
            seed=args.seed,
        )
        # Headless BẬT reachability check để exercise C2 logic (SimRobot có
        # reach model riêng, không tốn API). Bỏ delay giữa trials để chạy nhanh.
        config["skip_reachability_check"] = False
        config["inter_trial_delay_s"] = 0.0
        config["gripper_delay_s"] = 0.0
        log.info("HEADLESS mode — SimRobot (base=%s, grasp_fail=%.0f%%, miss=%.0f%%).",
                 list(cell_config.robot.pose.xyz_mm),
                 args.grasp_fail_rate * 100, args.detection_miss_rate * 100)
    elif args.no_build:
        log.info("Bỏ qua dựng cell (--no-build) — dùng station RoboDK đã có.")
    else:
        # Dựng cell trong RoboDK.
        loader = CellLoader(cell_config, project_root=PROJECT_ROOT,
                            minimal_build=args.minimal_build)
        items = loader.build()
        # Lưu references để Orchestrator attach/detach object khi gắp-thả.
        robodk_objects = items.get("objects", {}) or {}
        robodk_tool = items.get("tool")
        log.info("Cell đã dựng trong RoboDK%s.",
                 " (minimal)" if args.minimal_build else "")

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
                        logger_obj=trial_logger,
                        robodk_objects=robodk_objects,
                        robodk_tool=robodk_tool)

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
        finally:
            perception.stop()

    summary = trial_logger.summarize()
    log.info("─" * 60)
    log.info("KẾT QUẢ: success_rate=%.1f%% (%d/%d)",
             stats["success_rate"] * 100, stats["successful"], stats["attempted"])
    log.info("Failure modes: %s", summary["failure_modes"])
    log.info("CSV: %s", trial_logger.csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
