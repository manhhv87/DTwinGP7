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


def build_perception(mode: str, config: dict, args=None):
    """Tạo (camera, detector) theo chế độ chạy."""
    if mode == "real":
        from src.perception import D455Camera, ObjectDetector

        camera = D455Camera()
        detector = ObjectDetector(
            model_path=config.get("model_path", "models/yolov8s-seg_best.pt"),
            conf=config.get("conf_threshold", 0.5),
        )
        return camera, detector

    # mode == "sim" (cả RoboDK sim lẫn headless): kịch bản detection giả lập.
    # Depth phải khớp scene thật: camera ở Z=1200, bottle top ở Z=710
    # → khoảng cách = 490mm = 0.49m.
    import numpy as np

    from src.perception import MockCamera, MockDetector

    sim_depth_m = config.get("sim_depth_m", 0.49)
    h, w = 720, 1280
    camera = MockCamera(
        depth_frames=[np.full((h, w), sim_depth_m, np.float32)],
    )

    headless = bool(getattr(args, "headless", False))
    if not headless:
        # RoboDK sim: 1 kịch bản cố định, mask wide → yaw≈0 → orientation đơn giản.
        scripted = [[MockDetector.make_detection(
            "bottle", mask_box=(580, 350, 700, 390),
        )]]
    else:
        # Headless: sinh nhiều kịch bản đa dạng để thống kê có ý nghĩa.
        # - detection_miss_rate: tỉ lệ trial không có vật
        # - mask center span theo u → world X span → 1 phần ngoài reach (unreachable)
        import random

        n = getattr(args, "trials", 50)
        miss_rate = getattr(args, "detection_miss_rate", 0.0)
        rng = random.Random(getattr(args, "seed", 42))
        classes = ["bottle", "cup", "bolt"]
        scripted = []
        for _ in range(n):
            if rng.random() < miss_rate:
                scripted.append([])                       # detection miss
                continue
            cls = rng.choice(classes)
            cu = rng.randint(450, 1050)                   # pixel u → world X span
            cv = rng.randint(330, 410)
            scripted.append([MockDetector.make_detection(
                cls, mask_box=(cu - 60, cv - 20, cu + 60, cv + 20),
            )])
    detector = MockDetector(scripted=scripted)
    return camera, detector


def main() -> int:
    args = parse_args()
    log = setup_logging("experiment", log_file=PROJECT_ROOT / "logs/experiment.log")
    log.info("=" * 60)
    log.info("Thí nghiệm pick-and-place — mode=%s, trials=%d", args.mode, args.trials)
    log.info("=" * 60)

    config = load_yaml(PROJECT_ROOT / args.config)

    # ─── Robot + cell ───
    sim_robot = None
    if args.headless:
        # Headless: KHÔNG kết nối RoboDK, dùng SimRobot mock → 0 API call.
        from src.orchestrator.sim_robot import SimRobot

        cell_config = CellConfig.from_yaml(PROJECT_ROOT / args.cell_config)
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
        cell_config = CellConfig.from_yaml(PROJECT_ROOT / args.cell_config)
        loader = CellLoader(cell_config, project_root=PROJECT_ROOT,
                            minimal_build=args.minimal_build)
        loader.build()
        log.info("Cell đã dựng trong RoboDK%s.",
                 " (minimal)" if args.minimal_build else "")

    # ─── Perception ───
    camera, detector = build_perception(args.mode, config, args)
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
