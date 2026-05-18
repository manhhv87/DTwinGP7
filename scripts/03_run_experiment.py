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
    return parser.parse_args()


def build_perception(mode: str, config: dict):
    """Tạo (camera, detector) theo chế độ chạy."""
    if mode == "real":
        from src.perception import D455Camera, ObjectDetector

        camera = D455Camera()
        detector = ObjectDetector(
            model_path=config.get("model_path", "models/yolov8s-seg_best.pt"),
            conf=config.get("conf_threshold", 0.5),
        )
        return camera, detector

    # mode == "sim": kịch bản detection giả lập.
    from src.perception import MockCamera, MockDetector

    camera = MockCamera()
    scripted = [[MockDetector.make_detection("bottle")]]
    detector = MockDetector(scripted=scripted)
    return camera, detector


def main() -> int:
    args = parse_args()
    log = setup_logging("experiment", log_file=PROJECT_ROOT / "logs/experiment.log")
    log.info("=" * 60)
    log.info("Thí nghiệm pick-and-place — mode=%s, trials=%d", args.mode, args.trials)
    log.info("=" * 60)

    config = load_yaml(PROJECT_ROOT / args.config)

    # ─── Dựng cell trong RoboDK ───
    cell_config = CellConfig.from_yaml(PROJECT_ROOT / args.cell_config)
    loader = CellLoader(cell_config, project_root=PROJECT_ROOT)
    loader.build()
    log.info("Cell đã dựng trong RoboDK.")

    # ─── Perception ───
    camera, detector = build_perception(args.mode, config)
    det_queue: queue.Queue = queue.Queue(maxsize=3)
    perception = PerceptionNode(camera, detector, det_queue)

    # ─── Logger ───
    ts = timestamp()
    trial_logger = TrialLogger(
        PROJECT_ROOT / f"results/experiment_{args.mode}_{ts}.csv",
        extra_context={"lighting": args.lighting, "overlap": args.overlap,
                       "mode": args.mode},
    )

    # ─── Orchestrator ───
    orch = Orchestrator(det_queue, config=config, logger_obj=trial_logger)

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
