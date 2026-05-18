#!/usr/bin/env python
"""
02_run_calibration.py
─────────────────────
Hand-eye calibration eye-to-hand cho D455 + GP7 (xem mục 6 tài liệu).

Quy trình:
    1. Robot giữ ChArUco board trên end-effector
    2. Di chuyển robot tới 25-30 pose đa dạng (rotation ±30°)
    3. Tại mỗi pose: gõ ENTER để script ghi (ảnh ChArUco + robot.Pose())
    4. Gõ 's' để giải → lưu config/calibration/T_base_camera.npy

Usage:
    python scripts/02_run_calibration.py
    python scripts/02_run_calibration.py --method park --robot "Yaskawa GP7"

Lưu ý đơn vị: target2cam (từ OpenCV) tính bằng MÉT → script quy đổi
robot.Pose() từ mm sang mét trước khi giải, kết quả cuối xuất ra mm.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import CalibrationSession, CharucoBoardEstimator  # noqa: E402
from src.orchestrator.coord_conv import save_calibration  # noqa: E402
from src.utils import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default="Yaskawa GP7")
    # Mặc định "park": "tsai" sai với camera nhìn xuống (xoay ~180°).
    parser.add_argument("--method", default="park",
                        choices=["park", "horaud", "daniilidis", "andreff", "tsai"])
    parser.add_argument("--output", default="config/calibration/T_base_camera.npy")
    return parser.parse_args()


def robodk_pose_to_matrix_m(robodk_pose) -> np.ndarray:
    """RoboDK Mat (mm) → numpy 4x4, translation quy đổi sang mét."""
    T = np.array(robodk_pose.Rows()).reshape(4, 4)
    T[:3, 3] /= 1000.0
    return T


def main() -> int:
    args = parse_args()
    log = setup_logging("calibration", log_file=PROJECT_ROOT / "logs/calibration.log")

    try:
        from src.perception.camera import D455Camera
        from robodk.robolink import Robolink
        import cv2
    except ImportError as e:
        log.error("Thiếu dependency: %s", e)
        return 1

    # ─── Khởi tạo camera + robot ───
    camera = D455Camera()
    K = np.array([
        [camera.intrinsics["fx"], 0, camera.intrinsics["ppx"]],
        [0, camera.intrinsics["fy"], camera.intrinsics["ppy"]],
        [0, 0, 1],
    ])
    # D455 color stream đã rectified — dùng dist=0. Nếu cần độ chính xác cao
    # hơn, calibrate intrinsics riêng và nạp coeffs thực tế vào đây.
    dist = np.zeros(5)

    rdk = Robolink()
    robot = rdk.Item(args.robot)
    if not robot.Valid():
        log.error("Robot '%s' không có trong RoboDK station", args.robot)
        return 1

    estimator = CharucoBoardEstimator()
    session = CalibrationSession(estimator)

    log.info("=" * 60)
    log.info("Hand-eye calibration — di chuyển robot rồi gõ ENTER để ghi pose.")
    log.info("Gõ 's' để giải (cần >=10 pose), 'q' để huỷ.")
    log.info("=" * 60)

    try:
        while True:
            cmd = input(f"[pose {session.num_poses}] ENTER=ghi / s=giải / q=huỷ: ")
            cmd = cmd.strip().lower()
            if cmd == "q":
                log.info("Huỷ calibration.")
                return 1
            if cmd == "s":
                break
            # Ghi 1 pose.
            rgb, _ = camera.get_frame()
            if rgb is None:
                log.warning("Không lấy được frame, thử lại.")
                continue
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            T_gripper2base = robodk_pose_to_matrix_m(robot.Pose())
            if session.capture_pose(gray, T_gripper2base, K, dist):
                log.info("→ Đã ghi pose #%d", session.num_poses)
    finally:
        camera.stop()

    # ─── Giải + lưu ───
    T_BC_m = session.solve(method=args.method)
    T_BC_mm = T_BC_m.copy()
    T_BC_mm[:3, 3] *= 1000.0

    out_path = PROJECT_ROOT / args.output
    save_calibration(out_path, T_BC_mm)
    log.info("Lưu T_base_camera (mm) → %s", out_path)
    log.info("Camera tại base frame: %s mm", T_BC_mm[:3, 3].round(2))
    log.info("Kiểm chứng tiếp bằng touch test (mục 6.4).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
