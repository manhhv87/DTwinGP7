#!/usr/bin/env python
"""
02_run_calibration.py
─────────────────────
Hand-eye calibration eye-to-hand cho D455 + GP7 (xem mục 6 tài liệu).

Quy trình:
    1. Robot giữ ChArUco board trên end-effector
    2. Di chuyển robot tới 25-30 pose đa dạng (rotation ±30°) bằng teach pendant
    3. Tại mỗi pose: gõ ENTER → script đọc joints từ YRC1000 qua HSE, tính
       T_gripper2base bằng URDF FK pure-Python, kết hợp frame ChArUco
    4. Gõ 's' để giải → lưu config/calibration/T_base_camera.npy

Usage:
    python scripts/02_run_calibration.py --hse-ip 192.168.1.100
    python scripts/02_run_calibration.py --method park --hse-ip 192.168.1.100

Lưu ý:
  - YRC1000 phải bật High-Speed Ethernet Server function (Maintenance mode)
  - Robot mode: TEACH (không cần REMOTE — chỉ đọc joints, không gửi motion)
  - URDF chain forward kinematics đã verify match RoboDK SolveFK 0.00mm
    (xem scripts/13_verify_vs_robodk.py)
  - ĐƠN VỊ: toàn pipeline dùng mm. FK gripper2base trả mm; estimate_pose đã quy
    đổi ChArUco target2cam mét→mm. solve_hand_eye yêu cầu 2 input CÙNG đơn vị →
    giữ mm cả hai, output T_BC cũng mm → lưu thẳng (KHÔNG đổi mét).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import CalibrationSession, CharucoBoardEstimator  # noqa: E402
from src.cell import CellConfig  # noqa: E402
from src.orchestrator.backends.motoman_hse import MotomanHSEBackend  # noqa: E402
from src.orchestrator.coord_conv import save_calibration  # noqa: E402
from src.orchestrator.kinematics.urdf_chain import (  # noqa: E402
    forward_kinematics_urdf,
    gp7_urdf,
)
from src.utils import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell-config", default="config/cell_layout_real.yaml",
                        help="Cell config (lấy robot base pose + tool offset).")
    parser.add_argument("--hse-ip", default=None,
                        help="IP YRC1000. Default: từ robot_connection.ip trong cell config.")
    # Mặc định "park": "tsai" sai với camera nhìn xuống (xoay ~180°).
    parser.add_argument("--method", default="park",
                        choices=["park", "horaud", "daniilidis", "andreff", "tsai"])
    parser.add_argument("--output", default="config/calibration/T_base_camera.npy")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging("calibration", log_file=PROJECT_ROOT / "logs/calibration.log")

    try:
        from src.perception.camera import D455Camera
        import cv2
    except ImportError as e:
        log.error("Thiếu dependency: %s", e)
        return 1

    cell_config = CellConfig.from_yaml(PROJECT_ROOT / args.cell_config)
    hse_ip = args.hse_ip or cell_config.robot_connection.ip
    if not hse_ip:
        log.error("Cần IP YRC1000 (--hse-ip hoặc robot_connection.ip trong cell).")
        return 1

    tool_offset_mm = 0.0
    if hasattr(cell_config, "tool") and cell_config.tool:
        tcp = getattr(cell_config.tool, "tcp_offset_mm", None)
        if tcp:
            tool_offset_mm = float(tcp[2])

    urdf_model = gp7_urdf(
        base_xyz_mm=tuple(cell_config.robot.pose.xyz_mm),
        tool_offset_mm=tool_offset_mm,
    )

    camera = D455Camera()
    K = np.array([
        [camera.intrinsics["fx"], 0, camera.intrinsics["ppx"]],
        [0, camera.intrinsics["fy"], camera.intrinsics["ppy"]],
        [0, 0, 1],
    ])
    # D455 color stream đã rectified — dist=0.
    dist = np.zeros(5)

    backend = MotomanHSEBackend(ip=hse_ip)
    backend.connect()
    if not backend.Valid():
        log.error("HSE heartbeat fail — kiểm tra ping %s + HSE Server function", hse_ip)
        backend.disconnect()
        return 1
    log.info("HSE connected: %s", hse_ip)

    estimator = CharucoBoardEstimator()
    session = CalibrationSession(estimator)

    log.info("=" * 60)
    log.info("Hand-eye calibration — di chuyển robot bằng TP rồi gõ ENTER để ghi pose.")
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
            rgb, _ = camera.get_frame()
            if rgb is None:
                log.warning("Không lấy được frame, thử lại.")
                continue
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

            # Đọc joints từ YRC1000 qua HSE → FK pure-Python → T_gripper2base (mm)
            joints_deg = backend.Joints()
            joints_rad = [np.deg2rad(j) for j in joints_deg]
            # FK trả mm; session + estimate_pose đều mm → KHÔNG đổi sang mét
            # (solve_hand_eye yêu cầu gripper2base & target2cam CÙNG đơn vị).
            T_gripper2base = forward_kinematics_urdf(urdf_model, joints_rad)

            if session.capture_pose(gray, T_gripper2base, K, dist):
                log.info("→ Đã ghi pose #%d (joints=%s)",
                         session.num_poses,
                         [round(j, 1) for j in joints_deg])
    finally:
        camera.stop()
        backend.disconnect()

    # ─── Giải + lưu ───
    # Input toàn mm → solve trả T_BC mm (cùng đơn vị input) → lưu thẳng.
    T_BC_mm = session.solve(method=args.method)

    out_path = PROJECT_ROOT / args.output
    save_calibration(out_path, T_BC_mm)
    log.info("Lưu T_base_camera (mm) → %s", out_path)
    log.info("Camera tại base frame: %s mm", T_BC_mm[:3, 3].round(2))
    log.info("Kiểm chứng tiếp bằng touch test (mục 6.4).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
