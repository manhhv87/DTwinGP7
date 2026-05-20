#!/usr/bin/env python
"""
demo_reachability.py
────────────────────
Demo đóng góp C2 — lớp an toàn dựa trên digital twin (RoboDK).

Chứng minh hệ thống dùng RoboDK MoveJ_Test (digital twin) để kiểm tra
reachability của robot TRƯỚC khi gắp vật lý — đây là cổng an toàn ngăn lệnh
gắp tới vị trí ngoài tầm với / va chạm.

Script test một tập pose (trong tầm + ngoài tầm) và in quyết định của twin,
minh hoạ rằng orchestrator (qua _is_reachable) sẽ loại đúng pose ngoài tầm.

TIẾT KIỆM API (RoboDK Free quota): chỉ load DUY NHẤT robot (~5 calls) + ~1
call/pose. Không dựng full cell. Tổng ~11 calls cho 6 pose → fit budget Free.

Usage:
    # Mở RoboDK GUI (empty) hoặc để script tự khởi động:
    python scripts/demo_reachability.py
    python scripts/demo_reachability.py --robot-z 630
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.hand_eye_solver import invert_transform  # noqa: E402
from src.cell.pose_utils import (  # noqa: E402
    make_homogeneous,
    matrix_to_robodk_pose,
    robodk_pose_to_matrix,
)
from src.orchestrator.coord_conv import make_grasp_pose  # noqa: E402

DEFAULT_LIBRARY = Path("C:/RoboDK/Library")

# Tập pose test (world frame, mm). Mỗi tuple: (nhãn, x, y, z, kỳ vọng).
# Kỳ vọng do người thiết kế đoán theo reach GP7 ~927mm từ J1; twin sẽ xác nhận.
TEST_POSES = [
    ("Giữa bàn (pick zone)",      700, 0, 500, "reachable"),
    ("Gần robot",                 400, 0, 600, "reachable"),
    ("Vùng đặt (place)",          700, 200, 700, "reachable"),
    ("Xa ngoài tầm (X=1500)",     1500, 0, 500, "UNREACHABLE"),
    ("Quá cao (Z=1800)",          700, 0, 1800, "UNREACHABLE"),
    ("Sau lưng robot (X=-900)",   -900, 0, 500, "UNREACHABLE"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--robot-z", type=float, default=630.0,
                   help="Chiều cao J1 robot trong world (mm). Mặc định 630 (trên pedestal).")
    p.add_argument("--robot-name", default="Yaskawa GP7")
    p.add_argument("--library", default=None, help="RoboDK Library path.")
    return p.parse_args()


def main() -> int:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    args = parse_args()

    try:
        from robodk.robolink import ITEM_TYPE_ROBOT, Robolink
    except ImportError:
        print("✗ Chưa cài robodk: pip install robodk")
        return 1

    rdk = Robolink()

    # ─── Load DUY NHẤT robot (tiết kiệm API) ───
    robot = rdk.Item(args.robot_name, ITEM_TYPE_ROBOT)
    if not robot.Valid():
        library = Path(args.library) if args.library else DEFAULT_LIBRARY
        robot_file = library / "Yaskawa-GP7.robot"
        if not robot_file.exists():
            matches = list(library.rglob("Yaskawa-GP7.robot"))
            if not matches:
                print(f"✗ Không tìm thấy Yaskawa-GP7.robot trong {library}")
                return 1
            robot_file = matches[0]
        robot = rdk.AddFile(str(robot_file))
        robot.setName(args.robot_name)
        parent = robot.Parent()
        if parent and parent.Valid() and parent.item != robot.item:
            parent.setPose(matrix_to_robodk_pose(make_homogeneous([0, 0, args.robot_z])))
        robot.setJoints([0, 0, 0, 0, 0, 0])
        print(f"✓ Loaded {args.robot_name} tại world Z={args.robot_z}")
    else:
        print(f"✓ Dùng robot '{args.robot_name}' đã có trong station")

    # ─── World → parent frame conversion (giống Orchestrator) ───
    T_world_to_parent = np.eye(4)
    parent = robot.Parent()
    if parent and parent.Valid() and parent.item != robot.item:
        T_world_to_parent = invert_transform(robodk_pose_to_matrix(parent.PoseAbs()))

    joints_now = robot.Joints()

    # ─── Test từng pose ───
    print("\n" + "=" * 72)
    print(" DEMO C2 — KIỂM TRA REACHABILITY QUA DIGITAL TWIN (RoboDK MoveJ_Test)")
    print("=" * 72)
    print(f"{'Pose':<28}{'world (x,y,z)':<22}{'twin':<14}{'kỳ vọng'}")
    print("-" * 72)

    n_correct = 0
    for label, x, y, z, expected in TEST_POSES:
        grasp_T = make_grasp_pose(np.array([x, y, z], dtype=float), 0.0)
        target_parent = T_world_to_parent @ grasp_T
        try:
            code = robot.MoveJ_Test(joints_now, matrix_to_robodk_pose(target_parent))
        except Exception as e:  # noqa: BLE001
            print(f"{label:<28}{f'({x},{y},{z})':<22}{'LỖI':<14}{e}")
            continue
        # code: 0 = OK, >0 = collision (vẫn reach), <0 = ngoài tầm
        twin_reachable = code >= 0
        verdict = "reachable" if twin_reachable else "UNREACHABLE"
        match = "✓" if (verdict == expected) else "✗ LỆCH"
        if verdict == expected:
            n_correct += 1
        print(f"{label:<28}{f'({x},{y},{z})':<22}{verdict + f' ({code})':<14}{expected}  {match}")

    print("-" * 72)
    print(f"Twin khớp kỳ vọng: {n_correct}/{len(TEST_POSES)}")
    print("=" * 72)
    print("\nÝ nghĩa: orchestrator gọi đúng cổng kiểm tra này (qua _is_reachable)")
    print("TRƯỚC mỗi lệnh gắp → loại pose ngoài tầm, không bao giờ ra lệnh gắp lỗi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
