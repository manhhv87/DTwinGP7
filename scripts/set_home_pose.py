#!/usr/bin/env python
"""
set_home_pose.py
────────────────
Tìm tư thế "home" hợp lý cho GP7 bằng IK rồi ghi vào YAML config.

Vì sao cần: home pose [0,0,0,0,0,0] / [180,0,0,0,0,0] làm cánh tay GP7 duỗi
NGANG (flange ở Z≈485 mm) → robot trông "nằm bẹp" ngang tầm mặt bàn. Thực ra
robot thừa tầm với; chỉ là tư thế home xấu.

Script QUÉT nhiều pose ứng viên (flange nâng cao, gripper chỉ xuống), gọi
RoboDK SolveIK cho từng cái, verify bằng FK, lấy pose KHẢ THI đầu tiên (ưu tiên
flange cao nhất → robot đứng thẳng). Không cần đoán dấu joint.

Cách dùng (RoboDK đang mở + đã chạy build_station.py để có robot trong station):
    python scripts/set_home_pose.py            # quét + move robot + ghi YAML
    python scripts/set_home_pose.py --dry-run  # quét + move robot, KHÔNG ghi file

Sau khi chạy: chạy lại build_station.py để cell áp dụng home pose mới.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cell.pose_utils import make_homogeneous, matrix_to_robodk_pose

CONFIGS = ["config/cell_layout.yaml", "config/cell_layout_real.yaml"]
TCP_OFFSET_Z = 130.0          # khớp gripper.tcp_offset_xyz_mm trong YAML
POS_TOL_MM = 12.0             # sai số cho phép khi verify FK

# Ứng viên tư thế home, ưu tiên flange CAO (robot đứng thẳng) + hướng gripper
# chỉ xuống (sẵn sàng gắp). Quét theo thứ tự này, lấy cái khả thi đầu tiên.
ORIENTATIONS = [
    ("gripper chỉ xuống", [180.0, 0.0, 0.0]),
    ("gripper chéo xuống", [135.0, 0.0, 0.0]),
]
FLANGE_Z = [820.0, 780.0, 740.0, 700.0, 860.0, 880.0]   # ưu tiên trung bình
FLANGE_X = [550.0, 500.0, 600.0, 480.0, 450.0, 400.0]   # ưu tiên gần workspace


def _write_home(path: Path, joints: list[float]) -> bool:
    """Thay dòng `home_joints_deg:` trong file YAML."""
    if not path.exists():
        print(f"  ⚠ Bỏ qua (không có file): {path.name}")
        return False
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    val = "[" + ", ".join(f"{j:g}" for j in joints) + "]"
    for i, ln in enumerate(lines):
        stripped = ln.lstrip()
        if stripped.startswith("home_joints_deg:"):
            indent = ln[: len(ln) - len(stripped)]
            eol = "\n" if ln.endswith("\n") else ""
            lines[i] = (
                f"{indent}home_joints_deg: {val}"
                f"   # tư thế home tính bằng IK (set_home_pose.py){eol}"
            )
            path.write_text("".join(lines), encoding="utf-8")
            print(f"  ✓ {path.name}")
            return True
    print(f"  ⚠ Không tìm thấy 'home_joints_deg:' trong {path.name}")
    return False


def _try_pose(robot, x: float, z: float, rpy: list[float]):
    """Thử SolveIK cho flange tại (x,0,z) với hướng rpy. Verify bằng FK.

    Trả về (joints list, flange_xyz) nếu khả thi, None nếu không.
    """
    target = matrix_to_robodk_pose(make_homogeneous([x, 0.0, z], rpy))
    joints = robot.SolveIK(target)
    jlist = list(joints.list()) if hasattr(joints, "list") else list(joints)
    if len(jlist) != 6:
        return None

    # SolveIK có thể trả về joints "gần đúng" dù không có nghiệm → verify FK.
    fk = robot.SolveFK(jlist)
    fx, fy, fz = fk.Pos()
    err = max(abs(fx - x), abs(fy - 0.0), abs(fz - z))
    if err > POS_TOL_MM:
        return None
    return jlist, (fx, fy, fz)


def main() -> int:
    ap = argparse.ArgumentParser(description="Tìm + đặt home pose cho GP7.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Quét + move robot, không ghi YAML")
    args = ap.parse_args()

    try:
        from robodk.robolink import ITEM_TYPE_ROBOT, Robolink
    except ImportError:
        print("✗ Chưa cài robodk: pip install robodk")
        return 1

    rdk = Robolink()
    robots = rdk.ItemList(ITEM_TYPE_ROBOT)
    if not robots:
        print("✗ Không tìm thấy robot — chạy scripts/build_station.py trước.")
        return 1
    robot = robots[0]

    # Quét: orientation ngoài cùng, rồi z cao→thấp, rồi x.
    found = None
    tried = 0
    for ori_name, rpy in ORIENTATIONS:
        for z in FLANGE_Z:
            for x in FLANGE_X:
                tried += 1
                res = _try_pose(robot, x, z, rpy)
                if res is not None:
                    found = (ori_name, rpy, x, z, *res)
                    break
            if found:
                break
        if found:
            break

    if found is None:
        print(f"✗ Quét {tried} pose, không pose nào khả thi.")
        print("  → Báo lại cho tôi để nới dải FLANGE_X/FLANGE_Z.")
        return 1

    ori_name, rpy, x, z, jlist, (fx, fy, fz) = found
    robot.setJoints(jlist)   # move robot trong GUI để xem ngay

    jrounded = [round(j, 1) for j in jlist]
    print("=" * 62)
    print(f" ✓ Tìm được home pose sau {tried} lần thử")
    print(f"   Hướng       : {ori_name}")
    print(f"   Flange world: X={fx:.0f} Y={fy:.0f} Z={fz:.0f} mm")
    print(f"   Home joints : {jrounded}")
    print("=" * 62)

    if args.dry_run:
        print("(--dry-run) Xem robot trong RoboDK. Ưng thì chạy lại KHÔNG có --dry-run.")
        return 0

    print("Ghi home_joints_deg vào config:")
    for cfg in CONFIGS:
        _write_home(PROJECT_ROOT / cfg, jrounded)
    print("\nXong. Chạy lại 'python scripts/build_station.py' để áp dụng.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
