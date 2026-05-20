#!/usr/bin/env python
"""
diagnose_layout.py
──────────────────
Chẩn đoán bố cục cell ĐANG MỞ trong RoboDK — in ra toạ độ world THẬT của
mọi item, thay vì đoán qua screenshot.

Mục đích: trả lời dứt điểm câu hỏi "robot và bàn có đứng trên cùng mặt
phẳng không" bằng số liệu, không phỏng đoán.

Cách dùng (RoboDK GUI phải đang mở + đã chạy build_station.py):
    python scripts/diagnose_layout.py

Script in ra cho từng item:
  - Tên + loại
  - PoseAbs (toạ độ world của origin item)
  - Với item có mesh STL: world Z-min / Z-max (mép thấp/cao nhất)
  - Với robot: world Z của đế (base) + Z của flange ở home joints
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _stl_zbounds(stl_path: Path) -> tuple[float, float] | None:
    """Trả về (z_min, z_max) local của mesh STL, hoặc None nếu không đọc được."""
    try:
        import trimesh
        m = trimesh.load(str(stl_path), force="mesh")
        b = m.bounding_box.bounds
        return float(b[0, 2]), float(b[1, 2])
    except Exception as e:  # noqa: BLE001
        print(f"    (không đọc được STL: {e})")
        return None


def main() -> int:
    try:
        from robodk.robolink import ITEM_TYPE_ROBOT, Robolink
    except ImportError:
        print("✗ Chưa cài robodk: pip install robodk")
        return 1

    rdk = Robolink()
    items = rdk.ItemList()
    if not items:
        print("✗ Station trống — chạy scripts/build_station.py trước.")
        return 1

    print("=" * 66)
    print(" CHẨN ĐOÁN BỐ CỤC CELL — toạ độ world thật")
    print("=" * 66)

    for it in items:
        try:
            name = it.Name()
            pose = it.PoseAbs()
            x, y, z = pose.Pos()
            print(f"\n• {name}")
            print(f"    origin world : X={x:8.1f}  Y={y:8.1f}  Z={z:8.1f}  mm")
        except Exception as e:  # noqa: BLE001
            print(f"    (lỗi đọc item: {e})")
            continue

        # Robot: in thêm Z flange ở joints hiện tại. Chỉ gọi Joints()/SolveFK()
        # khi item đúng là robot — tránh RoboDK in "Invalid item provided".
        try:
            if it.Type() == ITEM_TYPE_ROBOT:
                fk = it.SolveFK(it.Joints())
                fz = fk.Pos()[2]
                print(f"    → ROBOT: đế (base) ở world Z={z:.1f} mm")
                print(f"            flange ở world Z={fz:.1f} mm (joints hiện tại)")
        except Exception as e:  # noqa: BLE001
            print(f"    (không đọc được FK robot: {e})")


    # Đối chiếu STL bbox để biết mép đáy thật của bàn / vật
    print("\n" + "-" * 66)
    print(" MÉP ĐÁY THẬT (origin world Z + STL z-min)")
    print("-" * 66)
    models = PROJECT_ROOT / "models"
    for stl_name, label in [
        ("worktable.stl", "Worktable"),
        ("pedestal.stl", "Pedestal"),
        ("gripper.stl", "Gripper"),
    ]:
        stl = models / stl_name
        if not stl.exists():
            continue
        zb = _stl_zbounds(stl)
        if zb:
            print(f"  {label:12s} STL local Z: {zb[0]:.1f} → {zb[1]:.1f}  "
                  f"(cao {zb[1] - zb[0]:.1f} mm)")

    print("\nGhi chú: nếu Worktable origin world Z = 0 và STL z-min = 0")
    print("→ đáy bàn chạm sàn. So với robot đế world Z để biết có đồng phẳng.")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
