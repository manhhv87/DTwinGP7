#!/usr/bin/env python
"""
gen_primitive_meshes.py
───────────────────────
Generate các mesh STL nguyên bản (primitive) cho cell:
  - worktable.stl  : bàn 600 × 400 × 500 mm (tabletop + 4 chân)
  - pedestal.stl   : cột 300 × 300 × 500 mm (cho hướng pedestal)
  - gripper.stl    : parallel-jaw 2 ngón, opening 90mm, cao 110mm
                     (thiết kế cho bottle / cup / bolt — xem make_parallel_gripper)

Mục đích: thay thế các mesh CAD download có thể chứa decoration/accessory
thừa, hoặc kích thước/đơn vị không phù hợp.

Usage:
    python scripts/gen_primitive_meshes.py
    python scripts/gen_primitive_meshes.py --only worktable
    python scripts/gen_primitive_meshes.py --table-size 800 600 700

Dependency:
    pip install trimesh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import trimesh

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def make_worktable(
    width_x: float = 600.0,
    depth_y: float = 400.0,
    height_z: float = 500.0,
    top_thickness: float = 30.0,
    leg_size: float = 50.0,
) -> trimesh.Trimesh:
    """Bàn workbench: tabletop dày + 4 chân hình hộp ở 4 góc.

    Origin ở **đáy chân bàn** (Z=0), center theo X-Y → bàn dễ đặt tại
    pose nào cũng trực quan.
    """
    top = trimesh.creation.box([width_x, depth_y, top_thickness])
    top.apply_translation([0, 0, height_z - top_thickness / 2])

    leg_h = height_z - top_thickness
    leg_offset_x = (width_x - leg_size) / 2
    leg_offset_y = (depth_y - leg_size) / 2

    parts = [top]
    for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        leg = trimesh.creation.box([leg_size, leg_size, leg_h])
        leg.apply_translation([sx * leg_offset_x, sy * leg_offset_y, leg_h / 2])
        parts.append(leg)

    return trimesh.util.concatenate(parts)


def make_pedestal(
    width: float = 300.0,
    depth: float = 300.0,
    height: float = 500.0,
) -> trimesh.Trimesh:
    """Pedestal: hộp chữ nhật, origin ở đáy."""
    box = trimesh.creation.box([width, depth, height])
    box.apply_translation([0, 0, height / 2])
    return box


def make_parallel_gripper(
    palm_width: float = 150.0,        # X: 110 opening + 2×10 finger + 20 slack
    palm_depth: float = 50.0,         # Y: bề dày thân
    palm_height: float = 60.0,        # Z: cao của palm (chứa motor khí nén)
    finger_thickness: float = 10.0,   # X: bề dày custom finger
    finger_depth: float = 40.0,       # Y: bề dày ngón
    finger_length: float = 40.0,      # Z: vừa ôm chiều cao tray 25mm + clearance
    finger_inner_gap: float = 110.0,  # X: opening 110mm — vừa khay Y=100mm + 5mm clearance/bên
) -> trimesh.Trimesh:
    """Gripper PNEUMATIC parallel-jaw (kẹp chiều rộng khay Galaxy S23).

    **PLACEHOLDER values** — gripper khít với chiều rộng tray (100mm Y):
    opening 110mm > 100mm để jaws ôm 2 cạnh DÀI 180mm (X) của khay với
    clearance 5mm/bên.

    Default specs:
      - Total height = palm 60 + finger 40 = 100mm
      - Opening max  = 110mm (vừa khít chiều rộng khay)
      - Stroke       = 55mm per finger
      - Driving      = compressed air (pneumatic)

    Orchestrator dùng yaw_offset_deg=90 để gripper jaws aligns vuông góc PCA
    major axis (= longest tray dim) → jaws spread theo chiều rộng khay.

    Origin: **tại fingertip (TCP)**, mesh extend theo -Z về phía flange.
    Cần đồng bộ cell_layout.yaml: tcp_offset_xyz_mm: [0, 0, 100].
    """
    # Palm tại Z = -(finger_length + palm_height/2 ... -finger_length)
    palm = trimesh.creation.box([palm_width, palm_depth, palm_height])
    palm.apply_translation([0, 0, -finger_length - palm_height / 2])

    # finger center offset từ trục tool: nửa khoảng hở trong + nửa bề dày ngón
    finger_center_x = (finger_inner_gap + finger_thickness) / 2
    parts = [palm]
    for sign in (1, -1):
        finger = trimesh.creation.box([finger_thickness, finger_depth, finger_length])
        finger.apply_translation([
            sign * finger_center_x,
            0,
            -finger_length / 2,         # finger từ Z=-finger_length đến Z=0
        ])
        parts.append(finger)

    return trimesh.util.concatenate(parts)


def make_tray(
    width: float = 180.0,    # X: chiều rộng khay (Galaxy S23 phone 146mm + lề)
    depth: float = 100.0,    # Y: chiều sâu khay
    height: float = 25.0,    # Z: chiều cao khay
) -> trimesh.Trimesh:
    """Khay đựng điện thoại Galaxy S23 (simplified box, không handle).

    **PLACEHOLDER values** — gripper rộng (opening 200mm) kẹp 2 bên dài của
    khay trực tiếp, không cần handle. User đo khay thực tế → update params.

    Galaxy S23 dimensions: 146.3 × 70.9 × 7.6mm → khay 180×100mm vừa chứa.

    Origin: ở **TÂM ĐÁY** khay (Z=0), khay extend +Z → đặt tại pose Z=table_top
    thì đáy khay trùng mặt bàn. Top of tray at Z=25.
    """
    box = trimesh.creation.box([width, depth, height])
    box.apply_translation([0, 0, height / 2])
    return box


def make_floor(size: float = 3000.0, thickness: float = 20.0) -> trimesh.Trimesh:
    """Sàn nhà: tấm vuông mỏng. Origin ở tâm, mặt TRÊN ở Z=0 (tấm nằm dưới sàn)
    → đặt floor tại pose Z=0 thì mặt sàn trùng đúng Z=0, mọi vật đứng lên trên.
    """
    plate = trimesh.creation.box([size, size, thickness])
    plate.apply_translation([0, 0, -thickness / 2])
    return plate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate primitive mesh STLs cho cell.")
    p.add_argument(
        "--only", choices=["worktable", "pedestal", "gripper", "floor", "tray"],
        help="Chỉ generate 1 mesh; mặc định generate tất cả.",
    )
    p.add_argument(
        "--table-size", nargs=3, type=float, metavar=("W", "D", "H"),
        default=[600.0, 400.0, 500.0],
        help="Worktable size W×D×H mm (default 600×400×500).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    targets = {args.only} if args.only else {"worktable", "pedestal", "gripper", "floor", "tray"}

    if "worktable" in targets:
        w, d, h = args.table_size
        table = make_worktable(w, d, h)
        path = MODELS_DIR / "worktable.stl"
        table.export(str(path))
        print(f"OK worktable.stl: {w:.0f}x{d:.0f}x{h:.0f} mm, {len(table.faces)} triangles")

    if "pedestal" in targets:
        ped = make_pedestal()
        path = MODELS_DIR / "pedestal.stl"
        ped.export(str(path))
        print(f"OK pedestal.stl: 300x300x500 mm, {len(ped.faces)} triangles")

    if "gripper" in targets:
        grip = make_parallel_gripper()
        path = MODELS_DIR / "gripper.stl"
        grip.export(str(path))
        print(f"OK gripper.stl: pneumatic parallel-jaw, opening 110mm (fits tray Y=100mm), total height 100mm, {len(grip.faces)} triangles")

    if "tray" in targets:
        tray = make_tray()
        (MODELS_DIR / "objects").mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / "objects" / "tray.stl"
        tray.export(str(path))
        print(f"OK objects/tray.stl: 180x100x25mm (Galaxy S23 tray, no handle), {len(tray.faces)} triangles")

    if "floor" in targets:
        floor = make_floor()
        path = MODELS_DIR / "floor.stl"
        floor.export(str(path))
        print(f"OK floor.stl: 3000x3000 mm, top surface at Z=0, {len(floor.faces)} triangles")

    return 0


if __name__ == "__main__":
    sys.exit(main())
