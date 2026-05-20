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
    palm_width: float = 200.0,        # X: 150 opening + 2×12 finger + slack
    palm_depth: float = 50.0,         # Y: bề dày thân
    palm_height: float = 40.0,        # Z: cao của palm
    finger_thickness: float = 12.0,   # X: đủ mảnh đâm qua quai cup
    finger_depth: float = 30.0,       # Y: bề dày ngón
    finger_length: float = 50.0,      # Z: đủ ôm top phần lớn object, không xuyên bàn khi gắp bolt
    finger_inner_gap: float = 150.0,  # X: opening 150mm — cover cup width 137mm + slack
) -> trimesh.Trimesh:
    """Gripper parallel-jaw 2 ngón (universal — đủ cho cả 3 object class).

    Thiết kế cho 3 vật pick-and-place GP7:
      - bottle (68×68×210): opening 150mm > 68mm ✓, finger 50mm ôm top thân
      - cup    (167×136×100): opening 150 ≥ width 137 ✓ (gắp body trực tiếp,
                              không phải qua quai); finger 50mm phù hợp h=100mm
      - bolt   (9×8×24): opening 150mm rộng — chỉ visual demo, orchestrator
                         dùng adaptive grasp_depth_offset để fingertip không
                         xuyên bàn (~5mm trên table)

    Origin: **tại fingertip (TCP)**, mesh extend theo -Z về phía flange.
    Tổng cao 90mm (palm 40 + finger 50). Cần update cell_layout.yaml:
      tcp_offset_xyz_mm: [0, 0, 90]  (was 110 cho gripper cũ)
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
        "--only", choices=["worktable", "pedestal", "gripper", "floor"],
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

    targets = {args.only} if args.only else {"worktable", "pedestal", "gripper", "floor"}

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
        print(f"OK gripper.stl: parallel-jaw 2-finger, opening 150mm, total height 90mm, {len(grip.faces)} triangles")

    if "floor" in targets:
        floor = make_floor()
        path = MODELS_DIR / "floor.stl"
        floor.export(str(path))
        print(f"OK floor.stl: 3000x3000 mm, top surface at Z=0, {len(floor.faces)} triangles")

    return 0


if __name__ == "__main__":
    sys.exit(main())
