#!/usr/bin/env python
"""
gen_primitive_meshes.py
───────────────────────
Generate primitive STL meshes for the cell:
  - worktable.stl  : table 600 × 400 × 500 mm (tabletop + 4 legs)
  - pedestal.stl   : column 300 × 300 × 300 mm (top flush with robot mount face Z=300)
  - gripper.stl    : parallel-jaw 2-finger, opening 90mm, height 110mm
                     (designed for bottle / cup / bolt — see make_parallel_gripper)

Purpose: replace downloaded CAD meshes that may include unwanted decoration/accessories
or have incorrect dimensions/units.

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
    """Workbench table: thick tabletop + 4 box legs at the corners.

    Origin at **bottom of legs** (Z=0), centered in X-Y → easy placement at any pose.
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
    height: float = 300.0,
) -> trimesh.Trimesh:
    """Pedestal: rectangular box, origin at the base.

    Height 300mm: pedestal top (Z=300) must align with the robot mount face. Robot J1 at
    world Z=630, base_link mesh offset -330mm → mount face at Z=300. Pedestal taller than
    300mm will clip the robot base (see cell_layout_real.yaml robot.pose).
    """
    box = trimesh.creation.box([width, depth, height])
    box.apply_translation([0, 0, height / 2])
    return box


def make_parallel_gripper(
    palm_width: float = 150.0,        # X: 110 opening + 2×10 finger + 20 slack
    palm_depth: float = 50.0,         # Y: palm body thickness (motor housing)
    palm_height: float = 60.0,        # Z: palm height (pneumatic actuator inside)
    finger_thickness: float = 10.0,   # X: finger thickness (along opening direction)
    finger_depth: float = 220.0,      # Y: finger length (long bar) — > tray length 180mm
    finger_length: float = 40.0,      # Z: fits tray height 25mm + clearance
    finger_inner_gap: float = 110.0,  # X: opening 110mm — fits tray Y=100mm + 5mm clearance/side
) -> trimesh.Trimesh:
    """PNEUMATIC parallel-jaw gripper (grips Galaxy S23 tray across its width).

    **PLACEHOLDER values** — gripper fits tray width (100mm Y):
    opening 110mm > 100mm so jaws span the LONG 180mm (X) edges of the tray with
    5mm clearance per side.

    Default specs:
      - Total height = palm 60 + finger 40 = 100mm
      - Opening max  = 110mm (fits tray width snugly)
      - Stroke       = 55mm per finger
      - Driving      = compressed air (pneumatic)

    Orchestrator uses yaw_offset_deg=90 so gripper jaws align perpendicular to PCA
    major axis (= longest tray dim) → jaws spread along tray width.

    Origin: **at fingertip (TCP)**, mesh extends along -Z toward the flange.
    Sync with cell_layout.yaml: tcp_offset_xyz_mm: [0, 0, 100].
    """
    # Palm at Z=[-total_height, -finger_length], fingers at Z=[-finger_length, 0].
    # STL extends along -Z from origin (fingertip). rotx(pi) in grasp pose flips
    # tool Z → palm world Z = TCP + total_height (ABOVE TCP).
    palm = trimesh.creation.box([palm_width, palm_depth, palm_height])
    palm.apply_translation([0, 0, -finger_length - palm_height / 2])

    finger_center_x = (finger_inner_gap + finger_thickness) / 2
    parts = [palm]
    for sign in (1, -1):
        finger = trimesh.creation.box([finger_thickness, finger_depth, finger_length])
        finger.apply_translation([
            sign * finger_center_x,
            0,
            -finger_length / 2,
        ])
        parts.append(finger)

    return trimesh.util.concatenate(parts)


def make_tray(
    width: float = 180.0,    # X: tray width (Galaxy S23 phone 146mm + margin)
    depth: float = 100.0,    # Y: tray depth
    height: float = 25.0,    # Z: tray height
) -> trimesh.Trimesh:
    """Galaxy S23 phone tray (simplified box, no handle).

    **PLACEHOLDER values** — wide gripper (opening 200mm) grips the two long sides of the
    tray directly, no handle needed. User should measure the real tray → update params.

    Galaxy S23 dimensions: 146.3 × 70.9 × 7.6mm → 180×100mm tray fits the phone.

    Origin: at **tray bottom center** (Z=0), tray extends +Z → placed at pose Z=table_top
    the tray bottom aligns with the table surface. Top of tray at Z=25.
    """
    box = trimesh.creation.box([width, depth, height])
    box.apply_translation([0, 0, height / 2])
    return box


def make_floor(
    size: float = 3000.0,
    thickness: float = 20.0,
    tile_count_per_side: int = 5,
    gap_mm: float = 5.0,
) -> trimesh.Trimesh:
    """Tiled floor: n×n grid of square tiles with grout gaps between them.

    Each tile is a separate box; concatenated into one mesh. The 5mm gap between tiles
    renders as a grout line (empty gap, dark background). Total size = size × size mm.

    Default: 5×5 grid, tile 596×596mm, gap 5mm → 3000×3000mm total.

    Origin at center, tile top face at Z=0 (place floor at pose Z=0 → floor surface at Z=0).
    """
    n = max(int(tile_count_per_side), 1)
    if n == 1:
        # Degenerate case: single tile = flat plate (preserve backward compatibility)
        plate = trimesh.creation.box([size, size, thickness])
        plate.apply_translation([0, 0, -thickness / 2])
        return plate

    tile_size = (size - gap_mm * (n - 1)) / n
    start = -size / 2 + tile_size / 2
    parts = []
    for i in range(n):
        for j in range(n):
            x = start + i * (tile_size + gap_mm)
            y = start + j * (tile_size + gap_mm)
            tile = trimesh.creation.box([tile_size, tile_size, thickness])
            tile.apply_translation([x, y, -thickness / 2])
            parts.append(tile)
    return trimesh.util.concatenate(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate primitive mesh STLs for the cell.")
    p.add_argument(
        "--only", choices=["worktable", "pedestal", "gripper", "floor", "tray"],
        help="Generate only one mesh; default is to generate all.",
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
        print(f"OK pedestal.stl: 300x300x300 mm, {len(ped.faces)} triangles")

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
        print(f"OK floor.stl: 5x5 tiled grid (596x596mm each, 5mm gaps), 3000x3000mm total, {len(floor.faces)} triangles")

    return 0


if __name__ == "__main__":
    sys.exit(main())
