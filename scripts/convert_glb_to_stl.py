#!/usr/bin/env python
"""
convert_glb_to_stl.py
─────────────────────
Convert a .glb (glTF Binary) mesh file to .stl for loading into RoboDK.

Handles 3 standard differences between glTF and RoboDK:
  1. RoboDK only loads .stl/.iges/.step/.wrl — not .glb.
  2. glTF uses METRES, RoboDK uses MM   → scale ×1000.
  3. glTF Y-up convention, RoboDK Z-up  → rotate +90° around X axis.
  Also flattens the node tree into a single mesh.

Usage:
    python scripts/convert_glb_to_stl.py models/worktable.glb
    python scripts/convert_glb_to_stl.py models/worktable.glb models/worktable.stl
    python scripts/convert_glb_to_stl.py worktable.glb --no-scale  # GLB already in mm
    python scripts/convert_glb_to_stl.py worktable.glb --no-axis   # keep Y-up
    python scripts/convert_glb_to_stl.py worktable.glb --scale 600 # custom factor

Dependency:
    pip install trimesh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh


def convert(
    glb_path: Path,
    stl_path: Path,
    scale_factor: float = 1000.0,
    yup_to_zup: bool = True,
) -> None:
    """Load GLB, flatten scene → single mesh, scale + rotate, export STL.

    Args:
        glb_path: Path to the input .glb file.
        stl_path: Path to the output .stl file.
        scale_factor: Coordinate scale multiplier. Default 1000.0 (m → mm). Set 1.0
            if the GLB is already in mm. Set another value for a normalized model.
        yup_to_zup: If True (default), rotate +90° around X to convert
            glTF Y-up convention to RoboDK Z-up.

    Raises:
        FileNotFoundError: Input file does not exist.
        ValueError: GLB contains no valid geometry.
    """
    if not glb_path.exists():
        raise FileNotFoundError(f"GLB not found: {glb_path}")

    # force="mesh" → trimesh applies all node transforms and merges every mesh
    # in the scene into a single Trimesh. Avoids manual scene-tree iteration.
    mesh = trimesh.load(str(glb_path), force="mesh")

    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(
            f"{glb_path} contains no valid mesh "
            f"(empty scene or only point cloud/curve data)"
        )

    bbox_pre = mesh.bounding_box.extents.copy()

    if scale_factor != 1.0:
        mesh.apply_scale(scale_factor)

    if yup_to_zup:
        # Rotate +90° around X: +Y_old (up) → +Z_new (up), +Z_old → -Y_new.
        # glTF Y-up → RoboDK Z-up.
        T = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        mesh.apply_transform(T)

    stl_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(stl_path))

    bbox_post = mesh.bounding_box.extents
    print(f"OK  {glb_path.name} -> {stl_path}")
    print(f"  triangles : {len(mesh.faces):,}")
    print(f"  input bbox: {bbox_pre.round(3)}  (raw GLB units)")
    print(f"  output    : {bbox_post.round(1)} mm  "
          f"(scale x{scale_factor}, yup_to_zup={yup_to_zup})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert .glb -> .stl for RoboDK (flatten + scale + Y-up to Z-up).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=Path, help="Input .glb file")
    parser.add_argument(
        "output", type=Path, nargs="?",
        help="Output .stl file (default: same path, extension changed)",
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--no-scale", action="store_true",
        help="Skip scaling (factor = 1). Use if GLB is already in mm.",
    )
    g.add_argument(
        "--scale", type=float, default=None,
        help="Custom scale factor. Default 1000 (m -> mm).",
    )
    parser.add_argument(
        "--no-axis", action="store_true",
        help="Skip Y-up -> Z-up rotation. Use if GLB is already in Z-up.",
    )
    return parser.parse_args()


def main() -> int:
    # Windows cp1252 console cannot print non-ASCII characters → force stdout UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    args = parse_args()
    output = args.output or args.input.with_suffix(".stl")

    if args.no_scale:
        scale = 1.0
    elif args.scale is not None:
        scale = args.scale
    else:
        scale = 1000.0

    try:
        convert(args.input, output, scale_factor=scale, yup_to_zup=not args.no_axis)
        return 0
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
