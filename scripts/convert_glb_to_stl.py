#!/usr/bin/env python
"""
convert_glb_to_stl.py
─────────────────────
Convert file mesh .glb (glTF Binary) sang .stl để load vào RoboDK.

Xử lý 3 chênh lệch chuẩn giữa glTF và RoboDK:
  1. RoboDK chỉ load .stl/.iges/.step/.wrl — không load .glb.
  2. glTF dùng đơn vị MÉT, RoboDK dùng MM   → scale ×1000.
  3. glTF Y-up convention, RoboDK Z-up      → xoay +90° quanh trục X.
  Đồng thời flatten node tree thành 1 mesh duy nhất.

Usage:
    python scripts/convert_glb_to_stl.py models/worktable.glb
    python scripts/convert_glb_to_stl.py models/worktable.glb models/worktable.stl
    python scripts/convert_glb_to_stl.py worktable.glb --no-scale  # GLB đã ở mm
    python scripts/convert_glb_to_stl.py worktable.glb --no-axis   # giữ Y-up
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
        glb_path: Đường dẫn file .glb đầu vào.
        stl_path: Đường dẫn file .stl đầu ra.
        scale_factor: Hệ số nhân toạ độ. Default 1000.0 (m → mm). Đặt 1.0
            nếu GLB đã ở mm. Đặt giá trị khác cho normalized model.
        yup_to_zup: Nếu True (default), xoay +90° quanh trục X để chuyển
            convention glTF Y-up sang RoboDK Z-up.

    Raises:
        FileNotFoundError: File đầu vào không tồn tại.
        ValueError: GLB không chứa geometry hợp lệ.
    """
    if not glb_path.exists():
        raise FileNotFoundError(f"GLB không tồn tại: {glb_path}")

    # force="mesh" → trimesh tự apply mọi node transform rồi gộp tất cả mesh
    # trong scene thành 1 Trimesh duy nhất. Tránh phải tự iterate scene tree.
    mesh = trimesh.load(str(glb_path), force="mesh")

    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(
            f"{glb_path} không chứa mesh hợp lệ "
            f"(scene rỗng hoặc chỉ có point cloud/curve)"
        )

    bbox_pre = mesh.bounding_box.extents.copy()

    if scale_factor != 1.0:
        mesh.apply_scale(scale_factor)

    if yup_to_zup:
        # Xoay +90° quanh X: +Y_old (up) → +Z_new (up), +Z_old → -Y_new.
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
        description="Convert .glb -> .stl cho RoboDK (flatten + scale + Y-up to Z-up).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=Path, help="File .glb đầu vào")
    parser.add_argument(
        "output", type=Path, nargs="?",
        help="File .stl đầu ra (default: cùng path, đổi extension)",
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--no-scale", action="store_true",
        help="Không scale (factor = 1). Dùng nếu GLB đã ở mm.",
    )
    g.add_argument(
        "--scale", type=float, default=None,
        help="Custom scale factor. Default 1000 (m -> mm).",
    )
    parser.add_argument(
        "--no-axis", action="store_true",
        help="Không xoay Y-up -> Z-up. Dùng nếu GLB đã ở Z-up.",
    )
    return parser.parse_args()


def main() -> int:
    # Console Windows cp1252 không in được ký tự ngoài ASCII → ép stdout UTF-8.
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
