#!/usr/bin/env python
"""
calibration_from_layout.py
──────────────────────────
Sinh file calibration T_base_camera.npy TỪ cell_layout.yaml cho SIM mode.

Trong sim, ta BIẾT chính xác vị trí + hướng của camera ảo trong cell layout
(camera.pose trong YAML). Không cần chạy ChArUco calibration thật.

Script này:
  1. Đọc camera.pose từ cell_layout.yaml
  2. Tính T_BC = pose camera trong WORLD frame (mm + radian)
  3. Lưu ra config/calibration/T_base_camera.npy

Tham chiếu:
  - REAL mode → chạy scripts/02_run_calibration.py với ChArUco board.
  - Orchestrator load file này qua coord_conv.load_calibration().

Convention:
  - T_BC là 4x4 homogeneous matrix, đơn vị translation mm.
  - p_world = T_BC @ p_camera (p_camera đơn vị mm từ postprocess.deproject_pixel).
  - Robot's active reference frame phải = world (đã set ở cell_loader._load_robot
    qua robot.setPoseFrame(WorldRef)) để MoveJ hiểu pose là world coords.

Usage:
    python scripts/calibration_from_layout.py
    python scripts/calibration_from_layout.py --config config/cell_layout_real.yaml
    python scripts/calibration_from_layout.py --output custom/path.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cell.cell_models import CellConfig  # noqa: E402
from src.cell.pose_utils import make_homogeneous  # noqa: E402
from src.orchestrator.coord_conv import save_calibration  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        default="config/cell_layout.yaml",
        help="Đường dẫn cell layout YAML (default: config/cell_layout.yaml)",
    )
    p.add_argument(
        "--output",
        default="config/calibration/T_base_camera.npy",
        help="Đường dẫn output .npy (default: config/calibration/T_base_camera.npy)",
    )
    return p.parse_args()


def main() -> int:
    # Console Windows mặc định cp1252 → ép UTF-8 để in ✓ + tiếng Việt.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    print(f"Reading cell config: {cfg_path}")
    cfg = CellConfig.from_yaml(cfg_path)
    cam = cfg.camera
    if cam.pose is None:
        print(f"✗ camera.pose không có trong {cfg_path}", file=sys.stderr)
        return 1

    # T_BC = camera pose in world frame (mm + degrees → 4x4 mm).
    T_BC = make_homogeneous(cam.pose.xyz_mm, cam.pose.rpy_deg)

    save_calibration(out_path, T_BC)

    print(f"✓ Saved T_BC ({T_BC.shape}) → {out_path}")
    print(f"  Camera world position (mm): {T_BC[:3, 3].round(1).tolist()}")
    print(f"  Camera world rpy (deg):     {list(cam.pose.rpy_deg)}")
    print()
    print("Sanity: điểm camera (0, 0, z_depth_mm) sẽ chuyển sang world:")
    for z_mm in (0, 100, 700):
        p_cam = np.array([0, 0, z_mm, 1])
        p_world = T_BC @ p_cam
        print(f"  z_cam={z_mm:4d}mm → world {p_world[:3].round(1).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
