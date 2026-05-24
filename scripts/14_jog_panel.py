#!/usr/bin/env python
"""
14_jog_panel.py
───────────────
Standalone JOG mode — mở viewport Open3D + Control Panel (style RoboDK) cho phép
JOG robot bằng joint slider hoặc Cartesian button. Không chạy experiment, không
gửi command tới robot thật — sim only.

Usage:
    python scripts/14_jog_panel.py
    python scripts/14_jog_panel.py --config config/cell_layout_real.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cell import CellConfig                                       # noqa: E402
from src.orchestrator.viewports.control_panel import ControlPanel     # noqa: E402
from src.orchestrator.viewports.open3d_gui_sim_robot import O3DGuiSimRobot  # noqa: E402
from src.utils import setup_logging                                   # noqa: E402


def parse_args() -> argparse.Namespace:
    # ASCII-only description so --help works on Windows cp1252 consoles.
    # Full Vietnamese docstring vẫn ở đầu file để dev đọc.
    p = argparse.ArgumentParser(
        description=("Standalone JOG mode: open Open3D viewport + "
                     "RoboDK-style control panel for joint / Cartesian jog "
                     "(sim only)."),
    )
    p.add_argument("--config", default="config/cell_layout.yaml",
                   help="Cell layout YAML. Default: config/cell_layout.yaml.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="DEBUG-level logging.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(name="jog", verbose=args.verbose)

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Không tìm thấy cell config: {config_path}", file=sys.stderr)
        return 2
    cell_config = CellConfig.from_yaml(config_path)

    # Robot base position + home joints lấy từ cell config.
    base_xyz = tuple(cell_config.robot.pose.xyz_mm) \
        if getattr(cell_config.robot, "pose", None) else (0.0, 0.0, 630.0)
    home = list(cell_config.robot.home_joints_deg) \
        if getattr(cell_config.robot, "home_joints_deg", None) else [0.0] * 6

    # Viewport — phải tạo TRƯỚC khi run_gui() (init Application + add window).
    robot = O3DGuiSimRobot(
        base_xyz=base_xyz,
        home_joints=home,
        cell_config=cell_config,
        project_root=PROJECT_ROOT,
    )
    # Control panel — cùng gui.Application, cửa sổ thứ 2.
    _panel = ControlPanel(
        robot=robot,
        model=robot._model,
        cell_config=cell_config,
        home_joints_deg=home,
    )

    try:
        robot.run_gui(message=(
            "JOG mode — kéo slider/nhấn ± X/Y/Z trong panel để jog. "
            "Đóng cửa sổ viewport để thoát."
        ))
    finally:
        try:
            robot.disconnect()
        except Exception:                                              # noqa: BLE001
            pass

    # Force exit để bypass Filament/atexit non-daemon thread (xem 03 _shutdown).
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
