#!/usr/bin/env python
"""
14_jog_panel.py
───────────────
Standalone JOG mode — opens Open3D viewport + Control Panel (RoboDK style) allowing
jogging the robot via joint sliders or Cartesian buttons. No experiment runs, no
commands sent to the real robot — sim only.

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
    # Full docstring is at the top of the file for developer reference.
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
        print(f"Cell config not found: {config_path}", file=sys.stderr)
        return 2
    cell_config = CellConfig.from_yaml(config_path)

    # Robot base position + home joints loaded from cell config.
    base_xyz = tuple(cell_config.robot.pose.xyz_mm) \
        if getattr(cell_config.robot, "pose", None) else (0.0, 0.0, 630.0)
    home = list(cell_config.robot.home_joints_deg) \
        if getattr(cell_config.robot, "home_joints_deg", None) else [0.0] * 6

    # Viewport — must be created BEFORE run_gui() (initialises Application + adds window).
    robot = O3DGuiSimRobot(
        base_xyz=base_xyz,
        home_joints=home,
        cell_config=cell_config,
        project_root=PROJECT_ROOT,
    )
    # Control panel — same gui.Application, second window.
    _panel = ControlPanel(
        robot=robot,
        model=robot._model,
        cell_config=cell_config,
        home_joints_deg=home,
    )

    try:
        robot.run_gui(message=(
            "JOG mode — drag sliders or press ± X/Y/Z in the panel to jog. "
            "Close the viewport window to exit."
        ))
    finally:
        try:
            robot.disconnect()
        except Exception:                                              # noqa: BLE001
            pass

    # Force exit to bypass Filament/atexit non-daemon thread (see 03 _shutdown).
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
