#!/usr/bin/env python
"""
15_app.py
─────────
Launcher for the GP7 Digital Twin **unified app**: single window with a
3D scene + side panel tabs (Jog / View / Cell / Run).

Unlike `14_jog_panel.py` (which opened 2 Open3D windows — not stable on Windows),
this script uses `gui.Window + SceneWidget + TabControl` → 1 window, all
controls live inside the same window panel.

Usage:
    python scripts/15_app.py
    python scripts/15_app.py --config config/cell_layout_real.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cell import CellConfig                                  # noqa: E402
from src.orchestrator.viewports.gp7_app import GP7App            # noqa: E402
from src.utils import setup_logging                              # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Unified GP7 Digital Twin app: single window with 3D "
                     "scene + tabbed side panel (Jog / View / Cell / Run). "
                     "Sim only."),
    )
    p.add_argument("--config", default="config/cell_layout.yaml")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="DEBUG-level logging.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(name="gp7-app", verbose=args.verbose)

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Cell config not found: {config_path}", file=sys.stderr)
        return 2
    cell_config = CellConfig.from_yaml(config_path)

    app = GP7App(cell_config=cell_config, project_root=PROJECT_ROOT)
    try:
        app.run()
    finally:
        # Force exit to bypass Filament/atexit non-daemon thread.
        os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
