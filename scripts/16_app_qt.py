#!/usr/bin/env python
"""
16_app_qt.py
────────────
PyQt6 + pyvistaqt (VTK) launcher cho GP7 Digital Twin — industrial-standard
stack. Same kinematics (FK/IK verified against RoboDK 0.00 mm).

Usage:
    python scripts/16_app_qt.py
    python scripts/16_app_qt.py --config config/cell_layout_real.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtWidgets import QApplication                       # noqa: E402

from src.cell import CellConfig                                # noqa: E402
from src.orchestrator.viewports.gp7_app_qt import GP7AppQt     # noqa: E402
from src.utils import setup_logging                            # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("GP7 Digital Twin app — PyQt6 + pyvistaqt (VTK)."))
    p.add_argument("--config", default="config/cell_layout.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(name="gp7-qt", verbose=args.verbose)

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Không tìm thấy cell config: {config_path}", file=sys.stderr)
        return 2
    cell_config = CellConfig.from_yaml(config_path)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")                                     # cross-platform consistent

    win = GP7AppQt(cell_config=cell_config, project_root=PROJECT_ROOT)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
