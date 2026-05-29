#!/usr/bin/env python
"""
16_app_qt.py
────────────
PyQt6 + pyvistaqt (VTK) launcher cho GP7 Digital Twin — industrial-standard
stack. Same kinematics (FK/IK verified against RoboDK 0.00 mm).

Usage:
    python scripts/16_app_qt.py
    python scripts/16_app_qt.py --config config/cell_layout_real.yaml
    python scripts/16_app_qt.py --program examples/sample_program.json
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
from src.orchestrator.viewports.qt_theme import apply_dark_theme  # noqa: E402
from src.utils import setup_logging                            # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Yaskawa GP7 Programming — PyQt6 + pyvistaqt (VTK).\n"
                     "Cold start (no args) ⇒ app trống; user load robot/cell "
                     "qua menu File. Hoặc dùng --config để auto-load 1 cell."))
    p.add_argument("--config", default=None,
                    help="Cell YAML file (optional). Bỏ qua thì app start trống.")
    p.add_argument("--program", default=None,
                    help="Program JSON (optional). Auto-load + show Program panel "
                         "+ load robot GP7 → sẵn sàng bấm Run Sim. "
                         "Vd: examples/sample_program.json")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    # Console Windows cp1252 không in được ký tự Unicode (⇒, dấu tiếng Việt)
    # trong --help / messages → ép stdout/stderr UTF-8.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    args = parse_args()
    setup_logging(name="gp7-qt", verbose=args.verbose)

    cell_config = None
    if args.config:
        config_path = PROJECT_ROOT / args.config
        if not config_path.exists():
            print(f"Không tìm thấy cell config: {config_path}", file=sys.stderr)
            return 2
        cell_config = CellConfig.from_yaml(config_path)

    program_path = None
    if args.program:
        program_path = (PROJECT_ROOT / args.program if not Path(args.program).is_absolute()
                        else Path(args.program))
        if not program_path.exists():
            print(f"Không tìm thấy program file: {program_path}", file=sys.stderr)
            return 2

    app = QApplication(sys.argv)
    apply_dark_theme(app)                                      # Fluent-inspired dark QSS

    win = GP7AppQt(cell_config=cell_config, project_root=PROJECT_ROOT,
                   program_path=program_path)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
