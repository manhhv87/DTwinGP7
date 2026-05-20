#!/usr/bin/env python
"""
build_station.py
────────────────
CLI entry point để dựng RoboDK station từ config YAML.

Usage:
    python scripts/build_station.py                                  # default config
    python scripts/build_station.py --config config/cell_layout.yaml
    python scripts/build_station.py --config config/cell_layout_real.yaml
    python scripts/build_station.py --no-clear                       # giữ items hiện tại
    python scripts/build_station.py --verbose

Exit codes:
    0 — Cell built successfully
    1 — Connection / runtime error
    2 — Config validation error
    3 — Missing files
"""
from __future__ import annotations

import argparse
import logging
import struct
import sys
from pathlib import Path

# Ensure src/ là trong sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.cell import (  # noqa: E402
    CellConfig,
    CellLoader,
    InvalidConfigError,
    MissingMeshError,
    MissingRobotError,
    RoboDKConnectionError,
    RobotConnectionError,
)


def setup_logging(verbose: bool = False) -> None:
    """Cấu hình logging."""
    # Windows console mặc định cp1252 → không encode được ✓ + tiếng Việt.
    # Ép stdout/stderr sang UTF-8 trước khi gắn StreamHandler.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "build_station.log", encoding="utf-8"),
    ]

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dựng RoboDK cell từ YAML config (code-first approach).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/cell_layout.yaml",
        help="Đường dẫn config file (default: config/cell_layout.yaml)",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="KHÔNG xóa items hiện tại trước khi build (default: clear)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging (DEBUG level)",
    )
    parser.add_argument(
        "--library-path",
        type=str,
        default=None,
        help="Custom RoboDK Library path (default: C:/RoboDK/Library)",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Build tối giản: bỏ floor, Cam2D viewport, CalibrationTarget frame, "
             "2/3 object templates → tiết kiệm ~10 API call cho RoboDK Free quota.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger("build_station")
    logger.info("=" * 60)
    logger.info("RoboDK Cell Builder — Code-first approach")
    logger.info("=" * 60)
    logger.info("Config: %s", args.config)

    # ─── Step 1: Load + validate config ───
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    try:
        config = CellConfig.from_yaml(config_path)
        logger.info("✓ Config valid (version=%s)", config.metadata.version)
    except FileNotFoundError:
        logger.error("Config file không tồn tại: %s", config_path)
        return 3
    except (InvalidConfigError, Exception) as e:
        logger.error("Config validation failed: %s", e)
        return 2

    # ─── Step 2: Build cell ───
    library_path = Path(args.library_path) if args.library_path else None

    try:
        loader = CellLoader(
            config=config,
            project_root=PROJECT_ROOT,
            library_path=library_path,
            minimal_build=args.minimal,
        )
        items = loader.build(clear_existing=not args.no_clear)

    except RoboDKConnectionError as e:
        logger.error(str(e))
        return 1
    except MissingRobotError as e:
        logger.error(str(e))
        return 3
    except MissingMeshError as e:
        logger.error("Missing mesh: %s", e)
        return 3
    except RobotConnectionError as e:
        logger.error(str(e))
        logger.warning("Cell built nhưng không kết nối được robot thật.")
        # Cell vẫn build OK, chỉ là connection fail
        # Return non-zero để CI/CD biết
        return 1
    except struct.error:
        # Popup "API calls are limited" của RoboDK Free chèn vào luồng socket
        # API giữa chừng → desync → struct.error khi đọc số nguyên trạng thái.
        logger.error(
            "RoboDK Free đã bật popup 'API calls are limited' và ngắt kết nối "
            "API giữa chừng. Đây là GIỚI HẠN CỦA RoboDK FREE, không phải lỗi code."
        )
        logger.error(
            "Khắc phục: ĐÓNG HẲN RoboDK (tắt cả popup lẫn ứng dụng), rồi chạy "
            "lại build_station.py — phiên RoboDK mới sẽ reset bộ đếm API call."
        )
        return 1
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        return 1

    # ─── Step 3: Report ───
    logger.info("=" * 60)
    logger.info("✓ Build successful")
    logger.info("=" * 60)
    n_objects = len(items.get("objects", {}))
    n_frames = len(items.get("frames", {}))
    logger.info("Items created: robot, table, camera, gripper, %d frames, %d objects",
                n_frames, n_objects)

    if config.robot_connection.enabled:
        logger.info("Real robot connected at %s:%d",
                    config.robot_connection.ip, config.robot_connection.port)

    return 0


if __name__ == "__main__":
    sys.exit(main())
