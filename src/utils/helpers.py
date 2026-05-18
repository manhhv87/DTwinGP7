"""
helpers.py
──────────
Helper functions dùng chung: path resolution, logging setup, YAML I/O.

Tách riêng để mọi module/script khác import được mà không lặp code.
Không phụ thuộc RoboDK / RealSense → import được ở bất kỳ đâu.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    """Tạo thư mục (kể cả parent) nếu chưa có. Trả về Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Trả về timestamp hiện tại dạng string — dùng đặt tên file/run."""
    return datetime.now().strftime(fmt)


def setup_logging(
    name: str = "pickplace",
    verbose: bool = False,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Cấu hình logging nhất quán cho mọi script.

    Args:
        name: Tên logger.
        verbose: True → DEBUG level, False → INFO.
        log_file: Nếu có, ghi log thêm vào file này (ngoài stdout).

    Returns:
        Logger đã cấu hình.
    """
    # Console Windows mặc định cp1252 → không in được ký tự ✓ ─ → … gây crash.
    # Ép stdout/stderr sang UTF-8 (Python 3.7+).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_path = Path(log_file)
        ensure_dir(log_path.parent)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(name)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load file YAML thành dict. Raise FileNotFoundError nếu không có."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file không tồn tại: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}
