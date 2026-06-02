"""
helpers.py
──────────
Shared helper functions: path resolution, logging setup, YAML I/O.

Kept separate so any module/script can import without duplicating code.
No dependency on RoboDK / RealSense → importable anywhere.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create directory (including parents) if it does not exist. Return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Return the current timestamp as a string — used for file/run naming."""
    return datetime.now().strftime(fmt)


def setup_logging(
    name: str = "pickplace",
    verbose: bool = False,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Configure consistent logging for all scripts.

    Args:
        name: Logger name.
        verbose: True → DEBUG level, False → INFO.
        log_file: If provided, also write logs to this file (in addition to stdout).

    Returns:
        Configured logger.
    """
    # Windows console defaults to cp1252 → cannot print ✓ ─ → … without crashing.
    # Force stdout/stderr to UTF-8 (Python 3.7+).
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
    """Load a YAML file into a dict. Raise FileNotFoundError if missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}
