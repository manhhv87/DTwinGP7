"""
telemetry.py
────────────
Logger CSV cho joint state + IO state của robot thật (HSE backend).

Mục đích: ghi snapshot state mỗi tick để:
  - Post-analysis (vẽ trajectory, đo cycle time)
  - Drift detection (so sánh commanded vs actual)
  - Demo bidirectional digital twin (CSV → matplotlib animate)

Thread-safe — gọi `log_state()` từ mirror thread đồng thời orchestrator thread.
"""
from __future__ import annotations

import csv
import logging
import threading
import time
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)


class TelemetryLogger:
    """Log joint angles + optional IO state ra CSV với timestamp UNIX.

    Cấu trúc CSV:
        timestamp,j1,j2,j3,j4,j5,j6,running,alarm

    `running` và `alarm` optional (None nếu mirror thread không đọc được).
    """

    HEADER = ("timestamp", "j1", "j2", "j3", "j4", "j5", "j6", "running", "alarm")

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self._fp: IO[str] | None = None
        self._writer: csv.writer | None = None
        self._lock = threading.Lock()
        self._row_count = 0

    def open(self) -> None:
        """Mở file + ghi header. Idempotent."""
        if self._fp is not None:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fp)
        self._writer.writerow(self.HEADER)
        self._fp.flush()
        logger.info("Telemetry → %s", self.csv_path)

    def log_state(
        self,
        joints: list[float],
        running: bool | None = None,
        alarm: int | None = None,
    ) -> None:
        """Ghi 1 row state. Thread-safe."""
        if self._writer is None:
            return
        if len(joints) != 6:
            logger.warning("Telemetry joints có %d phần tử, kỳ vọng 6", len(joints))
            return
        row = (
            f"{time.time():.4f}",
            *(f"{j:.4f}" for j in joints),
            "" if running is None else int(running),
            "" if alarm is None else int(alarm),
        )
        with self._lock:
            self._writer.writerow(row)
            self._row_count += 1

    def flush(self) -> None:
        """Force flush — gọi định kỳ nếu buffer quá lâu (vd 1 lần / giây)."""
        if self._fp is not None:
            self._fp.flush()

    def close(self) -> None:
        """Đóng file. Idempotent."""
        if self._fp is not None:
            with self._lock:
                self._fp.flush()
                self._fp.close()
            self._fp = None
            logger.info("Telemetry closed: %d rows → %s", self._row_count, self.csv_path)

    def __enter__(self) -> "TelemetryLogger":
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
