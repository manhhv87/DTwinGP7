"""
telemetry.py
────────────
CSV logger for joint state + IO state of the real robot (HSE backend).

Purpose: record a state snapshot every tick for:
  - Post-analysis (plot trajectory, measure cycle time)
  - Drift detection (compare commanded vs actual)
  - Bidirectional digital twin demo (CSV → matplotlib animate)

Thread-safe — call `log_state()` from the mirror thread concurrently with the orchestrator thread.
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
    """Log joint angles + optional IO state to CSV with a UNIX timestamp.

    CSV structure:
        timestamp,j1,j2,j3,j4,j5,j6,running,alarm

    `running` and `alarm` are optional (None if the mirror thread could not read them).
    """

    HEADER = ("timestamp", "j1", "j2", "j3", "j4", "j5", "j6", "running", "alarm")

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self._fp: IO[str] | None = None
        self._writer: csv.writer | None = None
        self._lock = threading.Lock()
        self._row_count = 0

    def open(self) -> None:
        """Open the file and write the header. Idempotent."""
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
        """Write one state row. Thread-safe."""
        if len(joints) != 6:
            logger.warning("Telemetry joints has %d elements, expected 6", len(joints))
            return
        row = (
            f"{time.time():.4f}",
            *(f"{j:.4f}" for j in joints),
            "" if running is None else int(running),
            "" if alarm is None else int(alarm),
        )
        # Check + write INSIDE the lock so a concurrent close() (which nulls _writer
        # under the same lock) cannot let us writerow() into a just-closed file.
        with self._lock:
            if self._writer is None:
                return
            self._writer.writerow(row)
            self._row_count += 1

    def flush(self) -> None:
        """Force flush — call periodically if the buffer lags too long (e.g. once per second)."""
        with self._lock:
            if self._fp is not None:
                self._fp.flush()

    def close(self) -> None:
        """Close the file. Idempotent."""
        with self._lock:
            if self._fp is None:
                return
            self._fp.flush()
            self._fp.close()
            self._fp = None
            self._writer = None          # block any concurrent log_state row
        logger.info("Telemetry closed: %d rows → %s", self._row_count, self.csv_path)

    def __enter__(self) -> "TelemetryLogger":
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
