"""
test_telemetry.py
─────────────────
Verify TelemetryLogger thread-safe + CSV format đúng.
"""
from __future__ import annotations

import csv
import threading

import pytest

from src.orchestrator.telemetry import TelemetryLogger


class TestTelemetryLogger:
    def test_header_written_on_open(self, tmp_path):
        log = TelemetryLogger(tmp_path / "tel.csv")
        log.open()
        log.close()
        with (tmp_path / "tel.csv").open() as f:
            rows = list(csv.reader(f))
        assert rows[0] == list(TelemetryLogger.HEADER)

    def test_log_state_writes_row(self, tmp_path):
        with TelemetryLogger(tmp_path / "tel.csv") as log:
            log.log_state([1, 2, 3, 4, 5, 6], running=True, alarm=0)
        with (tmp_path / "tel.csv").open() as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2                       # header + 1 row
        row = rows[1]
        # joints
        assert [float(x) for x in row[1:7]] == [1, 2, 3, 4, 5, 6]
        assert row[7] == "1"                        # running
        assert row[8] == "0"                        # alarm

    def test_log_state_wrong_joint_count_skipped(self, tmp_path):
        with TelemetryLogger(tmp_path / "tel.csv") as log:
            log.log_state([1, 2, 3])                # 3 ≠ 6
            log.log_state([1, 2, 3, 4, 5, 6])
        with (tmp_path / "tel.csv").open() as f:
            rows = list(csv.reader(f))
        # 1 header + 1 valid row (3-element joints skipped)
        assert len(rows) == 2

    def test_log_state_optional_fields_blank_if_none(self, tmp_path):
        with TelemetryLogger(tmp_path / "tel.csv") as log:
            log.log_state([1, 2, 3, 4, 5, 6])       # no running/alarm
        with (tmp_path / "tel.csv").open() as f:
            rows = list(csv.reader(f))
        assert rows[1][7] == ""                     # running blank
        assert rows[1][8] == ""                     # alarm blank

    def test_thread_safety_concurrent_writes(self, tmp_path):
        """Stress: 4 threads × 100 writes mỗi cái → 400 rows + header."""
        log = TelemetryLogger(tmp_path / "tel.csv")
        log.open()

        def worker():
            for _ in range(100):
                log.log_state([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        log.close()

        with (tmp_path / "tel.csv").open() as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1 + 400

    def test_open_idempotent(self, tmp_path):
        log = TelemetryLogger(tmp_path / "tel.csv")
        log.open()
        log.open()                                  # gọi lần 2 không error
        log.close()

    def test_close_idempotent(self, tmp_path):
        log = TelemetryLogger(tmp_path / "tel.csv")
        log.open()
        log.close()
        log.close()                                 # gọi lần 2 không error

    def test_log_before_open_silently_skipped(self, tmp_path):
        log = TelemetryLogger(tmp_path / "tel.csv")
        log.log_state([1] * 6)                      # không raise
        # File chưa tồn tại vì chưa open
        assert not (tmp_path / "tel.csv").exists()

    def test_log_after_close_no_writerow_on_closed_file(self, tmp_path):
        # Regression: close() nulls _writer under the lock so a racing log_state()
        # drops the row instead of writing to a closed file (ValueError).
        log = TelemetryLogger(tmp_path / "tel.csv")
        log.open()
        log.log_state([1] * 6)
        log.close()
        log.log_state([2] * 6)                      # must NOT raise
        log.flush()                                 # must NOT raise after close
        with open(tmp_path / "tel.csv", newline="") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1 + 1                   # header + the single pre-close row
