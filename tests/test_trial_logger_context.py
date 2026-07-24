"""
test_trial_logger_context.py
────────────────────────────
Tests for the C3 failure-context extension of TrialLogger:
new files carry the context columns; files created with the OLD schema keep
their original header on append (no silent column misalignment).

Run:
    pytest tests/test_trial_logger_context.py -v
"""
from __future__ import annotations

import csv

import pytest

from src.logging.logger import CONTEXT_FIELDNAMES, FIELDNAMES, TrialLogger


class TestNewSchema:
    def test_header_includes_context_columns(self, tmp_path):
        logger = TrialLogger(tmp_path / "run.csv")
        with (tmp_path / "run.csv").open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        assert header == FIELDNAMES + CONTEXT_FIELDNAMES

    def test_extra_fields_written(self, tmp_path):
        logger = TrialLogger(tmp_path / "run.csv",
                             extra_context={"lighting": "dim", "mode": "real"})
        logger.log_trial(
            1, False, class_name="bottle", cycle_time_s=7.2,
            failure_reason="grasp_failed", final_state="error",
            extra={"det_x_mm": 450.0, "det_y_mm": -150.0, "det_z_mm": 512.0,
                   "det_yaw_deg": 33.1, "confidence": 0.87, "mask_area": 5120,
                   "frame_path": "results/frames/f1.png", "pose_id": "P0007"},
        )
        with (tmp_path / "run.csv").open(newline="", encoding="utf-8") as f:
            row = list(csv.DictReader(f))[0]
        assert row["success"] == "0"
        assert row["failure_reason"] == "grasp_failed"
        assert row["lighting"] == "dim"
        assert row["det_x_mm"] == "450.0"
        assert row["pose_id"] == "P0007"
        assert row["frame_path"].endswith("f1.png")

    def test_missing_extra_yields_empty_cells(self, tmp_path):
        logger = TrialLogger(tmp_path / "run.csv")
        logger.log_trial(1, True, class_name="cup")
        with (tmp_path / "run.csv").open(newline="", encoding="utf-8") as f:
            row = list(csv.DictReader(f))[0]
        assert row["det_x_mm"] == "" and row["pose_id"] == ""


class TestOldSchemaCompat:
    def test_append_to_old_file_keeps_old_header(self, tmp_path):
        old = tmp_path / "old.csv"
        with old.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

        logger = TrialLogger(old)
        logger.log_trial(1, True, class_name="tray",
                         extra={"det_x_mm": 700.0, "pose_id": "P0001"})

        with old.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            row = next(reader)
        assert header == FIELDNAMES            # unchanged
        assert len(row) == len(FIELDNAMES)     # extras dropped, no misalignment

    def test_summarize_still_works(self, tmp_path):
        logger = TrialLogger(tmp_path / "run.csv")
        logger.log_trial(1, True, class_name="bolt")
        logger.log_trial(2, False, class_name="bolt", failure_reason="unreachable")
        s = logger.summarize()
        assert s["total"] == 2 and s["successful"] == 1
        assert s["failure_modes"] == {"unreachable": 1}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
