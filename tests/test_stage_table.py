"""The cycle-time table of E6 from the stage columns (04_analyze_results._stage_table)."""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
analyze = import_module("04_analyze_results")


def test_every_row_rests_on_the_complete_cycles_and_the_total_is_cycle_ms():
    moved = pd.DataFrame({
        "failure_reason": ["", "", "", "motion_error: alarm 4315"],
        "detect_window_ms": [5.0, 6.0, 7.0, 8.0],
        "inference_ms": [30.0, 40.0, None, 35.0],
        "job_ms": [9000.0, 9500.0, 10000.0, 3000.0],
        "cycle_ms": [12000.0, 13000.0, 14000.0, 5000.0],
        "cycle_time_s": [13.0, 15.0, 20.0, 6.0]})          # verdict included: not used
    table = analyze._stage_table(moved)
    assert table.loc["job_ms", "n"] == 3                    # the alarm cycle is left out
    assert table.loc["inference_ms", "n"] == 2
    assert table.loc["job_ms", "P50"] == pytest.approx(9500.0)
    assert table.loc["cycle_ms", "P50"] == pytest.approx(13000.0)
    assert table.loc["cycle_ms", "P95"] == pytest.approx(13900.0)


def test_runs_without_stage_columns_give_no_table():
    old = pd.DataFrame({"cycle_time_s": [12.0], "failure_reason": [""]})
    assert analyze._stage_table(old) is None
