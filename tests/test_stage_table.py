"""The cycle-time table of E6 from the stage columns (04_analyze_results._stage_table),
over the complete cycles kept by the rules fixed before the E3 sessions."""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
analyze = import_module("04_analyze_results")

# Trials 2-4 are kept. Set aside: 5 failed, 6 was held, 7 had an HSE timeout and 8
# a failed read during the job, 9 a bracket 2.1 s wide, 1 is the first of its run.
# 10 succeeded with no bracketed end (held+race).
MOVED = pd.DataFrame({
    "trial_id": [2, 3, 4, 5, 6, 7, 8, 9, 1, 10],
    "failure_reason": ["", "", "", "motion_error: alarm 4315", "", "", "", "", "", ""],
    "job_flags": ["", "", "", "", "held", "timeout", "error", "", "", "held+race"],
    "detect_window_ms": [5.0, 6.0, 7.0, 8.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    "inference_ms": [30.0, 40.0, None, 35.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
    "job_ms": [9000.0, 9500.0, 10000.0, 3000.0, 30000.0, 9200.0, 9300.0, 9900.0, 9100.0, None],
    "cycle_ms": [12000.0, 13000.0, 14000.0, 5000.0, 33000.0, 12200.0, 12300.0, 12900.0,
                 12100.0, None],
    "cycle_pc_ms": [12100.0, 13100.0, 14100.0, 5100.0, 33100.0, 12300.0, 12400.0, 13000.0,
                    12200.0, 12500.0],
    "job_end_lo_ms": [11950.0, 12950.0, 13950.0, 4950.0, 32950.0, 12150.0, 12250.0, 11900.0,
                      12050.0, None],
    "job_end_hi_ms": [12060.0, 13060.0, 14060.0, 5060.0, 33060.0, 12260.0, 12360.0, 14000.0,
                      12160.0, None],
    "frame_capture_ms": [-200.0, -250.0, -300.0, -200.0, -200.0, -200.0, -200.0, -200.0,
                         -200.0, -200.0],
    "frame_capture_src": ["arrival", "arrival", "got"] + ["arrival"] * 7,
    "cycle_time_s": [13.0, 15.0, 20.0, 6.0, 40.0, 13.0, 13.0, 13.0, 13.0, 13.0],  # not used
})


def test_every_row_rests_on_the_kept_complete_cycles_and_the_total_is_cycle_ms():
    table = analyze._stage_table(MOVED)
    assert table.loc["job_ms", "n"] == 3
    assert table.loc["inference_ms", "n"] == 2
    assert table.loc["job_ms", "P50"] == pytest.approx(9500.0)
    assert table.loc["cycle_ms", "P50"] == pytest.approx(13000.0)
    assert table.loc["cycle_ms", "P95"] == pytest.approx(13900.0)
    assert table.loc["cycle_pc_ms", "P50"] == pytest.approx(13100.0)
    assert table.loc["frame_age_ms", "P50"] == pytest.approx(256.0)   # 6 + 250


def test_what_was_set_aside_is_counted():
    _, counts = analyze._complete_cycles(MOVED)
    assert counts == {"complete": 8, "held": 1, "hse_fault": 2, "wide": 1, "first_of_run": 1,
                      "no_end": {"held+race": 1}, "frame_src": {"arrival": 2, "got": 1}}


def test_runs_without_bracketed_job_ends_give_no_table():
    old = pd.DataFrame({"cycle_time_s": [12.0], "failure_reason": [""]})
    assert analyze._stage_table(old) is None
    # 378d59f and f6f2830 logged a job_ms that ran to the PC's confirmation, and no flags
    earlier = pd.DataFrame({"job_ms": [9100.0], "failure_reason": [""], "trial_id": [2]})
    assert analyze._stage_table(earlier) is None
    untimed = pd.DataFrame({"job_ms": [None], "job_flags": ["untimed"], "failure_reason": [""],
                            "trial_id": [2]})
    assert analyze._stage_table(untimed) is None
