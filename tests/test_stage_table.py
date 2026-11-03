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


def test_each_stage_and_the_total_get_p50_p95_and_their_count():
    moved = pd.DataFrame({"cycle_time_s": [12.0, 13.0, 14.0],
                          "inference_ms": [30.0, 40.0, None],
                          "job_ms": [9000.0, 9500.0, 10000.0]})
    table = analyze._stage_table(moved)
    assert table.loc["inference_ms", "n"] == 2
    assert table.loc["job_ms", "P50"] == pytest.approx(9500.0)
    # The total is the cycles' own duration, not a sum of stage percentiles.
    assert table.loc["cycle_ms", "P50"] == pytest.approx(13000.0)
    assert table.loc["cycle_ms", "P95"] == pytest.approx(13900.0)


def test_runs_without_stage_columns_give_no_table():
    assert analyze._stage_table(pd.DataFrame({"cycle_time_s": [12.0]})) is None
