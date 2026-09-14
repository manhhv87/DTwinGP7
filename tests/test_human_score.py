"""
test_human_score.py
───────────────────
The operator's verdict (`human_ok`): asked once per trial that actually moved
the robot, written next to the machine's `success`, and selectable as the
outcome the analysis scores on.

Why it matters: the gripper detect sensor proves a part was held when the jaws
closed, and nothing after that. A part dropped in transfer, or dropped beside
the place point, is success=1 and human_ok=0.

Run:
    pytest tests/test_human_score.py -v
"""
from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logging.logger import CONTEXT_FIELDNAMES, TrialLogger  # noqa: E402
from src.orchestrator.orchestrator import Orchestrator  # noqa: E402

log = logging.getLogger("test_human_score")


def _analyze_mod():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from importlib import import_module
    return import_module("04_analyze_results")


def _row(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))[0]


def _bare_orch(trial_logger, hook):
    """An Orchestrator with only what _log_trial touches — no robot, no camera."""
    o = Orchestrator.__new__(Orchestrator)
    o.trial_logger = trial_logger
    o._last_frame_path = ""
    o.current_pose_id = "P0001"
    o.current_condition = ""
    o.current_stack_on = ""
    o.session_id = o.block_id = o.operator_id = ""
    o.sm = SimpleNamespace(state=SimpleNamespace(value="done"))
    o.human_score_hook = hook
    return o


class TestColumn:
    def test_in_schema(self):
        assert "human_ok" in CONTEXT_FIELDNAMES

    def test_value_round_trips(self, tmp_path):
        trial_logger = TrialLogger(tmp_path / "run.csv")
        trial_logger.log_trial(1, True, class_name="carton", extra={"human_ok": 0})
        row = _row(tmp_path / "run.csv")
        assert row["success"] == "1"      # machine: the motion ran, part was gripped
        assert row["human_ok"] == "0"     # operator: it did not end up in place

    def test_unscored_trial_leaves_it_empty(self, tmp_path):
        trial_logger = TrialLogger(tmp_path / "run.csv")
        trial_logger.log_trial(1, True, class_name="carton")
        assert _row(tmp_path / "run.csv")["human_ok"] == ""


class TestHook:
    def test_asked_when_the_robot_moved(self, tmp_path):
        asked = []
        orch = _bare_orch(TrialLogger(tmp_path / "run.csv"),
                          lambda tid, ok: asked.append((tid, ok)) or 0)
        orch._log_trial(7, True, "", time.time(), {"class_name": "carton"},
                        ask_human=True)
        assert asked == [(7, True)]
        assert _row(tmp_path / "run.csv")["human_ok"] == "0"

    def test_not_asked_when_nothing_moved(self, tmp_path):
        asked = []
        orch = _bare_orch(TrialLogger(tmp_path / "run.csv"),
                          lambda tid, ok: asked.append(tid) or 1)
        # detection_miss / unreachable path: default ask_human=False
        orch._log_trial(7, False, "detection_miss", time.time(), None)
        assert asked == []
        assert _row(tmp_path / "run.csv")["human_ok"] == ""

    def test_no_hook_is_fine(self, tmp_path):
        orch = _bare_orch(TrialLogger(tmp_path / "run.csv"), None)
        orch._log_trial(7, True, "", time.time(), {"class_name": "carton"},
                        ask_human=True)
        assert _row(tmp_path / "run.csv")["human_ok"] == ""


class TestScoreCol:
    def test_success_is_the_identity(self):
        import pandas as pd
        mod = _analyze_mod()
        df = pd.DataFrame({"success": [1, 0]})
        assert mod._apply_score_col(log, df, "success", "x") is df

    def test_human_ok_replaces_success(self):
        import pandas as pd
        mod = _analyze_mod()
        df = pd.DataFrame({"success": [1, 1, 1], "human_ok": [1, 0, 0]})
        out = mod._apply_score_col(log, df, "human_ok", "x")
        assert list(out["success"]) == [1, 0, 0]
        assert list(df["success"]) == [1, 1, 1]     # caller's frame untouched

    def test_missing_column_refused(self):
        import pandas as pd
        mod = _analyze_mod()
        df = pd.DataFrame({"success": [1, 0]})
        assert mod._apply_score_col(log, df, "human_ok", "x") is None

    def test_partly_filled_refused(self):
        """Scoring the filled rows only would drop trials from one arm."""
        import pandas as pd
        mod = _analyze_mod()
        df = pd.DataFrame({"success": [1, 1], "human_ok": [1, None]})
        assert mod._apply_score_col(log, df, "human_ok", "x") is None

    def test_non_binary_refused(self):
        import pandas as pd
        mod = _analyze_mod()
        df = pd.DataFrame({"success": [1, 1], "human_ok": ["yes", "maybe"]})
        assert mod._apply_score_col(log, df, "human_ok", "x") is None


class TestSplitCol:
    """A blinded campaign writes one CSV per block, so the arm is a column."""

    @staticmethod
    def _blocks(tmp_path, rates, n_block=2, per_block=10):
        import pandas as pd
        base = dict(class_name="carton", cycle_time_s=9.0, failure_reason="",
                    lighting="", overlap="", mode="real")
        for arm, ok_upto in rates.items():
            for b in range(n_block):
                rows = []
                for j in range(per_block):
                    ok = int(j < ok_upto)
                    rows.append({**base, "trial_id": j + 1,
                                 "pose_id": f"P{b * per_block + j:04d}",
                                 "success": ok, "human_ok": ok, "depth_mode": arm})
                pd.DataFrame(rows).to_csv(tmp_path / f"khoi_{arm}_{b}.csv", index=False)
        return str(tmp_path / "khoi_*.csv")

    def _run(self, monkeypatch, tmp_path, *extra):
        mod = _analyze_mod()
        pattern = self._blocks(tmp_path, {"rgbd": 9, "plane": 6, "fusion": 8})
        argv = ["04_analyze_results.py", "--csv", pattern,
                "--score-col", "human_ok", "--out", str(tmp_path / "fig.png"), *extra]
        monkeypatch.setattr(sys, "argv", argv)
        return mod.main()

    def test_one_command_over_block_files(self, monkeypatch, tmp_path):
        assert self._run(monkeypatch, tmp_path,
                         "--split-col", "depth_mode", "--baseline", "rgbd") == 0

    def test_unknown_baseline_refused(self, monkeypatch, tmp_path):
        assert self._run(monkeypatch, tmp_path,
                         "--split-col", "depth_mode", "--baseline", "khong_co") == 1

    def test_unknown_column_refused(self, monkeypatch, tmp_path):
        assert self._run(monkeypatch, tmp_path, "--split-col", "khong_co") == 1

    def test_paired_with_accepts_a_glob(self, monkeypatch, tmp_path):
        """One arm spread over several block files is still one arm."""
        mod = _analyze_mod()
        self._blocks(tmp_path, {"rgbd": 9, "plane": 6})
        argv = ["04_analyze_results.py",
                "--csv", str(tmp_path / "khoi_rgbd_*.csv"),
                "--paired-with", str(tmp_path / "khoi_plane_*.csv"),
                "--label-a", "rgbd", "--labels-b", "plane",
                "--score-col", "human_ok", "--out", str(tmp_path / "fig.png")]
        monkeypatch.setattr(sys, "argv", argv)
        assert mod.main() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
