"""
test_campaign_sittings.py
─────────────────────────
A blinded campaign longer than one sitting is split into sittings that each cover
their own pose rows with every arm (run_blinded_campaign --first-pose), and the
analysis reads each row's arm back from the sealed keys (04 --key), which is the only
place the arm lives when arms differ by model rather than by depth mode.

Run:
    pytest tests/test_campaign_sittings.py -v
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
RUNNER = PROJECT_ROOT / "tools" / "run_blinded_campaign.py"


def _dry(*extra):
    cmd = [sys.executable, str(RUNNER), "--arm", "a", "--model-path models/a.pt",
           "--arm", "b", "--model-path models/b.pt",
           "--pose-list", "config/pose_lists/hard_v2.csv", "--block", "25",
           "--session", "test", "--seed", "1", "--dry-run", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=str(PROJECT_ROOT))


def test_second_sitting_covers_its_own_rows():
    r = _dry("--first-pose", "100", "--trials", "100")
    assert r.returncode == 0, r.stderr
    slices = [ln.split("poses ")[1] for ln in r.stdout.splitlines() if "poses " in ln]
    assert len(slices) == 8                      # 4 blocks x 2 arms
    assert sorted(set(slices)) == ["100:125", "125:150", "150:175", "175:200"]


def test_misaligned_first_pose_refused():
    assert _dry("--first-pose", "10", "--trials", "100").returncode == 2


def test_running_past_the_list_refused():
    assert _dry("--first-pose", "250", "--trials", "100").returncode == 2


def _analyze_mod():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from importlib import import_module
    return import_module("04_analyze_results")


def _campaign(tmp_path, stray=False):
    import pandas as pd
    base = dict(class_name="carton", cycle_time_s=9.0, failure_reason="", lighting="dim",
                overlap="", mode="real", depth_mode="rgbd")
    schedule = []
    for b, (arm, ok_upto) in enumerate([("anchored", 9), ("wide", 6),
                                        ("wide", 7), ("anchored", 8)]):
        block_id = f"e3-1-b{b + 1:03d}"
        lo = (b // 2) * 10
        schedule.append({"block_id": block_id, "code": "A", "arm": arm,
                         "slice": f"{lo}:{lo + 10}"})
        rows = [{**base, "trial_id": j + 1, "pose_id": f"P{lo + j:04d}", "block_id": block_id,
                 "success": int(j < ok_upto), "human_ok": int(j < ok_upto)}
                for j in range(10)]
        pd.DataFrame(rows).to_csv(tmp_path / f"experiment_real_{b}.csv", index=False)
    if stray:
        pd.DataFrame([{**base, "trial_id": 1, "pose_id": "P0200", "block_id": "doichung-dau",
                       "success": 1, "human_ok": 1}]).to_csv(
            tmp_path / "experiment_real_9.csv", index=False)
    (tmp_path / "key_e3-1.json").write_text(json.dumps({"schedule": schedule}),
                                            encoding="utf-8")


def _run(monkeypatch, tmp_path, *extra):
    mod = _analyze_mod()
    argv = ["04_analyze_results.py", "--csv", str(tmp_path / "experiment_real_*.csv"),
            "--key", str(tmp_path / "key_*.json"), "--split-col", "arm",
            "--baseline", "anchored", "--score-col", "human_ok",
            "--out", str(tmp_path / "fig.png"), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    return mod.main()


def test_only_keeps_the_named_arms(monkeypatch, tmp_path):
    _campaign(tmp_path)
    assert _run(monkeypatch, tmp_path, "--only", "anchored", "wide") == 0
    # one arm left is not a comparison; an arm that is not there is a typo
    assert _run(monkeypatch, tmp_path, "--only", "anchored") == 1
    assert _run(monkeypatch, tmp_path, "--only", "anchored", "khong_co") == 1


def test_key_decodes_model_arms(monkeypatch, tmp_path):
    _campaign(tmp_path)
    assert _run(monkeypatch, tmp_path) == 0


def test_row_outside_the_key_refused(monkeypatch, tmp_path):
    _campaign(tmp_path, stray=True)
    assert _run(monkeypatch, tmp_path) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
