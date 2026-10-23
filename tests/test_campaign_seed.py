"""The blinded campaign's seed is secret unless given, and its key records the code.

Session 1 of E2 printed the seed in the operator guide (S=1), and the code-to-arm
mapping was recomputed from it with run_blinded_campaign.py. Without --seed the
tool now draws a seed from the operating system and writes it only to the sealed
key file. Every run's metadata and the key also record the git commit, branch and
uncommitted changes (src/utils/provenance.py).

Run:
    pytest tests/test_campaign_seed.py -v
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.logging.logger import TrialLogger
from src.perception.model_artifact import run_metadata
from src.perception import MockDetector
from src.utils.provenance import git_provenance

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _tool():
    path = PROJECT_ROOT / "tools" / "run_blinded_campaign.py"
    spec = importlib.util.spec_from_file_location("campaign_seed_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _campaign(tmp_path, monkeypatch, capsys, *extra, session="s"):
    """Run the tool up to its first block, which 'fails' at once; return the key and
    everything printed."""
    tool = _tool()
    launched = []
    monkeypatch.setattr(tool.subprocess, "call", lambda cmd, cwd=None: launched.append(cmd) or 9)
    monkeypatch.setattr(sys, "argv", [
        "run_blinded_campaign.py", "--arm", "rgbd", "--depth-mode rgbd",
        "--arm", "plane", "--depth-mode plane", "--arm", "fusion", "--depth-mode fusion",
        "--pose-list", "config/pose_lists/std_v2.csv", "--trials", "50", "--block", "25",
        "--session", session, "--key-out", str(tmp_path), *extra])
    assert tool.main() == 9
    out = capsys.readouterr()
    key = json.loads((tmp_path / f"key_{session}.json").read_text(encoding="utf-8"))
    return key, out.out + out.err, launched


def test_without_a_seed_the_seed_is_secret(tmp_path, monkeypatch, capsys):
    key, printed, launched = _campaign(tmp_path, monkeypatch, capsys)
    assert key["seed_source"] == "secrets"
    assert isinstance(key["seed"], int) and key["seed"].bit_length() > 32
    assert str(key["seed"]) not in printed
    assert " ".join(launched[0]).count(str(key["seed"])) == 0


def test_two_campaigns_draw_different_seeds(tmp_path, monkeypatch, capsys):
    a, _, _ = _campaign(tmp_path, monkeypatch, capsys, session="a")
    b, _, _ = _campaign(tmp_path, monkeypatch, capsys, session="b")
    assert a["seed"] != b["seed"]


def test_a_given_seed_reproduces_the_schedule(tmp_path, monkeypatch, capsys):
    a, _, _ = _campaign(tmp_path, monkeypatch, capsys, "--seed", "1", session="a")
    b, _, _ = _campaign(tmp_path, monkeypatch, capsys, "--seed", "1", session="b")
    assert a["seed_source"] == "cli" and a["seed"] == 1
    assert a["arm_of_code"] == b["arm_of_code"]
    assert [s["code"] for s in a["schedule"]] == [s["code"] for s in b["schedule"]]


def test_session1_seed_still_gives_session1_codes(tmp_path, monkeypatch, capsys):
    """key_e2chuan-1.json: seed 1 gave A=fusion, B=rgbd, C=plane."""
    key, _, _ = _campaign(tmp_path, monkeypatch, capsys, "--seed", "1")
    assert key["arm_of_code"] == {"A": "fusion", "B": "rgbd", "C": "plane"}


def test_the_key_records_the_code(tmp_path, monkeypatch, capsys):
    key, _, _ = _campaign(tmp_path, monkeypatch, capsys)
    assert key["code"] == git_provenance(PROJECT_ROOT)


def _git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    try:
        _git(tmp_path, "init", "-q", "-b", "main")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-q", "-m", "first")
    return tmp_path


def test_provenance_of_a_clean_checkout(repo):
    info = git_provenance(repo)
    assert len(info["commit"]) == 40 and info["branch"] == "main"
    assert info["dirty"] is False and info["changed"] == [] and info["untracked"] == []


def test_provenance_of_a_changed_checkout(repo):
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    (repo / "new.py").write_text("", encoding="utf-8")
    info = git_provenance(repo)
    assert info["dirty"] is True
    assert info["changed"] == ["a.py"] and info["untracked"] == ["new.py"]


def test_an_untracked_file_alone_is_not_dirty(repo):
    (repo / "notes.txt").write_text("", encoding="utf-8")
    assert git_provenance(repo)["dirty"] is False


def test_no_checkout_is_reported_not_raised(tmp_path):
    info = git_provenance(tmp_path / "nowhere")
    assert set(info) == {"error"}


def test_run_metadata_carries_the_code(tmp_path):
    code = {"commit": "c" * 40, "branch": "b", "dirty": False}
    logger = TrialLogger(tmp_path / "run.csv", run_metadata=lambda: run_metadata(
        {"k": 1}, MockDetector(), entrypoint="t", code=code))
    meta = json.loads(logger.metadata_path.read_text(encoding="utf-8"))
    assert meta["provenance"]["code"] == code
    assert "code" not in run_metadata({"k": 1}, MockDetector(), entrypoint="t")


def test_the_runner_writes_the_code_into_its_metadata(monkeypatch):
    """03 hands git_provenance's answer to run_metadata for every run."""
    path = PROJECT_ROOT / "scripts" / "03_run_experiment.py"
    spec = importlib.util.spec_from_file_location("runner_code_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    sentinel = {"commit": "f" * 40, "branch": "x", "dirty": True}
    captured = {}

    class Stop(Exception):
        pass

    def fake_logger(path, run_metadata=None, **kw):
        captured["meta"] = run_metadata
        raise Stop

    import logging
    monkeypatch.setattr(runner, "setup_logging", lambda *a, **k: logging.getLogger("x"))
    monkeypatch.setattr(runner, "git_provenance", lambda root: sentinel)
    monkeypatch.setattr(runner, "TrialLogger", fake_logger)
    monkeypatch.setattr(sys, "argv", ["03_run_experiment.py", "--mode", "sim",
                                      "--headless", "--trials", "1"])
    with pytest.raises(Stop):
        runner.main()
    assert captured["meta"]()["code"] == sentinel


def test_a_given_seed_on_a_live_campaign_is_warned_about(tmp_path, monkeypatch, capsys):
    key, printed, _ = _campaign(tmp_path, monkeypatch, capsys, "--seed", "2")
    assert "WARNING: --seed was given" in printed and key["seed"] == 2


def test_no_warning_without_a_seed(tmp_path, monkeypatch, capsys):
    key, printed, _ = _campaign(tmp_path, monkeypatch, capsys)
    assert "WARNING" not in printed


def test_depth_mode_arms_keep_the_placement_check(tmp_path, monkeypatch, capsys):
    key, printed, launched = _campaign(tmp_path, monkeypatch, capsys)
    assert "--no-placement-check" not in launched[0]


def test_model_arms_switch_the_placement_check_off(tmp_path, monkeypatch, capsys):
    """E3 to E5: the arms differ in the detector, so would the flags."""
    tool = _tool()
    launched = []
    monkeypatch.setattr(tool.subprocess, "call", lambda cmd, cwd=None: launched.append(cmd) or 9)
    monkeypatch.setattr(sys, "argv", [
        "run_blinded_campaign.py",
        "--arm", "anchored", "--depth-mode rgbd --model-path models/e3_anchored.pt",
        "--arm", "wide", "--depth-mode rgbd --model-path models/e3_wide.pt",
        "--pose-list", "config/pose_lists/hard_v2.csv", "--trials", "25", "--block", "25",
        "--session", "e3", "--key-out", str(tmp_path),
        "--common", "--mode real --confirm-each-trial"])
    assert tool.main() == 9
    assert launched[0].count("--no-placement-check") == 1
    assert "placement check is off for every arm" in capsys.readouterr().out


def test_the_runner_accepts_the_flag(monkeypatch):
    path = PROJECT_ROOT / "scripts" / "03_run_experiment.py"
    spec = importlib.util.spec_from_file_location("runner_flag_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(sys, "argv", ["03_run_experiment.py", "--no-placement-check"])
    assert runner.parse_args().no_placement_check is True
    monkeypatch.setattr(sys, "argv", ["03_run_experiment.py"])
    assert runner.parse_args().no_placement_check is False
