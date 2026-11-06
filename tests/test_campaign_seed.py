"""The blinded campaign's seed is secret unless given, and its key records the code.

Session 1 of E2 printed the seed in the operator guide (S=1), and the code-to-arm
mapping was recomputed from it with run_blinded_campaign.py. Without --seed the
tool draws a seed from the operating system and writes it only to the sealed key
file; a live campaign given --seed is refused unless --public-seed says it is not
blinded. Every run's metadata and the key also record the git commit, branch and
uncommitted changes (src/utils/provenance.py), and the tool decides once for every
arm whether 03's placement check runs.

Run:
    pytest tests/test_campaign_seed.py -v
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from src.logging.logger import TrialLogger
from src.perception.model_artifact import run_metadata
from src.perception import MockDetector
import src.utils.provenance as provenance
from src.utils.provenance import git_provenance, save_git_diff

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN = {"commit": "c" * 40, "branch": "fix/b3-review", "dirty": False, "changed": [],
         "untracked": []}
# Lenh A of the operator guide for session 2 (e2chuan-2) as printed on 23 Oct 2026,
# with $F = 100, $B = "e2chuan-2", $S = 2 filled in.
GUIDE_LENH_A_SESSION2 = [
    "--arm", "rgbd", "--depth-mode rgbd", "--arm", "plane", "--depth-mode plane",
    "--arm", "fusion", "--depth-mode fusion", "--pose-list", "config/pose_lists/std_v2.csv",
    "--first-pose", "100", "--trials", "100", "--block", "25", "--session", "e2chuan-2",
    "--operator", "PQM", "--seed", "2", "--common",
    "--mode real --ik-source yrc --tool-no 1 --confirm-each-trial --no-viewport-mirror "
    "--save-frames --frames-dir D:/Scientific/Dataset/DigitalTwin/frames"]


def _tool():
    path = PROJECT_ROOT / "tools" / "run_blinded_campaign.py"
    spec = importlib.util.spec_from_file_location("campaign_seed_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _campaign(tmp_path, monkeypatch, capsys, *extra, session="s", code=CLEAN):
    """Run the tool up to its first block, which 'fails' at once; return the key and
    everything printed. code stands in for git_provenance (None: the real one)."""
    tool = _tool()
    launched = []
    monkeypatch.setattr(tool.subprocess, "call", lambda cmd, cwd=None: launched.append(cmd) or 9)
    if code is not None:
        monkeypatch.setattr(provenance, "git_provenance", lambda root: dict(code))
    monkeypatch.setattr(sys, "argv", [
        "run_blinded_campaign.py", "--arm", "rgbd", "--depth-mode rgbd",
        "--arm", "plane", "--depth-mode plane", "--arm", "fusion", "--depth-mode fusion",
        "--pose-list", "config/pose_lists/std_v2.csv", "--trials", "50", "--block", "25",
        "--session", session, "--key-out", str(tmp_path), *extra])
    assert tool.main() == 9
    out = capsys.readouterr()
    key = json.loads((tmp_path / f"key_{session}.json").read_text(encoding="utf-8"))
    return key, out.out + out.err, launched


def _forms(seed):
    """The seed written in any base a script might print it in."""
    return {str(seed), hex(seed), f"{seed:x}", f"{seed:X}", oct(seed)[2:], bin(seed)[2:]}


def test_without_a_seed_the_seed_is_secret(tmp_path, monkeypatch, capsys):
    key, printed, launched = _campaign(tmp_path, monkeypatch, capsys)
    assert key["seed_source"] == "secrets"
    assert isinstance(key["seed"], int) and key["seed"].bit_length() > 32
    for form in _forms(key["seed"]):
        assert form not in printed
        assert form not in " ".join(launched[0])


def test_two_campaigns_draw_different_seeds(tmp_path, monkeypatch, capsys):
    a, _, _ = _campaign(tmp_path, monkeypatch, capsys, session="a")
    b, _, _ = _campaign(tmp_path, monkeypatch, capsys, session="b")
    assert a["seed"] != b["seed"]


def test_the_guide_command_for_session2_is_refused(tmp_path, monkeypatch, capsys):
    """Lenh A as the guide printed it for e2chuan-2, --seed 2 included: nothing is
    started and no key is written."""
    tool = _tool()
    launched = []
    monkeypatch.setattr(tool.subprocess, "call", lambda cmd, cwd=None: launched.append(cmd) or 0)
    monkeypatch.setattr(sys, "argv", ["run_blinded_campaign.py", *GUIDE_LENH_A_SESSION2,
                                      "--key-out", str(tmp_path)])
    assert tool.main() == 2
    err = capsys.readouterr().err
    assert launched == [] and list(tmp_path.iterdir()) == []
    assert "--public-seed" in err and "Nothing was started and no key was written" in err


def test_a_dry_run_accepts_a_seed(tmp_path, monkeypatch, capsys):
    tool = _tool()
    monkeypatch.setattr(sys, "argv", ["run_blinded_campaign.py", *GUIDE_LENH_A_SESSION2,
                                      "--key-out", str(tmp_path), "--dry-run"])
    assert tool.main() == 0
    assert list(tmp_path.iterdir()) == [] and "Dry run" in capsys.readouterr().out


def test_a_public_seed_reproduces_the_schedule(tmp_path, monkeypatch, capsys):
    a, printed, _ = _campaign(tmp_path, monkeypatch, capsys, "--seed", "1",
                              "--public-seed", session="a")
    b, _, _ = _campaign(tmp_path, monkeypatch, capsys, "--seed", "1", "--public-seed",
                        session="b")
    assert a["seed_source"] == "cli-public" and a["seed"] == 1
    assert a["arm_of_code"] == b["arm_of_code"]
    assert [s["code"] for s in a["schedule"]] == [s["code"] for s in b["schedule"]]
    assert "WARNING: --public-seed" in printed


def test_session1_seed_still_gives_session1_codes(tmp_path, monkeypatch, capsys):
    """key_e2chuan-1.json: seed 1 gave A=fusion, B=rgbd, C=plane."""
    key, _, _ = _campaign(tmp_path, monkeypatch, capsys, "--seed", "1", "--public-seed")
    assert key["arm_of_code"] == {"A": "fusion", "B": "rgbd", "C": "plane"}


def test_the_guide_seed_for_session2_repeats_the_session1_mapping(tmp_path, monkeypatch,
                                                                  capsys):
    """Why S=2 must not be used: it gives session 1's A=fusion, B=rgbd, C=plane."""
    key, _, _ = _campaign(tmp_path, monkeypatch, capsys, "--seed", "2", "--public-seed",
                          "--first-pose", "100")
    assert key["arm_of_code"] == {"A": "fusion", "B": "rgbd", "C": "plane"}


def test_the_key_records_the_code(tmp_path, monkeypatch, capsys):
    key, _, _ = _campaign(tmp_path, monkeypatch, capsys, code=None)
    assert key["code"] == git_provenance(PROJECT_ROOT)


def test_a_dirty_checkout_is_warned_about(tmp_path, monkeypatch, capsys):
    dirty = dict(CLEAN, dirty=True, changed=["src/x.py"], diff_sha256="d" * 64)
    key, printed, _ = _campaign(tmp_path, monkeypatch, capsys, code=dirty)
    assert "uncommitted changes in 1 file(s)" in printed
    assert key["code"]["diff_sha256"] == "d" * 64


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


def test_the_diff_of_a_dirty_checkout_is_hashed_and_can_be_saved(repo, tmp_path):
    assert "diff_sha256" not in git_provenance(repo)
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    info = git_provenance(repo)
    out = tmp_path / "run.diff"
    assert save_git_diff(repo, out) == info["diff_sha256"]
    assert hashlib.sha256(out.read_bytes()).hexdigest() == info["diff_sha256"]
    assert b"+x = 2" in out.read_bytes()


def test_paths_are_listed_plain(repo):
    """Spaces and non-ASCII names come back as they are, not quoted and escaped."""
    _git(repo, "mv", "a.py", "b c.py")
    (repo / "hướng dẫn.md").write_text("", encoding="utf-8")
    info = git_provenance(repo)
    assert info["changed"] == ["b c.py"]                 # a rename: the new path
    assert info["untracked"] == ["hướng dẫn.md"]


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


def test_the_runner_saves_the_diff_of_a_dirty_checkout(monkeypatch, tmp_path):
    path = PROJECT_ROOT / "scripts" / "03_run_experiment.py"
    spec = importlib.util.spec_from_file_location("runner_diff_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    dirty = dict(CLEAN, dirty=True, changed=["src/x.py"], diff_sha256="d" * 64)

    def save(repo, out):
        Path(out).write_text("diff --git a/src/x.py b/src/x.py\n", encoding="utf-8")
        return "d" * 64

    monkeypatch.setattr(runner, "setup_logging", lambda *a, **k: logging.getLogger("x"))
    monkeypatch.setattr(runner, "git_provenance", lambda root: dict(dirty))
    monkeypatch.setattr(runner, "save_git_diff", save)
    monkeypatch.setattr(runner, "TrialLogger", lambda p, **kw: TrialLogger(
        tmp_path / Path(p).name, **kw))
    monkeypatch.setattr(sys, "argv", ["03_run_experiment.py", "--mode", "sim",
                                      "--headless", "--trials", "1"])
    assert runner.main() == 0
    meta = json.loads(next(tmp_path.glob("*.metadata.json")).read_text(encoding="utf-8"))
    code = meta["provenance"]["code"]
    assert code["diff_sha256"] == "d" * 64
    assert (tmp_path / code["diff_file"]).read_text(encoding="utf-8").startswith("diff --git")


def test_no_warning_without_a_seed(tmp_path, monkeypatch, capsys):
    key, printed, _ = _campaign(tmp_path, monkeypatch, capsys)
    assert "WARNING" not in printed


def test_depth_mode_arms_keep_the_placement_check(tmp_path, monkeypatch, capsys):
    key, printed, launched = _campaign(
        tmp_path, monkeypatch, capsys, "--common", "--mode real --confirm-each-trial")
    assert "--no-placement-check" not in launched[0]
    assert key["placement_check"] == {"on": True,
                                      "reason": "the arms differ in --depth-mode only"}
    assert "Placement check: on for every arm" in printed


@pytest.mark.parametrize("arms,common,on", [
    ([("a", "--depth-mode rgbd"), ("b", "--depth-mode=plane")],
     "--confirm-each-trial", True),
    ([("a", "--depth-mode rgbd --model-path m.pt"), ("b", "--depth-mode plane --model-path m.pt")],
     "--confirm-each-trial", True),
    ([("a", "--depth-mode rgbd --model-path=models/a.pt"),
      ("b", "--depth-mode rgbd --model-path=models/b.pt")], "--confirm-each-trial", False),
    ([("a", "--depth-mode rgbd --config a.yaml"), ("b", "--depth-mode plane --config b.yaml")],
     "--confirm-each-trial", False),
    ([("real_only", "--depth-mode rgbd")], "--mode real --confirm-each-trial", True),
    ([("a", "--depth-mode rgbd"), ("b", "--depth-mode plane")],
     "--confirm-each-trial --no-placement-check", False),
    ([("a", "--depth-mode rgbd"), ("b", "--depth-mode plane")], "--mode real", False),
])
def test_the_check_stays_on_only_when_the_arms_differ_in_depth_mode_alone(arms, common, on):
    assert _tool().placement_check_decision(arms, common)[0] is on


def test_the_reasons_printed():
    tool = _tool()
    two = [("a", "--depth-mode rgbd"), ("b", "--depth-mode plane")]
    assert tool.placement_check_decision(two, "--mode real --confirm-each-trial") == (
        True, "the arms differ in --depth-mode only")
    assert tool.placement_check_decision(two[:1], "--confirm-each-trial") == (True, "one arm")
    assert tool.placement_check_decision(
        [("a", "--model-path a.pt"), ("b", "--model-path b.pt")], "--confirm-each-trial") == (
        False, tool.ARMS_DIFFER)
    assert tool.ARMS_DIFFER == "the arms differ in more than --depth-mode"


def test_a_per_arm_no_placement_check_is_refused(tmp_path, monkeypatch, capsys):
    tool = _tool()
    launched = []
    monkeypatch.setattr(tool.subprocess, "call", lambda cmd, cwd=None: launched.append(cmd) or 0)
    monkeypatch.setattr(sys, "argv", [
        "run_blinded_campaign.py", "--arm", "rgbd", "--depth-mode rgbd --no-placement-check",
        "--arm", "plane", "--depth-mode plane", "--pose-list", "config/pose_lists/std_v2.csv",
        "--trials", "25", "--block", "25", "--session", "s", "--key-out", str(tmp_path),
        "--common", "--mode real --confirm-each-trial"])
    assert tool.main() == 2
    assert launched == [] and "for that arm alone" in capsys.readouterr().err


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
    assert ("Placement check: off for every arm (the arms differ in more than "
            "--depth-mode).") in capsys.readouterr().out
    key = json.loads((tmp_path / "key_e3.json").read_text(encoding="utf-8"))
    assert key["placement_check"]["on"] is False


def test_the_runner_accepts_the_flag(monkeypatch):
    path = PROJECT_ROOT / "scripts" / "03_run_experiment.py"
    spec = importlib.util.spec_from_file_location("runner_flag_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(sys, "argv", ["03_run_experiment.py", "--no-placement-check"])
    assert runner.parse_args().no_placement_check is True
    monkeypatch.setattr(sys, "argv", ["03_run_experiment.py"])
    assert runner.parse_args().no_placement_check is False
