"""What a blinded run shows on the console, and the operator prompts of 03.

Session 1 of E2 (e2chuan-1, 23 Oct 2026) kept warnings and errors on the blinded
console, and the plane mode's refusal "pick refused (depth_geometry_reject):
commanded top face X mm, depth sees Y mm" appeared 14 times under one code. Under
src/utils/blind_console.py the console shows only lines marked as the same in every
arm; refusals become one neutral line and the reasons stay in the log file.

Run:
    pytest tests/test_blind_console.py -v
"""
from __future__ import annotations

import importlib.util
import io
import json
import logging
import queue
import sys
from pathlib import Path

import pytest

from src.logging.logger import TrialLogger
from src.orchestrator.orchestrator import Orchestrator
from src.utils.blind_console import (
    BlindConsoleFilter, BlindConsoleFormatter, install_blind_console,
)
from tests.test_real_cycle import MotomanHSEBackend, _detection, _plane_z, _real_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Text that tells the depth modes apart, or the outcome, and must stay off the console.
ARM_TEXT = ("rgbd", "plane", "fusion", "depth", "commanded", "reject", "FAIL", "mm")


def _runner():
    path = PROJECT_ROOT / "scripts" / "03_run_experiment.py"
    spec = importlib.util.spec_from_file_location("runner_blind_console_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def logs(tmp_path):
    """Root logger with a blinded console (a string buffer) and a log file."""
    stream = io.StringIO()
    console = logging.StreamHandler(stream)
    log_file = tmp_path / "run.log"
    disk = logging.FileHandler(log_file, encoding="utf-8")
    root = logging.getLogger()
    old_handlers, old_level = root.handlers[:], root.level
    root.handlers = [console, disk]
    root.setLevel(logging.INFO)
    try:
        yield stream, log_file, root
    finally:
        root.handlers, root.level = old_handlers, old_level
        console.close()
        disk.close()


def _record(name, level, msg, console=None):
    rec = logging.LogRecord(name, level, __file__, 1, msg, None, None)
    if console is not None:
        rec.console = console
    return rec


class TestFilter:
    def test_a_marked_line_passes_with_its_replacement(self):
        rec = _record("src.orchestrator.orchestrator", logging.ERROR,
                      "pick refused (depth_geometry_reject): depth sees 195 mm",
                      console="Trial 3: refused before motion (details in the log)")
        assert BlindConsoleFilter().filter(rec)
        assert BlindConsoleFormatter().format(rec) == (
            "Trial 3: refused before motion (details in the log)")

    def test_a_line_marked_true_passes_as_it_is(self):
        rec = _record("experiment", logging.ERROR, "STARTUP REFUSED: HOLD", console=True)
        assert BlindConsoleFilter().filter(rec)
        assert BlindConsoleFormatter().format(rec) == "STARTUP REFUSED: HOLD"

    @pytest.mark.parametrize("name,level", [
        ("src.orchestrator.orchestrator", logging.ERROR),
        ("src.orchestrator.orchestrator", logging.WARNING),
        ("src.perception.postprocess", logging.WARNING),
        ("experiment", logging.ERROR),
        ("src.logging.logger", logging.INFO),
        ("src.orchestrator.backends.motoman_hse", logging.INFO),
    ])
    def test_unmarked_lines_stay_off(self, name, level):
        assert not BlindConsoleFilter().filter(_record(name, level, "x"))

    def test_controller_warnings_pass(self):
        rec = _record("src.orchestrator.digital_twin", logging.ERROR,
                      "Auto-stop triggered by alarm 4315")
        assert BlindConsoleFilter().filter(rec)
        assert BlindConsoleFormatter().format(rec) == (
            "ERROR: Auto-stop triggered by alarm 4315")

    def test_file_handlers_are_left_alone(self, logs):
        stream, log_file, root = logs
        assert install_blind_console(root) >= 1
        console = [h for h in root.handlers if getattr(h, "stream", None) is stream][0]
        disk = [h for h in root.handlers if isinstance(h, logging.FileHandler)][0]
        assert any(isinstance(f, BlindConsoleFilter) for f in console.filters)
        assert disk.filters == [] and not isinstance(disk.formatter, BlindConsoleFormatter)


def _refused_cycle(mode, det_kwargs, pos=(578.3, -29.1), top=40.0):
    config = _real_config(mode)
    x, y = pos
    det = _detection(config, x, y, _plane_z(config, x, y) + top, "inox_box", **det_kwargs)
    q = queue.Queue()
    orch = Orchestrator(q, config=config, robot=MotomanHSEBackend())
    orch.current_expected_class = "inox_box"
    q.put({"timestamp": 0.0, "objects": [det], "fps": 20.0})
    assert orch.run_one_cycle(trial_id=7) is False
    return orch


class TestRefusals:
    def test_a_plane_refusal_reads_as_any_refusal(self, logs):
        """Session 1, code C: the plane estimate on a stack, refused by depth."""
        stream, log_file, root = logs
        install_blind_console(root)
        orch = _refused_cycle("plane", dict(depth_used="plane", frac=0.267, h_depth=194.8,
                                            yaw_deg=48.3))
        assert orch.sm.history[-1].note == "depth_geometry_reject"
        public = stream.getvalue()
        assert public.strip() == "Trial 7: refused before motion (details in the log)"
        for word in ARM_TEXT:
            assert word not in public
        for h in root.handlers:
            h.flush()
        private = log_file.read_text(encoding="utf-8")
        assert "depth sees 194.8 mm" in private and "depth_geometry_reject" in private

    def test_an_rgbd_refusal_reads_the_same(self, logs):
        """A depth valid fraction under 0.15: refused only when depth commands."""
        stream, log_file, root = logs
        install_blind_console(root)
        orch = _refused_cycle("rgbd", dict(depth_used="rgbd", frac=0.05, h_depth=40.8,
                                           yaw_deg=-40.2), pos=(525.0, 140.0))
        assert orch.sm.history[-1].note == "pick_z_safety_reject"
        assert stream.getvalue().strip() == (
            "Trial 7: refused before motion (details in the log)")

    def test_an_unblinded_console_keeps_the_reason(self, logs):
        stream, log_file, root = logs
        _refused_cycle("plane", dict(depth_used="plane", frac=0.267, h_depth=194.8,
                                     yaw_deg=48.3))
        assert "depth sees 194.8 mm" in stream.getvalue()


class TestPrompt:
    ROW = {"pose_id": "P0021", "card_id": "20", "x_mm": "575.0", "y_mm": "200.0",
           "yaw_deg": "85.0", "class_hint": "inox_box", "condition": "",
           "stack_on": "carton"}

    def test_the_blinded_prompt_carries_position_size_and_stack(self):
        lines = _runner().placement_prompt_lines(3, 25, self.ROW, {"P0021": (None, 120)}, "B")
        assert lines[0] == ("[B] Trial 3/25, PLACE: card=20  x=575.0 mm  y=200.0 mm  "
                            "yaw=85.0 deg  class=inox_box")
        assert lines[1] == ("    STACKED: put the carton(120) down on the card FIRST, "
                            "then the inox_box on top of it.")

    def test_a_carton_target_names_its_size(self):
        row = dict(self.ROW, pose_id="P0033", class_hint="carton", stack_on="")
        lines = _runner().placement_prompt_lines(8, 25, row, {"P0033": (120, None)})
        assert lines[0].endswith("class=carton(120)") and len(lines) == 1

    def test_without_a_size_the_class_alone(self):
        row = dict(self.ROW, class_hint="carton", stack_on="")
        assert _runner().placement_prompt_lines(1, 1, row, {})[0].endswith("class=carton")


class TestAsk:
    def _ask(self, answers, can=True):
        said, calls = [], []
        answers = list(answers)

        def ask(prompt):
            calls.append(("ask", prompt))
            if not answers:
                raise EOFError
            return answers.pop(0)

        out = _runner().ask_placement_fix(
            5, ["STACK: top at 42 mm"], can, ask=ask,
            say=lambda *a, **k: said.append(" ".join(str(x) for x in a)),
            make_safe=lambda: calls.append(("safe",)),
            make_ready=lambda: calls.append(("ready",)))
        return out, said, calls

    def test_enter_goes_on(self):
        out, said, calls = self._ask([""])
        assert out == "proceed" and ("safe",) not in calls
        assert any("STACK: top at 42 mm" in s for s in said)

    def test_a_fix_makes_the_cell_safe_first(self):
        """The check comes after the operator confirmed the cell clear and the
        servo went ON: the prompt says so, and s switches it OFF before the fix."""
        out, said, calls = self._ask(["s", ""])
        assert out == "recapture"
        kinds = [c[0] for c in calls]
        assert kinds == ["ask", "safe", "ask", "ready"]
        assert "Servo is ON" in calls[0][1] and "[s] = switch it OFF" in calls[0][1]
        assert "Servo OFF" in calls[2][1] and "TAKE YOUR HAND OUT" in calls[2][1]

    def test_without_a_servo_the_prompt_does_not_name_one(self):
        prompts = []
        answers = ["s", ""]
        out = _runner().ask_placement_fix(
            1, ["YAW: part turned 30 deg"], True,
            ask=lambda p: prompts.append(p) or answers.pop(0), say=lambda *a, **k: None)
        assert out == "recapture" and not any("Servo" in p for p in prompts)

    def test_other_keys_ask_again(self):
        out, said, calls = self._ask(["x", "y", ""])
        assert out == "proceed" and [c[0] for c in calls] == ["ask", "ask", "ask"]

    def test_at_the_limit_nothing_is_asked(self):
        out, said, calls = self._ask([], can=False)
        assert out == "proceed" and calls == []
        assert any("limit" in s for s in said)

    def test_a_closed_stdin_stops_the_run(self):
        with pytest.raises(KeyboardInterrupt):
            self._ask([])
        with pytest.raises(KeyboardInterrupt):
            self._ask(["s"])


def test_a_blinded_sim_run_shows_prompts_and_neutral_lines_only(tmp_path, monkeypatch,
                                                                capsys, logs):
    """03 itself, headless sim, blinded, on two pose-list rows. The scripted sim
    detections are of other classes, so both trials end before motion."""
    stream, log_file, root = logs
    runner = _runner()
    monkeypatch.setattr(runner, "setup_logging",
                        lambda *a, **k: logging.getLogger("experiment"))
    real_load = runner.load_yaml

    def load_yaml(path):
        cfg = real_load(path)
        cfg["detection_timeout_s"] = 0.2
        return cfg

    monkeypatch.setattr(runner, "load_yaml", load_yaml)
    monkeypatch.setattr(runner, "TrialLogger", lambda path, **kw: TrialLogger(
        tmp_path / Path(path).name, **kw))
    monkeypatch.setattr(sys, "argv", [
        "03_run_experiment.py", "--mode", "sim", "--headless", "--trials", "2",
        "--pose-list", "config/pose_lists/std_v2.csv", "--pose-slice", "20:22",
        "--blind-label", "B"])
    assert runner.main() == 0
    public = stream.getvalue() + capsys.readouterr().out
    assert "BLINDED RUN, arm B" in public
    assert ("[B] Trial 1/2, PLACE: card=20  x=575.0 mm  y=200.0 mm  yaw=85.0 deg  "
            "class=inox_box") in public
    assert "STACKED: put the carton(120) down on the card FIRST" in public
    assert "Trial 1: refused before motion (details in the log)" in public
    for word in ("rgbd", "depth", "FAIL", "NOT FOUND", "success_rate"):
        assert word not in public
    for h in root.handlers:
        h.flush()
    assert "POSE-LIST CLASS LOCK" in log_file.read_text(encoding="utf-8")
    meta = json.loads(next(tmp_path.glob("experiment_headless_*.metadata.json"))
                      .read_text(encoding="utf-8"))
    code = meta["provenance"]["code"]
    assert "commit" in code or "error" in code
