"""
test_operator_hooks.py
──────────────────────
The operator side of scripts/03_run_experiment.py: the servo gates, the placement
prompt, the placement check's question, the verdict, and how they are wired.

What must hold for session 2 of E2:

  * whenever the robot waits for the operator, the servo is OFF and verified;
  * at a placement check the flags are shown, the servo goes OFF, and only a typed
    letter answers: g goes on with the detection that was checked, f captures again;
    a bare ENTER, or a key typed before the prompt, answers nothing;
  * after either answer the motion gate switches the servo ON and stale frames are
    dropped before another frame is read;
  * the hooks are installed only where they belong;
  * a blinded run reports an unexpected error with one neutral line.

The controller is a FakeCell (servo flag and the calls made) and the console a
script of answers that records, for every prompt, whether the servo was ON.

Run:
    pytest tests/test_operator_hooks.py -v
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import logging
import queue
import sys
import types
from pathlib import Path

import pytest

from src.logging.logger import TrialLogger
from src.orchestrator.orchestrator import Orchestrator
from src.utils.blind_console import install_blind_console
from tests.test_placement_cycle import _part
from tests.test_real_cycle import MotomanHSEBackend, _real_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG = logging.getLogger("test_operator_hooks")
STACK_LINE = ("STACK: top at 42 mm, 126 mm expected for inox_box on metal_box; "
              "check the support")


def _load_runner():
    path = PROJECT_ROOT / "scripts" / "03_run_experiment.py"
    spec = importlib.util.spec_from_file_location("runner_operator_hooks_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUNNER = _load_runner()


class FakeCell:
    """The YRC1000 as the gates see it: a status dict and the calls made."""

    def __init__(self, servo=False, servo_off_works=True, **status):
        self.status = {"remote": True, "servo": servo, "running": False,
                       "alarm": False, "error": False, "hold": False, **status}
        self.servo_off_works = servo_off_works
        self.calls = []

    def read_controller_status(self):
        self.calls.append("status")
        return dict(self.status)

    def servo_off(self):
        self.calls.append("servo_off")
        if self.servo_off_works:
            self.status["servo"] = False

    def servo_on(self):
        self.calls.append("servo_on")
        self.status["servo"] = True

    def Stop(self):
        self.calls.append("stop")


class Console:
    """Scripted answers; records each prompt with the servo state at that moment,
    every line said, and every flush, in one ordered event list."""

    def __init__(self, cell=None, *answers):
        self.cell = cell
        self.answers = list(answers)
        self.events = []

    def ask(self, prompt):
        self.events.append(("ask", prompt, None if self.cell is None
                            else self.cell.status["servo"]))
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)

    def say(self, *a, **k):
        self.events.append(("say", " ".join(str(x) for x in a)))

    def flush(self):
        self.events.append(("flush",))

    @property
    def prompts(self):
        return [e for e in self.events if e[0] == "ask"]

    @property
    def said(self):
        return [e[1] for e in self.events if e[0] == "say"]


class ScriptedQueue(queue.Queue):
    """Hands out one scripted perception message per detection window; the
    operator-side drains find nothing, so every scripted frame is read in turn."""

    def get_nowait(self):
        raise queue.Empty


def _msg(*objects, frame=""):
    return {"timestamp": 0.0, "objects": list(objects), "fps": 20.0, "frame_path": frame}


# ── the gates ─────────────────────────────────────────────────────────────────

class TestGates:
    def test_the_placement_gate_switches_the_servo_off_and_verifies_it(self):
        cell = FakeCell(servo=True)
        RUNNER.placement_gate(cell, 3, LOG)
        assert cell.calls == ["status", "servo_off", "status"]
        assert cell.status["servo"] is False

    def test_a_servo_that_stays_on_refuses_placement(self):
        cell = FakeCell(servo=True, servo_off_works=False)
        with pytest.raises(RUNNER.CellGateError, match="SERVO OFF could not be verified"):
            RUNNER.placement_gate(cell, 3, LOG)

    def test_a_running_job_refuses_placement(self):
        with pytest.raises(RUNNER.CellGateError, match="still running"):
            RUNNER.placement_gate(FakeCell(running=True), 3, LOG)

    def test_the_motion_gate_switches_the_servo_on(self):
        cell = FakeCell()
        RUNNER.motion_gate(cell, LOG)
        assert cell.calls == ["status", "servo_on", "status"] and cell.status["servo"]

    @pytest.mark.parametrize("state", ["alarm", "error", "hold", "running"])
    def test_an_unsafe_controller_refuses_motion_before_servo_on(self, state):
        cell = FakeCell(**{state: True})
        with pytest.raises(RUNNER.CellGateError, match="Motion refused"):
            RUNNER.motion_gate(cell, LOG)
        assert "servo_on" not in cell.calls

    def test_not_remote_refuses_motion(self):
        with pytest.raises(RUNNER.CellGateError, match="REMOTE"):
            RUNNER.motion_gate(FakeCell(remote=False), LOG)

    def test_a_gate_error_is_still_a_runtime_error(self):
        assert issubclass(RUNNER.CellGateError, RuntimeError)

    def test_stale_frames_are_dropped(self):
        q = queue.Queue()
        for i in range(3):
            q.put(i)
        assert RUNNER.drop_stale_frames(q, LOG) == 3 and q.empty()


# ── keys typed ahead ──────────────────────────────────────────────────────────

class TestFlush:
    class _Tty:
        def isatty(self):
            return True

    def test_a_pipe_is_left_alone(self):
        stream = io.StringIO("g\n")
        assert RUNNER.flush_console_input(stream) == 0
        assert stream.read() == "g\n"

    def test_windows_discards_the_pending_keys(self, monkeypatch):
        pending = list("\r\rg")
        fake = types.SimpleNamespace(kbhit=lambda: bool(pending),
                                     getwch=lambda: pending.pop(0))
        monkeypatch.setitem(sys.modules, "msvcrt", fake)
        assert RUNNER.flush_console_input(self._Tty(), platform="nt") == 3
        assert pending == []

    def test_elsewhere_the_terminal_input_is_flushed(self, monkeypatch):
        flushed = []
        fake = types.SimpleNamespace(TCIFLUSH=7,
                                     tcflush=lambda fd, how: flushed.append(how))
        monkeypatch.setitem(sys.modules, "termios", fake)
        RUNNER.flush_console_input(self._Tty(), platform="posix")
        assert flushed == [7]


# ── the question at a placement check ─────────────────────────────────────────

class TestAsk:
    @staticmethod
    def _ask(*answers, can=True, servo=True):
        cell = FakeCell(servo=True) if servo else None
        con = Console(cell, *answers)

        def make_safe():
            con.events.append(("safe",))
            RUNNER.placement_gate(cell, 5, LOG)

        out = RUNNER.ask_placement_fix(
            5, [STACK_LINE], can, ask=con.ask, say=con.say,
            make_safe=make_safe if servo else None,
            make_ready=lambda: con.events.append(("ready",)), flush=con.flush)
        return out, con

    def test_g_goes_on_with_the_checked_detection(self):
        out, con = self._ask("g")
        assert out == "proceed"
        kinds = [e[0] for e in con.events]
        # flags first, then the servo OFF, then the question, then the motion gate
        assert kinds == ["say", "say", "safe", "say", "flush", "ask", "ready"]
        assert con.said[0] == "   Trial 5: the scene does not match the call:"
        assert con.said[1] == "     " + STACK_LINE
        assert con.said[2] == ("   Servo is OFF. Placed as called: move nothing. "
                               "Not as called: fix it now.")
        assert con.prompts[0][1] == ("   TAKE YOUR HAND OUT of the cell, then type  "
                                     "[g] = nothing moved, go on   "
                                     "[f] = fixed, capture again  ")
        assert con.prompts[0][2] is False                     # asked with servo OFF

    def test_f_captures_again(self):
        out, con = self._ask("F")
        assert out == "recapture" and [e[0] for e in con.events][-1] == "ready"

    def test_a_bare_enter_or_another_key_answers_nothing(self):
        """A second ENTER pressed at the placement prompt used to answer the check
        (b9f4e52, 'position:proceed' for a part 70 mm off its card)."""
        out, con = self._ask("", "s", "y", "g")
        assert out == "proceed"
        assert con.said.count("   Type g or f.") == 3
        assert [e[0] for e in con.events].count("flush") == 4   # before every prompt

    def test_at_the_limit_only_g_is_taken(self):
        out, con = self._ask("f", "g", can=False)
        assert out == "proceed"
        assert con.said[2] == ("   Servo is OFF. Recapture limit reached: "
                               "this capture is used.")
        assert con.prompts[0][1] == ("   TAKE YOUR HAND OUT of the cell, then type  "
                                     "[g] = go on (Ctrl+C stops the run)  ")
        assert con.said[3] == "   Type g, or press Ctrl+C to stop the run."

    def test_the_simulator_names_no_servo(self):
        out, con = self._ask("g", servo=False)
        assert out == "proceed"
        assert not any("Servo" in s for s in con.said)
        assert con.said[2] == "   Placed as called: move nothing. Not as called: fix it now."

    def test_a_closed_stdin_stops_the_run_and_says_so(self, caplog):
        with caplog.at_level(logging.ERROR, logger="experiment"):
            with pytest.raises(KeyboardInterrupt):
                self._ask()
        assert any("stdin closed at the placement check of trial 5" in r.getMessage()
                   and getattr(r, "console", None) for r in caplog.records)

    def test_the_answers(self):
        assert RUNNER.PLACEMENT_ANSWERS == {"g": "proceed", "f": "recapture"}


# ── the hook the orchestrator calls ───────────────────────────────────────────

class TestHookFactory:
    @staticmethod
    def _hook(cell, q, *answers, real=True):
        con = Console(cell, *answers)
        hook = RUNNER.make_placement_check_hook(cell, q, LOG, real=real, ask=con.ask,
                                                say=con.say, flush=con.flush)
        return hook, con

    @pytest.mark.parametrize("answer,out", [("f", "recapture"), ("g", "proceed")])
    def test_the_servo_is_off_while_asked_and_on_again_after(self, answer, out):
        cell = FakeCell(servo=True)          # the motion gate left it ON to detect
        q = queue.Queue()
        q.put("stale")
        hook, con = self._hook(cell, q, answer)
        assert hook(4, [STACK_LINE], True) == out
        assert [p[2] for p in con.prompts] == [False]
        assert cell.calls == ["status", "servo_off", "status",      # placement gate
                              "status", "servo_on", "status"]      # motion gate
        assert cell.status["servo"] is True and q.empty()

    def test_a_servo_that_will_not_switch_off_stops_before_the_question(self):
        cell = FakeCell(servo=True, servo_off_works=False)
        hook, con = self._hook(cell, queue.Queue(), "g")
        with pytest.raises(RUNNER.CellGateError):
            hook(4, [STACK_LINE], True)
        assert con.prompts == []

    def test_the_simulator_only_drops_the_frames(self):
        q = queue.Queue()
        q.put("stale")
        hook, con = self._hook(None, q, "f", real=False)
        assert hook(1, [STACK_LINE], True) == "recapture" and q.empty()


# ── wiring ────────────────────────────────────────────────────────────────────

ROWS = [{"pose_id": "P0021", "card_id": "20", "x_mm": "575.0", "y_mm": "200.0",
         "yaw_deg": "85.0", "class_hint": "inox_box", "condition": "",
         "stack_on": "carton"}]


def _args(**kw):
    base = dict(mode="real", confirm_each_trial=True, no_placement_check=False,
                blind_label=None, trials=1)
    base.update(kw)
    return argparse.Namespace(**base)


def _orch_stub():
    return types.SimpleNamespace(placement_check_hook=None, human_score_hook=None)


@pytest.mark.parametrize("confirm,rows,no_check,check,score,pre", [
    (True, ROWS, False, True, True, True),
    (True, ROWS, True, False, True, True),
    (True, None, False, False, True, True),
    (False, ROWS, False, False, False, True),
    (False, None, False, False, False, False),
])
def test_the_hooks_are_installed_where_they_belong(confirm, rows, no_check, check,
                                                   score, pre):
    orch = _orch_stub()
    out = RUNNER.wire_trial_hooks(orch, _args(confirm_each_trial=confirm,
                                              no_placement_check=no_check),
                                  rows, {}, FakeCell(), queue.Queue(), LOG)
    assert (orch.placement_check_hook is not None) is check
    assert (orch.human_score_hook is not None) is score
    assert (out is not None) is pre
    assert RUNNER.placement_check_enabled(
        _args(confirm_each_trial=confirm, no_placement_check=no_check), rows) is check


def test_the_part_is_placed_with_the_servo_off_and_the_frames_of_it_dropped():
    cell = FakeCell(servo=True)
    q = queue.Queue()
    q.put("frame taken while the hand was in the cell")
    con = Console(cell, "")
    orch = _orch_stub()
    pre = RUNNER.wire_trial_hooks(orch, _args(), ROWS, {"P0021": (None, 120)}, cell, q,
                                  LOG, ask=con.ask, say=con.say, flush=con.flush)
    pre(1)
    assert [p[2] for p in con.prompts] == [False]
    assert con.prompts[0][1] == ("   Placed it? TAKE YOUR HAND OUT of the cell, "
                                 "then press ENTER ")
    assert [e[0] for e in con.events] == ["flush", "ask"]
    assert cell.calls == ["status", "servo_off", "status",
                          "status", "servo_on", "status"]
    assert q.empty()
    assert orch.current_called_pose.stack_on_carton_mm == 120
    assert orch.current_expected_class == "inox_box"


def test_without_confirm_the_prompt_is_logged_and_nothing_asked(caplog):
    cell = FakeCell()
    con = Console(cell)
    orch = _orch_stub()
    pre = RUNNER.wire_trial_hooks(orch, _args(confirm_each_trial=False, mode="sim"), ROWS,
                                  {"P0021": (None, 120)}, None, queue.Queue(), LOG,
                                  ask=con.ask, say=con.say, flush=con.flush)
    with caplog.at_level(logging.INFO, logger=LOG.name):
        pre(1)
    assert con.prompts == [] and cell.calls == []
    assert any("STACKED: put the carton(120) down" in r.getMessage()
               for r in caplog.records)


@pytest.mark.parametrize("answers,verdict", [(["y"], 1), (["n"], 0), (["x", "y"], 1)])
def test_the_verdict_is_asked_with_the_servo_off(answers, verdict):
    cell = FakeCell(servo=True)
    con = Console(cell, *answers)
    orch = _orch_stub()
    RUNNER.wire_trial_hooks(orch, _args(), ROWS, {}, cell, queue.Queue(), LOG,
                            ask=con.ask, say=con.say, flush=con.flush)
    assert orch.human_score_hook(3, True) == verdict
    assert all(p[2] is False for p in con.prompts)
    assert cell.calls == ["servo_off", "status"]
    assert ("   Type y or n." in con.said) is (len(answers) > 1)
    assert [e[0] for e in con.events][0] == "flush"


def test_a_servo_that_stays_on_after_a_cycle_stops_the_run():
    cell = FakeCell(servo=True, servo_off_works=False)
    con = Console(cell, "y")
    orch = _orch_stub()
    RUNNER.wire_trial_hooks(orch, _args(), ROWS, {}, cell, queue.Queue(), LOG,
                            ask=con.ask, say=con.say, flush=con.flush)
    with pytest.raises(KeyboardInterrupt):
        orch.human_score_hook(3, True)
    assert "stop" in cell.calls and con.prompts == []


# ── end to end: the wired hooks around run_one_cycle ──────────────────────────

CALLED_ROW = {"pose_id": "P0100", "card_id": "11", "x_mm": "575.0", "y_mm": "20.0",
              "yaw_deg": "40.0", "class_hint": "inox_box", "condition": ""}


def _wired(tmp_path, mode, rows, frames, *answers, **kw):
    config = _real_config(mode)
    robot = MotomanHSEBackend()
    q = ScriptedQueue()
    for objects in frames:
        q.put(_msg(*objects))
    orch = Orchestrator(q, config=config, robot=robot,
                        logger_obj=TrialLogger(tmp_path / "run.csv"))
    cell = FakeCell()
    con = Console(cell, *answers)
    pre = RUNNER.wire_trial_hooks(orch, _args(trials=len(rows)), rows, {}, cell, q, LOG,
                                  ask=kw.get("ask", con.ask), say=con.say,
                                  flush=con.flush)
    return orch, robot, q, cell, con, pre, config


def _row(tmp_path):
    with (tmp_path / "run.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))[-1]


def test_a_fixed_stack_is_captured_again_with_the_servo_off_throughout(tmp_path):
    config = _real_config("rgbd")
    alone = _part(config, 575.0, 20.0, 41.5)
    stacked = _part(config, 577.0, 22.0, 41.5 + 84.7, rect=(193.0, 130.2))
    rows = [dict(CALLED_ROW, stack_on="metal_box")]
    orch, robot, q, cell, con, pre, _ = _wired(tmp_path, "rgbd", rows,
                                               [[alone], [stacked]], "", "f", "y")
    orch.run_n_trials(1, pre_trial_hook=pre)
    assert len(robot.sent) == 1
    row = _row(tmp_path)
    assert row["placement_events"] == "stack:recapture;none:clear"
    assert row["human_ok"] == "1" and row["placement_recaptures"] == "1"
    # placed, fixed, scored: the servo was OFF at every one of the three prompts
    assert [p[2] for p in con.prompts] == [False, False, False]
    assert cell.calls.count("servo_on") == 2          # after placing, after fixing
    assert [c["outcome"] for c in json.loads(row["placement_captures"])] == [
        "recapture", "clear"]


@pytest.mark.parametrize("mode", ["rgbd", "plane", "fusion"])
def test_looking_closely_does_not_retry_a_perception_failure(tmp_path, mode):
    """The case behind b9f4e52's blocker: a 227 mm mask for the 180 mm inox, which
    the shape guard refuses, read 30 deg off the call. The operator looks at the
    part with the servo OFF, finds it placed as called and types g: the refusal
    stands in every mode and the clean frame behind it is never read."""
    config = _real_config(mode)
    used = "rgbd" if mode == "rgbd" else "plane"
    spilled = _part(config, 575.0, 20.0, 42.6, yaw_off=30.0, rect=(227.2, 130.1),
                    depth_used=used)
    clean = _part(config, 575.0, 20.0, 41.5, depth_used=used)
    orch, robot, q, cell, con, pre, _ = _wired(tmp_path, mode, [CALLED_ROW],
                                               [[spilled], [clean]], "", "g")
    orch.run_n_trials(1, pre_trial_hook=pre)
    assert robot.sent == [] and orch.sm.history[-1].note == "yaw_geometry_reject"
    assert q.qsize() == 1                               # the clean frame, unread
    assert [p[2] for p in con.prompts] == [False, False]
    row = _row(tmp_path)
    assert row["placement_events"] == "yaw:proceed" and row["placement_override"] == "1"


def test_a_second_enter_at_the_placement_prompt_does_not_answer_the_check(
        tmp_path, monkeypatch):
    """b9f4e52: two ENTERs at 'Placed it?' and the second one answered the check,
    logged as the operator going on over a part 70 mm off its card."""
    config = _real_config("rgbd")
    off_card = _part(config, 575.0, 20.0 + 70.0, 41.5)
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n\nf\n"))
    on_card = _part(config, 575.0, 20.0, 41.5)
    orch, robot, q, cell, con, pre, _ = _wired(tmp_path, "rgbd", [CALLED_ROW],
                                               [[off_card], [on_card]], ask=input)
    orch.human_score_hook = None
    orch.run_n_trials(1, pre_trial_hook=pre)
    row = _row(tmp_path)
    assert "   Type g or f." in con.said
    assert row["placement_events"] == "position:recapture;none:clear"
    assert float(robot.sent[0]["pick"][1, 3]) == pytest.approx(20.0, abs=1.0)


def test_main_wires_the_hooks_through_one_function(monkeypatch, tmp_path):
    runner = _load_runner()
    calls = []
    real = runner.wire_trial_hooks

    def recorder(orch, args, *rest, **kw):
        calls.append(args)
        return real(orch, args, *rest, **kw)

    monkeypatch.setattr(runner, "wire_trial_hooks", recorder)
    monkeypatch.setattr(runner, "setup_logging", lambda *a, **k: logging.getLogger("x"))
    monkeypatch.setattr(runner, "TrialLogger", lambda path, **kw: TrialLogger(
        tmp_path / Path(path).name, **kw))
    monkeypatch.setattr(sys, "argv", ["03_run_experiment.py", "--mode", "sim",
                                      "--headless", "--trials", "1"])
    assert runner.main() == 0
    assert len(calls) == 1 and calls[0].confirm_each_trial is False


# ── a blinded run that stops on an error ──────────────────────────────────────

@pytest.fixture
def blind_logs(tmp_path):
    """Root logger with a blinded console (a string buffer) and a log file."""
    stream = io.StringIO()
    console = logging.StreamHandler(stream)
    log_file = tmp_path / "run.log"
    disk = logging.FileHandler(log_file, encoding="utf-8")
    root = logging.getLogger()
    old_handlers, old_level = root.handlers[:], root.level
    root.handlers = [console, disk]
    root.setLevel(logging.INFO)
    install_blind_console(root)
    try:
        yield stream, log_file, root
    finally:
        root.handlers, root.level = old_handlers, old_level
        console.close()
        disk.close()


def _raise(exc):
    def drive():
        raise exc
    return drive


def test_a_blinded_run_shows_a_controller_refusal_as_it_is(blind_logs):
    stream, log_file, root = blind_logs
    err = RUNNER.CellGateError("Motion refused: controller HOLD is active: {'hold': True}")
    assert RUNNER.run_reporting_errors(_raise(err), LOG, blinded=True) == 1
    assert stream.getvalue().strip() == (
        "Run stopped: Motion refused: controller HOLD is active: {'hold': True}")
    for h in root.handlers:
        h.flush()
    assert "Traceback" in log_file.read_text(encoding="utf-8")


def test_a_blinded_run_hides_the_text_of_any_other_error(blind_logs):
    stream, log_file, root = blind_logs
    err = RuntimeError("Invalid computed P2.Z: 712.3")
    assert RUNNER.run_reporting_errors(_raise(err), LOG, blinded=True) == 1
    assert stream.getvalue().strip() == RUNNER.BLIND_RUN_ERROR
    for h in root.handlers:
        h.flush()
    assert "712.3" in log_file.read_text(encoding="utf-8")


def test_unblinded_errors_and_a_stop_propagate(blind_logs):
    with pytest.raises(RuntimeError):
        RUNNER.run_reporting_errors(_raise(RuntimeError("x")), LOG, blinded=False)
    with pytest.raises(KeyboardInterrupt):
        RUNNER.run_reporting_errors(_raise(KeyboardInterrupt()), LOG, blinded=True)
    assert RUNNER.run_reporting_errors(lambda: None, LOG, blinded=True) == 0
