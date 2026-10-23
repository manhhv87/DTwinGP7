"""
test_operator_hooks.py
──────────────────────
The operator side of scripts/03_run_experiment.py: the servo gates, the placement
prompt, the placement check before the trial, the verdict, and how they are wired.

What must hold for session 2 of E2:

  * whenever the robot waits for the operator, the servo is OFF and verified;
  * after the operator's ENTER the placement is checked with the servo still OFF,
    stale frames dropped before every check capture; a flagged placement is shown
    and the operator asked to fix it, any line (a bare ENTER too) answering, until
    it is as called or three checks were made; nothing switches the servo meanwhile;
  * only after the last answer does the motion gate switch the servo ON, and the
    trial then detects afresh: no check capture is used for it;
  * keys typed ahead are flushed before every prompt;
  * Ctrl+C or a closed stdin at a check prompt writes aborted_at_check and stops;
  * the hooks are installed only where they belong;
  * a blinded run reports an unexpected error with one neutral line.

The controller is a FakeCell (servo flag and the calls made), the console a script
of answers, and the perception queue a Camera that hands one scripted frame to the
detection window after each drop of stale frames; all three write one timeline.

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
FIX = "   Fix it, TAKE YOUR HAND OUT of the cell, then press ENTER "
RUN = "   TAKE YOUR HAND OUT of the cell, then press ENTER to run the trial "


def _load_runner():
    path = PROJECT_ROOT / "scripts" / "03_run_experiment.py"
    spec = importlib.util.spec_from_file_location("runner_operator_hooks_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUNNER = _load_runner()


class FakeCell:
    """The YRC1000 as the gates see it: a status dict and the calls made, also put
    on a console's timeline when events is set."""

    def __init__(self, servo=False, servo_off_works=True, **status):
        self.status = {"remote": True, "servo": servo, "running": False,
                       "alarm": False, "error": False, "hold": False, **status}
        self.servo_off_works = servo_off_works
        self.calls = []
        self.events = None

    def _call(self, name):
        self.calls.append(name)
        if self.events is not None:
            self.events.append(("cell", name))

    def read_controller_status(self):
        self._call("status")
        return dict(self.status)

    def servo_off(self):
        self._call("servo_off")
        if self.servo_off_works:
            self.status["servo"] = False

    def servo_on(self):
        self._call("servo_on")
        self.status["servo"] = True

    def Stop(self):
        self._call("stop")


class Console:
    """Scripted answers (an exception class is raised instead); records each prompt
    with the servo state at that moment, every line said, and every flush, in one
    ordered event list. No answer left: the stdin is closed (EOFError)."""

    def __init__(self, cell=None, *answers):
        self.cell = cell
        self.answers = list(answers)
        self.events = []

    def ask(self, prompt):
        self.events.append(("ask", prompt, None if self.cell is None
                            else self.cell.status["servo"]))
        if not self.answers:
            raise EOFError
        answer = self.answers.pop(0)
        if isinstance(answer, type) and issubclass(answer, BaseException):
            raise answer
        return answer

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


class TypeAhead(Console):
    """An operator who presses every answer twice: the second press waits in the
    console buffer, where it would answer the next prompt unless flushed."""

    def __init__(self, cell=None, *answers):
        super().__init__(cell, *answers)
        self.pending = []
        self.answered_ahead = 0

    def flush(self):
        super().flush()
        self.pending.clear()

    def ask(self, prompt):
        if self.pending:
            self.answered_ahead += 1
            return self.pending.pop(0)
        answer = super().ask(prompt)
        self.pending.append(answer)
        return answer


class Camera(queue.Queue):
    """The perception queue as 03 uses it. 03 drops the stale frames after every
    ENTER and after the motion gate (get_nowait until Empty); the next detection
    window then reads one fresh frame, the next of the script, and finds nothing
    more. A window not preceded by a drop gets no frame. Every drop and every frame
    read goes on the timeline, the read with the servo state at that moment."""

    def __init__(self, frames, cell=None, events=None):
        super().__init__()
        self.frames = list(frames)
        self.cell = cell
        self.events = [] if events is None else events
        self._fresh = False

    def get_nowait(self):
        self.events.append(("drop",))
        self._fresh = True
        raise queue.Empty

    def get(self, block=True, timeout=None):
        if not (self._fresh and self.frames):
            raise queue.Empty
        self._fresh = False
        msg = self.frames.pop(0)
        self.events.append(("frame", msg.get("frame_path", ""),
                            None if self.cell is None else self.cell.status["servo"]))
        return msg


def _msg(*objects, frame=""):
    return {"timestamp": 0.0, "objects": list(objects), "fps": 20.0, "frame_path": frame}


_PROMPT_NAMES = {"Placed it?": "place", "Fix it": "fix", "run the trial": "run",
                 "did the robot": "verdict"}


def _timeline(events):
    """The flow in short, in order: prompts, drops of stale frames, frames read
    (with the servo state), and the servo switched."""
    out = []
    for e in events:
        if e[0] == "ask":
            out.append("ask:" + next(v for k, v in _PROMPT_NAMES.items() if k in e[1]))
        elif e[0] == "frame":
            out.append(f"frame:{e[1]}:{'on' if e[2] else 'off'}")
        elif e[0] == "drop":
            out.append("drop")
        elif e[0] == "cell" and e[1] in ("servo_on", "servo_off"):
            out.append(e[1])
    return out


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


# ── the prompt at a flagged check ─────────────────────────────────────────────

class TestAsk:
    @staticmethod
    def _ask(*answers, last=False):
        con = Console(None, *answers)
        out = RUNNER.ask_placement_fix(5, [STACK_LINE], last, ask=con.ask, say=con.say,
                                       flush=con.flush)
        return out, con

    def test_a_flag_asks_for_a_fix_after_a_flush(self):
        out, con = self._ask("")
        assert out is None
        assert [e[0] for e in con.events] == ["say", "say", "flush", "ask"]
        assert con.said == ["   Trial 5: not as called:", "     " + STACK_LINE]
        assert con.prompts[0][1] == FIX

    def test_the_last_check_asks_to_make_sure_and_runs_the_trial(self):
        out, con = self._ask("", last=True)
        assert out is None
        assert con.said == ["   Trial 5: still not as called after 3 checks:",
                            "     " + STACK_LINE,
                            "   Check that the part is placed as called (fix it if not)."]
        assert con.prompts[0][1] == RUN

    @pytest.mark.parametrize("answer", ["", "g", "anything"])
    def test_any_line_answers_a_bare_enter_too(self, answer):
        _, con = self._ask(answer, "never read")
        assert len(con.prompts) == 1 and con.answers == ["never read"]

    def test_a_closed_stdin_stops_the_run_and_says_so(self, caplog):
        with caplog.at_level(logging.ERROR, logger="experiment"):
            with pytest.raises(KeyboardInterrupt):
                self._ask()
        assert any("stdin closed at the placement check of trial 5" in r.getMessage()
                   and getattr(r, "console", None) for r in caplog.records)

    def test_the_letters_and_the_check_inside_the_cycle_are_gone(self):
        for name in ("PLACEMENT_ANSWERS", "make_placement_check_hook"):
            assert not hasattr(RUNNER, name)


# ── wiring ────────────────────────────────────────────────────────────────────

ROWS = [{"pose_id": "P0021", "card_id": "20", "x_mm": "575.0", "y_mm": "200.0",
         "yaw_deg": "85.0", "class_hint": "inox_box", "condition": "",
         "stack_on": "carton"}]


def _args(**kw):
    base = dict(mode="real", confirm_each_trial=True, no_placement_check=False,
                blind_label=None, trials=1)
    base.update(kw)
    return argparse.Namespace(**base)


def _orch_stub(cell=None, q=None):
    """What wire_trial_hooks touches on the orchestrator. run_placement_check
    records the trial, the servo state and the frames still queued."""
    stub = types.SimpleNamespace(human_score_hook=None, checks=[])

    def run_placement_check(trial_id, operator):
        stub.checks.append((trial_id, None if cell is None else cell.status["servo"],
                            None if q is None else q.qsize()))

    stub.run_placement_check = run_placement_check
    return stub


@pytest.mark.parametrize("confirm,rows,no_check,check,score,pre", [
    (True, ROWS, False, True, True, True),
    (True, ROWS, True, False, True, True),
    (True, None, False, False, True, True),
    (False, ROWS, False, False, False, True),
    (False, None, False, False, False, False),
])
def test_the_hooks_are_installed_where_they_belong(confirm, rows, no_check, check,
                                                   score, pre):
    cell = FakeCell()
    orch = _orch_stub()
    con = Console(cell, "")
    out = RUNNER.wire_trial_hooks(orch, _args(confirm_each_trial=confirm,
                                              no_placement_check=no_check),
                                  rows, {}, cell, queue.Queue(), LOG,
                                  ask=con.ask, say=con.say, flush=con.flush)
    if out is not None:
        out(1)
    assert (len(orch.checks) == 1) is check
    assert (orch.human_score_hook is not None) is score
    assert (out is not None) is pre
    assert RUNNER.placement_check_enabled(
        _args(confirm_each_trial=confirm, no_placement_check=no_check), rows) is check


def test_the_part_is_placed_and_checked_with_the_servo_off():
    cell = FakeCell(servo=True)
    q = queue.Queue()
    q.put("frame taken while the hand was in the cell")
    con = Console(cell, "")
    orch = _orch_stub(cell, q)
    pre = RUNNER.wire_trial_hooks(orch, _args(), ROWS, {"P0021": (None, 120)}, cell, q,
                                  LOG, ask=con.ask, say=con.say, flush=con.flush)
    pre(1)
    assert [p[2] for p in con.prompts] == [False]
    assert con.prompts[0][1] == ("   Placed it? TAKE YOUR HAND OUT of the cell, "
                                 "then press ENTER ")
    assert [e[0] for e in con.events] == ["flush", "ask"]
    # Checked after the ENTER, with the servo still OFF and the stale frame gone.
    assert orch.checks == [(1, False, 0)]
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
    assert con.prompts == [] and cell.calls == [] and orch.checks == []
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


def _wired(tmp_path, rows, frames, *answers, mode="rgbd", console=Console, ask=None):
    """The real Orchestrator, configured for --mode real, with 03's hooks, a
    FakeCell left with the servo ON, a Camera and a console on one timeline."""
    config = _real_config(mode)
    robot = MotomanHSEBackend()
    cell = FakeCell(servo=True)
    con = console(cell, *answers)
    cell.events = con.events
    cam = Camera(frames, cell, con.events)
    orch = Orchestrator(cam, config=config, robot=robot,
                        logger_obj=TrialLogger(tmp_path / "run.csv"))
    pre = RUNNER.wire_trial_hooks(orch, _args(trials=len(rows)), rows, {}, cell, cam,
                                  LOG, ask=ask or con.ask, say=con.say,
                                  flush=con.flush)
    return orch, robot, cell, con, pre


def _row(tmp_path):
    with (tmp_path / "run.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))[-1]


def _three_flags(config):
    """A part 60 to 62 mm off its card at three checks, set on it at the last
    prompt: the trial's own capture sees it there."""
    return ([_msg(_part(config, 575.0, 80.0 + k, 41.5), frame=f"check{k + 1}.png")
             for k in range(3)]
            + [_msg(_part(config, 575.0, 20.0, 41.5), frame="trial.png")])


def test_a_placement_as_called_runs_at_once_on_a_fresh_capture(tmp_path):
    config = _real_config("rgbd")
    frames = [_msg(_part(config, 578.0, 16.0, 41.5, yaw_off=3.0), frame="check1.png"),
              _msg(_part(config, 572.0, 24.0, 41.5), frame="trial.png")]
    orch, robot, cell, con, pre = _wired(tmp_path, [CALLED_ROW], frames, "", "y")
    orch.run_n_trials(1, pre_trial_hook=pre)
    assert _timeline(con.events) == [
        "servo_off", "ask:place", "drop", "frame:check1.png:off",
        "servo_on", "drop", "frame:trial.png:on",
        "servo_off", "ask:verdict"]
    assert con.said == []                       # nothing flagged, nothing shown
    row = _row(tmp_path)
    assert row["placement_events"] == "none:clear" and row["human_ok"] == "1"
    assert row["frame_path"] == "trial.png"
    assert float(robot.sent[0]["pick"][0, 3]) == pytest.approx(572.0, abs=1.0)


def test_a_flag_then_a_fix_asks_once_and_the_trial_detects_afresh(tmp_path):
    config = _real_config("rgbd")
    alone = _part(config, 575.0, 20.0, 41.5)
    stacked = _part(config, 577.0, 22.0, 41.5 + 84.7, rect=(193.0, 130.2))
    frames = [_msg(alone, frame="check1.png"), _msg(stacked, frame="check2.png"),
              _msg(stacked, frame="trial.png")]
    rows = [dict(CALLED_ROW, stack_on="metal_box")]
    orch, robot, cell, con, pre = _wired(tmp_path, rows, frames, "", "", "y")
    orch.run_n_trials(1, pre_trial_hook=pre)
    assert _timeline(con.events) == [
        "servo_off", "ask:place", "drop", "frame:check1.png:off",
        "ask:fix", "drop", "frame:check2.png:off",
        "servo_on", "drop", "frame:trial.png:on",
        "servo_off", "ask:verdict"]
    assert con.said == ["   Trial 1: not as called:", "     " + STACK_LINE]
    assert con.prompts[1][1] == FIX
    row = _row(tmp_path)
    assert row["placement_events"] == "stack:fixed;none:clear"
    assert row["placement_recaptures"] == "1" and row["placement_override"] == "0"
    assert row["human_ok"] == "1" and len(robot.sent) == 1


def test_three_flags_then_the_trial_runs_as_placed_on_a_fresh_capture(tmp_path):
    config = _real_config("rgbd")
    orch, robot, cell, con, pre = _wired(tmp_path, [CALLED_ROW], _three_flags(config),
                                         "", "", "", "", "y")
    orch.run_n_trials(1, pre_trial_hook=pre)
    timeline = _timeline(con.events)
    assert timeline == [
        "servo_off", "ask:place",
        "drop", "frame:check1.png:off", "ask:fix",
        "drop", "frame:check2.png:off", "ask:fix",
        "drop", "frame:check3.png:off", "ask:run",
        "drop", "servo_on", "drop", "frame:trial.png:on",
        "servo_off", "ask:verdict"]
    # The servo is not switched while the placement is checked, and goes ON only
    # after the last answer, before the trial's own capture.
    checks = timeline[timeline.index("frame:check1.png:off"):timeline.index("ask:run") + 1]
    assert "servo_on" not in checks and "servo_off" not in checks
    assert timeline.index("ask:run") < timeline.index("servo_on") < timeline.index(
        "frame:trial.png:on")
    assert con.said == [
        "   Trial 1: not as called:",
        "     POSITION: part centre 60 mm from card 11; check the card",
        "   Trial 1: not as called:",
        "     POSITION: part centre 61 mm from card 11; check the card",
        "   Trial 1: still not as called after 3 checks:",
        "     POSITION: part centre 62 mm from card 11; check the card",
        "   Check that the part is placed as called (fix it if not)."]
    assert [p[1] for p in con.prompts][1:4] == [FIX, FIX, RUN]
    assert all(p[2] is False for p in con.prompts)
    row = _row(tmp_path)
    assert row["placement_events"] == "position:fixed;position:fixed;position:limit"
    assert row["placement_recaptures"] == "2" and row["placement_override"] == "1"
    assert row["frame_path"] == "trial.png"
    assert float(robot.sent[0]["pick"][1, 3]) == pytest.approx(20.0, abs=1.0)


def test_keys_typed_ahead_are_flushed_before_every_prompt(tmp_path):
    config = _real_config("rgbd")
    orch, robot, cell, con, pre = _wired(tmp_path, [CALLED_ROW], _three_flags(config),
                                         "", "", "", "", "y", console=TypeAhead)
    orch.run_n_trials(1, pre_trial_hook=pre)
    assert con.answered_ahead == 0
    asks = [k for k, e in enumerate(con.events) if e[0] == "ask"]
    assert len(asks) == 5 and all(con.events[k - 1] == ("flush",) for k in asks)
    assert _row(tmp_path)["placement_events"] == (
        "position:fixed;position:fixed;position:limit")


def test_a_bare_enter_on_the_console_answers_the_check(tmp_path, monkeypatch, capsys):
    """input() on a stdin holding ENTER, ENTER, y: placed, fixed, scored."""
    config = _real_config("rgbd")
    stacked = _part(config, 577.0, 22.0, 41.5 + 84.7, rect=(193.0, 130.2))
    frames = [_msg(_part(config, 575.0, 20.0, 41.5)), _msg(stacked), _msg(stacked)]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n\ny\n"))
    orch, robot, cell, con, pre = _wired(tmp_path, [dict(CALLED_ROW, stack_on="metal_box")],
                                         frames, ask=input)
    orch.run_n_trials(1, pre_trial_hook=pre)
    assert FIX in capsys.readouterr().out
    row = _row(tmp_path)
    assert row["placement_events"] == "stack:fixed;none:clear" and row["human_ok"] == "1"


@pytest.mark.parametrize("stop", [KeyboardInterrupt, None])
def test_a_stop_at_a_check_prompt_writes_aborted_at_check(tmp_path, stop):
    """Ctrl+C, or a closed stdin (no answer left), at the fix prompt."""
    config = _real_config("rgbd")
    frames = [_msg(_part(config, 640.0, 20.0, 41.5), frame="check1.png")]
    answers = ("", stop) if stop else ("",)
    orch, robot, cell, con, pre = _wired(tmp_path, [CALLED_ROW], frames, *answers)
    with pytest.raises(KeyboardInterrupt):
        orch.run_n_trials(1, pre_trial_hook=pre)
    assert robot.sent == [] and "servo_on" not in cell.calls
    assert [p[1] for p in con.prompts][-1] == FIX
    row = _row(tmp_path)
    assert row["failure_reason"] == "aborted_at_check" and row["human_ok"] == ""
    assert row["placement_events"] == "position:abort"
    (cap,) = json.loads(row["placement_captures"])
    assert cap["frame"] == "check1.png" and cap["outcome"] == "abort"


@pytest.mark.parametrize("mode", ["rgbd", "plane", "fusion"])
def test_the_trial_stands_on_its_own_capture_in_every_mode(tmp_path, mode):
    """The case behind b9f4e52's blocker: a 227 mm mask for the 180 mm inox, which
    the shape guard refuses. The check saw the part placed as called; the trial's
    own capture is the spilled one, and its refusal stands in every mode."""
    config = _real_config(mode)
    used = "rgbd" if mode == "rgbd" else "plane"
    clean = _part(config, 575.0, 20.0, 41.5, depth_used=used)
    spilled = _part(config, 575.0, 20.0, 42.6, yaw_off=30.0, rect=(227.2, 130.1),
                    depth_used=used)
    frames = [_msg(clean, frame="check1.png"), _msg(spilled, frame="trial.png")]
    orch, robot, cell, con, pre = _wired(tmp_path, [CALLED_ROW], frames, "", mode=mode)
    orch.run_n_trials(1, pre_trial_hook=pre)
    assert robot.sent == [] and orch.sm.history[-1].note == "yaw_geometry_reject"
    row = _row(tmp_path)
    assert row["placement_events"] == "none:clear" and row["frame_path"] == "trial.png"


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
