"""
test_state_machine.py
─────────────────────
Unit tests cho state machine pick-and-place — pure logic.

Run:
    pytest tests/test_state_machine.py -v
"""
from __future__ import annotations

import pytest

from src.orchestrator.state_machine import (
    InvalidTransitionError,
    PickPlaceStateMachine,
    PickState,
)


class TestTransitions:
    def test_initial_state_is_idle(self):
        sm = PickPlaceStateMachine()
        assert sm.state == PickState.IDLE

    def test_full_happy_path(self):
        sm = PickPlaceStateMachine()
        path = [
            PickState.DETECT, PickState.PLAN, PickState.APPROACH,
            PickState.GRASP, PickState.LIFT, PickState.TRANSFER,
            PickState.PLACE, PickState.RETREAT, PickState.DONE,
        ]
        for st in path:
            sm.transition_to(st)
        assert sm.state == PickState.DONE

    def test_invalid_transition_raises(self):
        sm = PickPlaceStateMachine()
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(PickState.PLACE)  # IDLE → PLACE không hợp lệ

    def test_can_transition_query(self):
        sm = PickPlaceStateMachine()
        assert sm.can_transition(PickState.DETECT)
        assert not sm.can_transition(PickState.GRASP)


class TestFailAndReset:
    def test_fail_from_any_state(self):
        sm = PickPlaceStateMachine()
        sm.transition_to(PickState.DETECT)
        sm.transition_to(PickState.PLAN)
        sm.fail("gripper_slip")
        assert sm.state == PickState.ERROR
        assert sm.history[-1].note == "gripper_slip"

    def test_reset_returns_to_idle(self):
        sm = PickPlaceStateMachine()
        sm.transition_to(PickState.DETECT)
        sm.fail("x")
        sm.reset()
        assert sm.state == PickState.IDLE


class TestHistory:
    def test_history_records_transitions(self):
        sm = PickPlaceStateMachine()
        sm.transition_to(PickState.DETECT, note="got frame")
        # init + DETECT
        assert len(sm.history) == 2
        assert sm.history[-1].state == PickState.DETECT
        assert sm.history[-1].note == "got frame"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
