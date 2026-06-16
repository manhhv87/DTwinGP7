"""
state_machine.py
────────────────
State machine for a single pick-and-place cycle.

Pure logic — no RoboDK dependency → directly testable with pytest.
The orchestrator uses this class to control flow and catch invalid
state transitions (e.g. jumping straight from IDLE to PLACE).
"""
from __future__ import annotations

import time
from enum import Enum
from typing import NamedTuple


class PickState(Enum):
    """States in a pick-and-place cycle."""

    IDLE = "idle"            # Idle, doing nothing
    DETECT = "detect"        # Receiving detection from perception
    PLAN = "plan"            # Select object + check reachability/collision
    APPROACH = "approach"    # Move to approach point above the object
    GRASP = "grasp"          # Lower + close gripper
    LIFT = "lift"            # Lift the object
    TRANSFER = "transfer"    # Move to placement area
    PLACE = "place"          # Lower + open gripper
    RETREAT = "retreat"      # Withdraw, return home
    DONE = "done"            # Completed successfully
    ERROR = "error"          # Error — recovery needed


# Valid state transitions. Any state can jump to ERROR.
_VALID_TRANSITIONS: dict[PickState, set[PickState]] = {
    PickState.IDLE: {PickState.DETECT},
    PickState.DETECT: {PickState.PLAN, PickState.IDLE},
    PickState.PLAN: {PickState.APPROACH, PickState.IDLE},
    PickState.APPROACH: {PickState.GRASP},
    PickState.GRASP: {PickState.LIFT},
    PickState.LIFT: {PickState.TRANSFER},
    PickState.TRANSFER: {PickState.PLACE},
    PickState.PLACE: {PickState.RETREAT},
    PickState.RETREAT: {PickState.DONE},
    PickState.DONE: {PickState.IDLE},
    PickState.ERROR: {PickState.IDLE, PickState.RETREAT},
}


class StateRecord(NamedTuple):
    """A record in the state-transition history."""

    state: PickState
    timestamp: float
    note: str


class InvalidTransitionError(RuntimeError):
    """Transition not present in the valid-transition graph."""


class PickPlaceStateMachine:
    """State machine for a pick-and-place cycle.

    Usage:
        sm = PickPlaceStateMachine()
        sm.transition_to(PickState.DETECT)
        sm.transition_to(PickState.PLAN)
        ...
        sm.fail("gripper_slip")   # jump to ERROR at any time
    """

    def __init__(self) -> None:
        self._state = PickState.IDLE
        self.history: list[StateRecord] = [
            StateRecord(PickState.IDLE, time.time(), "init")
        ]

    @property
    def state(self) -> PickState:
        """Current state."""
        return self._state

    def can_transition(self, target: PickState) -> bool:
        """Check whether transitioning to `target` is allowed (does not execute it)."""
        if target == PickState.ERROR:
            return True  # ERROR can occur from any state
        return target in _VALID_TRANSITIONS.get(self._state, set())

    def transition_to(self, target: PickState, note: str = "") -> None:
        """Transition to state `target`.

        Raises:
            InvalidTransitionError: If the transition is not valid.
        """
        if not self.can_transition(target):
            raise InvalidTransitionError(
                f"Cannot transition {self._state.value} → {target.value}"
            )
        self._state = target
        self.history.append(StateRecord(target, time.time(), note))

    def fail(self, reason: str) -> None:
        """Jump to ERROR state with a reason (always valid)."""
        self._state = PickState.ERROR
        self.history.append(StateRecord(PickState.ERROR, time.time(), reason))

    def reset(self) -> None:
        """Reset the state machine to IDLE to start a new cycle.

        RE-INITIALISES history (does not append) — appending every cycle grew it
        unbounded across thousands of trials. Per-trial history is still complete
        within a cycle; the telemetry CSV keeps the durable record."""
        self._state = PickState.IDLE
        self.history = [StateRecord(PickState.IDLE, time.time(), "reset")]
