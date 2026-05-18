"""
state_machine.py
────────────────
State machine cho một chu trình pick-and-place.

Pure logic — không phụ thuộc RoboDK → testable trực tiếp bằng pytest.
Orchestrator dùng class này để kiểm soát luồng và bắt lỗi chuyển trạng thái
không hợp lệ (vd nhảy thẳng từ IDLE sang PLACE).
"""
from __future__ import annotations

import time
from enum import Enum
from typing import NamedTuple


class PickState(Enum):
    """Các trạng thái trong một chu trình pick-and-place."""

    IDLE = "idle"            # Chờ, chưa làm gì
    DETECT = "detect"        # Nhận detection từ perception
    PLAN = "plan"            # Chọn vật + kiểm tra reachability/collision
    APPROACH = "approach"    # Di chuyển tới điểm approach phía trên vật
    GRASP = "grasp"          # Hạ xuống + đóng gripper
    LIFT = "lift"            # Nâng vật lên
    TRANSFER = "transfer"    # Di chuyển tới vùng đặt
    PLACE = "place"          # Hạ + mở gripper
    RETREAT = "retreat"      # Rút về, quay home
    DONE = "done"            # Hoàn tất thành công
    ERROR = "error"          # Lỗi — cần recovery


# Các chuyển trạng thái hợp lệ. Mọi state có thể nhảy sang ERROR.
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
    """Một bản ghi trong lịch sử chuyển trạng thái."""

    state: PickState
    timestamp: float
    note: str


class InvalidTransitionError(RuntimeError):
    """Chuyển trạng thái không nằm trong sơ đồ hợp lệ."""


class PickPlaceStateMachine:
    """State machine cho chu trình pick-and-place.

    Sử dụng:
        sm = PickPlaceStateMachine()
        sm.transition_to(PickState.DETECT)
        sm.transition_to(PickState.PLAN)
        ...
        sm.fail("gripper_slip")   # nhảy sang ERROR bất cứ lúc nào
    """

    def __init__(self) -> None:
        self._state = PickState.IDLE
        self.history: list[StateRecord] = [
            StateRecord(PickState.IDLE, time.time(), "init")
        ]

    @property
    def state(self) -> PickState:
        """Trạng thái hiện tại."""
        return self._state

    def can_transition(self, target: PickState) -> bool:
        """Kiểm tra có thể chuyển sang `target` không (không thực hiện)."""
        if target == PickState.ERROR:
            return True  # Lỗi có thể xảy ra ở mọi trạng thái
        return target in _VALID_TRANSITIONS.get(self._state, set())

    def transition_to(self, target: PickState, note: str = "") -> None:
        """Chuyển sang trạng thái `target`.

        Raises:
            InvalidTransitionError: Nếu chuyển trạng thái không hợp lệ.
        """
        if not self.can_transition(target):
            raise InvalidTransitionError(
                f"Không thể chuyển {self._state.value} → {target.value}"
            )
        self._state = target
        self.history.append(StateRecord(target, time.time(), note))

    def fail(self, reason: str) -> None:
        """Nhảy sang trạng thái ERROR với lý do (luôn hợp lệ)."""
        self._state = PickState.ERROR
        self.history.append(StateRecord(PickState.ERROR, time.time(), reason))

    def reset(self) -> None:
        """Đưa state machine về IDLE để bắt đầu chu trình mới."""
        self._state = PickState.IDLE
        self.history.append(StateRecord(PickState.IDLE, time.time(), "reset"))
