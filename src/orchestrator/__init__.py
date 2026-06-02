"""
orchestrator — Coordinates pick-and-place: perception → transform → RoboDK.

Public API:
    Orchestrator: class that coordinates the pick-and-place cycle
    PickPlaceStateMachine, PickState: state machine
    camera_to_base, transform_point, make_grasp_pose,
    load_calibration, save_calibration: coordinate transforms
"""
from .coord_conv import (
    camera_to_base,
    load_calibration,
    make_grasp_pose,
    save_calibration,
    transform_point,
)
from .orchestrator import Orchestrator
from .state_machine import (
    InvalidTransitionError,
    PickPlaceStateMachine,
    PickState,
)

__all__ = [
    "Orchestrator",
    "PickPlaceStateMachine",
    "PickState",
    "InvalidTransitionError",
    "camera_to_base",
    "transform_point",
    "make_grasp_pose",
    "load_calibration",
    "save_calibration",
]
