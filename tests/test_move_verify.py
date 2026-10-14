"""The reached-pose check after each Cartesian motion."""
import numpy as np
import pytest

from src.orchestrator.kinematics.urdf_chain import forward_kinematics_urdf, gp7_urdf
from src.orchestrator.orchestrator import Orchestrator


class _Robot:
    """Reports a fixed joint vector, whatever it is told to do."""

    def __init__(self, joints_deg):
        self._j = list(joints_deg)
        self.moves = []

    def Joints(self):
        return list(self._j)

    def MoveJ(self, target):
        self.moves.append(("movj", target))

    def MoveL(self, target):
        self.moves.append(("movl", target))


def _orch(joints_deg, tol_mm=5.0):
    o = Orchestrator.__new__(Orchestrator)
    o.config = {
        "robot_base_xyz_mm": (0.0, 0.0, 630.0),
        "robot_base_rpy_deg": (0.0, 0.0, 0.0),
        "robot_tool_offset_mm": 0.0,
        "move_verify_tol_mm": tol_mm,
    }
    o.robot = _Robot(joints_deg)
    return o


JOINTS = [10.0, -20.0, 15.0, 5.0, -30.0, 25.0]


def _pose_of(joints_deg, orch):
    return forward_kinematics_urdf(
        orch._dh_model(), [float(np.deg2rad(q)) for q in joints_deg])


class TestVerifyReached:
    def test_passes_when_the_robot_is_at_the_target(self):
        o = _orch(JOINTS)
        o._verify_reached(_pose_of(JOINTS, o), "MoveJ")      # must not raise

    def test_raises_when_the_robot_stopped_short(self):
        o = _orch(JOINTS)
        target = _pose_of(JOINTS, o).copy()
        target[0, 3] += 40.0                                  # 40 mm away
        with pytest.raises(RuntimeError, match="stopped"):
            o._verify_reached(target, "MoveL")

    def test_tolerance_zero_disables_the_check(self):
        o = _orch(JOINTS, tol_mm=0.0)
        target = _pose_of(JOINTS, o).copy()
        target[0, 3] += 500.0
        o._verify_reached(target, "MoveJ")                    # must not raise

    def test_small_offset_within_tolerance_passes(self):
        o = _orch(JOINTS)
        target = _pose_of(JOINTS, o).copy()
        target[1, 3] += 3.0
        o._verify_reached(target, "MoveJ")

    def test_robot_without_joint_read_is_skipped(self):
        """A sim backend that cannot report joints must not break the cycle."""
        o = _orch(JOINTS)
        o.robot = type("NoRead", (), {"MoveJ": lambda *a: None})()
        o._verify_reached(np.eye(4), "MoveJ")                 # must not raise

    def test_frame_matches_the_model_used_by_client_ik(self):
        """FK here must use the same world-frame chain as the IK path."""
        o = _orch(JOINTS)
        model = o._dh_model()
        expect = gp7_urdf(base_xyz_mm=(0.0, 0.0, 630.0),
                          base_rpy_rad=(0.0, 0.0, 0.0), tool_offset_mm=0.0)
        a = forward_kinematics_urdf(model, [float(np.deg2rad(q)) for q in JOINTS])
        b = forward_kinematics_urdf(expect, [float(np.deg2rad(q)) for q in JOINTS])
        assert np.allclose(a, b, atol=1e-9)
