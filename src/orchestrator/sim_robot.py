"""
sim_robot.py
────────────
SimRobot — pure-Python robot simulator, NO RoboDK required.

Purpose: validate the full orchestration logic (state machine, coord transform,
perception → pose → grasp planning, trial logging) + GENERATE QUANTITATIVE METRICS
without consuming RoboDK Free API quota.

Compared with the real RoboDK digital twin:
  - HAS: geometric reachability model (GP7 reach envelope) → rejects out-of-reach poses
  - HAS: failure-mode injection (grasp slip) by probability → CSV reflects failure modes
  - NO: real collision check, accurate kinematics, visualization

Use for:
  - Pipeline development (unlimited runs)
  - Generating success-rate / failure-mode statistics for thesis (software-in-the-loop)
  - CI/CD (no RoboDK GUI needed)

The real RoboDK digital twin is still needed for: proving C2 (reachability/collision via
twin) with an accurate robot model, and for motion visualization.
"""
from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SimRobot:
    """Robot simulator for headless mode (no RoboDK).

    Interface matches the RoboDK Item methods called by Orchestrator:
        Joints, JointsHome, MoveJ, MoveL, MoveJ_Test, SolveIK,
        setDO, setSpeed, Parent, Valid.

    Args:
        home_joints: Home joint angles (6 elements). Default [0,0,0,0,0,0].
        base_xyz: J1 position in world frame (mm) — origin for reachability.
            Default (0, 0, 630) matches cell_layout.yaml (robot on pedestal).
        reach_max_mm: Maximum reach radius of GP7 (mm). Datasheet ~927.
        reach_min_mm: Minimum reach radius (dead zone around base). ~150.
        grasp_fail_rate: Grasp slip probability (0.0–1.0) → raised on GRASP move.
        seed: RNG seed for reproducible results.
    """

    # Class-level: SimRobot CAN accept a 4x4 pose target — uses DLS internally
    # as "fake YRC IK" to simulate the YRC IK pipeline. Only verifies wiring +
    # routing, does NOT verify YRC's actual IK accuracy (real controller required).
    supports_cartesian_pose: bool = True

    def __init__(
        self,
        home_joints: list[float] | None = None,
        base_xyz: tuple[float, float, float] = (0.0, 0.0, 630.0),
        reach_max_mm: float = 927.0,
        reach_min_mm: float = 150.0,
        grasp_fail_rate: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self._home = list(home_joints) if home_joints else [0.0] * 6
        self._joints = list(self._home)
        self._base = np.asarray(base_xyz, dtype=float)
        self.reach_max_mm = reach_max_mm
        self.reach_min_mm = reach_min_mm
        self.grasp_fail_rate = grasp_fail_rate
        self._rng = random.Random(seed)
        self._move_count = 0
        self._grasp_pending_check = False   # True immediately after gripper closes

    # ─── Query ───
    def Joints(self) -> list[float]:
        return list(self._joints)

    def JointsHome(self) -> list[float]:
        return list(self._home)

    def Valid(self) -> bool:
        return True

    def Parent(self) -> None:
        # No parent frame → Orchestrator uses identity world→base,
        # so all poses passed to SimRobot are in WORLD frame.
        return None

    # ─── Reachability model ───
    @staticmethod
    def _extract_xyz(pose: Any) -> np.ndarray | None:
        """Extract translation (x, y, z) from a pose (RoboDK Mat or numpy 4x4)."""
        # RoboDK Mat: .Pos() → [x, y, z]
        if hasattr(pose, "Pos") and callable(pose.Pos):
            try:
                return np.asarray(pose.Pos()[:3], dtype=float)
            except Exception:  # noqa: BLE001
                pass
        # numpy 4x4
        arr = np.asarray(pose, dtype=float) if not hasattr(pose, "Pos") else None
        if arr is not None and arr.shape == (4, 4):
            return arr[:3, 3].astype(float)
        # Mat indexing fallback
        try:
            return np.array([pose[0, 3], pose[1, 3], pose[2, 3]], dtype=float)
        except Exception:  # noqa: BLE001
            return None

    def _reachable(self, pose: Any) -> bool:
        xyz = self._extract_xyz(pose)
        if xyz is None:
            return True  # cannot determine → treat as reachable (safe for tests)
        dist = float(np.linalg.norm(xyz - self._base))
        return self.reach_min_mm <= dist <= self.reach_max_mm

    def MoveJ_Test(self, j1: Any, target: Any, *args: Any) -> int:
        """RoboDK convention: 0 = OK, < 0 = out of reach. Simulates reach envelope."""
        return 0 if self._reachable(target) else -1

    def SolveIK(self, pose: Any, joints_approx: Any = None) -> list[float] | None:
        """Returns placeholder joints if reachable, None if out of reach."""
        if not self._reachable(pose):
            return None
        return list(self._joints)

    # ─── Motion ───
    def MoveJ(self, target: Any) -> None:
        self._apply_move(target, "MoveJ")

    def MoveL(self, target: Any) -> None:
        self._apply_move(target, "MoveL")
        # Grasp slip injection: check ONCE on the move immediately after gripper closes
        # (i.e. LIFT — lifting the object). Slip → raise → orchestrator logs fail.
        if self._grasp_pending_check:
            self._grasp_pending_check = False
            if self._rng.random() < self.grasp_fail_rate:
                raise RuntimeError("grasp_slip")

    def _apply_move(self, target: Any, kind: str) -> None:
        self._move_count += 1
        # Joint list (6 floats)
        if isinstance(target, (list, tuple)) and len(target) >= 6:
            try:
                self._joints = [float(j) for j in target[:6]]
                logger.debug("%s #%d joints", kind, self._move_count)
                return
            except (TypeError, ValueError):
                pass
        # 4x4 pose — simulate YRC IK path via DLS internal
        if self._is_pose_4x4(target):
            self._joints = self._fake_yrc_ik(target)
            logger.debug("%s #%d Cartesian (fake YRC IK)", kind, self._move_count)
            return
        logger.debug("%s #%d (unknown target type)", kind, self._move_count)

    @staticmethod
    def _is_pose_4x4(target: Any) -> bool:
        """Detect 4x4 numpy array or list-of-lists."""
        if hasattr(target, "shape") and getattr(target, "shape", None) == (4, 4):
            return True
        if (isinstance(target, (list, tuple)) and len(target) == 4
                and all(hasattr(r, "__len__") and len(r) == 4 for r in target)):
            return True
        return False

    def _fake_yrc_ik(self, target_pose_base: Any) -> list[float]:
        """Stand-in for YRC's internal IK — uses DLS pure Python.

        Warning: For simulation wiring/pipeline tests only. Does NOT reflect YRC's actual IK accuracy.
        DH params in the kinematics module may not match the real robot.
        """
        try:
            from .kinematics import inverse_kinematics
            from .kinematics.urdf_chain import gp7_urdf
        except ImportError:
            return list(self._joints)            # fallback: keep current

        if not hasattr(self, "_sim_dh_model"):
            # URDF chain — match RoboDK SolveFK exactly (verified 0.00mm)
            self._sim_dh_model = gp7_urdf(
                base_xyz_mm=tuple(self._base.tolist()),
            )
        q_init_rad = [np.deg2rad(q) for q in self._joints]
        sol_rad = inverse_kinematics(self._sim_dh_model, target_pose_base, q_init_rad)
        if sol_rad is None:
            # IK failed in sim → keep current joints (orchestrator will log unreachable)
            return list(self._joints)
        return [float(np.rad2deg(q)) for q in sol_rad]

    # ─── Gripper / speed ───
    def setDO(self, index: int, value: int) -> None:
        # value=1 → gripper closes (holds object) → set flag so next move (LIFT) checks slip.
        self._grasp_pending_check = bool(value)
        logger.debug("setDO(%s, %s)", index, value)

    def setSpeed(self, linear: float, joint: float = -1) -> None:
        logger.debug("setSpeed(linear=%s, joint=%s)", linear, joint)

    # ─── PLC/CC-Link mock for sim mode ───
    def set_io(self, bit_addr: int, value: int) -> None:
        """Sim mock — no-op. Tracks gripper close command via bit pattern."""
        # Clamp ON bit → set grasp pending check (sim grasp slip injection)
        self._grasp_pending_check = bool(value)
        logger.debug("SimRobot.set_io(bit=%d) = %d", bit_addr, value)

    def read_io(self, bit_addr: int) -> int:
        """Sim mock — always returns "sensor confirmed" to let pipeline pass through.

        Real mode uses MotomanHSEBackend.read_io to read actual CC-Link bits.
        Sim: returns 1 immediately (gripper treated as at position + object present).
        """
        return 1

    # ─── DigitalTwinMirror compatibility ───
    def Stop(self) -> None:
        """No-op for sim (no hardware to stop)."""
        logger.debug("SimRobot.Stop() no-op")

    def disconnect(self) -> None:
        """No-op for sim — allows DigitalTwinMirror to clean up uniformly."""
        logger.debug("SimRobot.disconnect() no-op")
