"""
sim_robot.py
────────────
SimRobot — robot giả lập thuần Python, KHÔNG cần RoboDK.

Mục đích: validate toàn bộ logic orchestration (state machine, coord transform,
perception → pose → grasp planning, trial logging) + SINH SỐ LIỆU ĐỊNH LƯỢNG
mà KHÔNG tốn RoboDK Free API quota.

So với RoboDK digital twin thật:
  - CÓ: mô hình reachability hình học (GP7 reach envelope) → reject pose ngoài tầm
  - CÓ: inject failure modes (grasp slip) theo xác suất → CSV phản ánh failure modes
  - KHÔNG: collision check thật, kinematics chính xác, visualize

Dùng cho:
  - Phát triển pipeline (chạy không giới hạn)
  - Sinh thống kê success-rate / failure-mode cho luận văn (software-in-the-loop)
  - CI/CD (không cần RoboDK GUI)

RoboDK digital twin thật vẫn cần cho: chứng minh C2 (reachability/collision qua
twin) với robot model chính xác, và visualize chuyển động.
"""
from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SimRobot:
    """Robot giả lập cho headless mode (không RoboDK).

    Interface khớp các method RoboDK Item mà Orchestrator gọi:
        Joints, JointsHome, MoveJ, MoveL, MoveJ_Test, SolveIK,
        setDO, setSpeed, Parent, Valid.

    Args:
        home_joints: Joints home (6 phần tử). Mặc định [0,0,0,0,0,0].
        base_xyz: Vị trí J1 trong world (mm) — gốc tính reachability.
            Mặc định (0, 0, 630) khớp cell_layout.yaml (robot trên pedestal).
        reach_max_mm: Bán kính with-tới tối đa của GP7 (mm). Datasheet ~927.
        reach_min_mm: Bán kính tối thiểu (vùng chết quanh base). ~150.
        grasp_fail_rate: Xác suất grasp slip (0.0–1.0) → raise lúc GRASP move.
        seed: Seed RNG để tái lập kết quả.
    """

    # Class-level: SimRobot CÓ thể nhận 4x4 pose target — dùng DLS internal
    # làm "fake YRC IK" để sim được YRC IK pipeline. Chỉ verify wiring +
    # routing, KHÔNG verify được YRC's actual IK accuracy (cần controller thật).
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
        self._grasp_pending_check = False   # True ngay sau khi đóng gripper

    # ─── Query ───
    def Joints(self) -> list[float]:
        return list(self._joints)

    def JointsHome(self) -> list[float]:
        return list(self._home)

    def Valid(self) -> bool:
        return True

    def Parent(self) -> None:
        # Không có parent frame → Orchestrator dùng identity world→base,
        # nên mọi pose truyền vào SimRobot ở WORLD frame.
        return None

    # ─── Reachability model ───
    @staticmethod
    def _extract_xyz(pose: Any) -> np.ndarray | None:
        """Lấy translation (x, y, z) từ pose (RoboDK Mat hoặc numpy 4x4)."""
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
            return True  # không xác định được → coi như với tới (an toàn cho test)
        dist = float(np.linalg.norm(xyz - self._base))
        return self.reach_min_mm <= dist <= self.reach_max_mm

    def MoveJ_Test(self, j1: Any, target: Any, *args: Any) -> int:
        """RoboDK convention: 0 = OK, < 0 = ngoài tầm. Mô phỏng reach envelope."""
        return 0 if self._reachable(target) else -1

    def SolveIK(self, pose: Any, joints_approx: Any = None) -> list[float] | None:
        """Trả joints placeholder nếu với tới, None nếu ngoài tầm."""
        if not self._reachable(pose):
            return None
        return list(self._joints)

    # ─── Motion ───
    def MoveJ(self, target: Any) -> None:
        self._apply_move(target, "MoveJ")

    def MoveL(self, target: Any) -> None:
        self._apply_move(target, "MoveL")
        # Grasp slip injection: kiểm tra MỘT lần ở move ngay sau khi đóng gripper
        # (chính là LIFT — nhấc vật lên). Vật slip → raise → orchestrator log fail.
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
        # 4x4 pose — sim YRC IK path qua DLS internal
        if self._is_pose_4x4(target):
            self._joints = self._fake_yrc_ik(target)
            logger.debug("%s #%d Cartesian (fake YRC IK)", kind, self._move_count)
            return
        logger.debug("%s #%d (unknown target type)", kind, self._move_count)

    @staticmethod
    def _is_pose_4x4(target: Any) -> bool:
        """Detect 4x4 numpy hoặc list-of-lists."""
        if hasattr(target, "shape") and getattr(target, "shape", None) == (4, 4):
            return True
        if (isinstance(target, (list, tuple)) and len(target) == 4
                and all(hasattr(r, "__len__") and len(r) == 4 for r in target)):
            return True
        return False

    def _fake_yrc_ik(self, target_pose_base: Any) -> list[float]:
        """Stand-in cho YRC's internal IK — dùng DLS pure Python.

        ⚠ Chỉ để sim test wiring/pipeline. KHÔNG reflect YRC's actual IK accuracy.
        DH params trong kinematics module có thể chưa khớp robot thật.
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
            # IK fail trong sim → keep current joints (orchestrator sẽ log unreachable)
            return list(self._joints)
        return [float(np.rad2deg(q)) for q in sol_rad]

    # ─── Gripper / speed ───
    def setDO(self, index: int, value: int) -> None:
        # value=1 → gripper đóng (giữ vật) → bật cờ để move kế (LIFT) check slip.
        self._grasp_pending_check = bool(value)
        logger.debug("setDO(%s, %s)", index, value)

    def setSpeed(self, linear: float, joint: float = -1) -> None:
        logger.debug("setSpeed(linear=%s, joint=%s)", linear, joint)

    # ─── PLC/CC-Link mock cho sim mode ───
    def set_io(self, bit_addr: int, value: int) -> None:
        """Sim mock — no-op. Track gripper close command qua bit pattern."""
        # Bít clamp ON → set grasp pending check (sim grasp slip injection)
        self._grasp_pending_check = bool(value)
        logger.debug("SimRobot.set_io(bit=%d) = %d", bit_addr, value)

    def read_io(self, bit_addr: int) -> int:
        """Sim mock — luôn trả "sensor confirmed" để pipeline pass-through.

        Real mode dùng MotomanHSEBackend.read_io đọc CC-Link bit thật.
        Sim: trả 1 ngay (gripper coi như đã đến position + có vật trong gripper).
        """
        return 1

    # ─── DigitalTwinMirror compatibility ───
    def Stop(self) -> None:
        """No-op cho sim (không có hardware để stop)."""
        logger.debug("SimRobot.Stop() no-op")

    def disconnect(self) -> None:
        """No-op cho sim — cho phép DigitalTwinMirror cleanup thống nhất."""
        logger.debug("SimRobot.disconnect() no-op")
