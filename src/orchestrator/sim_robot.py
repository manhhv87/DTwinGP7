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
        if isinstance(target, (list, tuple)) and len(target) >= 6:
            try:
                self._joints = [float(j) for j in target[:6]]
            except (TypeError, ValueError):
                pass
        logger.debug("%s #%d", kind, self._move_count)

    # ─── Gripper / speed ───
    def setDO(self, index: int, value: int) -> None:
        # value=1 → gripper đóng (giữ vật) → bật cờ để move kế (LIFT) check slip.
        self._grasp_pending_check = bool(value)
        logger.debug("setDO(%s, %s)", index, value)

    def setSpeed(self, linear: float, joint: float = -1) -> None:
        logger.debug("setSpeed(linear=%s, joint=%s)", linear, joint)
