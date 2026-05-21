"""
reach_envelope.py
─────────────────
Mô hình reachability đơn giản client-side cho Yaskawa GP7.

Không phải IK chính xác — chỉ check sphere envelope từ J1: pose nằm trong
vỏ cầu [reach_min, reach_max] thì coi như với tới.

Mục đích: cho HSE backend có reachability check khi không có RoboDK item làm
kinematic helper (vd setup standalone không cần RoboDK GUI).

Datasheet GP7:
  - Reach max  ~927 mm (J1 to wrist center)
  - Reach min  ~150 mm (dead zone quanh J1)
  - Joint limits: tùy axis, không model trong sphere (cần full DH)

Pattern này cùng nguồn với SimRobot._reachable() — extract thành module dùng
chung giữa các backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ReachEnvelope:
    """Sphere envelope cho 1 robot.

    Args:
        base_xyz_mm: Vị trí J1 trong world frame (mm).
        reach_max_mm: Bán kính tối đa với tới.
        reach_min_mm: Bán kính tối thiểu (vùng chết quanh base).
    """

    base_xyz_mm: tuple[float, float, float]
    reach_max_mm: float
    reach_min_mm: float

    @staticmethod
    def gp7_default(base_xyz_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> "ReachEnvelope":
        """Default cho Yaskawa GP7 — datasheet specs."""
        return ReachEnvelope(
            base_xyz_mm=base_xyz_mm,
            reach_max_mm=927.0,
            reach_min_mm=150.0,
        )

    def can_reach(self, target_xyz_mm: Any) -> bool:
        """True nếu target trong vỏ cầu envelope."""
        xyz = self._extract_xyz(target_xyz_mm)
        if xyz is None:
            return True                              # không xác định được → permissive
        dist = float(np.linalg.norm(xyz - np.asarray(self.base_xyz_mm, dtype=float)))
        return self.reach_min_mm <= dist <= self.reach_max_mm

    def distance_from_base(self, target_xyz_mm: Any) -> float | None:
        """Khoảng cách từ J1 tới target (mm). None nếu extract không được."""
        xyz = self._extract_xyz(target_xyz_mm)
        if xyz is None:
            return None
        return float(np.linalg.norm(xyz - np.asarray(self.base_xyz_mm, dtype=float)))

    @staticmethod
    def _extract_xyz(target: Any) -> np.ndarray | None:
        """Trích (x, y, z) từ nhiều dạng pose."""
        # numpy 4x4
        if isinstance(target, np.ndarray) and target.shape == (4, 4):
            return target[:3, 3].astype(float)
        # numpy 3-vector
        if isinstance(target, np.ndarray) and target.size == 3:
            return target.flatten().astype(float)
        # RoboDK Mat with .Pos()
        if hasattr(target, "Pos") and callable(target.Pos):
            try:
                return np.asarray(target.Pos()[:3], dtype=float)
            except Exception:                       # noqa: BLE001
                return None
        # Tuple/list (x, y, z)
        if isinstance(target, (list, tuple)) and len(target) >= 3:
            try:
                return np.asarray(target[:3], dtype=float)
            except (TypeError, ValueError):
                return None
        # Mat-like indexable
        try:
            return np.array([target[0, 3], target[1, 3], target[2, 3]], dtype=float)
        except Exception:                           # noqa: BLE001
            return None
