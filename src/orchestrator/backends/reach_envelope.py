"""
reach_envelope.py
─────────────────
Simple client-side reachability model for the Yaskawa GP7.

Not a precise IK — only checks the sphere envelope from J1: if a pose lies
within [reach_min, reach_max] it is considered reachable.

Purpose: give the HSE backend a reachability check when no RoboDK item is
available as a kinematic helper (e.g. standalone setup without RoboDK GUI).

Datasheet GP7:
  - Reach max  ~927 mm (J1 to wrist center)
  - Reach min  ~150 mm (dead zone around J1)
  - Joint limits: per axis, not modelled in sphere (full DH required)

This pattern shares its origin with SimRobot._reachable() — extracted as a
module shared across backends.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ReachEnvelope:
    """Sphere envelope for one robot.

    Args:
        base_xyz_mm: J1 position in world frame (mm).
        reach_max_mm: Maximum reachable radius.
        reach_min_mm: Minimum reachable radius (dead zone around base).
    """

    base_xyz_mm: tuple[float, float, float]
    reach_max_mm: float
    reach_min_mm: float

    @staticmethod
    def gp7_default(base_xyz_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> "ReachEnvelope":
        """Default envelope for the Yaskawa GP7 — datasheet specs."""
        return ReachEnvelope(
            base_xyz_mm=base_xyz_mm,
            reach_max_mm=927.0,
            reach_min_mm=150.0,
        )

    def can_reach(self, target_xyz_mm: Any) -> bool:
        """True if target lies within the sphere envelope."""
        xyz = self._extract_xyz(target_xyz_mm)
        if xyz is None:
            return True                              # cannot determine → permissive
        dist = float(np.linalg.norm(xyz - np.asarray(self.base_xyz_mm, dtype=float)))
        return self.reach_min_mm <= dist <= self.reach_max_mm

    def distance_from_base(self, target_xyz_mm: Any) -> float | None:
        """Distance from J1 to target (mm). None if extraction fails."""
        xyz = self._extract_xyz(target_xyz_mm)
        if xyz is None:
            return None
        return float(np.linalg.norm(xyz - np.asarray(self.base_xyz_mm, dtype=float)))

    @staticmethod
    def _extract_xyz(target: Any) -> np.ndarray | None:
        """Extract (x, y, z) from various pose representations."""
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
