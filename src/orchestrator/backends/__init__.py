"""
backends — pluggable robot driver implementations.

Each backend implements a "RoboDK Item-like" interface (see base.py) so the
Orchestrator can call them uniformly regardless of the underlying target:
  - RoboDK API (sim / real via RoboDK Educational+)
  - SimRobot (headless, no RoboDK)
  - MotomanHSEBackend (real, bypass RoboDK driver — requires only RoboDK Free)
"""
from .base import RobotBackend

__all__ = ["RobotBackend"]
