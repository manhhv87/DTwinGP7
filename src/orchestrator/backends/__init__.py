"""
backends — pluggable robot driver implementations.

Mỗi backend implement interface "RoboDK Item-like" (xem base.py) để Orchestrator
gọi đồng nhất bất kể đang nói chuyện với:
  - RoboDK API (sim / real qua RoboDK Educational+)
  - SimRobot (headless, không RoboDK)
  - MotomanHSEBackend (real, bypass RoboDK driver — chỉ cần RoboDK Free)
"""
from .base import RobotBackend

__all__ = ["RobotBackend"]
