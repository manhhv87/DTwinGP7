"""
viewports — Pure-Python 3D rendering engine (Open3D + Filament GUI).

The motion + viewport stack no longer uses RoboDK. Non-headless sim uses
O3DGuiSimRobot as the motion backend + viewport mirror. Real mode uses
O3DGuiSimRobot as a viewport-only mirror (joints fed from the HSE backend).

Modules:
  - urdf_gen.py          : generates the GP7 URDF from kinematics.urdf_chain
                           (verified to match RoboDK SolveFK at 0.00 mm).
                           For pybullet / general use.
  - open3d_gui_sim_robot : SimRobot + Open3D O3DVisualizer (Filament, smooth
                           navigation; GUI runs on the main thread + worker
                           experiments).

open3d is imported LAZILY inside each class __init__ — importing this package
does NOT pull in open3d, so the test suite (no open3d installed) still runs
normally.
"""
from .urdf_gen import build_gp7_urdf_xml, ensure_gp7_urdf

__all__ = ["build_gp7_urdf_xml", "ensure_gp7_urdf"]
