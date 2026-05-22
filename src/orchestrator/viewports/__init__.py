"""
viewports — Engine hiển thị 3D pure-Python (Open3D + Filament GUI).

Stack motion + viewport KHÔNG còn dùng RoboDK. Sim non-headless dùng
O3DGuiSimRobot làm motion backend + viewport mirror. Real mode dùng
O3DGuiSimRobot làm viewport-only mirror (joints từ HSE backend).

Modules:
  - urdf_gen.py          : sinh URDF GP7 từ kinematics.urdf_chain (verified
                           match RoboDK SolveFK 0.00mm). Cho pybullet/general use.
  - open3d_gui_sim_robot : SimRobot + Open3D O3DVisualizer (Filament, navigate
                           mượt; GUI chạy main thread + thí nghiệm worker).

open3d import LAZY trong __init__ của class — import package này KHÔNG kéo
theo open3d, nên test suite (không cài open3d) vẫn chạy bình thường.
"""
from .urdf_gen import build_gp7_urdf_xml, ensure_gp7_urdf

__all__ = ["build_gp7_urdf_xml", "ensure_gp7_urdf"]
