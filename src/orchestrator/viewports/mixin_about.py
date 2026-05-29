"""
mixin_about.py
──────────────
AboutMixin: cell info + about dialog (info-only message boxes).

Host class (GP7AppQt) phải cung cấp:
  attributes: _cell_config, _base_xyz, _home_joints, _ref_frames
  methods:    _set_status
"""
from __future__ import annotations

from PyQt6.QtWidgets import QMessageBox

from .control_panel import _matrix_to_xyz_rpy_deg


class AboutMixin:
    """Cell info + about dialog."""

    def _show_cell_info(self) -> None:
        # Chưa load robot/cell → _cell_config còn None, không có gì để show.
        if self._cell_config is None:
            self._set_status(
                "Chưa load cell — File → Load Robot GP7 / Load Cell from YAML",
                level="warn")
            return
        lines = [
            f"<b>Robot</b><br>",
            f"&nbsp;&nbsp;Name: {self._cell_config.robot.name}<br>",
            f"&nbsp;&nbsp;Base xyz: {list(self._base_xyz)} mm<br>",
            f"&nbsp;&nbsp;Home joints: "
            + ", ".join(f"{q:+.2f}" for q in self._home_joints) + " deg<br>",
            f"<br><b>Reference frames ({len(self._ref_frames)})</b><br>",
        ]
        for name, T in self._ref_frames:
            x, y, z, rx, ry, rz = _matrix_to_xyz_rpy_deg(T)
            lines.append(
                f"&nbsp;&nbsp;{name}: xyz=({x:.0f},{y:.0f},{z:.0f}) "
                f"rpy=({rx:.0f},{ry:.0f},{rz:.0f})<br>")
        objs = list(getattr(self._cell_config, "objects", []) or [])
        lines.append(f"<br><b>Objects ({len(objs)})</b><br>")
        for o in objs:
            xyz = list(o.pose.xyz_mm) if getattr(o, "pose", None) else [0, 0, 0]
            lines.append(
                f"&nbsp;&nbsp;{o.name} ({getattr(o, 'parent_frame', '-')}) "
                f"xyz={xyz}<br>")
        QMessageBox.information(self, "Cell info", "".join(lines))

    def _show_about(self) -> None:
        QMessageBox.about(
            self, "About",
            "<b>GP7 Digital Twin</b><br><br>"
            "Yaskawa GP7 + PyQt6 + pyvistaqt (VTK).<br>"
            "Industrial-standard 3D stack.<br><br>"
            "Kinematics (FK/IK) verified against RoboDK to 0.00 mm.<br>"
            "INFORM .JBI export ready for YRC1000 controller.")
