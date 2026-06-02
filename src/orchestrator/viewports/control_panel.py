"""
control_panel.py
────────────────
GUI Control Panel for GP7 — RoboDK style, runs IN PARALLEL with O3DGuiSimRobot.

Opens a secondary window on the same `gui.Application` as the viewport — does not
touch the existing O3DVisualizer. Includes:

  * Joint axis jog        — 6 sliders θ1..θ6 ± min/max of model + Home
  * TCP pose readout       — Tool/Reference/Tool-w.r.t.-Ref (X,Y,Z mm + RPY°)
  * Tool / Reference combo — loaded from cell_config (gripper, frames)
  * Cartesian jog          — ± X/Y/Z translate / rotate buttons in selected frame
  * Step size adjustable   — mm and deg
  * Other configurations   — find alternate IK branches for the current TCP
  * Gripper Open / Close   — calls setDO(1, ·) (triggers grasp/release in sim)

**Sim-only**: callbacks only write `robot._joints` + `robot._apply_frames(...)`.
No MoveJ sent to the real robot.

Kinematics use URDFRobot (FK matches RoboDK 0.00mm) for both FK and IK — ensures
TCP shown in panel matches the viewport.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from ..kinematics.inverse_kinematics import inverse_kinematics_seeded
from ..kinematics.urdf_chain import URDFRobot, forward_kinematics_urdf

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Math helpers — pose ↔ XYZ+RPY (ZYX intrinsic = Rz·Ry·Rx, Fanuc/Motoman)
# ──────────────────────────────────────────────────────────────────────────

def _matrix_to_xyz_rpy_deg(
    T: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """4x4 → (X, Y, Z mm, Rx, Ry, Rz deg) — ZYX intrinsic (Fanuc/Motoman)."""
    R = T[:3, :3]
    sp = max(-1.0, min(1.0, -float(R[2, 0])))
    pitch = math.asin(sp)
    if abs(math.cos(pitch)) > 1e-6:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:                                              # gimbal lock
        roll = 0.0
        yaw = math.atan2(-float(R[0, 1]), float(R[1, 1]))
    return (
        float(T[0, 3]), float(T[1, 3]), float(T[2, 3]),
        math.degrees(roll), math.degrees(pitch), math.degrees(yaw),
    )


def _xyz_rpy_to_matrix(
    x: float, y: float, z: float,
    rx_deg: float, ry_deg: float, rz_deg: float,
) -> np.ndarray:
    r, p, yaw = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(yaw), math.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])
    T[:3, 3] = [x, y, z]
    return T


def _rotation_about_axis_3x3(axis_world: np.ndarray, theta_rad: float) -> np.ndarray:
    """Rodrigues 3x3 rotation about axis (world vector, auto-normalized)."""
    n = float(np.linalg.norm(axis_world))
    if n < 1e-12:
        return np.eye(3)
    x, y, z = axis_world / n
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    C = 1.0 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


# ──────────────────────────────────────────────────────────────────────────
# Frame loaders from cell_config
# ──────────────────────────────────────────────────────────────────────────

def _build_tool_frames(cell_config) -> list[tuple[str, np.ndarray]]:
    """[(name, T_flange_tool)] — list tool frame relative to flange."""
    out: list[tuple[str, np.ndarray]] = [("Flange (no tool)", np.eye(4))]
    g = getattr(cell_config, "gripper", None) if cell_config is not None else None
    if g is not None and getattr(g, "tcp_offset_xyz_mm", None) is not None:
        xyz = g.tcp_offset_xyz_mm
        rpy = getattr(g, "tcp_offset_rpy_deg", (0.0, 0.0, 0.0))
        T = _xyz_rpy_to_matrix(xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2])
        out.append((getattr(g, "name", "Gripper"), T))
    return out


def _build_ref_frames(cell_config) -> list[tuple[str, np.ndarray]]:
    """[(name, T_base_ref)] — reference frame relative to robot base."""
    out: list[tuple[str, np.ndarray]] = [("Base (0)", np.eye(4))]
    if cell_config is None:
        return out
    for f in getattr(cell_config, "frames", None) or []:
        pose = getattr(f, "pose", None)
        if pose is None:
            continue
        xyz = pose.xyz_mm
        rpy = pose.rpy_deg
        T = _xyz_rpy_to_matrix(xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2])
        out.append((f.name, T))
    return out


# ──────────────────────────────────────────────────────────────────────────
# Control Panel
# ──────────────────────────────────────────────────────────────────────────

class ControlPanel:
    """GP7 control window in RoboDK panel style — sim only.

    Instantiate AFTER `O3DGuiSimRobot(...)` and BEFORE `robot.run_gui()` so
    both windows are managed by the same Application:

        robot = O3DGuiSimRobot(base_xyz=..., cell_config=..., project_root=...)
        panel = ControlPanel(robot, robot._model, cell_config=cell_config)
        robot.run_gui()          # blocks until both windows closed
    """

    JOG_FRAMES = ("Tool Frame", "Reference Frame", "Base")
    AXIS_NAMES = ("X", "Y", "Z")
    _AXIS_VEC = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )

    def __init__(
        self,
        robot,                                # O3DGuiSimRobot (duck-typed)
        model: URDFRobot,
        cell_config: Any = None,
        home_joints_deg: list[float] | None = None,
        window_size: tuple[int, int] = (640, 820),
        window_pos: tuple[int, int] = (40, 60),
    ) -> None:
        import open3d.visualization.gui as gui              # lazy import
        self._gui = gui
        self._robot = robot
        self._model = model
        self._home_joints = list(
            home_joints_deg
            if home_joints_deg is not None else robot._joints
        )

        self._tool_frames = _build_tool_frames(cell_config)
        self._ref_frames = _build_ref_frames(cell_config)
        # Default selections: prefer gripper if present; ref = base.
        self._tool_idx = len(self._tool_frames) - 1 if len(self._tool_frames) > 1 else 0
        self._ref_idx = 0
        self._jog_frame_idx = 0                            # Tool Frame default
        self._jog_step_mm = 10.0
        self._jog_step_deg = 5.0
        self._alternates: list[list[float]] = []

        # Set explicit (x, y) position so the panel is NOT hidden behind the
        # viewport in the middle of the screen. Open3D create_window defaults
        # to (-1,-1) auto-position; on Windows multi-window tends to stack —
        # the panel may end up beneath the viewport.
        self._window = gui.Application.instance.create_window(
            "GP7 Control Panel — RoboDK style",
            window_size[0], window_size[1],
            window_pos[0], window_pos[1],
        )
        logger.info("Control Panel window opened at (%d, %d) size=%dx%d",
                    window_pos[0], window_pos[1], window_size[0], window_size[1])

        em = self._window.theme.font_size
        m = 0.5 * em
        root = gui.Vert(0.3 * em, gui.Margins(m, m, m, m))

        title = gui.Label("Yaskawa GP7  —  Sim Jog Panel")
        root.add_child(title)
        root.add_fixed(0.4 * em)

        root.add_child(self._build_cartesian_section(em))
        root.add_fixed(0.5 * em)
        root.add_child(self._build_joint_section(em))
        root.add_fixed(0.5 * em)
        root.add_child(self._build_alternates_section(em))
        root.add_fixed(0.5 * em)
        root.add_child(self._build_gripper_section(em))
        root.add_fixed(0.3 * em)

        self._status = gui.Label("Ready")
        root.add_child(self._status)

        self._window.add_child(root)
        try:
            self._window.show(True)                    # belt-and-suspenders
        except Exception as e:                         # noqa: BLE001
            logger.debug("window.show(True) error: %s", e)
        self._refresh_pose_readout()

    # ──────────────────────────────────────────────────────────────────
    # Section builders
    # ──────────────────────────────────────────────────────────────────

    def _build_cartesian_section(self, em):
        gui = self._gui
        box = gui.CollapsableVert(
            "Cartesian Jog", 0.25 * em, gui.Margins(0.3 * em, 0.2 * em, 0, 0))

        # Tool Frame combo
        row = gui.Horiz(0.3 * em)
        row.add_child(gui.Label("Tool Frame:   "))
        self._tool_combo = gui.Combobox()
        for name, _T in self._tool_frames:
            self._tool_combo.add_item(name)
        self._tool_combo.selected_index = self._tool_idx
        self._tool_combo.set_on_selection_changed(self._on_tool_changed)
        row.add_child(self._tool_combo)
        box.add_child(row)
        self._tool_pose_labels = self._add_pose_row(box, "  w.r.t. Flange:")

        # Reference Frame combo
        row = gui.Horiz(0.3 * em)
        row.add_child(gui.Label("Reference:    "))
        self._ref_combo = gui.Combobox()
        for name, _T in self._ref_frames:
            self._ref_combo.add_item(name)
        self._ref_combo.selected_index = self._ref_idx
        self._ref_combo.set_on_selection_changed(self._on_ref_changed)
        row.add_child(self._ref_combo)
        box.add_child(row)
        self._ref_pose_labels = self._add_pose_row(box, "  w.r.t. Base:")
        self._tcp_pose_labels = self._add_pose_row(box, "Tool w.r.t. Reference:")

        # Jog frame + step size
        row = gui.Horiz(0.3 * em)
        row.add_child(gui.Label("Jog in:"))
        self._jog_frame_combo = gui.Combobox()
        for n in self.JOG_FRAMES:
            self._jog_frame_combo.add_item(n)
        self._jog_frame_combo.selected_index = self._jog_frame_idx
        self._jog_frame_combo.set_on_selection_changed(self._on_jog_frame_changed)
        row.add_child(self._jog_frame_combo)
        row.add_fixed(em)
        row.add_child(gui.Label("Step:"))
        self._step_mm_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._step_mm_edit.set_limits(0.1, 500.0)
        self._step_mm_edit.double_value = self._jog_step_mm
        self._step_mm_edit.set_on_value_changed(
            lambda v: setattr(self, "_jog_step_mm", float(v)))
        row.add_child(self._step_mm_edit)
        row.add_child(gui.Label("mm /"))
        self._step_deg_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._step_deg_edit.set_limits(0.1, 90.0)
        self._step_deg_edit.double_value = self._jog_step_deg
        self._step_deg_edit.set_on_value_changed(
            lambda v: setattr(self, "_jog_step_deg", float(v)))
        row.add_child(self._step_deg_edit)
        row.add_child(gui.Label("deg"))
        box.add_child(row)

        box.add_child(self._build_jog_buttons(em, "Translation", self._on_translate))
        box.add_child(self._build_jog_buttons(em, "Rotation   ", self._on_rotate))
        return box

    def _build_jog_buttons(self, em, label, callback):
        gui = self._gui
        row = gui.Horiz(0.2 * em)
        row.add_child(gui.Label(label))
        for axis_idx, name in enumerate(self.AXIS_NAMES):
            for sign, sym in ((-1, "-"), (+1, "+")):
                btn = gui.Button(f"{sym}{name}")
                btn.set_on_clicked(
                    (lambda a=axis_idx, s=sign: callback(a, s))
                )
                row.add_child(btn)
            row.add_fixed(0.2 * em)
        return row

    def _add_pose_row(self, parent, label_text):
        gui = self._gui
        parent.add_child(gui.Label(label_text))
        row = gui.Horiz(0.15 * self._window.theme.font_size)
        labels = []
        for axis in ("X", "Y", "Z", "Rx", "Ry", "Rz"):
            row.add_child(gui.Label(f"{axis}"))
            v = gui.Label("       0.000")
            row.add_child(v)
            labels.append(v)
        parent.add_child(row)
        return labels                                       # [X, Y, Z, Rx, Ry, Rz]

    def _build_joint_section(self, em):
        gui = self._gui
        box = gui.CollapsableVert(
            "Joint axis jog", 0.25 * em, gui.Margins(0.3 * em, 0.2 * em, 0, 0))

        hdr = gui.Horiz(0.3 * em)
        hdr.add_stretch()
        home_btn = gui.Button("Home")
        home_btn.set_on_clicked(self._on_home)
        zero_btn = gui.Button("Zero")
        zero_btn.set_on_clicked(self._on_zero)
        hdr.add_child(zero_btn)
        hdr.add_child(home_btn)
        box.add_child(hdr)

        self._joint_sliders = []
        self._joint_value_labels = []
        for i, joint in enumerate(self._model.joints):
            row = gui.Horiz(0.25 * em)
            row.add_child(gui.Label(f"θ{i+1}"))
            value_lbl = gui.Label(f"{self._robot._joints[i]:8.2f}°")
            row.add_child(value_lbl)
            self._joint_value_labels.append(value_lbl)

            jmin_deg = math.degrees(joint.joint_min)
            jmax_deg = math.degrees(joint.joint_max)
            row.add_child(gui.Label(f"{jmin_deg:7.1f}"))
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(jmin_deg, jmax_deg)
            slider.double_value = self._robot._joints[i]
            slider.set_on_value_changed(
                (lambda v, idx=i: self._on_joint_slider(idx, float(v)))
            )
            row.add_child(slider)
            row.add_child(gui.Label(f"{jmax_deg:7.1f}"))
            self._joint_sliders.append(slider)
            box.add_child(row)
        return box

    def _build_alternates_section(self, em):
        gui = self._gui
        box = gui.CollapsableVert(
            "Other configurations", 0.25 * em,
            gui.Margins(0.3 * em, 0.2 * em, 0, 0))
        hdr = gui.Horiz(0.3 * em)
        hdr.add_child(gui.Label("Find alternate IK branches for current TCP:"))
        hdr.add_stretch()
        find_btn = gui.Button("Find")
        find_btn.set_on_clicked(self._on_find_alternates)
        hdr.add_child(find_btn)
        box.add_child(hdr)
        self._alt_combo = gui.Combobox()
        self._alt_combo.add_item("(not searched yet)")
        self._alt_combo.set_on_selection_changed(self._on_alternate_selected)
        box.add_child(self._alt_combo)
        return box

    def _build_gripper_section(self, em):
        gui = self._gui
        box = gui.Horiz(0.3 * em, self._gui.Margins(0.3 * em, 0.2 * em, 0, 0))
        box.add_child(gui.Label("Gripper:"))
        open_btn = gui.Button("Open")
        close_btn = gui.Button("Close")
        open_btn.set_on_clicked(lambda: self._toggle_gripper(False))
        close_btn.set_on_clicked(lambda: self._toggle_gripper(True))
        box.add_child(open_btn)
        box.add_child(close_btn)
        box.add_stretch()
        return box

    # ──────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────

    def _on_joint_slider(self, idx: int, value_deg: float) -> None:
        joints = list(self._robot._joints)
        joints[idx] = value_deg
        self._set_joints(joints, source=f"θ{idx+1}={value_deg:+.1f}°")

    def _on_home(self) -> None:
        self._set_joints(self._home_joints, source="Home")

    def _on_zero(self) -> None:
        self._set_joints([0.0] * len(self._model.joints), source="Zero")

    def _on_tool_changed(self, _text: str, idx: int) -> None:
        self._tool_idx = int(idx)
        self._refresh_pose_readout()

    def _on_ref_changed(self, _text: str, idx: int) -> None:
        self._ref_idx = int(idx)
        self._refresh_pose_readout()

    def _on_jog_frame_changed(self, _text: str, idx: int) -> None:
        self._jog_frame_idx = int(idx)
        self._set_status(f"Jog frame = {self.JOG_FRAMES[self._jog_frame_idx]}")

    def _on_translate(self, axis_idx: int, sign: int) -> None:
        T_world_tool = self._current_tool_world()
        if T_world_tool is None:
            return
        axis_world = self._jog_axis_world(axis_idx, T_world_tool)
        step = sign * self._jog_step_mm
        T_target = T_world_tool.copy()
        T_target[:3, 3] = T_world_tool[:3, 3] + axis_world * step
        self._apply_cartesian_target(
            T_target, f"T{'+' if sign > 0 else '-'}{self.AXIS_NAMES[axis_idx]}"
            f" {abs(step):.1f}mm"
        )

    def _on_rotate(self, axis_idx: int, sign: int) -> None:
        T_world_tool = self._current_tool_world()
        if T_world_tool is None:
            return
        axis_world = self._jog_axis_world(axis_idx, T_world_tool)
        step_rad = sign * math.radians(self._jog_step_deg)
        R_step = _rotation_about_axis_3x3(axis_world, step_rad)
        T_target = T_world_tool.copy()
        T_target[:3, :3] = R_step @ T_world_tool[:3, :3]
        self._apply_cartesian_target(
            T_target, f"R{'+' if sign > 0 else '-'}{self.AXIS_NAMES[axis_idx]}"
            f" {abs(math.degrees(step_rad)):.1f}°"
        )

    def _on_find_alternates(self) -> None:
        T_target = self._current_tool_world()
        if T_target is None:
            return
        self._alternates = self._find_alternate_solutions(T_target)
        try:
            self._alt_combo.clear_items()
        except AttributeError:                              # Open3D <0.17 fallback
            pass
        if not self._alternates:
            self._alt_combo.add_item("(no alternate IK branches)")
            self._set_status("No alternate branches found")
            return
        for sol in self._alternates:
            label = "[" + ", ".join(f"{q:+7.1f}°" for q in sol) + "]"
            self._alt_combo.add_item(label)
        self._set_status(f"Found {len(self._alternates)} alternate branch(es)")

    def _on_alternate_selected(self, _text: str, idx: int) -> None:
        i = int(idx)
        if 0 <= i < len(self._alternates):
            self._set_joints(self._alternates[i], source=f"alt #{i+1}")

    def _toggle_gripper(self, close: bool) -> None:
        try:
            self._robot.setDO(1, 1 if close else 0)
            self._set_status(f"Gripper {'CLOSE' if close else 'OPEN'}")
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"Gripper error: {e}")

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _set_joints(self, joints_deg, source: str = "") -> None:
        """Update robot state + viewport (main thread)."""
        joints_deg = [float(v) for v in joints_deg]
        # Clamp to model joint limits (safe guard if IK returns out-of-range values).
        for i, joint in enumerate(self._model.joints):
            lo = math.degrees(joint.joint_min)
            hi = math.degrees(joint.joint_max)
            joints_deg[i] = min(hi, max(lo, joints_deg[i]))
        self._robot._joints = list(joints_deg)
        try:
            self._robot._apply_frames(joints_deg, threadsafe=False)
        except Exception as e:                              # noqa: BLE001
            logger.debug("apply_frames error: %s", e)
        self._refresh_joint_sliders()
        self._refresh_pose_readout()
        if source:
            self._set_status(source)

    def _refresh_joint_sliders(self) -> None:
        for i, slider in enumerate(self._joint_sliders):
            v = self._robot._joints[i]
            slider.double_value = v
            self._joint_value_labels[i].text = f"{v:8.2f}°"

    def _refresh_pose_readout(self) -> None:
        self._fill_pose_row(self._tool_pose_labels,
                            self._tool_frames[self._tool_idx][1])
        self._fill_pose_row(self._ref_pose_labels,
                            self._ref_frames[self._ref_idx][1])
        T_world_tool = self._current_tool_world()
        if T_world_tool is not None:
            T_world_base = np.eye(4)
            T_world_base[:3, 3] = self._model.base_xyz_mm
            T_world_ref = T_world_base @ self._ref_frames[self._ref_idx][1]
            T_ref_tool = np.linalg.inv(T_world_ref) @ T_world_tool
            self._fill_pose_row(self._tcp_pose_labels, T_ref_tool)

    def _fill_pose_row(self, labels, T) -> None:
        x, y, z, rx, ry, rz = _matrix_to_xyz_rpy_deg(T)
        for lbl, val in zip(labels, (x, y, z, rx, ry, rz)):
            lbl.text = f"{val:11.3f}"

    def _set_status(self, msg: str) -> None:
        self._status.text = msg

    def _current_tool_world(self) -> np.ndarray | None:
        """T_world_tool = FK_URDF(joints) @ T_flange_tool (currently selected tool offset)."""
        try:
            q_rad = [math.radians(q) for q in self._robot._joints]
            T_world_tool0 = forward_kinematics_urdf(self._model, q_rad)
            return T_world_tool0 @ self._tool_frames[self._tool_idx][1]
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"FK error: {e}")
            return None

    def _jog_axis_world(
        self, axis_idx: int, T_world_tool: np.ndarray,
    ) -> np.ndarray:
        """Axis vector in WORLD coordinates for the selected jog frame (Tool / Reference / Base)."""
        unit = self._AXIS_VEC[axis_idx]
        frame = self.JOG_FRAMES[self._jog_frame_idx]
        if frame == "Tool Frame":
            return T_world_tool[:3, :3] @ unit
        if frame == "Reference Frame":
            T_world_base = np.eye(4)
            T_world_base[:3, 3] = self._model.base_xyz_mm
            T_world_ref = T_world_base @ self._ref_frames[self._ref_idx][1]
            return T_world_ref[:3, :3] @ unit
        return unit                                         # Base = world

    def _apply_cartesian_target(
        self, T_world_tool_target: np.ndarray, source: str,
    ) -> None:
        """IK seeded URDF → joints; clamp + apply."""
        T_flange_tool = self._tool_frames[self._tool_idx][1]
        T_world_tool0_target = T_world_tool_target @ np.linalg.inv(T_flange_tool)
        q_init = [math.radians(q) for q in self._robot._joints]
        sol = inverse_kinematics_seeded(
            self._model, T_world_tool0_target, q_init,
            tol_mm=0.5, tol_rad=1e-3, max_iter=100,
        )
        if sol is None:
            self._set_status(f"IK fail: {source}")
            return
        self._set_joints([math.degrees(q) for q in sol], source=source)

    def _find_alternate_solutions(
        self, T_world_tool_target: np.ndarray,
    ) -> list[list[float]]:
        """Generate diverse seeds → IK → collect ≤8 solutions differing >5° per joint."""
        T_flange_tool = self._tool_frames[self._tool_idx][1]
        T_world_tool0_target = T_world_tool_target @ np.linalg.inv(T_flange_tool)
        rng = np.random.RandomState(0)
        q_min = np.array([j.joint_min for j in self._model.joints])
        q_max = np.array([j.joint_max for j in self._model.joints])
        current_deg = list(self._robot._joints)
        seen: list[list[float]] = [current_deg]
        out: list[list[float]] = []

        def _is_new(sol_deg):
            for u in seen:
                if max(abs(a - b) for a, b in zip(sol_deg, u)) <= 5.0:
                    return False
            return True

        for _ in range(40):
            q_seed = rng.uniform(q_min, q_max)
            sol = inverse_kinematics_seeded(
                self._model, T_world_tool0_target, q_seed.tolist(),
                tol_mm=0.5, tol_rad=1e-3, max_iter=60, n_random_seeds=0,
            )
            if sol is None:
                continue
            sol_deg = [math.degrees(q) for q in sol]
            if _is_new(sol_deg):
                seen.append(sol_deg)
                out.append(sol_deg)
                if len(out) >= 8:
                    break
        return out
