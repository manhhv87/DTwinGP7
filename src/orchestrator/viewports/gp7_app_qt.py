"""
gp7_app_qt.py
─────────────
GP7 Digital Twin app — PyQt6 + pyvistaqt (VTK) industrial-standard stack.

Tách lớp triệt để:
  • Kinematics math (FK/IK/collision/trajectory) — REUSE NGUYÊN từ
    `src/orchestrator/kinematics/` (đã verify khớp RoboDK 0.00mm).
  • 3D rendering — VTK qua `pyvistaqt.QtInteractor` (cùng stack ROS RViz,
    MoveIt). Camera arcball/pan/zoom built-in.
  • GUI shell — PyQt6 native widgets (QMenuBar, QToolBar, QDockWidget,
    QFormLayout, QSlider, QPushButton). Stylesheet QSS có thể tùy biến.

Tại sao chọn stack này (so với Open3D gui hiện tại):
  • Industrial standard — VTK + Qt là chuẩn de-facto của robotics 3D apps.
  • UI polish — QFormLayout auto align label+widget, QDockWidget hide/show
    panel kiểu VSCode, QToolBar có icons + accelerators.
  • Math độc lập viewport — FK/IK/JBI export 100% giữ nguyên độ chính xác.

Launcher: `python scripts/16_app_qt.py`.
"""
from __future__ import annotations

import json
import logging
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import vtk
from PyQt6.QtCore import QEvent, QObject, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction, QActionGroup, QCloseEvent, QColor, QIcon, QKeySequence,
    QPainter, QPen, QPixmap, QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDial,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QInputDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from ..backends.inform_codegen import InformJobBuilder
from ..backends.motoman_hse import MotomanHSEBackend
from ..kinematics.inverse_kinematics import (
    inverse_kinematics, inverse_kinematics_batch, inverse_kinematics_seeded,
)
from ..kinematics.urdf_chain import (
    URDFRobot, forward_kinematics_urdf, gp7_urdf, link_frames_urdf,
)
from .control_panel import (
    _build_ref_frames,
    _build_tool_frames,
    _matrix_to_xyz_rpy_deg,
    _rotation_about_axis_3x3,
    _xyz_rpy_to_matrix,
)
from .gp7_app import Instruction  # reuse Program instruction model
from .open3d_gui_sim_robot import _GP7_MESH_MAP, _YASKAWA_BLUE

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _numpy_to_vtk_matrix(T: np.ndarray) -> vtk.vtkMatrix4x4:
    """numpy 4x4 → vtkMatrix4x4 (cho actor.SetUserMatrix)."""
    m = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            m.SetElement(i, j, float(T[i, j]))
    return m


# ──────────────────────────────────────────────────────────────────────────
# CollapsibleSection — header button + content area, click to expand/collapse
# ──────────────────────────────────────────────────────────────────────────

class CollapsibleSection(QWidget):
    """Section có nút header xổ/gập content. Default expanded=True or False.

    Qt KHÔNG có collapsible widget built-in (QGroupBox.checkable chỉ disable
    children, không hide). Custom: QPushButton header (text-align left,
    ▼/▶ arrow) + QWidget content có thể toggle visible.
    """

    def __init__(self, title: str, expanded: bool = True,
                  parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._expanded = expanded

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._toggle_btn = QPushButton()
        self._toggle_btn.setStyleSheet(
            "QPushButton {"
            "  text-align: left; padding: 5px 10px; "
            "  background-color: #3a3a3a; color: #e0e0e0; "
            "  border: none; border-bottom: 1px solid #2a2a2a; "
            "  font-weight: bold;"
            "}"
            "QPushButton:hover { background-color: #4a4a4a; }"
        )
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_text()
        self._toggle_btn.clicked.connect(self._toggle)
        outer.addWidget(self._toggle_btn)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 6, 8, 6)
        self._content_layout.setSpacing(4)
        self._content.setVisible(expanded)
        outer.addWidget(self._content)

    def _update_text(self) -> None:
        arrow = "▼" if self._expanded else "▶"
        self._toggle_btn.setText(f"{arrow}  {self._title}")

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        self._update_text()

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    def add_widget(self, w: QWidget) -> None:
        self._content_layout.addWidget(w)

    def add_layout(self, layout) -> None:
        self._content_layout.addLayout(layout)


# ──────────────────────────────────────────────────────────────────────────
# Worker signals — emit từ worker thread, slot chạy trên main thread
# ──────────────────────────────────────────────────────────────────────────

class _WorkerSignals(QObject):
    """Bridge worker thread → main thread (PyQt6 thread-safe pattern)."""

    joints_update = pyqtSignal(list)        # joints_deg
    status        = pyqtSignal(str, str)    # message, level (info/ok/warn/err)
    gripper       = pyqtSignal(bool)        # close
    demo_done     = pyqtSignal()
    program_done  = pyqtSignal()


# ──────────────────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────────────────

class _ScriptProgramAPI:
    """Read/write façade exposed tới user Python script qua biến `p`.

    Cho phép script append instruction vào job hiện tại + đọc target library.
    Tách lớp khỏi GP7AppQt để hạn chế surface mà script có thể đụng.
    """
    def __init__(self, app: "GP7AppQt") -> None:
        self._app = app

    @property
    def targets(self) -> dict:
        """Read-only view của target library."""
        return dict(self._app._targets)

    @property
    def active_job(self) -> str:
        return self._app._active_job

    def add_movej(self, joints: list[float]) -> None:
        """MoveJ với joints (6 deg)."""
        if len(joints) != 6:
            raise ValueError(f"joints phải 6 phần tử, got {len(joints)}")
        self._app._program.append(
            Instruction(type="MoveJ", joints=[float(q) for q in joints]))

    def add_movel(self, tcp_pose: list[float]) -> None:
        """MoveL với TCP pose [X,Y,Z mm, Rx,Ry,Rz deg] (WORLD frame)."""
        if len(tcp_pose) != 6:
            raise ValueError(f"tcp_pose phải 6 phần tử, got {len(tcp_pose)}")
        self._app._program.append(
            Instruction(type="MoveL", tcp_pose=[float(v) for v in tcp_pose]))

    def add_movej_to(self, target_name: str) -> None:
        """MoveJ → named target."""
        if target_name not in self._app._targets:
            raise KeyError(f"Target '{target_name}' không tồn tại")
        self._app._program.append(
            Instruction(type="MoveJ", target_name=target_name))

    def add_movel_to(self, target_name: str) -> None:
        """MoveL → named target."""
        if target_name not in self._app._targets:
            raise KeyError(f"Target '{target_name}' không tồn tại")
        self._app._program.append(
            Instruction(type="MoveL", target_name=target_name))

    def add_grip(self, close: bool) -> None:
        """SetGripper. close=True → CLOSE / False → OPEN."""
        self._app._program.append(
            Instruction(type="SetGripper", gripper_close=bool(close)))

    def add_wait(self, seconds: float) -> None:
        self._app._program.append(
            Instruction(type="Wait", wait_seconds=float(seconds)))

    def add_setspeed(self, vj_pct: float, v_mm_s: float) -> None:
        self._app._program.append(Instruction(
            type="SetSpeed",
            speed_joint_pct=float(vj_pct),
            speed_linear_mm_s=float(v_mm_s)))

    def add_msg(self, text: str) -> None:
        self._app._program.append(
            Instruction(type="ShowMessage", message=str(text)[:32]))

    def add_call(self, job_name: str) -> None:
        safe = "".join(c for c in str(job_name) if c.isalnum() or c == "_")[:32].upper()
        if not safe:
            raise ValueError(f"job_name không hợp lệ: '{job_name}'")
        self._app._program.append(Instruction(type="CallJob", job_name=safe))


class GP7AppQt(QMainWindow):
    """GP7 Digital Twin — PyQt6 main window + pyvistaqt 3D viewport."""

    JOG_FRAMES = ("Tool Frame", "Reference Frame", "Base")
    AXIS_NAMES = ("X", "Y", "Z")
    _AXIS_VEC = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    _CAM_PRESETS: dict[str, tuple[tuple[float, float, float],
                                    tuple[float, float, float],
                                    tuple[float, float, float]]] = {
        # name: (eye, center, up) in meters. Center = robot workspace mid.
        # Eye scaled ~1.5× từ center cho khoảng cách thoải mái khi startup
        # (trước scene to do camera gần). Đủ 6 góc chuẩn như RoboDK + Iso.
        "Iso":   (( 3.05, -2.25, 1.93), (0.35, 0.0, 0.50), (0.0, 0.0, 1.0)),
        "Front": (( 3.65,  0.00, 0.95), (0.35, 0.0, 0.50), (0.0, 0.0, 1.0)),
        "Back":  ((-2.95,  0.00, 0.95), (0.35, 0.0, 0.50), (0.0, 0.0, 1.0)),
        "Right": (( 0.35, -3.75, 1.25), (0.35, 0.0, 0.50), (0.0, 0.0, 1.0)),
        "Left":  (( 0.35,  3.75, 1.25), (0.35, 0.0, 0.50), (0.0, 0.0, 1.0)),
        "Top":   (( 0.36,  0.02, 5.00), (0.35, 0.0, 0.50), (1.0, 0.0, 0.0)),
    }

    # GP7 reach radii (m) — approximation. Real envelope is toroidal nhưng
    # sphere centered at J1 đủ visualize cho most use case.
    _REACH_FLANGE = 0.927          # max flange reach từ J1
    _REACH_WRIST  = 0.847          # = flange − 80mm tool0/flange offset
    _REACH_TOOL_EXTRA = 0.10       # tool offset typical ~100mm

    def __init__(
        self,
        cell_config: Any,
        project_root: str | Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("GP7 Digital Twin — PyQt6 + VTK")
        self.resize(1600, 950)

        self._cell_config = cell_config
        self._project_root = Path(project_root)

        # Robot model + state
        base_xyz = tuple(cell_config.robot.pose.xyz_mm) \
            if getattr(cell_config.robot, "pose", None) else (0.0, 0.0, 630.0)
        self._base_xyz = base_xyz
        self._model = gp7_urdf(base_xyz_mm=base_xyz)
        self._home_joints = list(cell_config.robot.home_joints_deg) \
            if getattr(cell_config.robot, "home_joints_deg", None) else [0.0] * 6
        self._joints = list(self._home_joints)

        # Frames + jog state
        self._tool_frames = _build_tool_frames(cell_config)
        self._ref_frames = _build_ref_frames(cell_config)
        self._tool_idx = len(self._tool_frames) - 1 if len(self._tool_frames) > 1 else 0
        self._ref_idx = 0
        self._jog_frame_idx = 0
        self._jog_step_mm = 10.0
        self._jog_step_deg = 5.0

        # Scene actors (link_name → vtkActor)
        self._link_actors: dict[str, Any] = {}
        # Object follow-gripper state
        self._objects: dict[str, dict[str, Any]] = {}
        self._grasped_name: str | None = None
        self._grasp_offset_m: np.ndarray | None = None

        # Show Frames — dynamic triad actors (vtkAxesActor) per frame key
        self._frame_actors: dict[str, Any] = {}
        self._frame_triad_size = 0.12                     # m, resizable
        # WorkSpace — single translucent sphere actor (or None)
        self._workspace_actor: Any = None
        # Fullscreen toggle state
        self._fullscreen = False

        # Custom toggle action state: action → [bool], action → base_text.
        # Vì native QAction.checkable + QSS không loại bỏ được ô vuông qua
        # Qt platform style trên Windows, ta KHÔNG dùng checkable. Thay vào
        # đó manage state qua dict + prefix text "✓ " khi active.
        self._toggle_states: dict[QAction, list] = {}
        self._toggle_texts: dict[QAction, str] = {}
        # Press-hold continuous jog timer
        self._jog_timer = QTimer(self)
        self._jog_timer.setInterval(120)
        self._jog_timer.timeout.connect(self._on_jog_tick)
        self._jog_active: tuple[str, int, int] | None = None  # (mode, axis, sign)

        # Worker control
        self._demo_thread: threading.Thread | None = None
        self._demo_stop = threading.Event()
        self._prog_thread: threading.Thread | None = None
        self._prog_stop = threading.Event()
        self._prog_pause = threading.Event()        # set = pause (held)
        # Multi-job project. Một dock chứa nhiều job; chỉ 1 job active tại
        # 1 thời điểm (combo box). self._program (property) trỏ vào job đang
        # active. Targets là project-global (RoboDK convention).
        self._jobs: dict[str, list[Instruction]] = {"MAIN": []}
        self._active_job: str = "MAIN"
        # Target library: name → {"joints": [..6 deg..], "tcp_pose": [..6..]}
        self._targets: dict[str, dict] = {}
        # Sim speed multiplier (1.0 = nominal). Tăng → animate faster.
        self._sim_speed_mult: float = 1.0
        # Post-processor settings (INFORM .JBI generation tuning).
        # max_speed_pct = safety cap cho VJ. default_vj/v = initial modal state.
        self._pp_max_speed_pct: float = 30.0
        self._pp_default_vj: float = 10.0
        self._pp_default_v_mms: float = 100.0
        # Teach-on-surface mode (click 3D scene → create target on picked cell).
        self._surface_pick_mode: bool = False
        # Cache: id(vtkDataSet) → (mesh_with_normals, cell_normals_array).
        # mesh.compute_normals() là O(N_cells) — expensive; cache để re-pick
        # cùng mesh không recompute.
        self._normal_cache: dict[int, tuple] = {}

        # HSE robot connection state — populated qua "Connection settings…" dialog.
        # Default lấy từ cell_config.robot_connection nếu có, không thì để trống.
        rc = getattr(cell_config, "robot_connection", None)
        self._hse_ip: str = (getattr(rc, "ip", "") or "")
        self._hse_tool_no: int = int(getattr(rc, "tool_no", 1) or 1)
        self._hse_ftp_user: str = (getattr(rc, "ftp_user", "") or "")
        self._hse_ftp_pass: str = (getattr(rc, "ftp_pass", "") or "")
        self._hse_ftp_dir: str = (getattr(rc, "ftp_job_dir", "/MPRAM1/JBI")
                                    or "/MPRAM1/JBI")
        self._hse_thread: threading.Thread | None = None
        self._hse_stop = threading.Event()

        # Worker → main signals
        self._signals = _WorkerSignals()
        self._signals.joints_update.connect(self._apply_joints_main)
        self._signals.status.connect(self._set_status)
        self._signals.gripper.connect(self._toggle_gripper)
        self._signals.demo_done.connect(self._on_demo_done)
        self._signals.program_done.connect(self._on_program_done)

        # Build UI (fast — Qt widget construction only)
        self._build_viewport()
        self._build_menu_bar()
        self._build_jog_dock()
        self._build_program_dock()
        self._build_status_bar()

        # Defer scene load (STL parse + VTK actor creation ~1-2s) — window
        # pop ra ngay, scene load sau khi event loop start → startup mượt.
        from PyQt6.QtCore import QTimer as _QT
        _QT.singleShot(0, self._post_show_setup)

    def _post_show_setup(self) -> None:
        """Chạy sau khi window đã show() — load scene + cam preset."""
        self._set_status("Loading scene...", level="info")
        self._load_scene()
        self._apply_joints_main(self._joints)
        self._set_camera_preset("Iso")
        self._set_status("Ready", level="ok")

    # ══════════════════════════════════════════════════════════════════
    # UI construction
    # ══════════════════════════════════════════════════════════════════

    def _build_viewport(self) -> None:
        """3D viewport bằng pyvistaqt.QtInteractor (VTK).

        Camera arcball/pan/zoom built-in. Background gradient set qua
        `set_background(color, top=color_top)`.
        """
        self._plotter = QtInteractor(self)
        self._plotter.set_background([95/255, 65/255, 175/255],          # bot (purple-blue)
                                      top=[5/255, 5/255, 28/255])         # top (near-black navy)
        try:
            self._plotter.enable_anti_aliasing()
        except Exception:                                  # noqa: BLE001
            pass
        # World axes widget ở góc trên-trái viewport
        self._plotter.add_axes(line_width=3, labels_off=False)
        self.setCentralWidget(self._plotter.interactor)

    # ── Custom toggle helpers (text prefix "✓ ", NOT native checkable) ──
    def _make_toggle(self, base_text: str, initial: bool = False,
                       callback=None) -> QAction:
        """QAction với visual ✓ prefix khi state True. KHÔNG dùng
        `checkable=True` để tránh native indicator (Qt platform style vẫn
        vẽ box trên Windows kể cả khi QSS hide).

        callback(new_state: bool) — fire khi user click. Có thể None.
        """
        act = QAction(self)
        self._toggle_states[act] = [bool(initial)]
        self._toggle_texts[act] = base_text

        def _on_clicked():
            new_state = not self._toggle_states[act][0]
            self._toggle_states[act][0] = new_state
            self._refresh_toggle_text(act)
            if callback is not None:
                callback(new_state)
        act.triggered.connect(_on_clicked)
        self._refresh_toggle_text(act)
        return act

    def _refresh_toggle_text(self, act: QAction) -> None:
        state = self._toggle_states[act][0]
        base = self._toggle_texts[act]
        # EM SPACE ( ) ≈ width of '✓' trong font tỉ lệ → text luôn
        # thẳng hàng giữa rows checked/unchecked.
        prefix = "✓ " if state else "  "
        act.setText(prefix + base)

    def _set_toggle(self, act: QAction, state: bool) -> None:
        """External setter: cập nhật state + refresh text WITHOUT fire callback."""
        if act in self._toggle_states:
            self._toggle_states[act][0] = bool(state)
            self._refresh_toggle_text(act)

    def _build_menu_bar(self) -> None:
        """Menu bar — File / View / Robot / Run / Program / Help.

        Mọi toggle action dùng custom prefix "✓  " (visible) hoặc "      "
        (6 spaces, ~ width của "✓  "). Không native checkable → không box.
        """
        mb = self.menuBar()

        # ── FILE ── File ops (Open / Save / Export) + Exit
        m_file = mb.addMenu("&File")
        act_open = QAction("&Open program (.json)...", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._on_prog_load_dlg)
        m_file.addAction(act_open)
        act_save = QAction("&Save program (.json)...", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self._on_prog_save_dlg)
        m_file.addAction(act_save)
        act_export = QAction("&Export .JBI (Yaskawa INFORM)...", self)
        act_export.triggered.connect(self._on_prog_export_dlg)
        m_file.addAction(act_export)
        m_file.addSeparator()
        act_exit = QAction("E&xit", self)
        act_exit.setShortcut(QKeySequence.StandardKey.Quit)
        act_exit.triggered.connect(self.close)
        m_file.addAction(act_exit)

        # ── VIEW ── 2 submenus + camera ops + window ops
        m_view = mb.addMenu("&View")

        # Camera submenu: Iso + 5 chính diện (RoboDK convention).
        # QActionGroup exclusive → checkable, click 1 preset thì các preset
        # khác auto-uncheck (kiểu radio button menu).
        # Camera presets — radio-style (exclusive) qua manual click handler.
        # KHÔNG dùng QActionGroup vì cần plain QAction (no checkable).
        cam_menu = m_view.addMenu("&Camera")
        self._cam_actions: dict[str, QAction] = {}
        for name, key in (("Iso", "1"), ("Top", "2"), ("Front", "3"),
                           ("Back", "4"), ("Right", "5"), ("Left", "6")):
            act = self._make_toggle(
                name, initial=(name == "Iso"),
                callback=lambda _state=False, n=name: self._set_camera_preset(n))
            act.setShortcut(f"Alt+{key}")
            cam_menu.addAction(act)
            self._cam_actions[name] = act

        # Visibility submenu — Background / Floor / Axes + size adjust
        vis_menu = m_view.addMenu("&Visibility")
        self._act_bg = self._make_toggle(
            "Background (gradient)", initial=True,
            callback=self._toggle_background)
        vis_menu.addAction(self._act_bg)
        self._act_floor = self._make_toggle(
            "Floor", initial=True,
            callback=lambda c: self._toggle_actor("__floor", c))
        vis_menu.addAction(self._act_floor)
        self._act_axes = self._make_toggle(
            "World axes triad", initial=True,
            callback=lambda c: self._toggle_actor("__world_axes", c))
        vis_menu.addAction(self._act_axes)
        vis_menu.addSeparator()
        act_triads_bigger = QAction("Reference frames &larger (+)", self)
        act_triads_bigger.setShortcut("+")
        act_triads_bigger.triggered.connect(lambda: self._resize_triads(1.25))
        vis_menu.addAction(act_triads_bigger)
        act_triads_smaller = QAction("Reference frames &smaller (-)", self)
        act_triads_smaller.setShortcut("-")
        act_triads_smaller.triggered.connect(lambda: self._resize_triads(0.8))
        vis_menu.addAction(act_triads_smaller)

        m_view.addSeparator()
        # Camera ops: Fit + Perspective toggle
        act_fit = QAction("&Fit all (reset camera)", self)
        act_fit.setShortcut("Alt+7")                       # RoboDK: Alt+7
        act_fit.triggered.connect(self._on_fit_all)
        m_view.addAction(act_fit)
        self._act_perspective = self._make_toggle(
            "Perspective view", initial=True,
            callback=self._on_toggle_perspective)
        m_view.addAction(self._act_perspective)

        m_view.addSeparator()
        # Window ops: Fullscreen + Close side panels
        self._act_fullscreen = self._make_toggle(
            "Fullscreen", initial=False,
            callback=self._on_toggle_fullscreen)
        self._act_fullscreen.setShortcut("F11")
        m_view.addAction(self._act_fullscreen)
        act_close_panels = QAction("Close side &windows", self)
        act_close_panels.setShortcut("Alt+C")              # RoboDK: Alt+C
        act_close_panels.triggered.connect(self._on_close_side_panels)
        m_view.addAction(act_close_panels)

        m_view.addSeparator()
        # Controls panel toggle (HIDDEN by default — user bật khi cần jog)
        self._act_jog_dock = self._make_toggle(
            "Show controls panel", initial=False,
            callback=lambda c: self._jog_dock.setVisible(c))
        m_view.addAction(self._act_jog_dock)

        # ── ROBOT ── motion + cell context
        m_robot = mb.addMenu("&Robot")
        for label, cb in (("Home", self._on_home), ("Zero", self._on_zero)):
            act = QAction(label, self); act.triggered.connect(cb)
            m_robot.addAction(act)
        m_robot.addSeparator()
        act_params = QAction("&Parameters (URDF/DH)...", self)
        act_params.triggered.connect(self._show_parameters_dlg)
        m_robot.addAction(act_params)
        act_cellinfo = QAction("Cell &info...", self)
        act_cellinfo.triggered.connect(self._show_cell_info)
        m_robot.addAction(act_cellinfo)
        m_robot.addSeparator()
        # Teach on surface — toggle mode để pick scene 3D tạo target
        self._act_surface_pick = self._make_toggle(
            "Teach on surface (Ctrl+Shift+T)", initial=False,
            callback=self._on_toggle_surface_pick)
        self._act_surface_pick.setShortcut("Ctrl+Shift+T")
        m_robot.addAction(self._act_surface_pick)
        m_robot.addSeparator()
        # HSE connection
        act_conn = QAction("&Connection settings... (HSE IP)", self)
        act_conn.triggered.connect(self._on_show_connection_settings)
        m_robot.addAction(act_conn)
        act_ping = QAction("&Test connection (ping HSE)", self)
        act_ping.triggered.connect(self._on_test_connection)
        m_robot.addAction(act_ping)

        # ── RUN ── runtime / scene operations
        m_run = mb.addMenu("R&un")
        self._act_demo = QAction("&Demo motion (start/stop)", self)
        self._act_demo.triggered.connect(self._on_demo_toggle)
        m_run.addAction(self._act_demo)
        act_reset = QAction("&Reset scene (restore objects)", self)
        act_reset.triggered.connect(self._on_reset_scene)
        m_run.addAction(act_reset)

        # ── PROGRAM ── panel toggle + runtime ops (Save/Load/Export ở File)
        m_prog = mb.addMenu("&Program")
        self._act_prog_dock = self._make_toggle(
            "Show program panel", initial=False,
            callback=lambda c: self._program_dock.setVisible(c))
        m_prog.addAction(self._act_prog_dock)
        m_prog.addSeparator()
        act_play = QAction("P&lay", self); act_play.triggered.connect(self._on_prog_play)
        m_prog.addAction(act_play)
        act_stop = QAction("&Stop", self); act_stop.triggered.connect(self._on_prog_stop)
        m_prog.addAction(act_stop)
        m_prog.addSeparator()
        act_clr = QAction("&Clear all", self); act_clr.triggered.connect(self._on_prog_clear)
        m_prog.addAction(act_clr)
        m_prog.addSeparator()
        act_pp = QAction("Post-&processor settings…", self)
        act_pp.triggered.connect(self._on_show_pp_settings)
        m_prog.addAction(act_pp)
        act_script = QAction("&Generate from Python script…", self)
        act_script.triggered.connect(self._on_show_script_editor)
        m_prog.addAction(act_script)

        # ── HELP ──
        m_help = mb.addMenu("&Help")
        act_about = QAction("&About...", self)
        act_about.triggered.connect(self._show_about)
        m_help.addAction(act_about)

    # NOTE: Separate QToolBar removed. Direct quick-action QActions added
    # to QMenuBar (same level as File/View/Robot/...) via _build_menu_bar().

    def _build_status_bar(self) -> None:
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        self._status_lbl = QLabel("Ready")
        sb.addWidget(self._status_lbl, 1)

    # ── Jog dock (left) — RoboDK-style: Cartesian TRÊN, Joint DƯỚI ────
    def _build_jog_dock(self) -> None:
        """Left dock — RoboDK panel layout:
          1. Cartesian Jog (Tool combo, Ref combo, 3 pose rows color-coded,
             Translate radio+buttons, Rotate radio+buttons, gripper)
          2. Joint axis jog (Align/Home buttons, 6 sliders)
          3. Other configurations (alternative IK solutions dropdown)
        """
        dock = QDockWidget("Yaskawa GP7 panel", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                              | Qt.DockWidgetArea.RightDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                          | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                          | QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self._jog_dock = dock

        inner = QWidget()
        vbox = QVBoxLayout(inner)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        # ──────────────────────────────────────────────────────
        # 1. CARTESIAN JOG  (Name + Parameters moved to toolbar)
        # ──────────────────────────────────────────────────────
        grp_cart = QGroupBox("Cartesian Jog")
        cv = QVBoxLayout(grp_cart)
        cv.setSpacing(4)

        # Tool + Ref combos on 1 row each (compact, no extra "w.r.t." text)
        combo_form = QFormLayout()
        combo_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        combo_form.setHorizontalSpacing(8)
        combo_form.setVerticalSpacing(4)

        self._tool_combo = QComboBox()
        for name, _T in self._tool_frames: self._tool_combo.addItem(name)
        self._tool_combo.setCurrentIndex(self._tool_idx)
        self._tool_combo.currentIndexChanged.connect(self._on_tool_changed)
        combo_form.addRow("Tool", self._tool_combo)

        self._ref_combo = QComboBox()
        for name, _T in self._ref_frames: self._ref_combo.addItem(name)
        self._ref_combo.setCurrentIndex(self._ref_idx)
        self._ref_combo.currentIndexChanged.connect(self._on_ref_changed)
        combo_form.addRow("Ref", self._ref_combo)
        cv.addLayout(combo_form)

        # ── TCP pose (Tool / Reference) — PRIMARY readout, always visible ──
        tcp_lbl = QLabel("Tool / Reference  (live TCP pose)")
        tcp_lbl.setStyleSheet("color: #88c8ff; font-weight: bold; padding-top: 4px;")
        cv.addWidget(tcp_lbl)
        self._tcp_pose_lbls = self._make_colored_pose_row(cv)

        # ── Tool/Flange + Ref/Base poses — COLLAPSED by default (static info) ──
        adv_sec = CollapsibleSection(
            "Frame poses — Tool / Flange + Ref / Base (static, advanced)",
            expanded=False)
        adv_sec.add_widget(QLabel("Tool / Flange:"))
        self._tool_pose_lbls = self._make_colored_pose_row(adv_sec.content_layout())
        adv_sec.add_widget(QLabel("Reference / Base:"))
        self._ref_pose_lbls = self._make_colored_pose_row(adv_sec.content_layout())
        cv.addWidget(adv_sec)

        # Jog frame + Step combined into 1 compact row
        jog_step_row = QHBoxLayout()
        jog_step_row.addWidget(QLabel("Jog"))
        self._jog_frame_combo = QComboBox()
        for n in self.JOG_FRAMES: self._jog_frame_combo.addItem(n)
        self._jog_frame_combo.setCurrentIndex(self._jog_frame_idx)
        self._jog_frame_combo.currentIndexChanged.connect(
            lambda i: setattr(self, "_jog_frame_idx", int(i)))
        jog_step_row.addWidget(self._jog_frame_combo, 1)
        self._step_mm_spin = QDoubleSpinBox()
        self._step_mm_spin.setRange(0.1, 500.0); self._step_mm_spin.setValue(self._jog_step_mm)
        self._step_mm_spin.setSuffix(" mm"); self._step_mm_spin.setFixedWidth(90)
        self._step_mm_spin.valueChanged.connect(
            lambda v: setattr(self, "_jog_step_mm", float(v)))
        jog_step_row.addWidget(self._step_mm_spin)
        self._step_deg_spin = QDoubleSpinBox()
        self._step_deg_spin.setRange(0.1, 90.0); self._step_deg_spin.setValue(self._jog_step_deg)
        self._step_deg_spin.setSuffix(" °"); self._step_deg_spin.setFixedWidth(80)
        self._step_deg_spin.valueChanged.connect(
            lambda v: setattr(self, "_jog_step_deg", float(v)))
        jog_step_row.addWidget(self._step_deg_spin)
        cv.addLayout(jog_step_row)

        # ── 3-column layout: jog grid+dial | WorkSpace | Show Frames ──
        # Như RoboDK panel: 3 columns side-by-side, không stack vertical.
        # RADIO BUTTONS: gộp 6 radio (Translate X/Y/Z + Rotate X/Y/Z) vào
        # CÙNG 1 QButtonGroup — exclusive → user chỉ chọn 1 trong 6 → dial
        # điều khiển ĐÚNG 1 selection đó.
        cols_row = QHBoxLayout()
        cols_row.setSpacing(8)

        # === Column 1: axis grid + dial ===
        left_col = QVBoxLayout(); left_col.setSpacing(6)
        mode_grid = QGridLayout()
        mode_grid.setHorizontalSpacing(6); mode_grid.setVerticalSpacing(2)
        for c, axis in enumerate(self.AXIS_NAMES):
            lbl = QLabel(axis); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-weight: bold;")
            mode_grid.addWidget(lbl, 0, c + 1)
        mode_grid.addWidget(QLabel("Translation"), 1, 0)
        mode_grid.addWidget(QLabel("Rotation"),    2, 0)

        # SINGLE shared group cho cả 6 radio — exclusive across BOTH rows.
        # Button id encoding: 0/1/2 = Trans X/Y/Z, 3/4/5 = Rot X/Y/Z.
        self._jog_axis_group = QButtonGroup(self)
        for i in range(3):
            rb = QRadioButton()
            if i == 0: rb.setChecked(True)                  # default: Translate X
            self._jog_axis_group.addButton(rb, i)
            cell = QWidget(); ch = QHBoxLayout(cell)
            ch.setContentsMargins(0, 0, 0, 0); ch.addStretch()
            ch.addWidget(rb); ch.addStretch()
            mode_grid.addWidget(cell, 1, i + 1)
        for i in range(3):
            rb = QRadioButton()
            self._jog_axis_group.addButton(rb, 3 + i)
            cell = QWidget(); ch = QHBoxLayout(cell)
            ch.setContentsMargins(0, 0, 0, 0); ch.addStretch()
            ch.addWidget(rb); ch.addStretch()
            mode_grid.addWidget(cell, 2, i + 1)
        left_col.addLayout(mode_grid)

        # Jog Dial — ROTARY ENCODER style:
        # • Mỗi notch xoay qua = 1 step jog (axis + sign theo radio đã chọn).
        # • Thả chuột ra → dial GIỮ NGUYÊN vị trí (không snap về 0).
        # • Tiếp tục xoay → tiếp tục jog. Wrap=True để xoay không giới hạn.
        # Cách track: lưu `_last_dial_value`, mỗi valueChanged tính delta (có
        # xử lý wrap-around 360°→0°), accumulate, mỗi STEP_THRESHOLD = 1 step.
        self._jog_dial = QDial()
        self._jog_dial.setRange(0, 359)
        self._jog_dial.setValue(0)
        self._jog_dial.setNotchesVisible(True)
        self._jog_dial.setWrapping(True)                  # xoay không giới hạn
        self._jog_dial.setFixedSize(80, 80)
        self._jog_dial.setToolTip(
            "Xoay dial (như rotary encoder) → jog từng step theo radio đã chọn.\n"
            "Xoay phải = sign +, xoay trái = sign −. Thả ra giữ nguyên vị trí.")
        self._last_dial_value = 0
        self._dial_accumulator = 0.0
        self._jog_dial.valueChanged.connect(self._on_dial_value_changed)
        left_col.addWidget(self._jog_dial, 0, Qt.AlignmentFlag.AlignCenter)
        cols_row.addLayout(left_col, 1)

        # === Column 2: WorkSpace ===
        cols_row.addWidget(self._build_workspace_group(), 1)
        # === Column 3: Show Frames ===
        cols_row.addWidget(self._build_show_frames_group(), 1)
        cv.addLayout(cols_row)

        vbox.addWidget(grp_cart)

        # ──────────────────────────────────────────────────────
        # 2. JOINT AXIS JOG — collapsible, OPEN by default (primary control)
        # ──────────────────────────────────────────────────────
        joints_sec = CollapsibleSection("Joint axis jog", expanded=True)
        jhdr = QHBoxLayout()
        jhdr.addStretch()
        self._align_btn = QPushButton("Align")
        self._align_btn.setToolTip(
            "Align with closest target (placeholder — snaps to home in sim)")
        self._align_btn.clicked.connect(self._on_home)
        jhdr.addWidget(self._align_btn)
        home_btn = QPushButton("Home"); home_btn.clicked.connect(self._on_home)
        jhdr.addWidget(home_btn)
        joints_sec.add_layout(jhdr)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8); grid.setVerticalSpacing(4)
        self._joint_sliders: list[QSlider] = []
        self._joint_value_lbls: list[QLabel] = []
        for i, joint in enumerate(self._model.joints):
            jmin = math.degrees(joint.joint_min)
            jmax = math.degrees(joint.joint_max)
            tlbl = QLabel(f"θ{i+1}")
            tlbl.setStyleSheet("font-weight: bold;")
            grid.addWidget(tlbl, i, 0)
            val_lbl = QLabel(f"{self._joints[i]:+7.2f}°")
            val_lbl.setFixedWidth(70)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            grid.addWidget(val_lbl, i, 1)
            min_lbl = QLabel(f"{jmin:+.0f}")
            min_lbl.setFixedWidth(40)
            min_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            grid.addWidget(min_lbl, i, 2)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(int(jmin * 100), int(jmax * 100))
            slider.setValue(int(self._joints[i] * 100))
            slider.setMinimumWidth(180)
            slider.valueChanged.connect(
                lambda v, idx=i: self._on_joint_slider(idx, v / 100.0))
            grid.addWidget(slider, i, 3)
            max_lbl = QLabel(f"{jmax:+.0f}")
            max_lbl.setFixedWidth(40)
            grid.addWidget(max_lbl, i, 4)
            self._joint_sliders.append(slider)
            self._joint_value_lbls.append(val_lbl)
        joints_sec.add_layout(grid)
        vbox.addWidget(joints_sec)

        # ──────────────────────────────────────────────────────
        # 3. OTHER CONFIGURATIONS — collapsible, CLOSED by default
        # ──────────────────────────────────────────────────────
        other_sec = CollapsibleSection(
            "Other configurations — alternative IK branches", expanded=False)
        oh = QHBoxLayout()
        oh.addWidget(QLabel("(θ1..θ6)"))
        oh.addStretch()
        find_btn = QPushButton("Find branches")
        find_btn.setToolTip("Search IK alternative solutions for current TCP pose")
        find_btn.clicked.connect(self._on_find_alternates)
        oh.addWidget(find_btn)
        other_sec.add_layout(oh)
        self._alt_combo = QComboBox()
        self._alt_combo.addItem("(no alternates — click \"Find branches\")")
        self._alt_combo.currentIndexChanged.connect(self._on_alternate_selected)
        other_sec.add_widget(self._alt_combo)
        self._alt_solutions: list[list[float]] = []
        vbox.addWidget(other_sec)

        vbox.addStretch()

        # Wrap inner trong QScrollArea — panel rất dài (Cartesian + WorkSpace +
        # Show Frames + Joints + Other configs), không scroll thì content bị
        # cắt mất phần dưới.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        # 6 colored cells × 50px + dial 80px + grid → ~400px đủ
        dock.setMinimumWidth(400)
        # ẨN MẶC ĐỊNH — user vào menu View > Show controls panel để bật.
        dock.setVisible(False)
        # SYNC menu checkbox khi user đóng dock bằng X trên title bar
        dock.visibilityChanged.connect(self._sync_jog_dock_check)

    # ── WorkSpace + Show Frames groupboxes (used INSIDE Cartesian Jog) ──
    def _build_workspace_group(self) -> QGroupBox:
        grp = QGroupBox("WorkSpace")
        v = QVBoxLayout(grp)
        self._ws_group = QButtonGroup(self)
        opts = [
            ("none",   "Do not show"),
            ("wrist",  "Show for wrist center"),
            ("flange", "Show for robot flange"),
            ("tool",   "Show for current tool"),
        ]
        for i, (key, label) in enumerate(opts):
            rb = QRadioButton(label)
            if i == 0: rb.setChecked(True)
            self._ws_group.addButton(rb, i)
            v.addWidget(rb)
        self._ws_group.idClicked.connect(
            lambda i: self._on_workspace_changed(opts[i][0]))
        return grp

    def _build_show_frames_group(self) -> QGroupBox:
        grp = QGroupBox("Show Frames")
        g = QGridLayout(grp)
        g.setHorizontalSpacing(8); g.setVerticalSpacing(2)
        self._frame_checks: dict[str, QCheckBox] = {}
        # Row 0: All/None + Base
        cb_all = QCheckBox("All/None")
        cb_all.stateChanged.connect(self._on_show_frames_all)
        g.addWidget(cb_all, 0, 0)
        cb_base = QCheckBox("Base (0)")
        cb_base.stateChanged.connect(
            lambda s: self._on_toggle_frame("base", bool(s)))
        g.addWidget(cb_base, 0, 1)
        self._frame_checks["base"] = cb_base
        # Row 1: Tool + Robot Flange
        cb_tool = QCheckBox("Tool Frame")
        cb_tool.stateChanged.connect(
            lambda s: self._on_toggle_frame("tool", bool(s)))
        g.addWidget(cb_tool, 1, 0)
        self._frame_checks["tool"] = cb_tool
        cb_fl = QCheckBox("Robot Flange")
        cb_fl.stateChanged.connect(
            lambda s: self._on_toggle_frame("flange", bool(s)))
        g.addWidget(cb_fl, 1, 1)
        self._frame_checks["flange"] = cb_fl
        # Row 2: Ref Frame
        cb_ref = QCheckBox("Ref. Frame")
        cb_ref.stateChanged.connect(
            lambda s: self._on_toggle_frame("ref", bool(s)))
        g.addWidget(cb_ref, 2, 0)
        self._frame_checks["ref"] = cb_ref
        # Row 3-4: J1..J6 in 2 rows × 3 cols
        for i in range(6):
            key = f"joint_{i+1}"
            cb = QCheckBox(str(i + 1))
            cb.stateChanged.connect(
                lambda s, k=key: self._on_toggle_frame(k, bool(s)))
            g.addWidget(cb, 3 + (i // 3), i % 3)
            self._frame_checks[key] = cb
        return grp

    # ── Pose row helper: 6 colored boxes (RoboDK X/Y/Z/Rx/Ry/Rz pastels) ──
    AXIS_BG_HEX = ("#f4a8b0", "#a8e8a8", "#a8c4f0",
                    "#a8e8e8", "#f0a8f0", "#f0f0a8")
    AXIS_NAMES_FULL = ("X", "Y", "Z", "Rx", "Ry", "Rz")

    def _make_colored_pose_row(self, parent_layout) -> list[QLabel]:
        """1 dòng duy nhất: 6 cells color-coded narrow (50px) + ⎘ 📋 icons.

        Header format ("[X,Y,Z]mm Rot[X,Y,Z]deg") đã bỏ — title của
        CollapsibleSection (hoặc Label trên) đã chỉ rõ pose nào. Tiết kiệm
        1 dòng/pose × 3 poses = 3 dòng.
        """
        row = QHBoxLayout()
        row.setSpacing(1)
        labels = []
        for bg in self.AXIS_BG_HEX:
            lbl = QLabel("0.000")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumWidth(50)
            lbl.setStyleSheet(
                f"background-color: {bg}; color: #000; "
                f"padding: 3px; border: 1px solid #555;"
                f"font-family: Consolas, 'Courier New', monospace;"
                f"font-size: 11px;"
            )
            row.addWidget(lbl, 1)
            labels.append(lbl)
        cpy = QPushButton("⎘"); cpy.setFixedWidth(24)
        cpy.setToolTip("Copy pose to clipboard")
        pst = QPushButton("📋"); pst.setFixedWidth(24)
        pst.setToolTip("Paste pose from clipboard (TCP target)")
        row.addWidget(cpy); row.addWidget(pst)
        parent_layout.addLayout(row)
        cpy.clicked.connect(lambda: self._copy_pose_to_clipboard(labels))
        pst.clicked.connect(self._paste_pose_from_clipboard)
        return labels

    def _copy_pose_to_clipboard(self, labels: list[QLabel]) -> None:
        """Copy 6 raw values từ labels (no axis prefix) ra clipboard."""
        vals = []
        for lbl in labels:
            try:
                vals.append(float(lbl.text()))
            except ValueError:
                vals.append(0.0)
        s = ", ".join(f"{v:.3f}" for v in vals)
        QApplication.clipboard().setText(s)
        self._set_status(f"Pose copied: [{s}]", level="ok")

    def _paste_pose_from_clipboard(self) -> None:
        """Đọc 'X,Y,Z,Rx,Ry,Rz' từ clipboard → IK → set TCP target.
        Interpretation: pose là Tool / Reference (giống pose row dưới cùng).
        """
        text = QApplication.clipboard().text().strip()
        parts = [p.strip() for p in text.replace(";", ",").split(",")]
        if len(parts) != 6:
            self._set_status(
                f"Clipboard '{text[:40]}' không phải 6-value pose", level="err")
            return
        try:
            x, y, z, rx, ry, rz = (float(p) for p in parts)
        except ValueError:
            self._set_status("Cannot parse clipboard as pose values", level="err")
            return
        # Pose w.r.t. Reference → world frame TCP target
        T_ref_tool = _xyz_rpy_to_matrix(x, y, z, rx, ry, rz)
        T_world_base = np.eye(4); T_world_base[:3, 3] = self._base_xyz
        T_world_ref = T_world_base @ self._ref_frames[self._ref_idx][1]
        T_world_tool = T_world_ref @ T_ref_tool
        self._apply_cartesian_target(T_world_tool, "Paste pose")

    # ── Program dock (right) ──────────────────────────────────────────
    def _build_program_dock(self) -> None:
        """Layout: program list (top, primary) → edit toolbar → Targets group →
        Add tabs (Motion / Logic / Modal) → Playback bar → File bar.
        Workflow top-down: Teach targets → Add instructions → Run → Save."""
        dock = QDockWidget("Program", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                              | Qt.DockWidgetArea.RightDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                          | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                          | QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self._program_dock = dock

        root = QWidget()
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(6)

        # ══ 0. JOB SELECTOR (multi-job project) ══════════════════════
        job_row = QHBoxLayout(); job_row.setSpacing(4)
        job_row.addWidget(QLabel("Job"))
        self._job_combo = QComboBox()
        self._job_combo.setMinimumWidth(120)
        self._job_combo.currentTextChanged.connect(self._on_job_changed)
        job_row.addWidget(self._job_combo, 1)
        for label, cb, tip in (
            ("+",  self._on_job_add,    "Add new job"),
            ("⟲",  self._on_job_rename, "Rename current job"),
            ("✕",  self._on_job_delete, "Delete current job"),
        ):
            b = QPushButton(label); b.setToolTip(tip); b.setFixedWidth(28)
            b.clicked.connect(cb)
            job_row.addWidget(b)
        jw = QWidget(); jw.setLayout(job_row)
        vbox.addWidget(jw)

        # ══ 1. PROGRAM LIST (primary surface, stretch=3) ═════════════
        self._prog_list = QListWidget()
        self._prog_list.setMinimumHeight(180)
        vbox.addWidget(self._prog_list, 3)

        # ── 1b. Edit toolbar (↑ ↓ Edit ✕) ─────────────────────────────
        edit_row = QHBoxLayout(); edit_row.setSpacing(4)
        for label, cb, tip, width in (
            ("↑",      self._on_prog_move_up,   "Move selected up",      40),
            ("↓",      self._on_prog_move_down, "Move selected down",    40),
            ("Edit",   self._on_prog_modify,    "F2 — edit selected",    50),
            ("✕",      self._on_prog_delete,    "Delete selected",       40),
        ):
            b = QPushButton(label); b.setToolTip(tip); b.setFixedWidth(width)
            b.clicked.connect(cb)
            edit_row.addWidget(b)
        edit_row.addStretch()
        eb = QWidget(); eb.setLayout(edit_row)
        vbox.addWidget(eb)
        # Double-click instruction in list → Edit
        self._prog_list.itemDoubleClicked.connect(
            lambda *_: self._on_prog_modify())
        # F2 app-wide shortcut → modify selected instruction
        QShortcut(QKeySequence("F2"), self,
                   activated=self._on_prog_modify)

        # ══ 2. TARGETS group (anchored — workflow starts here) ═══════
        tgt_grp = QGroupBox("Targets")
        tgt_lay = QVBoxLayout(tgt_grp); tgt_lay.setSpacing(4)
        self._tgt_list = QListWidget()
        self._tgt_list.setMaximumHeight(90)
        tgt_lay.addWidget(self._tgt_list)
        # Name + Teach row
        tname_row = QHBoxLayout()
        self._tgt_name_edit = QLineEdit()
        self._tgt_name_edit.setPlaceholderText("name (e.g. PICK, HOME)")
        self._tgt_name_edit.setMaxLength(24)
        b_teach = QPushButton("+ Teach")
        b_teach.setToolTip("Ctrl+T — capture current pose as new target")
        b_teach.setShortcut("Ctrl+T")
        b_teach.clicked.connect(self._on_tgt_teach)
        tname_row.addWidget(QLabel("Name")); tname_row.addWidget(self._tgt_name_edit, 1)
        tname_row.addWidget(b_teach)
        tnw = QWidget(); tnw.setLayout(tname_row)
        tgt_lay.addWidget(tnw)
        # Modify/Delete/Go-to/Config + use-as-move (1 row, separated by stretch)
        tact_row = QHBoxLayout()
        b_mod = QPushButton("Modify"); b_mod.setShortcut("F3")
        b_mod.setToolTip("F3 — replace selected target with current pose")
        b_mod.clicked.connect(self._on_tgt_modify)
        b_del = QPushButton("Delete"); b_del.clicked.connect(self._on_tgt_delete)
        b_goto = QPushButton("Go to")
        b_goto.setToolTip("Animate robot to selected target (preview)")
        b_goto.clicked.connect(self._on_tgt_goto)
        b_cfg = QPushButton("Config")
        b_cfg.setToolTip("F4 — pick alternative IK configuration for selected target")
        b_cfg.setShortcut("F4")
        b_cfg.clicked.connect(self._on_tgt_change_config)
        b_uj = QPushButton("+ MoveJ→")
        b_uj.setToolTip("Add MoveJ → selected target")
        b_uj.clicked.connect(lambda: self._on_prog_add_move_to_target("MoveJ"))
        b_ul = QPushButton("+ MoveL→")
        b_ul.setToolTip("Add MoveL → selected target")
        b_ul.clicked.connect(lambda: self._on_prog_add_move_to_target("MoveL"))
        tact_row.addWidget(b_mod); tact_row.addWidget(b_del)
        tact_row.addWidget(b_goto); tact_row.addWidget(b_cfg)
        tact_row.addStretch()
        tact_row.addWidget(b_uj); tact_row.addWidget(b_ul)
        taw = QWidget(); taw.setLayout(tact_row)
        tgt_lay.addWidget(taw)
        vbox.addWidget(tgt_grp)

        # ══ 3. ADD INSTRUCTION (3 tabs: Motion / Logic / Modal) ═════
        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        # ─ Tab: Motion ─
        mot_w = QWidget()
        mot_lay = QGridLayout(mot_w); mot_lay.setSpacing(4)
        b_mj = QPushButton("+ MoveJ"); b_mj.setToolTip("Joint move tới pose hiện tại")
        b_mj.clicked.connect(self._on_prog_add_movej)
        b_ml = QPushButton("+ MoveL"); b_ml.setToolTip("Linear move tới pose hiện tại")
        b_ml.clicked.connect(self._on_prog_add_movel)
        self._btn_movec = QPushButton("+ MoveC (set MID)")
        self._btn_movec.setToolTip("2-step: click 1 = MID waypoint, click 2 = END")
        self._btn_movec.clicked.connect(self._on_prog_add_movec)
        self._pending_movc_mid: list[float] | None = None
        mot_lay.addWidget(b_mj,            0, 0)
        mot_lay.addWidget(b_ml,            0, 1)
        mot_lay.addWidget(self._btn_movec, 1, 0, 1, 2)
        mot_lay.setRowStretch(2, 1)
        tabs.addTab(mot_w, "Motion")

        # ─ Tab: Logic (gripper + waits + msg) ─
        log_w = QWidget()
        log_lay = QVBoxLayout(log_w); log_lay.setSpacing(4)
        # Gripper row
        gr_row = QHBoxLayout()
        b_go = QPushButton("+ Grip OPEN")
        b_go.clicked.connect(lambda: self._on_prog_add_gripper(False))
        b_gc = QPushButton("+ Grip CLOSE")
        b_gc.clicked.connect(lambda: self._on_prog_add_gripper(True))
        gr_row.addWidget(b_go); gr_row.addWidget(b_gc)
        gw = QWidget(); gw.setLayout(gr_row); log_lay.addWidget(gw)
        # Wait row
        w_row = QHBoxLayout()
        w_row.addWidget(QLabel("Wait"))
        self._prog_wait_spin = QDoubleSpinBox()
        self._prog_wait_spin.setRange(0.0, 600.0); self._prog_wait_spin.setValue(0.5)
        self._prog_wait_spin.setSuffix(" s")
        w_row.addWidget(self._prog_wait_spin)
        b_wait = QPushButton("+ Wait"); b_wait.clicked.connect(self._on_prog_add_wait)
        w_row.addWidget(b_wait)
        ww = QWidget(); ww.setLayout(w_row); log_lay.addWidget(ww)
        # WaitIO row
        io_row = QHBoxLayout()
        io_row.addWidget(QLabel("IN#"))
        self._prog_io_idx = QSpinBox(); self._prog_io_idx.setRange(1, 1024); self._prog_io_idx.setValue(1)
        io_row.addWidget(self._prog_io_idx)
        io_row.addWidget(QLabel("="))
        self._prog_io_state = QComboBox(); self._prog_io_state.addItems(["ON", "OFF"])
        io_row.addWidget(self._prog_io_state)
        io_row.addWidget(QLabel("T"))
        self._prog_io_tout = QDoubleSpinBox()
        self._prog_io_tout.setRange(0.0, 600.0); self._prog_io_tout.setValue(0.0)
        self._prog_io_tout.setSuffix("s"); self._prog_io_tout.setToolTip("0 = block vô hạn")
        io_row.addWidget(self._prog_io_tout)
        b_wio = QPushButton("+ WaitIO"); b_wio.clicked.connect(self._on_prog_add_waitio)
        io_row.addWidget(b_wio)
        iow = QWidget(); iow.setLayout(io_row); log_lay.addWidget(iow)
        # MSG row
        msg_row = QHBoxLayout()
        msg_row.addWidget(QLabel("MSG"))
        self._prog_msg_edit = QLineEdit()
        self._prog_msg_edit.setMaxLength(32)
        self._prog_msg_edit.setPlaceholderText("≤ 32 ASCII")
        msg_row.addWidget(self._prog_msg_edit, 1)
        b_msg = QPushButton("+ MSG"); b_msg.clicked.connect(self._on_prog_add_msg)
        msg_row.addWidget(b_msg)
        mw = QWidget(); mw.setLayout(msg_row); log_lay.addWidget(mw)
        # CallJob row (sub-program invocation)
        call_row = QHBoxLayout()
        call_row.addWidget(QLabel("Call"))
        self._prog_call_edit = QLineEdit()
        self._prog_call_edit.setMaxLength(32)
        self._prog_call_edit.setPlaceholderText("sub-job name (e.g. WELD_A)")
        call_row.addWidget(self._prog_call_edit, 1)
        b_call = QPushButton("+ Call"); b_call.clicked.connect(self._on_prog_add_calljob)
        call_row.addWidget(b_call)
        cw = QWidget(); cw.setLayout(call_row); log_lay.addWidget(cw)
        # SimEvent row (sim-only checkpoint, không export INFORM)
        ev_row = QHBoxLayout()
        ev_row.addWidget(QLabel("Event"))
        self._prog_ev_edit = QLineEdit()
        self._prog_ev_edit.setMaxLength(32)
        self._prog_ev_edit.setPlaceholderText("sim checkpoint name")
        ev_row.addWidget(self._prog_ev_edit, 1)
        b_ev = QPushButton("+ SimEvent")
        b_ev.setToolTip("Sim checkpoint — không export ra .JBI")
        b_ev.clicked.connect(self._on_prog_add_simevent)
        ev_row.addWidget(b_ev)
        ew = QWidget(); ew.setLayout(ev_row); log_lay.addWidget(ew)
        log_lay.addStretch()
        tabs.addTab(log_w, "Logic")

        # ─ Tab: Modal (Speed / Rounding / Tool / RefFrame) ─
        mod_w = QWidget()
        mod_lay = QVBoxLayout(mod_w); mod_lay.setSpacing(4)
        # SetSpeed
        sp_row = QHBoxLayout()
        sp_row.addWidget(QLabel("VJ"))
        self._prog_spd_vj = QDoubleSpinBox()
        self._prog_spd_vj.setRange(1.0, 30.0); self._prog_spd_vj.setValue(10.0); self._prog_spd_vj.setSuffix("%")
        sp_row.addWidget(self._prog_spd_vj)
        sp_row.addWidget(QLabel("V"))
        self._prog_spd_v = QDoubleSpinBox()
        self._prog_spd_v.setRange(1.0, 250.0); self._prog_spd_v.setValue(100.0); self._prog_spd_v.setSuffix("mm/s")
        sp_row.addWidget(self._prog_spd_v)
        b_spd = QPushButton("+ SetSpeed"); b_spd.clicked.connect(self._on_prog_add_setspeed)
        sp_row.addWidget(b_spd)
        spw = QWidget(); spw.setLayout(sp_row); mod_lay.addWidget(spw)
        # SetRounding
        pl_row = QHBoxLayout()
        pl_row.addWidget(QLabel("PL"))
        self._prog_pl = QSpinBox(); self._prog_pl.setRange(0, 8); self._prog_pl.setValue(0)
        pl_row.addWidget(self._prog_pl); pl_row.addStretch()
        b_pl = QPushButton("+ SetRounding"); b_pl.clicked.connect(self._on_prog_add_setrounding)
        pl_row.addWidget(b_pl)
        plw = QWidget(); plw.setLayout(pl_row); mod_lay.addWidget(plw)
        # SetTool
        tl_row = QHBoxLayout()
        tl_row.addWidget(QLabel("TL#"))
        self._prog_tool_no = QSpinBox(); self._prog_tool_no.setRange(0, 15); self._prog_tool_no.setValue(0)
        tl_row.addWidget(self._prog_tool_no); tl_row.addStretch()
        b_tool = QPushButton("+ SetTool"); b_tool.clicked.connect(self._on_prog_add_settool)
        tl_row.addWidget(b_tool)
        tlw = QWidget(); tlw.setLayout(tl_row); mod_lay.addWidget(tlw)
        # SetRefFrame
        uf_row = QHBoxLayout()
        uf_row.addWidget(QLabel("UF#"))
        self._prog_uf_no = QSpinBox(); self._prog_uf_no.setRange(0, 15); self._prog_uf_no.setValue(0)
        uf_row.addWidget(self._prog_uf_no); uf_row.addStretch()
        b_uf = QPushButton("+ SetRefFrame"); b_uf.clicked.connect(self._on_prog_add_setrefframe)
        uf_row.addWidget(b_uf)
        ufw = QWidget(); ufw.setLayout(uf_row); mod_lay.addWidget(ufw)
        mod_lay.addStretch()
        tabs.addTab(mod_w, "Modal")

        vbox.addWidget(tabs)

        # ══ 4. PLAYBACK bar (always visible) ═════════════════════════
        pb_row = QHBoxLayout(); pb_row.setSpacing(6)
        b_play = QPushButton("▶ Sim")
        b_play.setMinimumHeight(34)
        b_play.setToolTip("Play program in simulation (no robot)")
        b_play.setStyleSheet(
            "QPushButton { background-color: #2da44e; color: white; "
            "font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #2c974b; }")
        b_play.clicked.connect(self._on_prog_play)
        b_run_robot = QPushButton("⚙ Run on Robot")
        b_run_robot.setMinimumHeight(34)
        b_run_robot.setToolTip(
            "Upload + execute current job trên YRC1000 thật via HSE")
        b_run_robot.setStyleSheet(
            "QPushButton { background-color: #fb8500; color: white; "
            "font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #d96e00; }")
        b_run_robot.clicked.connect(self._on_run_on_robot)
        self._btn_pause = QPushButton("⏸")
        self._btn_pause.setCheckable(True)
        self._btn_pause.setMinimumHeight(34)
        self._btn_pause.setToolTip("Pause/Resume sim playback")
        self._btn_pause.clicked.connect(self._on_prog_toggle_pause)
        b_stop = QPushButton("⏹ Stop")
        b_stop.setMinimumHeight(34)
        b_stop.setToolTip("Stop sim OR emergency-stop robot (servo OFF)")
        b_stop.setStyleSheet(
            "QPushButton { background-color: #cf222e; color: white; "
            "font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #a40e26; }")
        b_stop.clicked.connect(self._on_stop_all)
        pb_row.addWidget(b_play, 2)
        pb_row.addWidget(b_run_robot, 2)
        pb_row.addWidget(self._btn_pause, 1)
        pb_row.addWidget(b_stop, 1)
        pb_row.addWidget(QLabel("Speed"))
        self._sim_speed_spin = QDoubleSpinBox()
        self._sim_speed_spin.setRange(0.25, 5.0); self._sim_speed_spin.setSingleStep(0.25)
        self._sim_speed_spin.setValue(1.0); self._sim_speed_spin.setSuffix("×")
        self._sim_speed_spin.valueChanged.connect(
            lambda v: setattr(self, "_sim_speed_mult", float(v)))
        pb_row.addWidget(self._sim_speed_spin)
        pbw = QWidget(); pbw.setLayout(pb_row)
        vbox.addWidget(pbw)

        # ══ 5. FILE bar (Save / Load / Export / Clear) ════════════════
        file_row = QHBoxLayout(); file_row.setSpacing(4)
        for label, cb in (
            ("Save",        self._on_prog_save_dlg),
            ("Load",        self._on_prog_load_dlg),
            ("Export .JBI", self._on_prog_export_dlg),
            ("Clear all",   self._on_prog_clear),
        ):
            b = QPushButton(label); b.clicked.connect(cb)
            file_row.addWidget(b)
        fw = QWidget(); fw.setLayout(file_row)
        vbox.addWidget(fw)

        # Populate job combo from initial _jobs dict.
        self._refresh_job_combo()

        # Wrap trong QScrollArea để dock width nhỏ vẫn không cắt content.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(root)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        dock.setVisible(False)
        dock.visibilityChanged.connect(self._sync_prog_dock_check)

    # ══════════════════════════════════════════════════════════════════
    # Scene loading (pyvista / VTK)
    # ══════════════════════════════════════════════════════════════════

    def _load_scene(self) -> None:
        """Load tất cả mesh + lights vào pyvista plotter."""
        self._add_floor()
        self._add_world_axes_triad()
        self._load_robot_links()
        self._load_gripper()
        self._add_cell_meshes()
        self._setup_lighting()

    def _add_floor(self) -> None:
        try:
            # 4×3m plane lát gạch — pyvista Plane primitive + procedural texture
            plane = pv.Plane(center=(0.8, 0.0, 0.0), direction=(0, 0, 1),
                              i_size=4.0, j_size=3.0, i_resolution=10, j_resolution=8)
            self._floor_actor = self._plotter.add_mesh(
                plane, color=[0.92, 0.92, 0.95], show_edges=True,
                edge_color=[0.55, 0.55, 0.60], line_width=2,
                name="__floor", lighting=True)
        except Exception as e:                              # noqa: BLE001
            logger.debug("Floor lỗi: %s", e)

    def _add_world_axes_triad(self) -> None:
        """Triad nhỏ 300mm ở gốc world. pyvista có add_axes() ở góc viewport,
        nhưng đây là triad ngay trong scene để align với robot base.
        """
        try:
            triad = pv.Axes(show_actor=True, actor_scale=0.3, line_width=4)
            self._plotter.add_actor(triad.actor, name="__world_axes")
        except Exception as e:                              # noqa: BLE001
            logger.debug("World axes lỗi: %s", e)

    def _setup_lighting(self) -> None:
        # pyvista default lighting đủ tốt; có thể thêm light tùy chỉnh nếu cần.
        try:
            self._plotter.enable_shadows = False            # tắt shadow cho perf
        except Exception:                                   # noqa: BLE001
            pass

    def _load_robot_links(self) -> bool:
        """Load 7 STL ros-industrial → add vào plotter, lưu actor để animate.

        NOTE: thử ThreadPoolExecutor cho STL parse — chậm hơn 13% do GIL +
        per-file work nhỏ (~10ms). pv.read mặc dù gọi C code nhưng Python-side
        dataset wrapping vẫn hold GIL. STL load tổng ~65ms (SSD warm cache),
        không đáng parallelize.
        """
        mesh_dir = self._project_root / "models" / "gp7_links"
        if not mesh_dir.exists():
            logger.warning("Robot mesh dir không có: %s", mesh_dir)
            return False
        loaded = 0
        for key, fname, off in _GP7_MESH_MAP:
            path = mesh_dir / fname
            if not path.exists():
                continue
            try:
                mesh = pv.read(str(path))
                if any(off):
                    mesh.translate([off[0], off[1], off[2]], inplace=True)
                actor = self._plotter.add_mesh(
                    mesh, color=list(_YASKAWA_BLUE), name=key,
                    smooth_shading=True, pbr=False)
                self._link_actors[key] = actor
                loaded += 1
            except Exception as e:                          # noqa: BLE001
                logger.debug("Lỗi load mesh %s: %s", path, e)
        logger.info("GP7AppQt: %d/%d GP7 link mesh", loaded, len(_GP7_MESH_MAP))
        return loaded > 0

    def _load_gripper(self) -> None:
        cfg = getattr(self._cell_config, "gripper", None)
        if cfg is None or not getattr(cfg, "mesh", None):
            return
        path = Path(cfg.mesh)
        if not path.is_absolute():
            path = self._project_root / path
        if not path.exists():
            return
        try:
            mesh = pv.read(str(path))
            mesh.points *= 0.001                            # mm → m
            mesh.translate([0.0, 0.0, 0.1], inplace=True)   # offset palm
            actor = self._plotter.add_mesh(
                mesh, color=[0.78, 0.78, 0.80], name="gripper",
                smooth_shading=True)
            self._link_actors["gripper"] = actor
        except Exception as e:                              # noqa: BLE001
            logger.debug("Gripper lỗi: %s", e)

    def _add_cell_meshes(self) -> None:
        cfg = self._cell_config
        for attr, drgb in (("worktable", [0.52, 0.55, 0.58]),
                            ("robot_pedestal", [0.40, 0.40, 0.40]),
                            ("camera_mount", [0.50, 0.50, 0.50])):
            item = getattr(cfg, attr, None)
            if item is None or not getattr(item, "mesh", None):
                continue
            rgb = getattr(item, "color_rgb", None) or drgb
            self._add_static_mesh(attr, item.mesh, item.pose.xyz_mm,
                                    item.pose.rpy_deg, rgb)
        frames = {f.name: f for f in getattr(cfg, "frames", []) or []}
        for obj in getattr(cfg, "objects", []) or []:
            if not getattr(obj, "mesh", None):
                continue
            pxyz = list(frames[obj.parent_frame].pose.xyz_mm) \
                if getattr(obj, "parent_frame", None) in frames else [0, 0, 0]
            off = list(obj.pose.xyz_mm) if getattr(obj, "pose", None) else [0, 0, 0]
            world_xyz_mm = [pxyz[k] + off[k] for k in range(3)]
            self._register_object(obj.name, obj.mesh, world_xyz_mm,
                                   rgb=[0.80, 0.75, 0.20])

    def _add_static_mesh(self, name, mesh_rel, xyz_mm, rpy_deg, rgb) -> None:
        try:
            path = Path(mesh_rel)
            if not path.is_absolute():
                path = self._project_root / path
            if not path.exists():
                return
            mesh = pv.read(str(path))
            mesh.points *= 0.001
            # Apply rpy rotation + xyz translation
            T = _xyz_rpy_to_matrix(0, 0, 0, *rpy_deg)
            mesh.transform(T, inplace=True)
            mesh.translate([xyz_mm[0] / 1000.0, xyz_mm[1] / 1000.0,
                             xyz_mm[2] / 1000.0], inplace=True)
            self._plotter.add_mesh(mesh, color=rgb, name=name,
                                     smooth_shading=True)
        except Exception as e:                              # noqa: BLE001
            logger.debug("Static mesh '%s' lỗi: %s", mesh_rel, e)

    def _register_object(self, name, mesh_rel, world_xyz_mm, rgb) -> None:
        try:
            path = Path(mesh_rel)
            if not path.is_absolute():
                path = self._project_root / path
            if not path.exists():
                return
            mesh = pv.read(str(path))
            mesh.points *= 0.001
            actor = self._plotter.add_mesh(mesh, color=rgb, name=name,
                                              smooth_shading=True)
            world_T = np.eye(4)
            world_T[:3, 3] = [v / 1000.0 for v in world_xyz_mm]
            actor.SetUserMatrix(_numpy_to_vtk_matrix(world_T))
            self._objects[name] = {
                "actor": actor,
                "world_T": world_T.copy(),
                "initial_world_T": world_T.copy(),
            }
        except Exception as e:                              # noqa: BLE001
            logger.debug("Object '%s' lỗi: %s", name, e)

    def _set_camera_preset(self, name: str) -> None:
        eye, ctr, up = self._CAM_PRESETS.get(name, self._CAM_PRESETS["Iso"])
        try:
            self._plotter.camera_position = [eye, ctr, up]
            self._plotter.render()
        except Exception as e:                              # noqa: BLE001
            logger.debug("camera preset lỗi: %s", e)
        # Sync menu visual: exclusive — chỉ `name` checked, others unchecked.
        if hasattr(self, "_cam_actions"):
            for n, a in self._cam_actions.items():
                self._set_toggle(a, n == name)

    def _toggle_actor(self, name: str, visible: bool) -> None:
        try:
            actor = self._plotter.renderer.actors.get(name)
            if actor is None:
                return
            actor.SetVisibility(bool(visible))
            self._plotter.render()
        except Exception as e:                              # noqa: BLE001
            logger.debug("Toggle %s lỗi: %s", name, e)

    def _toggle_background(self, gradient_on: bool) -> None:
        """Toggle background giữa gradient navy/purple ↔ solid dark gray."""
        try:
            if gradient_on:
                self._plotter.set_background(
                    [95 / 255, 65 / 255, 175 / 255],            # bottom purple-blue
                    top=[5 / 255, 5 / 255, 28 / 255])           # top near-black navy
            else:
                self._plotter.set_background([0.10, 0.10, 0.13])
            self._plotter.render()
        except Exception as e:                              # noqa: BLE001
            logger.debug("Toggle background lỗi: %s", e)

    # ══════════════════════════════════════════════════════════════════
    # View ops: Fit all / Perspective / Fullscreen / Close panels / Resize
    # ══════════════════════════════════════════════════════════════════
    def _on_fit_all(self) -> None:
        """Reset camera để fit toàn scene (như RoboDK 'Fit all' Alt+7).
        Sau Fit all → camera không match preset → uncheck tất cả preset.
        """
        try:
            self._plotter.reset_camera()
            self._plotter.render()
            if hasattr(self, "_cam_actions"):
                for act in self._cam_actions.values():
                    self._set_toggle(act, False)
            self._set_status("Camera reset (fit all)", level="ok")
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"Fit all lỗi: {e}", level="err")

    def _on_toggle_perspective(self, perspective_on: bool) -> None:
        """Toggle perspective ↔ orthographic projection (engineering view)."""
        try:
            cam = self._plotter.camera
            cam.SetParallelProjection(not perspective_on)
            self._plotter.render()
            mode = "Perspective" if perspective_on else "Orthographic"
            self._set_status(f"Projection: {mode}", level="ok")
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"Projection toggle lỗi: {e}", level="err")

    def _on_toggle_fullscreen(self, fullscreen_on: bool) -> None:
        """Toggle window fullscreen (F11). Bỏ menu + dock + status."""
        self._fullscreen = bool(fullscreen_on)
        if self._fullscreen:
            self.showFullScreen()
        else:
            self.showNormal()

    # ══════════════════════════════════════════════════════════════════
    # Menu checkbox sync — đồng bộ khi user thay đổi qua route khác
    # (vd. close dock bằng X title bar, hoặc thoát fullscreen bằng Esc)
    # ══════════════════════════════════════════════════════════════════
    def _sync_jog_dock_check(self, visible: bool) -> None:
        """Khi controls panel ẩn/hiện qua bất kỳ route nào → sync menu tick."""
        if hasattr(self, "_act_jog_dock"):
            self._set_toggle(self._act_jog_dock, bool(visible))

    def _sync_prog_dock_check(self, visible: bool) -> None:
        """Tương tự cho program panel."""
        if hasattr(self, "_act_prog_dock"):
            self._set_toggle(self._act_prog_dock, bool(visible))

    def changeEvent(self, event) -> None:
        """Bắt window state change (Esc thoát fullscreen, maximize, v.v.) →
        sync menu Fullscreen tick state.
        """
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            if hasattr(self, "_act_fullscreen"):
                is_fs = self.isFullScreen()
                self._set_toggle(self._act_fullscreen, is_fs)
                self._fullscreen = is_fs

    def _on_close_side_panels(self) -> None:
        """Đóng cả controls + program panel — xem full viewport (Alt+C)."""
        for dock, act in (
            (self._jog_dock, self._act_jog_dock),
            (self._program_dock, self._act_prog_dock),
        ):
            dock.setVisible(False)
            self._set_toggle(act, False)
        self._set_status("Side panels closed", level="ok")

    def _resize_triads(self, factor: float) -> None:
        """Scale tất cả visible frame triads lên/xuống theo `factor` (1.25 / 0.8).
        VTK vtkAxesActor có `SetTotalLength` nên chỉ cần update size + re-set
        trên mọi actor đang hiện."""
        self._frame_triad_size = max(0.02, min(1.0, self._frame_triad_size * factor))
        s = self._frame_triad_size
        for actor in self._frame_actors.values():
            try:
                actor.SetTotalLength(s, s, s)
            except Exception:                              # noqa: BLE001
                pass
        try:
            self._plotter.render()
        except Exception:                                   # noqa: BLE001
            pass
        self._set_status(f"Frame triads size: {s*1000:.0f} mm", level="ok")

    # ══════════════════════════════════════════════════════════════════
    # Animation (joint → actor transform)
    # ══════════════════════════════════════════════════════════════════

    def _apply_joints_main(self, joints_deg: list[float]) -> None:
        """Main-thread slot: cập nhật joint state + actor transforms + readouts."""
        self._joints = list(joints_deg)
        self._render_scene_frame(joints_deg)
        self._refresh_joint_sliders()
        self._refresh_pose_readout()

    def _render_scene_frame(self, joints_deg: list[float]) -> None:
        """Compute FK qua link_frames_urdf → SetUserMatrix mỗi link actor."""
        frames = dict(link_frames_urdf(
            self._model, [math.radians(q) for q in joints_deg]))
        for link_key, actor in self._link_actors.items():
            # gripper bám link_tool0
            frame_key = "link_tool0" if link_key == "gripper" else link_key
            T = frames.get(frame_key)
            if T is None:
                continue
            Tm = T.copy()
            Tm[:3, 3] = T[:3, 3] / 1000.0                  # mm → m
            try:
                actor.SetUserMatrix(_numpy_to_vtk_matrix(Tm))
            except Exception:                               # noqa: BLE001
                pass
        # Object follow gripper
        T_tool0_mm = frames.get("link_tool0")
        if T_tool0_mm is not None:
            for name, obj in self._objects.items():
                if (self._grasped_name == name
                        and self._grasp_offset_m is not None):
                    T_tool_m = T_tool0_mm.copy()
                    T_tool_m[:3, 3] = T_tool0_mm[:3, 3] / 1000.0
                    obj["world_T"] = T_tool_m @ self._grasp_offset_m
                try:
                    obj["actor"].SetUserMatrix(
                        _numpy_to_vtk_matrix(obj["world_T"]))
                except Exception:                           # noqa: BLE001
                    pass
        # Update visible frame triads (joint_N, tool, flange — these depend on joints)
        self._update_dynamic_frames()
        self._plotter.render()

    # Throttle animation render rate. 30Hz đủ smooth (mắt nhận ~24Hz là phim),
    # ít hơn render() invocation 25% so với 40Hz.
    _ANIM_MAX_FPS: float = 30.0

    def _animate_to(self, end_deg: list[float], steps: int = 36, dt: float = 0.02,
                     stop_event: threading.Event | None = None,
                     pause_event: threading.Event | None = None) -> None:
        """Worker-thread slerp animation: emit joints_update per frame.

        Honors stop_event (abort) và pause_event (block while set).
        Throttle: skip emit nếu chưa đủ 1/_ANIM_MAX_FPS từ frame trước → giảm
        Qt event queue + VTK render() load.
        """
        start = list(self._joints)
        n = min(len(start), len(end_deg))
        min_emit_dt = 1.0 / self._ANIM_MAX_FPS
        last_emit_t = 0.0
        for s in range(1, steps + 1):
            if stop_event is not None and stop_event.is_set():
                return
            if pause_event is not None:
                while pause_event.is_set():
                    if stop_event is not None and stop_event.is_set():
                        return
                    time.sleep(0.05)
            t = s / steps
            frame = [start[k] + (end_deg[k] - start[k]) * t for k in range(n)]
            # Emit chỉ khi đủ throttle interval HOẶC final frame (đảm bảo
            # arrive exactly at target).
            now = time.monotonic()
            is_final = (s == steps)
            if is_final or (now - last_emit_t) >= min_emit_dt:
                self._signals.joints_update.emit(frame)
                last_emit_t = now
            time.sleep(dt)

    # ══════════════════════════════════════════════════════════════════
    # State helpers
    # ══════════════════════════════════════════════════════════════════

    def _refresh_joint_sliders(self) -> None:
        for i, slider in enumerate(self._joint_sliders):
            slider.blockSignals(True)
            slider.setValue(int(self._joints[i] * 100))
            slider.blockSignals(False)
            self._joint_value_lbls[i].setText(f"{self._joints[i]:+7.2f}°")

    def _fill_pose_row(self, labels: list[QLabel], T: np.ndarray) -> None:
        """Cập nhật 6 colored labels (X/Y/Z/Rx/Ry/Rz) từ matrix 4x4.
        Format CHỈ giá trị (không "X:" prefix) — narrow cells."""
        x, y, z, rx, ry, rz = _matrix_to_xyz_rpy_deg(T)
        for lbl, v in zip(labels, (x, y, z, rx, ry, rz)):
            lbl.setText(f"{v:.3f}")

    def _refresh_pose_readout(self) -> None:
        # Tool / Flange (static — chỉ đổi khi đổi Tool combo)
        T_flange_tool = self._tool_frames[self._tool_idx][1]
        self._fill_pose_row(self._tool_pose_lbls, T_flange_tool)
        # Reference / Base (static — chỉ đổi khi đổi Ref combo)
        T_base_ref = self._ref_frames[self._ref_idx][1]
        self._fill_pose_row(self._ref_pose_lbls, T_base_ref)
        # Tool / Reference (live — phụ thuộc joint state)
        T_world_tool = self._current_tool_world()
        if T_world_tool is not None:
            T_world_base = np.eye(4); T_world_base[:3, 3] = self._base_xyz
            T_world_ref = T_world_base @ T_base_ref
            T_ref_tool = np.linalg.inv(T_world_ref) @ T_world_tool
            self._fill_pose_row(self._tcp_pose_lbls, T_ref_tool)

    # ── Other configurations (IK branches) ────────────────────────────
    def _on_find_alternates(self) -> None:
        """Tìm alternative IK solutions cho TCP pose hiện tại — RoboDK 'Other
        configurations' equivalent. Dùng inverse_kinematics_seeded với
        nhiều random seeds, gom unique solutions (cách nhau >5° per joint).
        """
        T = self._current_tool_world()
        if T is None:
            return
        T_flange_tool = self._tool_frames[self._tool_idx][1]
        T_target_tool0 = T @ np.linalg.inv(T_flange_tool)
        rng = np.random.RandomState(0)
        q_min = np.array([j.joint_min for j in self._model.joints])
        q_max = np.array([j.joint_max for j in self._model.joints])
        current_deg = list(self._joints)
        seen: list[list[float]] = [current_deg]
        unique: list[list[float]] = []

        def _is_new(sol_deg):
            for u in seen:
                if max(abs(a - b) for a, b in zip(sol_deg, u)) <= 5.0:
                    return False
            return True

        for _ in range(40):
            q_seed = rng.uniform(q_min, q_max)
            sol = inverse_kinematics_seeded(
                self._model, T_target_tool0, q_seed.tolist(),
                tol_mm=0.5, tol_rad=1e-3, max_iter=60, n_random_seeds=0)
            if sol is None:
                continue
            sol_deg = [math.degrees(q) for q in sol]
            if _is_new(sol_deg):
                seen.append(sol_deg)
                unique.append(sol_deg)
                if len(unique) >= 8:
                    break

        self._alt_solutions = unique
        self._alt_combo.blockSignals(True)
        self._alt_combo.clear()
        if not unique:
            self._alt_combo.addItem("(no alternative IK branches)")
            self._set_status("No alternate IK branches found", level="warn")
        else:
            for sol in unique:
                label = "( * )-[ " + ",  ".join(f"{q:+7.2f}°" for q in sol) + " ]"
                self._alt_combo.addItem(label)
            self._set_status(f"Found {len(unique)} alternative IK branch(es)", "ok")
        self._alt_combo.blockSignals(False)

    def _on_alternate_selected(self, idx: int) -> None:
        i = int(idx)
        if 0 <= i < len(self._alt_solutions):
            self._apply_joints_main(self._alt_solutions[i])
            self._set_status(f"Switched to IK branch #{i+1}", level="ok")

    # ══════════════════════════════════════════════════════════════════
    # Parameters dialog
    # ══════════════════════════════════════════════════════════════════
    def _show_parameters_dlg(self) -> None:
        from PyQt6.QtWidgets import QDialog, QTextEdit
        dlg = QDialog(self)
        dlg.setWindowTitle("Robot parameters (read-only)")
        dlg.resize(700, 560)
        lay = QVBoxLayout(dlg)
        text = QTextEdit(); text.setReadOnly(True)
        lines = [
            f"<h3>{self._cell_config.robot.name}</h3>",
            f"<b>Base:</b> xyz = {list(self._base_xyz)} mm<br><br>",
            f"<b>URDF joints ({len(self._model.joints)}):</b>",
            "<table border='1' cellpadding='6' cellspacing='0'>",
            "<tr style='background:#444;color:#eee;'>"
            "<th>Joint</th><th>Origin (mm)</th><th>Axis</th>"
            "<th>Min (°)</th><th>Max (°)</th></tr>",
        ]
        for j in self._model.joints:
            lines.append(
                f"<tr><td><b>{j.name}</b></td>"
                f"<td>{list(j.origin_mm)}</td>"
                f"<td>{list(j.axis)}</td>"
                f"<td>{math.degrees(j.joint_min):+7.1f}</td>"
                f"<td>{math.degrees(j.joint_max):+7.1f}</td></tr>")
        lines.append("</table><br>")
        lines.append(f"<b>Flange offset (fixed):</b> "
                     f"{list(self._model.flange_xyz_mm)} mm<br>")
        lines.append(f"<b>Tool0 rotation rpy (rad):</b> "
                     f"{[round(v, 4) for v in self._model.tool0_rpy_rad]}<br>")
        lines.append(f"<b>Home joints (deg):</b> "
                     + ", ".join(f"{q:+7.2f}" for q in self._home_joints) + "<br><br>")
        lines.append("<b>Verification:</b> FK match RoboDK SolveFK to 0.00 mm.<br>")
        text.setHtml("".join(lines))
        lay.addWidget(text)
        ok = QPushButton("Close"); ok.clicked.connect(dlg.accept)
        lay.addWidget(ok)
        dlg.exec()

    # ══════════════════════════════════════════════════════════════════
    # WorkSpace sphere
    # ══════════════════════════════════════════════════════════════════
    def _on_workspace_changed(self, mode: str) -> None:
        # Clear existing
        if self._workspace_actor is not None:
            try:
                self._plotter.remove_actor("__workspace")
            except Exception:                              # noqa: BLE001
                pass
            self._workspace_actor = None
        if mode == "none":
            self._plotter.render()
            return
        radius = {
            "wrist":  self._REACH_WRIST,
            "flange": self._REACH_FLANGE,
            "tool":   self._REACH_FLANGE + self._REACH_TOOL_EXTRA,
        }.get(mode)
        if radius is None:
            return
        center = (self._base_xyz[0] / 1000.0,
                  self._base_xyz[1] / 1000.0,
                  self._base_xyz[2] / 1000.0)
        try:
            sphere = pv.Sphere(radius=radius, center=center,
                                theta_resolution=48, phi_resolution=24)
            self._workspace_actor = self._plotter.add_mesh(
                sphere, color=[0.4, 0.85, 1.0], opacity=0.12,
                name="__workspace", show_edges=False, lighting=False,
                style="surface")
            self._plotter.render()
            self._set_status(
                f"Workspace ({mode}): radius {radius*1000:.0f} mm", level="ok")
        except Exception as e:                             # noqa: BLE001
            self._set_status(f"Workspace error: {e}", level="err")

    # ══════════════════════════════════════════════════════════════════
    # Show Frames (triad axes per frame)
    # ══════════════════════════════════════════════════════════════════
    # Default 0.12m; resize qua View > Visibility > Reference frames ± .
    # Instance attr (không class const) để resize động.

    def _on_show_frames_all(self, state: int) -> None:
        """All/None toggle — set tất cả frame checkbox theo state."""
        checked = bool(state)
        for cb in self._frame_checks.values():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        # Apply each manually
        for key in list(self._frame_checks.keys()):
            self._on_toggle_frame(key, checked)

    def _on_toggle_frame(self, key: str, visible: bool) -> None:
        if not visible:
            self._remove_frame_triad(key)
            return
        T = self._frame_world_matrix(key)
        if T is None:
            return
        self._add_frame_triad(key, T)

    def _frame_world_matrix(self, key: str) -> np.ndarray | None:
        """Tính world transform (meters) cho frame key. Returns None nếu N/A."""
        # Base = robot base position (no rotation)
        T_world_base = np.eye(4); T_world_base[:3, 3] = self._base_xyz
        T_world_base_m = T_world_base.copy()
        T_world_base_m[:3, 3] /= 1000.0
        if key == "base":
            return T_world_base_m
        if key == "ref":
            T_world_ref = T_world_base @ self._ref_frames[self._ref_idx][1]
            T = T_world_ref.copy(); T[:3, 3] /= 1000.0
            return T
        # Dynamic frames depending on current joints
        try:
            q_rad = [math.radians(q) for q in self._joints]
            frames = dict(link_frames_urdf(self._model, q_rad))
        except Exception:                                  # noqa: BLE001
            return None
        if key == "flange":
            T = frames.get("link_flange")
        elif key == "tool":
            T_tool0 = frames.get("link_tool0")
            if T_tool0 is None: return None
            T_flange_tool = self._tool_frames[self._tool_idx][1]
            T = T_tool0 @ T_flange_tool
        elif key.startswith("joint_"):
            idx = int(key.split("_")[1])
            joint_name = self._model.joints[idx - 1].name
            T = frames.get(f"link_{joint_name}")
        else:
            return None
        if T is None:
            return None
        T_m = T.copy(); T_m[:3, 3] /= 1000.0                # mm → m
        return T_m

    def _add_frame_triad(self, key: str, T_world_m: np.ndarray) -> None:
        if key in self._frame_actors:
            # Already exists; just update transform
            self._update_frame_triad(key, T_world_m); return
        actor = vtk.vtkAxesActor()
        s = self._frame_triad_size                          # resizable instance attr
        actor.SetTotalLength(s, s, s)
        actor.SetShaftType(0)                              # cylinder
        # Bỏ XYZ text labels cho gọn (axes có sẵn 3 màu R/G/B)
        actor.GetXAxisCaptionActor2D().SetVisibility(False)
        actor.GetYAxisCaptionActor2D().SetVisibility(False)
        actor.GetZAxisCaptionActor2D().SetVisibility(False)
        transform = vtk.vtkTransform()
        transform.SetMatrix(_numpy_to_vtk_matrix(T_world_m))
        actor.SetUserTransform(transform)
        self._plotter.renderer.AddActor(actor)
        self._frame_actors[key] = actor
        self._plotter.render()

    def _update_frame_triad(self, key: str, T_world_m: np.ndarray) -> None:
        actor = self._frame_actors.get(key)
        if actor is None: return
        transform = vtk.vtkTransform()
        transform.SetMatrix(_numpy_to_vtk_matrix(T_world_m))
        actor.SetUserTransform(transform)

    def _remove_frame_triad(self, key: str) -> None:
        actor = self._frame_actors.pop(key, None)
        if actor is not None:
            try:
                self._plotter.renderer.RemoveActor(actor)
                self._plotter.render()
            except Exception:                              # noqa: BLE001
                pass

    def _update_dynamic_frames(self) -> None:
        """Khi robot move → update transform cho mọi frame triad đang visible."""
        for key in list(self._frame_actors.keys()):
            T = self._frame_world_matrix(key)
            if T is not None:
                self._update_frame_triad(key, T)

    # ══════════════════════════════════════════════════════════════════
    # Press-hold continuous jog
    # ══════════════════════════════════════════════════════════════════
    def _on_jog_tick(self) -> None:
        if self._jog_active is None:
            return
        mode, axis, sign = self._jog_active
        if mode == "T":
            self._on_translate(axis, sign)
        else:
            self._on_rotate(axis, sign)

    def _start_continuous_jog(self, mode: str, axis: int, sign: int) -> None:
        """Press-hold continuous jog — auto-fire mỗi 120ms cho tới khi released."""
        self._jog_active = (mode, axis, sign)
        # Delay 250ms trước khi auto-fire (tránh trigger khi user chỉ click)
        QTimer.singleShot(250, self._jog_timer.start)

    def _stop_continuous_jog(self) -> None:
        self._jog_active = None
        self._jog_timer.stop()

    # ── Circular jog dial (QDial) — rotary encoder semantics ──────────
    # Mỗi DIAL_DEG_PER_STEP độ xoay qua = 1 jog step. Persistent vị trí
    # (không snap về 0 khi release chuột).
    DIAL_DEG_PER_STEP = 30                                  # độ/step

    def _on_dial_value_changed(self, v: int) -> None:
        """Tính delta angle (xử lý wrap 0↔359) → accumulate → fire jog mỗi
        khi vượt qua DIAL_DEG_PER_STEP. Axis + sign theo radio đã chọn.
        """
        delta = v - self._last_dial_value
        # Wrap-around handling: 358° → 1° là +3°, không phải −357°.
        if delta > 180:
            delta -= 360
        elif delta < -180:
            delta += 360
        self._last_dial_value = v
        self._dial_accumulator += delta

        # Fire jog cho mỗi STEP_THRESHOLD độ đã accumulate
        while abs(self._dial_accumulator) >= self.DIAL_DEG_PER_STEP:
            sign = +1 if self._dial_accumulator > 0 else -1
            self._dial_accumulator -= sign * self.DIAL_DEG_PER_STEP
            bid = self._jog_axis_group.checkedId()
            if bid < 0:
                continue
            if bid < 3:
                self._on_translate(bid, sign)
            else:
                self._on_rotate(bid - 3, sign)

    def _set_status(self, msg: str, level: str = "info") -> None:
        color = {"info": "#d8d8d8", "ok": "#5cf08c",
                 "warn": "#ffc870", "err": "#ff7373"}.get(level, "#d8d8d8")
        self._status_lbl.setText(msg)
        self._status_lbl.setStyleSheet(f"color: {color};")

    def _current_tool_world(self) -> np.ndarray | None:
        try:
            q_rad = [math.radians(q) for q in self._joints]
            T_world_tool0 = dict(link_frames_urdf(self._model, q_rad)).get(
                "link_tool0")
            if T_world_tool0 is None:
                return None
            return T_world_tool0 @ self._tool_frames[self._tool_idx][1]
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"FK error: {e}", level="err")
            return None

    def _jog_axis_world(self, axis_idx, T_world_tool):
        unit = self._AXIS_VEC[axis_idx]
        frame = self.JOG_FRAMES[self._jog_frame_idx]
        if frame == "Tool Frame":
            return T_world_tool[:3, :3] @ unit
        if frame == "Reference Frame":
            T_world_base = np.eye(4); T_world_base[:3, 3] = self._base_xyz
            T_world_ref = T_world_base @ self._ref_frames[self._ref_idx][1]
            return T_world_ref[:3, :3] @ unit
        return unit

    def _apply_cartesian_target(self, T_world_tool_target, source: str) -> None:
        T_flange_tool = self._tool_frames[self._tool_idx][1]
        T_world_tool0_target = T_world_tool_target @ np.linalg.inv(T_flange_tool)
        q_init = [math.radians(q) for q in self._joints]
        sol = inverse_kinematics_seeded(
            self._model, T_world_tool0_target, q_init,
            tol_mm=0.5, tol_rad=1e-3, max_iter=100)
        if sol is None:
            self._set_status(f"IK fail: {source}", level="err")
            return
        self._apply_joints_main([math.degrees(q) for q in sol])
        self._set_status(source, level="ok")

    # ══════════════════════════════════════════════════════════════════
    # Callbacks (run on main thread)
    # ══════════════════════════════════════════════════════════════════

    def _on_joint_slider(self, idx: int, value_deg: float) -> None:
        j = list(self._joints); j[idx] = value_deg
        self._apply_joints_main(j)
        self._set_status(f"J{idx+1} = {value_deg:+.2f} deg")

    def _on_home(self) -> None:
        threading.Thread(target=self._animate_to,
                          args=(list(self._home_joints),), daemon=True).start()
        self._set_status("Move: Home", level="ok")

    def _on_zero(self) -> None:
        threading.Thread(target=self._animate_to,
                          args=([0.0] * 6,), daemon=True).start()
        self._set_status("Move: Zero", level="ok")

    def _on_tool_changed(self, idx: int) -> None:
        self._tool_idx = int(idx); self._refresh_pose_readout()
        # Tool triad depends on tool offset → update if visible
        if "tool" in self._frame_actors:
            T = self._frame_world_matrix("tool")
            if T is not None:
                self._update_frame_triad("tool", T)
                self._plotter.render()
        # Workspace radius for "tool" mode also changes
        if self._ws_group.checkedId() == 3:                # tool
            self._on_workspace_changed("tool")

    def _on_ref_changed(self, idx: int) -> None:
        self._ref_idx = int(idx); self._refresh_pose_readout()
        if "ref" in self._frame_actors:
            T = self._frame_world_matrix("ref")
            if T is not None:
                self._update_frame_triad("ref", T)
                self._plotter.render()

    def _on_translate(self, axis_idx: int, sign: int) -> None:
        T_tool = self._current_tool_world()
        if T_tool is None: return
        axis_world = self._jog_axis_world(axis_idx, T_tool)
        T_target = T_tool.copy()
        T_target[:3, 3] = T_tool[:3, 3] + axis_world * (sign * self._jog_step_mm)
        self._apply_cartesian_target(T_target,
            f"T{'+' if sign>0 else '-'}{self.AXIS_NAMES[axis_idx]} "
            f"{self._jog_step_mm:.1f}mm")

    def _on_rotate(self, axis_idx: int, sign: int) -> None:
        T_tool = self._current_tool_world()
        if T_tool is None: return
        axis_world = self._jog_axis_world(axis_idx, T_tool)
        R_step = _rotation_about_axis_3x3(axis_world,
            sign * math.radians(self._jog_step_deg))
        T_target = T_tool.copy()
        T_target[:3, :3] = R_step @ T_tool[:3, :3]
        self._apply_cartesian_target(T_target,
            f"R{'+' if sign>0 else '-'}{self.AXIS_NAMES[axis_idx]} "
            f"{self._jog_step_deg:.1f} deg")

    def _toggle_gripper(self, close: bool) -> None:
        if close:
            self._grasp_nearest_object()
        else:
            self._release_object()
        self._render_scene_frame(self._joints)
        self._set_status(f"Gripper {'CLOSE' if close else 'OPEN'}", level="ok")

    def _on_reset_scene(self) -> None:
        self._grasped_name = None
        self._grasp_offset_m = None
        for name, obj in self._objects.items():
            obj["world_T"] = obj["initial_world_T"].copy()
        self._render_scene_frame(self._joints)
        self._set_status("Scene reset", level="ok")

    # ── Grasp helpers ─────────────────────────────────────────────────
    def _grasp_nearest_object(self) -> None:
        T_tool0 = self._tool0_world()
        if T_tool0 is None or not self._objects:
            return
        tool0_pos_m = T_tool0[:3, 3] / 1000.0
        best_name, best_d = None, float("inf")
        for name, obj in self._objects.items():
            d = float(np.linalg.norm(obj["world_T"][:3, 3] - tool0_pos_m))
            if d < best_d: best_name, best_d = name, d
        if best_name is None or best_d > 0.25: return
        T_tool_m = T_tool0.copy(); T_tool_m[:3, 3] = tool0_pos_m
        self._grasp_offset_m = np.linalg.inv(T_tool_m) @ self._objects[best_name]["world_T"]
        self._grasped_name = best_name

    def _release_object(self) -> None:
        self._grasped_name = None
        self._grasp_offset_m = None

    def _tool0_world(self) -> np.ndarray | None:
        try:
            q_rad = [math.radians(q) for q in self._joints]
            return dict(link_frames_urdf(self._model, q_rad)).get("link_tool0")
        except Exception:                                   # noqa: BLE001
            return None

    # ── Demo motion (worker thread) ───────────────────────────────────
    def _on_demo_toggle(self) -> None:
        if self._demo_thread is not None and self._demo_thread.is_alive():
            self._demo_stop.set()
            self._set_status("Demo: stopping...", level="warn"); return
        self._demo_stop.clear()
        self._set_status("Demo: running", level="ok")
        self._demo_thread = threading.Thread(
            target=self._demo_loop, daemon=True)
        self._demo_thread.start()

    def _demo_loop(self) -> None:
        poses = [
            list(self._home_joints),
            [30, -30, 30, 0, 30, 0],
            [-30, 20, -40, 90, -30, 45],
            [0, 50, -80, 0, 60, 0],
        ]
        i = 0
        try:
            while not self._demo_stop.is_set():
                self._animate_to(poses[i % len(poses)], steps=40, dt=0.025,
                                  stop_event=self._demo_stop)
                i += 1
                for _ in range(20):
                    if self._demo_stop.is_set(): break
                    time.sleep(0.05)
        finally:
            self._signals.demo_done.emit()

    def _on_demo_done(self) -> None:
        self._set_status("Demo: idle")

    # ══════════════════════════════════════════════════════════════════
    # Program panel actions
    # ══════════════════════════════════════════════════════════════════

    @property
    def _program(self) -> list[Instruction]:
        """Active job's instruction list. List-mutating ops (.append/.clear)
        modify in-place; assignment goes via setter to replace the underlying
        list trong self._jobs."""
        return self._jobs[self._active_job]

    @_program.setter
    def _program(self, value: list[Instruction]) -> None:
        self._jobs[self._active_job] = list(value)

    def _refresh_program_list(self) -> None:
        self._prog_list.clear()
        if not self._program:
            self._prog_list.addItem("(empty)"); return
        for i, ins in enumerate(self._program):
            self._prog_list.addItem(f"{i+1:>2}. {ins.describe()}")

    def _on_prog_add_movej(self) -> None:
        self._program.append(Instruction(type="MoveJ", joints=list(self._joints)))
        self._refresh_program_list()
        self._set_status(f"Program +MoveJ (n={len(self._program)})", level="ok")

    def _on_prog_add_movel(self) -> None:
        T = self._current_tool_world()
        if T is None: return
        x, y, z, rx, ry, rz = _matrix_to_xyz_rpy_deg(T)
        self._program.append(Instruction(type="MoveL", tcp_pose=[x, y, z, rx, ry, rz]))
        self._refresh_program_list()
        self._set_status(f"Program +MoveL (n={len(self._program)})", level="ok")

    def _on_prog_add_gripper(self, close: bool) -> None:
        self._program.append(Instruction(type="SetGripper", gripper_close=close))
        self._refresh_program_list()
        self._set_status(
            f"Program +Gripper {'CLOSE' if close else 'OPEN'} "
            f"(n={len(self._program)})", level="ok")

    def _on_prog_add_wait(self) -> None:
        secs = float(self._prog_wait_spin.value())
        self._program.append(Instruction(type="Wait", wait_seconds=secs))
        self._refresh_program_list()
        self._set_status(f"Program +Wait {secs:.2f}s", level="ok")

    # ── Tier-1 new instructions ───────────────────────────────────────
    def _on_prog_add_movec(self) -> None:
        """2-step: lần 1 click → lưu pose hiện tại làm MID; lần 2 → end +
        commit MoveC instruction."""
        T = self._current_tool_world()
        if T is None: return
        pose = list(_matrix_to_xyz_rpy_deg(T))
        if self._pending_movc_mid is None:
            self._pending_movc_mid = pose
            self._btn_movec.setText("+ MoveC (set END)")
            self._set_status(
                "MoveC: MID captured — di chuyển robot tới END pose rồi click lại",
                level="info")
            return
        # 2nd click → commit
        self._program.append(Instruction(
            type="MoveC",
            tcp_pose_mid=self._pending_movc_mid,
            tcp_pose=pose,
        ))
        self._pending_movc_mid = None
        self._btn_movec.setText("+ MoveC (set MID)")
        self._refresh_program_list()
        self._set_status(f"Program +MoveC (n={len(self._program)})", level="ok")

    def _on_prog_add_waitio(self) -> None:
        ins = Instruction(
            type="WaitIO",
            io_index=int(self._prog_io_idx.value()),
            io_state=(self._prog_io_state.currentText() == "ON"),
            io_timeout_s=float(self._prog_io_tout.value()),
        )
        self._program.append(ins)
        self._refresh_program_list()
        self._set_status(f"Program +{ins.describe()}", level="ok")

    def _on_prog_add_setspeed(self) -> None:
        self._program.append(Instruction(
            type="SetSpeed",
            speed_joint_pct=float(self._prog_spd_vj.value()),
            speed_linear_mm_s=float(self._prog_spd_v.value()),
        ))
        self._refresh_program_list()
        self._set_status(
            f"Program +SetSpeed VJ={self._prog_spd_vj.value():.1f}% "
            f"V={self._prog_spd_v.value():.0f}mm/s", level="ok")

    def _on_prog_add_setrounding(self) -> None:
        self._program.append(Instruction(
            type="SetRounding", rounding_pl=int(self._prog_pl.value())))
        self._refresh_program_list()
        self._set_status(
            f"Program +SetRounding PL={self._prog_pl.value()}", level="ok")

    def _on_prog_add_settool(self) -> None:
        self._program.append(Instruction(
            type="SetTool", tool_no=int(self._prog_tool_no.value())))
        self._refresh_program_list()
        self._set_status(
            f"Program +SetTool TL#{self._prog_tool_no.value()}", level="ok")

    def _on_prog_add_setrefframe(self) -> None:
        self._program.append(Instruction(
            type="SetRefFrame", ref_frame_no=int(self._prog_uf_no.value())))
        self._refresh_program_list()
        self._set_status(
            f"Program +SetRefFrame UF#{self._prog_uf_no.value()}", level="ok")

    def _on_prog_add_msg(self) -> None:
        text = self._prog_msg_edit.text().strip()
        if not text:
            self._set_status("MSG empty — nhập text trước", level="warn"); return
        self._program.append(Instruction(type="ShowMessage", message=text))
        self._refresh_program_list()
        self._prog_msg_edit.clear()
        self._set_status(f'Program +MSG "{text[:32]}"', level="ok")

    def _on_prog_modify(self) -> None:
        """F2 / double-click / Edit button — edit selected instruction.

        Hành vi per-type:
          • MoveJ/MoveL/MoveC (inline pose) → Replace with current pose (sau
            confirm). Target-referencing → dialog chọn target khác.
          • SetGripper → flip OPEN/CLOSE.
          • Wait / WaitIO / SetSpeed / SetRounding / SetTool / SetRefFrame /
            ShowMessage / CallJob → dialog với editable fields.
        """
        idx = self._prog_list.currentRow()
        if idx < 0 or idx >= len(self._program):
            self._set_status("Chọn instruction để Edit", level="warn"); return
        ins = self._program[idx]
        t = ins.type
        # ── Motion (inline vs target-ref) ─────────────────────────────
        if t in ("MoveJ", "MoveL"):
            if ins.target_name:
                # Target-ref → chọn target khác qua combo
                new_name = self._dlg_pick_target(ins.target_name)
                if new_name is None: return
                ins.target_name = new_name
            else:
                r = QMessageBox.question(
                    self, "Modify", f"Replace step {idx+1} pose with current pose?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if r != QMessageBox.StandardButton.Yes: return
                if t == "MoveJ":
                    ins.joints = list(self._joints)
                else:
                    T = self._current_tool_world()
                    if T is None: return
                    ins.tcp_pose = list(_matrix_to_xyz_rpy_deg(T))
        elif t == "MoveC":
            r = QMessageBox.question(
                self, "Modify MoveC",
                f"Replace step {idx+1} END với current pose? (MID giữ nguyên)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes: return
            T = self._current_tool_world()
            if T is None: return
            ins.tcp_pose = list(_matrix_to_xyz_rpy_deg(T))
        # ── Logic / Modal ─────────────────────────────────────────────
        elif t == "SetGripper":
            ins.gripper_close = not ins.gripper_close
        elif t == "Wait":
            v, ok = QInputDialog.getDouble(
                self, "Modify Wait", "Seconds:",
                ins.wait_seconds, 0.0, 600.0, 2)
            if not ok: return
            ins.wait_seconds = float(v)
        elif t == "WaitIO":
            new = self._dlg_edit_waitio(ins)
            if new is None: return
            ins.io_index, ins.io_state, ins.io_timeout_s = new
        elif t == "SetSpeed":
            new = self._dlg_edit_setspeed(ins)
            if new is None: return
            ins.speed_joint_pct, ins.speed_linear_mm_s = new
        elif t == "SetRounding":
            v, ok = QInputDialog.getInt(
                self, "Modify Rounding", "PL (0..8):", ins.rounding_pl, 0, 8)
            if not ok: return
            ins.rounding_pl = int(v)
        elif t == "SetTool":
            v, ok = QInputDialog.getInt(
                self, "Modify Tool", "TL# (0..15):", ins.tool_no, 0, 15)
            if not ok: return
            ins.tool_no = int(v)
        elif t == "SetRefFrame":
            v, ok = QInputDialog.getInt(
                self, "Modify Ref frame", "UF# (0..15):", ins.ref_frame_no, 0, 15)
            if not ok: return
            ins.ref_frame_no = int(v)
        elif t == "ShowMessage":
            v, ok = QInputDialog.getText(
                self, "Modify MSG", "Text (≤32 ASCII):",
                QLineEdit.EchoMode.Normal, ins.message)
            if not ok: return
            ins.message = str(v)[:32]
        elif t == "CallJob":
            v, ok = QInputDialog.getText(
                self, "Modify Call", "Sub-job name:",
                QLineEdit.EchoMode.Normal, ins.job_name)
            if not ok: return
            safe = "".join(c for c in v if c.isalnum() or c == "_")[:32].upper()
            if not safe:
                self._set_status("Job name không hợp lệ", level="warn"); return
            ins.job_name = safe
        elif t == "SimEvent":
            new = self._dlg_edit_simevent(ins)
            if new is None: return
            ins.event_name, ins.event_payload = new
        else:
            self._set_status(f"Edit not supported for type {t}", level="warn"); return
        self._refresh_program_list()
        self._prog_list.setCurrentRow(idx)
        self._set_status(f"Modified step {idx+1}: {ins.describe()}", level="ok")

    def _dlg_pick_target(self, current: str) -> str | None:
        """QDialog chọn target khác từ list. Return new name hoặc None nếu cancel."""
        if not self._targets:
            self._set_status("No targets defined", level="warn"); return None
        names = list(self._targets.keys())
        idx = names.index(current) if current in names else 0
        v, ok = QInputDialog.getItem(
            self, "Pick target", "Target:", names, idx, False)
        return v if ok else None

    def _dlg_edit_waitio(self, ins: Instruction):
        """Trả về (io_index, io_state_bool, timeout_s) hoặc None."""
        dlg = QDialog(self); dlg.setWindowTitle("Modify WaitIO")
        form = QFormLayout(dlg)
        sp_idx = QSpinBox(); sp_idx.setRange(1, 1024); sp_idx.setValue(ins.io_index)
        cb_state = QComboBox(); cb_state.addItems(["ON", "OFF"])
        cb_state.setCurrentIndex(0 if ins.io_state else 1)
        sp_t = QDoubleSpinBox(); sp_t.setRange(0.0, 600.0); sp_t.setSuffix(" s")
        sp_t.setValue(ins.io_timeout_s)
        form.addRow("IN#", sp_idx); form.addRow("State", cb_state); form.addRow("Timeout", sp_t)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted: return None
        return (int(sp_idx.value()), cb_state.currentText() == "ON", float(sp_t.value()))

    def _on_show_pp_settings(self) -> None:
        """Edit post-processor parameters (INFORM codegen tuning)."""
        dlg = QDialog(self); dlg.setWindowTitle("Post-processor settings")
        form = QFormLayout(dlg)
        sp_max = QDoubleSpinBox()
        sp_max.setRange(1.0, 100.0); sp_max.setSuffix(" %")
        sp_max.setValue(self._pp_max_speed_pct)
        sp_max.setToolTip("Cap VJ safety. Yaskawa convention ≤ 30%.")
        sp_vj = QDoubleSpinBox()
        sp_vj.setRange(1.0, 30.0); sp_vj.setSuffix(" %")
        sp_vj.setValue(self._pp_default_vj)
        sp_v = QDoubleSpinBox()
        sp_v.setRange(1.0, 250.0); sp_v.setSuffix(" mm/s")
        sp_v.setValue(self._pp_default_v_mms)
        form.addRow("Max VJ cap", sp_max)
        form.addRow("Initial VJ", sp_vj)
        form.addRow("Initial V (linear)", sp_v)
        info = QLabel(
            "<small><i>Initial speeds áp dụng cho MOVJ/MOVL trước khi user "
            "đặt SetSpeed. Max VJ giới hạn an toàn cho mọi move.</i></small>")
        info.setWordWrap(True)
        form.addRow(info)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted: return
        self._pp_max_speed_pct = float(sp_max.value())
        self._pp_default_vj = float(sp_vj.value())
        self._pp_default_v_mms = float(sp_v.value())
        self._set_status(
            f"Post-processor: max={self._pp_max_speed_pct:.0f}%, "
            f"VJ₀={self._pp_default_vj:.0f}%, V₀={self._pp_default_v_mms:.0f}mm/s",
            level="ok")

    # ══════════════════════════════════════════════════════════════════
    # Robot connection (HSE) — Run on Robot pipeline
    # ══════════════════════════════════════════════════════════════════
    def _on_show_connection_settings(self) -> None:
        """Dialog edit HSE connection — IP, tool_no, FTP creds."""
        dlg = QDialog(self); dlg.setWindowTitle("Robot connection (HSE)")
        form = QFormLayout(dlg)
        ed_ip = QLineEdit(self._hse_ip)
        ed_ip.setPlaceholderText("e.g. 192.168.1.100")
        sp_tool = QSpinBox(); sp_tool.setRange(0, 63); sp_tool.setValue(self._hse_tool_no)
        ed_user = QLineEdit(self._hse_ftp_user)
        ed_user.setPlaceholderText("empty = anonymous")
        ed_pass = QLineEdit(self._hse_ftp_pass)
        ed_pass.setEchoMode(QLineEdit.EchoMode.Password)
        ed_dir = QLineEdit(self._hse_ftp_dir)
        form.addRow("HSE IP", ed_ip)
        form.addRow("Tool # (TL=)", sp_tool)
        form.addRow("FTP user", ed_user)
        form.addRow("FTP pass", ed_pass)
        form.addRow("FTP job dir", ed_dir)
        info = QLabel(
            "<small><i>⚠ Robot phải ở REMOTE mode + HSE Server function enabled."
            "<br>Speed slider trên TP nên ≤ 10% lần đầu.</i></small>")
        info.setWordWrap(True); form.addRow(info)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted: return
        self._hse_ip = ed_ip.text().strip()
        self._hse_tool_no = int(sp_tool.value())
        self._hse_ftp_user = ed_user.text()
        self._hse_ftp_pass = ed_pass.text()
        self._hse_ftp_dir = ed_dir.text().strip() or "/MPRAM1/JBI"
        self._set_status(
            f"Connection: {self._hse_ip} TL#{self._hse_tool_no}", level="ok")

    def _on_test_connection(self) -> None:
        """Ping HSE — gửi READ_STATUS để verify socket + controller alive."""
        if not self._hse_ip:
            self._set_status(
                "Chưa cấu hình HSE IP — Robot → Connection settings", level="warn")
            return
        self._set_status(f"Pinging {self._hse_ip}…", level="info")
        QApplication.processEvents()
        backend = MotomanHSEBackend(
            ip=self._hse_ip, timeout_s=2.0,
            ftp_user=self._hse_ftp_user, ftp_pass=self._hse_ftp_pass,
            ftp_job_dir=self._hse_ftp_dir, tool_no=self._hse_tool_no)
        try:
            backend.connect()
            ok = backend.Valid()
            if ok:
                # Đọc joints + alarm để verify deeper
                try:
                    joints = backend.Joints()
                    alarm_code, _ = backend.read_alarm()
                    alarm_str = ("✓ no alarm" if alarm_code == 0
                                 else f"⚠ alarm 0x{alarm_code:04X}")
                    msg = (f"Connected. Joints: ["
                           + ", ".join(f"{j:+.1f}°" for j in joints) +
                           f"]  {alarm_str}")
                    self._set_status(msg, level="ok")
                    QMessageBox.information(self, "Connection OK", msg)
                except Exception as e:                      # noqa: BLE001
                    self._set_status(
                        f"Connected but deep probe fail: {e}", level="warn")
            else:
                self._set_status(
                    f"Connection FAIL — kiểm tra HSE Server enable",
                    level="err")
                QMessageBox.warning(
                    self, "Connection failed",
                    f"YRC1000 {self._hse_ip} không phản hồi READ_STATUS.\n"
                    "Verify:\n"
                    " • Ping {ip} OK?\n"
                    " • HSE Server function enabled trong Maintenance mode?\n"
                    " • PC cùng subnet với YRC1000?".format(ip=self._hse_ip))
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"Connection error: {e}", level="err")
            QMessageBox.critical(self, "Connection error", str(e))
        finally:
            backend.disconnect()

    def _on_run_on_robot(self) -> None:
        """Render current job → upload .JBI → JOB_SELECT + START → wait_idle.

        Chạy trong worker thread để UI không block. Stop button → servo OFF.
        """
        if not self._hse_ip:
            r = QMessageBox.question(
                self, "Run on Robot",
                "Chưa cấu hình HSE IP. Mở Connection settings bây giờ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r == QMessageBox.StandardButton.Yes:
                self._on_show_connection_settings()
            return
        if not self._program:
            self._set_status("Current job empty", level="warn"); return
        if self._hse_thread is not None and self._hse_thread.is_alive():
            self._set_status("Robot đang chạy job — wait done hoặc Stop",
                              level="warn"); return
        # Safety confirm
        n_steps = len(self._program)
        r = QMessageBox.warning(
            self, "Run on Robot — Safety check",
            f"<b>⚠ ROBOT SẼ CHUYỂN ĐỘNG THẬT</b><br><br>"
            f"Job: <code>{self._active_job}</code> ({n_steps} instructions)<br>"
            f"HSE IP: <code>{self._hse_ip}</code><br>"
            f"Max VJ: {self._pp_max_speed_pct:.0f}%<br><br>"
            f"Trước khi tiếp tục, verify:<br>"
            f"&nbsp;✓ YRC1000 ở REMOTE mode<br>"
            f"&nbsp;✓ Speed slider TP ≤ 10%<br>"
            f"&nbsp;✓ Workspace clear, tay sẵn sàng E-stop<br>"
            f"&nbsp;✓ Không có alarm active<br><br>"
            f"Tiếp tục?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if r != QMessageBox.StandardButton.Yes: return
        # Render JBI trong main thread (cần access self._targets atomically),
        # rồi pass text + name vào worker.
        try:
            stem = self._safe_job_name(self._active_job) or "PROG"
            jbi_path = Path.cwd() / f"{stem}.JBI"
            self._export_job_to_path(self._program, stem, jbi_path)
            jbi_text = jbi_path.read_text(encoding="utf-8")
            jbi_path.unlink(missing_ok=True)                # tmp file
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"JBI render fail: {e}", level="err")
            return
        # Worker thread chạy upload + JOB_SELECT + START + wait_idle
        self._hse_stop.clear()
        self._hse_thread = threading.Thread(
            target=self._run_on_robot_worker,
            args=(jbi_text, stem),
            daemon=True)
        self._hse_thread.start()
        self._set_status(f"Robot: uploading job '{stem}'…", level="info")

    def _run_on_robot_worker(self, jbi_text: str, job_name: str) -> None:
        backend = MotomanHSEBackend(
            ip=self._hse_ip, timeout_s=3.0,
            ftp_user=self._hse_ftp_user, ftp_pass=self._hse_ftp_pass,
            ftp_job_dir=self._hse_ftp_dir, tool_no=self._hse_tool_no,
            max_speed_pct=self._pp_max_speed_pct,
            wait_completion_timeout_s=120.0)
        try:
            backend.connect()
            if not backend.Valid():
                self._signals.status.emit(
                    "Robot: HSE not responding — abort", "err"); return
            # Alarm pre-check
            code, sub = backend.read_alarm()
            if code != 0:
                self._signals.status.emit(
                    f"Robot: ALARM 0x{code:04X} (sub 0x{sub:04X}) — reset TP trước",
                    "err"); return
            if self._hse_stop.is_set():
                self._signals.status.emit("Robot: aborted before upload", "warn"); return
            self._signals.status.emit(f"Robot: FTP uploading '{job_name}.JBI'…", "info")
            backend.upload_job(jbi_text, job_name)
            if self._hse_stop.is_set():
                self._signals.status.emit("Robot: aborted before start", "warn"); return
            self._signals.status.emit(f"Robot: JOB_SELECT + START '{job_name}'…", "info")
            backend.job_select(job_name)
            backend.job_start()
            # Poll status until idle hoặc stop
            import time as _time
            t_start = _time.monotonic()
            poll_dt = 0.3
            timeout = backend.wait_completion_timeout_s
            while True:
                if self._hse_stop.is_set():
                    backend.Stop()                          # servo off
                    self._signals.status.emit(
                        "Robot: STOP triggered — servo OFF", "warn"); return
                try:
                    running = backend.read_status_running()
                except Exception as e:                      # noqa: BLE001
                    self._signals.status.emit(
                        f"Robot: status poll error: {e}", "warn"); break
                if not running: break
                if _time.monotonic() - t_start > timeout:
                    self._signals.status.emit(
                        f"Robot: timeout {timeout:.0f}s — check TP", "err"); return
                _time.sleep(poll_dt)
            # Done — alarm post-check
            code, sub = backend.read_alarm()
            if code != 0:
                self._signals.status.emit(
                    f"Robot: completed WITH ALARM 0x{code:04X}", "warn")
            else:
                self._signals.status.emit(
                    f"Robot: job '{job_name}' completed OK", "ok")
        except Exception as e:                              # noqa: BLE001
            self._signals.status.emit(f"Robot error: {e}", "err")
        finally:
            try:
                backend.disconnect()
            except Exception:                               # noqa: BLE001
                pass

    def _on_stop_all(self) -> None:
        """Dual-purpose stop: sim playback + robot job (servo OFF nếu đang HSE)."""
        # Sim stop
        self._on_prog_stop()
        # Robot stop
        if self._hse_thread is not None and self._hse_thread.is_alive():
            self._hse_stop.set()
            self._set_status("Robot: STOP signaled (will servo-off)", level="warn")

    def _on_show_script_editor(self) -> None:
        """Mở Python script editor — user nhập code dùng `p.add_*()` API
        để generate instructions programmatically. Run → append vào current job.
        """
        dlg = QDialog(self); dlg.setWindowTitle("Generate from Python script")
        dlg.resize(720, 520)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(
            "<b>Python script</b> — dùng <code>p.add_*()</code> để thêm "
            "instruction vào job hiện tại. Helpers: <code>math</code>, "
            "<code>np</code>, target dict <code>p.targets</code>."))
        editor = QPlainTextEdit()
        editor.setPlaceholderText(
            "# Example: tạo 8 MoveJ điểm vòng tròn quanh Z=500\n"
            "import math\n"
            "for i in range(8):\n"
            "    angle = i * 2 * math.pi / 8\n"
            "    x = 500 + 200 * math.cos(angle)\n"
            "    y = 0   + 200 * math.sin(angle)\n"
            "    # Cần solve IK riêng, hoặc dùng target có sẵn:\n"
            "    p.add_movej_to('HOME')")
        from PyQt6.QtGui import QFont
        mono = QFont("Consolas"); mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(10)
        editor.setFont(mono)
        # Restore previous script nếu có
        editor.setPlainText(getattr(self, "_last_script_text", ""))
        lay.addWidget(editor, 1)
        result_lbl = QLabel("")
        result_lbl.setWordWrap(True)
        lay.addWidget(result_lbl)
        # Buttons
        bb = QDialogButtonBox()
        b_run = bb.addButton("▶ Run", QDialogButtonBox.ButtonRole.ActionRole)
        b_close = bb.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        b_close.clicked.connect(dlg.reject)
        lay.addWidget(bb)

        def _run():
            self._last_script_text = editor.toPlainText()
            n_before = len(self._program)
            api = _ScriptProgramAPI(self)
            globs = {
                "__builtins__": {
                    k: getattr(__builtins__, k) if hasattr(__builtins__, k)
                    else __builtins__[k]
                    for k in ("range", "len", "enumerate", "zip", "min", "max",
                              "int", "float", "list", "tuple", "dict", "set",
                              "sum", "abs", "round", "print", "bool", "str",
                              "isinstance", "type")
                },
            }
            try:
                exec(editor.toPlainText(), globs,
                      {"p": api, "math": math, "np": np})
                added = len(self._program) - n_before
                result_lbl.setText(
                    f"<span style='color:#0a7d2c'><b>OK</b> — added {added} "
                    f"instruction(s) to job '{self._active_job}'</span>")
                self._refresh_program_list()
                self._set_status(
                    f"Script: +{added} instructions vào '{self._active_job}'",
                    level="ok")
            except Exception as e:                          # noqa: BLE001
                result_lbl.setText(
                    f"<span style='color:#cf222e'><b>Error:</b> "
                    f"{type(e).__name__}: {e}</span>")
        b_run.clicked.connect(_run)
        dlg.exec()

    def _dlg_edit_simevent(self, ins: Instruction):
        """Trả về (name, payload) hoặc None."""
        dlg = QDialog(self); dlg.setWindowTitle("Modify SimEvent")
        form = QFormLayout(dlg)
        ed_name = QLineEdit(ins.event_name); ed_name.setMaxLength(32)
        ed_pl = QLineEdit(ins.event_payload); ed_pl.setMaxLength(80)
        form.addRow("Name", ed_name); form.addRow("Payload", ed_pl)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted: return None
        return (ed_name.text().strip()[:32], ed_pl.text().strip()[:80])

    def _dlg_edit_setspeed(self, ins: Instruction):
        """Trả về (vj_pct, v_mm_s) hoặc None."""
        dlg = QDialog(self); dlg.setWindowTitle("Modify SetSpeed")
        form = QFormLayout(dlg)
        sp_vj = QDoubleSpinBox(); sp_vj.setRange(1.0, 30.0); sp_vj.setSuffix(" %")
        sp_vj.setValue(ins.speed_joint_pct)
        sp_v = QDoubleSpinBox(); sp_v.setRange(1.0, 250.0); sp_v.setSuffix(" mm/s")
        sp_v.setValue(ins.speed_linear_mm_s)
        form.addRow("VJ (joint)", sp_vj); form.addRow("V (linear)", sp_v)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted: return None
        return (float(sp_vj.value()), float(sp_v.value()))

    def _on_prog_add_calljob(self) -> None:
        raw = self._prog_call_edit.text().strip()
        if not raw:
            self._set_status("Call job name empty", level="warn"); return
        # Sanitize: uppercase, alphanumeric + _ (INFORM job name rules)
        safe = "".join(c for c in raw if c.isalnum() or c == "_")[:32].upper()
        if not safe:
            self._set_status("Job name không hợp lệ", level="warn"); return
        self._program.append(Instruction(type="CallJob", job_name=safe))
        self._refresh_program_list()
        self._prog_call_edit.clear()
        self._set_status(f"Program +Call JOB:{safe}", level="ok")

    def _on_prog_add_simevent(self) -> None:
        name = self._prog_ev_edit.text().strip()
        if not name:
            self._set_status("SimEvent name empty", level="warn"); return
        self._program.append(Instruction(type="SimEvent", event_name=name[:32]))
        self._refresh_program_list()
        self._prog_ev_edit.clear()
        self._set_status(f"Program +SimEvent '{name[:32]}'", level="ok")

    # ══════════════════════════════════════════════════════════════════
    # Multi-job project — job selector
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _safe_job_name(name: str) -> str:
        """Sanitize → uppercase alphanumeric/_  (INFORM job name rules)."""
        s = "".join(c for c in name.strip() if c.isalnum() or c == "_")
        return s.upper()[:32]

    def _refresh_job_combo(self) -> None:
        self._job_combo.blockSignals(True)
        self._job_combo.clear()
        for name in self._jobs.keys():
            self._job_combo.addItem(name)
        idx = self._job_combo.findText(self._active_job)
        if idx >= 0: self._job_combo.setCurrentIndex(idx)
        self._job_combo.blockSignals(False)

    def _on_job_changed(self, name: str) -> None:
        if not name or name not in self._jobs: return
        self._active_job = name
        self._refresh_program_list()
        self._set_status(
            f"Active job: {name}  ({len(self._program)} steps)", level="info")

    def _on_job_add(self) -> None:
        v, ok = QInputDialog.getText(
            self, "Add job", "New job name:", QLineEdit.EchoMode.Normal, "SUB1")
        if not ok: return
        name = self._safe_job_name(v)
        if not name:
            self._set_status("Job name không hợp lệ", level="warn"); return
        if name in self._jobs:
            self._set_status(f"Job '{name}' đã tồn tại", level="warn"); return
        self._jobs[name] = []
        self._active_job = name
        self._refresh_job_combo()
        self._refresh_program_list()
        self._set_status(f"Added job '{name}'", level="ok")

    def _on_job_rename(self) -> None:
        old = self._active_job
        v, ok = QInputDialog.getText(
            self, "Rename job", f"New name for '{old}':",
            QLineEdit.EchoMode.Normal, old)
        if not ok: return
        new = self._safe_job_name(v)
        if not new or new == old: return
        if new in self._jobs:
            self._set_status(f"Job '{new}' đã tồn tại", level="warn"); return
        # Preserve dict order
        new_jobs: dict[str, list[Instruction]] = {}
        for k, v_list in self._jobs.items():
            new_jobs[new if k == old else k] = v_list
        self._jobs = new_jobs
        self._active_job = new
        # Update CallJob refs nếu có (trong tất cả jobs)
        for job_prog in self._jobs.values():
            for ins in job_prog:
                if ins.type == "CallJob" and ins.job_name == old:
                    ins.job_name = new
        self._refresh_job_combo()
        self._refresh_program_list()
        self._set_status(f"Renamed '{old}' → '{new}'", level="ok")

    def _on_job_delete(self) -> None:
        if len(self._jobs) <= 1:
            self._set_status("Phải có ít nhất 1 job", level="warn"); return
        old = self._active_job
        # Check CallJob refs trong các job khác
        refs = sum(
            1
            for k, prog in self._jobs.items() if k != old
            for ins in prog if ins.type == "CallJob" and ins.job_name == old)
        msg = f"Delete job '{old}' ({len(self._program)} steps)?"
        if refs > 0:
            msg += f"\n\nCảnh báo: {refs} CallJob instruction(s) ở job khác đang reference."
        r = QMessageBox.question(
            self, "Delete job", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if r != QMessageBox.StandardButton.Yes: return
        del self._jobs[old]
        self._active_job = next(iter(self._jobs.keys()))
        self._refresh_job_combo()
        self._refresh_program_list()
        self._set_status(f"Deleted '{old}', switched to '{self._active_job}'",
                          level="ok")

    # ══════════════════════════════════════════════════════════════════
    # Target library (RoboDK-style Teach Target)
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _safe_target_name(name: str) -> str:
        """Sanitize → uppercase, alphanumeric + _ (INFORM C-var safe)."""
        s = "".join(c if (c.isalnum() or c == "_") else "_" for c in name.strip())
        return s.upper()[:24]

    def _refresh_target_list(self) -> None:
        self._tgt_list.clear()
        if not self._targets:
            self._tgt_list.addItem("(no targets)"); return
        for name, pose in self._targets.items():
            j = pose["joints"]
            self._tgt_list.addItem(
                f"{name}  [{', '.join(f'{q:+5.0f}' for q in j)}]")

    def _capture_current_pose(self) -> dict | None:
        """Snapshot current joints + tcp_pose từ robot state."""
        T = self._current_tool_world()
        if T is None: return None
        x, y, z, rx, ry, rz = _matrix_to_xyz_rpy_deg(T)
        return {
            "joints": list(self._joints),
            "tcp_pose": [x, y, z, rx, ry, rz],
        }

    def _on_tgt_teach(self) -> None:
        name_raw = self._tgt_name_edit.text().strip()
        if not name_raw:
            self._set_status("Target name empty", level="warn"); return
        name = self._safe_target_name(name_raw)
        if not name:
            self._set_status("Target name không hợp lệ", level="warn"); return
        if name in self._targets:
            self._set_status(
                f"Target '{name}' đã tồn tại — dùng Modify (F3) để cập nhật",
                level="warn"); return
        pose = self._capture_current_pose()
        if pose is None: return
        self._targets[name] = pose
        self._refresh_target_list()
        self._tgt_name_edit.clear()
        self._set_status(f"Target '{name}' taught (n={len(self._targets)})",
                          level="ok")

    def _on_tgt_modify(self) -> None:
        """F3 — replace selected target's pose với current pose."""
        idx = self._tgt_list.currentRow()
        if idx < 0 or not self._targets:
            self._set_status("Chọn target để Modify", level="warn"); return
        name = list(self._targets.keys())[idx]
        pose = self._capture_current_pose()
        if pose is None: return
        self._targets[name] = pose
        self._refresh_target_list()
        self._tgt_list.setCurrentRow(idx)
        self._set_status(f"Target '{name}' modified", level="ok")

    def _on_tgt_delete(self) -> None:
        idx = self._tgt_list.currentRow()
        if idx < 0 or not self._targets: return
        name = list(self._targets.keys())[idx]
        refs = sum(
            1 for ins in self._program
            if ins.type in ("MoveJ", "MoveL") and ins.target_name == name)
        if refs > 0:
            r = QMessageBox.question(
                self, "Delete target",
                f"Target '{name}' đang được {refs} instruction(s) tham chiếu.\n"
                "Xoá sẽ làm các move đó không resolve được khi play/export.\n\n"
                "Tiếp tục?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes: return
        del self._targets[name]
        self._refresh_target_list()
        self._set_status(f"Target '{name}' deleted", level="ok")

    def _on_prog_add_move_to_target(self, kind: str) -> None:
        """kind ∈ {'MoveJ','MoveL'}. Thêm instruction tham chiếu target đang
        select trong list."""
        idx = self._tgt_list.currentRow()
        if idx < 0 or not self._targets:
            self._set_status(
                "Chọn target trong list trước (hoặc Teach mới)", level="warn"); return
        name = list(self._targets.keys())[idx]
        self._program.append(Instruction(type=kind, target_name=name))
        self._refresh_program_list()
        self._set_status(f"Program +{kind} → {name}", level="ok")

    # ══════════════════════════════════════════════════════════════════
    # Teach on Surface (Ctrl+Shift+T) — click 3D scene → create target
    # ══════════════════════════════════════════════════════════════════
    def _on_toggle_surface_pick(self, enabled: bool) -> None:
        self._surface_pick_mode = bool(enabled)
        if enabled:
            try:
                self._plotter.enable_surface_point_picking(
                    callback=self._on_surface_pick,
                    show_message=False,
                    show_point=True,
                    point_size=14,
                    color="magenta",
                    left_clicking=True,
                )
                self._set_status(
                    "Teach on surface ON — click vào cell/floor để tạo target",
                    level="info")
            except Exception as e:                          # noqa: BLE001
                self._set_status(f"Surface pick fail: {e}", level="err")
                self._surface_pick_mode = False
                self._set_toggle(self._act_surface_pick, False)
        else:
            try:
                self._plotter.disable_picking()
            except Exception:                               # noqa: BLE001
                pass
            self._set_status("Teach on surface OFF", level="ok")

    def _on_surface_pick(self, picked_point) -> None:
        """Callback từ pyvista picker. picked_point in METERS (pyvista internal)."""
        if picked_point is None: return
        pt_m = np.asarray(picked_point, dtype=float)
        pt_mm = pt_m * 1000.0
        # Lấy surface normal tại pick point từ picked actor (nếu có).
        normal = self._surface_normal_at(pt_m)
        if normal is None:
            normal = np.array([0.0, 0.0, 1.0])              # fallback: assume +Z (floor)
        # Build TCP target: Z_tcp = -normal (tool point INTO surface),
        # X_tcp = world X projected lên plane vuông góc Z_tcp.
        z_tcp = -normal / max(1e-9, np.linalg.norm(normal))
        ref_x = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(ref_x, z_tcp))) > 0.99:
            ref_x = np.array([0.0, 1.0, 0.0])
        x_tcp = ref_x - float(np.dot(ref_x, z_tcp)) * z_tcp
        x_tcp = x_tcp / max(1e-9, np.linalg.norm(x_tcp))
        y_tcp = np.cross(z_tcp, x_tcp)
        T = np.eye(4)
        T[:3, 0] = x_tcp; T[:3, 1] = y_tcp; T[:3, 2] = z_tcp
        T[:3, 3] = pt_mm
        # Solve IK
        T_flange_tool = self._tool_frames[self._tool_idx][1]
        T_target_tool0 = T @ np.linalg.inv(T_flange_tool)
        q_init = [math.radians(q) for q in self._joints]
        sol = inverse_kinematics_seeded(
            self._model, T_target_tool0, q_init,
            tol_mm=0.5, tol_rad=1e-3, max_iter=100)
        if sol is None:
            self._set_status(
                f"IK fail tại ({pt_mm[0]:.0f},{pt_mm[1]:.0f},{pt_mm[2]:.0f})mm — "
                "ngoài tầm với", level="err")
            return
        # Prompt name + save
        default_name = f"SURF_{len(self._targets)+1:02d}"
        v, ok = QInputDialog.getText(
            self, "Teach on surface",
            f"Target name (pick @ XYZ = "
            f"{pt_mm[0]:.0f}, {pt_mm[1]:.0f}, {pt_mm[2]:.0f} mm):",
            QLineEdit.EchoMode.Normal, default_name)
        if not ok or not v.strip(): return
        name = self._safe_target_name(v)
        if not name:
            self._set_status("Name không hợp lệ", level="warn"); return
        if name in self._targets:
            self._set_status(
                f"Target '{name}' đã tồn tại — Modify (F3) để update",
                level="warn"); return
        joints_deg = [math.degrees(q) for q in sol]
        tcp_pose = list(_matrix_to_xyz_rpy_deg(T))
        self._targets[name] = {"joints": joints_deg, "tcp_pose": tcp_pose}
        self._refresh_target_list()
        self._set_status(f"Taught '{name}' on surface", level="ok")

    def _surface_normal_at(self, point_m: np.ndarray) -> np.ndarray | None:
        """Trả về world-frame surface normal tại 3D point.

        Optimization: cache `mesh + cell_normals` per dataset id. First call cho
        mỗi mesh chạy `compute_normals()` (O(N) cells, có thể vài ms cho mesh
        lớn). Subsequent calls cùng mesh → chỉ `find_closest_cell` + index lookup,
        ~100µs.
        """
        try:
            actor = getattr(self._plotter, "picked_actor", None)
            if actor is None: return None
            mapper = actor.GetMapper()
            if mapper is None: return None
            dataset = mapper.GetInputAsDataSet()
            if dataset is None: return None
            # Lookup cache by dataset id (static meshes — id stable).
            key = id(dataset)
            cached = self._normal_cache.get(key)
            if cached is not None:
                mesh, arr = cached
            else:
                mesh = pv.wrap(dataset)
                normals = mesh.compute_normals(
                    cell_normals=True, point_normals=False,
                    consistent_normals=True, auto_orient_normals=True,
                    inplace=False)
                arr = normals.cell_data.get("Normals", None)
                if arr is None: return None
                arr = np.asarray(arr, dtype=float)
                self._normal_cache[key] = (mesh, arr)
            # Transform point to actor local frame nếu có UserMatrix
            user_mat = actor.GetUserMatrix()
            if user_mat is not None:
                inv = vtk.vtkMatrix4x4()
                vtk.vtkMatrix4x4.Invert(user_mat, inv)
                out = [0.0, 0.0, 0.0, 1.0]
                inv.MultiplyPoint(
                    [float(point_m[0]), float(point_m[1]),
                     float(point_m[2]), 1.0], out)
                pt_local = out[:3]
            else:
                pt_local = [float(point_m[0]), float(point_m[1]), float(point_m[2])]
            cell_id = mesh.find_closest_cell(pt_local)
            if cell_id is None or cell_id < 0:
                return None
            if int(cell_id) >= len(arr): return None
            n_local = arr[int(cell_id)]
            # Transform normal back to world frame (rotation part only)
            if user_mat is not None:
                R = np.array([[user_mat.GetElement(i, j) for j in range(3)]
                              for i in range(3)])
                n_world = R @ n_local
            else:
                n_world = n_local
            n = float(np.linalg.norm(n_world))
            if n < 1e-9: return None
            return n_world / n
        except Exception as e:                              # noqa: BLE001
            logger.debug("surface_normal_at fail: %s", e)
            return None

    def _enumerate_ik_solutions(
        self, T_target_tool0: np.ndarray,
        max_solutions: int = 8, n_seeds: int = 30,
        dedupe_deg: float = 5.0,
    ) -> list[list[float]]:
        """Tìm IK solutions qua **BATCHED** IK — 1 numpy pipeline cho N=30 seeds
        thay vì N Python loops.

        Threading thử trước → chậm hơn 16% (GIL contention). Batched numpy
        operations là true vectorization: 1 outer iter loop chạy cho cả batch,
        mỗi step là single np.linalg.solve trên (N, 6, 6) stacked — BLAS handle
        toàn bộ trong C code.

        Benchmark: N=30 sequential ~290ms → batched ~5-15ms (~20-50× speedup).
        """
        link_attr = getattr(self._model, "joints", None) or getattr(
            self._model, "links", None)
        q_min = np.array([j.joint_min for j in link_attr])
        q_max = np.array([j.joint_max for j in link_attr])
        rng = np.random.RandomState(42)
        # Pre-generate (N, 6) seed batch
        q_init_batch = np.array([rng.uniform(q_min, q_max) for _ in range(n_seeds)])

        # Single batched call — all N IK problems song song
        results = inverse_kinematics_batch(
            self._model, T_target_tool0, q_init_batch,
            max_iter=100, tol_mm=0.5, tol_rad=1e-3)

        # Dedupe theo joint distance (5° threshold)
        thresh = np.deg2rad(dedupe_deg)
        solutions_rad: list[np.ndarray] = []
        for sol in results:
            if sol is None: continue
            sol_arr = np.asarray(sol)
            if any(np.max(np.abs(sol_arr - ex)) < thresh for ex in solutions_rad):
                continue
            solutions_rad.append(sol_arr)
            if len(solutions_rad) >= max_solutions:
                break
        return [[math.degrees(q) for q in s.tolist()] for s in solutions_rad]

    def _on_tgt_change_config(self) -> None:
        """F4 — enumerate IK solutions for selected target's TCP pose, let user
        pick alternative joint configuration."""
        idx = self._tgt_list.currentRow()
        if idx < 0 or not self._targets:
            self._set_status("Chọn target để Change Config", level="warn"); return
        name = list(self._targets.keys())[idx]
        tgt = self._targets[name]
        # Need TCP pose to enumerate. Compute T_target_tool0 từ stored tcp_pose
        T_target = _xyz_rpy_to_matrix(*tgt["tcp_pose"])
        T_flange_tool = self._tool_frames[self._tool_idx][1]
        T_target_tool0 = T_target @ np.linalg.inv(T_flange_tool)
        self._set_status(f"Enumerating IK solutions for '{name}'…", level="info")
        QApplication.processEvents()
        sols = self._enumerate_ik_solutions(T_target_tool0)
        if not sols:
            self._set_status(f"No IK solutions found for '{name}'", level="err"); return
        # Dialog: list of solutions, user picks one
        dlg = QDialog(self); dlg.setWindowTitle(f"Change Config — {name}")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel(
            f"<b>{len(sols)} IK solution(s)</b> for TCP "
            f"({tgt['tcp_pose'][0]:.0f}, {tgt['tcp_pose'][1]:.0f}, "
            f"{tgt['tcp_pose'][2]:.0f}) mm:"))
        lw = QListWidget(); lw.setMinimumHeight(180)
        current_jr = np.asarray(tgt["joints"])
        best_idx = 0; best_dist = float("inf")
        for i, s in enumerate(sols):
            arr = np.asarray(s)
            d = float(np.max(np.abs(arr - current_jr)))
            mark = "★" if d < 0.5 else " "
            if d < best_dist: best_dist, best_idx = d, i
            lw.addItem(
                f"{mark} #{i+1}: [" +
                ", ".join(f"{q:+6.1f}" for q in s) +
                f"]  Δ={d:.1f}°")
        lw.setCurrentRow(best_idx)
        layout.addWidget(lw)
        layout.addWidget(QLabel(
            "<small><i>★ = config hiện tại (Δ ≈ 0°). Δ = max joint difference "
            "so với current. Pick & OK để swap.</i></small>"))
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        layout.addWidget(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted: return
        pick = lw.currentRow()
        if pick < 0 or pick >= len(sols): return
        self._targets[name]["joints"] = list(sols[pick])
        self._refresh_target_list()
        self._tgt_list.setCurrentRow(idx)
        self._set_status(
            f"Target '{name}' config #{pick+1} applied", level="ok")

    def _on_tgt_goto(self) -> None:
        """Preview: animate robot to selected target's joints. KHÔNG thêm
        instruction — chỉ jog robot để verify pose hợp lệ."""
        idx = self._tgt_list.currentRow()
        if idx < 0 or not self._targets:
            self._set_status("Chọn target để Go to", level="warn"); return
        if self._prog_thread is not None and self._prog_thread.is_alive():
            self._set_status("Program đang chạy — không thể Go to", level="warn"); return
        name = list(self._targets.keys())[idx]
        target_joints = list(self._targets[name]["joints"])
        # Chạy animation trong worker thread (như Play, nhưng 1-step).
        self._set_status(f"Going to '{name}'…", level="info")
        def _worker():
            self._animate_to(target_joints, steps=40, dt=0.025)
            self._signals.status.emit(f"Reached '{name}'", "ok")
        threading.Thread(target=_worker, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════
    # Pause / Resume
    # ══════════════════════════════════════════════════════════════════
    def _on_prog_toggle_pause(self) -> None:
        if self._btn_pause.isChecked():
            self._prog_pause.set()
            self._btn_pause.setText("Resume")
            self._set_status("Program: paused", level="warn")
        else:
            self._prog_pause.clear()
            self._btn_pause.setText("Pause")
            self._set_status("Program: resumed", level="ok")

    def _on_prog_delete(self) -> None:
        idx = self._prog_list.currentRow()
        if 0 <= idx < len(self._program):
            del self._program[idx]
            self._refresh_program_list()

    def _on_prog_move_up(self) -> None:
        idx = self._prog_list.currentRow()
        if 0 < idx < len(self._program):
            self._program[idx-1], self._program[idx] = \
                self._program[idx], self._program[idx-1]
            self._refresh_program_list()
            self._prog_list.setCurrentRow(idx - 1)

    def _on_prog_move_down(self) -> None:
        idx = self._prog_list.currentRow()
        if 0 <= idx < len(self._program) - 1:
            self._program[idx+1], self._program[idx] = \
                self._program[idx], self._program[idx+1]
            self._refresh_program_list()
            self._prog_list.setCurrentRow(idx + 1)

    def _on_prog_clear(self) -> None:
        total_steps = sum(len(p) for p in self._jobs.values())
        if total_steps == 0 and not self._targets:
            return
        r = QMessageBox.question(
            self, "Clear all",
            f"Xoá toàn bộ {len(self._jobs)} job(s), {total_steps} instructions, "
            f"và {len(self._targets)} target(s)?\n\n"
            "Reset project về MAIN job rỗng.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if r != QMessageBox.StandardButton.Yes: return
        self._jobs = {"MAIN": []}
        self._active_job = "MAIN"
        self._targets.clear()
        self._refresh_job_combo()
        self._refresh_program_list()
        self._refresh_target_list()
        self._set_status("Project cleared (reset to MAIN/empty)", level="ok")

    def _on_prog_play(self) -> None:
        if self._prog_thread is not None and self._prog_thread.is_alive():
            self._set_status("Program already running", level="warn"); return
        if not self._program:
            self._set_status("Program empty", level="warn"); return
        self._prog_stop.clear()
        self._set_status(f"Program: running ({len(self._program)} steps)", "ok")
        self._prog_thread = threading.Thread(
            target=self._play_program_loop, daemon=True)
        self._prog_thread.start()

    def _on_prog_stop(self) -> None:
        if self._prog_thread is not None and self._prog_thread.is_alive():
            self._prog_stop.set()
            self._prog_pause.clear()                # unstuck nếu đang pause
            self._set_status("Program: stopping...", level="warn")

    def _on_program_done(self) -> None:
        # Reset pause UI (program kết thúc → không thể pause nữa).
        if hasattr(self, "_btn_pause"):
            self._btn_pause.setChecked(False)
            self._btn_pause.setText("Pause")
        self._prog_pause.clear()

    def _wait_while_paused(self) -> None:
        """Block player loop khi pause is set, exit nhanh khi stop is set."""
        while self._prog_pause.is_set() and not self._prog_stop.is_set():
            time.sleep(0.05)

    def _resolve_move_target(
        self, ins: Instruction, want: str,
    ) -> list[float] | None:
        """Resolve joints (want='joints') hoặc tcp_pose (want='tcp_pose') cho
        MoveJ/MoveL — từ target library nếu ins.target_name set, không thì
        inline fields. Return None + emit err nếu missing."""
        if ins.target_name:
            tgt = self._targets.get(ins.target_name)
            if tgt is None:
                self._signals.status.emit(
                    f"Target '{ins.target_name}' không tồn tại", "err")
                return None
            return list(tgt[want])
        return list(ins.joints if want == "joints" else ins.tcp_pose)

    def _play_program_loop(self) -> None:
        n = len(self._program)
        # Modal sim state (scale animation speed; modifiers chỉ là metadata).
        sim_vj_pct: float = 10.0
        try:
            for i, ins in enumerate(self._program):
                if self._prog_stop.is_set():
                    self._signals.status.emit("Program: stopped", "warn"); return
                self._wait_while_paused()
                if self._prog_stop.is_set():
                    self._signals.status.emit("Program: stopped", "warn"); return
                self._signals.status.emit(
                    f"Step {i+1}/{n}: {ins.describe()}", "info")
                t = ins.type
                # Animation step count: VJ% nhỏ + sim_speed_mult thấp → chậm.
                base_steps = int(40 * 30.0 / max(5.0, sim_vj_pct))
                steps = max(8, int(base_steps / max(0.1, self._sim_speed_mult)))
                if t == "MoveJ":
                    joints = self._resolve_move_target(ins, "joints")
                    if joints is None: return
                    self._animate_to(joints, steps=steps, dt=0.025,
                                      stop_event=self._prog_stop,
                                      pause_event=self._prog_pause)
                elif t == "MoveL":
                    if ins.target_name:
                        # Target stored joints — bypass IK, animate trực tiếp.
                        joints = self._resolve_move_target(ins, "joints")
                        if joints is None: return
                        self._animate_to(joints, steps=steps, dt=0.025,
                                          stop_event=self._prog_stop,
                                          pause_event=self._prog_pause)
                    else:
                        sol = self._solve_movel(ins.tcp_pose)
                        if sol is None:
                            self._signals.status.emit(
                                f"Step {i+1}: IK fail, abort", "err"); return
                        self._animate_to(sol, steps=steps, dt=0.025,
                                          stop_event=self._prog_stop,
                                          pause_event=self._prog_pause)
                elif t == "MoveC":
                    # Sim đơn giản: chạy MoveL tới mid rồi end (không nội suy
                    # circular thực — đủ để verify trình tự, .JBI vẫn MOVC).
                    for pose in (ins.tcp_pose_mid, ins.tcp_pose):
                        sol = self._solve_movel(pose)
                        if sol is None:
                            self._signals.status.emit(
                                f"Step {i+1}: IK fail (MoveC), abort", "err"); return
                        self._animate_to(sol, steps=steps, dt=0.025,
                                          stop_event=self._prog_stop,
                                          pause_event=self._prog_pause)
                elif t == "SetGripper":
                    self._signals.gripper.emit(bool(ins.gripper_close))
                    time.sleep(0.25 / max(0.1, self._sim_speed_mult))
                elif t == "Wait":
                    deadline = time.monotonic() + max(0.0, ins.wait_seconds) \
                        / max(0.1, self._sim_speed_mult)
                    while time.monotonic() < deadline:
                        if self._prog_stop.is_set(): return
                        self._wait_while_paused()
                        if self._prog_stop.is_set(): return
                        time.sleep(0.05)
                elif t == "WaitIO":
                    # Sim không có IO thật → chỉ log + short delay.
                    self._signals.status.emit(
                        f"Step {i+1}: (sim) WaitIO IN#{ins.io_index}="
                        f"{'ON' if ins.io_state else 'OFF'} → assumed satisfied",
                        "warn")
                    time.sleep(0.3 / max(0.1, self._sim_speed_mult))
                elif t == "SetSpeed":
                    sim_vj_pct = float(ins.speed_joint_pct)
                elif t == "ShowMessage":
                    self._signals.status.emit(
                        f"Step {i+1}: MSG \"{ins.message[:32]}\"", "info")
                    time.sleep(0.4 / max(0.1, self._sim_speed_mult))
                elif t == "CallJob":
                    # Sim không thực thi sub-job — chỉ log để xác nhận order.
                    self._signals.status.emit(
                        f"Step {i+1}: (sim) CALL JOB:{ins.job_name} → skipped",
                        "warn")
                    time.sleep(0.2 / max(0.1, self._sim_speed_mult))
                elif t == "SimEvent":
                    # Sim hook — emit signal + log. Không có side effect mặc định
                    # (downstream code có thể subscribe nếu cần custom action).
                    pl = f" — {ins.event_payload}" if ins.event_payload else ""
                    self._signals.status.emit(
                        f"⚑ Step {i+1}: SimEvent '{ins.event_name}'{pl}", "info")
                    time.sleep(0.15 / max(0.1, self._sim_speed_mult))
                # SetRounding / SetTool / SetRefFrame: pure metadata cho .JBI,
                # sim không cần làm gì — đã hiện trong status.
            self._signals.status.emit(f"Program: done ({n} steps)", "ok")
        except Exception as e:                              # noqa: BLE001
            self._signals.status.emit(f"Program error: {e}", "err")
        finally:
            self._signals.program_done.emit()

    def _solve_movel(self, tcp_pose_6: list[float]) -> list[float] | None:
        """Solve IK + verify post-FK pose error. Return None nếu:
          - IK không converge
          - Convergent solution có pose error > tol (do joint limit clipping
            kéo solution ra biên — silent failure mode trước đây)."""
        TOL_POS_MM = 0.5
        TOL_ROT_RAD = 1e-3
        T_target = _xyz_rpy_to_matrix(*tcp_pose_6)
        T_flange_tool = self._tool_frames[self._tool_idx][1]
        T_target_tool0 = T_target @ np.linalg.inv(T_flange_tool)
        q_init = [math.radians(q) for q in self._joints]
        sol = inverse_kinematics_seeded(
            self._model, T_target_tool0, q_init,
            tol_mm=TOL_POS_MM, tol_rad=TOL_ROT_RAD, max_iter=100)
        if sol is None:
            return None
        # Post-FK verify: nếu IK clip joint limit, may converge tới biên với
        # error > tol. Recompute FK + check.
        T_actual = forward_kinematics_urdf(self._model, sol)
        pos_err = float(np.linalg.norm(
            T_actual[:3, 3] - T_target_tool0[:3, 3]))
        # Rotation error via axis-angle log map
        R_err = T_target_tool0[:3, :3] @ T_actual[:3, :3].T
        cos_theta = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
        rot_err_rad = float(np.arccos(cos_theta))
        if pos_err > TOL_POS_MM or rot_err_rad > TOL_ROT_RAD:
            logger.warning(
                "IK convergent but pose error > tol: pos=%.3fmm rot=%.4frad "
                "(target có thể ngoài tầm hoặc joint limit clip)",
                pos_err, rot_err_rad)
            return None
        return [math.degrees(q) for q in sol]

    # ── Save / Load JSON + Export .JBI ────────────────────────────────
    def _on_prog_save_dlg(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project", "", "Program JSON (*.json)")
        if not path: return
        try:
            # v3 format: project với nhiều jobs + global targets.
            doc = {
                "version": 3,
                "active_job": self._active_job,
                "targets": self._targets,
                "jobs": {
                    name: [ins.to_dict() for ins in prog]
                    for name, prog in self._jobs.items()
                },
            }
            Path(path).write_text(json.dumps(doc, indent=2), encoding="utf-8")
            total_steps = sum(len(p) for p in self._jobs.values())
            self._set_status(
                f"Saved {len(self._jobs)} job(s), {total_steps} steps total, "
                f"{len(self._targets)} targets", level="ok")
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"Save failed: {e}", level="err")

    def _on_prog_load_dlg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load project", "", "Program JSON (*.json)")
        if not path: return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            # Backward compat across 3 formats:
            #   v1: bare list of instructions → single MAIN job
            #   v2: {"targets":..., "program":[...]} → single MAIN job
            #   v3: {"jobs":{name:[...]}, "active_job":..., "targets":...}
            if isinstance(data, list):
                self._jobs = {"MAIN": [Instruction.from_dict(d) for d in data]}
                self._active_job = "MAIN"
                self._targets = {}
            elif isinstance(data, dict):
                ver = int(data.get("version", 1))
                tgt_raw = data.get("targets", {}) or {}
                self._targets = {
                    str(k): {
                        "joints": list(v["joints"]),
                        "tcp_pose": list(v["tcp_pose"]),
                    } for k, v in tgt_raw.items()
                }
                if "jobs" in data:                              # v3
                    self._jobs = {
                        str(name): [Instruction.from_dict(d) for d in prog]
                        for name, prog in data["jobs"].items()
                    }
                    self._active_job = str(data.get("active_job",
                                                       next(iter(self._jobs.keys()))))
                    if self._active_job not in self._jobs:
                        self._active_job = next(iter(self._jobs.keys()))
                else:                                          # v2
                    prog_list = data.get("program", [])
                    self._jobs = {"MAIN": [Instruction.from_dict(d) for d in prog_list]}
                    self._active_job = "MAIN"
            else:
                raise ValueError("Unknown JSON shape")
            if not self._jobs:
                self._jobs = {"MAIN": []}; self._active_job = "MAIN"
            self._refresh_job_combo()
            self._refresh_program_list()
            self._refresh_target_list()
            total_steps = sum(len(p) for p in self._jobs.values())
            self._set_status(
                f"Loaded {len(self._jobs)} job(s), {total_steps} steps, "
                f"{len(self._targets)} targets", level="ok")
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"Load failed: {e}", level="err")

    def _on_prog_export_dlg(self) -> None:
        # Multi-job: nếu project có > 1 job, hỏi user mode export.
        non_empty_jobs = [n for n, p in self._jobs.items() if p]
        if not non_empty_jobs:
            self._set_status("All jobs empty", level="warn"); return
        export_all = False
        if len(non_empty_jobs) > 1:
            r = QMessageBox.question(
                self, "Export INFORM .JBI",
                f"Project có {len(non_empty_jobs)} non-empty jobs.\n\n"
                f"  Yes → Export ALL jobs (separate .JBI files vào 1 thư mục)\n"
                f"  No  → Export only current ({self._active_job})",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel)
            if r == QMessageBox.StandardButton.Cancel: return
            export_all = (r == QMessageBox.StandardButton.Yes)

        if export_all:
            return self._export_all_jobs(non_empty_jobs)

        if not self._program:
            self._set_status("Current job empty", level="warn"); return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Yaskawa INFORM .JBI",
            f"{self._active_job}.JBI", "INFORM (*.JBI)")
        if not path: return
        try:
            stem = Path(path).stem[:32].replace(" ", "_") or "PROG"
            self._export_job_to_path(self._program, stem, Path(path))
            self._set_status(
                f"Exported '{self._active_job}' → {Path(path).name}", level="ok")
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"Export failed: {e}", level="err")

    def _export_all_jobs(self, job_names: list[str]) -> None:
        out_dir = QFileDialog.getExistingDirectory(
            self, "Export ALL jobs — chọn thư mục output")
        if not out_dir: return
        # Sequential — IK solving (~1.4ms hot, ~10ms cold) per MoveL có GIL
        # contention nếu parallel; benchmark confirms threading SLOWER for sub-ms
        # numpy ops. Đa phần jobs ≤ 50 instructions → tổng <200ms acceptable.
        n_ok = 0; errors: list[str] = []
        for name in job_names:
            try:
                stem = self._safe_job_name(name)
                path = Path(out_dir) / f"{stem}.JBI"
                self._export_job_to_path(self._jobs[name], stem, path)
                n_ok += 1
            except Exception as e:                          # noqa: BLE001
                errors.append(f"{name}: {e}")
        if errors:
            self._set_status(
                f"Exported {n_ok}/{len(job_names)} OK, {len(errors)} fail: "
                + "; ".join(errors[:3]), level="warn")
        else:
            self._set_status(
                f"Exported {n_ok} .JBI files → {out_dir}", level="ok")

    def _export_job_to_path(
        self, program: list[Instruction], job_name: str, path: Path,
    ) -> None:
        """Helper: export 1 job's instruction list ra .JBI file tại path.

        Raises trên IK failure / invalid CallJob name. Caller catch để show
        status. KHÔNG đụng self._program — chỉ dùng `program` param.
        """
        builder = InformJobBuilder(
            name=job_name, max_speed_pct=self._pp_max_speed_pct)
        # Pre-pass: collect referenced targets, add as named C-vars upfront
        # (RoboDK convention — multiple references share single C-var).
        target_cvars: dict[str, str] = {}
        for ins in program:
            if ins.type in ("MoveJ", "MoveL") and ins.target_name:
                if ins.target_name not in target_cvars:
                    if ins.target_name not in self._targets:
                        raise KeyError(
                            f"Target '{ins.target_name}' không tồn tại")
                    cvar_name = f"T_{ins.target_name}"[:32]
                    builder.add_position(
                        cvar_name, self._targets[ins.target_name]["joints"])
                    target_cvars[ins.target_name] = cvar_name
        # Modal state — áp lên MOVJ/MOVL/MOVC kế tiếp. Initial từ pp settings.
        cur_vj_pct: float = self._pp_default_vj
        cur_v_mm_s: float = self._pp_default_v_mms
        cur_pl: int | None = None
        cur_tl: int | None = None
        cur_uf: int | None = None
        pos_idx = 0
        for i, ins in enumerate(program):
            t = ins.type
            if t == "MoveJ":
                if ins.target_name:
                    pname = target_cvars[ins.target_name]
                else:
                    pname = f"P{pos_idx:03d}"
                    builder.add_position(pname, list(ins.joints))
                    pos_idx += 1
                builder.movj(pname, speed_pct=cur_vj_pct,
                             tool_no=cur_tl, pl=cur_pl, user_frame=cur_uf)
            elif t == "MoveL":
                if ins.target_name:
                    pname = target_cvars[ins.target_name]
                else:
                    sol = self._solve_movel(ins.tcp_pose)
                    if sol is None:
                        raise RuntimeError(
                            f"IK fail at step {i+1} ({job_name})")
                    pname = f"P{pos_idx:03d}"
                    builder.add_position(pname, sol)
                    pos_idx += 1
                builder.movl(pname, speed_mm_s=cur_v_mm_s,
                             tool_no=cur_tl, pl=cur_pl, user_frame=cur_uf)
            elif t == "MoveC":
                sol_m = self._solve_movel(ins.tcp_pose_mid)
                sol_e = self._solve_movel(ins.tcp_pose)
                if sol_m is None or sol_e is None:
                    raise RuntimeError(
                        f"IK fail at step {i+1} MoveC ({job_name})")
                pname_m = f"P{pos_idx:03d}"; pos_idx += 1
                pname_e = f"P{pos_idx:03d}"; pos_idx += 1
                builder.add_position(pname_m, sol_m)
                builder.add_position(pname_e, sol_e)
                builder.movc(pname_m, speed_mm_s=cur_v_mm_s,
                             tool_no=cur_tl, pl=cur_pl, user_frame=cur_uf)
                builder.movc(pname_e, speed_mm_s=cur_v_mm_s,
                             tool_no=cur_tl, pl=cur_pl, user_frame=cur_uf)
            elif t == "SetGripper":
                builder.dout(1, on=ins.gripper_close)
            elif t == "Wait":
                builder.timer(max(0.0, ins.wait_seconds))
            elif t == "WaitIO":
                builder.wait_in(int(ins.io_index), on=bool(ins.io_state),
                                timeout_s=max(0.0, float(ins.io_timeout_s)))
            elif t == "SetSpeed":
                cur_vj_pct = float(ins.speed_joint_pct)
                cur_v_mm_s = float(ins.speed_linear_mm_s)
                builder.comment(
                    f"SetSpeed VJ={cur_vj_pct:.1f}% V={cur_v_mm_s:.0f}mm/s")
            elif t == "SetRounding":
                cur_pl = int(ins.rounding_pl)
                builder.comment(f"SetRounding PL={cur_pl}")
            elif t == "SetTool":
                cur_tl = int(ins.tool_no)
                builder.comment(f"SetTool TL#{cur_tl}")
            elif t == "SetRefFrame":
                cur_uf = int(ins.ref_frame_no)
                builder.comment(f"SetRefFrame UF#{cur_uf}")
            elif t == "ShowMessage":
                builder.msg(ins.message)
            elif t == "CallJob":
                builder.call_job(ins.job_name)
            elif t == "SimEvent":
                # Không export ra INFORM (sim-only). Emit comment để traceable.
                pl = f" — {ins.event_payload}" if ins.event_payload else ""
                builder.comment(f"SimEvent: {ins.event_name}{pl}")
        path.write_bytes(builder.render().encode("utf-8"))

    # ══════════════════════════════════════════════════════════════════
    # Dialogs
    # ══════════════════════════════════════════════════════════════════

    def _show_cell_info(self) -> None:
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

    # ══════════════════════════════════════════════════════════════════
    # Lifecycle
    # ══════════════════════════════════════════════════════════════════

    def closeEvent(self, event: QCloseEvent) -> None:
        # Stop worker threads
        self._demo_stop.set()
        self._prog_stop.set()
        try:
            self._plotter.close()
        except Exception:                                   # noqa: BLE001
            pass
        event.accept()
