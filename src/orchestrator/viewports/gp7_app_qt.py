"""
gp7_app_qt.py
─────────────
GP7 Digital Twin app — PyQt6 + pyvistaqt (VTK) industrial-standard stack.

Clean layer separation:
  • Kinematics math (FK/IK/collision/trajectory) — fully reused from
    `src/orchestrator/kinematics/` (verified to match RoboDK 0.00mm).
  • 3D rendering — VTK via `pyvistaqt.QtInteractor` (same stack as ROS RViz,
    MoveIt). Camera arcball/pan/zoom built-in.
  • GUI shell — PyQt6 native widgets (QMenuBar, QToolBar, QDockWidget,
    QFormLayout, QSlider, QPushButton). QSS stylesheet customizable.

Why this stack (vs. current Open3D gui):
  • Industrial standard — VTK + Qt are the de-facto standard for robotics 3D apps.
  • UI polish — QFormLayout auto-aligns label+widget, QDockWidget hide/show
    panels VSCode-style, QToolBar with icons + accelerators.
  • Math independent of viewport — FK/IK/JBI export accuracy 100% preserved.

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
from PyQt6.QtCore import QEvent, QPoint, QSize, Qt, QTimer
from PyQt6.QtGui import (
    QAction, QActionGroup, QCloseEvent, QColor, QFont, QImage, QKeySequence,
    QPixmap, QShortcut,
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
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QTabWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from ..backends.inform_codegen import InformJobBuilder
from ..backends.motoman_hse import MotomanHSEBackend
from ..kinematics.inverse_kinematics import (
    inverse_kinematics, inverse_kinematics_batch, inverse_kinematics_seeded,
    manipulability,
)
from ..kinematics.pieper_gp7 import (
    inverse_kinematics_pieper_gp7,
    inverse_kinematics_pieper_gp7_nearest,
    inverse_kinematics_pieper_gp7_tagged,
)
from ..kinematics.urdf_chain import (
    URDFRobot, forward_kinematics_urdf, gp7_urdf, link_frames_batch_urdf,
    link_frames_urdf,
)
from ...cell import CellConfig
from ...cell.cell_models import (
    FrameConfig, ObjectConfig, FloorConfig, WorktableConfig,
    PedestalConfig, CameraMountConfig, CameraConfig, CameraIntrinsics, PoseConfig,
)
from .control_panel import (
    _build_ref_frames,
    _build_tool_frames,
    _matrix_to_xyz_rpy_deg,
    _rotation_about_axis_3x3,
    _xyz_rpy_to_matrix,
)
from .mixin_about import AboutMixin
from .mixin_camera import CameraMixin
from .mixin_connection import ConnectionMixin
from .mixin_experiment import ExperimentMixin
from .mixin_job_target import JobTargetMixin
from .mixin_program_io import ProgramIOMixin
from .mixin_program_playback import ProgramPlaybackMixin
from .open3d_gui_sim_robot import _GP7_MESH_MAP, _YASKAWA_BLUE
from .program_model import Instruction
from .qt_helpers import (
    draw_arrow_down_icon as _draw_arrow_down_icon,
    draw_arrow_up_icon as _draw_arrow_up_icon,
    draw_copy_icon as _draw_copy_icon,
    draw_menu_icon as _draw_menu_icon,
    draw_paste_icon as _draw_paste_icon,
    draw_open_icon as _draw_open_icon,
    draw_plus_icon as _draw_plus_icon,
    draw_rename_icon as _draw_rename_icon,
    draw_trash_icon as _draw_trash_icon,
    draw_x_icon as _draw_x_icon,
    numpy_to_vtk_matrix as _numpy_to_vtk_matrix,
)
from .qt_widgets import CollapsibleSection, WorkerSignals as _WorkerSignals
from .script_api import ScriptProgramAPI as _ScriptProgramAPI

logger = logging.getLogger(__name__)

# Preferred dock widths (px). Jog + Cell + Program docks tabified in the left
# area → Qt forces them to the same width. When a tab is activated, resize the area
# to the preferred width of that tab (jog wide for 3-col layout, cell narrow for tree,
# program medium for playback bar). minimumWidth of jog/program lowered to cell width
# so the area can shrink when cell tab is active (hidden tabs have QScrollArea, no issue).
_JOG_DOCK_W = 580
_CELL_DOCK_W = 180
# Wide enough to fit all program dock content without horizontal scroll: widest row
# (Modal "Speed": VJ + V spinbox + button 138px) min ~449px + vbox margin + scrollbar
# ~480px; extra margin for real Windows fonts → 540.
_PROG_DOCK_W = 540

# Camera frustum depth: by default follows the optical axis until hitting the floor
# (Z=0) — the real observation area. If the camera is not pointing down, use this
# default far value (m). Clamped to the useful range of the D455 (~0.4–6 m).
_CAM_FRUSTUM_FAR_M = 1.5
_CAM_FRUSTUM_MIN_M = 0.15
_CAM_FRUSTUM_MAX_M = 6.0


class GP7AppQt(
    QMainWindow,
    ConnectionMixin,
    ProgramIOMixin,
    JobTargetMixin,
    ProgramPlaybackMixin,
    AboutMixin,
    CameraMixin,
    ExperimentMixin,
):
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
        # Eye scaled ~1.5× from center for comfortable viewing distance at startup
        # (scene was too close before). Covers 6 standard views like RoboDK + Iso.
        "Iso":   (( 3.05, -2.25, 1.93), (0.35, 0.0, 0.50), (0.0, 0.0, 1.0)),
        "Front": (( 3.65,  0.00, 0.95), (0.35, 0.0, 0.50), (0.0, 0.0, 1.0)),
        "Back":  ((-2.95,  0.00, 0.95), (0.35, 0.0, 0.50), (0.0, 0.0, 1.0)),
        "Right": (( 0.35, -3.75, 1.25), (0.35, 0.0, 0.50), (0.0, 0.0, 1.0)),
        "Left":  (( 0.35,  3.75, 1.25), (0.35, 0.0, 0.50), (0.0, 0.0, 1.0)),
        "Top":   (( 0.36,  0.02, 5.00), (0.35, 0.0, 0.50), (1.0, 0.0, 0.0)),
    }

    # GP7 reach radii (m) — approximation. Real envelope is toroidal but
    # a sphere centered at J1 is sufficient to visualize for most use cases.
    _REACH_FLANGE = 0.927          # max flange reach from J1
    _REACH_WRIST  = 0.847          # = flange − 80mm tool0/flange offset
    _REACH_TOOL_EXTRA = 0.10       # tool offset typical ~100mm

    def __init__(
        self,
        cell_config: Any = None,
        project_root: str | Path = ".",
        parent: QWidget | None = None,
        program_path: str | Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Yaskawa GP7 Programming")
        self.resize(1600, 950)
        self.setMinimumSize(1200, 720)
        # Modern unified appearance: borderless dock margins, no system separator.
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks)

        # Cell config may be None on cold start — robot/cell load deferred via
        # menu File → Load Robot GP7 / Load Cell from YAML.
        self._cell_config = cell_config
        self._project_root = Path(project_root)
        # Optional: auto-load a program (.json) at startup (--program). Loaded
        # after the robot is ready in _post_show_setup.
        self._startup_program_path = (Path(program_path) if program_path
                                       else None)

        # Robot model + state — INITIALIZED EMPTY. Will be filled when _load_robot_gp7
        # runs (from menu or auto-triggered in _post_show_setup if cell_config
        # is already available).
        self._base_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._model = None
        self._home_joints: list[float] = [0.0] * 6
        self._joints: list[float] = [0.0] * 6

        # Frames — empty defaults (1 neutral entry). _build_tool_frames /
        # _build_ref_frames already handle cell_config=None.
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
        # 240mm (= 300mm - tip portion 60mm). RoboDK-style.
        self._frame_triad_size = 0.24
        # WorkSpace — single translucent sphere actor (or None)
        self._workspace_actor: Any = None
        # Fullscreen toggle state
        self._fullscreen = False

        # (Text-prefix toggle — no icon cache needed)
        # Press-hold continuous jog timer
        self._jog_timer = QTimer(self)
        self._jog_timer.setInterval(120)
        self._jog_timer.timeout.connect(self._on_jog_tick)
        self._jog_active: tuple[str, int, int] | None = None  # (mode, axis, sign)

        # Worker control
        self._prog_thread: threading.Thread | None = None
        self._prog_stop = threading.Event()
        self._prog_pause = threading.Event()        # set = pause (held)
        # Simulated digital inputs for IN#(n) conditions in JUMP/IF/WHILE. Sim has
        # no real IO → default OFF (False); user/script can set them here.
        self._sim_io: dict[int, bool] = {}
        # Multi-job project. One dock holds multiple jobs; only 1 job active at
        # a time (combo box). self._program (property) points to the active job.
        # Targets are project-global (RoboDK convention).
        self._jobs: dict[str, list[Instruction]] = {"MAIN": []}
        self._active_job: str = "MAIN"
        # Per-job ///FOLDERNAME from imported .JBI (job name → folder). Preserved
        # through save/load and re-emitted on export. Absent = no folder line.
        self._job_folders: dict[str, str] = {}
        # Verbatim re-export cache: job key → {text, name, sig}. An imported job
        # UNCHANGED since import (sig match) re-exports byte-for-byte from `text`.
        self._jbi_raw: dict[str, dict] = {}
        # Global //POS table (key 'P<idx>' → joints_deg) for P[Bxxx] indirect.
        self._jbi_positions: dict[str, list[float]] = {}
        # Target library: name → {"joints": [..6 deg..], "tcp_pose": [..6..]}
        self._targets: dict[str, dict] = {}
        # Post-processor settings (INFORM .JBI generation tuning).
        # max_speed_pct = safety cap for VJ. default_vj/v = initial modal state.
        # Set BEFORE _project_signature() below — the signature includes them.
        self._pp_max_speed_pct: float = 30.0
        self._pp_default_vj: float = 10.0
        self._pp_default_v_mms: float = 100.0
        # Dirty-check baseline: project signature at the last Save/Load.
        # closeEvent compares it to warn about unsaved changes. Empty app on open =
        # clean (won't prompt if the user hasn't added anything).
        self._saved_signature: str = self._project_signature()
        # Sim speed multiplier (1.0 = nominal). Higher → animate faster.
        self._sim_speed_mult: float = 1.0
        # Digital-output index that actuates the gripper (OT#n). DOUT/SetDO on this
        # index drives grasp(ON)/release(OFF) in the sim, matching the real robot.
        # Loaded from config/experiment.yaml (gripper_do_index); default 1.
        self._gripper_do_index: int = self._load_gripper_do_index()
        # Teach-on-surface mode (click 3D scene → create target on picked surface).
        self._surface_pick_mode: bool = False
        # Cache: id(vtkDataSet) → (mesh_with_normals, cell_normals_array).
        # mesh.compute_normals() is O(N_cells) — expensive; cache so re-picking
        # the same mesh avoids recomputation.
        self._normal_cache: dict[int, tuple] = {}

        # HSE robot connection state — populated via "Connection settings…" dialog.
        # Defaults taken from cell_config.robot_connection if available, otherwise empty.
        # Defaults pre-filled with the verified YRC1000+GP7 connection (see
        # tools/probe_hse.py): IP 192.168.125.100, FTP rcmaster / 16×'9' (Yaskawa
        # documented default), job dir /JOB. cell_config.robot_connection (if set)
        # overrides any of these.
        rc = getattr(cell_config, "robot_connection", None)
        self._hse_ip: str = (getattr(rc, "ip", "") or "192.168.125.100")
        self._hse_tool_no: int = int(getattr(rc, "tool_no", 1) or 1)
        self._hse_ftp_user: str = (getattr(rc, "ftp_user", "") or "rcmaster")
        self._hse_ftp_pass: str = (getattr(rc, "ftp_pass", "") or "9999999999999999")
        self._hse_ftp_dir: str = (getattr(rc, "ftp_job_dir", "/JOB") or "/JOB")
        self._hse_thread: threading.Thread | None = None
        self._hse_stop = threading.Event()
        self._send_pose_busy: bool = False     # Phase-1 discrete send in progress (re-entrancy guard)
        self._send_pose_stop = threading.Event()   # abort the discrete send (Stop-all)
        self._send_pose_thread: threading.Thread | None = None  # joinable on close/Stop-all
        # Sim move animation (Home / Zero / Align) — tracked so it can be cancelled
        # before a new move and joined on app close (no untracked teleport threads).
        self._anim_thread: threading.Thread | None = None
        self._anim_stop = threading.Event()

        # Phase-2 live jog → REAL robot (streaming HSE move, RoboDK-style). OFF by
        # default. The worker holds one HSE connection + servo on and chases the
        # latest jogged joints (latest-value-wins, coalesced). ⚠ REAL MOTION.
        self._live_jog_thread: threading.Thread | None = None
        self._live_jog_stop = threading.Event()
        self._live_jog_lock = threading.Lock()
        self._live_jog_target: list[float] | None = None   # latest jogged joints (deg)
        self._live_jog_dirty: bool = False                 # new target pending send

        # Camera (D455) state — live capture + vision-guided control (CameraMixin).
        self._cam_thread: threading.Thread | None = None
        self._cam_stop = threading.Event()
        self._cam_running: bool = False
        self._cam_source: str = "Auto"
        self._cam_use_detector: bool = False
        self._cam_color_size: tuple[int, int] = (1280, 720)
        self._cam_fps: int = 30
        self._last_camera_objects: list[dict] = []
        self._last_depth = None
        self._last_rgb = None
        self._last_frame = None         # atomic (rgb, depth) coherent pair for capture
        self._last_display = None            # processed RGB image (built by worker)
        self._last_fps: float = 0.0
        self._last_intrinsics = None
        self._last_source = None
        self._cam_frame_pending: bool = False  # backpressure worker→UI
        self._cam_closing: bool = False        # block slots from touching widgets on app close
        self._cam_capture_seq: int = 0         # capture filename counter
        self._cam_show_depth: bool = False
        self._cam_show_overlay: bool = True
        self._last_grasp_target: str | None = None
        self._last_grasp_T = None

        # Worker → main signals
        self._signals = _WorkerSignals()
        self._signals.joints_update.connect(self._apply_joints_main)
        self._signals.status.connect(self._set_status)
        self._signals.gripper.connect(self._toggle_gripper)
        self._signals.sim_reset.connect(self._on_reset_scene)
        self._signals.program_done.connect(self._on_program_done)
        self._signals.prog_step.connect(self._on_prog_step_highlight)
        self._signals.prog_show_job.connect(self._on_prog_show_job)
        self._signals.camera_result.connect(self._on_camera_result)
        self._signals.exp_progress.connect(self._on_exp_progress)
        self._signals.exp_done.connect(self._on_exp_done)
        self._signals.live_jog_off.connect(self._on_live_jog_worker_exit)
        self._signals.run_overwrite_blocked.connect(self._on_run_overwrite_blocked)

        # Build UI (fast — Qt widget construction only)
        self._build_viewport()
        self._build_menu_bar()
        self._build_jog_dock()
        self._build_cell_tree_dock()
        self._build_program_dock()
        self._build_camera_dock()
        self._build_experiment_dock()
        self._connect_group_dock_redock()
        self._build_status_bar()

        # Defer scene load (STL parse + VTK actor creation ~1-2s) — window
        # pops up immediately, scene loads after event loop starts → smooth startup.
        from PyQt6.QtCore import QTimer as _QT
        _QT.singleShot(0, self._post_show_setup)

    def _load_gripper_do_index(self) -> int:
        """Read gripper_do_index from config/experiment.yaml (default 1).

        Determines which DOUT/SetDO output actuates the gripper in the sim so
        imported DOUT-based pick-place jobs grasp/release like the real robot.
        """
        try:
            from ...utils import load_yaml
            exp_yaml = self._project_root / "config" / "experiment.yaml"
            if exp_yaml.exists():
                loaded = load_yaml(exp_yaml)
                if isinstance(loaded, dict) and "gripper_do_index" in loaded:
                    return int(loaded["gripper_do_index"])
        except Exception:                                       # noqa: BLE001
            pass
        return 1

    def _post_show_setup(self) -> None:
        """Runs after the window has been shown — initialise the base scene. Robot/cell
        will be loaded deferred via the File menu. If cell_config was passed to __init__
        (backward compat with `python scripts/16_app_qt.py --config ...`),
        auto-load it immediately to preserve legacy behaviour.
        """
        self._set_status("Initializing…", level="info")
        # Scene base: axes + lighting. Floor NOT added — it is off by default;
        # user enables it via View → Visibility → Floor.
        self._add_world_axes_triad()
        self._setup_lighting()
        self._set_camera_preset("Iso")

        # Backward compat: if cell_config was passed in the constructor, auto-load
        # both the robot and cell meshes.
        if self._cell_config is not None:
            self._load_robot_gp7()
            self._load_cell_assets()
        elif self._startup_program_path is not None:
            # --program: robot needed for sim playback → auto-load GP7 default.
            self._load_robot_gp7()
        else:
            # Empty state: disable robot-dependent UI; status hint.
            self._set_robot_dependent_enabled(False)
            self._set_status(
                "Robot not loaded — File → Load Robot GP7 to begin",
                level="info")

        # Auto-load the program file (--program) — after the robot is ready.
        if self._startup_program_path is not None:
            self._load_program_file(self._startup_program_path)
            self._program_dock.setVisible(True)
            self._program_dock.raise_()

    # ══════════════════════════════════════════════════════════════════
    # UI construction
    # ══════════════════════════════════════════════════════════════════

    def _build_viewport(self) -> None:
        """3D viewport using pyvistaqt.QtInteractor (VTK).

        Camera arcball/pan/zoom built-in. Background gradient set via
        `set_background(color, top=color_top)`.
        """
        self._plotter = QtInteractor(self)
        self._plotter.set_background([95/255, 65/255, 175/255],          # bot (purple-blue)
                                      top=[5/255, 5/255, 28/255])         # top (near-black navy)
        try:
            self._plotter.enable_anti_aliasing()
        except Exception:                                  # noqa: BLE001
            pass
        # World axes widget in the top-left corner of the viewport
        self._plotter.add_axes(line_width=3, labels_off=False)
        self.setCentralWidget(self._plotter.interactor)

    # ── Toggle action: Qt native checkable + custom indicator QSS ──
    # Indicator column reserved at the same width for ALL menu items
    # (checkable + plain) → text aligned pixel-perfectly. ✓ rendered via
    # generated check.png referenced from QSS (see qt_theme.py).

    def _make_toggle(self, base_text: str, initial: bool = False,
                       callback=None) -> QAction:
        """QAction Word/VSCode-style — ✓ in the native indicator column, text
        aligned perfectly (uniform across checked / unchecked / plain items).

        callback(new_state: bool) — fired when the user clicks. May be None.
        """
        act = QAction(base_text, self)
        act.setCheckable(True)
        act.setChecked(bool(initial))
        if callback is not None:
            act.toggled.connect(callback)
        return act

    def _set_toggle(self, act: QAction, state: bool) -> None:
        """External setter: update state WITHOUT firing the toggled signal."""
        act.blockSignals(True)
        act.setChecked(bool(state))
        act.blockSignals(False)

    def _build_menu_bar(self) -> None:
        """Menu bar — File / Edit / View / Robot / Program / Help.

        All toggle actions use a custom "✓  " prefix (visible) or "      "
        (6 spaces, approx width of "✓  "). Not natively checkable → no box.

        Structure:
          • File:    cell I/O + program file I/O + cell info + exit
          • Edit:    add components (same actions as cell tree context menu)
          • View:    Camera ▶ / Visibility ▶ / Window ▶ + Reset scene
          • Robot:   Home/Zero + Demo motion + Parameters + Teach surface
                     + Connection settings (Test button inside the dialog)
          • Program: Play / Pause / Stop / Run on Robot + Clear + PP/Script
          • Help:    About
        """
        mb = self.menuBar()

        # ── FILE ── Cell + Program file I/O + Cell info + Exit
        m_file = mb.addMenu("&File")
        # Robot/Cell load — deferred (cold start: app empty, user picks from menu)
        act_load_robot = QAction("Load Robot &GP7", self)
        act_load_robot.triggered.connect(self._on_action_load_robot_gp7)
        m_file.addAction(act_load_robot)
        act_load_cell = QAction("Load &Cell from YAML…", self)
        act_load_cell.triggered.connect(self._on_action_load_cell)
        m_file.addAction(act_load_cell)
        act_save_cell = QAction("Sa&ve Cell to YAML…", self)
        act_save_cell.triggered.connect(self._on_action_save_cell)
        m_file.addAction(act_save_cell)
        # Cell info — cell-context dialog (frames + objects + base xyz)
        act_cellinfo = QAction("Cell &info...", self)
        act_cellinfo.triggered.connect(self._show_cell_info)
        m_file.addAction(act_cellinfo)
        m_file.addSeparator()
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
        act_import_jbi = QAction("&Import .JBI (Yaskawa INFORM)...", self)
        act_import_jbi.triggered.connect(self._on_prog_import_jbi_dlg)
        m_file.addAction(act_import_jbi)
        m_file.addSeparator()
        act_exit = QAction("E&xit", self)
        act_exit.setShortcut(QKeySequence.StandardKey.Quit)
        act_exit.triggered.connect(self.close)
        m_file.addAction(act_exit)

        # ── EDIT ── Cell design: Add components (Robot/Object/Frame/...)
        # Same actions as the Cell tree context menu — more discoverable.
        m_edit = mb.addMenu("&Edit")
        act_add_robot = QAction("Add &Robot…", self)
        act_add_robot.triggered.connect(self._show_add_robot_dlg)
        m_edit.addAction(act_add_robot)
        act_add_gripper = QAction("Add &Gripper…", self)
        act_add_gripper.triggered.connect(self._show_add_gripper_dlg)
        m_edit.addAction(act_add_gripper)
        m_edit.addSeparator()
        act_add_obj = QAction("Add &Object…", self)
        act_add_obj.triggered.connect(self._show_add_object_dlg)
        m_edit.addAction(act_add_obj)
        act_add_frame = QAction("Add &Frame…", self)
        act_add_frame.triggered.connect(self._show_add_frame_dlg)
        m_edit.addAction(act_add_frame)
        m_edit.addSeparator()
        for kind, label in (("worktable", "Add &Worktable…"),
                              ("robot_pedestal", "Add &Pedestal…"),
                              ("floor", "Add F&loor…"),
                              ("camera_mount", "Add Camera &Mount…")):
            act = QAction(label, self)
            act.triggered.connect(
                lambda _checked=False, k=kind: self._show_add_single_dlg(k))
            m_edit.addAction(act)
        act_add_cam = QAction("Add &Camera…", self)
        act_add_cam.triggered.connect(self._show_add_camera_dlg)
        m_edit.addAction(act_add_cam)

        # ── VIEW ── Camera ▶ / Visibility ▶ / Window ▶ + Reset scene
        m_view = mb.addMenu("&View")

        # Camera submenu: presets (Iso + 5 orthogonal views, RoboDK convention)
        # + camera ops (Fit all, Perspective).
        # Camera presets — radio-style (exclusive) via manual click handler.
        # NOT using QActionGroup since plain QActions are needed (not checkable).
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
        cam_menu.addSeparator()
        # Camera ops: Fit + Perspective toggle (merged into Camera submenu)
        act_fit = QAction("&Fit all (reset camera)", self)
        act_fit.setShortcut("Alt+7")                       # RoboDK: Alt+7
        act_fit.triggered.connect(self._on_fit_all)
        cam_menu.addAction(act_fit)
        self._act_perspective = self._make_toggle(
            "Perspective view", initial=True,
            callback=self._on_toggle_perspective)
        cam_menu.addAction(self._act_perspective)

        # Visibility submenu — Floor / Axes + triad size adjust.
        # (Background gradient is a FIXED default, NO toggle.)
        vis_menu = m_view.addMenu("&Visibility")
        self._act_floor = self._make_toggle(
            "Floor", initial=False,
            callback=lambda c: self._toggle_floor(c))
        vis_menu.addAction(self._act_floor)
        self._act_axes = self._make_toggle(
            "World axes triad", initial=True,
            callback=lambda c: self._toggle_actor("__world_axes", c))
        vis_menu.addAction(self._act_axes)
        # Camera frustum (viewing cone of cell `camera`) — RoboDK-style camera viz.
        self._act_cam_frustum = self._make_toggle(
            "Camera frustum", initial=True,
            callback=lambda c: self._toggle_camera_frustum(c))
        vis_menu.addAction(self._act_cam_frustum)
        # (Removed: Reference frames +/- — niche; +/- shortcuts could conflict
        # with jog. Default triad size set via self._frame_triad_size init.)

        # Window submenu — Fullscreen + dock toggles (grouped in one place).
        win_menu = m_view.addMenu("&Window")
        self._act_fullscreen = self._make_toggle(
            "Fullscreen", initial=False,
            callback=self._on_toggle_fullscreen)
        self._act_fullscreen.setShortcut("F11")
        win_menu.addAction(self._act_fullscreen)
        win_menu.addSeparator()
        # Controls panel (HIDDEN by default — user enables it when jogging).
        self._act_jog_dock = self._make_toggle(
            "Controls panel", initial=False,
            callback=lambda c: self._open_dock_tab(self._jog_dock, c))
        win_menu.addAction(self._act_jog_dock)
        # Cell tree dock (visible by default — main editor).
        self._act_cell_dock = self._make_toggle(
            "Cell components", initial=True,
            callback=lambda c: self._open_dock_tab(self._cell_tree_dock, c))
        self._act_cell_dock.setShortcut("Ctrl+Shift+C")
        win_menu.addAction(self._act_cell_dock)
        # Program dock (HIDDEN by default) — grouped under View → Window instead
        # of scattered in the Program menu.
        self._act_prog_dock = self._make_toggle(
            "Program panel", initial=False,
            callback=lambda c: self._open_dock_tab(self._program_dock, c))
        win_menu.addAction(self._act_prog_dock)
        # Camera dock (HIDDEN by default) — live D455 + vision-guided control.
        # (visibilityChanged sync wired in _build_camera_dock — dock built after menu.)
        self._act_camera_dock = self._make_toggle(
            "Camera (D455)", initial=False,
            callback=lambda c: self._open_dock_tab(self._camera_dock, c))
        win_menu.addAction(self._act_camera_dock)

        m_view.addSeparator()
        # Reset scene — restore object positions after move/grasp.
        # Placed under View (scene op) rather than a separate Run menu (removed).
        act_reset = QAction("&Reset scene (restore objects)", self)
        act_reset.triggered.connect(self._on_reset_scene)
        m_view.addAction(act_reset)

        # ── ROBOT ── motion + Demo + Parameters + Teach + Connection
        m_robot = mb.addMenu("&Robot")
        for label, cb in (("Home", self._on_home), ("Zero", self._on_zero)):
            act = QAction(label, self); act.triggered.connect(cb)
            m_robot.addAction(act)
            if label == "Home":
                self._act_move_home = act        # tracked for robot-dependent enable
        m_robot.addSeparator()
        # (Removed: Demo motion — dev-test 4-pose hard-coded loop, replaced
        # by Program → Play with a fully user-defined sequence.)
        act_params = QAction("&Parameters (URDF/DH)...", self)
        act_params.triggered.connect(self._show_parameters_dlg)
        m_robot.addAction(act_params)
        self._act_robot_params = act_params       # tracked for robot-dependent enable
        m_robot.addSeparator()
        # Teach on surface — toggle mode to pick the 3D scene to create a target
        self._act_surface_pick = self._make_toggle(
            "Teach on surface (Ctrl+Shift+T)", initial=False,
            callback=self._on_toggle_surface_pick)
        self._act_surface_pick.setShortcut("Ctrl+Shift+T")
        m_robot.addAction(self._act_surface_pick)
        m_robot.addSeparator()
        # HSE connection — dialog has a Test button inside (merged from the old
        # "Test connection" menu entry → one fewer item, better UX).
        act_conn = QAction("&Connection settings... (HSE IP)", self)
        act_conn.triggered.connect(self._on_show_connection_settings)
        m_robot.addAction(act_conn)
        m_robot.addSeparator()
        # Direct control (Phase 1, discrete): send the current jogged pose to the
        # REAL robot via HSE MOVE. ⚠ real motion — REMOTE mode + E-stop required.
        act_send_pose = QAction("⬇ Send current pose to REAL robot (HSE move)", self)
        act_send_pose.triggered.connect(self._on_send_pose_to_robot)
        m_robot.addAction(act_send_pose)
        # Direct control (Phase 2, streaming): while ON, every jog (Cartesian / dial
        # / joint sliders) streams to the REAL robot at ~8 Hz (RoboDK-style online
        # jog). ⚠ continuous real motion — REMOTE mode + E-stop required.
        self._act_live_jog = QAction(
            "◀▶ Live jog → REAL robot (streaming HSE)", self)
        self._act_live_jog.setCheckable(True)
        self._act_live_jog.setChecked(False)
        self._act_live_jog.toggled.connect(self._on_toggle_live_jog)
        m_robot.addAction(self._act_live_jog)

        # ── PROGRAM ── Play / Pause / Stop / Run on Robot + ops
        # Program panel moved to View → Window submenu.
        m_prog = mb.addMenu("&Program")
        act_play = QAction("P&lay", self)
        act_play.triggered.connect(self._on_prog_play)
        m_prog.addAction(act_play)
        # Pause — supplements the dock playback bar for keyboard-only users.
        act_pause = QAction("Pa&use / Resume", self)
        act_pause.triggered.connect(self._on_prog_toggle_pause)
        m_prog.addAction(act_pause)
        act_stop = QAction("&Stop", self)
        # Stop callback = _on_stop_all (dual-purpose: sim playback + servo OFF
        # robot). Same behaviour as the ⏹ Stop button in the dock — avoids the bug
        # where the user thinks menu Stop = full stop but it only stops sim.
        act_stop.triggered.connect(self._on_stop_all)
        m_prog.addAction(act_stop)
        m_prog.addSeparator()
        # Run on Robot — supplements the dock playback bar for keyboard-only users.
        act_run_robot = QAction("&Run on Robot…", self)
        act_run_robot.triggered.connect(self._on_run_on_robot)
        m_prog.addAction(act_run_robot)
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

        # ── DIGITAL TWIN (real robot) ──
        # Start/Stop are buttons inside the Digital Twin panel → menu only opens the panel.
        m_dtwin = mb.addMenu("&Digital Twin")
        act_dt_panel = QAction("Show Digital &Twin panel", self)
        act_dt_panel.triggered.connect(
            lambda: self._open_dock_tab(self._experiment_dock, True))
        m_dtwin.addAction(act_dt_panel)

        # ── HELP ──
        m_help = mb.addMenu("&Help")
        act_about = QAction("&About...", self)
        act_about.triggered.connect(self._show_about)
        m_help.addAction(act_about)

    # NOTE: Separate QToolBar removed. Direct quick-action QActions added
    # to QMenuBar (same level as File/View/Robot/...) via _build_menu_bar().

    def _build_status_bar(self) -> None:
        sb = QStatusBar(self)
        sb.setContentsMargins(8, 0, 8, 0)
        self.setStatusBar(sb)
        # Colored dot badge (level indicator) + text label.
        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet("color: #969696; font-size: 16px;")
        self._status_dot.setFixedWidth(20)
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet("color: #cccccc;")
        sb.addWidget(self._status_dot)
        sb.addWidget(self._status_lbl, 1)
        # Right-side: job + active joint summary chips (read-only info).
        self._status_right_lbl = QLabel("")
        self._status_right_lbl.setStyleSheet(
            "color: #969696; padding: 0 8px;")
        sb.addPermanentWidget(self._status_right_lbl)

    # ── Jog dock (left) — RoboDK-style: Cartesian TOP, Joint BOTTOM ────
    def _build_jog_dock(self) -> None:
        """Left dock — RoboDK panel layout:
          1. Cartesian Jog (Tool combo, Ref combo, 3 pose rows color-coded,
             Translate radio+buttons, Rotate radio+buttons, gripper)
          2. Joint axis jog (Align/Home buttons, 6 sliders)
          3. Other configurations (alternative IK solutions dropdown)
        """
        dock = QDockWidget("Yaskawa GP7 panel", self)
        # FIXED to Left — not Floatable / Movable (Qt 6 bug when re-docking from
        # floating). User shows/hides only via menu or the ✕ button.
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self._jog_dock = dock

        inner = QWidget()
        vbox = QVBoxLayout(inner)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)

        # ──────────────────────────────────────────────────────
        # 1. CARTESIAN JOG  — compact: combo row, pose row, jog row, 2-col grid
        # ──────────────────────────────────────────────────────
        grp_cart = QGroupBox("Cartesian Jog")
        grp_cart.setObjectName("cardGroup")     # title-inside-card (qt_theme)
        cv = QVBoxLayout(grp_cart)
        cv.setContentsMargins(2, 2, 2, 2)
        cv.setSpacing(3)

        # ── Tool + Ref combos + Step (mm/°) all on ONE ROW ──
        # Tool/Ref combos stretch; Step spinbox is fixed-width and compact at
        # the end of the row. All "jog settings" packed into one line → saves height.
        tr_row = QHBoxLayout(); tr_row.setSpacing(4)
        l_tool = QLabel("Tool"); l_tool.setStyleSheet("font-size: 9pt;")
        tr_row.addWidget(l_tool)
        self._tool_combo = QComboBox()
        for name, _T in self._tool_frames: self._tool_combo.addItem(name)
        self._tool_combo.setCurrentIndex(self._tool_idx)
        self._tool_combo.currentIndexChanged.connect(self._on_tool_changed)
        tr_row.addWidget(self._tool_combo, 2)
        l_ref = QLabel("Ref"); l_ref.setStyleSheet("font-size: 9pt;")
        tr_row.addWidget(l_ref)
        self._ref_combo = QComboBox()
        for name, _T in self._ref_frames: self._ref_combo.addItem(name)
        self._ref_combo.setCurrentIndex(self._ref_idx)
        self._ref_combo.currentIndexChanged.connect(self._on_ref_changed)
        tr_row.addWidget(self._ref_combo, 1)
        # Step (jog increment) — same row, fixed width
        ls = QLabel("Step"); ls.setStyleSheet("font-size: 9pt; color: #969696;")
        tr_row.addWidget(ls)
        self._step_mm_spin = QDoubleSpinBox()
        self._step_mm_spin.setRange(0.1, 500.0)
        self._step_mm_spin.setValue(self._jog_step_mm)
        self._step_mm_spin.setSuffix(" mm"); self._step_mm_spin.setFixedWidth(94)
        self._step_mm_spin.setToolTip("Translation step (Translation jog)")
        self._step_mm_spin.valueChanged.connect(
            lambda v: setattr(self, "_jog_step_mm", float(v)))
        tr_row.addWidget(self._step_mm_spin)
        self._step_deg_spin = QDoubleSpinBox()
        self._step_deg_spin.setRange(0.1, 90.0)
        self._step_deg_spin.setValue(self._jog_step_deg)
        self._step_deg_spin.setSuffix(" °"); self._step_deg_spin.setFixedWidth(74)
        self._step_deg_spin.setToolTip("Rotation step (Rotation jog)")
        self._step_deg_spin.valueChanged.connect(
            lambda v: setattr(self, "_jog_step_deg", float(v)))
        tr_row.addWidget(self._step_deg_spin)
        cv.addLayout(tr_row)

        # ── TCP pose — pose row directly, no label header needed ──
        # Pose row is already color-coded X/Y/Z/Rx/Ry/Rz and the "Tool / Ref" combo
        # above already clarifies context → drop the "Tool / Reference (live TCP pose)" label.
        self._tcp_pose_lbls = self._make_colored_pose_row(cv)

        # ── Frame poses (Tool/Flange + Ref/Base) — collapsed, advanced ──
        adv_sec = CollapsibleSection("Frame poses (advanced)", expanded=False)
        adv_sec.add_widget(QLabel("Tool / Flange:"))
        self._tool_pose_lbls = self._make_colored_pose_row(adv_sec.content_layout())
        adv_sec.add_widget(QLabel("Reference / Base:"))
        self._ref_pose_lbls = self._make_colored_pose_row(adv_sec.content_layout())
        cv.addWidget(adv_sec)

        # ── 3-column layout (RoboDK-style): jog control | WorkSpace | Show Frames ──
        # Column 1 = "Jog control" group: ALL jog settings in one place —
        #   frame combo + axis grid (X/Y/Z × Trans/Rot radio) + Step (mm/°) + dial.
        # Step is the JOG INCREMENT so it belongs in the same group (previously it
        # stood alone; the "Step" label was too far from the spinbox due to addStretch).
        # RADIO: 6 radios (Trans X/Y/Z + Rot X/Y/Z) share one exclusive QButtonGroup
        # → user selects one → dial drives that selection.
        cols_row = QHBoxLayout()
        cols_row.setSpacing(6)

        # === Column 1: jog control group ===
        left_col = QVBoxLayout(); left_col.setSpacing(4)
        self._jog_frame_combo = QComboBox()
        for n in self.JOG_FRAMES: self._jog_frame_combo.addItem(n)
        self._jog_frame_combo.setCurrentIndex(self._jog_frame_idx)
        self._jog_frame_combo.currentIndexChanged.connect(
            lambda i: setattr(self, "_jog_frame_idx", int(i)))
        left_col.addWidget(self._jog_frame_combo)

        mode_grid = QGridLayout()
        mode_grid.setHorizontalSpacing(4); mode_grid.setVerticalSpacing(2)
        for c, axis in enumerate(self.AXIS_NAMES):
            lbl = QLabel(axis); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-weight: bold; font-size: 9pt;")
            mode_grid.addWidget(lbl, 0, c + 1)
        l_tr = QLabel("Translation"); l_tr.setStyleSheet("font-size: 9pt;")
        mode_grid.addWidget(l_tr, 1, 0)
        l_rt = QLabel("Rotation"); l_rt.setStyleSheet("font-size: 9pt;")
        mode_grid.addWidget(l_rt, 2, 0)

        # SINGLE shared group for all 6 radios — exclusive across BOTH rows.
        # Button id encoding: 0/1/2 = Trans X/Y/Z, 3/4/5 = Rot X/Y/Z.
        self._jog_axis_group = QButtonGroup(self)

        def _radio_cell(rb: QRadioButton) -> QWidget:
            # Wrapper QWidget to center the radio. Must be transparent — otherwise
            # global QWidget{bg:#1e1e1e} creates a black box behind the radio on the group background.
            cell = QWidget()
            cell.setStyleSheet("background-color: transparent;")
            ch = QHBoxLayout(cell)
            ch.setContentsMargins(0, 0, 0, 0); ch.addStretch()
            ch.addWidget(rb); ch.addStretch()
            return cell

        for i in range(3):
            rb = QRadioButton()
            if i == 0: rb.setChecked(True)                  # default: Translate X
            self._jog_axis_group.addButton(rb, i)
            mode_grid.addWidget(_radio_cell(rb), 1, i + 1)
        for i in range(3):
            rb = QRadioButton()
            self._jog_axis_group.addButton(rb, 3 + i)
            mode_grid.addWidget(_radio_cell(rb), 2, i + 1)
        left_col.addLayout(mode_grid)

        # Jog Dial — ROTARY ENCODER style:
        # • Each notch turned = 1 jog step (axis + sign per selected radio).
        # • Releasing the mouse → dial STAYS in position (no snap to 0).
        # • Continue turning → continue jogging. Wrap=True for unlimited rotation.
        # Tracking: store `_last_dial_value`, each valueChanged computes delta (with
        # wrap-around 360°→0° handling), accumulate, every STEP_THRESHOLD = 1 step.
        self._jog_dial = QDial()
        self._jog_dial.setRange(0, 359)
        self._jog_dial.setValue(0)
        self._jog_dial.setNotchesVisible(True)
        self._jog_dial.setWrapping(True)                  # unlimited rotation
        self._jog_dial.setFixedSize(64, 64)
        self._jog_dial.setToolTip(
            "Turn the dial (like a rotary encoder) → jog one step per notch for the selected radio.\n"
            "Turn right = sign +, turn left = sign −. Release keeps the position.")
        self._last_dial_value = 0
        self._dial_accumulator = 0.0
        self._jog_dial.valueChanged.connect(self._on_dial_value_changed)
        left_col.addWidget(self._jog_dial, 0, Qt.AlignmentFlag.AlignCenter)
        left_col.addStretch()
        cols_row.addLayout(left_col, 0)

        # === Column 2: WorkSpace | Column 3: Show Frames (side-by-side) ===
        cols_row.addWidget(self._build_workspace_group(), 0)
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
            "Align the tool orientation to the reference frame "
            "(keep TCP position, snap wrist perpendicular)")
        self._align_btn.clicked.connect(self._on_align)
        jhdr.addWidget(self._align_btn)
        home_btn = QPushButton("Home"); home_btn.clicked.connect(self._on_home)
        jhdr.addWidget(home_btn)
        joints_sec.add_layout(jhdr)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8); grid.setVerticalSpacing(4)
        self._joint_sliders: list[QSlider] = []
        self._joint_value_lbls: list[QLabel] = []
        # When building the jog dock, the robot may NOT yet be loaded (deferred).
        # Use GP7 datasheet limits as placeholders; after _load_robot_gp7 runs,
        # sliders will apply the correct values (range UNCHANGED since it is the same GP7).
        _GP7_LIMITS_DEG = [(-170, 170), (-65, 145), (-70, 190),
                           (-190, 190), (-135, 135), (-360, 360)]   # GP7 datasheet
        if self._model is not None:
            joints_iter = [(math.degrees(j.joint_min), math.degrees(j.joint_max))
                           for j in self._model.joints]
        else:
            joints_iter = _GP7_LIMITS_DEG
        for i, (jmin, jmax) in enumerate(joints_iter):
            tlbl = QLabel(f"θ{i+1}")
            tlbl.setStyleSheet("font-weight: bold; font-size: 9pt;")
            grid.addWidget(tlbl, i, 0)
            val_lbl = QLabel(f"{self._joints[i]:+7.2f}°")
            val_lbl.setFixedWidth(60)
            val_lbl.setStyleSheet("font-size: 9pt;")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            grid.addWidget(val_lbl, i, 1)
            min_lbl = QLabel(f"{jmin:+.0f}")
            min_lbl.setFixedWidth(32)
            min_lbl.setStyleSheet("font-size: 9pt; color: #969696;")
            min_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            grid.addWidget(min_lbl, i, 2)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(int(jmin * 100), int(jmax * 100))
            slider.setValue(int(self._joints[i] * 100))
            slider.setMinimumWidth(140)                     # compact drag
            slider.valueChanged.connect(
                lambda v, idx=i: self._on_joint_slider(idx, v / 100.0))
            grid.addWidget(slider, i, 3)
            max_lbl = QLabel(f"{jmax:+.0f}")
            max_lbl.setFixedWidth(32)
            max_lbl.setStyleSheet("font-size: 9pt; color: #969696;")
            grid.addWidget(max_lbl, i, 4)
            self._joint_sliders.append(slider)
            self._joint_value_lbls.append(val_lbl)
        joints_sec.add_layout(grid)
        vbox.addWidget(joints_sec)

        # ──────────────────────────────────────────────────────
        # 3. OTHER CONFIGURATIONS — collapsible, CLOSED by default
        # ──────────────────────────────────────────────────────
        other_sec = CollapsibleSection(
            "Other configurations — IK solutions (postures + joint turns)",
            expanded=False)
        oh = QHBoxLayout()
        oh.addWidget(QLabel("(θ1..θ6)"))
        oh.addStretch()
        find_btn = QPushButton("Find branches")
        find_btn.setToolTip(
            "List ALL IK solutions for the current TCP pose (Pieper analytical), "
            "like RoboDK: up to 8 postures (Front/Rear · Up/Down · Flip) × ±360° "
            "joint-turn variants. Switching posture crosses a singularity.")
        find_btn.clicked.connect(self._on_find_alternates)
        oh.addWidget(find_btn)
        cfg_btn = QPushButton("Configurations…")
        cfg_btn.setToolTip("Robot Configurations table (Front/Rear · Elbow "
                            "Up/Down · Flip/Non-Flip)")
        cfg_btn.clicked.connect(self._show_configurations_dlg)
        oh.addWidget(cfg_btn)
        other_sec.add_layout(oh)
        self._alt_combo = QComboBox()
        self._alt_combo.addItem("(no configurations — click \"Find branches\")")
        self._alt_combo.currentIndexChanged.connect(self._on_alternate_selected)
        other_sec.add_widget(self._alt_combo)
        self._alt_solutions: list[list[float]] = []
        vbox.addWidget(other_sec)

        vbox.addStretch()

        # Wrap inner in QScrollArea — panel is very tall (Cartesian + WorkSpace +
        # Show Frames + Joints + Other configs); without scroll, content is
        # clipped at the bottom.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        # Allow horizontal scroll when width is narrow — content overflow remains viewable.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        # minimumWidth = _CELL_DOCK_W (NOT 580) so the shared tab area can shrink
        # to cell width when the cell tab is active. When the jog tab is active,
        # _sync_jog_dock_check resizes the area to _JOG_DOCK_W (580) → 3-col layout
        # wide enough, no scroll. Content has QScrollArea (ScrollBarAsNeeded) so
        # it is safe when squished (hidden behind a tab).
        dock.setMinimumWidth(_CELL_DOCK_W)
        # HIDDEN BY DEFAULT — user enables via menu View > Controls panel.
        dock.setVisible(False)
        dock.visibilityChanged.connect(self._sync_jog_dock_check)
        # Set initial width — resizeDocks only works after window show().
        QTimer.singleShot(0, lambda: self.resizeDocks(
            [dock], [_JOG_DOCK_W], Qt.Orientation.Horizontal))

    # ── WorkSpace + Show Frames groupboxes (used INSIDE Cartesian Jog) ──
    def _build_workspace_group(self) -> QGroupBox:
        grp = QGroupBox("WorkSpace")
        # title bg = #252526 (SURFACE of the parent Cartesian Jog group) to AVOID
        # creating a dark #1e1e1e box behind the text. Nested groups must match the parent background.
        grp.setStyleSheet(
            "QGroupBox { margin-top: 9px; padding: 4px 4px 2px 4px; }"
            "QGroupBox::title { background-color: #252526; }")
        v = QVBoxLayout(grp)
        v.setContentsMargins(2, 2, 2, 2); v.setSpacing(1)
        self._ws_group = QButtonGroup(self)
        opts = [
            ("none",   "Do not show"),
            ("wrist",  "Show for wrist center"),
            ("flange", "Show for robot flange"),
            ("tool",   "Show for current tool"),
        ]
        for i, (key, label) in enumerate(opts):
            rb = QRadioButton(label)
            rb.setStyleSheet("font-size: 9pt;")
            if i == 0: rb.setChecked(True)
            self._ws_group.addButton(rb, i)
            v.addWidget(rb)
        self._ws_group.idClicked.connect(
            lambda i: self._on_workspace_changed(opts[i][0]))
        return grp

    def _build_show_frames_group(self) -> QGroupBox:
        grp = QGroupBox("Show Frames")
        grp.setStyleSheet(
            "QGroupBox { margin-top: 9px; padding: 4px 4px 2px 4px; }"
            "QGroupBox::title { background-color: #252526; }")
        g = QGridLayout(grp)
        g.setContentsMargins(2, 2, 2, 2)
        g.setHorizontalSpacing(4); g.setVerticalSpacing(1)
        self._frame_checks: dict[str, QCheckBox] = {}
        def _mk(text: str) -> QCheckBox:
            cb = QCheckBox(text)
            cb.setStyleSheet("font-size: 9pt;")
            return cb
        # Layout matches the RoboDK reference image:
        #   row0: All/None  | Base (0)
        #   row1: Tool Frame | Robot Flange
        #   row2: Ref. Frame
        #   row3: 1 2 3   row4: 4 5 6   (J1..J6 as 3 columns)
        cb_all = _mk("All/None")
        cb_all.stateChanged.connect(self._on_show_frames_all)
        g.addWidget(cb_all, 0, 0, 1, 2)
        cb_base = _mk("Base (0)")
        cb_base.stateChanged.connect(
            lambda s: self._on_toggle_frame("base", bool(s)))
        g.addWidget(cb_base, 0, 2); self._frame_checks["base"] = cb_base
        cb_tool = _mk("Tool Frame")
        cb_tool.stateChanged.connect(
            lambda s: self._on_toggle_frame("tool", bool(s)))
        g.addWidget(cb_tool, 1, 0, 1, 2); self._frame_checks["tool"] = cb_tool
        cb_fl = _mk("Robot Flange")
        cb_fl.stateChanged.connect(
            lambda s: self._on_toggle_frame("flange", bool(s)))
        g.addWidget(cb_fl, 1, 2); self._frame_checks["flange"] = cb_fl
        cb_ref = _mk("Ref. Frame")
        cb_ref.stateChanged.connect(
            lambda s: self._on_toggle_frame("ref", bool(s)))
        g.addWidget(cb_ref, 2, 0, 1, 2); self._frame_checks["ref"] = cb_ref
        # J1..J6 — 2 rows × 3 cols, narrow digit checkboxes
        for i in range(6):
            key = f"joint_{i+1}"
            cb = _mk(str(i + 1))
            cb.stateChanged.connect(
                lambda s, k=key: self._on_toggle_frame(k, bool(s)))
            g.addWidget(cb, 3 + (i // 3), i % 3)
            self._frame_checks[key] = cb
        return grp

    # ── Pose row helper: 6 colored boxes (RoboDK X/Y/Z/Rx/Ry/Rz pastels) ──
    AXIS_BG_HEX = ("#f4a8b0", "#a8e8a8", "#a8c4f0",
                    "#a8e8e8", "#f0a8f0", "#f0f0a8")
    AXIS_NAMES_FULL = ("X", "Y", "Z", "Rx", "Ry", "Rz")

    # Header text describing pose format (RoboDK style). GP7 = Yaskawa Motoman →
    # Motoman convention. Pre-defined and shared by all 3 pose rows.
    _POSE_FMT_HEADER = "[X,Y,Z] mm  |  Rot[X,Y,Z] deg  ·  Motoman"

    def _make_colored_pose_row(self, parent_layout) -> list[QLabel]:
        """Layout like RoboDK:
          • Header: format label "[X,Y,Z]mm | Rot[X,Y,Z]deg · Motoman"
            + 3 buttons (copy / paste / menu ≡) right-aligned
          • Value row: 6 color-coded cells, values RIGHT-aligned (RoboDK convention)
        """
        container = QVBoxLayout()
        container.setSpacing(2)

        # ── Header row: format label (left) + copy/paste/menu (right) ──
        hdr = QHBoxLayout(); hdr.setSpacing(3)
        fmt_lbl = QLabel(self._POSE_FMT_HEADER)
        fmt_lbl.setStyleSheet(
            "color: #969696; font-size: 8pt; background-color: transparent;")
        hdr.addWidget(fmt_lbl)
        hdr.addStretch()

        _btn_style = "QPushButton { padding: 2px; }"
        cpy = QPushButton(); cpy.setIcon(_draw_copy_icon())
        cpy.setIconSize(QSize(14, 14)); cpy.setFixedWidth(26)
        cpy.setStyleSheet(_btn_style); cpy.setToolTip("Copy pose to clipboard")
        pst = QPushButton(); pst.setIcon(_draw_paste_icon())
        pst.setIconSize(QSize(14, 14)); pst.setFixedWidth(26)
        pst.setStyleSheet(_btn_style)
        pst.setToolTip("Paste pose from clipboard (TCP target)")
        mnu = QPushButton(); mnu.setIcon(_draw_menu_icon())
        mnu.setIconSize(QSize(14, 14)); mnu.setFixedWidth(26)
        mnu.setStyleSheet(_btn_style)
        mnu.setToolTip("More pose options (copy as list / JSON)")
        hdr.addWidget(cpy); hdr.addWidget(pst); hdr.addWidget(mnu)
        container.addLayout(hdr)

        # ── Value row: 6 colored cells, values right-aligned ──
        row = QHBoxLayout()
        row.setSpacing(1)
        labels = []
        for bg in self.AXIS_BG_HEX:
            lbl = QLabel("0.000")
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight
                              | Qt.AlignmentFlag.AlignVCenter)
            lbl.setMinimumWidth(46)
            lbl.setStyleSheet(
                f"background-color: {bg}; color: #000; "
                f"padding: 3px 6px 3px 2px; border: 1px solid #555;"
                f"font-family: Consolas, 'Courier New', monospace;"
                f"font-size: 10px; font-weight: 600;"
            )
            row.addWidget(lbl, 1)
            labels.append(lbl)
        container.addLayout(row)

        parent_layout.addLayout(container)
        cpy.clicked.connect(lambda: self._copy_pose_to_clipboard(labels))
        pst.clicked.connect(self._paste_pose_from_clipboard)
        mnu.clicked.connect(lambda: self._show_pose_menu(mnu, labels))
        return labels

    def _show_pose_menu(self, anchor: QPushButton,
                         labels: list[QLabel]) -> None:
        """Popup menu (≡) — copy/paste + export pose as list / JSON."""
        m = QMenu(self)
        m.addAction("Copy values",
                    lambda: self._copy_pose_to_clipboard(labels))
        m.addAction("Paste values", self._paste_pose_from_clipboard)
        m.addSeparator()
        m.addAction("Copy as Python list",
                    lambda: self._copy_pose_as(labels, "py"))
        m.addAction("Copy as JSON",
                    lambda: self._copy_pose_as(labels, "json"))
        m.exec(anchor.mapToGlobal(anchor.rect().bottomLeft()))

    def _copy_pose_as(self, labels: list[QLabel], fmt: str) -> None:
        """Export 6 pose values in `py` (list) or `json` format."""
        vals = []
        for lbl in labels:
            try:
                vals.append(float(lbl.text()))
            except ValueError:
                vals.append(0.0)
        if fmt == "py":
            s = "[" + ", ".join(f"{v:.3f}" for v in vals) + "]"
        else:                                          # json
            s = json.dumps(
                {"xyz_mm": vals[:3], "rpy_deg": vals[3:]})
        QApplication.clipboard().setText(s)
        self._set_status(f"Pose copied ({fmt})", level="ok")

    def _copy_pose_to_clipboard(self, labels: list[QLabel]) -> None:
        """Copy 6 raw values from labels (no axis prefix) to clipboard."""
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
        """Read 'X,Y,Z,Rx,Ry,Rz' from clipboard → IK → set TCP target.
        Interpretation: pose is Tool / Reference (same as the bottom pose row).
        """
        text = QApplication.clipboard().text().strip()
        parts = [p.strip() for p in text.replace(";", ",").split(",")]
        if len(parts) != 6:
            self._set_status(
                f"Clipboard '{text[:40]}' is not a 6-value pose", level="err")
            return
        try:
            x, y, z, rx, ry, rz = (float(p) for p in parts)
        except ValueError:
            self._set_status("Cannot parse clipboard as pose values", level="err")
            return
        # Pose w.r.t. Reference → world frame TCP target
        T_ref_tool = _xyz_rpy_to_matrix(x, y, z, rx, ry, rz)
        # The "Base (0)" readout is rendered in the YRC pendant convention (tool
        # orientation · Rz180), so a value COPIED from it carries that Rz180.
        # Undo it here (Rz180 is its own inverse) before building the target, or
        # paste would command a 180° tool-Z reorientation. Position is unaffected.
        if self._ref_frames[self._ref_idx][0] == "Base (0)":
            Rz180 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
            T_ref_tool[:3, :3] = T_ref_tool[:3, :3] @ Rz180
        T_world_base = np.eye(4); T_world_base[:3, 3] = self._base_xyz
        T_world_ref = T_world_base @ self._ref_frames[self._ref_idx][1]
        T_world_tool = T_world_ref @ T_ref_tool
        self._apply_cartesian_target(T_world_tool, "Paste pose")

    # ── Program dock (left, tabified with Jog + Cell) ──────────────────
    def _build_program_dock(self) -> None:
        """Layout: program list (top, primary) → edit toolbar → Targets group →
        Add tabs (Motion / Logic / Modal) → Playback bar → File bar.
        Workflow top-down: Teach targets → Add instructions → Run → Save.

        Dock is in the SAME left area, tabified with Jog + Cell (1 tab group)
        instead of a separate right panel."""
        dock = QDockWidget("Program", self)
        # FIXED to Left — tabified with jog/cell, non-floatable, predictable layout.
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
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
        # Hand-drawn icons (QPainter) — ⟲/✕ glyphs don't render in Segoe UI so
        # buttons were previously blank. Consistent with copy/paste/menu icons.
        for icon, cb, tip in (
            (_draw_open_icon(),   self._on_prog_open_file_dlg,
             "Open job code file (.JBI / .json)"),
            (_draw_plus_icon(),   self._on_job_add,    "Add new job"),
            (_draw_rename_icon(), self._on_job_rename, "Rename current job"),
            (_draw_trash_icon(),  self._on_job_delete, "Delete current job"),
        ):
            b = QPushButton(); b.setIcon(icon); b.setToolTip(tip); b.setFixedWidth(28)
            b.clicked.connect(cb)
            job_row.addWidget(b)
        jw = QWidget(); jw.setLayout(job_row)
        vbox.addWidget(jw)

        # ══ 1. PROGRAM LIST (primary surface, stretch=3) ═════════════
        self._prog_list = QListWidget()
        self._prog_list.setMinimumHeight(180)
        vbox.addWidget(self._prog_list, 3)

        # ── 1b. Edit toolbar — icon buttons for move/delete (↑↓✕ glyphs don't
        # render in Segoe UI), text buttons for Edit/Replace. Edit & Replace
        # stretch to fill so their labels are never clipped at narrow dock widths.
        edit_row = QHBoxLayout(); edit_row.setSpacing(4)
        edit_row.setContentsMargins(0, 0, 0, 0)
        for icon, cb, tip in (
            (_draw_arrow_up_icon(),   self._on_prog_move_up,   "Move selected instruction up"),
            (_draw_arrow_down_icon(), self._on_prog_move_down, "Move selected instruction down"),
        ):
            b = QPushButton(); b.setIcon(icon); b.setToolTip(tip)
            b.setFixedWidth(38); b.clicked.connect(cb)
            edit_row.addWidget(b)
        b_edit = QPushButton("Edit")
        b_edit.setToolTip("F2 — edit the selected instruction's parameters "
                          "(or double-click the list)")
        b_edit.setMinimumWidth(60); b_edit.clicked.connect(self._on_prog_modify)
        edit_row.addWidget(b_edit, 1)               # stretch
        b_repl = QPushButton("Replace")
        b_repl.setToolTip("Change the selected instruction to a different type "
                          "(keeps its position), then edit its parameters")
        b_repl.setMinimumWidth(72); b_repl.clicked.connect(self._on_prog_replace)
        edit_row.addWidget(b_repl, 1)               # stretch
        b_pdel = QPushButton(); b_pdel.setIcon(_draw_x_icon())
        b_pdel.setToolTip("Delete the selected instruction")
        b_pdel.setFixedWidth(38); b_pdel.clicked.connect(self._on_prog_delete)
        edit_row.addWidget(b_pdel)
        eb = QWidget(); eb.setLayout(edit_row)
        vbox.addWidget(eb)
        # Double-click instruction in list → Edit
        self._prog_list.itemDoubleClicked.connect(
            lambda *_: self._on_prog_modify())
        # F2 app-wide shortcut → modify selected instruction
        QShortcut(QKeySequence("F2"), self,
                   activated=self._on_prog_modify)

        # ══ 2. TARGETS group (manage targets — teach/modify/goto/config) ═
        tgt_grp = QGroupBox("Targets")
        tgt_grp.setObjectName("cardGroup")      # title-inside-card (qt_theme)
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
        # Manage row: Modify / Delete / Go to / Config (4 equal buttons, full-width).
        # (+ MoveJ→/MoveL→ moved to Motion tab to group all "create move" actions in one place.)
        tact_row = QHBoxLayout(); tact_row.setSpacing(4)
        b_mod = QPushButton("Modify"); b_mod.setShortcut("F3")
        b_mod.setToolTip("F3 — replace selected target with current pose")
        b_mod.clicked.connect(self._on_tgt_modify)
        b_del = QPushButton("Delete"); b_del.setToolTip("Delete selected target")
        b_del.clicked.connect(self._on_tgt_delete)
        b_goto = QPushButton("Go to")
        b_goto.setToolTip("Animate robot to selected target (preview)")
        b_goto.clicked.connect(self._on_tgt_goto)
        b_cfg = QPushButton("Config"); b_cfg.setShortcut("F4")
        b_cfg.setToolTip("F4 — pick alternative IK configuration for selected target")
        b_cfg.clicked.connect(self._on_tgt_change_config)
        for b in (b_mod, b_del, b_goto, b_cfg):
            tact_row.addWidget(b)
        taw = QWidget(); taw.setLayout(tact_row)
        tgt_lay.addWidget(taw)
        vbox.addWidget(tgt_grp)

        # ══ 3. ADD INSTRUCTION (Motion / I-O & Flow / Modal) ═════════
        # Every row uses a 3-column grid [label | inputs(stretch) | + button(fixed)]
        # → "+ Add" buttons align vertically, clean & professional. Helper below.
        _BTN_W = 138        # wide enough for the longest label (+ WaitIO / + SetVar) — no clipping
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        # Maximum vertical policy → tabs widget height = content only, does NOT expand
        # to fill. Extra space goes to the program list (stretch=3, primary) instead
        # of blank space below the Motion tab. (Previously tabs expanded → large gap.)
        tabs.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        def _hwrap(*widgets, fill=False):
            """Pack small inputs into one widget. fill=True: LAST widget stretches to fill
            the column (no gap between input and button). fill=False: stretch pushes
            inputs to the left."""
            w = QWidget(); h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0); h.setSpacing(3)
            last = len(widgets) - 1
            for i, x in enumerate(widgets):
                h.addWidget(x, 1 if (fill and i == last) else 0)
            if not fill:
                h.addStretch()
            return w

        def _btn_fill_row(*btns):
            """Row of buttons that stretch evenly to fill the full width (each button stretch=1)
            — no packing left with empty space on the right."""
            w = QWidget(); h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0); h.setSpacing(4)
            for b in btns:
                h.addWidget(b, 1)
            return w

        # Label col fixed (aligns labels), input col STRETCH fills remaining
        # width → NO empty right margin at any dock width; inputs fill
        # col1 (fill=True) so also no gap. Button col fixed → +Add aligned in column.
        # Wide enough for the longest label ("Pulse OT#", "DOUT OG#") — narrower
        # values clipped those labels at 52px.
        _LABEL_W = 72

        def _grid_row(grid, r, label, mid, btn):
            """label@col0 (fixed) | mid@col1 (stretch, fill) | btn@col2 (fixed)."""
            lbl = QLabel(label); lbl.setFixedWidth(_LABEL_W)
            grid.addWidget(lbl, r, 0)
            if mid is not None:
                grid.addWidget(mid, r, 1)
            btn.setFixedWidth(_BTN_W)
            grid.addWidget(btn, r, 2)

        # ─ Tab: Motion (inline + to-target moves) ─
        mot_w = QWidget()
        mot_lay = QGridLayout(mot_w); mot_lay.setSpacing(4)
        mot_lay.setColumnStretch(0, 1); mot_lay.setColumnStretch(1, 1)
        b_mj = QPushButton("+ MoveJ"); b_mj.setToolTip("Joint move to current pose")
        b_mj.clicked.connect(self._on_prog_add_movej)
        b_ml = QPushButton("+ MoveL"); b_ml.setToolTip("Linear move to current pose")
        b_ml.clicked.connect(self._on_prog_add_movel)
        self._btn_movec = QPushButton("+ MoveC (set MID)")
        self._btn_movec.setToolTip("2-step: click 1 = MID waypoint, click 2 = END")
        self._btn_movec.clicked.connect(self._on_prog_add_movec)
        self._pending_movc_mid: list[float] | None = None
        mot_lay.addWidget(QLabel("At current pose:"), 0, 0, 1, 2)
        mot_lay.addWidget(b_mj,            1, 0)
        mot_lay.addWidget(b_ml,            1, 1)
        mot_lay.addWidget(self._btn_movec, 2, 0, 1, 2)
        _line = QFrame(); _line.setFrameShape(QFrame.Shape.HLine)
        _line.setFrameShadow(QFrame.Shadow.Sunken)
        mot_lay.addWidget(_line, 3, 0, 1, 2)
        mot_lay.addWidget(QLabel("To selected target:"), 4, 0, 1, 2)
        b_uj = QPushButton("+ MoveJ →")
        b_uj.setToolTip("Add MoveJ → selected target in Targets list")
        b_uj.clicked.connect(lambda: self._on_prog_add_move_to_target("MoveJ"))
        b_ul = QPushButton("+ MoveL →")
        b_ul.setToolTip("Add MoveL → selected target in Targets list")
        b_ul.clicked.connect(lambda: self._on_prog_add_move_to_target("MoveL"))
        mot_lay.addWidget(b_uj, 5, 0); mot_lay.addWidget(b_ul, 5, 1)
        mot_lay.setRowStretch(6, 1)
        tabs.addTab(mot_w, "Motion")

        # ─ Tab: I/O & Flow (DOUT / WaitIO / Wait / MSG / Call / SimEvent) ─
        log_w = QWidget()
        log_lay = QGridLayout(log_w); log_lay.setSpacing(4)
        # col1 (input) stretch → fills full width, no empty right margin.
        log_lay.setColumnStretch(1, 1)
        r = 0
        # DOUT — generic digital output (REPLACING gripper-specific). General-purpose
        # software: control any output bit (gripper/valve/clamp…).
        self._prog_do_idx = QSpinBox(); self._prog_do_idx.setRange(1, 1024)
        self._prog_do_idx.setValue(1); self._prog_do_idx.setFixedWidth(64)
        self._prog_do_state = QComboBox(); self._prog_do_state.addItems(["ON", "OFF"])
        self._prog_do_state.setMinimumWidth(64)   # wide enough for "OFF" + arrow
        b_do = QPushButton("+ DOUT")
        b_do.setToolTip("Set digital output bit (DOUT OT#n = ON/OFF)")
        b_do.clicked.connect(self._on_prog_add_setdo)
        _grid_row(log_lay, r, "OUT#",
                  _hwrap(self._prog_do_idx, QLabel("="), self._prog_do_state, fill=True),
                  b_do); r += 1
        # WaitIO (input)
        self._prog_io_idx = QSpinBox(); self._prog_io_idx.setRange(1, 1024)
        self._prog_io_idx.setValue(1); self._prog_io_idx.setFixedWidth(64)
        self._prog_io_state = QComboBox(); self._prog_io_state.addItems(["ON", "OFF"])
        self._prog_io_state.setMinimumWidth(64)   # wide enough for "OFF" + arrow
        self._prog_io_tout = QDoubleSpinBox()
        self._prog_io_tout.setRange(0.0, 600.0); self._prog_io_tout.setValue(0.0)
        self._prog_io_tout.setSuffix("s"); self._prog_io_tout.setMinimumWidth(64)
        self._prog_io_tout.setToolTip("0 = block forever")
        b_wio = QPushButton("+ WaitIO"); b_wio.clicked.connect(self._on_prog_add_waitio)
        b_wio.setToolTip("Wait until input IN#n = ON/OFF (timeout 0 = block forever)")
        # idx/=/state packed left; addStretch pushes the "T + timeout" group right →
        # creates a clear gap to the left of the "T" label (instead of T touching the combo).
        _io_mid = QWidget(); _io_h = QHBoxLayout(_io_mid)
        _io_h.setContentsMargins(0, 0, 0, 0); _io_h.setSpacing(3)
        _io_h.addWidget(self._prog_io_idx)
        _io_h.addWidget(QLabel("="))
        _io_h.addWidget(self._prog_io_state)
        _io_h.addStretch(1)                            # gap to the left of "T"
        _io_h.addWidget(QLabel("T"))
        _io_h.addWidget(self._prog_io_tout)
        _grid_row(log_lay, r, "IN#", _io_mid, b_wio); r += 1
        # Wait (timer)
        self._prog_wait_spin = QDoubleSpinBox()
        self._prog_wait_spin.setRange(0.0, 600.0); self._prog_wait_spin.setValue(0.5)
        self._prog_wait_spin.setSuffix(" s"); self._prog_wait_spin.setMinimumWidth(80)
        b_wait = QPushButton("+ Wait"); b_wait.clicked.connect(self._on_prog_add_wait)
        b_wait.setToolTip("Pause program for N seconds (INFORM TIMER)")
        _grid_row(log_lay, r, "Wait", _hwrap(self._prog_wait_spin, fill=True), b_wait); r += 1
        # MSG
        self._prog_msg_edit = QLineEdit()
        self._prog_msg_edit.setMaxLength(32)
        self._prog_msg_edit.setPlaceholderText("≤ 32 ASCII")
        b_msg = QPushButton("+ MSG"); b_msg.clicked.connect(self._on_prog_add_msg)
        b_msg.setToolTip("Show a message on the teach pendant (INFORM MSG)")
        _grid_row(log_lay, r, "MSG", self._prog_msg_edit, b_msg); r += 1
        # CallJob (sub-program)
        self._prog_call_edit = QLineEdit()
        self._prog_call_edit.setMaxLength(32)
        self._prog_call_edit.setPlaceholderText("sub-job name (e.g. WELD_A)")
        b_call = QPushButton("+ Call"); b_call.clicked.connect(self._on_prog_add_calljob)
        b_call.setToolTip("Call another job as sub-program (INFORM CALL JOB)")
        _grid_row(log_lay, r, "Call", self._prog_call_edit, b_call); r += 1
        # SimEvent (sim-only checkpoint)
        self._prog_ev_edit = QLineEdit()
        self._prog_ev_edit.setMaxLength(32)
        self._prog_ev_edit.setPlaceholderText("sim checkpoint name")
        b_ev = QPushButton("+ Event")
        b_ev.setToolTip("Sim checkpoint (SimEvent) — not exported to .JBI")
        b_ev.clicked.connect(self._on_prog_add_simevent)
        _grid_row(log_lay, r, "Event", self._prog_ev_edit, b_ev); r += 1
        log_lay.setRowStretch(r, 1)
        tabs.addTab(log_w, "I/O && Flow")

        # ─ Tab: Modal (Speed / Rounding / Tool / RefFrame) ─
        mod_w = QWidget()
        mod_lay = QGridLayout(mod_w); mod_lay.setSpacing(4)
        # Same column config as the I/O & Flow tab: col1 (input) stretch fills all.
        mod_lay.setColumnStretch(1, 1)
        r = 0
        # Speed row has 2 inputs (VJ% joint + V mm/s linear). Both spinboxes keep
        # natural width (no stretch); only addStretch(1) between VJ and "V" absorbs
        # extra space → all the gap goes to the left of the "V" label (large gap,
        # clearly separating the two visual groups) instead of stretching the V spinbox.
        self._prog_spd_vj = QDoubleSpinBox()
        self._prog_spd_vj.setRange(1.0, 30.0); self._prog_spd_vj.setValue(10.0)
        self._prog_spd_vj.setDecimals(2); self._prog_spd_vj.setSuffix(" %")
        self._prog_spd_vj.setMinimumWidth(100); self._prog_spd_vj.setMaximumWidth(120)
        self._prog_spd_v = QDoubleSpinBox()
        self._prog_spd_v.setRange(1.0, 250.0); self._prog_spd_v.setValue(100.0)
        self._prog_spd_v.setDecimals(1); self._prog_spd_v.setSuffix(" mm/s")
        self._prog_spd_v.setMinimumWidth(100)
        _spd_mid = QWidget(); _spd_h = QHBoxLayout(_spd_mid)
        _spd_h.setContentsMargins(0, 0, 0, 0); _spd_h.setSpacing(4)
        _spd_h.addWidget(QLabel("VJ"))
        _spd_h.addWidget(self._prog_spd_vj)            # no stretch → ~100px
        _spd_h.addStretch(1)                            # gap between VJ and "V"
        _spd_h.addWidget(QLabel("V"))
        _spd_h.addWidget(self._prog_spd_v)             # natural width; gap is to the left of "V"
        b_spd = QPushButton("+ Speed"); b_spd.clicked.connect(self._on_prog_add_setspeed)
        b_spd.setToolTip("Set speed VJ% (joint) + V mm/s (linear) for following moves")
        _grid_row(mod_lay, r, "Speed", _spd_mid, b_spd); r += 1
        self._prog_pl = QSpinBox(); self._prog_pl.setRange(0, 8); self._prog_pl.setValue(0)
        self._prog_pl.setMinimumWidth(64)
        b_pl = QPushButton("+ Round"); b_pl.clicked.connect(self._on_prog_add_setrounding)
        b_pl.setToolTip("Corner rounding PL (0 = sharp/exact … 8 = smooth/fast)")
        _grid_row(mod_lay, r, "PL", _hwrap(self._prog_pl, fill=True), b_pl); r += 1
        self._prog_tool_no = QSpinBox(); self._prog_tool_no.setRange(0, 15)
        self._prog_tool_no.setValue(0); self._prog_tool_no.setMinimumWidth(64)
        b_tool = QPushButton("+ Tool"); b_tool.clicked.connect(self._on_prog_add_settool)
        b_tool.setToolTip("Select tool coordinate TL# for following moves")
        _grid_row(mod_lay, r, "TL#", _hwrap(self._prog_tool_no, fill=True), b_tool); r += 1
        self._prog_uf_no = QSpinBox(); self._prog_uf_no.setRange(0, 15)
        self._prog_uf_no.setValue(0); self._prog_uf_no.setMinimumWidth(64)
        b_uf = QPushButton("+ Frame"); b_uf.clicked.connect(self._on_prog_add_setrefframe)
        b_uf.setToolTip("Select user/reference frame UF# for following moves")
        _grid_row(mod_lay, r, "UF#", _hwrap(self._prog_uf_no, fill=True), b_uf); r += 1
        mod_lay.setRowStretch(r, 1)
        tabs.addTab(mod_w, "Modal")

        # ─ Tab: Logic (flow control + variables — INFORM JUMP/LABEL/SET) ─
        lgc_w = QWidget()
        lgc_lay = QGridLayout(lgc_w); lgc_lay.setSpacing(4)
        lgc_lay.setColumnStretch(1, 1)
        r = 0
        # Label (*LABEL) — jump target
        self._prog_lbl_edit = QLineEdit(); self._prog_lbl_edit.setMaxLength(32)
        self._prog_lbl_edit.setPlaceholderText("label name (e.g. LOOP)")
        b_lbl = QPushButton("+ Label")
        b_lbl.setToolTip("Jump target: *LABEL")
        b_lbl.clicked.connect(self._on_prog_add_label)
        _grid_row(lgc_lay, r, "Label", _hwrap(self._prog_lbl_edit, fill=True), b_lbl); r += 1
        # Jump *LABEL [IF cond]
        self._prog_jmp_edit = QLineEdit(); self._prog_jmp_edit.setMaxLength(32)
        self._prog_jmp_edit.setPlaceholderText("target label")
        b_jmp = QPushButton("+ Jump")
        b_jmp.setToolTip("JUMP *LABEL — optionally only IF the condition below holds")
        b_jmp.clicked.connect(self._on_prog_add_jump)
        _grid_row(lgc_lay, r, "Jump *", _hwrap(self._prog_jmp_edit, fill=True), b_jmp); r += 1
        # Condition row (applied to the Jump above). "(uncond)" = unconditional jump.
        self._prog_jc_lhs = QLineEdit(); self._prog_jc_lhs.setPlaceholderText("B000 / IN#(1)")
        self._prog_jc_lhs.setMinimumWidth(78)
        self._prog_jc_op = QComboBox()
        self._prog_jc_op.addItems(["(uncond)", "=", "<>", ">", "<", ">=", "<="])
        self._prog_jc_op.setMinimumWidth(74)
        self._prog_jc_rhs = QLineEdit(); self._prog_jc_rhs.setPlaceholderText("value / var")
        self._prog_jc_rhs.setMinimumWidth(78)
        lgc_lay.addWidget(QLabel("IF"), r, 0)
        lgc_lay.addWidget(
            _hwrap(self._prog_jc_lhs, self._prog_jc_op, self._prog_jc_rhs, fill=True),
            r, 1, 1, 2)
        r += 1
        # SetVar: SET/ADD/SUB/MUL/DIV Bxxx arg | INC/DEC Bxxx
        self._prog_var_name = QLineEdit(); self._prog_var_name.setMaxLength(5)
        self._prog_var_name.setPlaceholderText("B000"); self._prog_var_name.setMinimumWidth(64)
        self._prog_var_op = QComboBox()
        self._prog_var_op.addItems(["SET", "ADD", "SUB", "MUL", "DIV", "INC", "DEC"])
        self._prog_var_op.setMinimumWidth(64)
        self._prog_var_arg = QLineEdit(); self._prog_var_arg.setPlaceholderText("value / var")
        self._prog_var_arg.setMinimumWidth(70)
        b_var = QPushButton("+ SetVar")
        b_var.setToolTip("Variable op: SET/ADD/SUB/MUL/DIV need an operand; INC/DEC don't")
        b_var.clicked.connect(self._on_prog_add_setvar)
        _grid_row(
            lgc_lay, r, "Var",
            _hwrap(self._prog_var_name, self._prog_var_op, self._prog_var_arg, fill=True),
            b_var); r += 1
        # ── Structured blocks (IFTHEN/ELSEIF/ELSE/ENDIF + WHILE/ENDWHILE) ──
        _sep = QFrame(); _sep.setFrameShape(QFrame.Shape.HLine)
        _sep.setFrameShadow(QFrame.Shadow.Sunken)
        lgc_lay.addWidget(_sep, r, 0, 1, 3); r += 1
        lgc_lay.addWidget(QLabel("Structured blocks:"), r, 0, 1, 3); r += 1
        # Shared condition for IfThen / ElseIf / While (always requires a condition).
        self._prog_sc_lhs = QLineEdit(); self._prog_sc_lhs.setPlaceholderText("B000 / IN#(1)")
        self._prog_sc_lhs.setMinimumWidth(78)
        self._prog_sc_op = QComboBox()
        self._prog_sc_op.addItems(["=", "<>", ">", "<", ">=", "<="])
        self._prog_sc_op.setMinimumWidth(64)
        self._prog_sc_rhs = QLineEdit(); self._prog_sc_rhs.setPlaceholderText("value / var")
        self._prog_sc_rhs.setMinimumWidth(78)
        lgc_lay.addWidget(QLabel("Cond"), r, 0)
        lgc_lay.addWidget(
            _hwrap(self._prog_sc_lhs, self._prog_sc_op, self._prog_sc_rhs, fill=True),
            r, 1, 1, 2)
        r += 1
        # Row: buttons that require a condition (reads Cond above).
        b_if = QPushButton("+ IfThen"); b_if.clicked.connect(self._on_prog_add_ifthen)
        b_if.setToolTip("IFTHEN <cond> … (closed by ENDIF)")
        b_elif = QPushButton("+ ElseIf"); b_elif.clicked.connect(self._on_prog_add_elseif)
        b_elif.setToolTip("ELSEIF <cond> — next conditional branch (inside IFTHEN)")
        b_while = QPushButton("+ While"); b_while.clicked.connect(self._on_prog_add_while)
        b_while.setToolTip("WHILE <cond> … (closed by ENDWHILE)")
        lgc_lay.addWidget(_btn_fill_row(b_if, b_elif, b_while), r, 0, 1, 3); r += 1
        # Row: closing/unconditioned buttons.
        b_else = QPushButton("+ Else"); b_else.clicked.connect(self._on_prog_add_else)
        b_endif = QPushButton("+ EndIf"); b_endif.clicked.connect(self._on_prog_add_endif)
        b_endw = QPushButton("+ EndWhile"); b_endw.clicked.connect(self._on_prog_add_endwhile)
        lgc_lay.addWidget(_btn_fill_row(b_else, b_endif, b_endw), r, 0, 1, 3); r += 1
        _hint = QLabel("Operands: B###/I### variable, integer literal, or IN#(n). "
                       "IF/WHILE blocks must be balanced (validated before Run/Export).")
        _hint.setWordWrap(True); _hint.setStyleSheet("color: #8a8a8a; font-size: 11px;")
        lgc_lay.addWidget(_hint, r, 0, 1, 3); r += 1

        # ── I/O & registers (extended INFORM: PULSE / CLEAR / DIN / DOUT OG#) ──
        _sep2 = QFrame(); _sep2.setFrameShape(QFrame.Shape.HLine)
        _sep2.setFrameShadow(QFrame.Shadow.Sunken)
        lgc_lay.addWidget(_sep2, r, 0, 1, 3); r += 1
        lgc_lay.addWidget(QLabel("I/O & registers:"), r, 0, 1, 3); r += 1
        # PULSE OT#(n)
        self._prog_pulse_idx = QSpinBox(); self._prog_pulse_idx.setRange(1, 1024)
        self._prog_pulse_idx.setValue(1)
        b_pulse = QPushButton("+ Pulse")
        b_pulse.setToolTip("PULSE OT#(n) — momentary output pulse")
        b_pulse.clicked.connect(self._on_prog_add_pulse)
        _grid_row(lgc_lay, r, "Pulse OT#", _hwrap(self._prog_pulse_idx, fill=True),
                  b_pulse); r += 1
        # CLEAR Ixxx n|ALL
        self._prog_clear_var = QLineEdit(); self._prog_clear_var.setMaxLength(5)
        self._prog_clear_var.setPlaceholderText("I010")
        self._prog_clear_cnt = QLineEdit(); self._prog_clear_cnt.setPlaceholderText("2 / ALL")
        b_clear = QPushButton("+ Clear")
        b_clear.setToolTip("CLEAR Ixxx n | CLEAR Ixxx ALL — zero consecutive registers")
        b_clear.clicked.connect(self._on_prog_add_clearvar)
        _grid_row(lgc_lay, r, "Clear",
                  _hwrap(self._prog_clear_var, self._prog_clear_cnt, fill=True),
                  b_clear); r += 1
        b_cstk = QPushButton("+ Clear STACK")
        b_cstk.setToolTip("CLEAR STACK — clear the call stack")
        b_cstk.clicked.connect(self._on_prog_add_clearstack)
        lgc_lay.addWidget(_btn_fill_row(b_cstk), r, 0, 1, 3); r += 1
        # DIN Bxxx IG#(n)/SOUT#(n)
        self._prog_din_var = QLineEdit(); self._prog_din_var.setMaxLength(5)
        self._prog_din_var.setPlaceholderText("B005")
        self._prog_din_kind = QComboBox(); self._prog_din_kind.addItems(["IG", "SOUT"])
        self._prog_din_grp = QSpinBox(); self._prog_din_grp.setRange(0, 4095)
        b_din = QPushButton("+ DIN")
        b_din.setToolTip("DIN Bxxx IG#(n)/SOUT#(n) — read input/status group into a register")
        b_din.clicked.connect(self._on_prog_add_din)
        _grid_row(lgc_lay, r, "DIN",
                  _hwrap(self._prog_din_var, self._prog_din_kind, self._prog_din_grp,
                         fill=True), b_din); r += 1
        # DOUT OG#(n) Bxxx
        self._prog_dog_grp = QSpinBox(); self._prog_dog_grp.setRange(0, 4095)
        self._prog_dog_var = QLineEdit(); self._prog_dog_var.setMaxLength(5)
        self._prog_dog_var.setPlaceholderText("B005")
        b_dog = QPushButton("+ DOUT OG#")
        b_dog.setToolTip("DOUT OG#(n) Bxxx — write a register to an output group")
        b_dog.clicked.connect(self._on_prog_add_doutgroup)
        _grid_row(lgc_lay, r, "DOUT OG#",
                  _hwrap(self._prog_dog_grp, self._prog_dog_var, fill=True),
                  b_dog); r += 1

        lgc_lay.setRowStretch(r, 1)
        tabs.addTab(lgc_w, "Logic")

        # Center values in all spinboxes — because inputs fill col1 (wide),
        # centering looks balanced instead of left-aligned with empty space on the right.
        for _sp in (self._prog_do_idx, self._prog_io_idx, self._prog_io_tout,
                    self._prog_wait_spin, self._prog_spd_vj, self._prog_spd_v,
                    self._prog_pl, self._prog_tool_no, self._prog_uf_no):
            _sp.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Tab area shrinks to the content of the CURRENT tab (not as tall as the
        # tallest tab) → no blank space when on a short tab (e.g. Motion). Hidden
        # pages use Ignored policy so they don't push the QTabWidget sizeHint up to max.
        self._prog_tabs = tabs
        tabs.currentChanged.connect(self._on_prog_tab_changed)
        self._on_prog_tab_changed(tabs.currentIndex())

        vbox.addWidget(tabs)

        # ══ 4. PLAYBACK bar — 2 rows separating Sim ↔ Real robot ══════════════
        # Row 1 = SIMULATION controls (safe): Run Sim / Pause / Stop +
        # speed. Row 2 = REAL ROBOT (dangerous, intentionally separated):
        # Run on Robot full-width. All buttons have full labels + tooltips.
        _GREEN = ("QPushButton { background-color: #2da44e; color: white; "
                  "font-weight: bold; border-radius: 4px; padding: 4px 8px; }"
                  "QPushButton:hover { background-color: #2c974b; }")
        _RED = ("QPushButton { background-color: #cf222e; color: white; "
                "font-weight: bold; border-radius: 4px; padding: 4px 8px; }"
                "QPushButton:hover { background-color: #a40e26; }")
        _ORANGE = ("QPushButton { background-color: #fb8500; color: white; "
                   "font-weight: bold; border-radius: 4px; padding: 6px 8px; }"
                   "QPushButton:hover { background-color: #d96e00; }")

        # ─ Row 1: Sim controls ─
        sim_row = QHBoxLayout(); sim_row.setSpacing(6)
        b_play = QPushButton("▶  Run Sim")
        b_play.setMinimumHeight(34); b_play.setStyleSheet(_GREEN)
        b_play.setToolTip("Run the program in simulation (no real robot moves)")
        b_play.clicked.connect(self._on_prog_play)
        self._btn_pause = QPushButton("▮▮  Pause")
        self._btn_pause.setCheckable(True); self._btn_pause.setMinimumHeight(34)
        self._btn_pause.setToolTip("Pause / Resume the running simulation")
        self._btn_pause.clicked.connect(self._on_prog_toggle_pause)
        b_stop = QPushButton("■  Stop")
        b_stop.setMinimumHeight(34); b_stop.setStyleSheet(_RED)
        b_stop.setToolTip(
            "Stop simulation — or emergency-stop the real robot (servo OFF)")
        b_stop.clicked.connect(self._on_stop_all)
        sim_row.addWidget(b_play, 3)
        sim_row.addWidget(self._btn_pause, 2)
        sim_row.addWidget(b_stop, 2)
        lbl_spd = QLabel("Speed")
        lbl_spd.setToolTip("Simulation playback speed multiplier (0.25× – 5×)")
        sim_row.addWidget(lbl_spd)
        self._sim_speed_spin = QDoubleSpinBox()
        self._sim_speed_spin.setRange(0.25, 5.0); self._sim_speed_spin.setSingleStep(0.25)
        self._sim_speed_spin.setValue(1.0); self._sim_speed_spin.setSuffix("×")
        self._sim_speed_spin.setToolTip("Simulation playback speed multiplier")
        self._sim_speed_spin.valueChanged.connect(
            lambda v: setattr(self, "_sim_speed_mult", float(v)))
        sim_row.addWidget(self._sim_speed_spin)
        simw = QWidget(); simw.setLayout(sim_row)
        vbox.addWidget(simw)

        # ─ Row 2: Real robot (intentionally separated) ─
        robot_row = QHBoxLayout(); robot_row.setSpacing(6)
        b_run_robot = QPushButton("⚙  Run on Robot  (real — HSE)")
        b_run_robot.setMinimumHeight(34); b_run_robot.setStyleSheet(_ORANGE)
        b_run_robot.setToolTip(
            "Upload + execute the current job on the real YRC1000 controller "
            "via HSE. Robot WILL move — safety dialog appears first.")
        b_run_robot.clicked.connect(self._on_run_on_robot)
        robot_row.addWidget(b_run_robot, 1)
        rbw = QWidget(); rbw.setLayout(robot_row)
        vbox.addWidget(rbw)

        # ══ 5. FILE bar (Save / Load / Export / Clear) ════════════════
        file_row = QHBoxLayout(); file_row.setSpacing(4)
        for label, cb, tip in (
            ("Save",        self._on_prog_save_dlg,   "Save project (all jobs + targets) to JSON"),
            ("Load",        self._on_prog_load_dlg,   "Load project from JSON (v1/v2/v3 compatible)"),
            ("Export .JBI", self._on_prog_export_dlg, "Export current job to Yaskawa INFORM .JBI"),
            ("Clear all",   self._on_prog_clear,      "Delete all jobs + targets (reset to empty MAIN)"),
        ):
            b = QPushButton(label); b.setToolTip(tip); b.clicked.connect(cb)
            file_row.addWidget(b)
        fw = QWidget(); fw.setLayout(file_row)
        vbox.addWidget(fw)

        # Populate job combo from initial _jobs dict.
        self._refresh_job_combo()

        # Wrap in QScrollArea so narrow dock width still shows all content.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(root)
        dock.setWidget(scroll)
        # minimumWidth = _CELL_DOCK_W so the shared tab area can shrink to cell
        # width when cell is active. When program tab is active, _sync_prog_dock_check
        # resizes to _PROG_DOCK_W. Content has QScrollArea so it is fine when hidden.
        dock.setMinimumWidth(_CELL_DOCK_W)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        # Tabify into the left group (jog + cell). jog/cell built first (order:
        # jog → cell → program in __init__).
        if hasattr(self, "_jog_dock"):
            self.tabifyDockWidget(self._jog_dock, dock)
        dock.setVisible(False)
        dock.visibilityChanged.connect(self._sync_prog_dock_check)

    # ══════════════════════════════════════════════════════════════════
    # Scene loading (pyvista / VTK)
    # ══════════════════════════════════════════════════════════════════

    def _load_scene(self) -> None:
        """Load all meshes + lights into the pyvista plotter (legacy — by calling
        _load_robot_gp7 + _load_cell_assets after the base scene is initialised).
        Floor NOT loaded by default (toggle via menu)."""
        self._add_world_axes_triad()
        self._setup_lighting()
        self._load_robot_assets()
        self._load_cell_assets()

    # ══════════════════════════════════════════════════════════════════
    # Deferred robot/cell loaders (menu-triggered or auto from ctor)
    # ══════════════════════════════════════════════════════════════════

    def _load_robot_gp7(self,
                          base_xyz_mm: tuple[float, float, float] | None = None,
                          home_joints_deg: list[float] | None = None,
                          ) -> None:
        """Build GP7 URDF model + STL meshes + enable robot-dependent UI.
        Idempotent — if already loaded, just reports status.

        Priority for base_xyz / home_joints:
          1. Caller-provided params (e.g. from Add Robot dialog).
          2. cell_config.robot if available.
          3. Defaults: (0, 0, 630) mm and [0]*6 deg.
        """
        if self._model is not None:
            self._set_status("Robot GP7 already loaded", level="info")
            return
        cc = self._cell_config
        if base_xyz_mm is not None:
            self._base_xyz = tuple(base_xyz_mm)
        elif cc is not None and getattr(cc, "robot", None) is not None:
            self._base_xyz = tuple(cc.robot.pose.xyz_mm)
        else:
            # Default Z=330: GP7 built-in stand (330mm) sits on the floor (Z=0).
            # If the user adds a 300mm pedestal later, change Z to 630 in the Edit dialog.
            self._base_xyz = (0.0, 0.0, 330.0)
        if home_joints_deg is not None:
            self._home_joints = list(home_joints_deg)
        elif cc is not None and getattr(cc, "robot", None) is not None:
            self._home_joints = list(cc.robot.home_joints_deg)
        else:
            self._home_joints = [0.0] * 6
        self._joints = list(self._home_joints)
        self._model = gp7_urdf(base_xyz_mm=self._base_xyz)

        # Load STL meshes + apply home pose
        self._load_robot_assets()
        self._apply_joints_main(self._joints)
        self._refresh_pose_readout()
        # Re-build jog dock sliders with joint limits from the model (needs live model)
        self._rebuild_jog_sliders_if_needed()
        self._set_robot_dependent_enabled(True)
        # Auto-fit camera so the whole robot fits in frame — avoids the user
        # having to scroll/zoom out after loading. Then set Iso preset for
        # a consistent viewing angle.
        try:
            self._plotter.reset_camera()
            self._set_camera_preset("Iso")
        except Exception:                                  # noqa: BLE001
            pass
        # Ensure there is a minimal _cell_config so the user can add components
        # directly to the tree without needing to Load Cell first.
        self._ensure_cell_config()
        # Sync robot pose/home into _cell_config so Save Cell picks up the correct
        # values just loaded (especially when the user calls Add Robot with custom params).
        self._cell_config.robot.pose = PoseConfig(
            xyz_mm=self._base_xyz, rpy_deg=(0.0, 0.0, 0.0))
        self._cell_config.robot.home_joints_deg = list(self._home_joints)
        self._refresh_cell_tree()
        # Auto-show Base + Tool triads (RoboDK convention): user immediately sees
        # XYZ axes at the base + end-effector without a manual toggle.
        if hasattr(self, "_frame_checks"):
            for key in ("base", "tool"):
                cb = self._frame_checks.get(key)
                if cb is not None and not cb.isChecked():
                    cb.setChecked(True)                     # fire toggle → add triad
        self._set_status("Robot GP7 loaded", level="ok")

    def _load_cell_from_yaml(self, path: str | Path) -> None:
        """Load cell config from a YAML file and apply it to the scene. Cell config
        includes robot pose + home joints, so it also **auto-loads robot GP7**
        (if not yet loaded) — the user usually does not want a cell without a robot."""
        path = Path(path)
        try:
            cfg = CellConfig.from_yaml(path)
        except Exception as e:                                   # noqa: BLE001
            self._set_status(f"Error loading cell '{path.name}': {e}", level="err")
            return
        self._cell_config = cfg
        # Auto-load robot GP7 first (if not yet) so base_xyz + home joints are
        # taken from cfg.robot, and STL meshes appear with the cell. If robot is
        # already loaded but base_xyz differs from cfg.robot.pose, rebuild model
        # + apply home joints so the robot is at the correct cell position.
        if self._model is None:
            self._load_robot_gp7()
        else:
            new_base = tuple(cfg.robot.pose.xyz_mm) \
                if getattr(cfg.robot, "pose", None) else self._base_xyz
            if new_base != self._base_xyz:
                self._base_xyz = new_base
                self._model = gp7_urdf(base_xyz_mm=new_base)
            self._home_joints = list(cfg.robot.home_joints_deg) \
                if getattr(cfg.robot, "home_joints_deg", None) \
                else self._home_joints
            self._joints = list(self._home_joints)
            self._apply_joints_main(self._joints)
        # Re-build tool/ref frames so jog/IK use the correct config
        self._tool_frames = _build_tool_frames(cfg)
        self._ref_frames = _build_ref_frames(cfg)
        self._tool_idx = (len(self._tool_frames) - 1
                          if len(self._tool_frames) > 1 else 0)
        self._ref_idx = 0
        # Refresh combo UI if present (tool/ref combos)
        self._refresh_tool_ref_combos_if_present()
        # Load cell meshes (worktable, pedestal, objects)
        self._load_cell_assets()
        # Refresh Cell tree to show the new structure
        self._refresh_cell_tree()
        # HSE connection update if config has it
        rc = getattr(cfg, "robot_connection", None)
        if rc is not None:
            # Only override when the cell YAML actually CARRIES a value — a cell
            # without connection keys must NOT wipe the configured/default IP, tool#
            # or FTP creds (those drive Run-on-Robot / Send-pose / the real TL frame).
            self._hse_ip = getattr(rc, "ip", None) or self._hse_ip
            self._hse_tool_no = int(getattr(rc, "tool_no", None) or self._hse_tool_no)
            self._hse_ftp_user = getattr(rc, "ftp_user", None) or self._hse_ftp_user
            self._hse_ftp_pass = getattr(rc, "ftp_pass", None) or self._hse_ftp_pass
            self._hse_ftp_dir = getattr(rc, "ftp_job_dir", None) or self._hse_ftp_dir
        # Auto-fit camera including the new cell meshes
        try:
            self._plotter.reset_camera()
            self._set_camera_preset("Iso")
        except Exception:                                  # noqa: BLE001
            pass
        # Restore visibility state from metadata if present (B8 persistence)
        vis = getattr(cfg.metadata, "visibility_state", None) if cfg else None
        if vis:
            self._component_visibility = dict(vis)
            # Apply saved Hide states after meshes are loaded
            self._reapply_visibility_state()
        if self._model is not None:
            self._plotter.render()
        self._set_status(f"Cell loaded: {path.name}", level="ok")

    def _save_cell_to_yaml(self, path: str | Path) -> None:
        """Dump the current CellConfig to a YAML file. Also saves the visibility
        state (Show/Hide from the cell tree) into metadata for persistence."""
        if self._cell_config is None:
            self._set_status("No cell config to save", level="warn")
            return
        # Snapshot visibility state into metadata before dumping
        vis = getattr(self, "_component_visibility", {})
        if vis:
            self._cell_config.metadata.visibility_state = dict(vis)
        try:
            self._cell_config.to_yaml(Path(path))
            self._set_status(f"Cell saved: {Path(path).name}", level="ok")
        except Exception as e:                                   # noqa: BLE001
            self._set_status(f"Error saving cell: {e}", level="err")

    def _set_robot_dependent_enabled(self, enabled: bool) -> None:
        """Enable/disable all widgets that depend on the robot model. When enabled=False,
        the jog dock + kinematics actions are all disabled (tooltip hints the user
        to load the robot first)."""
        # Jog dock: only disable the CONTENT (scroll area + controls inside),
        # NOT the whole dock — disabling the dock greys out the title bar +
        # ✕ button ⇒ the panel cannot be closed. Title bar must always be enabled.
        if hasattr(self, "_jog_dock") and self._jog_dock.widget() is not None:
            self._jog_dock.widget().setEnabled(enabled)
        # Status console stays enabled to show hints
        # Robot-dependent menu actions (config dialog, params, IK)
        # Only the actions that actually exist as menu items (the old list named two
        # — _act_configurations/_act_find_alts — that were never created, so those
        # robot-dependent items silently stayed enabled with no robot).
        for attr in ("_act_robot_params", "_act_move_home", "_act_surface_pick"):
            act = getattr(self, attr, None)
            if act is not None:
                act.setEnabled(enabled)

    def _rebuild_jog_sliders_if_needed(self) -> None:
        """After loading the robot, joint sliders need range from the model's
        joint limits. If the jog_dock was built with model=None, sliders already
        have default ±180° range — that is fine, slider values are still physically
        correct. This method is a placeholder for future refinement (if needed)."""
        # Placeholder: currently _build_jog_dock reads self._model in the slider
        # creation loop. When robot loads late, sliders keep the old range. To keep
        # commits small we do NOT rebuild the jog dock — just update values when
        # _apply_joints_main runs.
        return

    def _refresh_tool_ref_combos_if_present(self) -> None:
        """Update tool/ref combos if the combo attributes exist after loading
        the cell. _build_jog_dock usually creates these combos — refresh if present."""
        for attr, items_attr in (("_tool_combo", "_tool_frames"),
                                   ("_ref_combo", "_ref_frames")):
            combo = getattr(self, attr, None)
            if combo is None:
                continue
            items = getattr(self, items_attr, [])
            try:
                combo.blockSignals(True)
                combo.clear()
                for name, _T in items:
                    combo.addItem(name)
                combo.blockSignals(False)
            except Exception:                                   # noqa: BLE001
                pass

    # ── Menu action handlers ──────────────────────────────────────────

    def _on_action_load_robot_gp7(self) -> None:
        self._load_robot_gp7()

    def _on_action_load_cell(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Cell from YAML",
            str(self._project_root / "config"),
            "YAML files (*.yaml *.yml);;All files (*.*)")
        if path:
            self._load_cell_from_yaml(path)

    def _on_action_save_cell(self) -> None:
        # Automatically takes the current CellConfig from the tree (including
        # components added interactively by the user); _cell_config is kept
        # in sync by _refresh_cell_tree.
        if self._cell_config is None:
            self._ensure_cell_config()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Cell to YAML",
            str(self._project_root / "config" / "cell_layout.yaml"),
            "YAML files (*.yaml *.yml)")
        if path:
            self._save_cell_to_yaml(path)

    # ══════════════════════════════════════════════════════════════════
    # Cell Editor — interactive cell design (RoboDK Station Tree style)
    # ══════════════════════════════════════════════════════════════════
    # Tree dock shows the current cell structure. User adds/edits/deletes via
    # context menu (right-click). All mutations update self._cell_config
    # ⇒ Save Cell automatically dumps the full design including newly added components.

    def _build_cell_tree_dock(self) -> None:
        """Cell Tree dock (left, tabbed with Jog). Narrow ~180px (≈½ the old width)
        — the main dock uses less space, giving the viewport more room."""
        dock = QDockWidget("Cell", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                             | Qt.DockWidgetArea.RightDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable
                         | QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self._cell_tree = QTreeWidget()
        self._cell_tree.setHeaderLabel("Cell components")
        self._cell_tree.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self._cell_tree.customContextMenuRequested.connect(
            self._on_cell_tree_context_menu)
        self._cell_tree.itemDoubleClicked.connect(
            self._on_cell_tree_double_click)
        dock.setWidget(self._cell_tree)
        # Width policy: prefer 180px, min 140px, no stretching
        dock.setMinimumWidth(140)
        self._cell_tree.setMinimumWidth(140)
        dock.resize(_CELL_DOCK_W, dock.height())
        self._cell_tree_dock = dock
        # Tabbed with Jog (same left panel); Cell tree visible by default
        # (Jog dock starts hidden in _build_jog_dock).
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        if hasattr(self, "_jog_dock"):
            self.tabifyDockWidget(self._jog_dock, dock)
            dock.raise_()
        # Force width after the dock is added to the main window (Qt requires
        # resizeDocks for actual width to take effect, dock.resize alone is NOT enough).
        QTimer.singleShot(0, lambda: self.resizeDocks(
            [dock], [_CELL_DOCK_W], Qt.Orientation.Horizontal))
        # Sync menu tick when user closes the dock with the title bar X button
        dock.visibilityChanged.connect(self._sync_cell_dock_check)
        self._refresh_cell_tree()

    def _ensure_cell_config(self) -> None:
        """Create a minimal CellConfig (robot only) if None. Worktable/camera/
        gripper/floor/pedestal/camera_mount are all None — user adds via UI."""
        if self._cell_config is not None:
            return
        from ...cell.cell_models import MetadataConfig, RobotConfig
        self._cell_config = CellConfig(
            metadata=MetadataConfig(version="0.0.0",
                                    notes="Created from Cell Editor"),
            robot=RobotConfig(
                name="Yaskawa GP7", source="library",
                library_name="Yaskawa-GP7.robot",
                pose=PoseConfig(xyz_mm=(0.0, 0.0, 330.0),
                                rpy_deg=(0.0, 0.0, 0.0)),
                home_joints_deg=[0.0] * 6),
        )

    def _refresh_cell_tree(self) -> None:
        """Re-build the tree from self._cell_config. RoboDK-style: only shows
        items that EXIST (no "(none) — Add" placeholder). Add components via
        context menu on the root node or the top-level Edit menu."""
        if not hasattr(self, "_cell_tree"):
            return
        self._cell_tree.clear()
        cfg = self._cell_config

        def _add(parent, label, kind, ref=None):
            it = QTreeWidgetItem(parent, [label])
            it.setData(0, Qt.ItemDataRole.UserRole, (kind, ref))
            return it

        # Count items to show at root like RoboDK ("Cell (12)")
        def _count(c):
            if c is None: return 0
            n = sum(1 for v in (getattr(c, k, None) for k in
                    ("worktable", "robot_pedestal", "floor", "camera",
                     "camera_mount")) if v is not None)
            n += len(c.frames or []) + len(c.objects or [])
            n += 1 if (c.robot is not None and self._model is not None) else 0
            # Count gripper only if it has a mesh (default empty does not count)
            g = getattr(c, "gripper", None) if c else None
            if g is not None and getattr(g, "mesh", None):
                n += 1
            return n

        n = _count(cfg)
        root = QTreeWidgetItem(self._cell_tree, [f"Cell ({n})"])
        root.setData(0, Qt.ItemDataRole.UserRole, ("root", None))

        if cfg is None or n == 0:
            root.setExpanded(True)
            return

        # ── Robot (root level) — Gripper nested under robot ──
        if cfg.robot is not None and self._model is not None:
            rob_node = _add(root, cfg.robot.name, "robot", cfg.robot)
            g = getattr(cfg, "gripper", None)
            if g is not None and getattr(g, "mesh", None):
                _add(rob_node, f"{g.name} (TCP)", "gripper", g)
            rob_node.setExpanded(True)

        # (World/Base/Tool triads toggled via right-click Robot → Triads submenu)

        # ── Cell items group (Worktable/Pedestal/Floor/Camera/Camera Mount) ──
        cell_items = []
        for kind, label in (("worktable", "Worktable"),
                              ("robot_pedestal", "Pedestal"),
                              ("floor", "Floor"),
                              ("camera", "Camera"),
                              ("camera_mount", "Camera Mount")):
            ref = getattr(cfg, kind, None)
            if ref is not None:
                cell_items.append((kind, label, ref))
        if cell_items:
            ci_group = QTreeWidgetItem(self._cell_tree,
                                        [f"Cell items ({len(cell_items)})"])
            ci_group.setData(0, Qt.ItemDataRole.UserRole,
                             ("cell_items_group", None))
            for kind, label, ref in cell_items:
                _add(ci_group, label, kind, ref)
            ci_group.setExpanded(True)

        # ── Frames group ──
        frames = cfg.frames or []
        if frames:
            fr_group = QTreeWidgetItem(self._cell_tree,
                                        [f"Frames ({len(frames)})"])
            fr_group.setData(0, Qt.ItemDataRole.UserRole,
                             ("frames_group", None))
            for fr in frames:
                _add(fr_group, fr.name, "frame", fr)
            fr_group.setExpanded(True)

        # ── Objects group ──
        objs = cfg.objects or []
        if objs:
            obj_group = QTreeWidgetItem(self._cell_tree,
                                         [f"Objects ({len(objs)})"])
            obj_group.setData(0, Qt.ItemDataRole.UserRole,
                              ("objects_group", None))
            for o in objs:
                _add(obj_group, o.name, "object", o)
            obj_group.setExpanded(True)

        root.setExpanded(True)

    # ── Context menu ──────────────────────────────────────────────────

    def _on_cell_tree_context_menu(self, pos: QPoint) -> None:
        item = self._cell_tree.itemAt(pos)
        kind, ref = (item.data(0, Qt.ItemDataRole.UserRole)
                     if item is not None else (None, None))
        menu = QMenu(self)
        # If move widget is active ⇒ show Commit/Cancel at the top of the menu
        if getattr(self, "_move_widget", None) is not None:
            menu.addAction("Commit move",
                           lambda: self._stop_move_widget(commit=True))
            menu.addAction("Cancel move",
                           lambda: self._stop_move_widget(commit=False))
            menu.addSeparator()
        # Show/Hide at the top for every leaf (real component) — checkable toggle.
        if kind in ("robot", "gripper", "worktable", "robot_pedestal",
                     "floor", "camera", "camera_mount", "frame", "object"):
            vis = self._is_component_visible(kind, ref)
            act_vis = menu.addAction("Hide" if vis else "Show")
            act_vis.triggered.connect(
                lambda _checked=False, k=kind, r=ref: self._toggle_component_visibility(k, r))
            menu.addSeparator()
        # (world_axes branch removed — toggle via robot → Triads submenu)
        # Leaf nodes — Edit/Delete/Move
        if kind == "frame":
            menu.addAction("Edit…", lambda: self._show_edit_frame_dlg(ref))
            menu.addAction("Delete", lambda: self._delete_frame(ref))
        elif kind == "object":
            menu.addAction("Edit…", lambda: self._show_edit_object_dlg(ref))
            menu.addAction("Move (drag in viewport)",
                           lambda: self._start_move_widget("object", ref))
            menu.addAction("Delete", lambda: self._delete_object(ref))
        elif kind in ("worktable", "robot_pedestal", "floor",
                      "camera_mount", "camera"):
            menu.addAction("Edit…",
                           lambda k=kind, r=ref: self._show_edit_single_dlg(k, r))
            if kind != "camera":  # camera has no dedicated actor to drag
                menu.addAction("Move (drag in viewport)",
                               lambda k=kind, r=ref: self._start_move_widget(k, r))
            if kind in ("robot_pedestal", "floor", "camera_mount", "worktable"):
                menu.addAction("Delete",
                               lambda k=kind: self._delete_single(k))
        elif kind == "robot":
            menu.addAction("Edit base pose…",
                           lambda: self._show_edit_robot_dlg(ref))
            # Sub-actions: add/edit gripper directly from the robot item
            g = (self._cell_config.gripper
                 if self._cell_config is not None else None)
            if g is None or not getattr(g, "mesh", None):
                menu.addAction("Add Gripper…", self._show_add_gripper_dlg)
            # Triads submenu — World/Base/Tool toggle directly on the robot item
            self._add_triads_submenu(menu)
        elif kind == "gripper":
            menu.addAction("Edit…", lambda: self._show_edit_gripper_dlg(ref))
            menu.addAction("Delete", self._delete_gripper)
        elif kind == "frames_group":
            menu.addAction("Add Frame…", self._show_add_frame_dlg)
        elif kind == "objects_group":
            menu.addAction("Add Object…", self._show_add_object_dlg)
        elif kind == "cell_items_group":
            cfg = self._cell_config
            if cfg is None or cfg.worktable is None:
                menu.addAction("Add Worktable…",
                    lambda: self._show_add_single_dlg("worktable"))
            if cfg is None or cfg.robot_pedestal is None:
                menu.addAction("Add Pedestal…",
                    lambda: self._show_add_single_dlg("robot_pedestal"))
            if cfg is None or cfg.floor is None:
                menu.addAction("Add Floor…",
                    lambda: self._show_add_single_dlg("floor"))
            if cfg is None or cfg.camera_mount is None:
                menu.addAction("Add Camera Mount…",
                    lambda: self._show_add_single_dlg("camera_mount"))
            if cfg is None or cfg.camera is None:
                menu.addAction("Add Camera…", self._show_add_camera_dlg)
        elif kind == "world_axes":
            pass            # Show/Hide actions already added in the block above
        else:
            # Root, background, or unknown → "Add ..." menu like RoboDK.
            cfg = self._cell_config
            self._add_cell_menu_actions(menu, cfg)
        if menu.actions():
            menu.exec(self._cell_tree.viewport().mapToGlobal(pos))

    def _add_cell_menu_actions(self, menu: QMenu, cfg) -> None:
        """Populate the menu with appropriate Add actions. Existing items are
        not listed again — keeps the menu clean like RoboDK."""
        if self._model is None:
            menu.addAction("Add Robot…", self._show_add_robot_dlg)
            menu.addSeparator()
        menu.addAction("Add Object…", self._show_add_object_dlg)
        menu.addAction("Add Frame…", self._show_add_frame_dlg)
        menu.addSeparator()
        if cfg is None or cfg.worktable is None:
            menu.addAction("Add Worktable…",
                           lambda: self._show_add_single_dlg("worktable"))
        if cfg is None or cfg.robot_pedestal is None:
            menu.addAction("Add Pedestal…",
                           lambda: self._show_add_single_dlg("robot_pedestal"))
        if cfg is None or cfg.floor is None:
            menu.addAction("Add Floor…",
                           lambda: self._show_add_single_dlg("floor"))
        if cfg is None or cfg.camera_mount is None:
            menu.addAction("Add Camera Mount…",
                           lambda: self._show_add_single_dlg("camera_mount"))
        if cfg is None or cfg.camera is None:
            menu.addAction("Add Camera…", self._show_add_camera_dlg)
        # Gripper: only shown when robot is loaded (gripper attaches to flange)
        # and no gripper exists yet.
        if (self._model is not None and (cfg is None
                or cfg.gripper is None
                or not getattr(cfg.gripper, "mesh", None))):
            menu.addAction("Add Gripper…", self._show_add_gripper_dlg)

    def _on_cell_tree_double_click(self, item: QTreeWidgetItem,
                                     col: int) -> None:
        kind, ref = item.data(0, Qt.ItemDataRole.UserRole) or (None, None)
        if kind == "frame":
            self._show_edit_frame_dlg(ref)
        elif kind == "object":
            self._show_edit_object_dlg(ref)
        elif kind == "robot":
            self._show_edit_robot_dlg(ref)
        elif kind in ("worktable", "robot_pedestal", "floor",
                      "camera_mount", "camera"):
            self._show_edit_single_dlg(kind, ref)
        elif kind == "gripper":
            self._show_edit_gripper_dlg(ref)

    # ── Pose form helper ──────────────────────────────────────────────

    def _make_pose_form(self, xyz=(0.0, 0.0, 0.0),
                         rpy=(0.0, 0.0, 0.0)) -> tuple[QFormLayout, dict]:
        """Build a form with 3 xyz_mm spinboxes + 3 rpy_deg spinboxes. Returns
        (layout, widgets_dict) for the caller to wrap in a QDialog."""
        form = QFormLayout()
        widgets = {}
        def _spin(val, lo, hi, suffix, dec=1):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi); sp.setDecimals(dec); sp.setSingleStep(1.0)
            sp.setValue(float(val)); sp.setSuffix(suffix); sp.setFixedWidth(110)
            return sp
        for axis, v in zip("XYZ", xyz):
            w = _spin(v, -5000.0, 5000.0, " mm")
            form.addRow(f"{axis} (mm):", w); widgets[f"x{axis.lower()}"] = w
        for axis, v in zip(("R", "P", "Y"), rpy):
            w = _spin(v, -360.0, 360.0, " °")
            form.addRow(f"{axis} (deg):", w); widgets[f"r{axis.lower()}"] = w
        return form, widgets

    @staticmethod
    def _read_pose(widgets: dict) -> tuple[tuple, tuple]:
        xyz = (widgets["xx"].value(), widgets["xy"].value(),
               widgets["xz"].value())
        rpy = (widgets["rr"].value(), widgets["rp"].value(),
               widgets["ry"].value())
        return xyz, rpy

    # ── Add Object / Frame dialogs ────────────────────────────────────

    def _show_add_robot_dlg(self) -> None:
        """Add Robot — pop-up dialog with variant combo, base pose, home joints.
        Currently only supports GP7 (single variant); design remains extensible."""
        if self._model is not None:
            QMessageBox.information(self, "Add Robot",
                "Robot already loaded. To change pose/home, use Edit "
                "on the robot item in the cell tree.")
            return
        dlg = QDialog(self); dlg.setWindowTitle("Add Robot")
        dlg.setMinimumWidth(560)
        v = QVBoxLayout(dlg)
        # Variant combo (full-width, at the top)
        form = QFormLayout()
        variant_combo = QComboBox()
        variant_combo.addItem("Yaskawa GP7 (6-DOF)", "gp7")
        form.addRow("Robot variant:", variant_combo)
        v.addLayout(form)
        # Middle row: [base pose fields | robot preview]. Preview fills the
        # empty area to the right of the pose fields (previously blank). Z=330mm:
        # GP7 stand sits on the floor (no pedestal); add a 300mm pedestal and change Z to 630.
        mid = QHBoxLayout()
        pose_col = QVBoxLayout()
        pose_col.addWidget(QLabel("<b>Base pose (world frame):</b>"))
        pose_form, pw = self._make_pose_form(xyz=(0.0, 0.0, 330.0))
        pose_col.addLayout(pose_form)
        pose_col.addStretch()
        mid.addLayout(pose_col, 0)
        preview = self._make_preview_label((300, 300))
        mid.addWidget(preview, 1)
        v.addLayout(mid)
        # Home joints (6 spinboxes, full-width at the bottom)
        v.addWidget(QLabel("<b>Home joints (S, L, U, R, B, T) °:</b>"))
        home_grid = QHBoxLayout()
        home_spins: list[QDoubleSpinBox] = []
        for i, name in enumerate(("S", "L", "U", "R", "B", "T")):
            col = QVBoxLayout()
            col.addWidget(QLabel(name))
            sp = QDoubleSpinBox()
            sp.setRange(-360.0, 360.0); sp.setDecimals(2); sp.setSingleStep(5.0)
            sp.setSuffix(" °"); sp.setFixedWidth(80)
            sp.setValue(0.0)
            col.addWidget(sp); home_spins.append(sp); home_grid.addLayout(col)
        v.addLayout(home_grid)
        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                 | QDialogButtonBox.StandardButton.Cancel)
        def _ok():
            xyz, rpy = self._read_pose(pw)
            if rpy != (0.0, 0.0, 0.0):
                QMessageBox.warning(dlg, "Add Robot",
                    "Robot base currently supports xyz only (rpy=0). RPY field ignored.")
            home = [sp.value() for sp in home_spins]
            # variant lookup — future: could switch _load_robot_*()
            variant = variant_combo.currentData()
            if variant != "gp7":
                QMessageBox.warning(dlg, "Add Robot",
                    f"Variant '{variant}' not supported yet.")
                return
            self._load_robot_gp7(base_xyz_mm=xyz, home_joints_deg=home)
            dlg.accept()
        btns.accepted.connect(_ok); btns.rejected.connect(dlg.reject)
        v.addWidget(btns)

        def _refresh_preview():
            self._set_preview(preview, lambda sz: self._render_robot_thumbnail(
                [sp.value() for sp in home_spins], size=sz))
        for sp in home_spins:
            sp.editingFinished.connect(_refresh_preview)
        _refresh_preview()
        dlg.exec()

    def _show_add_gripper_dlg(self) -> None:
        """Add Gripper — attaches to the robot flange (link_tool0). Unlike
        Add Object: gripper has a TCP offset (not a base offset), and the mesh
        actor automatically follows the flange via _render_scene_frame."""
        self._ensure_cell_config()
        dlg = QDialog(self); dlg.setWindowTitle("Add Gripper")
        dlg.setMinimumWidth(440)
        v = QVBoxLayout()
        form = QFormLayout()
        name_edit = QLineEdit("Gripper")
        form.addRow("Name:", name_edit)
        mesh_edit = QLineEdit()
        mesh_edit.setPlaceholderText("models/gripper.stl")
        mesh_row = QHBoxLayout()
        mesh_row.addWidget(mesh_edit, 1)
        btn_pick = QPushButton("Browse…")
        def _pick():
            p, _ = QFileDialog.getOpenFileName(
                dlg, "Pick gripper mesh",
                str(self._project_root / "models"),
                "Mesh files (*.stl *.obj);;All (*.*)")
            if p:
                try:
                    p = str(Path(p).resolve().relative_to(
                        self._project_root.resolve()))
                except ValueError: pass
                mesh_edit.setText(p)
        btn_pick.clicked.connect(_pick)
        mesh_row.addWidget(btn_pick)
        form.addRow("Mesh:", mesh_row)
        v.addLayout(form)
        v.addWidget(QLabel("<b>TCP offset (from flange link_tool0):</b>"))
        pose_form, pw = self._make_pose_form(xyz=(0.0, 0.0, 100.0))
        v.addLayout(pose_form)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                 | QDialogButtonBox.StandardButton.Cancel)
        def _ok():
            name = name_edit.text().strip()
            if not name:
                QMessageBox.warning(dlg, "Add Gripper",
                                     "Name cannot be empty"); return
            mesh = mesh_edit.text().strip()
            if not mesh:
                QMessageBox.warning(dlg, "Add Gripper",
                                     "Mesh path is required"); return
            xyz, rpy = self._read_pose(pw)
            try:
                from ...cell.cell_models import GripperConfig
                self._cell_config.gripper = GripperConfig(
                    name=name, mesh=mesh,
                    tcp_offset_xyz_mm=xyz, tcp_offset_rpy_deg=rpy)
            except Exception as e:                          # noqa: BLE001
                QMessageBox.warning(dlg, "Add Gripper", f"Invalid: {e}")
                return
            # Reload gripper actor + tool frames so jog/IK use the correct TCP
            self._reload_gripper_actor()
            self._tool_frames = _build_tool_frames(self._cell_config)
            self._tool_idx = len(self._tool_frames) - 1
            self._refresh_tool_ref_combos_if_present()
            self._refresh_cell_tree()
            self._set_status(f"Added gripper '{name}'", level="ok")
            dlg.accept()
        btns.accepted.connect(_ok); btns.rejected.connect(dlg.reject)
        v.addWidget(btns)
        preview = self._attach_preview(dlg, v)

        def _refresh_preview():
            self._set_preview(preview, lambda sz: self._render_mesh_thumbnail(
                mesh_edit.text().strip(), [0.78, 0.78, 0.80], size=sz, zoom=0.82))
        mesh_edit.editingFinished.connect(_refresh_preview)
        btn_pick.clicked.connect(_refresh_preview)          # update sau Browse
        _refresh_preview()
        dlg.exec()

    def _show_edit_gripper_dlg(self, g) -> None:
        if g is None: return
        dlg = QDialog(self); dlg.setWindowTitle(f"Edit Gripper: {g.name}")
        dlg.setMinimumWidth(440)
        v = QVBoxLayout(dlg)
        form = QFormLayout()
        form.addRow("Name:", QLabel(f"<b>{g.name}</b>"))
        mesh_edit = QLineEdit(g.mesh or "")
        mesh_row = QHBoxLayout()
        mesh_row.addWidget(mesh_edit, 1)
        btn_pick = QPushButton("Browse…")
        def _pick():
            p, _ = QFileDialog.getOpenFileName(
                dlg, "Pick gripper mesh",
                str(self._project_root / "models"),
                "Mesh files (*.stl *.obj);;All (*.*)")
            if p:
                try:
                    p = str(Path(p).resolve().relative_to(
                        self._project_root.resolve()))
                except ValueError: pass
                mesh_edit.setText(p)
        btn_pick.clicked.connect(_pick)
        mesh_row.addWidget(btn_pick)
        form.addRow("Mesh:", mesh_row)
        v.addLayout(form)
        v.addWidget(QLabel("<b>TCP offset (from flange link_tool0):</b>"))
        pose_form, pw = self._make_pose_form(
            xyz=g.tcp_offset_xyz_mm, rpy=g.tcp_offset_rpy_deg)
        v.addLayout(pose_form)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                 | QDialogButtonBox.StandardButton.Cancel)
        def _ok():
            xyz, rpy = self._read_pose(pw)
            g.mesh = mesh_edit.text().strip() or None
            g.tcp_offset_xyz_mm = xyz
            g.tcp_offset_rpy_deg = rpy
            self._reload_gripper_actor()
            self._tool_frames = _build_tool_frames(self._cell_config)
            self._refresh_tool_ref_combos_if_present()
            self._refresh_cell_tree()
            self._set_status(f"Updated gripper '{g.name}'", level="ok")
            dlg.accept()
        btns.accepted.connect(_ok); btns.rejected.connect(dlg.reject)
        v.addWidget(btns)
        dlg.exec()

    def _delete_gripper(self) -> None:
        if self._cell_config is None or self._cell_config.gripper is None:
            return
        g = self._cell_config.gripper
        ret = QMessageBox.question(
            self, "Delete Gripper",
            f"Delete gripper '{g.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ret != QMessageBox.StandardButton.Yes: return
        # Schema is now Optional → set None cleanly
        self._cell_config.gripper = None
        # Remove mesh actor + revert tool frames to Flange
        try:
            self._plotter.remove_actor("gripper")
        except Exception: pass                              # noqa: BLE001
        self._link_actors.pop("gripper", None)
        self._tool_frames = _build_tool_frames(self._cell_config)
        self._tool_idx = 0
        self._refresh_tool_ref_combos_if_present()
        self._refresh_cell_tree()
        self._plotter.render()
        self._set_status(f"Deleted gripper '{g.name}'", level="ok")

    # ══════════════════════════════════════════════════════════════════
    # Move widget — interactive drag-to-move (vtkBoxWidget)
    # ══════════════════════════════════════════════════════════════════

    # ── Triads submenu (right-click robot → toggle World/Base/Tool) ──

    def _add_triads_submenu(self, parent_menu: QMenu) -> None:
        """Triads submenu to toggle World/Base/Tool triads + labels from the
        robot context menu. Checkable items sync with jog dock checkboxes."""
        parent_menu.addSeparator()
        sub = parent_menu.addMenu("Triads")
        # World axes
        a_world = sub.addAction("World axes")
        a_world.setCheckable(True)
        a_world.setChecked(self._is_world_axes_visible())
        a_world.triggered.connect(self._toggle_world_axes)
        # World labels
        a_lbls = sub.addAction("World axis labels")
        a_lbls.setCheckable(True)
        a_lbls.setChecked(self._is_world_axes_labels_visible())
        a_lbls.triggered.connect(self._toggle_world_axes_labels)
        sub.addSeparator()
        # Base triad
        a_base = sub.addAction("Base triad")
        a_base.setCheckable(True)
        a_base.setChecked("base" in self._frame_actors)
        a_base.triggered.connect(
            lambda _checked=False: self._toggle_frame_triad_from_menu("base"))
        # Tool triad
        a_tool = sub.addAction("Tool triad")
        a_tool.setCheckable(True)
        a_tool.setChecked("tool" in self._frame_actors)
        a_tool.triggered.connect(
            lambda _checked=False: self._toggle_frame_triad_from_menu("tool"))
        # Flange (optional, rarely used)
        a_fl = sub.addAction("Flange triad")
        a_fl.setCheckable(True)
        a_fl.setChecked("flange" in self._frame_actors)
        a_fl.triggered.connect(
            lambda _checked=False: self._toggle_frame_triad_from_menu("flange"))

    def _toggle_frame_triad_from_menu(self, key: str) -> None:
        """Toggle frame triad (base/tool/flange/ref/joint_N). Sync with
        the corresponding checkbox in the jog dock 'Show Frames' group."""
        currently_on = key in self._frame_actors
        new_state = not currently_on
        # Sync checkbox UI (will auto-fire _on_toggle_frame)
        cb = self._frame_checks.get(key) if hasattr(self, "_frame_checks") else None
        if cb is not None:
            cb.setChecked(new_state)
        else:
            # No checkbox (e.g. jog dock not yet built) → toggle directly
            self._on_toggle_frame(key, new_state)
        self._set_status(
            f"{'Showed' if new_state else 'Hid'} {key} triad", level="ok")

    # ── World axes visibility (axes + labels separately) ──────────────

    def _is_world_axes_visible(self) -> bool:
        actor = getattr(self, "_world_axes_actor", None)
        if actor is None: return False
        try: return bool(actor.GetVisibility())
        except Exception: return True                      # noqa: BLE001

    def _is_world_axes_labels_visible(self) -> bool:
        actor = getattr(self, "_world_axes_actor", None)
        if actor is None: return False
        try:
            # Label "visible" = has text (empty text ⇒ not rendered)
            return bool(actor.GetXAxisLabelText())
        except Exception:                                  # noqa: BLE001
            return False

    def _toggle_world_axes(self) -> None:
        actor = getattr(self, "_world_axes_actor", None)
        if actor is None: return
        new = not bool(actor.GetVisibility())
        actor.SetVisibility(new)
        # Sync View menu tick if present
        if hasattr(self, "_act_axes"):
            self._set_toggle(self._act_axes, new)
        self._plotter.render()
        self._set_status(
            f"{'Showed' if new else 'Hid'} world axes", level="ok")

    def _toggle_world_axes_labels(self) -> None:
        actor = getattr(self, "_world_axes_actor", None)
        if actor is None: return
        visible = self._is_world_axes_labels_visible()
        if visible:
            # Hide: clear text → no render
            actor.SetXAxisLabelText("")
            actor.SetYAxisLabelText("")
            actor.SetZAxisLabelText("")
        else:
            # Show: restore "X","Y","Z" + force small size via caption width
            actor.SetXAxisLabelText("X")
            actor.SetYAxisLabelText("Y")
            actor.SetZAxisLabelText("Z")
            for cap in (actor.GetXAxisCaptionActor2D(),
                          actor.GetYAxisCaptionActor2D(),
                          actor.GetZAxisCaptionActor2D()):
                cap.SetWidth(0.020); cap.SetHeight(0.018)
                tp = cap.GetCaptionTextProperty()
                tp.SetBold(False); tp.SetShadow(False); tp.SetItalic(False)
        self._plotter.render()
        self._set_status(
            f"{'Showed' if not visible else 'Hid'} world axes labels",
            level="ok")

    # ── Component visibility (Show/Hide from tree context menu) ─────────

    def _component_visibility_key(self, kind: str, ref) -> str:
        """Unique key per visible component for state tracking."""
        if kind == "frame":
            return f"frame::{getattr(ref, 'name', '')}"
        if kind == "object":
            return f"object::{getattr(ref, 'name', '')}"
        if kind == "gripper":
            return "gripper"
        return kind                                          # robot/worktable/etc.

    def _component_actor_names(self, kind: str, ref) -> list[str]:
        """Names of the actors whose visibility should be toggled for a component."""
        if kind == "robot":
            # All 7 GP7 link actors
            return [k for k, _f, _o in _GP7_MESH_MAP]
        if kind == "gripper":
            return ["gripper"]
        if kind in ("worktable", "robot_pedestal", "floor",
                     "camera", "camera_mount"):
            return [kind]
        if kind == "object":
            return [getattr(ref, "name", "")]
        return []

    def _is_component_visible(self, kind: str, ref) -> bool:
        """Read visibility from the tracking dict (default visible if not set)."""
        if not hasattr(self, "_component_visibility"):
            self._component_visibility: dict[str, bool] = {}
        if kind == "frame":
            # Frame triad: state from _frame_checks (jog dock)
            name = getattr(ref, "name", "")
            # frames in the cell config have no dedicated checkbox — tracked separately
            key = self._component_visibility_key(kind, ref)
            return self._component_visibility.get(key, False)
        key = self._component_visibility_key(kind, ref)
        return self._component_visibility.get(key, True)

    def _set_component_visibility(self, kind: str, ref, visible: bool) -> None:
        """Toggle mesh actors. For frames, add/remove the triad via _frame_actors."""
        if not hasattr(self, "_component_visibility"):
            self._component_visibility: dict[str, bool] = {}
        key = self._component_visibility_key(kind, ref)
        self._component_visibility[key] = visible
        if kind == "frame":
            # Frame triad: add/remove triad via _frame_world_matrix
            name = getattr(ref, "name", "")
            triad_key = f"frame::{name}"
            if visible:
                # Compute world matrix from frame pose (mm → m)
                xyz = ref.pose.xyz_mm; rpy = ref.pose.rpy_deg
                T = _xyz_rpy_to_matrix(xyz[0], xyz[1], xyz[2],
                                         rpy[0], rpy[1], rpy[2])
                T[:3, 3] /= 1000.0
                self._add_frame_triad(triad_key, T)
            else:
                self._remove_frame_triad(triad_key)
            return
        # Mesh actors: SetVisibility
        for actor_name in self._component_actor_names(kind, ref):
            actor = self._find_actor(actor_name)
            if actor is not None:
                actor.SetVisibility(bool(visible))
        self._plotter.render()

    def _reapply_visibility_state(self) -> None:
        """After loading the cell, apply visibility from _component_visibility dict
        (restored from metadata.visibility_state). Iterate config items,
        set hidden for keys = False."""
        if not hasattr(self, "_component_visibility"):
            return
        cfg = self._cell_config
        if cfg is None: return
        # Iterate kinds + refs
        items_to_check: list[tuple] = []
        if cfg.robot is not None:
            items_to_check.append(("robot", cfg.robot))
        g = getattr(cfg, "gripper", None)
        if g is not None and getattr(g, "mesh", None):
            items_to_check.append(("gripper", g))
        for kind_name in ("worktable", "robot_pedestal", "floor",
                           "camera", "camera_mount"):
            r = getattr(cfg, kind_name, None)
            if r is not None:
                items_to_check.append((kind_name, r))
        for fr in (cfg.frames or []):
            items_to_check.append(("frame", fr))
        for o in (cfg.objects or []):
            items_to_check.append(("object", o))
        # Apply state (only if explicitly False — True is default)
        for kind, ref in items_to_check:
            key = self._component_visibility_key(kind, ref)
            saved = self._component_visibility.get(key)
            if saved is False:
                self._set_component_visibility(kind, ref, False)

    def _toggle_component_visibility(self, kind: str, ref) -> None:
        cur = self._is_component_visible(kind, ref)
        self._set_component_visibility(kind, ref, not cur)
        # Show status feedback + update tree (eye icon if present)
        label = (getattr(ref, "name", kind) if ref is not None else kind)
        self._set_status(
            f"{'Hidden' if cur else 'Shown'}: {label}", level="ok")

    def _find_actor(self, name: str):
        """Find an actor by name in the viewport (renderer actors or _link_actors)."""
        actor = self._link_actors.get(name)
        if actor is not None:
            return actor
        try:
            return self._plotter.renderer.actors.get(name)
        except Exception:                                   # noqa: BLE001
            return None

    def _start_move_widget(self, kind: str, ref) -> None:
        """Enable the drag gizmo (vtkBoxWidget) around the actor.

        Avoid SetProp3D — it overrides UserMatrix and snaps the actor to
        identity. Instead, place the widget at the actor world bounds + cache
        T_initial; each InteractionEvent reads the widget LOCAL transform and
        composes it with T_initial → SetUserMatrix actor.
        """
        if getattr(self, "_move_widget", None) is not None:
            self._stop_move_widget(commit=False)

        actor_name = kind if kind != "object" else getattr(ref, "name", "")
        actor = self._find_actor(actor_name)
        if actor is None:
            self._set_status(
                f"Mesh actor for '{kind}' not found — not loaded?",
                level="warn")
            return

        widget = vtk.vtkBoxWidget()
        rwin = self._plotter.render_window
        widget.SetInteractor(rwin.GetInteractor())
        widget.SetRotationEnabled(True)
        widget.SetTranslationEnabled(True)
        widget.SetScalingEnabled(False)
        widget.SetHandleSize(0.015)                      # larger handles for easier grabbing
        widget.SetPlaceFactor(1.2)
        # Place box at the actor's world bounds. For small objects (e.g. bolt 50mm)
        # bounds are tiny ⇒ handles are hidden/hard to grab. Expand to min 200mm/axis
        # for a sufficiently large interaction area.
        b = list(actor.GetBounds())                       # [xmin,xmax,ymin,ymax,zmin,zmax]
        min_size = 0.20                                   # 200mm
        for i in (0, 2, 4):
            if b[i+1] - b[i] < min_size:
                center = (b[i] + b[i+1]) / 2.0
                b[i] = center - min_size / 2.0
                b[i+1] = center + min_size / 2.0
        widget.PlaceWidget(b)
        self._move_widget = widget
        self._move_target = (kind, ref, actor_name)
        self._move_initial_xform = self._get_actor_world_matrix(actor)
        # Wire callbacks AFTER PlaceWidget so they don't fire immediately
        widget.AddObserver(
            "InteractionEvent",
            lambda obj, evt: self._on_move_interact(actor))
        widget.AddObserver(
            "EndInteractionEvent",
            lambda obj, evt: None)
        widget.On()
        self._plotter.render()
        self._set_status(
            "Drag handles → move/rotate. Right-click the cell tree to Commit "
            "move (save into cell) or Cancel move (revert).",
            level="info")

    def _on_move_interact(self, actor) -> None:
        """Each interaction tick: read widget transform → compose with T_initial
        → SetUserMatrix actor (real-time preview)."""
        w = getattr(self, "_move_widget", None)
        if w is None or actor is None: return
        t = vtk.vtkTransform()
        w.GetTransform(t)
        m = vtk.vtkMatrix4x4(); t.GetMatrix(m)
        T_widget = np.array([[m.GetElement(i, j) for j in range(4)]
                              for i in range(4)])
        T_new = T_widget @ self._move_initial_xform
        actor.SetUserMatrix(_numpy_to_vtk_matrix(T_new))
        # Render is triggered by the Qt event loop via the interactor

    def _stop_move_widget(self, commit: bool = True) -> None:
        """Turn off the widget. commit=True ⇒ write pose to _cell_config; False ⇒
        revert actor to its initial transform. Clean up observers to avoid memory
        leak (B4 fix)."""
        w = getattr(self, "_move_widget", None)
        if w is None:
            return
        if commit:
            self._commit_move_pose()
        else:
            # Revert: re-apply initial transform
            t = getattr(self, "_move_initial_xform", None)
            if t is not None:
                kind, ref, name = self._move_target
                actor = self._find_actor(name)
                if actor is not None:
                    actor.SetUserMatrix(_numpy_to_vtk_matrix(t))
        w.Off()
        # Cleanup: remove all observers (lambda closures capture self ⇒
        # circular ref if not cleaned up; rapid toggling accumulates them).
        try: w.RemoveAllObservers()
        except Exception: pass                              # noqa: BLE001
        self._move_widget = None
        self._move_target = None
        self._move_initial_xform = None
        self._plotter.render()

    @staticmethod
    def _get_actor_world_matrix(actor) -> np.ndarray:
        """Get 4x4 numpy from actor UserMatrix (defaults to identity if
        SetUserMatrix was never called). Units: meters (viewport scale)."""
        m = actor.GetUserMatrix()
        if m is None:
            return np.eye(4)
        return np.array([[m.GetElement(i, j) for j in range(4)]
                          for i in range(4)])

    def _on_move_end_event(self, obj, event) -> None:
        """EndInteractionEvent from vtkBoxWidget — only updates _cell_config when
        the user calls commit. This is just a hook for optional auto-commit on release.
        Currently we defer commit to _stop_move_widget(commit=True)."""
        # No-op: keep widget open for preview; commit/cancel is manual.
        pass

    def _commit_move_pose(self) -> None:
        """Read the current transform from the actor → convert to pose mm/deg →
        write to _cell_config. Objects with a parent_frame: convert world→relative."""
        if self._move_target is None: return
        kind, ref, name = self._move_target
        actor = self._find_actor(name)
        if actor is None: return
        T_world_m = self._get_actor_world_matrix(actor)   # meters
        # Convert to mm (config uses mm)
        T_world_mm = T_world_m.copy()
        T_world_mm[:3, 3] *= 1000.0

        # Object with parent_frame ⇒ convert world → parent-relative
        if kind == "object" and getattr(ref, "parent_frame", None):
            parent_name = ref.parent_frame
            parent_T = None
            for f in (self._cell_config.frames or []):
                if f.name == parent_name:
                    parent_T = _xyz_rpy_to_matrix(
                        f.pose.xyz_mm[0], f.pose.xyz_mm[1], f.pose.xyz_mm[2],
                        f.pose.rpy_deg[0], f.pose.rpy_deg[1], f.pose.rpy_deg[2])
                    break
            if parent_T is not None:
                T_rel = np.linalg.inv(parent_T) @ T_world_mm
                xyz, rpy = _matrix_to_xyz_rpy_deg(T_rel)[:3], \
                           _matrix_to_xyz_rpy_deg(T_rel)[3:]
            else:
                xyz, rpy = T_world_mm[:3, 3], _matrix_to_xyz_rpy_deg(T_world_mm)[3:]
        else:
            x, y, z, rx, ry, rz = _matrix_to_xyz_rpy_deg(T_world_mm)
            xyz = (x, y, z); rpy = (rx, ry, rz)

        ref.pose = PoseConfig(xyz_mm=tuple(xyz), rpy_deg=tuple(rpy))
        # Refresh _objects dict (object has a world_T cache for gripper follow)
        if kind == "object" and name in self._objects:
            self._objects[name]["world_T"] = T_world_m.copy()
        self._refresh_cell_tree()
        self._set_status(
            f"Moved {kind} → xyz=({xyz[0]:.1f}, {xyz[1]:.1f}, "
            f"{xyz[2]:.1f}) mm", level="ok")

    def _reload_gripper_actor(self) -> None:
        """Idempotent reload: remove old "gripper" actor (if any) then call
        _load_gripper() to re-add it with the new config."""
        try:
            self._plotter.remove_actor("gripper")
        except Exception: pass                              # noqa: BLE001
        self._link_actors.pop("gripper", None)
        self._load_gripper()
        # Apply joints so the mesh immediately snaps to the correct flange position
        if self._model is not None:
            self._apply_joints_main(self._joints)

    def _show_add_object_dlg(self) -> None:
        self._ensure_cell_config()
        # Smart default pose — avoids overlapping the robot stand (Z=0..330)
        # and previous objects (scatter along Y by 100mm per added object).
        frames = self._cell_config.frames or []
        wt = self._cell_config.worktable
        existing = self._cell_config.objects or []
        n = len(existing)
        # Scatter Y: objects 0..6 placed at Y = -300, -200, ..., +300 mm
        scatter_y = ((n % 7) - 3) * 100.0
        if frames:
            default_parent = frames[0].name
            default_xyz = (0.0, scatter_y, 50.0)
        elif wt is not None:
            h = self._mesh_height_mm(wt.mesh) or 500.0
            default_parent = None
            default_xyz = (wt.pose.xyz_mm[0],
                            wt.pose.xyz_mm[1] + scatter_y,
                            wt.pose.xyz_mm[2] + h + 50.0)
        else:
            default_parent = None
            default_xyz = (500.0, scatter_y, 500.0)

        dlg = QDialog(self); dlg.setWindowTitle("Add Object")
        dlg.setMinimumWidth(420)
        v = QVBoxLayout()
        form = QFormLayout()
        # Name
        name_edit = QLineEdit(); name_edit.setPlaceholderText("e.g., my_part")
        form.addRow("Name:", name_edit)
        # Mesh picker
        mesh_row = QHBoxLayout()
        mesh_edit = QLineEdit(); mesh_edit.setPlaceholderText(
            "models/objects/xxx.stl")
        btn_pick = QPushButton("Browse…")
        def _pick():
            p, _ = QFileDialog.getOpenFileName(
                dlg, "Pick mesh (.stl/.obj)",
                str(self._project_root / "models"),
                "Mesh files (*.stl *.obj *.ply);;All (*.*)")
            if p:
                try:
                    p = str(Path(p).resolve().relative_to(
                        self._project_root.resolve()))
                except ValueError:
                    pass
                mesh_edit.setText(p)
        btn_pick.clicked.connect(_pick)
        mesh_row.addWidget(mesh_edit, 1); mesh_row.addWidget(btn_pick)
        form.addRow("Mesh:", mesh_row)
        # Parent frame combo (pre-select default theo context)
        frame_combo = QComboBox()
        frame_combo.addItem("(base / none)", None)
        default_idx = 0
        for i, f in enumerate(frames):
            frame_combo.addItem(f.name, f.name)
            if f.name == default_parent:
                default_idx = i + 1
        frame_combo.setCurrentIndex(default_idx)
        form.addRow("Parent frame:", frame_combo)
        v.addLayout(form)
        # Pose form
        v.addWidget(QLabel("<b>Offset from parent frame:</b>"))
        pose_form, pw = self._make_pose_form(xyz=default_xyz)
        v.addLayout(pose_form)
        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                 | QDialogButtonBox.StandardButton.Cancel)
        def _ok():
            name = name_edit.text().strip()
            if not name:
                QMessageBox.warning(dlg, "Add Object", "Name cannot be empty")
                return
            existing = {o.name for o in (self._cell_config.objects or [])}
            if name in existing:
                QMessageBox.warning(dlg, "Add Object",
                                     f"Name '{name}' already exists")
                return
            mesh = mesh_edit.text().strip()
            if not mesh:
                QMessageBox.warning(dlg, "Add Object", "Mesh path is required")
                return
            # Heuristic: if the user adds an object named/pathed "gripper", suggest
            # the correct workflow (Add Gripper) instead of Add Object — gripper differs
            # in schema (TCP offset, follows flange).
            if "gripper" in name.lower() or "gripper" in mesh.lower():
                ret = QMessageBox.question(dlg, "Is this a gripper?",
                    "The name/path contains 'gripper'. Grippers have a separate "
                    "workflow (Edit → Add Gripper) to attach to the robot flange "
                    "and set the correct TCP offset.\n\nContinue with a normal Add Object?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if ret != QMessageBox.StandardButton.Yes:
                    return
            xyz, rpy = self._read_pose(pw)
            try:
                new_obj = ObjectConfig(
                    name=name, mesh=mesh,
                    parent_frame=frame_combo.currentData(),
                    pose=PoseConfig(xyz_mm=xyz, rpy_deg=rpy))
            except Exception as e:                          # noqa: BLE001
                QMessageBox.warning(dlg, "Add Object", f"Invalid: {e}")
                return
            if self._cell_config.objects is None:
                self._cell_config.objects = []
            self._cell_config.objects.append(new_obj)
            self._refresh_cell_tree()
            self._load_cell_assets()
            # Compute world coords for status (helps user locate in viewport)
            parent_xyz = (0.0, 0.0, 0.0)
            if new_obj.parent_frame:
                for fr in (self._cell_config.frames or []):
                    if fr.name == new_obj.parent_frame:
                        parent_xyz = fr.pose.xyz_mm; break
            wx = parent_xyz[0] + xyz[0]
            wy = parent_xyz[1] + xyz[1]
            wz = parent_xyz[2] + xyz[2]
            self._set_status(
                f"Added object '{name}' at world=({wx:.0f}, {wy:.0f}, "
                f"{wz:.0f}) mm", level="ok")
            dlg.accept()
        btns.accepted.connect(_ok); btns.rejected.connect(dlg.reject)
        v.addWidget(btns)
        preview = self._attach_preview(dlg, v)

        def _refresh_preview():
            self._set_preview(preview, lambda sz: self._render_mesh_thumbnail(
                mesh_edit.text().strip(), [0.74, 0.76, 0.78], size=sz, zoom=0.82))
        mesh_edit.editingFinished.connect(_refresh_preview)
        btn_pick.clicked.connect(_refresh_preview)          # update sau Browse
        _refresh_preview()
        dlg.exec()

    def _show_add_frame_dlg(self) -> None:
        self._ensure_cell_config()
        # Default pose: on top of the worktable if present (common teach position)
        # else (500, 0, 500) world coords (in front of the robot)
        wt = self._cell_config.worktable
        if wt is not None:
            h = self._mesh_height_mm(wt.mesh) or 500.0
            default_xyz = (wt.pose.xyz_mm[0], wt.pose.xyz_mm[1],
                            wt.pose.xyz_mm[2] + h)
        else:
            default_xyz = (500.0, 0.0, 500.0)

        dlg = QDialog(self); dlg.setWindowTitle("Add Frame")
        dlg.setMinimumWidth(360)
        v = QVBoxLayout(dlg)
        form = QFormLayout()
        name_edit = QLineEdit(); name_edit.setPlaceholderText("e.g., PickZone")
        form.addRow("Name:", name_edit)
        parent_combo = QComboBox()
        parent_combo.addItem("(none = base)", None)
        for f in self._cell_config.frames or []:
            parent_combo.addItem(f.name, f.name)
        form.addRow("Parent:", parent_combo)
        v.addLayout(form)
        v.addWidget(QLabel("<b>Pose relative to parent:</b>"))
        pose_form, pw = self._make_pose_form(xyz=default_xyz)
        v.addLayout(pose_form)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                 | QDialogButtonBox.StandardButton.Cancel)
        def _ok():
            name = name_edit.text().strip()
            if not name:
                QMessageBox.warning(dlg, "Add Frame", "Name cannot be empty")
                return
            existing = {f.name for f in (self._cell_config.frames or [])}
            if name in existing:
                QMessageBox.warning(dlg, "Add Frame",
                                     f"Name '{name}' already exists")
                return
            xyz, rpy = self._read_pose(pw)
            new_fr = FrameConfig(
                name=name, parent=parent_combo.currentData(),
                pose=PoseConfig(xyz_mm=xyz, rpy_deg=rpy))
            if self._cell_config.frames is None:
                self._cell_config.frames = []
            self._cell_config.frames.append(new_fr)
            # Update jog/IK tool/ref frames (ref_frames read from cell.frames)
            self._ref_frames = _build_ref_frames(self._cell_config)
            self._refresh_tool_ref_combos_if_present()
            self._refresh_cell_tree()
            self._set_status(f"Added frame '{name}'", level="ok")
            dlg.accept()
        btns.accepted.connect(_ok); btns.rejected.connect(dlg.reject)
        v.addWidget(btns)
        dlg.exec()

    # ── Edit dialogs ──────────────────────────────────────────────────

    def _show_edit_frame_dlg(self, fr: FrameConfig) -> None:
        if fr is None: return
        dlg = QDialog(self); dlg.setWindowTitle(f"Edit Frame: {fr.name}")
        v = QVBoxLayout(dlg)
        form = QFormLayout()
        form.addRow("Name:", QLabel(f"<b>{fr.name}</b>  (rename: delete & re-add)"))
        v.addLayout(form)
        v.addWidget(QLabel("<b>Pose:</b>"))
        pose_form, pw = self._make_pose_form(fr.pose.xyz_mm, fr.pose.rpy_deg)
        v.addLayout(pose_form)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                 | QDialogButtonBox.StandardButton.Cancel)
        def _ok():
            xyz, rpy = self._read_pose(pw)
            fr.pose = PoseConfig(xyz_mm=xyz, rpy_deg=rpy)
            self._ref_frames = _build_ref_frames(self._cell_config)
            self._refresh_tool_ref_combos_if_present()
            self._refresh_cell_tree()
            self._set_status(f"Updated frame '{fr.name}'", level="ok")
            dlg.accept()
        btns.accepted.connect(_ok); btns.rejected.connect(dlg.reject)
        v.addWidget(btns)
        dlg.exec()

    def _show_edit_object_dlg(self, obj: ObjectConfig) -> None:
        if obj is None: return
        dlg = QDialog(self); dlg.setWindowTitle(f"Edit Object: {obj.name}")
        dlg.setMinimumWidth(420)
        v = QVBoxLayout(dlg)
        form = QFormLayout()
        form.addRow("Name:", QLabel(f"<b>{obj.name}</b>"))
        form.addRow("Mesh:", QLabel(obj.mesh))
        frame_combo = QComboBox()
        frame_combo.addItem("(base / none)", None)
        for f in self._cell_config.frames or []:
            frame_combo.addItem(f.name, f.name)
        # Pre-select
        if obj.parent_frame is not None:
            for i in range(frame_combo.count()):
                if frame_combo.itemData(i) == obj.parent_frame:
                    frame_combo.setCurrentIndex(i); break
        form.addRow("Parent frame:", frame_combo)
        v.addLayout(form)
        v.addWidget(QLabel("<b>Offset from parent frame:</b>"))
        init_xyz = obj.pose.xyz_mm if obj.pose else (0.0, 0.0, 0.0)
        init_rpy = obj.pose.rpy_deg if obj.pose else (0.0, 0.0, 0.0)
        pose_form, pw = self._make_pose_form(init_xyz, init_rpy)
        v.addLayout(pose_form)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                 | QDialogButtonBox.StandardButton.Cancel)
        def _ok():
            xyz, rpy = self._read_pose(pw)
            obj.parent_frame = frame_combo.currentData()
            obj.pose = PoseConfig(xyz_mm=xyz, rpy_deg=rpy)
            self._refresh_cell_tree()
            self._load_cell_assets()
            self._set_status(f"Updated object '{obj.name}'", level="ok")
            dlg.accept()
        btns.accepted.connect(_ok); btns.rejected.connect(dlg.reject)
        v.addWidget(btns)
        dlg.exec()

    def _show_edit_robot_dlg(self, rob) -> None:
        if rob is None: return
        dlg = QDialog(self); dlg.setWindowTitle("Edit Robot Base Pose")
        v = QVBoxLayout(dlg)
        form = QFormLayout()
        form.addRow("Robot:", QLabel(f"<b>{rob.name}</b>"))
        v.addLayout(form)
        v.addWidget(QLabel("<b>Base pose (world frame):</b>"))
        pose_form, pw = self._make_pose_form(rob.pose.xyz_mm, rob.pose.rpy_deg)
        v.addLayout(pose_form)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                 | QDialogButtonBox.StandardButton.Cancel)
        def _ok():
            xyz, rpy = self._read_pose(pw)
            rob.pose = PoseConfig(xyz_mm=xyz, rpy_deg=rpy)
            self._base_xyz = tuple(xyz)
            self._base_rpy = tuple(rpy)
            # Rebuild URDF model with the new base — INCLUDING rpy (was dropped, so a
            # base rotation was silently ignored in the model/3D view).
            if self._model is not None:
                self._model = gp7_urdf(
                    base_xyz_mm=self._base_xyz,
                    base_rpy_rad=tuple(math.radians(d) for d in rpy))
                self._apply_joints_main(self._joints)
            self._refresh_cell_tree()
            self._set_status("Updated robot base pose", level="ok")
            dlg.accept()
        btns.accepted.connect(_ok); btns.rejected.connect(dlg.reject)
        v.addWidget(btns)
        dlg.exec()

    def _show_edit_single_dlg(self, kind: str, ref) -> None:
        """Edit single-instance cell items (Worktable, Pedestal, Floor,
        Camera, Camera Mount). Form: mesh + pose [+ color]."""
        if ref is None: return
        labels = {"worktable": "Worktable", "robot_pedestal": "Pedestal",
                  "floor": "Floor", "camera": "Camera",
                  "camera_mount": "Camera Mount"}
        dlg = QDialog(self); dlg.setWindowTitle(f"Edit {labels.get(kind, kind)}")
        dlg.setMinimumWidth(420)
        v = QVBoxLayout(dlg)
        form = QFormLayout()
        # Mesh (camera has no mesh)
        mesh_edit = None
        if kind != "camera":
            mesh_edit = QLineEdit(getattr(ref, "mesh", "") or "")
            mesh_row = QHBoxLayout()
            mesh_row.addWidget(mesh_edit, 1)
            btn_pick = QPushButton("Browse…")
            def _pick():
                p, _ = QFileDialog.getOpenFileName(
                    dlg, "Pick mesh", str(self._project_root / "models"),
                    "Mesh files (*.stl *.obj);;All (*.*)")
                if p:
                    try:
                        p = str(Path(p).resolve().relative_to(
                            self._project_root.resolve()))
                    except ValueError: pass
                    mesh_edit.setText(p)
            btn_pick.clicked.connect(_pick)
            mesh_row.addWidget(btn_pick)
            form.addRow("Mesh:", mesh_row)
        # Camera: type/model/mount + intrinsics (fov + size). Camera has no mesh.
        cam_w: dict = {}
        if kind == "camera":
            cam_w["type"] = QComboBox(); cam_w["type"].addItems(["virtual", "real"])
            cam_w["type"].setCurrentText(getattr(ref, "type", "virtual"))
            cam_w["model"] = QLineEdit(getattr(ref, "model", "") or "")
            cam_w["mount"] = QComboBox()
            cam_w["mount"].addItems(["eye_to_hand", "eye_in_hand"])
            cam_w["mount"].setCurrentText(getattr(ref, "mount", "eye_to_hand"))
            form.addRow("Type:", cam_w["type"])
            form.addRow("Model:", cam_w["model"])
            form.addRow("Mount:", cam_w["mount"])
            intr = getattr(ref, "intrinsics", None)
            cam_w["fov"] = QDoubleSpinBox(); cam_w["fov"].setRange(1.0, 179.0)
            cam_w["fov"].setSuffix(" °")
            cam_w["fov"].setValue(float(getattr(intr, "fov_deg", None) or 87.0))
            sz = getattr(intr, "size_px", (1280, 720)) if intr else (1280, 720)
            cam_w["w"] = QSpinBox(); cam_w["w"].setRange(100, 8192)
            cam_w["h"] = QSpinBox(); cam_w["h"].setRange(100, 8192)
            cam_w["w"].setValue(int(sz[0])); cam_w["h"].setValue(int(sz[1]))
            form.addRow("Horizontal FOV:", cam_w["fov"])
            form.addRow("Width (px):", cam_w["w"])
            form.addRow("Height (px):", cam_w["h"])
        v.addLayout(form)
        v.addWidget(QLabel("<b>Pose (world frame):</b>"))
        pose_form, pw = self._make_pose_form(ref.pose.xyz_mm, ref.pose.rpy_deg)
        v.addLayout(pose_form)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                 | QDialogButtonBox.StandardButton.Cancel)
        def _ok():
            xyz, rpy = self._read_pose(pw)
            ref.pose = PoseConfig(xyz_mm=xyz, rpy_deg=rpy)
            if mesh_edit is not None:
                ref.mesh = mesh_edit.text().strip()
            if kind == "camera":
                ref.type = cam_w["type"].currentText()
                ref.model = cam_w["model"].text().strip() or None
                ref.mount = cam_w["mount"].currentText()
                old = getattr(ref, "intrinsics", None)
                # Preserve real pixel intrinsics (fx/fy/cx/cy) if present — only update
                # fov + size from the form (real intrinsics take priority when drawing the frustum).
                ref.intrinsics = CameraIntrinsics(
                    fov_deg=cam_w["fov"].value(),
                    size_px=(cam_w["w"].value(), cam_w["h"].value()),
                    fx=getattr(old, "fx", None), fy=getattr(old, "fy", None),
                    cx=getattr(old, "cx", None), cy=getattr(old, "cy", None),
                    focal_length_mm=getattr(old, "focal_length_mm", None))
            self._refresh_cell_tree()
            self._load_cell_assets()
            self._set_status(f"Updated {labels.get(kind, kind)}", level="ok")
            dlg.accept()
        btns.accepted.connect(_ok); btns.rejected.connect(dlg.reject)
        v.addWidget(btns)
        dlg.exec()

    def _show_add_camera_dlg(self) -> None:
        """Add Camera (CameraConfig) — unified RoboDK-style camera node:
        type/model/mount + pose (extrinsics) + intrinsics (FOV + size). Draws
        frustum in the scene. Real pixel intrinsics filled later via D455 dock →
        'Sync Camera → Cell'."""
        self._ensure_cell_config()
        if self._cell_config.camera is not None:
            QMessageBox.information(
                self, "Add Camera",
                "Cell already has a Camera. Double-click the Camera node to Edit.")
            return
        dlg = QDialog(self); dlg.setWindowTitle("Add Camera")
        dlg.setMinimumWidth(420)
        v = QVBoxLayout(dlg)
        form = QFormLayout()
        cb_type = QComboBox(); cb_type.addItems(["virtual", "real"])
        ed_model = QLineEdit("Intel RealSense D455")
        cb_mount = QComboBox(); cb_mount.addItems(["eye_to_hand", "eye_in_hand"])
        sp_fov = QDoubleSpinBox(); sp_fov.setRange(1.0, 179.0)
        sp_fov.setSuffix(" °"); sp_fov.setValue(87.0)
        sp_w = QSpinBox(); sp_w.setRange(100, 8192); sp_w.setValue(1280)
        sp_h = QSpinBox(); sp_h.setRange(100, 8192); sp_h.setValue(720)
        form.addRow("Type:", cb_type)
        form.addRow("Model:", ed_model)
        form.addRow("Mount:", cb_mount)
        form.addRow("Horizontal FOV:", sp_fov)
        form.addRow("Width (px):", sp_w)
        form.addRow("Height (px):", sp_h)
        v.addLayout(form)
        v.addWidget(QLabel("<b>Pose (world frame):</b>"))
        # Default: 700mm above table looking down (D455 sweet spot, matches sample YAML).
        pose_form, pw = self._make_pose_form(
            xyz=(700.0, 0.0, 1200.0), rpy=(180.0, 0.0, 0.0))
        v.addLayout(pose_form)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                 | QDialogButtonBox.StandardButton.Cancel)

        def _ok():
            xyz, rpy = self._read_pose(pw)
            try:
                cam = CameraConfig(
                    type=cb_type.currentText(),
                    model=ed_model.text().strip() or None,
                    mount=cb_mount.currentText(),
                    pose=PoseConfig(xyz_mm=xyz, rpy_deg=rpy),
                    intrinsics=CameraIntrinsics(
                        fov_deg=sp_fov.value(),
                        size_px=(sp_w.value(), sp_h.value())))
            except Exception as e:                          # noqa: BLE001
                QMessageBox.warning(dlg, "Add Camera", f"Invalid: {e}")
                return
            self._cell_config.camera = cam
            self._refresh_cell_tree()
            self._load_cell_assets()
            self._set_status("Added Camera", level="ok")
            dlg.accept()
        btns.accepted.connect(_ok); btns.rejected.connect(dlg.reject)
        v.addWidget(btns)
        dlg.exec()

    def _show_add_single_dlg(self, kind: str) -> None:
        """Add single-instance item (worktable/pedestal/floor/camera_mount)."""
        self._ensure_cell_config()
        # Defaults by kind — avoid placing everything at (0,0,0) (overlaps robot)
        # Worktable: 700mm in front of robot, rotated 90° (long side along Y)
        # Pedestal: below robot (Z=0)
        # Floor: below the entire cell (Z=0)
        # Camera mount: above table (Z=500mm)
        defaults = {
            "worktable":     ((700.0, 0.0,   0.0), (0.0, 0.0, 90.0),
                              "models/worktable.stl"),
            "robot_pedestal":((0.0,   0.0,   0.0), (0.0, 0.0,  0.0),
                              "models/pedestal.stl"),
            "floor":         ((300.0, 0.0,   0.0), (0.0, 0.0,  0.0),
                              "models/floor.stl"),
            "camera_mount":  ((700.0, 0.0, 500.0), (0.0, 0.0,  0.0),
                              "models/camera_mount.stl"),
        }
        def_xyz, def_rpy, def_mesh = defaults.get(
            kind, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), f"models/{kind}.stl"))
        type_map = {"worktable": (WorktableConfig, "Worktable"),
                    "robot_pedestal": (PedestalConfig, "Pedestal"),
                    "floor": (FloorConfig, "Floor"),
                    "camera_mount": (CameraMountConfig, "Camera Mount")}
        if kind not in type_map:
            return
        Cls, label = type_map[kind]
        prev_color = {"worktable": [0.66, 0.67, 0.66],
                      "robot_pedestal": [0.28, 0.29, 0.31],
                      "floor": [0.50, 0.52, 0.55],
                      "camera_mount": [0.68, 0.69, 0.71]}.get(kind, [0.7, 0.7, 0.7])
        dlg = QDialog(self); dlg.setWindowTitle(f"Add {label}")
        dlg.setMinimumWidth(420)
        v = QVBoxLayout()
        form = QFormLayout()
        mesh_edit = QLineEdit(def_mesh)
        mesh_edit.setPlaceholderText(f"models/{kind}.stl")
        mesh_row = QHBoxLayout()
        mesh_row.addWidget(mesh_edit, 1)
        btn_pick = QPushButton("Browse…")
        def _pick():
            p, _ = QFileDialog.getOpenFileName(
                dlg, "Pick mesh", str(self._project_root / "models"),
                "Mesh files (*.stl *.obj);;All (*.*)")
            if p:
                try:
                    p = str(Path(p).resolve().relative_to(
                        self._project_root.resolve()))
                except ValueError: pass
                mesh_edit.setText(p)
        btn_pick.clicked.connect(_pick)
        mesh_row.addWidget(btn_pick)
        form.addRow("Mesh:", mesh_row)
        v.addLayout(form)
        v.addWidget(QLabel("<b>Pose:</b>"))
        pose_form, pw = self._make_pose_form(xyz=def_xyz, rpy=def_rpy)
        v.addLayout(pose_form)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                 | QDialogButtonBox.StandardButton.Cancel)
        def _ok():
            mesh = mesh_edit.text().strip()
            if not mesh:
                QMessageBox.warning(dlg, f"Add {label}",
                                     "Mesh path is required")
                return
            xyz, rpy = self._read_pose(pw)
            try:
                obj = Cls(mesh=mesh,
                          pose=PoseConfig(xyz_mm=xyz, rpy_deg=rpy))
            except Exception as e:                          # noqa: BLE001
                QMessageBox.warning(dlg, f"Add {label}", f"Invalid: {e}")
                return
            setattr(self._cell_config, kind, obj)
            self._refresh_cell_tree()
            self._load_cell_assets()
            self._set_status(f"Added {label}", level="ok")
            # Auto-detect: Pedestal added while robot is loaded ⇒ offer to lift robot
            # to pedestal top so the stand rests on the pedestal (sit-on relationship).
            if kind == "robot_pedestal" and self._model is not None:
                self._offer_lift_robot_onto_pedestal(obj)
            dlg.accept()
        btns.accepted.connect(_ok); btns.rejected.connect(dlg.reject)
        v.addWidget(btns)
        preview = self._attach_preview(dlg, v)

        def _refresh_preview():
            self._set_preview(preview, lambda sz: self._render_mesh_thumbnail(
                mesh_edit.text().strip(), prev_color, size=sz, zoom=0.82))
        mesh_edit.editingFinished.connect(_refresh_preview)
        btn_pick.clicked.connect(_refresh_preview)          # update sau Browse
        _refresh_preview()
        dlg.exec()

    # ── Delete handlers ───────────────────────────────────────────────

    def _delete_frame(self, fr: FrameConfig) -> None:
        if fr is None: return
        # Orphan check: which objects reference this frame?
        orphans = [o.name for o in (self._cell_config.objects or [])
                   if o.parent_frame == fr.name]
        msg = f"Delete frame '{fr.name}'?"
        if orphans:
            msg += (f"\n\n{len(orphans)} object(s) reference this frame: "
                    f"{', '.join(orphans)}. "
                    "After deletion, those objects will revert to base.")
        ret = QMessageBox.question(self, "Delete Frame", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ret != QMessageBox.StandardButton.Yes: return
        self._cell_config.frames = [f for f in self._cell_config.frames
                                     if f.name != fr.name]
        # Reset parent_frame of orphan objects → None
        for o in (self._cell_config.objects or []):
            if o.parent_frame == fr.name:
                o.parent_frame = None
        self._ref_frames = _build_ref_frames(self._cell_config)
        # Deleting a frame shrinks _ref_frames; clamp the selected index so the
        # live readout/jog (which index _ref_frames[_ref_idx]) cannot IndexError
        # on a now-out-of-range selection. _refresh_tool_ref_combos rebuilds the
        # combo items with signals blocked, so it does NOT resync _ref_idx itself.
        self._ref_idx = max(0, min(self._ref_idx, len(self._ref_frames) - 1))
        self._refresh_tool_ref_combos_if_present()
        if getattr(self, "_ref_combo", None) is not None:
            self._ref_combo.setCurrentIndex(self._ref_idx)
        self._refresh_cell_tree()
        self._set_status(f"Deleted frame '{fr.name}'", level="ok")

    def _delete_object(self, obj: ObjectConfig) -> None:
        if obj is None: return
        ret = QMessageBox.question(self, "Delete Object",
            f"Delete object '{obj.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ret != QMessageBox.StandardButton.Yes: return
        self._cell_config.objects = [o for o in self._cell_config.objects
                                      if o.name != obj.name]
        self._refresh_cell_tree()
        self._load_cell_assets()
        self._set_status(f"Deleted object '{obj.name}'", level="ok")

    def _delete_single(self, kind: str) -> None:
        labels = {"worktable": "Worktable", "robot_pedestal": "Pedestal",
                  "floor": "Floor", "camera_mount": "Camera Mount"}
        label = labels.get(kind, kind)
        ret = QMessageBox.question(self, f"Delete {label}",
            f"Delete {label}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ret != QMessageBox.StandardButton.Yes: return
        prev_ref = getattr(self._cell_config, kind, None)
        setattr(self._cell_config, kind, None)
        # Remove actor from the viewport
        try: self._plotter.remove_actor(kind)
        except Exception: pass                              # noqa: BLE001
        self._refresh_cell_tree()
        self._plotter.render()
        self._set_status(f"Deleted {label}", level="ok")
        # Symmetric auto-detect: Pedestal deleted while robot base Z>~330
        # ⇒ offer to lower robot to Z=330 (stand touching floor).
        if (kind == "robot_pedestal" and self._model is not None
                and prev_ref is not None):
            self._offer_lower_robot_off_pedestal(prev_ref)

    # ── Component preview thumbnails (Add dialogs) ────────────────────
    _PREVIEW_BG = [0.12, 0.12, 0.15]

    def _np_to_pixmap(self, img) -> QPixmap:
        """numpy RGB (H,W,3) uint8 → QPixmap (copy buffer for safe lifetime)."""
        img = np.ascontiguousarray(img[:, :, :3])
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    def _make_preview_label(self, size=(300, 300)) -> QLabel:
        """QLabel preview frame (shared style) — place in any layout."""
        lbl = QLabel("Generating preview…")
        lbl.setMinimumSize(size[0], size[1])
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("background:#16161c; border:1px solid #2a2a33; "
                          "border-radius:6px; color:#888;")
        return lbl

    def _attach_preview(self, dlg, content_layout, size=(300, 300)):
        """Wrap content (left) + QLabel preview (right) into dlg. Returns QLabel."""
        outer = QHBoxLayout(dlg)
        left = QWidget(); left.setLayout(content_layout)
        outer.addWidget(left, 0)
        lbl = self._make_preview_label(size)
        outer.addWidget(lbl, 1)
        return lbl

    def _set_preview(self, lbl, render_fn) -> None:
        """Defer render via QTimer so the dialog paints first, then fill the thumbnail.

        render_fn receives (w, h) = actual label size at render time
        → render off-screen at the correct aspect ratio, avoiding letterboxing (black bars)."""
        lbl.setText("Generating preview…")

        def _do():
            size = (max(120, lbl.width()), max(120, lbl.height()))
            try:
                pm = render_fn(size)
            except Exception as e:                          # noqa: BLE001
                logger.debug("preview render error: %s", e); pm = None
            if pm is not None:
                lbl.setPixmap(pm.scaled(
                    lbl.size(), Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
            else:
                lbl.setText("No preview\n(select a valid mesh)")
        QTimer.singleShot(0, _do)

    def _render_mesh_thumbnail(self, mesh_rel, color, size=(300, 300), zoom=1.05):
        """Render one STL off-screen → QPixmap. None if file missing/error.

        zoom<1 adds padding around the mesh (for large/flat components like floor, tray)."""
        if not mesh_rel:
            return None
        path = Path(mesh_rel)
        if not path.is_absolute():
            path = self._project_root / path
        if not path.exists():
            return None
        p = pv.Plotter(off_screen=True, window_size=[size[0], size[1]])
        try:
            p.set_background(self._PREVIEW_BG)
            p.add_mesh(pv.read(str(path)), color=color, smooth_shading=True)
            p.view_isometric()
            # Zoom gently: view_isometric already auto-fits; high zoom would clip the mesh.
            p.camera.zoom(zoom)
            img = p.screenshot(return_img=True)
        finally:
            p.close()
        return self._np_to_pixmap(img)

    def _render_robot_thumbnail(self, home_joints, size=(300, 300)):
        """Assemble 7 GP7 links at home pose (base at origin) off-screen → QPixmap."""
        mesh_dir = self._project_root / "models" / "gp7_links"
        if not mesh_dir.exists():
            return None
        model = gp7_urdf(base_xyz_mm=(0.0, 0.0, 0.0))
        frames = dict(link_frames_urdf(
            model, [math.radians(q) for q in home_joints]))
        p = pv.Plotter(off_screen=True, window_size=[size[0], size[1]])
        try:
            p.set_background(self._PREVIEW_BG)
            added = 0
            for key, fname, off in _GP7_MESH_MAP:
                fp = mesh_dir / fname
                if not fp.exists():
                    continue
                mesh = pv.read(str(fp))
                if any(off):
                    mesh.translate([off[0], off[1], off[2]], inplace=True)
                T = frames.get(key)
                if T is not None:
                    Tm = T.copy(); Tm[:3, 3] = T[:3, 3] / 1000.0
                    mesh.transform(Tm, inplace=True)
                p.add_mesh(mesh, color=list(_YASKAWA_BLUE), smooth_shading=True)
                added += 1
            if added == 0:
                return None
            p.view_isometric()
            # Do NOT zoom further: view_isometric already auto-fits the full robot.
            # zoom>1 would clip the upper arm (robot is tall at home pose).
            p.camera.zoom(1.05)
            img = p.screenshot(return_img=True)
        finally:
            p.close()
        return self._np_to_pixmap(img)

    # ── Auto pedestal-robot z-sync ────────────────────────────────────

    # GP7 stand built-in: 330mm (xem _GP7_MESH_MAP[base_link] offset -0.330m).
    _GP7_STAND_MM = 330.0

    def _mesh_height_mm(self, mesh_rel: str) -> float | None:
        """Read STL mesh → compute height (Z range) in mm. Return None if
        unreadable. Lightweight cache via instance dict to avoid re-parsing."""
        if not hasattr(self, "_mesh_height_cache"):
            self._mesh_height_cache: dict[str, float] = {}
        if mesh_rel in self._mesh_height_cache:
            return self._mesh_height_cache[mesh_rel]
        try:
            path = Path(mesh_rel)
            if not path.is_absolute():
                path = self._project_root / path
            if not path.exists():
                return None
            m = pv.read(str(path))
            # bounds = [xmin, xmax, ymin, ymax, zmin, zmax] in native mesh units.
            # STL meshes are conventionally in mm; after pv.read points remain mm.
            h = float(m.bounds[5] - m.bounds[4])
            self._mesh_height_cache[mesh_rel] = h
            return h
        except Exception:                                  # noqa: BLE001
            return None

    def _offer_lift_robot_onto_pedestal(self, pedestal_cfg) -> None:
        """Pedestal added after robot ⇒ ASK user whether to lift robot onto the
        pedestal top (Yes/No). User wants confirmation rather than an automatic base pose change.
        """
        ped_top = pedestal_cfg.pose.xyz_mm[2]
        h = self._mesh_height_mm(pedestal_cfg.mesh)
        if h is not None:
            ped_top += h
        target_z = ped_top + self._GP7_STAND_MM
        current_z = self._cell_config.robot.pose.xyz_mm[2]
        if abs(target_z - current_z) < 1.0:                # already in position
            return
        r = QMessageBox.question(
            self, "Lift robot onto pedestal",
            f"The pedestal just added reaches Z={ped_top:.0f} mm.\n\n"
            f"Lift the robot onto the top of the pedestal (base Z: {current_z:.0f} → "
            f"{target_z:.0f} mm) so the robot base sits on the pedestal top?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes)
        if r != QMessageBox.StandardButton.Yes:
            return
        self._set_robot_base_z(target_z)
        self._set_status(
            f"Lifted robot Z={current_z:.0f}→{target_z:.0f} mm "
            f"(stand on pedestal top)", level="ok")

    def _offer_lower_robot_off_pedestal(self, prev_pedestal_cfg) -> None:
        """Pedestal deleted ⇒ AUTO-LOWER robot to Z=330 (stand touching floor).
        Same UX rationale as _offer_lift_robot_onto_pedestal."""
        target_z = self._GP7_STAND_MM
        current_z = self._cell_config.robot.pose.xyz_mm[2]
        if abs(target_z - current_z) < 1.0:
            return
        self._set_robot_base_z(target_z)
        self._set_status(
            f"Auto-lowered robot Z={current_z:.0f}→{target_z:.0f} mm "
            f"(stand on floor)", level="ok")

    def _set_robot_base_z(self, z_mm: float) -> None:
        """Update robot base Z + rebuild URDF model + refresh viewport."""
        cur = self._cell_config.robot.pose
        self._cell_config.robot.pose = PoseConfig(
            xyz_mm=(cur.xyz_mm[0], cur.xyz_mm[1], z_mm),
            rpy_deg=cur.rpy_deg)
        self._base_xyz = (cur.xyz_mm[0], cur.xyz_mm[1], z_mm)
        self._model = gp7_urdf(base_xyz_mm=self._base_xyz)
        self._apply_joints_main(self._joints)
        self._refresh_cell_tree()

    def _load_robot_assets(self) -> None:
        """Robot STL meshes (independent of cell). Idempotent — only loads
        if not already loaded."""
        if any(k in self._link_actors for k in ("base_link", "link_s")):
            return
        self._load_robot_links()
        self._load_gripper()

    def _load_cell_assets(self) -> None:
        """Cell meshes (worktable, pedestal, objects) — depends on
        cell_config. Idempotent — clears old cell actors before adding.
        Fix B5: also pops stale entries from _link_actors (if any) to
        prevent _render_scene_frame from accessing dangling actor refs."""
        if self._cell_config is None:
            return
        # Old datasets removed/replaced → invalidate normal cache (key by
        # id(dataset) may be reused for a different mesh → stale normals).
        self._normal_cache.clear()
        cell_names = ("worktable", "robot_pedestal", "camera_mount", "floor")
        for name in cell_names:
            try:
                self._plotter.remove_actor(name)
            except Exception:                                   # noqa: BLE001
                pass
            self._link_actors.pop(name, None)
        for name in list(self._objects.keys()):
            try:
                self._plotter.remove_actor(name)
            except Exception:                                   # noqa: BLE001
                pass
            self._objects.pop(name, None)
            self._link_actors.pop(name, None)
        for name in self._CAM_VIZ_ACTORS:                   # old camera frustum
            try:
                self._plotter.remove_actor(name)
            except Exception:                               # noqa: BLE001
                pass
        self._add_cell_meshes()
        self._add_camera_frustum()
        # Respect View → Visibility → Camera frustum toggle (if disabled).
        if hasattr(self, "_act_cam_frustum") and not self._act_cam_frustum.isChecked():
            self._toggle_camera_frustum(False)
        # add_mesh() in _add_cell_meshes uses render=False (avoids re-rendering
        # after EACH mesh → appearing one-by-one). Single render here ⇒ all cell
        # meshes appear at once. Reset clipping range first: large meshes (floor/
        # table) expand bounds → prevents being clipped to a "rectangle" until mouse move.
        self._plotter.reset_camera_clipping_range()
        self._plotter.render()
        # New cell may have different object_classes → update Class dock combo.
        if hasattr(self, "_refresh_class_combo"):
            self._refresh_class_combo()

    def _add_floor(self) -> None:
        """Procedural reference floor (toggled via View → Visibility → Floor).
        Flat 6×6m plane, single solid color (no grid edges) — industrial
        finish look. Used as a reference plane when no FloorConfig.mesh exists
        (that is the real STL floor, loaded via Edit → Add Floor).
        """
        try:
            plane = pv.Plane(center=(0.0, 0.0, 0.0), direction=(0, 0, 1),
                              i_size=6.0, j_size=6.0,
                              i_resolution=1, j_resolution=1)
            self._floor_actor = self._plotter.add_mesh(
                plane,
                color=[0.42, 0.44, 0.48],         # neutral mid-gray
                show_edges=False,                  # NO grid (clean finish)
                name="__floor",
                lighting=True,
                ambient=0.3,                       # low specular reflection
                diffuse=0.7,
                specular=0.1,
                render=False)                      # toggle/loader renders once
        except Exception as e:                              # noqa: BLE001
            logger.debug("Floor error: %s", e)

    def _add_world_axes_triad(self) -> None:
        """Triad in-scene to align with robot base. RoboDK-style:
        RGB arrows with cylinder shaft + cone tip. XYZ labels OFF by default.

        FIX: SetVisibility(False) on CaptionActor2D does not suppress vtkAxesActor
        label rendering in many VTK versions. The reliable approach: clear text with
        SetXAxisLabelText(""). No text ⇒ no render. User re-enables via
        right-click → Show labels (text reverts to "X","Y","Z").
        """
        try:
            actor = vtk.vtkAxesActor()
            # World at floor level like RoboDK: small (120mm) to not overpower the robot.
            # Length reduced ~tip portion (cone 20% default) vs previous version;
            # larger cylinder + cone ratio for a stocky look.
            actor.SetTotalLength(0.12, 0.12, 0.12)
            actor.SetShaftType(0)                          # cylinder
            actor.SetCylinderRadius(0.035)
            actor.SetConeRadius(0.55)
            actor.SetConeResolution(24)
            # Force labels OFF by clearing text
            actor.SetXAxisLabelText("")
            actor.SetYAxisLabelText("")
            actor.SetZAxisLabelText("")
            self._plotter.add_actor(actor, name="__world_axes")
            self._world_axes_actor = actor
        except Exception as e:                              # noqa: BLE001
            logger.debug("World axes error: %s", e)

    def _setup_lighting(self) -> None:
        # pyvista default lighting is sufficient; custom lights can be added if needed.
        try:
            self._plotter.enable_shadows = False            # disable shadows for perf
        except Exception:                                   # noqa: BLE001
            pass

    def _load_robot_links(self) -> bool:
        """Load 7 ros-industrial STL files → add to plotter, save actors for animation.

        NOTE: tried ThreadPoolExecutor for STL parsing — 13% slower due to GIL +
        small per-file work (~10ms). pv.read calls C code but Python-side
        dataset wrapping still holds the GIL. Total STL load ~65ms (SSD warm cache),
        not worth parallelizing.
        """
        mesh_dir = self._project_root / "models" / "gp7_links"
        if not mesh_dir.exists():
            logger.warning("Robot mesh dir not found: %s", mesh_dir)
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
                    smooth_shading=True, pbr=False, render=False)
                self._link_actors[key] = actor
                loaded += 1
            except Exception as e:                          # noqa: BLE001
                logger.debug("Error loading mesh %s: %s", path, e)
        logger.info("GP7AppQt: %d/%d GP7 link mesh", loaded, len(_GP7_MESH_MAP))
        return loaded > 0

    def _load_gripper(self) -> None:
        """Load gripper mesh from cell_config.gripper, attach to link_tool0.

        Mesh local frame = flange (link_tool0). tcp_offset_xyz_mm shifts the mesh
        so that the TCP (tool tip) aligns with that offset from the flange. During animation,
        _render_scene_frame maps actor name "gripper" → frame "link_tool0".
        """
        cfg = getattr(self._cell_config, "gripper", None)
        if cfg is None or not getattr(cfg, "mesh", None):
            return
        path = Path(cfg.mesh)
        if not path.is_absolute():
            path = self._project_root / path
        if not path.exists():
            logger.warning("Gripper mesh not found: %s", path)
            return
        try:
            mesh = pv.read(str(path))
            mesh.points *= 0.001                            # mm → m
            # Translate mesh by tcp_offset (mm → m). User edits offset via Edit
            # Gripper → tcp_offset_xyz_mm updates → reload shows immediately.
            off = cfg.tcp_offset_xyz_mm
            mesh.translate([off[0] / 1000.0, off[1] / 1000.0,
                             off[2] / 1000.0], inplace=True)
            actor = self._plotter.add_mesh(
                mesh, color=[0.78, 0.78, 0.80], name="gripper",
                smooth_shading=True, render=False)
            self._link_actors["gripper"] = actor
        except Exception as e:                              # noqa: BLE001
            logger.debug("Gripper error: %s", e)

    def _add_cell_meshes(self) -> None:
        cfg = self._cell_config
        # Include floor in the loop — previously omitted, Add Floor did not load
        # STL mesh into the viewport.
        # Standard industrial color palette (RAL/real materials):
        #   worktable     RAL 7035 light grey (machine/workbench finish)
        #   robot_pedestal RAL 7016 anthracite (base/frame)
        #   camera_mount  anodized aluminium (80/20 extrusion)
        #   floor         neutral grey concrete/epoxy (not white)
        for attr, drgb in (("worktable", [0.66, 0.67, 0.66]),
                            ("robot_pedestal", [0.28, 0.29, 0.31]),
                            ("camera_mount", [0.68, 0.69, 0.71]),
                            ("floor", [0.50, 0.52, 0.55])):
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
            # Graspable object (bolt/part) = polished steel/zinc (light metallic grey),
            # replacing the old amber "lamp" color → standard machined-part finish.
            self._register_object(obj.name, obj.mesh, world_xyz_mm,
                                   rgb=[0.74, 0.76, 0.78])

    # ── Camera frustum (RoboDK-style camera item viz) ──────────────────
    _CAM_VIZ_ACTORS = ("__camera_frustum", "__camera_plane", "__camera_axes")

    def _add_camera_frustum(self) -> None:
        """Draw view frustum + axes for the cell `camera` at its pose.

        RoboDK-style camera item: extrinsics = camera.pose (world/base),
        frustum size from FOV (intrinsics). Optical axis = +Z local."""
        cfg = self._cell_config
        cam = getattr(cfg, "camera", None) if cfg is not None else None
        if cam is None:
            return
        try:
            intr = getattr(cam, "intrinsics", None)
            hfov, vfov = intr.hfov_vfov_deg() if intr is not None else (87.0, 56.0)
            # Pose: xyz_mm+rpy_deg → 4×4, translation mm→m, applied via UserMatrix.
            T = _xyz_rpy_to_matrix(
                cam.pose.xyz_mm[0], cam.pose.xyz_mm[1], cam.pose.xyz_mm[2],
                cam.pose.rpy_deg[0], cam.pose.rpy_deg[1], cam.pose.rpy_deg[2])
            T[:3, 3] /= 1000.0
            # Frustum depth = distance along the optical axis (+Z local) until
            # hitting the FLOOR (Z=0 world) → frustum extends "to the ground", matching real FOV.
            # Camera not pointing down ⇒ use default far. Clamp to D455 range.
            optical = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
            pz = float(T[2, 3])
            if optical[2] < -1e-3 and pz > 0.0:
                d = -pz / float(optical[2])                 # intersection of optical axis with floor
            else:
                d = _CAM_FRUSTUM_FAR_M
            d = float(np.clip(d, _CAM_FRUSTUM_MIN_M, _CAM_FRUSTUM_MAX_M))
            hw = d * math.tan(math.radians(hfov) / 2.0)
            hh = d * math.tan(math.radians(vfov) / 2.0)
            corners = np.array([[-hw, -hh, d], [hw, -hh, d],
                                [hw, hh, d], [-hw, hh, d]], dtype=float)
            pts = np.vstack([[0.0, 0.0, 0.0], corners])     # apex + 4 corners
            frustum = pv.PolyData(pts)
            frustum.lines = np.array([
                2, 0, 1, 2, 0, 2, 2, 0, 3, 2, 0, 4,         # apex→4 corners
                2, 1, 2, 2, 2, 3, 2, 3, 4, 2, 4, 1,         # rectangular base
            ])
            vtk_T = _numpy_to_vtk_matrix(T)
            actor = self._plotter.add_mesh(
                frustum, color=[0.40, 0.75, 1.0], style="wireframe",
                line_width=2, opacity=0.9, lighting=False,
                name="__camera_frustum", render=False)
            actor.SetUserMatrix(vtk_T)
            # Image plane (translucent) at frustum base — evokes sensor plane.
            plane = pv.PolyData(corners, np.array([4, 0, 1, 2, 3]))
            pactor = self._plotter.add_mesh(
                plane, color=[0.30, 0.55, 0.85], opacity=0.16,
                lighting=False, name="__camera_plane", render=False)
            pactor.SetUserMatrix(vtk_T)
            # Camera axes (RGB triad) at pose — reuse vtkAxesActor.
            ax = vtk.vtkAxesActor()
            ax.SetTotalLength(0.08, 0.08, 0.08)
            ax.SetShaftType(0); ax.SetCylinderRadius(0.04)
            ax.SetConeRadius(0.5); ax.SetConeResolution(20)
            ax.SetXAxisLabelText(""); ax.SetYAxisLabelText("")
            ax.SetZAxisLabelText("")
            tf = vtk.vtkTransform(); tf.SetMatrix(vtk_T)
            ax.SetUserTransform(tf)
            self._plotter.add_actor(ax, name="__camera_axes")
        except Exception as e:                              # noqa: BLE001
            logger.debug("Camera frustum error: %s", e)

    def _toggle_camera_frustum(self, visible: bool) -> None:
        """Hide/show the frustum + image plane + camera axes."""
        for n in self._CAM_VIZ_ACTORS:
            actor = self._plotter.renderer.actors.get(n)
            if actor is not None:
                actor.SetVisibility(bool(visible))
        self._plotter.render()

    def _add_static_mesh(self, name, mesh_rel, xyz_mm, rpy_deg, rgb) -> None:
        """Add one static mesh (worktable/pedestal/floor/camera_mount/etc.).

        Mesh geometry stays AT ORIGIN (identity); pose applied via SetUserMatrix.
        This allows vtkBoxWidget drag-to-move to work — the widget
        modifies UserMatrix without touching mesh geometry. Previously baking
        the transform into mesh.points caused a "double-transform" bug.
        """
        try:
            path = Path(mesh_rel)
            if not path.is_absolute():
                path = self._project_root / path
            if not path.exists():
                return
            mesh = pv.read(str(path))
            mesh.points *= 0.001                              # mm → m
            actor = self._plotter.add_mesh(mesh, color=rgb, name=name,
                                              smooth_shading=True, render=False)
            # Pose qua UserMatrix (translation: mm → m)
            T = _xyz_rpy_to_matrix(xyz_mm[0], xyz_mm[1], xyz_mm[2],
                                     rpy_deg[0], rpy_deg[1], rpy_deg[2])
            T[:3, 3] /= 1000.0
            actor.SetUserMatrix(_numpy_to_vtk_matrix(T))
        except Exception as e:                              # noqa: BLE001
            logger.debug("Static mesh '%s' error: %s", mesh_rel, e)

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
                                              smooth_shading=True, render=False)
            world_T = np.eye(4)
            world_T[:3, 3] = [v / 1000.0 for v in world_xyz_mm]
            actor.SetUserMatrix(_numpy_to_vtk_matrix(world_T))
            self._objects[name] = {
                "actor": actor,
                "world_T": world_T.copy(),
                "initial_world_T": world_T.copy(),
            }
        except Exception as e:                              # noqa: BLE001
            logger.debug("Object '%s' error: %s", name, e)

    def _set_camera_preset(self, name: str) -> None:
        """Switch camera to preset + auto-adjust projection per CAD convention:
          • **Iso** → Perspective (natural 3D feel for navigation/demo)
          • **Top/Front/Back/Right/Left** → Orthographic (engineering view —
            accurate measurement, no perspective distortion)
        """
        eye, ctr, up = self._CAM_PRESETS.get(name, self._CAM_PRESETS["Iso"])
        try:
            self._plotter.camera_position = [eye, ctr, up]
            # CAD convention: orthographic cho orthogonal views, perspective cho Iso
            is_perspective = (name == "Iso")
            self._plotter.camera.SetParallelProjection(not is_perspective)
            self._plotter.render()
            # Sync Perspective view menu toggle (do NOT fire toggled signal to
            # avoid infinite recursion via _on_toggle_perspective).
            if hasattr(self, "_act_perspective"):
                self._set_toggle(self._act_perspective, is_perspective)
        except Exception as e:                              # noqa: BLE001
            logger.debug("camera preset error: %s", e)
        # Sync menu visual: exclusive — only `name` checked, others unchecked.
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
            logger.debug("Toggle %s error: %s", name, e)

    def _toggle_floor(self, visible: bool) -> None:
        """Floor lazy-loaded: first enable calls _add_floor() to build the mesh,
        subsequent calls only SetVisibility."""
        try:
            actor = self._plotter.renderer.actors.get("__floor")
            if actor is None:
                if not visible:
                    return
                self._add_floor()                          # build mesh on first enable
            else:
                actor.SetVisibility(bool(visible))
            # The 6×6m floor expands scene bounds → old camera clipping range
            # clips the mesh → looks like a "rectangle" until mouse move (which
            # triggers ResetCameraClippingRange). Reset now so floor shows fully.
            self._plotter.reset_camera_clipping_range()
            self._plotter.render()
        except Exception as e:                              # noqa: BLE001
            logger.debug("Toggle floor error: %s", e)

    def _toggle_background(self, gradient_on: bool) -> None:
        """Toggle background between navy/purple gradient ↔ solid dark gray."""
        try:
            if gradient_on:
                self._plotter.set_background(
                    [95 / 255, 65 / 255, 175 / 255],            # bottom purple-blue
                    top=[5 / 255, 5 / 255, 28 / 255])           # top near-black navy
            else:
                self._plotter.set_background([0.10, 0.10, 0.13])
            self._plotter.render()
        except Exception as e:                              # noqa: BLE001
            logger.debug("Toggle background error: %s", e)

    # ══════════════════════════════════════════════════════════════════
    # View ops: Fit all / Perspective / Fullscreen / Close panels / Resize
    # ══════════════════════════════════════════════════════════════════
    def _on_fit_all(self) -> None:
        """Reset camera to fit the full scene (like RoboDK 'Fit all' Alt+7).
        After Fit all → camera no longer matches any preset → uncheck all presets.
        """
        try:
            self._plotter.reset_camera()
            self._plotter.render()
            if hasattr(self, "_cam_actions"):
                for act in self._cam_actions.values():
                    self._set_toggle(act, False)
            self._set_status("Camera reset (fit all)", level="ok")
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"Fit all error: {e}", level="err")

    def _on_toggle_perspective(self, perspective_on: bool) -> None:
        """Toggle perspective ↔ orthographic projection (engineering view)."""
        try:
            cam = self._plotter.camera
            cam.SetParallelProjection(not perspective_on)
            self._plotter.render()
            mode = "Perspective" if perspective_on else "Orthographic"
            self._set_status(f"Projection: {mode}", level="ok")
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"Projection toggle error: {e}", level="err")

    def _on_toggle_fullscreen(self, fullscreen_on: bool) -> None:
        """Toggle window fullscreen (F11). Hides menu + dock + status."""
        self._fullscreen = bool(fullscreen_on)
        if self._fullscreen:
            self.showFullScreen()
        else:
            self.showNormal()

    # ══════════════════════════════════════════════════════════════════
    # Menu checkbox sync — sync when user changes state via another route
    # (e.g. close dock via X title bar, or exit fullscreen via Esc)
    # ══════════════════════════════════════════════════════════════════
    # ── Left tab group: maintain tab order Cell | Control | Program | Camera ──
    # When enabling a panel, re-order tabs of all OPEN panels per the standard order
    # ⇒ newly opened panel's tab is to the RIGHT. ONLY uses tabifyDockWidget (no
    # splitDockWidget/resizeDocks so layout/viewport is NOT broken).
    def _group_docks_ordered(self) -> list:
        """Left dock group in standard tab order (left→right), excluding None."""
        return [d for d in (getattr(self, "_cell_tree_dock", None),
                            getattr(self, "_jog_dock", None),
                            getattr(self, "_program_dock", None),
                            getattr(self, "_camera_dock", None),
                            getattr(self, "_experiment_dock", None))
                if d is not None]

    def _open_dock_tab(self, dock, show: bool) -> None:
        """Enable/disable a left dock panel. When ENABLING: re-order tabs of open panels
        per standard order (Cell|Control|Program|Camera) then raise the just-enabled panel ⇒
        its tab sits to the right of previously open panels. When DISABLING: hide the panel."""
        if not show:
            dock.setVisible(False)
            return
        dock.setVisible(True)
        # Re-order tabs: tabify sequentially in standard order for open panels
        # (not floating). tabifyDockWidget(a, b) places b immediately right of a.
        ordered = [d for d in self._group_docks_ordered()
                   if d.isVisible() and not d.isFloating()]
        for prev, cur in zip(ordered, ordered[1:]):
            self.tabifyDockWidget(prev, cur)
        dock.raise_()

    def _connect_group_dock_redock(self) -> None:
        """Connect topLevelChanged for all left dock panels: when a panel STOPS FLOATING
        (dragged floating window dropped back into dock area), auto-merge it back into the tab group —
        Qt by default docks it to a separate area, not returning to the group."""
        for d in self._group_docks_ordered():
            d.topLevelChanged.connect(
                lambda floating, dock=d: self._on_group_dock_floated(
                    dock, floating))

    def _on_group_dock_floated(self, dock, floating: bool) -> None:
        """floating=False ⇒ panel just dropped back into dock area → re-tabify into
        group. Deferred via singleShot so Qt completes the drop before re-ordering tabs
        (re-tabifying immediately in the handler can cause layout state errors)."""
        if floating:
            return
        QTimer.singleShot(0, lambda: (
            self._open_dock_tab(dock, True)
            if dock.isVisible() and not dock.isFloating() else None))

    def _apply_active_dock_min(self, active_dock, pref_w: int) -> None:
        """DYNAMIC min-width for the left tab group (jog/cell/program/camera): the active
        dock gets min = pref_w (cannot shrink below it, only expand); all other docks
        get floor _CELL_DOCK_W to NOT force a wide region when they are not active."""
        for d in (getattr(self, "_jog_dock", None),
                  getattr(self, "_cell_tree_dock", None),
                  getattr(self, "_program_dock", None),
                  getattr(self, "_camera_dock", None)):
            if d is not None:
                d.setMinimumWidth(pref_w if d is active_dock else _CELL_DOCK_W)

    def _sync_jog_dock_check(self, visible: bool) -> None:
        """When controls panel is shown/hidden or raised (tab switch) → sync
        menu tick + dynamic min _JOG_DOCK_W (active cannot shrink) + resize region."""
        if hasattr(self, "_act_jog_dock"):
            self._set_toggle(self._act_jog_dock, bool(visible))
        if visible and hasattr(self, "_jog_dock"):
            self._apply_active_dock_min(self._jog_dock, _JOG_DOCK_W)
            QTimer.singleShot(0, lambda: self.resizeDocks(
                [self._jog_dock], [_JOG_DOCK_W], Qt.Orientation.Horizontal))

    def _sync_prog_dock_check(self, visible: bool) -> None:
        """Same as above for the program panel — sync menu tick + dynamic min _PROG_DOCK_W
        + resize region when program becomes the active tab (tabified with jog/cell)."""
        if hasattr(self, "_act_prog_dock"):
            self._set_toggle(self._act_prog_dock, bool(visible))
        if visible and hasattr(self, "_program_dock"):
            self._apply_active_dock_min(self._program_dock, _PROG_DOCK_W)
            QTimer.singleShot(0, lambda: self.resizeDocks(
                [self._program_dock], [_PROG_DOCK_W], Qt.Orientation.Horizontal))

    def _on_prog_tab_changed(self, index: int) -> None:
        """Shrink the tab area (Add Instruction) to the content of the current tab.

        QTabWidget defaults to the height of the TALLEST tab (QStackedWidget uses max sizeHint
        of all pages) → short tabs (e.g. Motion) have excess space below. Set maximumHeight
        of QTabWidget = current page sizeHint + tab bar → area shrinks/expands
        to match the viewed tab; spare space goes to the program list (stretch=3).
        """
        tabs = getattr(self, "_prog_tabs", None)
        if tabs is None:
            return
        cur = tabs.widget(index)
        if cur is None:
            return
        # page content + tab bar + frame padding (+12 to avoid clipping the last row)
        h = (cur.sizeHint().height()
             + tabs.tabBar().sizeHint().height() + 12)
        tabs.setMaximumHeight(h)
        tabs.updateGeometry()

    def _sync_cell_dock_check(self, visible: bool) -> None:
        """Same as above for the Cell components dock. Also forces width back to 180px
        when the dock is shown again — Qt by default stretches the dock to fill the area
        after setVisible(True), not restoring the previous size."""
        if hasattr(self, "_act_cell_dock"):
            self._set_toggle(self._act_cell_dock, bool(visible))
        if visible and hasattr(self, "_cell_tree_dock"):
            self._apply_active_dock_min(self._cell_tree_dock, _CELL_DOCK_W)
            QTimer.singleShot(0, lambda: self.resizeDocks(
                [self._cell_tree_dock], [_CELL_DOCK_W], Qt.Orientation.Horizontal))

    def changeEvent(self, event) -> None:
        """Catch window state changes (Esc exits fullscreen, maximize, etc.) →
        sync menu Fullscreen tick state.
        """
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            if hasattr(self, "_act_fullscreen"):
                is_fs = self.isFullScreen()
                self._set_toggle(self._act_fullscreen, is_fs)
                self._fullscreen = is_fs

    # (Removed: _resize_triads — menu entries gone, niche feature)

    # ══════════════════════════════════════════════════════════════════
    # Animation (joint → actor transform)
    # ══════════════════════════════════════════════════════════════════

    def _apply_joints_main(self, joints_deg: list[float]) -> None:
        """Main-thread slot: update joint state + actor transforms + readouts.
        Robot must be loaded — if None, skip silently (no-op)."""
        if self._model is None:
            return
        self._joints = list(joints_deg)
        self._render_scene_frame(joints_deg)
        self._refresh_joint_sliders()
        self._refresh_pose_readout()

    def _render_scene_frame(self, joints_deg: list[float]) -> None:
        """Compute FK via link_frames_urdf → SetUserMatrix on each link actor.

        Perf: cache FK result in self._cached_frames for _update_dynamic_frames
        to reuse (avoids 2-3 repeated link_frames_urdf/frame calls). Cache invalidated by
        joints tuple key (immutable hash).
        Bug fix B2: if move_widget is active on any actor, skip updating
        that actor (widget needs to own UserMatrix).
        """
        if self._model is None:
            return
        # FK cache key INCLUDES id(model) — if model is rebuilt (e.g. auto-lift
        # robot when adding pedestal, _set_robot_base_z rebuilds gp7_urdf), key
        # differs ⇒ cache miss ⇒ recompute with new base. WITHOUT id(model)
        # the cache returns stale frames ⇒ mesh renders at wrong position.
        jk = (id(self._model), tuple(round(q, 4) for q in joints_deg))
        cached = getattr(self, "_cached_fk", None)
        if cached is not None and cached[0] == jk:
            frames = cached[1]
        else:
            frames = dict(link_frames_urdf(
                self._model, [math.radians(q) for q in joints_deg]))
            self._cached_fk = (jk, frames)
        self._cached_frames = frames
        # Actor currently controlled by move widget — skip FK override
        move_target_name = (self._move_target[2]
                            if getattr(self, "_move_target", None)
                            else None)
        for link_key, actor in self._link_actors.items():
            if link_key == move_target_name:
                continue
            # gripper follows link_tool0
            frame_key = "link_tool0" if link_key == "gripper" else link_key
            T = frames.get(frame_key)
            if T is None:
                continue
            Tm = T.copy()
            Tm[:3, 3] = T[:3, 3] / 1000.0                  # mm → m
            try:
                actor.SetUserMatrix(_numpy_to_vtk_matrix(Tm))
            except Exception as e:                          # noqa: BLE001
                logger.debug("SetUserMatrix error (%s): %s", link_key, e)
        # Object follow gripper
        T_tool0_mm = frames.get("link_tool0")
        if T_tool0_mm is not None:
            for name, obj in self._objects.items():
                if name == move_target_name:
                    continue
                if (self._grasped_name == name
                        and self._grasp_offset_m is not None):
                    T_tool_m = T_tool0_mm.copy()
                    T_tool_m[:3, 3] = T_tool0_mm[:3, 3] / 1000.0
                    obj["world_T"] = T_tool_m @ self._grasp_offset_m
                try:
                    obj["actor"].SetUserMatrix(
                        _numpy_to_vtk_matrix(obj["world_T"]))
                except Exception as e:                      # noqa: BLE001
                    logger.debug("Object SetUserMatrix error (%s): %s", name, e)
        # Update visible frame triads (joint_N, tool, flange — these depend on joints)
        self._update_dynamic_frames()
        # Caller may suspend the (expensive) VTK render to coalesce several
        # sub-steps into a single repaint — e.g. the jog dial firing multiple
        # steps per event. Actor matrices are still updated above; only the
        # render is deferred to the caller.
        if not getattr(self, "_suspend_render", False):
            self._plotter.render()

    # Throttle animation render rate. 30Hz is smooth enough (eye perceives ~24Hz as film),
    # 25% fewer render() invocations than 40Hz.
    _ANIM_MAX_FPS: float = 30.0

    def _animate_to(self, end_deg: list[float], steps: int = 36, dt: float = 0.02,
                     stop_event: threading.Event | None = None,
                     pause_event: threading.Event | None = None) -> None:
        """Worker-thread slerp animation: emit joints_update per frame.

        Honors stop_event (abort) and pause_event (block while set).
        Throttle: skip emit if less than 1/_ANIM_MAX_FPS since last frame → reduces
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
            # Emit only when throttle interval elapsed OR final frame (ensures
            # arrival exactly at target).
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
        # Perf P7: skip if value unchanged (avoids blockSignals + setValue
        # round-trip for every animation tick, every joint).
        for i, slider in enumerate(self._joint_sliders):
            new_val = int(self._joints[i] * 100)
            if slider.value() != new_val:
                slider.blockSignals(True)
                slider.setValue(new_val)
                slider.blockSignals(False)
            new_text = f"{self._joints[i]:+7.2f}°"
            if self._joint_value_lbls[i].text() != new_text:
                self._joint_value_lbls[i].setText(new_text)

    def _fill_pose_row(self, labels: list[QLabel], T: np.ndarray) -> None:
        """Update 6 colored labels (X/Y/Z/Rx/Ry/Rz) from a 4x4 matrix.
        Format: value ONLY (no "X:" prefix) — narrow cells."""
        x, y, z, rx, ry, rz = _matrix_to_xyz_rpy_deg(T)
        for lbl, v in zip(labels, (x, y, z, rx, ry, rz)):
            lbl.setText(f"{v:.3f}")

    def _refresh_pose_readout(self) -> None:
        # Tool / Flange (static — only changes when Tool combo changes)
        T_flange_tool = self._tool_frames[self._tool_idx][1]
        self._fill_pose_row(self._tool_pose_lbls, T_flange_tool)
        # Reference / Base (static — only changes when Ref combo changes)
        T_base_ref = self._ref_frames[self._ref_idx][1]
        self._fill_pose_row(self._ref_pose_lbls, T_base_ref)
        # Tool / Reference (live — depends on joint state)
        T_world_tool = self._current_tool_world()
        if T_world_tool is not None:
            # When the reference is the robot base ("Base (0)"), show the pose
            # EXACTLY as the YRC1000 teach pendant does — BASE frame + the TOOL
            # Rz(180°) naming convention — so the panel cross-checks 1:1 with the
            # TP (no more Rz180/angle mismatch vs CURRENT POSITION). _pendant_pose
            # is verified <0.02 vs the real pendant. Other (user) frames keep the
            # generic tool-in-frame readout. Display-only: jog/IK/motion unaffected.
            pend = (self._pendant_pose()
                    if self._ref_frames[self._ref_idx][0] == "Base (0)"
                    else None)
            if pend is not None:
                for lbl, v in zip(self._tcp_pose_lbls, pend):
                    lbl.setText(f"{v:.3f}")
            else:
                T_world_base = np.eye(4); T_world_base[:3, 3] = self._base_xyz
                T_world_ref = T_world_base @ T_base_ref
                T_ref_tool = np.linalg.inv(T_world_ref) @ T_world_tool
                self._fill_pose_row(self._tcp_pose_lbls, T_ref_tool)

    def _pendant_pose(self) -> tuple[float, float, float, float, float, float] | None:
        """Current TCP expressed exactly as the YRC1000 teach-pendant shows it, so the
        operator can cross-check the app against the TP 1:1.

        Verified against a real GP7 send-pose measurement:
          • Position = TCP in the robot BASE frame (origin = _base_xyz, no offset;
            this matched the pendant exactly. GP7_CTRL_BASE_Z_MM is now 0.0 so the
            "Base (0)" reference frame matches too).
          • Orientation = base-frame orientation · Rz(180° about the TOOL Z) — the app
            tool0 frame and the controller TOOL frame differ by exactly that (proven:
            R_pendant = R_app·Rz180 to 0.001°). Tool Z (approach) is identical; only
            the roll naming flips.
        Returns (x,y,z mm, rx,ry,rz deg) or None.
        """
        T_world_tool = self._current_tool_world()
        if T_world_tool is None:
            return None
        T_world_base = np.eye(4); T_world_base[:3, 3] = self._base_xyz
        T_base_tool = np.linalg.inv(T_world_base) @ T_world_tool
        Rz180 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        T_pend = T_base_tool.copy()
        T_pend[:3, :3] = T_base_tool[:3, :3] @ Rz180
        return _matrix_to_xyz_rpy_deg(T_pend)

    # ── Other configurations (IK solutions) ───────────────────────────
    @staticmethod
    def _solution_turns(joints_deg: list[float]) -> list[int]:
        """Per-joint turn number = how many full ±360° revolutions a joint is wound
        away from its principal value in (-180°, 180°]. Yaskawa/RoboDK convention:
        turn 0 = principal solution; ±1 = one extra revolution (possible only on
        axes with range > 360°, e.g. GP7 J6 ±360°)."""
        turns = []
        for a in joints_deg:
            principal = ((a + 180.0) % 360.0) - 180.0
            turns.append(int(round((a - principal) / 360.0)))
        return turns

    @staticmethod
    def _turn_label(turns: list[int]) -> str:
        """Compact label of the non-zero turns, e.g. 'J6+1' or 'J4-1 J6+1'.
        'turn 0' when the solution is the principal (no winding)."""
        parts = [f"J{i + 1}{t:+d}" for i, t in enumerate(turns) if t != 0]
        return " ".join(parts) if parts else "turn 0"

    def _compute_configurations(self) -> list[dict]:
        """All IK solutions for the current TCP pose (Pieper analytical), enumerated
        like RoboDK. Each row is one joint solution reaching the same TCP. Empty if
        unreachable.

        Two things distinguish the rows:
          • Configuration — the (up to 8) posture regions Front/Rear · Elbow Up/Down ·
            Flip/Non-Flip. These ARE separated by singularities: moving between two of
            them requires crossing a singularity.
          • Joint turns — the SAME posture wound differently (±360°). The GP7 has wide
            axes (J6 ±360°, J4 ±190°, J3 spans 371°), so one posture can be reached at
            several turn counts that all stay within limits. RoboDK lists these too,
            which is why a pose can have "from 1 to more than 100" configurations.

        `include_turns=True` so the count matches RoboDK (dropping turns would show
        roughly half the rows).

        Perf P8: cache result by TCP pose tuple — if pose unchanged, reuse
        cache instead of calling Pieper IK again (~20-50ms). Invalidate when tool_idx
        changes or joints change (T is derived).
        """
        T = self._current_tool_world()
        if T is None:
            return []
        T_flange_tool = self._tool_frames[self._tool_idx][1]
        T_target_tool0 = T @ np.linalg.inv(T_flange_tool)
        # Cache key: round T_target_tool0 elements + tool_idx
        # (rounded for stable hash, avoids float-equality flakiness)
        key = (tuple(round(v, 4) for v in T_target_tool0.ravel()),
               self._tool_idx)
        cached = getattr(self, "_pieper_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            result = inverse_kinematics_pieper_gp7_tagged(
                self._model, T_target_tool0, include_turns=True)
            for c in result:                # annotate per-joint turn numbers
                c["turns"] = self._solution_turns(c["joints_deg"])
                c["turn_label"] = self._turn_label(c["turns"])
            self._pieper_cache = (key, result)
            return result
        except Exception as e:                              # noqa: BLE001
            logger.warning("Pieper tagged IK failed: %s", e)
            return []

    @staticmethod
    def _config_flag_text(cfg: dict) -> tuple[str, str, str]:
        """(F/R, U/D, F/N) text cho 1 config."""
        return (
            "Front" if cfg["front"] else "Rear",
            "Up" if cfg["elbow_up"] else "Down",
            "NoFlip" if cfg["no_flip"] else "Flip",
        )

    def _on_find_alternates(self) -> None:
        """List ALL IK solutions for the current TCP pose (Pieper analytical), like
        RoboDK: up to 8 postures (Front/Rear · Up/Down · Flip/Non-Flip), each with
        its ±360° joint-turn variants. The current solution is included.
        """
        cfgs = self._compute_configurations()
        self._alt_solutions = [c["joints_deg"] for c in cfgs]
        self._alt_combo.blockSignals(True)
        self._alt_combo.clear()
        if not cfgs:
            self._alt_combo.addItem("(unreachable — no IK configs)")
            self._set_status("No robot configurations (pose unreachable)",
                             level="warn")
        else:
            for c in cfgs:
                fr, ud, fn = self._config_flag_text(c)
                self._alt_combo.addItem(
                    f"id {c['id']} · {fr}/{ud}/{fn} · {c['turn_label']}")
            n_post = len({c["id"] for c in cfgs})
            self._set_status(
                f"Found {len(cfgs)} IK solution(s) in {n_post} posture(s) "
                f"— switching posture crosses a singularity", "ok")
        self._alt_combo.blockSignals(False)

    def _on_alternate_selected(self, idx: int) -> None:
        i = int(idx)
        if 0 <= i < len(self._alt_solutions):
            self._apply_joints_main(self._alt_solutions[i])
            self._set_status(
                f"Switched → {self._alt_combo.itemText(i)}", level="ok")

    def _show_configurations_dlg(self) -> None:
        """'Robot Configurations' dialog like RoboDK — config table + filters
        Front/Rear · Elbow Up/Down · Flip/Non-Flip + Show all / Recommended /
        Config.id. Flags shown as green(primary)/grey(other) dots like RoboDK.
        Double-click / Ok → apply config.
        """
        cfgs = self._compute_configurations()
        if not cfgs:
            self._set_status("No robot configurations (pose unreachable)",
                             level="warn")
            QMessageBox.information(
                self, "Robot Configurations",
                "Current TCP pose has no IK solution (out of reach / "
                "joint limits exceeded).")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Yaskawa Motoman GP7: Robot Configurations")
        dlg.setMinimumWidth(620)
        v = QVBoxLayout(dlg)

        # ── Top: 3 filter groups (left) + Show all/Recommended/Config.id (right) ──
        top = QHBoxLayout()
        groups = {
            "fr": ("Front/Rear",
                   [("Front", True), ("Rear", False), ("Both", None)]),
            "ud": ("Elbow up/down",
                   [("Elbow Up", True), ("Elbow Down", False), ("Both", None)]),
            "fn": ("Flip/non-Flip",
                   [("Non-Flip", True), ("Flip", False), ("Both", None)]),
        }
        self._cfg_filter = {"fr": None, "ud": None, "fn": None, "id": None}
        self._cfg_radio_groups: dict[str, QButtonGroup] = {}

        def _on_posture_change(key: str, val):
            self._cfg_filter[key] = val
            # Radio changed → reset id filter to joined (representing "all ids
            # matching posture"). Keeping the old id would make clicking "Both" seem to
            # broaden the filter but the table still pins the old id → confusing.
            self._cfg_filter["id"] = None
            _rebuild_id_combo()
            _refill()

        for key, (title, opts) in groups.items():
            gb = QGroupBox(title); gv = QVBoxLayout(gb); gv.setSpacing(1)
            bgrp = QButtonGroup(dlg)
            for i, (lab, val) in enumerate(opts):
                rb = QRadioButton(lab); rb.setStyleSheet("font-size: 9pt;")
                if val is None: rb.setChecked(True)
                bgrp.addButton(rb, i)
                gv.addWidget(rb)
            bgrp.idClicked.connect(
                lambda i, k=key, o=opts: _on_posture_change(k, o[i][1]))
            self._cfg_radio_groups[key] = bgrp
            top.addWidget(gb)

        # Right control column
        rcol = QVBoxLayout(); rcol.setSpacing(4)
        btn_all = QPushButton("Show all")
        btn_rec = QPushButton("Recommended")
        rcol.addWidget(btn_all); rcol.addWidget(btn_rec)
        idrow = QHBoxLayout()
        idrow.addWidget(QLabel("Config. id"))
        # Read-only combo (derived from posture like RoboDK). userData is
        # frozenset[int] of ids satisfying posture; user can only pick 1 id from
        # that set to narrow further — NOT editable. Content is updated by
        # `_rebuild_id_combo()` each time F/R · U/D · F/N changes.
        id_combo = QComboBox(); id_combo.setEditable(False)
        idrow.addWidget(id_combo, 1)
        rcol.addLayout(idrow)
        rcol.addStretch()
        top.addLayout(rcol)
        v.addLayout(top)

        info = QLabel(""); info.setStyleSheet("color: #cccccc; font-weight: 600;")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(info)

        note = QLabel(
            "Each row is an IK solution reaching the same TCP. Up to 8 postures "
            "(Front/Rear · Elbow Up/Down · Flip/Non-Flip) are bounded by "
            "singularities — switching posture crosses a singularity. Rows sharing "
            "an id are ±360° joint-turn variants of the same posture (GP7 has wide "
            "axes), so a pose may list many solutions, as in RoboDK.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #9a9a9a; font-style: italic;")
        v.addWidget(note)

        # ── Table ──
        cols = ["id", "F/R", "U/D", "F/N", "Turn",
                "J1", "J2", "J3", "J4", "J5", "J6"]
        table = QTableWidget(0, len(cols))
        table.setHorizontalHeaderLabels(cols)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        hh = table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for c in (0, 1, 2, 3, 4):                            # id + dots + turn: narrow
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        v.addWidget(table)

        # Large font for status dots — easier to distinguish green/grey than default ●
        # character (closer to RoboDK config table UX).
        _dot_font = QFont(); _dot_font.setPointSize(18); _dot_font.setBold(True)

        def _dot(on: bool) -> QTableWidgetItem:
            # Green = primary state (Front/Up/NoFlip), grey = opposite.
            it = QTableWidgetItem("●")
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            it.setForeground(QColor("#2da44e") if on else QColor("#5a5a5a"))
            it.setFont(_dot_font)
            return it

        def _refill():
            f = self._cfg_filter
            shown = [c for c in cfgs
                     if (f["fr"] is None or c["front"] == f["fr"])
                     and (f["ud"] is None or c["elbow_up"] == f["ud"])
                     and (f["fn"] is None or c["no_flip"] == f["fn"])
                     and (f["id"] is None or c["id"] in f["id"])]
            table.setRowCount(len(shown))
            table._shown = shown
            for r, c in enumerate(shown):
                id_it = QTableWidgetItem(str(c["id"]))
                id_it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(r, 0, id_it)
                table.setItem(r, 1, _dot(c["front"]))
                table.setItem(r, 2, _dot(c["elbow_up"]))
                table.setItem(r, 3, _dot(c["no_flip"]))
                turn_it = QTableWidgetItem(
                    c.get("turn_label", self._turn_label(
                        self._solution_turns(c["joints_deg"]))))
                turn_it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(r, 4, turn_it)
                for jc, a in enumerate(c["joints_deg"]):
                    it = QTableWidgetItem(f"{a:.1f}")
                    it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    table.setItem(r, 5 + jc, it)
            n_post = len({c["id"] for c in shown})
            info.setText(
                f"Showing: {len(shown)} solution(s) in {n_post} posture(s) "
                f"/ {len(cfgs)} total")
            if shown:
                table.selectRow(0)
        _refill()

        # ── Show all: reset EVERYTHING (3 radios → Both + id → joined) ──
        def _show_all():
            for k in ("fr", "ud", "fn"):
                self._cfg_filter[k] = None
                self._cfg_radio_groups[k].button(2).setChecked(True)  # "Both"
            self._cfg_filter["id"] = None
            _rebuild_id_combo()
            _refill()
        btn_all.clicked.connect(_show_all)

        # ── Recommended: pick config with the shortest move time ──
        # (like RoboDK): axes run in parallel so T_move = max_i(|Δθᵢ|/v_maxᵢ).
        # v_max per Yaskawa GP7 datasheet (deg/s) — larger axes are slower so have
        # higher "weight" (1/v_max is larger).
        _v_max_dps = (220.0, 200.0, 220.0, 410.0, 410.0, 610.0)  # J1..J6

        def _recommend():
            # Pick from ALL cfgs (independent of current filter) — even
            # when radios filter to an empty table, Recommended still finds
            # the best config and auto-snaps the UI to it.
            if not cfgs:
                return
            cur = self._joints
            best_cfg = min(
                cfgs,
                key=lambda c: max(abs(a - b) / v for a, b, v in
                                  zip(c["joints_deg"], cur, _v_max_dps)))
            # Sync combo+radios to best id (equivalent to user picking the id):
            # setCurrentIndex fires _on_id_index ⇒ auto-link radios + refilter.
            id_combo.setCurrentIndex(1 + best_cfg["id"])
            # After refilter, find the correct row (J6 turn variant) and select.
            new_shown = getattr(table, "_shown", [])
            for r, c in enumerate(new_shown):
                if c["joints_deg"] == best_cfg["joints_deg"]:
                    table.selectRow(r)
                    break
        btn_rec.clicked.connect(_recommend)

        # Combo has 9 items:
        #   • item 0 = "joined" — list of ids implied by posture (text varies
        #     with radios, e.g. Rear+Both+Both ⇒ "4,5,6,7"). This is the Config.id
        #     "corresponding" to the current radio state.
        #   • item 1..8 = 0..7 fixed — user can always pick any id;
        #     auto-link will snap radios to the id's bits.
        # Bidirectional binding: radio changes → joined text + selection refresh;
        # combo pick → radios snap.
        def _rebuild_id_combo():
            f = self._cfg_filter
            fr_bits = (0, 1) if f["fr"] is None else (0 if f["fr"] else 1,)
            ud_bits = (0, 1) if f["ud"] is None else (0 if f["ud"] else 1,)
            fn_bits = (0, 1) if f["fn"] is None else (0 if f["fn"] else 1,)
            implied = sorted(b_fr * 4 + b_ud * 2 + b_fn
                             for b_fr in fr_bits
                             for b_ud in ud_bits
                             for b_fn in fn_bits)
            prev = self._cfg_filter["id"]
            keep_single = (prev is not None and len(prev) == 1
                           and 0 <= next(iter(prev)) <= 7)
            id_combo.blockSignals(True)
            id_combo.clear()
            id_combo.addItem(",".join(map(str, implied)), frozenset(implied))
            for cid in range(8):
                id_combo.addItem(str(cid), frozenset({cid}))
            if keep_single:
                id_combo.setCurrentIndex(1 + next(iter(prev)))
            else:
                id_combo.setCurrentIndex(0)
            id_combo.blockSignals(False)
            self._cfg_filter["id"] = id_combo.currentData()

        # User selects from dropdown only — userData is frozenset[int]. Picking a specific
        # id (frozenset with 1 element) **auto-links** posture radios
        # to the id's bits (like RoboDK): id=7 (bits 111) ⇒ Rear/Down/Flip.
        # Picking the "joined set" item (e.g. "0,1,2,3,4,5,6,7") leaves radios unchanged.
        def _set_radio(key: str, target_val):
            bg = self._cfg_radio_groups[key]
            btn_idx = 0 if target_val is True else (1 if target_val is False else 2)
            bg.blockSignals(True)
            bg.button(btn_idx).setChecked(True)
            bg.blockSignals(False)

        def _on_id_index(_i: int):
            data = id_combo.currentData()
            auto_linked = False
            if data is not None and len(data) == 1:
                cid = next(iter(data))
                new_fr = (cid & 4) == 0   # bit 2: 0=Front (True), 1=Rear (False)
                new_ud = (cid & 2) == 0   # bit 1: 0=Up,           1=Down
                new_fn = (cid & 1) == 0   # bit 0: 0=NoFlip,       1=Flip
                self._cfg_filter["fr"] = new_fr
                self._cfg_filter["ud"] = new_ud
                self._cfg_filter["fn"] = new_fn
                _set_radio("fr", new_fr)
                _set_radio("ud", new_ud)
                _set_radio("fn", new_fn)
                auto_linked = True
            self._cfg_filter["id"] = data
            # After auto-link, combo must rebuild for the new posture (e.g. Rear/Down/Flip
            # ⇒ only id=7 remains). _rebuild_id_combo blocks signals so no recursion.
            if auto_linked:
                _rebuild_id_combo()
            _refill()
        id_combo.currentIndexChanged.connect(_on_id_index)

        # First open: build combo from full set (no posture filter yet).
        _rebuild_id_combo()

        def _apply_selected():
            shown = getattr(table, "_shown", [])
            r = table.currentRow()
            if 0 <= r < len(shown):
                self._apply_joints_main(shown[r]["joints_deg"])
                fr, ud, fn = self._config_flag_text(shown[r])
                self._set_status(
                    f"Config id {shown[r]['id']}: {fr}/{ud}/{fn}", level="ok")

        table.itemDoubleClicked.connect(
            lambda *_: (_apply_selected(), dlg.accept()))
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(lambda: (_apply_selected(), dlg.accept()))
        btns.rejected.connect(dlg.reject)
        v.addWidget(btns)
        dlg.exec()

    # ══════════════════════════════════════════════════════════════════
    # Parameters dialog
    # ══════════════════════════════════════════════════════════════════
    def _show_parameters_dlg(self) -> None:
        from PyQt6.QtWidgets import QDialog
        if self._model is None:
            self._set_status("Robot not loaded — File → Load Robot GP7", "warn")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Robot parameters (read-only)")
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(16, 16, 16, 12)
        lay.setSpacing(12)
        robot_name = (getattr(self._cell_config.robot, "name", "GP7")
                      if self._cell_config is not None else "GP7")
        lines = [
            f"<h3 style='margin:0 0 6px 0;'>{robot_name}</h3>",
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
        lines.append("<b>Verification:</b> FK match RoboDK SolveFK to 0.00 mm.")
        # QLabel sizes to its content (no scroll area) → dialog fits the table with
        # no wasted vertical space (the old QTextEdit was a fixed 700×560 box).
        body = QLabel("".join(lines))
        body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(body)
        ok = QPushButton("Close"); ok.setMinimumHeight(32)
        ok.clicked.connect(dlg.accept)
        lay.addWidget(ok)
        dlg.adjustSize()                        # shrink-wrap to content
        dlg.exec()

    # ══════════════════════════════════════════════════════════════════
    # WorkSpace sphere
    # ══════════════════════════════════════════════════════════════════
    def _compute_reach_envelope_mesh(self, mode: str):
        """Reach (work) envelope, RoboDK-style: BRUTE-FORCE forward kinematics.

        Instead of a hand-derived L1/L2/L3 formula, sample the position joints
        through their REAL limits, run FK on the URDF model for each sample, collect
        the reachable point of interest, then build the envelope as a surface of
        revolution about the J1 axis. This is exact to the URDF and — for 'tool'/
        'flange' — also sweeps the wrist (J4/J5) so the envelope grows correctly by
        the tool length in every orientation the wrist can reach (something a fixed
        offset along the forearm cannot capture).

          • mode 'wrist'  → wrist-centre point (depends on J2,J3 only).
          • mode 'flange' → flange point      (also sweeps J4,J5).
          • mode 'tool'   → current TCP point (also sweeps J4,J5).

        Built as a DENTED ball (no wedge cut): per world-azimuth bin, the reachable
        (r,z) cross-section is taken from the points reachable there for some
        q1∈[-170°,+170°]. The radius dents inward where max reach is reduced. There is
        NO binary hole — every azimuth is reachable (the arm folds to either side of
        its plane); only the radius shrinks.

          • OUTER = max reach over all configs (outer kidney shell).
          • INNER = the R233 concave VOID wall — the MIN reachable radius at each
            height. This is the central dent of the single envelope (matches the
            datasheet HW1483944 Fig 5-3(b) inner R233 arc): a spindle that pinches to
            the J1 axis at top & bottom (where the arm folds through the centre) and
            bulges out in the mid-band. NOT the old "J2<0" surface.

        Verified by FK against the datasheet (top 1217, bottom 476, R927, R233).
        FK output mm; scene metres (÷1000). Cached by (model, mode, tool, base).
        """
        if self._model is None:
            return None
        cache = getattr(self, "_reach_envelope_cache", {})
        # Include the tool offset MATRIX (not just its index) — the 'tool' envelope
        # depends on the TCP offset, so editing a gripper's offset (same index) must
        # invalidate the cached mesh.
        _tool_T = self._tool_frames[self._tool_idx][1]
        _tool_sig = tuple(np.round(np.asarray(_tool_T, float).ravel(), 3))
        ckey = (id(self._model), mode, self._tool_idx, tuple(self._base_xyz),
                tuple(getattr(self, "_base_rpy", (0.0, 0.0, 0.0))), _tool_sig)
        if ckey in cache:
            return cache[ckey]

        bx, by, bz = self._base_xyz
        jl = [(j.joint_min, j.joint_max) for j in self._model.joints]

        def _lin(i, n):
            return np.linspace(jl[i][0], jl[i][1], n)

        # ── Sample the q1=0 cross-section. For each reachable point record its
        # NATURAL azimuth (atan2 at q1=0), radius and height. J1 is then applied
        # ANALYTICALLY: a point at natural azimuth a is reachable at world azimuth
        # a+q1 for q1∈[-170°,+170°]. Split by shoulder: ALL configs → OUTER (max
        # reach overall), backward (J2<0) → INNER ("reach when joint 2 is negative",
        # per RoboDK). ──
        j1_half = math.degrees(min(-jl[0][0], jl[0][1]))     # 170° for GP7
        allp = []                                            # (natural_az, r, z) mm
        try:
            if mode == "wrist":
                grid = [(qL, qU, 0.0, 0.0)
                        for qL in _lin(1, 28) for qU in _lin(2, 32)]
            else:
                grid = [(qL, qU, qR, qB)
                        for qL in _lin(1, 16) for qU in _lin(2, 18)
                        for qR in _lin(3, 5) for qB in _lin(4, 9)]
            # Batched FK over the whole grid in one call. link_frames_batch_urdf is
            # BIT-IDENTICAL to per-sample link_frames_urdf (same _urdf_consts, matmul
            # order and Rodrigues) → the sampled points, and thus the envelope shape,
            # are UNCHANGED; just ~20× faster than a Python loop of thousands of FK
            # calls. (Verified: max abs frame diff = 0.0 vs per-sample.)
            q_arr = np.array([[0.0, qL, qU, qR, qB, 0.0]
                              for qL, qU, qR, qB in grid], dtype=float)
            fr = link_frames_batch_urdf(self._model, q_arr)   # {name: (N,4,4)}
            if mode == "wrist":
                P = None
                for k in ("link_B", "link_T", "link_tool0"):
                    if k in fr:
                        P = fr[k][:, :3, 3]; break
            elif mode == "flange":
                src = fr["link_flange"] if "link_flange" in fr else fr["link_tool0"]
                P = src[:, :3, 3]
            else:                                            # tool
                P = (fr["link_tool0"]
                     @ self._tool_frames[self._tool_idx][1])[:, :3, 3]
            X = P[:, 0] - bx; Y = P[:, 1] - by; Z = P[:, 2] - bz
            R = np.hypot(X, Y)
            AZ = np.degrees(np.arctan2(Y, X))
            for i in np.nonzero(R >= 1.0)[0]:                # skip r<1mm (on-axis)
                allp.append((float(AZ[i]), float(R[i]), float(Z[i])))
        except Exception as e:                              # noqa: BLE001
            logger.warning("Reach envelope FK failed: %s", e)
            return None
        if len(allp) < 8:
            return None

        def _kidney_mesh(recs, nb=72, nz=44):
            """Single CLOSED reach envelope WITH the inner concave void (the datasheet
            HW1483944 Fig 5-3(b) R233 dent), as a surface of revolution about the J1
            axis. For each world-azimuth bin the reachable (r,z) cross-section is the
            band between the OUTER boundary rmax(z) and the INNER boundary rmin(z)
            (the void near the central axis). The closed outline = outer boundary
            (ascending z) then inner boundary (descending z); revolving it sweeps the
            full kidney shell — outer wall, inner R233 dent, and the top/bottom rims
            where they meet. The inner boundary is the SMOOTHED concave cap of the FK
            min-reach (see _concave_cap) → one clean R233 bulge, not a stack of lenses.

            Phase is consistent across bins because point k is ALWAYS the same
            fractional height level → adjacent rings stitch without twisting (the
            reason the old convex-hull arc-length resample needed _resample_by_angle).
            Per-azimuth z-range is used (not a global one) so the rear dent — where the
            arm reaches neither as far NOR as high/low — shrinks correctly."""
            a = np.asarray(recs, dtype=float)
            if len(a) < 8:
                return None
            nsec = 2 * nz

            def _concave_cap(zc, r):
                """Upper convex hull of (zc, r) — the smallest CONCAVE curve ≥ r at
                every level. Turns the noisy FK min-reach (which spikes to the axis
                wherever the wrist folds through the centre) into ONE smooth inner
                bulge ≈ R233, pinching at top & bottom — the idealised datasheet dent
                instead of a stack of lenses. Monotone chain, no scipy."""
                hull = []
                for x, y in sorted(zip([float(t) for t in zc],
                                       [float(t) for t in r])):
                    while len(hull) >= 2:
                        (x1, y1), (x2, y2) = hull[-2], hull[-1]
                        if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= 0.0:
                            hull.pop()
                        else:
                            break
                    hull.append((x, y))
                cap = np.interp(zc, [h[0] for h in hull], [h[1] for h in hull])
                if len(cap) >= 5:                         # round the peak
                    k = np.array([1., 2., 3., 2., 1.]); k /= k.sum()
                    cap = np.convolve(cap, k, mode="same")
                return cap

            rings, off, pts = [None] * nb, [None] * nb, []
            for b in range(nb):
                beta = 360.0 * b / nb
                d = np.abs(((a[:, 0] - beta + 180.0) % 360.0) - 180.0)
                sel = a[d <= j1_half + 1e-6]            # reachable at this azimuth
                if len(sel) < 3:
                    continue
                zlo, zhi = float(sel[:, 2].min()), float(sel[:, 2].max())
                if zhi - zlo < 1.0:
                    continue
                edges = np.linspace(zlo, zhi, nz + 1)
                zc = 0.5 * (edges[:-1] + edges[1:])
                zc[0] = zlo; zc[-1] = zhi          # reach true extremes (no half-bin inset)
                idx = np.clip(np.digitize(sel[:, 2], edges) - 1, 0, nz - 1)
                rmin = np.full(nz, np.nan); rmax = np.full(nz, np.nan)
                for k in range(nz):
                    rs = sel[idx == k, 1]
                    if len(rs):
                        rmin[k] = float(rs.min()); rmax[k] = float(rs.max())
                v = ~np.isnan(rmax)
                if int(v.sum()) < 2:
                    continue
                rmin = np.interp(zc, zc[v], rmin[v])      # fill empty levels
                rmax = np.interp(zc, zc[v], rmax[v])
                r_inner = _concave_cap(zc, rmin)          # one smooth R233 bulge
                r_inner = np.clip(np.minimum(r_inner, rmax), 0.0, None)
                loop_r = list(rmax) + list(r_inner[::-1])  # outer up, inner down
                loop_z = list(zc) + list(zc[::-1])
                br = math.radians(beta); ca, sa = math.cos(br), math.sin(br)
                rings[b] = [(bx / 1000.0 + (rr / 1000.0) * ca,
                             by / 1000.0 + (rr / 1000.0) * sa,
                             bz / 1000.0 + zz / 1000.0)
                            for rr, zz in zip(loop_r, loop_z)]
            for b in range(nb):
                if rings[b] is not None:
                    off[b] = len(pts); pts.extend(rings[b])
            if len(pts) < nsec * 3:
                return None
            faces = []
            for b in range(nb):
                b2 = (b + 1) % nb
                if rings[b] is None or rings[b2] is None:
                    continue
                o0, o1 = off[b], off[b2]
                for j in range(nsec):
                    jn = (j + 1) % nsec
                    faces.extend([4, o0 + j, o1 + j, o1 + jn, o0 + jn])
            return pv.PolyData(np.array(pts), faces)

        # ONE closed envelope with the R233 inner dent (datasheet HW1483944 Fig
        # 5-3(b)). No separate inner surface — the void is carved into the single
        # kidney shell. Verified by FK (top 1217, bottom 476, R927, inner R233).
        outer_mesh = _kidney_mesh(allp)
        if outer_mesh is None:
            return None
        inner_mesh = None

        result = (outer_mesh, inner_mesh)
        if not hasattr(self, "_reach_envelope_cache"):
            self._reach_envelope_cache = {}
        self._reach_envelope_cache[ckey] = result
        return result

    def _on_workspace_changed(self, mode: str) -> None:
        # Clear existing
        for name in ("__workspace_outer", "__workspace_inner", "__workspace"):
            try:
                self._plotter.remove_actor(name)
            except Exception:                              # noqa: BLE001
                pass
        self._workspace_actor = None
        if mode == "none":
            self._plotter.render()
            return
        if self._model is None:
            return
        try:
            meshes = self._compute_reach_envelope_mesh(mode)
            if meshes is None:
                # None = FK/build failed or too few points; surface it instead of a
                # silent no-op (the prior actors were already removed above).
                self._set_status(
                    f"Workspace ({mode}): could not build reach envelope "
                    f"(see log)", level="warn")
                self._plotter.render()
                return
            outer_mesh, inner_mesh = meshes
            self._plotter.add_mesh(
                outer_mesh,
                color=[0.70, 0.85, 1.0],
                style="wireframe", line_width=2, opacity=0.75,
                lighting=False, name="__workspace_outer")
            if inner_mesh is not None:
                self._workspace_actor = self._plotter.add_mesh(
                    inner_mesh,
                    color=[1.0, 0.70, 0.10],
                    style="wireframe", line_width=2, opacity=0.9,
                    lighting=False, name="__workspace_inner")
            else:
                self._workspace_actor = None
            self._plotter.render()
            self._set_status(
                f"Workspace ({mode}): brute-force FK reach envelope",
                level="ok")
        except Exception as e:                             # noqa: BLE001
            self._set_status(f"Workspace error: {e}", level="err")

    # ══════════════════════════════════════════════════════════════════
    # Show Frames (triad axes per frame)
    # ══════════════════════════════════════════════════════════════════
    # Default 0.12m; resized via View > Visibility > Reference frames ± .
    # Instance attr (not class const) for dynamic resize.

    def _on_show_frames_all(self, state: int) -> None:
        """All/None toggle — set all frame checkboxes to state."""
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
        """Compute world transform (meters) for frame key. Returns None if N/A.

        Perf: reuse the JOINT-KEYED FK cache (_cached_fk) iff it matches the current
        joints — recompute on mismatch so a caller that changed joints without a
        render never draws a stale triad.
        """
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
        if self._model is None:
            return None
        # Try the joint-keyed FK cache; recompute (and refresh it) on mismatch.
        jk = (id(self._model), tuple(round(q, 4) for q in self._joints))
        cached = getattr(self, "_cached_fk", None)
        if cached is not None and cached[0] == jk:
            frames = cached[1]
        else:
            try:
                q_rad = [math.radians(q) for q in self._joints]
                frames = dict(link_frames_urdf(self._model, q_rad))
                self._cached_fk = (jk, frames)
            except Exception:                              # noqa: BLE001
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
        # Same proportions as world axes — base + tool consistent, thick RGB.
        actor.SetCylinderRadius(0.035)
        actor.SetConeRadius(0.55)
        actor.SetConeResolution(24)
        # Force OFF labels — SetVisibility(False) on caption has no effect
        # in VTK 9.x. Clearing text is the reliable approach.
        actor.SetXAxisLabelText("")
        actor.SetYAxisLabelText("")
        actor.SetZAxisLabelText("")
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
        """When robot moves → update transform for all currently visible frame triads."""
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
        """Press-hold continuous jog — auto-fires every 120ms until released."""
        self._jog_active = (mode, axis, sign)
        # Delay 250ms before auto-fire (avoids triggering on a simple click)
        QTimer.singleShot(250, self._jog_timer.start)

    def _stop_continuous_jog(self) -> None:
        self._jog_active = None
        self._jog_timer.stop()

    # ── Circular jog dial (QDial) — rotary encoder semantics ──────────
    # Every DIAL_DEG_PER_STEP degrees of rotation = 1 jog step. Persistent position
    # (does not snap to 0 on mouse release). Lower = more sensitive (less dial
    # rotation per step); 15° ⇒ 24 steps per full turn (was 30° = 12 steps/turn).
    DIAL_DEG_PER_STEP = 15                                  # degrees/step

    def _on_dial_value_changed(self, v: int) -> None:
        """Compute delta angle (handles wrap 0↔359) → accumulate → fire jog each
        time DIAL_DEG_PER_STEP is exceeded. Axis + sign from selected radio.
        """
        delta = v - self._last_dial_value
        # Wrap-around handling: 358° → 1° is +3°, not −357°.
        if delta > 180:
            delta -= 360
        elif delta < -180:
            delta += 360
        self._last_dial_value = v
        self._dial_accumulator += delta

        # Fire jog for each STEP_THRESHOLD degrees accumulated. Suspend per-step
        # rendering and repaint once at the end — a fast dial spin can queue
        # several steps in one event; rendering each separately is the main jog
        # lag. Actor transforms still update per step; only the VTK render coalesces.
        fired = False
        self._suspend_render = True
        try:
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
                fired = True
        finally:
            self._suspend_render = False
        if fired:
            self._plotter.render()

    def _set_status(self, msg: str, level: str = "info") -> None:
        # Dot color theo level (Fluent-inspired status indicator).
        dot_color = {
            "info": "#38b6e7",  # cyan
            "ok":   "#2da44e",  # green
            "warn": "#fb8500",  # orange
            "err":  "#cf222e",  # red
        }.get(level, "#969696")
        text_color = {
            "info": "#cccccc",
            "ok":   "#cccccc",
            "warn": "#ffc870",
            "err":  "#ff7373",
        }.get(level, "#cccccc")
        self._status_dot.setStyleSheet(
            f"color: {dot_color}; font-size: 16px;")
        self._status_lbl.setText(msg)
        self._status_lbl.setStyleSheet(f"color: {text_color};")

    def _current_tool_world(self) -> np.ndarray | None:
        if self._model is None:
            return None
        try:
            # Reuse the FK that _render_scene_frame just computed for these exact
            # joints (same cache key scheme) instead of recomputing link_frames_urdf.
            # During jog/animation this is called right after a render, so the
            # cache hits and we avoid a second full FK per frame.
            jk = (id(self._model), tuple(round(q, 4) for q in self._joints))
            cached = getattr(self, "_cached_fk", None)
            if cached is not None and cached[0] == jk:
                frames = cached[1]
            else:
                frames = dict(link_frames_urdf(
                    self._model, [math.radians(q) for q in self._joints]))
                self._cached_fk = (jk, frames)
            T_world_tool0 = frames.get("link_tool0")
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

    def _solve_cartesian(self, T_world_tool_target,
                          seeded: bool = True) -> list[float] | None:
        """IK for one TCP target (Tool frame, world) → joints in rad or None.

        seeded=True: tries multiple seeds (robust, used for new targets). seeded=False:
        single DLS from current joints (fast, used in continuous bisection jog).
        """
        T_flange_tool = self._tool_frames[self._tool_idx][1]
        T_world_tool0 = T_world_tool_target @ np.linalg.inv(T_flange_tool)
        q_init = [math.radians(q) for q in self._joints]
        if seeded:
            return inverse_kinematics_seeded(
                self._model, T_world_tool0, q_init,
                tol_mm=0.5, tol_rad=1e-3, max_iter=100)
        return inverse_kinematics(
            self._model, T_world_tool0, q_init,
            tol_mm=0.5, tol_rad=1e-3, max_iter=100)

    def _limit_blocker(self, sol_rad, tol_deg: float = 1.5) -> str | None:
        """Describe the joint at limit in the solution (None if none at limit)."""
        for i, qr in enumerate(sol_rad):
            d = math.degrees(qr)
            lo = math.degrees(self._model.joints[i].joint_min)
            hi = math.degrees(self._model.joints[i].joint_max)
            if d >= hi - tol_deg:
                return f"θ{i+1} at limit +{hi:.0f}°"
            if d <= lo + tol_deg:
                return f"θ{i+1} at limit {lo:.0f}°"
        return None

    # Manipulability threshold for singularity warning (tuned for GP7:
    # healthy ~0.07-0.08, near singularity < 0.01). See kinematics.manipulability().
    _W_SINGULAR = 0.01

    def _jog_cartesian(self, target_at, full_amount: float, unit: str,
                        label: str) -> None:
        """Jog Cartesian per industrial convention:

          1. **Hold configuration** — use local IK (seed = current joints), NO
             random reseeding → no branch jumping / sudden wrist flip.
          2. **Stop at joint limit** — if full step fails, bisection finds the
             largest achievable fraction (jog up to the limit like a real teach pendant) +
             reports which joint blocked.
          3. **Singularity warning** — manipulability w < threshold (wrist θ5≈0,
             arm fully extended…) → warn even if jog succeeds (real controller
             decelerates / reports near singularity).
        """
        q_now = [math.radians(q) for q in self._joints]
        w_now = manipulability(self._model, q_now)

        # 1+3. Full step via local IK (config-continuous).
        sol = self._solve_cartesian(target_at(1.0), seeded=False)
        if sol is not None:
            self._apply_joints_main([math.degrees(q) for q in sol])
            self._stream_live_jog()                     # Phase-2: → REAL robot if ON
            w_new = manipulability(self._model, sol)
            if min(w_now, w_new) < self._W_SINGULAR:
                self._set_status(
                    f"{label} {full_amount:.1f}{unit} — ⚠ near singularity "
                    f"(w={w_new:.3f}); Cartesian motion less stable",
                    level="warn")
            else:
                self._set_status(f"{label} {full_amount:.1f}{unit}", level="ok")
            return

        # 2. Bisection: max reachable fraction, still local IK (hold branch).
        lo_f, hi_f, best = 0.0, 1.0, None
        for _ in range(14):
            mid = 0.5 * (lo_f + hi_f)
            s = self._solve_cartesian(target_at(mid), seeded=False)
            if s is not None:
                best, lo_f = s, mid
            else:
                hi_f = mid
        if best is None or lo_f < 0.02:
            # Cannot jog even a small step → classify the cause.
            cause = ("near singularity — change orientation or jog joints out of "
                     "the singularity first" if w_now < self._W_SINGULAR
                     else "out of reach / joint limits exceeded")
            self._set_status(
                f"IK fail: {label} {full_amount:.1f}{unit} — {cause}",
                level="err")
            return
        self._apply_joints_main([math.degrees(q) for q in best])
        self._stream_live_jog()                         # Phase-2: → REAL robot if ON
        blk = self._limit_blocker(best) or "workspace boundary / singularity"
        self._set_status(
            f"{label} {lo_f*full_amount:.1f}/{full_amount:.1f}{unit} — "
            f"stopped by {blk}", level="warn")

    def _apply_cartesian_target(self, T_world_tool_target, source: str) -> None:
        """1-shot Cartesian target (paste / teach) — no interpolation."""
        sol = self._solve_cartesian(T_world_tool_target)
        if sol is None:
            self._set_status(
                f"IK fail: {source} — out of reach / joint limits exceeded",
                level="err")
            return
        self._apply_joints_main([math.degrees(q) for q in sol])
        self._set_status(source, level="ok")

    # ══════════════════════════════════════════════════════════════════
    # Callbacks (run on main thread)
    # ══════════════════════════════════════════════════════════════════

    def _on_joint_slider(self, idx: int, value_deg: float) -> None:
        j = list(self._joints); j[idx] = value_deg
        self._apply_joints_main(j)
        self._stream_live_jog()                         # Phase-2: → REAL robot if ON
        self._set_status(f"J{idx+1} = {value_deg:+.2f} deg")

    def _start_animation(self, joints_deg: list[float], label: str) -> bool:
        """Start a single TRACKED sim animation. Cancels any in-flight animation
        first and refuses while Live jog is ON (an untracked sim teleport would
        desync from the real robot). Returns True if started."""
        if (getattr(self, "_act_live_jog", None) is not None
                and self._act_live_jog.isChecked()):
            self._set_status(f"{label}: turn Live jog OFF first", level="warn")
            return False
        prev = self._anim_thread
        if prev is not None and prev.is_alive():
            self._anim_stop.set()
            prev.join(timeout=1.0)
        self._anim_stop = threading.Event()
        self._anim_thread = threading.Thread(
            target=self._animate_to, args=(list(joints_deg),),
            kwargs={"stop_event": self._anim_stop}, daemon=True)
        self._anim_thread.start()
        return True

    def _on_home(self) -> None:
        if self._model is None:
            self._set_status("Robot not loaded", level="warn"); return
        if self._start_animation(list(self._home_joints), "Home"):
            self._set_status("Move: Home", level="ok")

    # Below this orientation change (deg) the tool is already aligned → no move.
    _ALIGN_MIN_ANGLE_DEG = 0.5

    @staticmethod
    def _signed_perm_rotations() -> list[np.ndarray]:
        """The 24 proper rotation matrices whose columns are signed unit axes
        (the octahedral group). Cached on the class after first build."""
        cache = getattr(GP7AppQt, "_SIGNED_PERMS_CACHE", None)
        if cache is not None:
            return cache
        perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2),
                 (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        mats: list[np.ndarray] = []
        for p in perms:
            for bits in range(8):
                s = [1 if (bits >> k) & 1 == 0 else -1 for k in range(3)]
                M = np.zeros((3, 3))
                for col, (row, sg) in enumerate(zip(p, s)):
                    M[row, col] = sg
                if round(float(np.linalg.det(M))) == 1:   # proper rotation only
                    mats.append(M)
        GP7AppQt._SIGNED_PERMS_CACHE = mats               # 24 matrices
        return mats

    def _align_target_rotation(self, R_cur: np.ndarray,
                               R_ref: np.ndarray) -> np.ndarray:
        """Snap R_cur so the tool frame is axis-aligned with the reference frame:
        nearest rotation R_ref @ P where P is a signed-permutation rotation. Picks
        the P maximising the Frobenius inner product with R_ref.T @ R_cur."""
        R_local = R_ref.T @ R_cur
        best_P, best_dot = None, -1e9
        for P in self._signed_perm_rotations():
            dot = float(np.sum(P * R_local))
            if dot > best_dot:
                best_dot, best_P = dot, P
        return R_ref @ best_P

    def _on_align(self) -> None:
        """Align the tool ORIENTATION to the active reference frame (RoboDK 'Align').

        Keeps the TCP point fixed and rotates the wrist so the tool frame becomes
        axis-aligned with the reference frame — snapping the orientation to the nearest
        signed-axis match (e.g. tool Z pointing straight down the frame's -Z). Solves
        IK holding the current configuration, then animates. Visible wrist re-orient."""
        if self._model is None:
            self._set_status("Robot not loaded", level="warn"); return
        T_cur = self._current_tool_world()
        if T_cur is None:
            self._set_status("Align: cannot read current TCP pose", level="warn")
            return
        T_world_base = np.eye(4); T_world_base[:3, 3] = self._base_xyz
        R_ref = (T_world_base @ self._ref_frames[self._ref_idx][1])[:3, :3]
        R_cur = T_cur[:3, :3]
        R_aligned = self._align_target_rotation(R_cur, R_ref)
        # Orientation change magnitude (rotation angle of R_cur^T R_aligned).
        dR = R_cur.T @ R_aligned
        ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1.0) / 2.0))))
        if ang < self._ALIGN_MIN_ANGLE_DEG:
            self._set_status("Align: tool already aligned to reference frame",
                             level="ok")
            return
        T_target = np.eye(4)
        T_target[:3, :3] = R_aligned
        T_target[:3, 3] = T_cur[:3, 3]                    # keep TCP position
        sol = self._solve_cartesian(T_target, seeded=False)
        if sol is None:
            sol = self._solve_cartesian(T_target, seeded=True)
        if sol is None:
            self._set_status(
                "Align: no IK solution for the aligned orientation", level="warn")
            return
        joints_deg = [math.degrees(q) for q in sol]
        if self._start_animation(joints_deg, "Align"):
            self._set_status(
                f"Align → tool ⟂ '{self._ref_frames[self._ref_idx][0]}' "
                f"(Δ={ang:.0f}°)", level="ok")

    def _on_zero(self) -> None:
        if self._model is None:
            self._set_status("Robot not loaded", level="warn"); return
        if self._start_animation([0.0] * 6, "Zero"):
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
        full = sign * self._jog_step_mm
        p0 = T_tool[:3, 3].copy()

        def target_at(frac: float):
            T = T_tool.copy()
            T[:3, 3] = p0 + axis_world * (full * frac)
            return T

        self._jog_cartesian(
            target_at, self._jog_step_mm, "mm",
            f"T{'+' if sign>0 else '-'}{self.AXIS_NAMES[axis_idx]}")

    def _on_rotate(self, axis_idx: int, sign: int) -> None:
        T_tool = self._current_tool_world()
        if T_tool is None: return
        axis_world = self._jog_axis_world(axis_idx, T_tool)
        R0 = T_tool[:3, :3].copy()

        def target_at(frac: float):
            R_step = _rotation_about_axis_3x3(
                axis_world, sign * math.radians(self._jog_step_deg * frac))
            T = T_tool.copy()
            T[:3, :3] = R_step @ R0
            return T

        self._jog_cartesian(
            target_at, self._jog_step_deg, "°",
            f"R{'+' if sign>0 else '-'}{self.AXIS_NAMES[axis_idx]}")

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
        if self._model is None:
            return None
        try:
            q_rad = [math.radians(q) for q in self._joints]
            return dict(link_frames_urdf(self._model, q_rad)).get("link_tool0")
        except Exception:                                   # noqa: BLE001
            return None

    # (Removed: Demo motion methods — replaced by Program → Play with
    # user-defined instruction sequence. Hard-coded 4-pose loop was dev-only.)

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
        self._prog_running_row = None       # list rebuilt → drop stale run marker
        if not self._program:
            self._prog_list.addItem("(empty)"); return
        # Track modal state (speed/PL/tool/uframe) in command order → move
        # displays inline INFORM tag with the correct value applied at that point (like
        # how export folds modal state into each MOV line).
        modal = {"vj": self._pp_default_vj, "v": self._pp_default_v_mms,
                 "pl": None, "tl": None, "uf": None}
        depth = 0       # nesting depth of IF/WHILE blocks → indent for readability
        for i, ins in enumerate(self._program):
            t = ins.type
            if t == "SetSpeed":
                modal["vj"] = ins.speed_joint_pct
                modal["v"] = ins.speed_linear_mm_s
            elif t == "SetRounding":
                modal["pl"] = ins.rounding_pl
            elif t == "SetTool":
                modal["tl"] = ins.tool_no
            elif t == "SetRefFrame":
                modal["uf"] = ins.ref_frame_no
            # Closing block → decrease depth BEFORE rendering (keyword aligns with opening block).
            if t in ("EndIf", "EndWhile"):
                depth = max(0, depth - 1)
            line_depth = depth
            if t in ("ElseIf", "Else"):        # mid-block keyword dedented 1 level
                line_depth = max(0, depth - 1)
            indent = "    " * line_depth
            text = ins.describe(modal)
            # V (linear mm/s) only affects MOVL/MOVC. When this SetSpeed governs only
            # joint moves, V is meaningless and is NOT written to .JBI (MOVJ carries
            # VJ= only) — so don't show a misleading V= value that would silently reset
            # to the default on re-import.
            if t == "SetSpeed" and not self._setspeed_v_effective(i):
                text = (f"SET SPEED  VJ={ins.speed_joint_pct:.2f}"
                        f"   (V n/a - applies to MOVL/MOVC only)")
            self._prog_list.addItem(f"{i+1:>2}. {indent}{text}")
            # Opening block → increase depth for subsequent lines.
            if t in ("IfThen", "While"):
                depth += 1

    # ── Playback live view: highlight running step + follow CALL JOB ──────
    def _on_prog_show_job(self, job_name: str) -> None:
        """Switch the program list to show `job_name` during playback (e.g. when
        a CALL JOB enters a sub-job). Remembers the user's pre-play selection so
        it can be restored when the program finishes."""
        if job_name not in self._jobs:
            return
        if not hasattr(self, "_playback_prev_job") or self._playback_prev_job is None:
            self._playback_prev_job = self._active_job   # remember to restore
        if self._active_job != job_name:
            self._active_job = job_name
            # Reflect in the combo without re-triggering refresh storms.
            self._job_combo.blockSignals(True)
            idx = self._job_combo.findText(job_name)
            if idx >= 0:
                self._job_combo.setCurrentIndex(idx)
            self._job_combo.blockSignals(False)
            self._refresh_program_list()

    def _on_prog_step_highlight(self, row: int) -> None:
        """Light up + scroll to the currently-executing step.

        NOTE: a QListWidget with a stylesheet ignores item.setBackground(), so we
        mark the running step with a BRIGHT amber foreground + bold font (both
        reliably honoured) plus a '▶ ' prefix — clearly visible on the dark theme.
        """
        from PyQt6.QtGui import QBrush, QColor

        def _restore(idx: int) -> None:
            if 0 <= idx < self._prog_list.count():
                it = self._prog_list.item(idx)
                if it is not None:
                    it.setForeground(QBrush())          # back to theme default
                    f = it.font(); f.setBold(False); it.setFont(f)
                    txt = it.text()
                    if txt.startswith("▶ "):
                        it.setText(txt[2:])

        prev = getattr(self, "_prog_running_row", None)
        if prev is not None:
            _restore(prev)
        if 0 <= row < self._prog_list.count():
            item = self._prog_list.item(row)
            if item is not None:
                item.setForeground(QColor("#ffb000"))   # bright amber
                f = item.font(); f.setBold(True); item.setFont(f)
                if not item.text().startswith("▶ "):
                    item.setText("▶ " + item.text())
                self._prog_list.scrollToItem(item)
            self._prog_running_row = row
        else:
            self._prog_running_row = None

    def _on_prog_show_job_restore(self, job_name: str) -> None:
        """Restore the program list to `job_name` after playback (no re-arming of
        the saved-selection state). Used by _on_program_done."""
        if job_name not in self._jobs:
            return
        self._active_job = job_name
        self._job_combo.blockSignals(True)
        idx = self._job_combo.findText(job_name)
        if idx >= 0:
            self._job_combo.setCurrentIndex(idx)
        self._job_combo.blockSignals(False)
        self._refresh_program_list()

    def _on_prog_add_movej(self) -> None:
        if self._guard_not_running("add MoveJ"): return
        self._program.append(Instruction(type="MoveJ", joints=list(self._joints)))
        self._refresh_program_list()
        self._set_status(f"Program +MoveJ (n={len(self._program)})", level="ok")

    def _on_prog_add_movel(self) -> None:
        if self._guard_not_running("add MoveL"): return
        T = self._current_tool_world()
        if T is None: return
        x, y, z, rx, ry, rz = _matrix_to_xyz_rpy_deg(T)
        self._program.append(Instruction(type="MoveL", tcp_pose=[x, y, z, rx, ry, rz]))
        self._refresh_program_list()
        self._set_status(f"Program +MoveL (n={len(self._program)})", level="ok")

    def _on_prog_add_setdo(self) -> None:
        """Generic digital output (DOUT) — replaces gripper-specific. General-purpose
        programming: control any output bit (gripper, valve, ...)."""
        if self._guard_not_running("add DOUT"): return
        ins = Instruction(
            type="SetDO",
            do_index=int(self._prog_do_idx.value()),
            do_state=(self._prog_do_state.currentText() == "ON"),
        )
        self._program.append(ins)
        self._refresh_program_list()
        self._set_status(f"Program +{ins.describe()}", level="ok")

    def _on_prog_add_wait(self) -> None:
        if self._guard_not_running("add Wait"): return
        secs = float(self._prog_wait_spin.value())
        self._program.append(Instruction(type="Wait", wait_seconds=secs))
        self._refresh_program_list()
        self._set_status(f"Program +Wait {secs:.2f}s", level="ok")

    # ── Tier-1 new instructions ───────────────────────────────────────
    def _on_prog_add_movec(self) -> None:
        """2-step: first click → store current pose as MID; second click → end +
        commit MoveC instruction."""
        if self._guard_not_running("add MoveC"): return
        T = self._current_tool_world()
        if T is None: return
        pose = list(_matrix_to_xyz_rpy_deg(T))
        if self._pending_movc_mid is None:
            self._pending_movc_mid = pose
            self._btn_movec.setText("+ MoveC (set END)")
            self._set_status(
                "MoveC: MID captured — move robot to the END pose then click again",
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
            self._set_status("MSG empty — enter text first", level="warn"); return
        self._program.append(Instruction(type="ShowMessage", message=text))
        self._refresh_program_list()
        self._prog_msg_edit.clear()
        self._set_status(f'Program +MSG "{text[:32]}"', level="ok")

    # ── Logic instructions (flow control + variables) ─────────────────
    def _on_prog_add_label(self) -> None:
        name = self._prog_lbl_edit.text().strip().upper()
        if (not name or not name[0].isalpha()
                or not name.replace("_", "").isalnum()):
            self._set_status(
                "Label: start with a letter; letters/digits/_ only", level="warn")
            return
        self._program.append(Instruction(type="Label", label_name=name[:32]))
        self._refresh_program_list(); self._prog_lbl_edit.clear()
        self._set_status(f"Program +*{name}", level="ok")

    def _read_jump_cond(self):
        """Read 3 condition widgets → (lhs,op,rhs) | None (uncond) | 'ERR'."""
        op = self._prog_jc_op.currentText()
        if op == "(uncond)":
            return None
        lhs = self._prog_jc_lhs.text().strip()
        rhs = self._prog_jc_rhs.text().strip()
        if not lhs or not rhs:
            return "ERR"
        return (lhs, op, rhs)

    def _on_prog_add_jump(self) -> None:
        label = self._prog_jmp_edit.text().strip().upper()
        if not label:
            self._set_status("Jump: enter a target label", level="warn"); return
        cond = self._read_jump_cond()
        if cond == "ERR":
            self._set_status(
                "Jump IF: fill both sides, or set op to (uncond)", level="warn")
            return
        ins = Instruction(type="Jump", label_name=label[:32])
        if cond:
            ins.cond_lhs, ins.cond_op, ins.cond_rhs = cond
        self._program.append(ins)
        self._refresh_program_list()
        self._set_status(f"Program +{ins.describe()}", level="ok")

    def _on_prog_add_setvar(self) -> None:
        from .program_logic import VarStore
        var = self._prog_var_name.text().strip().upper()
        op = self._prog_var_op.currentText()
        arg = self._prog_var_arg.text().strip()
        try:
            VarStore.validate(var)
        except ValueError as e:
            self._set_status(str(e), level="warn"); return
        if op not in ("INC", "DEC") and not arg:
            self._set_status(
                f"{op} needs an operand (value or variable)", level="warn")
            return
        ins = Instruction(
            type="SetVar", var_name=var, var_op=op,
            var_arg=("" if op in ("INC", "DEC") else arg))
        self._program.append(ins)
        self._refresh_program_list()
        self._set_status(f"Program +{ins.describe()}", level="ok")

    # ── Structured blocks (IF / WHILE) ────────────────────────────────
    def _read_struct_cond(self):
        """Read structured block condition builder → (lhs,op,rhs) | None."""
        lhs = self._prog_sc_lhs.text().strip()
        rhs = self._prog_sc_rhs.text().strip()
        if not lhs or not rhs:
            return None
        return (lhs, self._prog_sc_op.currentText(), rhs)

    def _append_block(self, ins_type: str, need_cond: bool) -> None:
        """Append one block instruction; read condition if need_cond."""
        if self._guard_not_running(f"add {ins_type}"): return
        kwargs = {}
        if need_cond:
            cond = self._read_struct_cond()
            if not cond:
                self._set_status(
                    f"{ins_type}: enter a condition (both sides)", level="warn")
                return
            kwargs = dict(cond_lhs=cond[0], cond_op=cond[1], cond_rhs=cond[2])
        ins = Instruction(type=ins_type, **kwargs)
        self._program.append(ins)
        self._refresh_program_list()
        self._set_status(f"Program +{ins.describe()}", level="ok")

    def _on_prog_add_ifthen(self) -> None:
        self._append_block("IfThen", need_cond=True)

    def _on_prog_add_elseif(self) -> None:
        self._append_block("ElseIf", need_cond=True)

    def _on_prog_add_while(self) -> None:
        self._append_block("While", need_cond=True)

    def _on_prog_add_else(self) -> None:
        self._append_block("Else", need_cond=False)

    def _on_prog_add_endif(self) -> None:
        self._append_block("EndIf", need_cond=False)

    def _on_prog_add_endwhile(self) -> None:
        self._append_block("EndWhile", need_cond=False)

    # ── I/O & registers (extended INFORM) ─────────────────────────────
    def _on_prog_add_pulse(self) -> None:
        ins = Instruction(type="PulseDO", do_index=int(self._prog_pulse_idx.value()))
        self._program.append(ins)
        self._refresh_program_list()
        self._set_status(f"Program +{ins.describe()}", level="ok")

    def _on_prog_add_clearstack(self) -> None:
        ins = Instruction(type="ClearStack")
        self._program.append(ins)
        self._refresh_program_list()
        self._set_status(f"Program +{ins.describe()}", level="ok")

    def _on_prog_add_clearvar(self) -> None:
        from .program_logic import VarStore
        var = self._prog_clear_var.text().strip().upper()
        cnt_raw = self._prog_clear_cnt.text().strip().upper()
        try:
            VarStore.validate(var)
        except ValueError as e:
            self._set_status(str(e), level="warn"); return
        if cnt_raw in ("ALL", ""):
            cnt = -1
        else:
            try:
                cnt = int(cnt_raw)
            except ValueError:
                self._set_status("Count must be a number or ALL", level="warn"); return
        ins = Instruction(type="ClearVar", var_name=var, clear_count=cnt)
        self._program.append(ins)
        self._refresh_program_list()
        self._set_status(f"Program +{ins.describe()}", level="ok")

    def _on_prog_add_din(self) -> None:
        from .program_logic import VarStore
        var = self._prog_din_var.text().strip().upper()
        try:
            VarStore.validate(var)
        except ValueError as e:
            self._set_status(str(e), level="warn"); return
        ins = Instruction(
            type="ReadGroupIn", var_name=var,
            io_group_kind=self._prog_din_kind.currentText(),
            io_group=int(self._prog_din_grp.value()))
        self._program.append(ins)
        self._refresh_program_list()
        self._set_status(f"Program +{ins.describe()}", level="ok")

    def _on_prog_add_doutgroup(self) -> None:
        from .program_logic import VarStore
        var = self._prog_dog_var.text().strip().upper()
        try:
            VarStore.validate(var)
        except ValueError as e:
            self._set_status(str(e), level="warn"); return
        ins = Instruction(
            type="WriteGroupOut", var_name=var, io_group_kind="OG",
            io_group=int(self._prog_dog_grp.value()))
        self._program.append(ins)
        self._refresh_program_list()
        self._set_status(f"Program +{ins.describe()}", level="ok")

    # Replace picker entries: (label, type, exp). `exp=True` builds the EXP
    # keyword variant (IFTHENEXP/ELSEIFEXP/WHILEEXP) of a flow-control type.
    _REPLACE_TYPES = [
        ("MoveJ — joint move", "MoveJ", False),
        ("MoveL — linear move", "MoveL", False),
        ("SetDO — DOUT OT#", "SetDO", False),
        ("PulseDO — PULSE OT#", "PulseDO", False),
        ("Wait — TIMER", "Wait", False),
        ("WaitIO — WAIT IN#", "WaitIO", False),
        ("SetSpeed — VJ/V", "SetSpeed", False),
        ("SetRounding — PL", "SetRounding", False),
        ("SetTool — TL#", "SetTool", False),
        ("SetRefFrame — UF#", "SetRefFrame", False),
        ("ShowMessage — MSG", "ShowMessage", False),
        ("CallJob — CALL JOB", "CallJob", False),
        ("Label — *LABEL", "Label", False),
        ("Jump — JUMP", "Jump", False),
        ("SetVar — SET/ADD/EXPRESS", "SetVar", False),
        ("IfThen — IFTHEN", "IfThen", False),
        ("IfThenExp — IFTHENEXP (I/O)", "IfThen", True),
        ("ElseIf — ELSEIF", "ElseIf", False),
        ("ElseIfExp — ELSEIFEXP (I/O)", "ElseIf", True),
        ("Else — ELSE", "Else", False),
        ("EndIf — ENDIF", "EndIf", False),
        ("While — WHILE", "While", False),
        ("WhileExp — WHILEEXP (I/O)", "While", True),
        ("EndWhile — ENDWHILE", "EndWhile", False),
        ("ClearStack — CLEAR STACK", "ClearStack", False),
        ("ClearVar — CLEAR", "ClearVar", False),
        ("ReadGroupIn — DIN", "ReadGroupIn", False),
        ("WriteGroupOut — DOUT OG#", "WriteGroupOut", False),
        ("SimEvent", "SimEvent", False),
    ]

    def _on_prog_replace(self) -> None:
        """Change the selected instruction to a DIFFERENT type, keeping its
        position in the list. Inserts a default instance of the new type, then
        routes to the parameter editor. Cancelling the editor keeps the default."""
        idx = self._prog_list.currentRow()
        if idx < 0 or idx >= len(self._program):
            self._set_status("Select an instruction to Replace", level="warn"); return
        cur = self._program[idx]
        labels = [e[0] for e in self._REPLACE_TYPES]
        # Pre-select the entry matching the current type + its EXP flavour.
        pre = 0
        for i, (_lbl, ty, exp) in enumerate(self._REPLACE_TYPES):
            if ty == cur.type and exp == bool(getattr(cur, "cond_exp", False)):
                pre = i; break
        lbl, ok = QInputDialog.getItem(
            self, "Replace instruction",
            f"Change step {idx+1} ({cur.describe()}) to:", labels, pre, False)
        if not ok: return
        _lbl, new_type, new_exp = self._REPLACE_TYPES[labels.index(lbl)]
        # Swap in a default instance of the new type at the SAME position, then
        # rebuild the list so the new line shows immediately. We do NOT auto-open
        # the parameter editor here (that second dialog confused users into
        # thinking nothing changed) — the user presses Edit/F2 to set params.
        self._program[idx] = self._default_instruction(new_type, new_exp)
        self._refresh_program_list()
        self._prog_list.setCurrentRow(idx)
        self._prog_list.repaint()                # force immediate repaint
        editable = new_type not in (
            "Else", "EndIf", "EndWhile", "ClearStack",
            "ReadGroupIn", "WriteGroupOut", "MoveJ", "MoveL")
        tail = " — press Edit (F2) to set its parameters" if editable else ""
        self._set_status(
            f"Replaced step {idx+1} → {self._program[idx].describe()}{tail}",
            level="ok")

    def _default_instruction(self, t: str, exp: bool = False) -> "Instruction":
        """Build a sensible default Instruction of type `t` for Replace.
        Motion types capture the current pose so the step is immediately valid;
        `exp` sets the EXP keyword flavour on flow-control conditions."""
        if t == "MoveJ":
            return Instruction(type="MoveJ", joints=list(self._joints))
        if t == "MoveL":
            T = self._current_tool_world()
            pose = list(_matrix_to_xyz_rpy_deg(T)) if T is not None else [0.0] * 6
            return Instruction(type="MoveL", tcp_pose=pose)
        if t == "SetVar":
            return Instruction(type="SetVar", var_name="B000", var_op="SET",
                               var_arg="0")
        if t in ("IfThen", "ElseIf", "While"):
            return Instruction(type=t, cond_lhs="B000", cond_op="=", cond_rhs="1",
                               cond_exp=exp)
        if t == "Jump":
            return Instruction(type="Jump", label_name="L1")
        if t == "Label":
            return Instruction(type="Label", label_name="L1")
        if t == "ClearVar":
            return Instruction(type="ClearVar", var_name="I000", clear_count=1)
        if t == "ReadGroupIn":
            return Instruction(type="ReadGroupIn", var_name="B000",
                               io_group=1, io_group_kind="IG")
        if t == "WriteGroupOut":
            return Instruction(type="WriteGroupOut", var_name="B000",
                               io_group=1, io_group_kind="OG")
        if t == "CallJob":
            return Instruction(type="CallJob", job_name="SUB")
        if t == "ShowMessage":
            return Instruction(type="ShowMessage", message="MSG")
        return Instruction(type=t)               # field defaults are fine

    def _on_prog_modify(self) -> None:
        """F2 / double-click / Edit button — edit selected instruction.

        Behaviour per type:
          • MoveJ/MoveL/MoveC (inline pose) → Replace with current pose (after
            confirm). Target-referencing → dialog to pick another target.
          • SetGripper → flip OPEN/CLOSE.
          • Wait / WaitIO / SetSpeed / SetRounding / SetTool / SetRefFrame /
            ShowMessage / CallJob / SimEvent → dialog with editable fields.
          • Label / Jump / IfThen / ElseIf / While / SetVar / PulseDO / ClearVar
            → dialog (logic instructions). The condition dialog edits compound
            (ANDEXP/OREXP) conditions as text and has an EXP-keyword toggle.
            ClearStack / group I/O have no inline editor (delete + re-add).
        Editing mutates the active job's list in place, which invalidates the
        verbatim re-export cache (signature mismatch) → export re-synthesises.
        """
        if self._guard_not_running("edit the program"): return
        idx = self._prog_list.currentRow()
        if idx < 0 or idx >= len(self._program):
            self._set_status("Select an instruction to Edit", level="warn"); return
        ins = self._program[idx]
        t = ins.type
        # ── Motion (inline vs target-ref) ─────────────────────────────
        if t in ("MoveJ", "MoveL"):
            if ins.target_name:
                # Target-ref → pick another target via combo
                new_name = self._dlg_pick_target(ins.target_name)
                if new_name is None: return
                ins.target_name = new_name
            else:
                r = QMessageBox.question(
                    self, "Modify", f"Replace step {idx+1} pose with current pose?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if r != QMessageBox.StandardButton.Yes: return
                # An indirect P[Bxxx] move ignores joints/tcp_pose (pos_index_var wins
                # at export/playback) — clear it so the captured pose actually applies.
                ins.pos_index_var = ""
                if t == "MoveJ":
                    ins.joints = list(self._joints)
                else:
                    T = self._current_tool_world()
                    if T is None: return
                    ins.tcp_pose = list(_matrix_to_xyz_rpy_deg(T))
        elif t == "MoveC":
            r = QMessageBox.question(
                self, "Modify MoveC",
                f"Replace step {idx+1} END with current pose? (MID unchanged)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes: return
            T = self._current_tool_world()
            if T is None: return
            ins.tcp_pose = list(_matrix_to_xyz_rpy_deg(T))
        # ── Logic / Modal ─────────────────────────────────────────────
        elif t == "SetGripper":
            ins.gripper_close = not ins.gripper_close
        elif t == "SetDO":
            idx_v, ok = QInputDialog.getInt(
                self, "Modify DOUT", "Output bit OT# (1..1024):",
                ins.do_index, 1, 1024)
            if not ok: return
            state_v, ok = QInputDialog.getItem(
                self, "Modify DOUT", "State:", ["ON", "OFF"],
                0 if ins.do_state else 1, False)
            if not ok: return
            ins.do_index = int(idx_v); ins.do_state = (state_v == "ON")
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
            new = self._dlg_edit_setspeed(ins, self._setspeed_v_effective(idx))
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
                self._set_status("Invalid job name", level="warn"); return
            ins.job_name = safe
        elif t == "SimEvent":
            new = self._dlg_edit_simevent(ins)
            if new is None: return
            ins.event_name, ins.event_payload = new
        # ── Flow control / variables (INFORM logic) ───────────────────
        elif t == "Label":
            v, ok = QInputDialog.getText(
                self, "Modify Label", "Label name (*LABEL):",
                QLineEdit.EchoMode.Normal, ins.label_name)
            if not ok: return
            name = "".join(c for c in v if c.isalnum() or c == "_")[:32]
            if not name:
                self._set_status("Invalid label name", level="warn"); return
            ins.label_name = name
        elif t == "Jump":
            v, ok = QInputDialog.getText(
                self, "Modify Jump", "Target label:",
                QLineEdit.EchoMode.Normal, ins.label_name)
            if not ok: return
            name = "".join(c for c in v if c.isalnum() or c == "_")[:32]
            if not name:
                self._set_status("Invalid label name", level="warn"); return
            ins.label_name = name
            cond = self._dlg_edit_condition(ins, allow_none=True,
                                            title="Modify Jump condition")
            if cond is False: return            # cancelled
            self._apply_cond(ins, cond)
        elif t in ("IfThen", "ElseIf", "While"):
            cond = self._dlg_edit_condition(ins, allow_none=False,
                                            title=f"Modify {t} condition")
            if cond is False: return
            self._apply_cond(ins, cond)
        elif t == "SetVar":
            new = self._dlg_edit_setvar(ins)
            if new is None: return
            ins.var_name, ins.var_op, ins.var_arg, ins.var_expr = new
        elif t == "PulseDO":
            v, ok = QInputDialog.getInt(
                self, "Modify Pulse", "Output bit OT# (1..1024):",
                ins.do_index, 1, 1024)
            if not ok: return
            ins.do_index = int(v)
        elif t == "ClearVar":
            v, ok = QInputDialog.getText(
                self, "Modify Clear", "Variable + count (e.g. 'I010 2' or 'I010 ALL'):",
                QLineEdit.EchoMode.Normal,
                f"{ins.var_name} {'ALL' if ins.clear_count < 0 else ins.clear_count}")
            if not ok: return
            parts = v.split()
            if len(parts) != 2:
                self._set_status("Format: <var> <count|ALL>", level="warn"); return
            ins.var_name = parts[0].upper()
            ins.clear_count = -1 if parts[1].upper() == "ALL" else int(parts[1])
        elif t in ("ReadGroupIn", "WriteGroupOut", "ClearStack"):
            self._set_status(
                f"{t}: no editable fields (delete + re-add to change)", level="info")
            return
        else:
            self._set_status(f"Edit not supported for type {t}", level="warn"); return
        self._refresh_program_list()
        self._prog_list.setCurrentRow(idx)
        self._set_status(f"Modified step {idx+1}: {ins.describe()}", level="ok")

    def _apply_cond(self, ins: Instruction, cond) -> None:
        """Apply a parsed condition to ins. `cond` is one of:
          • None                       → unconditional (clear everything)
          • {"terms": [...], "join": "AND"/"OR", "exp": bool}  → from the dialog.
        Sets single (cond_lhs/op/rhs) when one term, else compound (cond_terms);
        always sets cond_exp; mirrors the first term for back-compat."""
        if cond is None:
            ins.cond_terms = []
            ins.cond_lhs = ins.cond_op = ins.cond_rhs = ""
            ins.cond_join = ""
            ins.cond_exp = False
            return
        terms = cond["terms"]
        ins.cond_exp = bool(cond.get("exp", False))
        ins.cond_lhs, ins.cond_op, ins.cond_rhs = terms[0]
        if len(terms) > 1:
            ins.cond_terms = list(terms)
            ins.cond_join = cond.get("join", "AND")
        else:
            ins.cond_terms = []
            ins.cond_join = ""

    @staticmethod
    def _parse_cond_text(text: str):
        """Parse a condition string into (terms, join). Supports a single
        'lhs op rhs' or compound 'a op b ANDEXP c op d' / OREXP. Returns
        (list_of_(lhs,op,rhs), 'AND'/'OR'/'') or None if unparseable."""
        import re as _re
        parts = _re.split(r"\s+(ANDEXP|OREXP|AND|OR)\s+", text.strip(),
                          flags=_re.IGNORECASE)
        term_strs = parts[0::2]
        joiners = [j.upper().replace("EXP", "") for j in parts[1::2]]
        if joiners and len(set(joiners)) > 1:
            return None                         # mixed AND/OR not supported
        terms = []
        for ts in term_strs:
            m = _re.match(r"^\s*(\S+?)\s*(<>|>=|<=|=|>|<)\s*(\S+?)\s*$", ts)
            if not m:
                return None
            terms.append((m.group(1), m.group(2), m.group(3)))
        if not terms:
            return None
        return terms, (joiners[0] if joiners else "")

    def _dlg_edit_condition(self, ins: Instruction, allow_none: bool, title: str):
        """Dialog editing a condition (single OR compound) + an EXP toggle.

        Returns a dict {"terms":[...], "join":..., "exp":bool}, or None for
        unconditional (allow_none only), or False if cancelled. Supports
        compound 'a<>1 ANDEXP b<>2' / OREXP entered as text."""
        # Current text from the instruction.
        if ins.cond_terms:
            join = f" {ins.cond_join or 'AND'}EXP "
            cur = join.join(f"{l}{o}{r}" for l, o, r in ins.cond_terms)
        elif ins.cond_op:
            cur = f"{ins.cond_lhs}{ins.cond_op}{ins.cond_rhs}"
        else:
            cur = ""

        dlg = QDialog(self); dlg.setWindowTitle(title)
        form = QFormLayout(dlg)
        ed = QLineEdit(cur)
        ed.setPlaceholderText("B000>5  |  IN#(8)=ON  |  I010<>11 ANDEXP B010<>12")
        ed.setMinimumWidth(320)
        form.addRow("Condition", ed)
        cb_exp = QCheckBox("Use EXP keyword (IFTHENEXP/ELSEIFEXP/WHILEEXP)")
        cb_exp.setChecked(bool(ins.cond_exp))
        cb_exp.setToolTip(
            "Required for I/O conditions (IN#/OT#/ON/OFF). Auto-applied for "
            "those even if unchecked. Tick to force EXP on variable conditions.")
        form.addRow("", cb_exp)
        hint = QLabel("Operators: = <> > < >= <=. Combine terms with ANDEXP / "
                      "OREXP." + ("  Empty = unconditional." if allow_none else ""))
        hint.setWordWrap(True); hint.setStyleSheet("color:#8a8a8a; font-size:11px;")
        form.addRow("", hint)
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return False

        v = ed.text().strip()
        if not v:
            if allow_none:
                return None
            self._set_status("Condition required", level="warn"); return False
        parsed = self._parse_cond_text(v)
        if parsed is None:
            self._set_status(f"Cannot parse condition: {v!r}", level="warn")
            return False
        terms, join = parsed
        return {"terms": terms, "join": join or "AND", "exp": cb_exp.isChecked()}

    def _dlg_edit_setvar(self, ins: Instruction):
        """Dialog editing a SetVar. Returns (var_name, var_op, var_arg, var_expr)
        or None if cancelled."""
        ops = ["SET", "ADD", "SUB", "MUL", "DIV", "INC", "DEC", "EXPRESS"]
        cur_op = "EXPRESS" if ins.var_expr else (ins.var_op.upper() or "SET")
        op, ok = QInputDialog.getItem(
            self, "Modify SetVar", "Operation:", ops,
            ops.index(cur_op) if cur_op in ops else 0, False)
        if not ok: return None
        name, ok = QInputDialog.getText(
            self, "Modify SetVar", "Variable (B###/I###):",
            QLineEdit.EchoMode.Normal, ins.var_name)
        if not ok: return None
        name = name.strip().upper()
        if op in ("INC", "DEC"):
            return (name, op, "", "")
        if op == "EXPRESS":
            expr, ok = QInputDialog.getText(
                self, "Modify SetVar", "Expression (e.g. 5 * B005):",
                QLineEdit.EchoMode.Normal, ins.var_expr)
            if not ok: return None
            return (name, "SET", "", expr.strip())
        arg, ok = QInputDialog.getText(
            self, "Modify SetVar", "Operand (value or variable):",
            QLineEdit.EchoMode.Normal, ins.var_arg)
        if not ok: return None
        return (name, op, arg.strip(), "")

    def _dlg_pick_target(self, current: str) -> str | None:
        """QDialog to pick another target from the list. Returns new name or None if cancelled."""
        if not self._targets:
            self._set_status("No targets defined", level="warn"); return None
        names = list(self._targets.keys())
        idx = names.index(current) if current in names else 0
        v, ok = QInputDialog.getItem(
            self, "Pick target", "Target:", names, idx, False)
        return v if ok else None

    def _dlg_edit_waitio(self, ins: Instruction):
        """Returns (io_index, io_state_bool, timeout_s) or None."""
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
            "<small><i>Initial speeds apply to MOVJ/MOVL before the user "
            "sets SetSpeed. Max VJ is the safety limit for every move.</i></small>")
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
        # Default VJ/V feed the modal tail of moves before the first SetSpeed → re-render.
        self._refresh_program_list()
        self._set_status(
            f"Post-processor: max={self._pp_max_speed_pct:.0f}%, "
            f"VJ₀={self._pp_default_vj:.0f}%, V₀={self._pp_default_v_mms:.0f}mm/s",
            level="ok")

    # Robot connection (HSE) — Run on Robot pipeline → ConnectionMixin

    def _on_show_script_editor(self) -> None:
        """Open Python script editor — user enters code using `p.add_*()` API
        to generate instructions programmatically. Run → appends to current job.
        """
        dlg = QDialog(self); dlg.setWindowTitle("Generate from Python script")
        dlg.resize(720, 520)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(
            "<b>Python script</b> — use <code>p.add_*()</code> to add "
            "instructions to the current job. Helpers: <code>math</code>, "
            "<code>np</code>, target dict <code>p.targets</code>."))
        editor = QPlainTextEdit()
        editor.setPlaceholderText(
            "# Example: create 8 MoveJ points on a circle around Z=500\n"
            "import math\n"
            "for i in range(8):\n"
            "    angle = i * 2 * math.pi / 8\n"
            "    x = 500 + 200 * math.cos(angle)\n"
            "    y = 0   + 200 * math.sin(angle)\n"
            "    # Solve IK separately, or use an existing target:\n"
            "    p.add_movej_to('HOME')")
        from PyQt6.QtGui import QFont
        mono = QFont("Consolas"); mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(10)
        editor.setFont(mono)
        # Restore previous script if any
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
                    f"Script: +{added} instructions to '{self._active_job}'",
                    level="ok")
            except Exception as e:                          # noqa: BLE001
                result_lbl.setText(
                    f"<span style='color:#cf222e'><b>Error:</b> "
                    f"{type(e).__name__}: {e}</span>")
        b_run.clicked.connect(_run)
        dlg.exec()

    def _dlg_edit_simevent(self, ins: Instruction):
        """Returns (name, payload) or None."""
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

    def _setspeed_v_effective(self, idx: int) -> bool:
        """True if the SetSpeed at self._program[idx] actually governs a linear move.

        V (linear mm/s) only applies to MOVL/MOVC; MOVJ uses VJ only and carries no
        V= tag in INFORM. So V is meaningful — and survives a .JBI round-trip — only
        when a MOVL/MOVC follows this SetSpeed before the next SetSpeed overrides it."""
        if not (0 <= idx < len(self._program)):
            return True
        for k in range(idx + 1, len(self._program)):
            tk = self._program[k].type
            if tk == "SetSpeed":
                break
            if tk in ("MoveL", "MoveC"):
                return True
        return False

    def _dlg_edit_setspeed(self, ins: Instruction, v_effective: bool = True):
        """Returns (vj_pct, v_mm_s) or None.

        When v_effective is False (no MOVL/MOVC follows), V is disabled with a note:
        it has no effect and cannot be stored in .JBI (MOVJ carries VJ= only), so
        letting the user edit it would be misleading — it silently resets on reload."""
        dlg = QDialog(self); dlg.setWindowTitle("Modify SetSpeed")
        form = QFormLayout(dlg)
        sp_vj = QDoubleSpinBox(); sp_vj.setRange(1.0, 30.0); sp_vj.setSuffix(" %")
        sp_vj.setValue(ins.speed_joint_pct)
        sp_v = QDoubleSpinBox(); sp_v.setRange(1.0, 250.0); sp_v.setSuffix(" mm/s")
        sp_v.setValue(ins.speed_linear_mm_s)
        form.addRow("VJ (joint)", sp_vj); form.addRow("V (linear)", sp_v)
        if not v_effective:
            sp_v.setEnabled(False)
            sp_v.setToolTip("V applies to MOVL/MOVC only. The next move is MOVJ "
                            "(uses VJ only), so V has no effect and is not written "
                            "to .JBI.")
            note = QLabel("V (linear) applies to MOVL/MOVC only. No linear move "
                          "follows this SET SPEED, so V is ignored and not saved to "
                          ".JBI — only VJ (joint speed) is applied. Use a MOVL/MOVC "
                          "move, or save the project as .json to keep V.")
            note.setWordWrap(True)
            note.setStyleSheet("color:#E0A030; font-size:11px;")
            form.addRow(note)
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
            self._set_status("Invalid job name", level="warn"); return
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

    # Multi-job project + Target library → JobTargetMixin

    # ══════════════════════════════════════════════════════════════════
    # Teach on Surface (Ctrl+Shift+T) — click 3D scene → create target
    # ══════════════════════════════════════════════════════════════════
    def _teach_target_from_matrix(
        self, T, default_name: str, prompt_label: str | None = None,
        prompt: bool = True,
    ) -> str | None:
        """IK for TCP target T (4×4 world, mm) → (prompt name) → save to library.

        Shared by Teach-on-surface (mouse pick) and camera grasp-teach
        (pose from detection). Returns the saved target name, or None if IK fails /
        user cancels / name duplicate/invalid. Does NOT emit "taught" status (caller
        reports it, because context differs).

        prompt=False: skip name dialog, use `default_name` (auto-append suffix
        _2, _3… on collision) — used for auto-generated auxiliary targets (e.g. approach)."""
        if self._model is None:
            self._set_status(
                "Robot not loaded — IK needs a robot model", level="warn")
            return None
        sol = self._solve_cartesian(T, seeded=True)
        if sol is None:
            p = T[:3, 3]
            self._set_status(
                f"IK fail at ({p[0]:.0f},{p[1]:.0f},{p[2]:.0f})mm — "
                "out of reach", level="err")
            return None
        if prompt:
            if prompt_label is None:
                p = T[:3, 3]
                prompt_label = (f"Target name (XYZ = "
                                f"{p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f} mm):")
            v, ok = QInputDialog.getText(
                self, "Teach target", prompt_label,
                QLineEdit.EchoMode.Normal, default_name)
            if not ok or not v.strip():
                return None
            name = self._safe_target_name(v)
            if not name:
                self._set_status("Invalid name", level="warn"); return None
            if name in self._targets:
                self._set_status(
                    f"Target '{name}' already exists — Modify (F3) to update",
                    level="warn"); return None
        else:
            base = self._safe_target_name(default_name) or "TGT"
            name = base; i = 2
            while name in self._targets:
                if i > 999:                          # never spin (UI hang guard)
                    self._set_status(
                        "Cannot auto-name target (too many duplicates)", level="warn")
                    return None
                # Truncate the STEM (not the whole "base_i") so the _i suffix always
                # survives _safe_target_name's 24-char cap — otherwise a 24-char base
                # truncates back to itself and the loop never terminates.
                name = self._safe_target_name(f"{base[:20]}_{i}"); i += 1
        self._targets[name] = {
            "joints": [math.degrees(q) for q in sol],
            "tcp_pose": list(_matrix_to_xyz_rpy_deg(T)),
        }
        self._refresh_target_list()
        return name

    def _on_toggle_surface_pick(self, enabled: bool) -> None:
        if enabled and self._model is None:
            self._set_status(
                "Robot not loaded — load the robot before Teach on surface",
                level="warn")
            self._surface_pick_mode = False
            self._set_toggle(self._act_surface_pick, False)
            return
        self._surface_pick_mode = bool(enabled)
        if enabled:
            try:
                # Exclude robot mesh from pick → only teach on cell/floor/object,
                # avoiding accidental clicks on the robot body (target on itself = meaningless).
                self._set_robot_pickable(False)
                self._plotter.enable_surface_point_picking(
                    callback=self._on_surface_pick,
                    show_message=False,
                    show_point=True,
                    point_size=14,
                    color="magenta",
                    left_clicking=True,
                )
                self._set_status(
                    "Teach on surface ON — click cell/floor to create a target",
                    level="info")
            except Exception as e:                          # noqa: BLE001
                self._set_status(f"Surface pick fail: {e}", level="err")
                self._surface_pick_mode = False
                self._set_robot_pickable(True)
                self._set_toggle(self._act_surface_pick, False)
        else:
            try:
                self._plotter.disable_picking()
            except Exception:                               # noqa: BLE001
                pass
            self._set_robot_pickable(True)                  # restore robot picking
            self._set_status("Teach on surface OFF", level="ok")

    def _set_robot_pickable(self, flag: bool) -> None:
        """Enable/disable pickable for all robot actors (links + gripper). Used to exclude
        robot from surface-pick (teach only on cell/floor/object)."""
        for actor in self._link_actors.values():
            try:
                actor.SetPickable(bool(flag))
            except Exception:                               # noqa: BLE001
                pass

    def _on_surface_pick(self, picked_point) -> None:
        """Callback from pyvista picker. picked_point in METERS (pyvista internal)."""
        if picked_point is None: return
        if self._model is None:
            self._set_status(
                "Robot not loaded — cannot teach target (IK needs a robot model)",
                level="warn")
            return
        pt_m = np.asarray(picked_point, dtype=float)
        pt_mm = pt_m * 1000.0
        # Get surface normal at pick point from the picked actor (if available).
        normal = self._surface_normal_at(pt_m)
        if normal is None:
            normal = np.array([0.0, 0.0, 1.0])              # fallback: assume +Z (floor)
        # Orient normal toward the half-space containing the robot base. Open surfaces (floor/
        # single-sided worktable) have ambiguous normals — auto_orient may flip → tool
        # approaches from the wrong side (e.g. from below the floor). Robot always reaches
        # from the base side so forcing the normal toward the base is the physically correct
        # anchor for all surface orientations.
        to_base = np.asarray(self._base_xyz, dtype=float) - pt_mm
        if float(np.dot(normal, to_base)) < 0.0:
            normal = -normal
        # Build TCP target: Z_tcp = -normal (tool point INTO surface),
        # X_tcp = world X projected onto the plane perpendicular to Z_tcp.
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
        # Teach target from TCP matrix (IK + prompt + save) — shared helper
        # with camera grasp-teach.
        name = self._teach_target_from_matrix(
            T,
            default_name=f"SURF_{len(self._targets)+1:02d}",
            prompt_label=(
                f"Target name (pick @ XYZ = "
                f"{pt_mm[0]:.0f}, {pt_mm[1]:.0f}, {pt_mm[2]:.0f} mm):"))
        if name:
            self._set_status(f"Taught '{name}' on surface", level="ok")

    def _surface_normal_at(self, point_m: np.ndarray) -> np.ndarray | None:
        """Return the world-frame surface normal at a 3D point.

        Optimization: cache `mesh + cell_normals` per dataset id. First call per
        mesh runs `compute_normals()` (O(N) cells, may take a few ms for large meshes).
        Subsequent calls for the same mesh → only `find_closest_cell` + index lookup,
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
            # Transform point to actor local frame if UserMatrix is present
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
        """Find all IK solutions via **Pieper analytical** — returns 3-8 native
        solutions without stochastic seeding.

        Pieper returns configurations (front/back × elbow up/down × wrist normal/flip)
        deterministically, exactly (float precision ~1e-12mm). Fallback to batched IK
        if Pieper returns empty (rare — only when pose is completely out of reach).

        Performance: ~150µs (Pieper) vs ~12ms (batched DLS) — 80× faster.
        """
        sols_rad = inverse_kinematics_pieper_gp7(self._model, T_target_tool0)
        if not sols_rad:
            # Fallback: batched DLS with random seeds (legacy path)
            link_attr = getattr(self._model, "joints", None) or getattr(
                self._model, "links", None)
            q_min = np.array([j.joint_min for j in link_attr])
            q_max = np.array([j.joint_max for j in link_attr])
            rng = np.random.RandomState(42)
            q_init_batch = np.array([rng.uniform(q_min, q_max) for _ in range(n_seeds)])
            results = inverse_kinematics_batch(
                self._model, T_target_tool0, q_init_batch,
                max_iter=100, tol_mm=0.5, tol_rad=1e-3)
            sols_rad = [s for s in results if s is not None]
        # SAFETY: post-FK verify every candidate before it can be applied to a
        # target / drive the real robot. The DLS fallback can in principle report a
        # false-converged config; re-running FK and checking the TCP pose against the
        # request rejects any solution whose orientation/position does not match.
        def _fk_matches(q_rad) -> bool:
            try:
                fr = dict(link_frames_urdf(self._model, [float(q) for q in q_rad]))
            except Exception:                               # noqa: BLE001
                return False
            T = fr.get("link_tool0")
            if T is None:
                return False
            d_pos = float(np.linalg.norm(T[:3, 3] - T_target_tool0[:3, 3]))
            R = T_target_tool0[:3, :3] @ T[:3, :3].T
            cos = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
            d_ang = math.degrees(math.acos(cos))
            return d_pos < 1.0 and d_ang < 1.0              # 1 mm / 1°

        # Dedupe (Pieper returns uniquely-different configs but deduping is still safe)
        thresh = np.deg2rad(dedupe_deg)
        unique: list[np.ndarray] = []
        for s in sols_rad:
            arr = np.asarray(s)
            if any(np.max(np.abs(arr - ex)) < thresh for ex in unique):
                continue
            if not _fk_matches(arr):
                logger.warning("Dropping IK solution that fails FK re-check "
                               "(false convergence guard)")
                continue
            unique.append(arr)
            if len(unique) >= max_solutions:
                break
        return [[math.degrees(q) for q in s.tolist()] for s in unique]

    def _on_tgt_change_config(self) -> None:
        """F4 — enumerate IK solutions for selected target's TCP pose, let user
        pick alternative joint configuration."""
        idx = self._tgt_list.currentRow()
        if idx < 0 or not self._targets:
            self._set_status("Select a target to Change Config", level="warn"); return
        name = list(self._targets.keys())[idx]
        tgt = self._targets[name]
        # Need TCP pose to enumerate. Compute T_target_tool0 from stored tcp_pose
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
            "<small><i>★ = current config (Δ ≈ 0°). Δ = max joint difference "
            "from current. Pick & OK to swap.</i></small>"))
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
        """Preview: animate robot to selected target's joints. Does NOT add
        an instruction — only jogs the robot to verify the pose is valid."""
        idx = self._tgt_list.currentRow()
        if idx < 0 or not self._targets:
            self._set_status("Select a target to Go to", level="warn"); return
        if self._prog_thread is not None and self._prog_thread.is_alive():
            self._set_status("Program is running — cannot Go to", level="warn"); return
        name = list(self._targets.keys())[idx]
        target_joints = list(self._targets[name]["joints"])
        # Run animation in a worker thread (like Play, but 1-step).
        self._set_status(f"Going to '{name}'…", level="info")
        def _worker():
            self._animate_to(target_joints, steps=40, dt=0.025)
            self._signals.status.emit(f"Reached '{name}'", "ok")
        threading.Thread(target=_worker, daemon=True).start()

    # Pause / Resume + program list ops + play loop → ProgramPlaybackMixin

    # Cell info + About dialogs → AboutMixin

    # ══════════════════════════════════════════════════════════════════
    # Lifecycle
    # ══════════════════════════════════════════════════════════════════

    def _project_signature(self) -> str:
        """Deterministic snapshot of project DATA (jobs + targets) → JSON.

        Used for dirty-checking: compared against `self._saved_signature` (updated on
        each Save/Load). Does NOT include active_job — switching the viewed job is a
        'view' action, not a data edit → avoids spurious Save prompts."""
        doc = {
            "targets": self._targets,
            "jobs": {
                name: [ins.to_dict() for ins in prog]
                for name, prog in self._jobs.items()
            },
            # Post-processor settings are project data → editing them marks dirty.
            "post_processor": {
                "max_speed_pct": self._pp_max_speed_pct,
                "default_vj": self._pp_default_vj,
                "default_v_mms": self._pp_default_v_mms,
            },
        }
        return json.dumps(doc, sort_keys=True)

    def _has_unsaved_changes(self) -> bool:
        return self._project_signature() != self._saved_signature

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._has_unsaved_changes():
            r = QMessageBox.question(
                self, "Unsaved changes",
                "The project has unsaved changes.\n\n"
                "  • Save — save the project (.json) then exit\n"
                "  • Discard — exit now, discard all changes\n"
                "  • Cancel — go back, do not exit",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save)
            if r == QMessageBox.StandardButton.Cancel:
                event.ignore(); return
            if r == QMessageBox.StandardButton.Save:
                self._on_prog_save_dlg()
                # User cancelled file dialog → still dirty → abort close (avoids
                # unintentional data loss).
                if self._has_unsaved_changes():
                    event.ignore(); return
        # Stop worker threads. Set _cam_closing BEFORE joining the camera so the
        # camera_result slot (worker emits 'stopped' during join) does not touch already-destroyed widgets.
        self._cam_closing = True
        self._prog_stop.set()
        self._exp_stop.set()
        # Stop the sim move animation (Home/Zero/Align) so it cannot emit into a
        # tearing-down VTK plotter.
        if getattr(self, "_anim_stop", None) is not None:
            self._anim_stop.set()
            at = getattr(self, "_anim_thread", None)
            if at is not None and at.is_alive():
                at.join(timeout=1.0)
        # Halt any REAL-robot motion + servo-off before exit.
        self._hse_stop.set()                                 # Run-on-Robot job
        if getattr(self, "_send_pose_stop", None) is not None:
            self._send_pose_stop.set()                       # Phase-1 discrete send
        if getattr(self, "_live_jog_stop", None) is not None:
            self._live_jog_stop.set()                        # Phase-2 live jog
        # Experiment: stop + servo-OFF synchronously (mirror is read-only).
        try:
            if hasattr(self, "_on_stop_experiment"):
                self._on_stop_experiment()
        except Exception:                                    # noqa: BLE001
            pass
        # JOIN every real-robot worker so the servo-OFF / Stop in its finally
        # block actually runs before the interpreter kills the daemon thread at
        # exit — otherwise the robot can be left MOVING with servos ON.
        for _wname in ("_live_jog_thread", "_send_pose_thread",
                       "_hse_thread", "_exp_thread"):
            _wt = getattr(self, _wname, None)
            if _wt is not None and _wt.is_alive():
                _wt.join(timeout=2.0)
        try:
            if getattr(self, "_twin", None) is not None:
                self._twin.stop_mirror()
        except Exception:                                       # noqa: BLE001
            pass
        self._stop_camera()
        try:
            self._plotter.close()
        except Exception:                                   # noqa: BLE001
            pass
        event.accept()
