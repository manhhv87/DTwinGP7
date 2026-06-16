"""
mixin_camera.py
───────────────
CameraMixin: embeds Intel RealSense D455 camera into the Digital Twin Qt app —
live capture (RGB/depth), dataset image saving, and closed-loop control
(detect → grasp pose → teach target → pick sequence → Run on Robot).

REUSE perception/transform layers already in place (DO NOT rewrite):
  - src/perception: D455Camera, MockCamera, ObjectDetector, MockDetector,
                    field_dict, PoseExtractor   (camera + YOLO + pose 3D)
  - src/orchestrator/coord_conv: load_calibration, camera_to_base, make_grasp_pose
  - host (GP7AppQt): _np_to_pixmap, _solve_cartesian (via _teach_target_from_matrix),
                     _on_run_on_robot, _refresh_program_list, _refresh_target_list

Mixin pattern — not instantiated standalone. Host class (GP7AppQt) must provide:
  attributes: _signals, _project_root, _model, _targets, _program, _base_xyz,
              _cam_thread, _cam_stop, _cam_running, _last_camera_objects,
              _last_depth, _last_rgb, _last_grasp_target, _last_grasp_T
  methods:    _set_status, _np_to_pixmap, _teach_target_from_matrix,
              _on_run_on_robot, _refresh_program_list
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFormLayout, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

from .program_model import Instruction
from .qt_widgets import CollapsibleSection

logger = logging.getLogger(__name__)

# Class names matching the detector (DEFAULT_CLASS_NAMES) + metadata for dataset capture.
_CAM_CLASSES = ["tray", "bottle", "cup", "bolt"]
_CAM_LIGHTING = ["bright", "medium", "dim"]
_CAM_OVERLAP = ["none", "light", "medium"]
_CAM_BG = ["gray", "blue", "brown"]
# Mock detection box (pixel) — used by MockDetector + draws a fake object on the mock frame.
_MOCK_BOX = (560, 320, 760, 440)

# Minimum width of the Camera dock — enough to fit every row (source, resolution,
# 2-column button grid). Dock cannot be shrunk below this, only widened.
_CAM_DOCK_MIN_W = 380

# D455 resolution presets (color stream) → (color_size, fps). Depth stays at
# 848×480 (native D455) then aligns→color, so no need to change it. Only applies to
# real D455; Mock is always 1280×720. Changing preset requires Stop→Start (no hot-swap).
_CAM_RES_PRESETS: dict[str, tuple[tuple[int, int], int]] = {
    "1280×720 @30 (default)": ((1280, 720), 30),
    "848×480 @30": ((848, 480), 30),
    "640×480 @30": ((640, 480), 30),
    "640×360 @30": ((640, 360), 30),
    "1280×720 @15": ((1280, 720), 15),
}


class CameraMixin:
    """Live camera dock + dataset capture + vision-guided control (closed-loop)."""

    # ── UI ────────────────────────────────────────────────────────────
    def _build_camera_dock(self) -> None:
        """'Camera (D455)' dock on the right — live view + dataset + control."""
        dock = QDockWidget("Camera (D455)", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea
                             | Qt.DockWidgetArea.LeftDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable
                         | QDockWidget.DockWidgetFeature.DockWidgetMovable
                         | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        # All content wrapped in QScrollArea → short screen/dock can still
        # scroll to every button (no clipping like the old group). RoboDK-style: live
        # view stretches at top, function panels are collapsible below.
        body = QWidget(); v = QVBoxLayout(body)
        v.setContentsMargins(6, 6, 6, 6); v.setSpacing(6)

        # Live view — stretches within [180, 360]px. Use SizePolicy.Ignored so
        # QLabel does NOT use the pixmap as sizeHint: each frame sets a new pixmap
        # (scaled to label size) — Expanding would create a runaway grow loop
        # pushing buttons down. Ignored + maxHeight prevents that.
        self._cam_view = QLabel("Camera off — press Start")
        self._cam_view.setMinimumHeight(180)
        self._cam_view.setMaximumHeight(360)
        self._cam_view.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._cam_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_view.setStyleSheet(
            "background:#0d0d12; border:1px solid #2a2a33; "
            "border-radius:6px; color:#888;")
        v.addWidget(self._cam_view, 1)

        # Source + Start/Stop
        src_row = QHBoxLayout()
        self._cam_source_cb = QComboBox()
        self._cam_source_cb.addItems(["Auto (D455→Mock)", "D455", "Mock"])
        src_row.addWidget(QLabel("Source:"))
        src_row.addWidget(self._cam_source_cb, 1)
        self._cam_start_btn = QPushButton("Start")
        self._cam_start_btn.clicked.connect(self._start_camera)
        self._cam_stop_btn = QPushButton("Stop")
        self._cam_stop_btn.setEnabled(False)
        self._cam_stop_btn.clicked.connect(self._stop_camera)
        src_row.addWidget(self._cam_start_btn)
        src_row.addWidget(self._cam_stop_btn)
        v.addLayout(src_row)

        # Resolution (real D455 only; change takes effect after Stop→Start).
        res_row = QHBoxLayout()
        self._cam_res_cb = QComboBox()
        self._cam_res_cb.addItems(list(_CAM_RES_PRESETS.keys()))
        self._cam_res_cb.setToolTip(
            "Resolution + FPS for the real D455. To change while running: press "
            "Stop then Start again. (Mock is always 1280×720.)")
        res_row.addWidget(QLabel("Resolution:"))
        res_row.addWidget(self._cam_res_cb, 1)
        v.addLayout(res_row)

        # Display + detector toggles. Depth/Overlay are mirrored into bools that
        # the worker thread reads (never reads widget from worker) → instant, safe.
        opt_row = QHBoxLayout()
        self._cam_depth_chk = QCheckBox("Depth colormap")
        self._cam_detector_chk = QCheckBox("Detector")
        self._cam_detector_chk.setToolTip(
            "Enable object detection. Changes take effect after Stop → Start again "
            "(detector loads once at startup). Depth/Overlay change instantly.")
        self._cam_overlay_chk = QCheckBox("Overlay"); self._cam_overlay_chk.setChecked(True)
        self._cam_show_depth = False
        self._cam_show_overlay = True
        self._cam_depth_chk.toggled.connect(
            lambda c: setattr(self, "_cam_show_depth", bool(c)))
        self._cam_overlay_chk.toggled.connect(
            lambda c: setattr(self, "_cam_show_overlay", bool(c)))
        opt_row.addWidget(self._cam_depth_chk)
        opt_row.addWidget(self._cam_detector_chk)
        opt_row.addWidget(self._cam_overlay_chk)
        opt_row.addStretch(1)
        v.addLayout(opt_row)

        self._cam_info = QLabel("Source: —  |  0.0 FPS  |  0 objects")
        self._cam_info.setWordWrap(True)
        self._cam_info.setStyleSheet("color:#9aa0aa; font-size:11px;")
        v.addWidget(self._cam_info)

        # Dataset capture — collapsible (collapsed by default, less used than control).
        ds_sec = CollapsibleSection("Dataset — capture images", expanded=False)
        ds_form = QFormLayout()
        # Class: list defined by the task (CellConfig.object_classes) +
        # management button (add/delete/edit). Combo populated via _refresh_class_combo().
        self._cam_cls_cb = QComboBox()
        cls_row = QHBoxLayout()
        cls_row.setContentsMargins(0, 0, 0, 0); cls_row.setSpacing(4)
        cls_row.addWidget(self._cam_cls_cb, 1)
        btn_mng_cls = QPushButton("Manage…")
        btn_mng_cls.setToolTip("Define the class list for the task (saved into Cell)")
        btn_mng_cls.clicked.connect(self._manage_classes_dlg)
        cls_row.addWidget(btn_mng_cls)
        self._cam_light_cb = QComboBox(); self._cam_light_cb.addItems(_CAM_LIGHTING)
        self._cam_overlap_cb = QComboBox(); self._cam_overlap_cb.addItems(_CAM_OVERLAP)
        self._cam_bg_cb = QComboBox(); self._cam_bg_cb.addItems(_CAM_BG)
        ds_form.addRow("Class", cls_row)
        ds_form.addRow("Lighting", self._cam_light_cb)
        ds_form.addRow("Overlap", self._cam_overlap_cb)
        ds_form.addRow("Background", self._cam_bg_cb)
        ds_sec.add_layout(ds_form)
        self._cam_save_depth_chk = QCheckBox("Save depth (.npy)")
        self._cam_save_depth_chk.setChecked(True)
        self._cam_save_depth_chk.setToolTip(
            "On: save both RGB (.png) + depth (.npy, meters). Off: save RGB only.")
        ds_sec.add_widget(self._cam_save_depth_chk)
        cap_btn = QPushButton("📷 Capture")
        cap_btn.clicked.connect(self._camera_capture)
        ds_sec.add_widget(cap_btn)
        v.addWidget(ds_sec)

        # Control — vision-guided (closed-loop) — collapsible, expanded by default. 2×2
        # grid of ALL BUTTONS (balanced 2 columns, flush edges); Approach Z = own full-width row.
        ctl_sec = CollapsibleSection(
            "Control — vision-guided (closed-loop)", expanded=True)
        # Remove horizontal content indent → buttons span the full section width ⇒ left/
        # right edges align with Dataset & Control headers.
        ctl_sec.content_layout().setContentsMargins(0, 6, 0, 6)
        grid = QGridLayout()
        grid.setHorizontalSpacing(6); grid.setVerticalSpacing(6)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1)

        teach_btn = QPushButton("Detect → Teach grasp")
        teach_btn.setToolTip("Take the detected object → grasp pose (calibration) → "
                              "IK → save target")
        teach_btn.clicked.connect(self._camera_teach_grasp)
        pick_btn = QPushButton("Pick → Program")
        pick_btn.setToolTip("Add approach→grasp→close→retreat sequence to the current job")
        pick_btn.clicked.connect(self._camera_pick_to_program)
        run_btn = QPushButton("▶ Run on Robot")
        run_btn.clicked.connect(self._on_run_on_robot)
        sync_btn = QPushButton("Sync Camera → Cell")
        sync_btn.setToolTip("Write pose (from hand-eye calibration) + real intrinsics "
                            "into the Cell's Camera node → draw frustum (RoboDK-style)")
        sync_btn.clicked.connect(self._camera_sync_to_cell)
        # minWidth 0 + Expanding → 2 columns share width equally regardless of label length ⇒ flush edges.
        for b in (teach_btn, pick_btn, run_btn, sync_btn):
            b.setMinimumWidth(0)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        grid.addWidget(teach_btn, 0, 0)
        grid.addWidget(pick_btn,  0, 1)
        grid.addWidget(run_btn,   1, 0)
        grid.addWidget(sync_btn,  1, 1)
        ctl_sec.add_layout(grid)

        # Approach Z — own full-width row (parameter for 'Pick → Program').
        app_row = QHBoxLayout()
        app_row.addWidget(QLabel("Approach Z (for Pick):"))
        self._cam_approach_spin = QDoubleSpinBox()
        self._cam_approach_spin.setRange(10.0, 400.0)
        self._cam_approach_spin.setSingleStep(10.0)
        self._cam_approach_spin.setValue(80.0)
        self._cam_approach_spin.setSuffix(" mm")
        app_row.addWidget(self._cam_approach_spin, 1)
        ctl_sec.add_layout(app_row)
        v.addWidget(ctl_sec)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        dock.setWidget(scroll)
        self._camera_dock = dock
        # Same tab group as Control/Cell/Program on the left side — tabify into jog dock
        # so 4 panels share one tab group (jog/cell/program already tabified together).
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        if hasattr(self, "_jog_dock"):
            self.tabifyDockWidget(self._jog_dock, dock)
        dock.setVisible(False)                      # hidden by default — enable via View
        self._refresh_class_combo()                 # load classes from cell (or default)
        # DYNAMIC min-width: only apply _CAM_DOCK_MIN_W when Camera is the active tab (see
        # _sync_cam_dock_check) → cannot shrink below this when active, other tabs
        # can still shrink. Also sync the menu tick.
        dock.visibilityChanged.connect(self._sync_cam_dock_check)

    def _sync_cam_dock_check(self, visible: bool) -> None:
        """Camera becomes active tab → min = _CAM_DOCK_MIN_W (cannot shrink, only
        expand) + resize area to that value; sync menu tick."""
        if hasattr(self, "_act_camera_dock"):
            self._set_toggle(self._act_camera_dock, bool(visible))
        if visible and hasattr(self, "_camera_dock"):
            self._apply_active_dock_min(self._camera_dock, _CAM_DOCK_MIN_W)
            QTimer.singleShot(0, lambda: self.resizeDocks(
                [self._camera_dock], [_CAM_DOCK_MIN_W], Qt.Orientation.Horizontal))

    # ── Object classes (defined by the task) ───────────────────────────
    def _refresh_class_combo(self) -> None:
        """Reload the Class combo from CellConfig.object_classes (fallback to default),
        preserving the current selection if still present."""
        cfg = getattr(self, "_cell_config", None)
        classes = list(getattr(cfg, "object_classes", None) or _CAM_CLASSES)
        cur = self._cam_cls_cb.currentText()
        self._cam_cls_cb.blockSignals(True)
        self._cam_cls_cb.clear()
        self._cam_cls_cb.addItems(classes)
        if cur in classes:
            self._cam_cls_cb.setCurrentText(cur)
        self._cam_cls_cb.blockSignals(False)

    def _manage_classes_dlg(self) -> None:
        """Dialog for defining the task's class list → saved to CellConfig.

        Layout: list on left + button column on right (Add/Edit/Delete/Up/Down +
        OK/Cancel) — all buttons same width, aligned, no gaps."""
        from PyQt6.QtWidgets import (
            QDialog, QInputDialog, QListWidget, QListWidgetItem, QMessageBox)
        self._ensure_cell_config()
        cur = list(getattr(self._cell_config, "object_classes", None) or _CAM_CLASSES)
        dlg = QDialog(self); dlg.setWindowTitle("Manage classes")
        dlg.setMinimumSize(420, 300)
        outer = QHBoxLayout(dlg); outer.setSpacing(8)
        lst = QListWidget()
        lst.setAlternatingRowColors(True)

        def _mk_item(text: str):
            it = QListWidgetItem(text)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEditable)
            return it
        for c in cur:
            lst.addItem(_mk_item(c))
        outer.addWidget(lst, 1)

        # Right button column — includes OK/Cancel, all same width + UNIFORM spacing.
        col = QVBoxLayout(); col.setSpacing(8)
        b_add = QPushButton("＋  Add")
        b_edit = QPushButton("✎  Edit")
        b_del = QPushButton("－  Delete")
        b_up = QPushButton("↑  Up")
        b_dn = QPushButton("↓  Down")
        b_ok = QPushButton("OK"); b_cancel = QPushButton("Cancel")
        for b in (b_add, b_edit, b_del, b_up, b_dn, b_ok, b_cancel):
            b.setMinimumWidth(108)
            col.addWidget(b)
        col.addStretch(1)
        outer.addLayout(col)

        def _add():
            name, ok = QInputDialog.getText(dlg, "Add class", "Class name:")
            if ok and name.strip():
                it = _mk_item(name.strip()); lst.addItem(it); lst.setCurrentItem(it)

        def _edit():
            it = lst.currentItem()
            if it is not None:
                lst.editItem(it)

        def _del():
            r = lst.currentRow()
            if r >= 0:
                lst.takeItem(r)

        def _move(d: int):
            r = lst.currentRow()
            if r < 0:
                return
            nr = r + d
            if 0 <= nr < lst.count():
                lst.insertItem(nr, lst.takeItem(r)); lst.setCurrentRow(nr)
        b_add.clicked.connect(_add); b_edit.clicked.connect(_edit)
        b_del.clicked.connect(_del)
        b_up.clicked.connect(lambda: _move(-1)); b_dn.clicked.connect(lambda: _move(1))
        b_ok.setDefault(True)
        b_ok.clicked.connect(dlg.accept); b_cancel.clicked.connect(dlg.reject)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        names: list[str] = []; seen: set[str] = set()
        for i in range(lst.count()):
            s = lst.item(i).text().strip()
            if s and s not in seen:
                seen.add(s); names.append(s)
        if not names:
            QMessageBox.warning(self, "Manage classes",
                                "List is empty — keeping the previous one.")
            return
        self._cell_config.object_classes = names
        self._refresh_class_combo()
        self._set_status(
            f"Updated {len(names)} classes: {', '.join(names)}", level="ok")

    # ── Acquisition thread ─────────────────────────────────────────────
    def _start_camera(self) -> None:
        if self._cam_running:
            return
        if self._cam_thread is not None and self._cam_thread.is_alive():
            # A previous _stop_camera join timed out and its worker is still inside
            # the camera (get_frame / pipeline.stop in finally). Starting now would
            # open a SECOND pipeline on the same D455 → librealsense crash/hang.
            self._set_status(
                "Camera: previous session still closing — wait a moment, then retry",
                level="warn")
            return
        self._cam_source = self._cam_source_cb.currentText()
        self._cam_use_detector = self._cam_detector_chk.isChecked()
        self._cam_color_size, self._cam_fps = _CAM_RES_PRESETS[
            self._cam_res_cb.currentText()]
        self._cam_frame_pending = False
        self._last_display = None
        self._cam_stop.clear()
        self._cam_running = True
        self._cam_start_btn.setEnabled(False)
        self._cam_stop_btn.setEnabled(True)
        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()
        self._set_status("Camera: starting up…", level="info")

    def _stop_camera(self) -> None:
        self._cam_stop.set()
        t = self._cam_thread
        if t is not None and t.is_alive():
            t.join(timeout=3.5)                  # > camera frame timeout (~2s) + filters
        # If the worker is STILL alive after the join, keep the handle and stay
        # 'running' so _start_camera refuses to open a 2nd pipeline on the same
        # device until this worker actually exits (its finally calls camera.stop()).
        if t is not None and t.is_alive():
            self._set_status(
                "Camera: worker still closing (will free the device shortly)",
                level="warn")
            return
        self._cam_thread = None
        self._cam_running = False

    def _camera_loop(self) -> None:
        """Background thread: get_frame → (detect → pose) → emit camera_result."""
        from ...perception import (  # lazy — avoid heavy import at startup
            D455Camera, MockCamera, MockDetector, ObjectDetector, PoseExtractor,
        )
        from ...perception.detector import field_dict      # not exported in __init__
        try:
            camera, source_label = self._open_camera(
                self._cam_source, D455Camera, MockCamera,
                self._cam_color_size, self._cam_fps)
        except Exception as e:                              # noqa: BLE001
            self._signals.status.emit(f"Camera failed to open: {e}", "err")
            self._cam_running = False
            self._signals.camera_result.emit({"stopped": True})
            return
        detector = None
        if self._cam_use_detector:
            try:
                detector = self._open_detector(ObjectDetector, MockDetector)
            except Exception as e:                          # noqa: BLE001
                self._signals.status.emit(
                    f"Detector error → running without detection: {e}", "warn")
        extractor = PoseExtractor(camera.intrinsics)
        self._signals.status.emit(
            f"Camera ON — source {source_label}"
            + (" + detector" if detector is not None else ""), "ok")
        period = 1.0 / 15.0
        prev_t = time.time()
        try:
            while not self._cam_stop.is_set():
                t0 = time.time()
                rgb, depth = camera.get_frame()
                if rgb is None or depth is None:
                    time.sleep(0.02); continue
                objects: list[dict] = []
                if detector is not None:
                    for det in detector.detect(rgb):
                        enr = extractor.extract(field_dict(det), depth)
                        if enr is not None and enr.get("pose_camera") is not None:
                            objects.append(enr)
                # Actual FPS = reciprocal of time between two frames (including sleep).
                now = time.time()
                self._last_fps = 1.0 / max(now - prev_t, 1e-6)
                prev_t = now
                # Store the raw frame as ONE coherent (rgb, depth) tuple so capture
                # reads a matched pair in a single atomic ref-read (reading _last_rgb
                # then _last_depth separately could straddle a worker update → a
                # mismatched RGB/depth pair). Copy: the RealSense buffer is reused.
                _rgb_c = np.ascontiguousarray(rgb).copy()
                _depth_c = np.asarray(depth).copy()
                self._last_frame = (_rgb_c, _depth_c)   # atomic coherent pair
                self._last_rgb = _rgb_c                 # individual refs (display)
                self._last_depth = _depth_c
                self._last_camera_objects = objects
                self._last_intrinsics = dict(camera.intrinsics)
                self._last_source = source_label
                # BACKPRESSURE: only build the HEAVY display image (depth colormap/overlay)
                # + emit WHEN the main thread has consumed the previous frame → avoids
                # blocking the event loop (Stop always drains) and wastes no CPU when UI is slow.
                if not self._cam_frame_pending:
                    try:
                        if self._cam_show_depth and depth is not None:
                            self._last_display = self._depth_to_color(depth)
                        else:
                            disp = np.ascontiguousarray(rgb[:, :, ::-1])  # BGR→RGB
                            if self._cam_show_overlay and objects:
                                disp = self._draw_overlay(disp, objects)
                            self._last_display = disp
                    except Exception:                       # noqa: BLE001
                        self._last_display = np.ascontiguousarray(rgb[:, :, ::-1])
                    self._cam_frame_pending = True
                    self._signals.camera_result.emit({"tick": True})
                left = period - (time.time() - t0)
                if left > 0:
                    time.sleep(left)
        except Exception as e:                              # noqa: BLE001
            self._signals.status.emit(f"Camera loop error: {e}", "err")
        finally:
            try:
                camera.stop()
            except Exception:                               # noqa: BLE001
                pass
            self._cam_running = False
            self._signals.camera_result.emit({"stopped": True})

    def _open_camera(self, source: str, D455Camera, MockCamera,
                     color_size=(1280, 720), fps: int = 30):
        """Build camera from source selection. Returns (camera, source_label).

        Auto: try D455, on error → fallback to Mock. D455: real hardware required (raises on error).
        Mock: synthetic frame + fake object to demo the pipeline without hardware.
        color_size/fps only applies to real D455 (Mock uses fixed 1280×720 intrinsics).
        """
        want = source.split()[0].lower()                    # auto / d455 / mock
        if want != "mock":
            try:
                return D455Camera(color_size=color_size, fps=fps), "D455"
            except Exception as e:                          # noqa: BLE001
                if want == "d455":
                    raise
                self._signals.status.emit(
                    f"D455 not ready → fallback to Mock ({e})", "warn")
        return MockCamera(rgb_frames=[self._synthetic_mock_frame()]), "Mock"

    def _open_detector(self, ObjectDetector, MockDetector):
        """YOLO if weights models/*.pt|onnx are present, else MockDetector (1 fake object)."""
        weights = (sorted(self._project_root.glob("models/*.pt"))
                   + sorted(self._project_root.glob("models/*.onnx")))
        if weights:
            return ObjectDetector(model_path=str(weights[0]))
        self._signals.status.emit(
            "No YOLO weights yet (models/*.pt) → MockDetector", "warn")
        return MockDetector(scripted=[[
            MockDetector.make_detection("tray", mask_box=_MOCK_BOX)]])

    @staticmethod
    def _synthetic_mock_frame() -> np.ndarray:
        """Synthetic BGR frame (gradient + 1 'object' at _MOCK_BOX) for mock view."""
        h, w = 720, 1280
        ramp = np.linspace(35, 80, h).astype(np.uint8)
        frame = np.repeat(ramp[:, None], w, axis=1)
        rgb = np.stack([frame, frame, frame], axis=-1)
        x1, y1, x2, y2 = _MOCK_BOX
        rgb[y1:y2, x1:x2] = (150, 150, 155)             # 'object' light gray
        return rgb

    # ── Main-thread slot ───────────────────────────────────────────────
    def _on_camera_result(self, payload) -> None:
        # App is closing → do NOT touch widgets (may already be destroyed when worker
        # emits 'stopped' during join in closeEvent).
        if getattr(self, "_cam_closing", False):
            return
        # LIGHTWEIGHT slot: worker already built _last_display (RGB) — main only does
        # QPixmap + setPixmap. Clear backpressure so worker can push the next frame.
        self._cam_frame_pending = False
        if isinstance(payload, dict) and payload.get("stopped"):
            # Ignore stale 'stopped' arriving late (fast Stop→Start): if a new camera
            # thread is already running, do not reset buttons to the off state.
            if self._cam_thread is not None and self._cam_thread.is_alive():
                return
            self._cam_start_btn.setEnabled(True)
            self._cam_stop_btn.setEnabled(False)
            self._cam_running = False
            self._cam_view.setText("Camera off — press Start")
            return
        disp = self._last_display
        if disp is None:
            return
        pm = self._np_to_pixmap(disp)
        self._cam_view.setPixmap(pm.scaled(
            self._cam_view.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        intr = self._last_intrinsics or {}
        self._cam_info.setText(
            f"Source: {self._last_source or '?'}  |  "
            f"{getattr(self, '_last_fps', 0.0):.1f} FPS  |  "
            f"{intr.get('width', '?')}×{intr.get('height', '?')}  |  "
            f"{len(self._last_camera_objects)} objects")

    def _draw_overlay(self, rgb, objects):
        """Draw bbox + centroid + class label onto an RGB image (in-place on a copy)."""
        import cv2
        img = np.ascontiguousarray(rgb)
        for o in objects:
            bbox = o.get("bbox")
            if bbox:
                x1, y1, x2, y2 = (int(v) for v in bbox)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{o.get('class_name', '?')} {o.get('confidence', 0.0):.2f}"
                cv2.putText(img, label, (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            uv = o.get("pixel_uv")
            if uv:
                cv2.circle(img, (int(uv[0]), int(uv[1])), 5, (255, 90, 90), -1)
        return img

    @staticmethod
    def _depth_to_color(depth):
        """Depth (meters) → RGB color image (JET), normalized by 5–95th percentile."""
        import cv2
        d = np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=0.0)
        valid = d[d > 0]
        if valid.size:
            lo, hi = float(np.percentile(valid, 5)), float(np.percentile(valid, 95))
        else:
            lo, hi = 0.0, 1.0
        hi = max(hi, lo + 1e-3)
        u8 = (np.clip((d - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)
        cm = cv2.applyColorMap(u8, cv2.COLORMAP_JET)        # BGR
        cm[d <= 0] = 0
        return np.ascontiguousarray(cm[:, :, ::-1])         # BGR→RGB

    # ── Dataset capture ────────────────────────────────────────────────
    def _camera_capture(self) -> None:
        """Save current frame: RGB (.png, BGR) + depth (.npy, meters) → data/raw.

        Unchecking 'Save depth' → RGB only."""
        frame = getattr(self, "_last_frame", None)
        if frame is None:
            self._set_status("No camera frame yet — press Start first", level="warn")
            return
        rgb, depth = frame                          # coherent pair (atomic read)
        save_depth = self._cam_save_depth_chk.isChecked()
        if save_depth and depth is None:
            self._set_status(
                "No depth to save — uncheck 'Save depth' or Start again",
                level="warn")
            return
        import cv2

        from ...utils import ensure_dir, timestamp
        out = ensure_dir(self._project_root / "data" / "raw")
        # Counter to prevent overwriting when capturing >1 image in the same second
        # (timestamp resolution is seconds). Matches dataset script 01 convention (_{NNNN}).
        seq = getattr(self, "_cam_capture_seq", 0)
        self._cam_capture_seq = seq + 1
        name = (f"{self._cam_cls_cb.currentText()}_"
                f"{self._cam_light_cb.currentText()}_0_"
                f"{self._cam_overlap_cb.currentText()}_"
                f"{self._cam_bg_cb.currentText()}_{timestamp('%H%M%S')}_{seq:04d}")
        try:
            cv2.imwrite(str(out / f"{name}_rgb.png"), rgb)  # rgb = BGR ✓
            if save_depth:
                np.save(str(out / f"{name}_depth.npy"), np.asarray(depth))
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"Capture error: {e}", level="err"); return
        kind = "RGB+depth" if save_depth else "RGB"
        self._set_status(f"Saved {name} ({kind}) → data/raw", level="ok")

    # ── Vision-guided control (closed-loop) ────────────────────────────
    def _camera_teach_grasp(self) -> None:
        """Latest detected object → camera_to_base → make_grasp_pose → teach target."""
        if self._model is None:
            self._set_status(
                "Robot not loaded — load the robot before teaching a grasp", level="warn")
            return
        objects = self._last_camera_objects
        if not objects:
            self._set_status(
                "No object detected — enable Detector + Start", level="warn")
            return
        target = max(objects, key=lambda o: o.get("confidence", 0.0))
        pose_cam = target.get("pose_camera")
        if pose_cam is None:
            self._set_status("Object missing 3D pose (no depth)", level="warn")
            return
        from ..coord_conv import (
            camera_to_base, camera_yaw_to_base, load_calibration, make_grasp_pose)
        calib = self._project_root / "config" / "calibration" / "T_base_camera.npy"
        try:
            t_bc = load_calibration(calib)
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"Calibration error: {e}", level="err"); return
        xyz_base = camera_to_base(np.asarray(pose_cam[:3], dtype=float), t_bc)
        # Transform the PCA yaw camera→base via the SHARED helper (the raw image-frame
        # yaw alone leaves the gripper mis-aligned with the object long axis — same
        # transform the orchestrator pick path uses). yaw_offset_deg is an optional
        # gripper-vs-PCA calibration constant; default 0 (operator reviews this teach).
        yaw_base = camera_yaw_to_base(float(pose_cam[3]), t_bc)
        yaw_off = float(getattr(self, "_grasp_yaw_offset_deg", 0.0) or 0.0)
        T = make_grasp_pose(xyz_base, yaw_base, yaw_off)
        cls = target.get("class_name", "?")
        name = self._teach_target_from_matrix(
            T, default_name=f"PICK_{len(self._targets) + 1:02d}",
            prompt_label=(
                f"Grasp target '{cls}' @ base XYZ = "
                f"{xyz_base[0]:.0f}, {xyz_base[1]:.0f}, {xyz_base[2]:.0f} mm:"))
        if name:
            self._last_grasp_target = name
            self._last_grasp_T = T
            self._set_status(
                f"Taught grasp '{name}' from object '{cls}'", level="ok")

    def _camera_pick_to_program(self) -> None:
        """Append pick sequence (open→approach→grasp→close→retreat) to the current job."""
        name = getattr(self, "_last_grasp_target", None)
        T = getattr(self, "_last_grasp_T", None)
        if not name or name not in self._targets or T is None:
            self._set_status(
                "No grasp target yet — press 'Detect → Teach grasp' first",
                level="warn")
            return
        approach_mm = float(self._cam_approach_spin.value())
        t_app = np.asarray(T, dtype=float).copy()
        t_app[2, 3] += approach_mm                          # lift along +Z base
        app_name = self._teach_target_from_matrix(
            t_app, default_name=f"{name}_APP", prompt=False)
        if not app_name:
            self._set_status(
                "IK approach failed — reduce Approach Z", level="warn")
            return
        seq = [
            Instruction(type="SetGripper", gripper_close=False),    # open gripper
            Instruction(type="MoveJ", target_name=app_name),        # move to approach
            Instruction(type="MoveL", target_name=name),            # descend straight to grasp
            Instruction(type="SetGripper", gripper_close=True),     # close gripper
            Instruction(type="Wait", wait_seconds=0.3),
            Instruction(type="MoveL", target_name=app_name),        # lift up (retreat)
        ]
        self._program.extend(seq)
        self._refresh_program_list()
        self._set_status(
            f"Added pick sequence for '{name}' ({len(seq)} instructions) to job",
            level="ok")

    def _camera_sync_to_cell(self) -> None:
        """Write real D455 camera into cell `camera` node (RoboDK single-source):
        pose from hand-eye calibration, real pixel intrinsics from current frame.
        → Cell tree + frustum reflect the physical camera accurately."""
        intr = getattr(self, "_last_intrinsics", None)
        if not intr:
            self._set_status(
                "No camera frame yet — Start the camera before syncing", level="warn")
            return
        from ...cell.cell_models import (
            CameraConfig, CameraIntrinsics, PoseConfig)
        from ..coord_conv import load_calibration
        from .control_panel import _matrix_to_xyz_rpy_deg
        self._ensure_cell_config()
        # Extrinsics: camera-in-base pose from hand-eye calibration.
        calib = self._project_root / "config" / "calibration" / "T_base_camera.npy"

        def _fallback_pose(reason: str):
            existing = getattr(self._cell_config, "camera", None)
            if existing is not None:
                return existing.pose, f"keeping previous pose ({reason})"
            return (PoseConfig(xyz_mm=(700.0, 0.0, 1200.0),
                               rpy_deg=(180.0, 0.0, 0.0)),
                    f"default pose ({reason})")
        # Distinguish a genuine calibration LOAD/shape error from a calibration that
        # loaded fine but produced an OUT-OF-BOUNDS PoseConfig — the old single try
        # mislabeled the latter as "not calibrated".
        try:
            x, y, z, rx, ry, rz = _matrix_to_xyz_rpy_deg(load_calibration(calib))
        except Exception as e:                              # noqa: BLE001
            pose, pose_note = _fallback_pose(f"not calibrated: {e}")
        else:
            try:
                pose = PoseConfig(xyz_mm=(x, y, z), rpy_deg=(rx, ry, rz))
                pose_note = "pose from calibration"
            except Exception as e:                          # noqa: BLE001
                pose, pose_note = _fallback_pose(
                    f"calibration pose out of cell bounds — check units/limits: {e}")
        is_d455 = "D455" in str(getattr(self, "_last_source", "") or "")
        w = int(intr.get("width", 1280)); h = int(intr.get("height", 720))
        cam = CameraConfig(
            type="real" if is_d455 else "virtual",
            model="Intel RealSense D455" if is_d455 else "Mock camera",
            mount="eye_to_hand",
            pose=pose,
            intrinsics=CameraIntrinsics(
                fx=float(intr["fx"]), fy=float(intr["fy"]),
                cx=float(intr["ppx"]), cy=float(intr["ppy"]),
                size_px=(w, h)))
        self._cell_config.camera = cam
        self._refresh_cell_tree()
        self._load_cell_assets()
        self._set_status(
            f"Sync Camera → Cell: {pose_note}, real intrinsics "
            f"fx={float(intr['fx']):.0f} ({w}×{h})", level="ok")
