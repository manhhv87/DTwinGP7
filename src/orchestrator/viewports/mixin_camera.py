"""
mixin_camera.py
───────────────
CameraMixin: nhúng camera Intel RealSense D455 vào app Digital Twin Qt —
thu ảnh live (RGB/depth), chụp lưu dataset, và điều khiển vòng kín
(detect → grasp pose → teach target → pick sequence → Run on Robot).

TÁI DÙNG tối đa tầng perception/transform đã có (KHÔNG viết lại):
  - src/perception: D455Camera, MockCamera, ObjectDetector, MockDetector,
                    field_dict, PoseExtractor   (camera + YOLO + pose 3D)
  - src/orchestrator/coord_conv: load_calibration, camera_to_base, make_grasp_pose
  - host (GP7AppQt): _np_to_pixmap, _solve_cartesian (qua _teach_target_from_matrix),
                     _on_run_on_robot, _refresh_program_list, _refresh_target_list

Mixin pattern — không khởi tạo độc lập. Host class (GP7AppQt) phải cung cấp:
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

# Class names khớp detector (DEFAULT_CLASS_NAMES) + metadata cho dataset capture.
_CAM_CLASSES = ["tray", "bottle", "cup", "bolt"]
_CAM_LIGHTING = ["bright", "medium", "dim"]
_CAM_OVERLAP = ["none", "light", "medium"]
_CAM_BG = ["gray", "blue", "brown"]
# Mock detection box (pixel) — dùng cho MockDetector + vẽ object giả lên frame mock.
_MOCK_BOX = (560, 320, 760, 440)

# Bề rộng tối thiểu của dock Camera — đủ chứa mọi hàng (nguồn, độ phân giải,
# lưới nút 2 cột). Dock không kéo nhỏ hơn mức này, chỉ kéo rộng ra.
_CAM_DOCK_MIN_W = 380

# Preset độ phân giải D455 (color stream) → (color_size, fps). Depth giữ
# 848×480 (native D455) rồi align→color nên không cần đổi theo. Chỉ áp cho D455
# thật; Mock luôn 1280×720. Đổi preset cần Stop→Start (không hot-swap stream).
_CAM_RES_PRESETS: dict[str, tuple[tuple[int, int], int]] = {
    "1280×720 @30 (default)": ((1280, 720), 30),
    "848×480 @30": ((848, 480), 30),
    "640×480 @30": ((640, 480), 30),
    "640×360 @30": ((640, 360), 30),
    "1280×720 @15": ((1280, 720), 15),
}


class CameraMixin:
    """Live camera dock + dataset capture + vision-guided control (vòng kín)."""

    # ── UI ────────────────────────────────────────────────────────────
    def _build_camera_dock(self) -> None:
        """Dock 'Camera (D455)' bên phải — live view + dataset + control."""
        dock = QDockWidget("Camera (D455)", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea
                             | Qt.DockWidgetArea.LeftDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable
                         | QDockWidget.DockWidgetFeature.DockWidgetMovable
                         | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        # Toàn bộ nội dung bọc trong QScrollArea → màn hình thấp/dock ngắn vẫn
        # cuộn tới được mọi nút (không bị cắt như group cũ). RoboDK-style: live
        # view co giãn ở trên, các panel chức năng thu gọn (collapsible) bên dưới.
        body = QWidget(); v = QVBoxLayout(body)
        v.setContentsMargins(6, 6, 6, 6); v.setSpacing(6)

        # Live view — co giãn trong [180, 360]px. Dùng SizePolicy.Ignored để
        # QLabel KHÔNG lấy kích thước pixmap làm sizeHint: mỗi frame set pixmap
        # mới (scale theo size label) — nếu Expanding sẽ tạo vòng lặp phình to
        # dần đẩy các nút xuống. Ignored + maxHeight chặn đứng việc đó.
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

        # Resolution (chỉ D455 thật; đổi xong Stop→Start để áp).
        res_row = QHBoxLayout()
        self._cam_res_cb = QComboBox()
        self._cam_res_cb.addItems(list(_CAM_RES_PRESETS.keys()))
        self._cam_res_cb.setToolTip(
            "Resolution + FPS for the real D455. To change while running: press "
            "Stop then Start again. (Mock is always 1280×720.)")
        res_row.addWidget(QLabel("Resolution:"))
        res_row.addWidget(self._cam_res_cb, 1)
        v.addLayout(res_row)

        # Display + detector toggles. Depth/Overlay phản chiếu vào bool mà
        # worker thread đọc (không đọc widget từ worker) → đổi tức thì, an toàn.
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

        # Dataset capture — collapsible (gập mặc định, ít dùng hơn control).
        ds_sec = CollapsibleSection("Dataset — capture images", expanded=False)
        ds_form = QFormLayout()
        # Class: danh sách do bài toán định nghĩa (CellConfig.object_classes) +
        # nút quản lý (thêm/xóa/sửa). Combo nạp qua _refresh_class_combo().
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

        # Control — vision-guided (vòng kín) — collapsible, xổ sẵn. Lưới 2×2
        # TOÀN NÚT (cân bằng 2 cột thẳng mép); Approach Z = hàng riêng full-width.
        ctl_sec = CollapsibleSection(
            "Control — vision-guided (closed-loop)", expanded=True)
        # Bỏ thụt lề ngang nội dung → nút span đúng bề rộng section ⇒ mép trái/
        # phải thẳng hàng với header Dataset & Control.
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
        # minWidth 0 + Expanding → 2 cột chia đều bất kể độ dài chữ ⇒ mép thẳng.
        for b in (teach_btn, pick_btn, run_btn, sync_btn):
            b.setMinimumWidth(0)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        grid.addWidget(teach_btn, 0, 0)
        grid.addWidget(pick_btn,  0, 1)
        grid.addWidget(run_btn,   1, 0)
        grid.addWidget(sync_btn,  1, 1)
        ctl_sec.add_layout(grid)

        # Approach Z — hàng riêng full-width (tham số cho 'Pick → Program').
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
        # Cùng cụm (tab) với Control/Cell/Program ở vùng trái — tabify vào jog dock
        # để 4 panel chia sẻ 1 nhóm tab (jog/cell/program đã tabified với nhau).
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        if hasattr(self, "_jog_dock"):
            self.tabifyDockWidget(self._jog_dock, dock)
        dock.setVisible(False)                      # ẩn mặc định — bật qua View
        self._refresh_class_combo()                 # nạp class từ cell (hoặc default)
        # Min-width ĐỘNG: chỉ áp _CAM_DOCK_MIN_W khi Camera là tab active (xem
        # _sync_cam_dock_check) → không kéo nhỏ hơn khi active, các tab khác
        # vẫn co được. Đồng thời sync menu tick.
        dock.visibilityChanged.connect(self._sync_cam_dock_check)

    def _sync_cam_dock_check(self, visible: bool) -> None:
        """Camera thành active tab → min = _CAM_DOCK_MIN_W (không kéo nhỏ, chỉ
        to ra) + resize vùng về mức đó; sync menu tick."""
        if hasattr(self, "_act_camera_dock"):
            self._set_toggle(self._act_camera_dock, bool(visible))
        if visible and hasattr(self, "_camera_dock"):
            self._apply_active_dock_min(self._camera_dock, _CAM_DOCK_MIN_W)
            QTimer.singleShot(0, lambda: self.resizeDocks(
                [self._camera_dock], [_CAM_DOCK_MIN_W], Qt.Orientation.Horizontal))

    # ── Object classes (do bài toán định nghĩa) ────────────────────────
    def _refresh_class_combo(self) -> None:
        """Nạp lại combo Class từ CellConfig.object_classes (fallback default),
        giữ lựa chọn hiện tại nếu còn."""
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
        """Dialog định nghĩa danh sách class của bài toán → lưu vào CellConfig.

        Bố cục: list bên trái + cột nút bên phải (Thêm/Sửa/Xóa/Lên/Xuống +
        OK/Cancel) — mọi nút cùng độ rộng, thẳng hàng, không khoảng trống."""
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

        # Cột nút bên phải — gồm cả OK/Cancel, cùng độ rộng + khoảng cách ĐỀU.
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
            t.join(timeout=2.5)
        self._cam_thread = None
        self._cam_running = False

    def _camera_loop(self) -> None:
        """Thread nền: get_frame → (detect → pose) → emit camera_result."""
        from ...perception import (  # lazy — tránh import nặng lúc startup
            D455Camera, MockCamera, MockDetector, ObjectDetector, PoseExtractor,
        )
        from ...perception.detector import field_dict      # không export ở __init__
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
                # FPS thực = nghịch đảo khoảng cách giữa 2 frame (gồm cả sleep).
                now = time.time()
                self._last_fps = 1.0 / max(now - prev_t, 1e-6)
                prev_t = now
                # Lưu frame thô (atomic ref-assign) cho capture/control — copy vì
                # buffer RealSense bị tái dụng frame sau. Luôn cập nhật mỗi frame.
                self._last_rgb = np.ascontiguousarray(rgb).copy()
                self._last_depth = np.asarray(depth).copy()
                self._last_camera_objects = objects
                self._last_intrinsics = dict(camera.intrinsics)
                self._last_source = source_label
                # BACKPRESSURE: chỉ dựng ảnh hiển thị NẶNG (depth colormap/overlay)
                # + emit KHI main thread đã tiêu thụ frame trước → vừa tránh nghẽn
                # event loop (Stop luôn ăn) vừa khỏi phí CPU khi UI chậm.
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
        """Dựng camera theo lựa chọn nguồn. Trả (camera, source_label).

        Auto: thử D455, lỗi → fallback Mock. D455: bắt buộc thật (raise nếu lỗi).
        Mock: frame tổng hợp + object giả để demo pipeline khi không có phần cứng.
        color_size/fps chỉ áp cho D455 thật (Mock dùng intrinsics 1280×720).
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
        """YOLO nếu có trọng số models/*.pt|onnx, else MockDetector (1 vật giả)."""
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
        """Frame BGR tổng hợp (gradient + 1 'vật' tại _MOCK_BOX) cho mock view."""
        h, w = 720, 1280
        ramp = np.linspace(35, 80, h).astype(np.uint8)
        frame = np.repeat(ramp[:, None], w, axis=1)
        rgb = np.stack([frame, frame, frame], axis=-1)
        x1, y1, x2, y2 = _MOCK_BOX
        rgb[y1:y2, x1:x2] = (150, 150, 155)             # 'object' xám sáng
        return rgb

    # ── Main-thread slot ───────────────────────────────────────────────
    def _on_camera_result(self, payload) -> None:
        # App đang đóng → KHÔNG chạm widget (có thể đã bị hủy khi worker emit
        # 'stopped' lúc join trong closeEvent).
        if getattr(self, "_cam_closing", False):
            return
        # Slot NHẸ: worker đã dựng sẵn _last_display (RGB) — main chỉ QPixmap +
        # setPixmap. Clear backpressure để worker được đẩy frame kế.
        self._cam_frame_pending = False
        if isinstance(payload, dict) and payload.get("stopped"):
            # Bỏ qua 'stopped' cũ đến trễ (Stop→Start nhanh): nếu đã có thread
            # camera mới đang chạy thì đừng reset nút về trạng thái tắt.
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
        """Vẽ bbox + centroid + nhãn class lên ảnh RGB (in-place trên copy)."""
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
        """Depth (mét) → ảnh màu RGB (JET), normalize theo percentile 5–95%."""
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
        """Lưu khung hiện tại: RGB (.png, BGR) + depth (.npy, mét) → data/raw.

        Checkbox 'Lưu depth' tắt → chỉ lưu RGB."""
        rgb = getattr(self, "_last_rgb", None)
        if rgb is None:
            self._set_status("No camera frame yet — press Start first", level="warn")
            return
        save_depth = self._cam_save_depth_chk.isChecked()
        depth = self._last_depth
        if save_depth and depth is None:
            self._set_status(
                "No depth to save — uncheck 'Save depth' or Start again",
                level="warn")
            return
        import cv2

        from ...utils import ensure_dir, timestamp
        out = ensure_dir(self._project_root / "data" / "raw")
        # Counter chống ghi đè khi chụp >1 ảnh trong cùng 1 giây (timestamp
        # độ phân giải giây). Khớp convention dataset script 01 (_{NNNN}).
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

    # ── Vision-guided control (vòng kín) ───────────────────────────────
    def _camera_teach_grasp(self) -> None:
        """Vật detect mới nhất → camera_to_base → make_grasp_pose → teach target."""
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
        from ..coord_conv import camera_to_base, load_calibration, make_grasp_pose
        calib = self._project_root / "config" / "calibration" / "T_base_camera.npy"
        try:
            t_bc = load_calibration(calib)
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"Calibration error: {e}", level="err"); return
        xyz_base = camera_to_base(np.asarray(pose_cam[:3], dtype=float), t_bc)
        T = make_grasp_pose(xyz_base, float(pose_cam[3]))
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
        """Thêm chuỗi pick (open→approach→grasp→close→retreat) vào job hiện tại."""
        name = getattr(self, "_last_grasp_target", None)
        T = getattr(self, "_last_grasp_T", None)
        if not name or name not in self._targets or T is None:
            self._set_status(
                "No grasp target yet — press 'Detect → Teach grasp' first",
                level="warn")
            return
        approach_mm = float(self._cam_approach_spin.value())
        t_app = np.asarray(T, dtype=float).copy()
        t_app[2, 3] += approach_mm                          # nâng theo +Z base
        app_name = self._teach_target_from_matrix(
            t_app, default_name=f"{name}_APP", prompt=False)
        if not app_name:
            self._set_status(
                "IK approach failed — reduce Approach Z", level="warn")
            return
        seq = [
            Instruction(type="SetGripper", gripper_close=False),    # mở kẹp
            Instruction(type="MoveJ", target_name=app_name),        # tới approach
            Instruction(type="MoveL", target_name=name),            # hạ thẳng tới grasp
            Instruction(type="SetGripper", gripper_close=True),     # đóng kẹp
            Instruction(type="Wait", wait_seconds=0.3),
            Instruction(type="MoveL", target_name=app_name),        # nhấc lên (retreat)
        ]
        self._program.extend(seq)
        self._refresh_program_list()
        self._set_status(
            f"Added pick sequence for '{name}' ({len(seq)} instructions) to job",
            level="ok")

    def _camera_sync_to_cell(self) -> None:
        """Ghi camera D455 thật vào node cell `camera` (RoboDK single-source):
        pose từ hand-eye calibration, intrinsics pixel thật từ frame hiện tại.
        → cây Cell + frustum phản ánh đúng camera vật lý."""
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
        # Extrinsics: pose camera-trong-base lấy từ hand-eye calibration.
        calib = self._project_root / "config" / "calibration" / "T_base_camera.npy"
        try:
            x, y, z, rx, ry, rz = _matrix_to_xyz_rpy_deg(load_calibration(calib))
            pose = PoseConfig(xyz_mm=(x, y, z), rpy_deg=(rx, ry, rz))
            pose_note = "pose from calibration"
        except Exception as e:                              # noqa: BLE001
            existing = getattr(self._cell_config, "camera", None)
            if existing is not None:
                pose = existing.pose; pose_note = "keeping previous pose"
            else:
                pose = PoseConfig(xyz_mm=(700.0, 0.0, 1200.0),
                                  rpy_deg=(180.0, 0.0, 0.0))
                pose_note = f"default pose (not calibrated: {e})"
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
