"""
mixin_experiment.py
───────────────────
ExperimentMixin — Digital Twin cho ROBOT THẬT trong app GP7AppQt.

  • Phase 1 — **Live mirror**: poll joints THẬT từ YRC1000 qua HSE @telemetry_hz,
    vẽ vào viewport @mirror_hz (~2Hz), ghi telemetry CSV. Robot KHÔNG nhận lệnh
    (chỉ đọc) → an toàn. Render 76ms << 500ms (2Hz) nên mượt.
  • Phase 2 — **Run experiment**: chạy pick-place tự động trên robot THẬT qua
    Orchestrator + perception (D455+YOLO thật, hoặc Mock dry-run) — đúng cách
    `scripts/03_run_experiment.py --mode real` ghép, nhưng vòng trial NGẮT ĐƯỢC
    (gọi `run_one_cycle` từng trial trong worker thay vì `run_n_trials` blocking).

Tái dùng nguyên: `MotomanHSEBackend` + `DigitalTwinMirror` + `TelemetryLogger` +
`Orchestrator` + `PerceptionNode`. Worker thread CHỈ emit signal
(`joints_update`/`status`/`gripper`/`sim_reset`/`exp_progress`/`exp_done`) — không
đụng VTK/widget; main thread vẽ (an toàn, đúng mẫu `_play_program_loop`).

Host (GP7AppQt) cung cấp: `_signals`, `_hse_ip`, `_hse_tool_no`, `_pp_max_speed_pct`,
`_model`, `_home_joints`, `_base_xyz`, `_cell_config`, `_project_root`, `_set_status`,
`_open_dock_tab`, `_toggle_gripper` (qua signal gripper), `_on_reset_scene` (sim_reset).
"""
from __future__ import annotations

import threading

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)


class ExperimentMixin:
    """Digital Twin (robot THẬT): live mirror + experiment runner."""

    # ── State (gọi 1 lần trong _build_experiment_dock) ───────────────────
    def _init_experiment_state(self) -> None:
        if hasattr(self, "_exp_stop"):
            return
        self._exp_thread: threading.Thread | None = None
        self._exp_stop = threading.Event()
        self._twin = None                       # DigitalTwinMirror đang chạy
        self._exp_running = False
        self._exp_is_experiment = False         # True = runner (robot chuyển động)
        # Seam test headless: factory(ip, tool_no, max_speed_pct) -> backend.
        # None = dùng MotomanHSEBackend thật.
        self._exp_backend_factory = None
        # Seam test headless: factory(config, queue, choice, n) -> (node, threaded).
        # None = dùng _build_experiment_perception (D455+YOLO / Mock).
        self._exp_perception_factory = None

    # ── Dock UI ──────────────────────────────────────────────────────────
    def _build_experiment_dock(self) -> None:
        self._init_experiment_state()
        dock = QDockWidget("Digital Twin", self)
        dock.setObjectName("ExperimentDock")
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                             | Qt.DockWidgetArea.RightDockWidgetArea)

        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # ── Tần số mirror/telemetry (dùng chung 2 chế độ) ─────────────────
        gb = QGroupBox("Digital Twin — real robot (HSE)")
        form = QFormLayout(gb)
        self._exp_mirror_hz = QDoubleSpinBox()
        self._exp_mirror_hz.setRange(0.5, 10.0)
        self._exp_mirror_hz.setSingleStep(0.5)
        self._exp_mirror_hz.setValue(2.0)
        self._exp_tele_hz = QDoubleSpinBox()
        self._exp_tele_hz.setRange(1.0, 30.0)
        self._exp_tele_hz.setSingleStep(1.0)
        self._exp_tele_hz.setValue(10.0)
        form.addRow("Mirror Hz (viewport):", self._exp_mirror_hz)
        form.addRow("Telemetry Hz (CSV):", self._exp_tele_hz)
        lay.addWidget(gb)

        # ── Tham số experiment (dùng cho nút Start experiment) ────────────
        self._exp_group = QGroupBox("Experiment parameters (autonomous pick-place)")
        eform = QFormLayout(self._exp_group)
        self._exp_trials = QSpinBox()
        self._exp_trials.setRange(1, 1000)
        self._exp_trials.setValue(10)
        eform.addRow("Trials:", self._exp_trials)
        self._exp_ik_cb = QComboBox()
        self._exp_ik_cb.addItems(
            ["yrc (YRC1000 onboard IK)", "client (Pieper analytical)"])
        eform.addRow("IK source:", self._exp_ik_cb)
        self._exp_perc_cb = QComboBox()
        self._exp_perc_cb.addItems(["D455 + YOLO (real)", "Mock (dry-run test)"])
        eform.addRow("Perception:", self._exp_perc_cb)
        self._exp_ultra = QCheckBox("Ultra-fast (P-var, upload template once)")
        eform.addRow(self._exp_ultra)
        lay.addWidget(self._exp_group)

        # ── Action buttons (gộp từ menu vào panel) ────────────────────────
        row1 = QHBoxLayout()
        self._exp_mirror_btn = QPushButton("▶  Start live mirror")
        self._exp_mirror_btn.setToolTip(
            "Read-only: mirror the real robot's joints into the viewport "
            "(no commands sent).")
        self._exp_mirror_btn.clicked.connect(self._on_start_live_mirror)
        self._exp_exp_btn = QPushButton("▶  Start experiment")
        self._exp_exp_btn.setToolTip(
            "Autonomous pick-place on the REAL robot (the robot WILL move).")
        self._exp_exp_btn.clicked.connect(self._on_start_experiment)
        row1.addWidget(self._exp_mirror_btn)
        row1.addWidget(self._exp_exp_btn)
        lay.addLayout(row1)

        self._exp_stop_btn = QPushButton("⏹  Stop Digital Twin")
        self._exp_stop_btn.setEnabled(False)
        self._exp_stop_btn.clicked.connect(self._on_stop_experiment)
        lay.addWidget(self._exp_stop_btn)

        self._exp_status_lbl = QLabel("Idle — real robot not mirrored yet.")
        self._exp_status_lbl.setWordWrap(True)
        lay.addWidget(self._exp_status_lbl)

        hint = QLabel(
            "ⓘ Live mirror = READ real joints @Hz and render — robot receives no "
            "commands.\nRun experiment = the Orchestrator drives the REAL robot to "
            "pick-and-place autonomously (the robot WILL move). Telemetry + results "
            "are written to results/.")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        lay.addStretch(1)

        dock.setWidget(body)
        self._experiment_dock = dock
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        if getattr(self, "_jog_dock", None) is not None:
            self.tabifyDockWidget(self._jog_dock, dock)
        dock.setVisible(False)

    # ── Viewport bridge: mirror thread gọi → marshal về main thread ──────
    def _exp_viewport_callback(self, joints) -> None:
        """DigitalTwinMirror gọi @mirror_hz từ MIRROR THREAD. Chỉ emit signal —
        Qt tự queue về main thread `_apply_joints_main` (vẽ). KHÔNG đụng VTK ở đây."""
        try:
            self._signals.joints_update.emit([float(j) for j in joints])
        except Exception:                                       # noqa: BLE001
            pass

    # ── Viewport-visual hooks: Orchestrator (worker thread) gọi qua twin ──
    # attach/detach/reset_scene → emit signal → main thread gắp/thả/reset vật.
    def _exp_grasp_cb(self, class_name=None) -> None:
        try:
            self._signals.gripper.emit(True)
        except Exception:                                       # noqa: BLE001
            pass

    def _exp_release_cb(self) -> None:
        try:
            self._signals.gripper.emit(False)
        except Exception:                                       # noqa: BLE001
            pass

    def _exp_reset_cb(self) -> None:
        try:
            self._signals.sim_reset.emit()
        except Exception:                                       # noqa: BLE001
            pass

    def _validate_dt_common(self) -> str | None:
        """Kiểm tra điều kiện chung (model + IP). Trả message lỗi hoặc None."""
        if self._exp_running:
            return "Digital Twin is already running"
        if getattr(self, "_model", None) is None:
            return "Load Robot GP7 first (a model is required to build the viewport)"
        if not (getattr(self, "_hse_ip", "") or ""):
            return "No YRC1000 IP — Robot → Connection settings…"
        return None

    # ── Start: live mirror (Phase 1) ─────────────────────────────────────
    def _on_start_live_mirror(self) -> None:
        err = self._validate_dt_common()
        if err:
            if "IP" in err:
                QMessageBox.information(
                    self, "Digital Twin",
                    "No YRC1000 IP.\n\nRobot → Connection settings… to enter the IP "
                    "before mirroring.")
            else:
                self._set_status(err, level="warn")
            return
        ip = self._hse_ip
        r = QMessageBox.question(
            self, "Live mirror — real robot",
            f"<b>Mirror the REAL robot state (READ-only, no commands sent)</b><br><br>"
            f"HSE IP: <code>{ip}</code><br>"
            f"Mirror {self._exp_mirror_hz.value():.1f} Hz · Telemetry "
            f"{self._exp_tele_hz.value():.0f} Hz → CSV<br><br>"
            f"Requires: YRC1000 HSE Server enabled, same network (ping OK).<br>"
            f"Start mirror?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        if r != QMessageBox.StandardButton.Yes:
            return

        self._exp_is_experiment = False
        self._begin_worker(
            target=self._live_mirror_worker,
            args=(ip, float(self._exp_mirror_hz.value()),
                  float(self._exp_tele_hz.value())),
            status=f"Digital Twin: connecting to {ip}…")

    # ── Start: experiment runner (Phase 2) ───────────────────────────────
    def _on_start_experiment(self) -> None:
        err = self._validate_dt_common()
        if err:
            if "IP" in err:
                QMessageBox.information(
                    self, "Digital Twin",
                    "No YRC1000 IP.\n\nRobot → Connection settings… to enter the IP "
                    "before running an experiment.")
            else:
                self._set_status(err, level="warn")
            return

        from pathlib import Path
        root = Path(self._project_root)
        # Orchestrator load calib khi __init__ → bắt buộc có T_base_camera.npy.
        calib = root / "config" / "calibration" / "T_base_camera.npy"
        if not calib.exists():
            QMessageBox.warning(
                self, "Missing calibration",
                f"Hand-eye calibration not found:\n{calib}\n\n"
                "Run the calibration (scripts/02_run_calibration.py) before "
                "running a real experiment.")
            return

        ik_source = "client" if self._exp_ik_cb.currentIndex() == 1 else "yrc"
        perc_choice = "mock" if self._exp_perc_cb.currentIndex() == 1 else "real"
        n = int(self._exp_trials.value())
        ultra = bool(self._exp_ultra.isChecked())

        # Perception thật cần YOLO weights — cảnh báo nếu thiếu.
        if perc_choice == "real":
            cfg = self._assemble_experiment_config_real(ik_source)
            mp = cfg.get("model_path", "models/yolov8s-seg_best.pt")
            wp = Path(mp)
            if not wp.is_absolute():
                wp = root / wp
            if not wp.exists():
                r = QMessageBox.question(
                    self, "Missing YOLO weights",
                    f"Model not found:\n{wp}\n\n"
                    "Use Perception = Mock (dry-run) to test the pipeline without a "
                    "camera/model, or Cancel to adjust.\n\nContinue anyway?",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.Cancel)
                if r != QMessageBox.StandardButton.Yes:
                    return

        ip = self._hse_ip
        warn_mock = ("<br><br><b style='color:#d83b01'>⚠ Mock dry-run:</b> the robot "
                     "still moves to poses computed from FAKE detections — make sure "
                     "the workspace is clear.") if perc_choice == "mock" else ""
        r = QMessageBox.warning(
            self, "Run experiment — the REAL robot WILL MOVE",
            f"<b>The Orchestrator will drive the REAL GP7 to pick-and-place "
            f"autonomously.</b><br><br>"
            f"HSE IP: <code>{ip}</code> · {n} trials · IK={ik_source} · "
            f"perception={perc_choice}{' · ultra-fast' if ultra else ''}<br>"
            f"Predictive safety + reach envelope: ON.{warn_mock}<br><br>"
            f"Ensure: workspace clear of people, E-stop within reach, "
            f"YRC in PLAY/REMOTE mode.<br><br>"
            f"Start experiment?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if r != QMessageBox.StandardButton.Yes:
            return

        self._exp_is_experiment = True
        self._begin_worker(
            target=self._experiment_worker,
            args=(n, ik_source, ultra, perc_choice),
            status=f"Digital Twin experiment: connecting to {ip}…")

    def _begin_worker(self, target, args, status: str) -> None:
        """Chung cho mirror/experiment: reset stop flag, khoá UI, spawn worker."""
        self._exp_stop.clear()
        self._exp_running = True
        self._exp_mirror_btn.setEnabled(False)
        self._exp_exp_btn.setEnabled(False)
        self._exp_group.setEnabled(False)
        self._exp_stop_btn.setEnabled(True)
        self._set_status(status, level="info")
        self._exp_thread = threading.Thread(target=target, args=args, daemon=True)
        self._exp_thread.start()

    def _on_stop_experiment(self) -> None:
        self._exp_stop.set()
        # Experiment: robot đang chuyển động → servo OFF NGAY (mirror chỉ đọc nên
        # không Stop để tránh dừng job robot đang tự chạy).
        if getattr(self, "_exp_is_experiment", False):
            twin = getattr(self, "_twin", None)
            if twin is not None and hasattr(twin, "Stop"):
                try:
                    twin.Stop()
                except Exception:                               # noqa: BLE001
                    pass
        self._set_status("Digital Twin: stopping…", level="warn")

    # ── Worker: live mirror only (Phase 1) ───────────────────────────────
    def _live_mirror_worker(self, ip, mirror_hz, telemetry_hz) -> None:
        backend = None
        twin = None
        try:
            from ..digital_twin import DigitalTwinMirror
            from ..telemetry import TelemetryLogger
            from ...utils import timestamp
            from pathlib import Path

            backend = self._build_backend(ip)   # MotomanHSEBackend (hoặc seam test)
            backend.connect()
            if not backend.Valid():
                self._signals.status.emit(
                    "Digital Twin: HSE not responding — check IP / HSE Server",
                    "err")
                return

            csv_path = (Path(self._project_root) / "results"
                        / f"telemetry_{timestamp()}.csv")
            telemetry = TelemetryLogger(csv_path)
            twin = DigitalTwinMirror(
                backend, viewport_callback=self._exp_viewport_callback,
                telemetry=telemetry, mirror_hz=mirror_hz,
                telemetry_hz=telemetry_hz, viewport_mirror_enabled=True)
            self._twin = twin
            twin.start_mirror()
            self._signals.status.emit(
                f"Digital Twin: mirror ON @{mirror_hz:.1f} Hz → {csv_path.name}",
                "ok")

            # Giữ worker sống; mirror thread (trong twin) tự poll + emit.
            while not self._exp_stop.is_set():
                self._exp_stop.wait(0.2)
        except Exception as e:                                  # noqa: BLE001
            self._signals.status.emit(f"Digital Twin error: {e}", "err")
        finally:
            self._teardown(twin, backend)

    # ── Worker: experiment runner (Phase 2) ──────────────────────────────
    def _experiment_worker(self, n, ik_source, ultra_fast, perc_choice) -> None:
        backend = None
        twin = None
        perception = None
        orch = None
        threaded = False
        try:
            from ..digital_twin import DigitalTwinMirror
            from ..telemetry import TelemetryLogger
            from ..orchestrator import Orchestrator
            from ...logging import TrialLogger
            from ...utils import timestamp
            from pathlib import Path
            import queue as _queue

            ip = getattr(self, "_hse_ip", "") or ""
            mirror_hz = float(self._exp_mirror_hz.value())
            telemetry_hz = float(self._exp_tele_hz.value())

            backend = self._build_backend(ip)
            backend.connect()
            if not backend.Valid():
                self._signals.status.emit(
                    "Digital Twin: HSE not responding — check IP / HSE Server",
                    "err")
                return
            if ultra_fast and hasattr(backend, "enable_ultra_fast"):
                try:
                    backend.enable_ultra_fast(True)
                except Exception:                               # noqa: BLE001
                    pass

            root = Path(self._project_root)
            ts = timestamp()
            telemetry = TelemetryLogger(root / "results" / f"telemetry_{ts}.csv")
            twin = DigitalTwinMirror(
                backend, viewport_callback=self._exp_viewport_callback,
                telemetry=telemetry, mirror_hz=mirror_hz,
                telemetry_hz=telemetry_hz, drift_warn_deg=2.0,
                viewport_mirror_enabled=True,
                grasp_callback=self._exp_grasp_cb,
                release_callback=self._exp_release_cb,
                reset_callback=self._exp_reset_cb)
            self._twin = twin
            twin.start_mirror()

            config = self._assemble_experiment_config_real(ik_source)

            qsize = (n + 1) if perc_choice == "mock" else 3
            det_queue: _queue.Queue = _queue.Queue(maxsize=max(3, qsize))
            pfactory = getattr(self, "_exp_perception_factory", None)
            if pfactory is not None:
                perception, threaded = pfactory(config, det_queue, perc_choice, n)
            else:
                perception, threaded = self._build_experiment_perception(
                    config, det_queue, perc_choice, n)

            trial_logger = TrialLogger(
                root / "results" / f"experiment_real_{ts}.csv",
                extra_context={"mode": "real", "ik": ik_source})
            orch = Orchestrator(det_queue, config=config, robot=twin,
                                logger_obj=trial_logger)

            self._signals.status.emit(
                f"Digital Twin experiment: {n} trials — IK={ik_source}, "
                f"perception={perc_choice} → {trial_logger.csv_path.name}", "ok")

            if threaded:
                perception.start()
            delay = float(config.get("inter_trial_delay_s", 1.0))
            for i in range(n):
                if self._exp_stop.is_set():
                    break
                if not threaded:
                    msg = perception.process_once()
                    if msg is not None:
                        try:
                            det_queue.put(msg, timeout=1.0)
                        except Exception:                       # noqa: BLE001
                            pass
                try:
                    orch.run_one_cycle(i + 1)
                except Exception as e:                          # noqa: BLE001
                    self._signals.status.emit(f"Trial {i + 1} error: {e}", "warn")
                self._signals.exp_progress.emit(i + 1, dict(orch.stats))
                self._exp_stop.wait(delay)
        except Exception as e:                                  # noqa: BLE001
            self._signals.status.emit(f"Digital Twin experiment error: {e}", "err")
        finally:
            try:
                if perception is not None and threaded:
                    perception.stop()
            except Exception:                                   # noqa: BLE001
                pass
            # Servo OFF sau khi xong / dừng (robot đã chuyển động).
            try:
                if twin is not None and hasattr(twin, "Stop"):
                    twin.Stop()
            except Exception:                                   # noqa: BLE001
                pass
            self._teardown(twin, backend,
                           stats=dict(orch.stats) if orch is not None else {})

    # ── Helpers chung ────────────────────────────────────────────────────
    def _build_backend(self, ip):
        """Tạo motion backend (HSE thật, hoặc fake qua seam test)."""
        factory = getattr(self, "_exp_backend_factory", None)
        if factory is not None:
            backend = factory(ip, getattr(self, "_hse_tool_no", 1),
                              getattr(self, "_pp_max_speed_pct", 20.0))
        else:
            from ..backends.motoman_hse import MotomanHSEBackend
            backend = MotomanHSEBackend(
                ip=ip, tool_no=getattr(self, "_hse_tool_no", 1),
                max_speed_pct=getattr(self, "_pp_max_speed_pct", 20.0))
        if hasattr(backend, "set_home_joints"):
            backend.set_home_joints(list(getattr(self, "_home_joints", [0.0] * 6)))
        return backend

    def _teardown(self, twin, backend, stats=None) -> None:
        """Stop mirror + disconnect + reset state + emit exp_done (main thread)."""
        try:
            if twin is not None:
                twin.stop_mirror()
        except Exception:                                       # noqa: BLE001
            pass
        try:
            if backend is not None and hasattr(backend, "disconnect"):
                backend.disconnect()
        except Exception:                                       # noqa: BLE001
            pass
        self._twin = None
        self._exp_running = False
        self._signals.exp_done.emit(stats if stats is not None else {})

    def _assemble_experiment_config_real(self, ik_source: str = "yrc") -> dict:
        """Config cho experiment ROBOT THẬT (khớp `03_run_experiment.py --mode real`).

        Nền experiment.yaml (model_path/conf/place…) + override real-mode flags +
        base/home/tool từ cell config. Trả dict OVERRIDE; Orchestrator tự merge
        với `_DEFAULT_CONFIG`.
        """
        from pathlib import Path
        cfg: dict = {}
        try:
            from ...utils import load_yaml
            exp_yaml = Path(self._project_root) / "config" / "experiment.yaml"
            if exp_yaml.exists():
                loaded = load_yaml(exp_yaml)
                if isinstance(loaded, dict):
                    cfg = dict(loaded)
        except Exception:                                       # noqa: BLE001
            cfg = {}

        # Real-mode safety flags (giống 03 --mode real).
        cfg["is_real_mode"] = True
        cfg["skip_reachability_check"] = False
        cfg["predictive_safety_enabled"] = True
        if not cfg.get("gripper_cc_link"):
            cfg["gripper_cc_link"] = {
                "clamp_bit":              30010,
                "unclamp_bit":            30011,
                "clamp_sensor_bit":       30050,
                "unclamp_sensor_bit":     30051,
                "detect_bit":             30052,
                "wait_sensor_timeout_s":  2.0,
                "wait_sensor_poll_s":     0.05,
                "require_detect_on_close": True,
            }

        # base / home / tool — ưu tiên cell config, fallback attr app.
        cc = getattr(self, "_cell_config", None)
        if cc is not None:
            try:
                cfg["home_joints_deg"] = list(cc.robot.home_joints_deg)
                cfg["robot_base_xyz_mm"] = tuple(cc.robot.pose.xyz_mm)
                cfg["robot_base_rpy_deg"] = tuple(cc.robot.pose.rpy_deg)
                tool_offset = 0.0
                if getattr(cc, "tool", None):
                    tcp = getattr(cc.tool, "tcp_offset_mm", None)
                    if tcp:
                        tool_offset = float(tcp[2])
                cfg["robot_tool_offset_mm"] = tool_offset
            except Exception:                                   # noqa: BLE001
                pass
        if "home_joints_deg" not in cfg:
            cfg["home_joints_deg"] = list(getattr(self, "_home_joints", [0.0] * 6))
        if "robot_base_xyz_mm" not in cfg:
            cfg["robot_base_xyz_mm"] = tuple(getattr(self, "_base_xyz",
                                                     (0.0, 0.0, 0.0)))

        # IK source (mặc định yrc cho HSE real).
        if ik_source == "client":
            cfg["use_client_ik"] = True
        else:
            cfg["use_yrc_ik"] = True
        return cfg

    def _build_experiment_perception(self, config, det_queue, choice, n):
        """Tạo PerceptionNode. Trả (node, threaded).

        • choice == "real": D455 + YOLO → threaded=True (perception.start()).
        • choice == "mock": MockCamera + MockDetector (1 kịch bản lặp vòng) →
          threaded=False (worker gọi process_once() từng trial, deterministic).
        """
        from ...perception import PerceptionNode
        if choice == "mock":
            import numpy as np
            from ...perception import MockCamera, MockDetector
            cc = getattr(self, "_cell_config", None)
            target = "tray"
            if cc is not None and getattr(cc, "objects", None):
                target = cc.objects[0].name
            depth_m = float(config.get("sim_depth_m", 0.49))
            camera = MockCamera(
                depth_frames=[np.full((720, 1280), depth_m, np.float32)])
            det = MockDetector.make_detection(target, mask_box=(580, 350, 700, 390))
            detector = MockDetector(scripted=[[det]])
            return PerceptionNode(camera, detector, det_queue), False

        from ...perception import D455Camera, ObjectDetector
        camera = D455Camera()
        detector = ObjectDetector(
            model_path=config.get("model_path", "models/yolov8s-seg_best.pt"),
            conf=config.get("conf_threshold", 0.5))
        return PerceptionNode(camera, detector, det_queue), True

    # ── Main-thread slots cho exp signals ────────────────────────────────
    def _on_exp_progress(self, completed: int, stats) -> None:
        """Cập nhật số trial + thống kê live (chạy trên main thread)."""
        if hasattr(self, "_exp_status_lbl"):
            try:
                s = stats or {}
                att = max(int(s.get("attempted", 0)), 0)
                ok = int(s.get("successful", 0))
                rate = (100.0 * ok / att) if att else 0.0
                self._exp_status_lbl.setText(
                    f"Trial {completed} — OK {ok} / Fail {s.get('failed', 0)} "
                    f"({rate:.0f}% of {att} attempts)")
            except Exception:                                   # noqa: BLE001
                pass

    def _on_exp_done(self, stats) -> None:
        """Mirror/experiment kết thúc → re-enable UI (chạy trên main thread)."""
        self._exp_running = False
        self._exp_is_experiment = False
        if hasattr(self, "_exp_mirror_btn"):
            self._exp_mirror_btn.setEnabled(True)
        if hasattr(self, "_exp_exp_btn"):
            self._exp_exp_btn.setEnabled(True)
        if hasattr(self, "_exp_group"):
            self._exp_group.setEnabled(True)
        if hasattr(self, "_exp_stop_btn"):
            self._exp_stop_btn.setEnabled(False)
        if hasattr(self, "_exp_status_lbl"):
            s = stats or {}
            if s.get("attempted"):
                att = int(s["attempted"])
                ok = int(s.get("successful", 0))
                rate = (100.0 * ok / att) if att else 0.0
                self._exp_status_lbl.setText(
                    f"Done — OK {ok}/{att} ({rate:.0f}%), "
                    f"Fail {s.get('failed', 0)}.")
            else:
                self._exp_status_lbl.setText("Idle — stopped.")
