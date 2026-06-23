"""
mixin_connection.py
───────────────────
ConnectionMixin: HSE connection settings + test-connection + Run on Robot
worker pipeline + emergency stop.

Mixin pattern — not instantiated standalone. Host class (GP7AppQt) must provide:
  attributes: _hse_ip, _hse_tool_no, _hse_ftp_user, _hse_ftp_pass, _hse_ftp_dir,
              _hse_thread, _hse_stop, _pp_max_speed_pct, _program, _active_job,
              _signals
  methods:    _set_status, _safe_job_name, _export_job_to_path, _on_prog_stop
"""
from __future__ import annotations

import threading
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit,
    QMessageBox, QSpinBox,
)

from ..backends.motoman_hse import JobUploadError, MotomanHSEBackend


class ConnectionMixin:
    """HSE connection + Run on Robot lifecycle (settings, test, upload, stop)."""

    # Phase-2 live jog: max per-send joint move (deg). A jog increment is small;
    # a bigger jump = the sim teleported (paste / Home) while live jog was ON →
    # blocked so the real robot never makes a surprise fast move. See _live_jog_worker.
    _LIVE_JOG_MAX_STEP_DEG = 30.0

    def _on_show_connection_settings(self) -> None:
        """Dialog edit HSE connection — IP, tool_no, FTP creds.

        Has a **Test** button inline in the dialog (replaces the old separate
        "Test connection" menu entry — no need to Apply then go back to the menu;
        clicking Test pings immediately with the values currently in the form).
        """
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
            "<small><i>⚠ Robot must be in REMOTE mode + HSE Server function enabled."
            "<br>TP speed slider should be ≤ 10% on first run.</i></small>")
        info.setWordWrap(True); form.addRow(info)
        # Buttonbox: add "Test" alongside OK/Cancel. Clicking Test pings with
        # the values currently in the form (not the previously saved self._hse_*).
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_test = bb.addButton("Test", QDialogButtonBox.ButtonRole.ActionRole)
        btn_test.setToolTip("Ping HSE with the values in this form (without saving)")
        btn_test.clicked.connect(lambda: self._test_hse_connection(
            ip=ed_ip.text().strip(),
            tool_no=int(sp_tool.value()),
            ftp_user=ed_user.text(),
            ftp_pass=ed_pass.text(),
            ftp_dir=ed_dir.text().strip() or "/JOB",
        ))
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted: return
        self._hse_ip = ed_ip.text().strip()
        self._hse_tool_no = int(sp_tool.value())
        self._hse_ftp_user = ed_user.text()
        self._hse_ftp_pass = ed_pass.text()
        self._hse_ftp_dir = ed_dir.text().strip() or "/JOB"
        self._set_status(
            f"Connection: {self._hse_ip} TL#{self._hse_tool_no}", level="ok")

    def _test_hse_connection(
        self, ip: str, tool_no: int,
        ftp_user: str = "", ftp_pass: str = "", ftp_dir: str = "/JOB",
    ) -> None:
        """Ping HSE using the provided connection params (does not touch self._hse_*).

        Called from:
          • Connection settings dialog → Test button (values currently in the form)
          • `_on_test_connection` (legacy wrapper using saved self._hse_*)
        """
        if not ip:
            self._set_status(
                "HSE IP not configured — Robot → Connection settings", level="warn")
            return
        self._set_status(f"Pinging {ip}…", level="info")
        QApplication.processEvents()
        backend = MotomanHSEBackend(
            ip=ip, timeout_s=2.0,
            ftp_user=ftp_user, ftp_pass=ftp_pass,
            ftp_job_dir=ftp_dir, tool_no=tool_no)
        try:
            backend.connect()
            ok = backend.Valid()
            if ok:
                # Read joints + alarm to verify deeper
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
                    "Connection FAIL — check HSE Server is enabled", level="err")
                QMessageBox.warning(
                    self, "Connection failed",
                    f"YRC1000 {ip} did not respond to READ_STATUS.\n"
                    "Verify:\n"
                    f" • Ping {ip} OK?\n"
                    " • HSE Server function enabled in Maintenance mode?\n"
                    " • PC on the same subnet as the YRC1000?")
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"Connection error: {e}", level="err")
            QMessageBox.critical(self, "Connection error", str(e))
        finally:
            backend.disconnect()

    def _on_test_connection(self) -> None:
        """Legacy wrapper — ping HSE using the saved self._hse_* values.

        The separate "Test connection" menu entry has been removed (merged into
        the dialog); this method is kept for compatibility if anything binds a
        shortcut or script call to it.
        """
        self._test_hse_connection(
            ip=self._hse_ip, tool_no=self._hse_tool_no,
            ftp_user=self._hse_ftp_user, ftp_pass=self._hse_ftp_pass,
            ftp_dir=self._hse_ftp_dir)

    def _resolve_job_key(self, call_name: str) -> "str | None":
        """Map a CALL JOB literal to a project job key. Mirrors playback's
        `_exec_call_job`: try the raw literal, then the sanitized form — the
        importer keys sub-jobs by the sanitized name (e.g. `CALL JOB:SPEED-1`
        is stored under key `SPEED1`). Returns None if no job matches."""
        for cand in (call_name, self._safe_job_name(call_name)):
            if cand and cand in self._jobs:
                return cand
        return None

    def _collect_run_jobs(self) -> tuple[list[str], list[str]]:
        """BFS from the active job over CALL JOB references.

        Returns (order, missing):
          order   = job KEYS to upload — active job first, then every reachable
                    sub-job (deduped, any depth).
          missing = CALL JOB literals referenced but absent from the project (they
                    must already live on the controller, else it alarms at runtime).
        CALL targets are resolved to a key via `_resolve_job_key` (raw → sanitized,
        same as playback) so dashed sub-jobs (SPEED-1 → key SPEED1) are found.
        """
        order: list[str] = []
        seen: set[str] = set()
        missing: list[str] = []
        queue: list[str] = [self._active_job]
        while queue:
            key = queue.pop(0)
            if key in seen:
                continue
            prog = self._jobs.get(key)
            if prog is None:
                continue                        # active job is always a valid key
            seen.add(key)
            order.append(key)
            for ins in prog:
                if ins.type == "CallJob" and ins.job_name:
                    sub = self._resolve_job_key(ins.job_name)
                    if sub is None:
                        if ins.job_name not in missing:
                            missing.append(ins.job_name)
                    elif sub not in seen:
                        queue.append(sub)
        return order, missing

    def _robot_job_name(self, key: str) -> str:
        """Controller job name for a project job KEY. Prefer the original .JBI
        //NAME (keeps dashes, e.g. SPEED-1) so `CALL JOB:<name>` resolves; fall
        back to the sanitized key. Mirrors `_export_all_jobs` naming."""
        raw = getattr(self, "_jbi_raw", {}).get(key)
        orig = (raw or {}).get("name", "")
        return (orig or self._safe_job_name(key))[:32]

    def _on_run_on_robot(self) -> None:
        """Render active job + all CALL JOB sub-jobs, then run in TWO key-switch
        phases (this YRC1000 rejects JOB_SELECT in REMOTE): key in PLAY/TEACH →
        upload + JOB_SELECT (no motion); key in REMOTE → servo ON + START +
        wait_idle. The worker picks the phase from the controller's reported mode.

        Runs in a worker thread so the UI stays responsive. Stop button → servo OFF.
        """
        if not self._hse_ip:
            r = QMessageBox.question(
                self, "Run on Robot",
                "HSE IP not configured. Open Connection settings now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r == QMessageBox.StandardButton.Yes:
                self._on_show_connection_settings()
            return
        if not self._program:
            self._set_status("Current job empty", level="warn"); return
        if self._hse_thread is not None and self._hse_thread.is_alive():
            self._set_status("Robot is running a job — wait for it to finish or Stop",
                              level="warn"); return
        if (getattr(self, "_live_jog_thread", None) is not None
                and self._live_jog_thread.is_alive()):
            self._set_status("Live jog is ON — turn it off before running a job",
                              level="warn"); return
        if getattr(self, "_exp_running", False):
            self._set_status("Digital Twin experiment is running — stop it first",
                              level="warn"); return
        # Collect active job + ALL sub-jobs reachable via CALL JOB (any depth).
        order, missing = self._collect_run_jobs()
        if missing:
            r = QMessageBox.warning(
                self, "Run on Robot — missing sub-jobs",
                "These <code>CALL JOB</code> targets are NOT in the project and "
                "will not be uploaded:<br><br>&nbsp;• "
                + "<br>&nbsp;• ".join(f"<code>{m}</code>" for m in missing)
                + "<br><br>They must already exist on the controller, otherwise "
                "the job will alarm at runtime. Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel)
            if r != QMessageBox.StandardButton.Yes:
                return
        # Safety confirm
        n_steps = len(self._program)
        sub_note = (f"<br>Sub-jobs uploaded: {len(order) - 1} "
                    f"(<code>{', '.join(order[1:])}</code>)"
                    if len(order) > 1 else "")
        r = QMessageBox.warning(
            self, "Run on Robot — Safety check",
            f"<b>⚠ Run on Robot — 2 steps (robot MOVES in step 2)</b><br><br>"
            f"This controller can't select a job in REMOTE, so it runs in two "
            f"key-switch steps:<br>"
            f"&nbsp;1) Key in <b>PLAY</b> → upload &amp; SELECT the job (no motion)<br>"
            f"&nbsp;2) Key in <b>REMOTE</b> → START it (⚠ robot moves)<br>"
            f"The app detects the key position and does the matching step.<br><br>"
            f"Job: <code>{self._active_job}</code> ({n_steps} instructions)"
            f"{sub_note}<br>"
            f"HSE IP: <code>{self._hse_ip}</code><br>"
            f"Max VJ: {self._pp_max_speed_pct:.0f}%<br><br>"
            f"Before STEP 2 (REMOTE), verify:<br>"
            f"&nbsp;✓ TP speed slider ≤ 10%<br>"
            f"&nbsp;✓ Workspace clear, hand ready on E-stop<br>"
            f"&nbsp;✓ No active alarm<br><br>"
            f"Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if r != QMessageBox.StandardButton.Yes: return
        # Render every job to .JBI text in the main thread (atomic _targets access),
        # then hand (name, text) pairs to the worker. jobs[0] = the job to
        # JOB_SELECT + START; the rest are sub-jobs uploaded so CALL JOB resolves.
        try:
            jobs_to_upload: list[tuple[str, str]] = []
            for key in order:
                rname = self._robot_job_name(key) or "PROG"
                jbi_path = Path.cwd() / f"{rname}.JBI"
                self._export_job_to_path(
                    self._jobs[key], rname, jbi_path,
                    folder_name=self._job_folders.get(key, ""), raw_key=key)
                jobs_to_upload.append((rname, jbi_path.read_text(encoding="utf-8")))
                jbi_path.unlink(missing_ok=True)            # tmp file
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"JBI render fail: {e}", level="err")
            return
        main_name = jobs_to_upload[0][0]
        # Worker thread runs upload(all) + JOB_SELECT + START + wait_idle
        self._hse_stop.clear()
        self._hse_thread = threading.Thread(
            target=self._run_on_robot_worker,
            args=(jobs_to_upload, main_name),
            daemon=True)
        self._hse_thread.start()
        self._set_status(
            f"Robot: uploading {len(jobs_to_upload)} job(s)…", level="info")

    def _run_on_robot_worker(self, jobs: "list[tuple[str, str]]",
                             main_name: str) -> None:
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
                    f"Robot: ALARM 0x{code:04X} (sub 0x{sub:04X}) — reset on TP first",
                    "err"); return
            import time as _time
            # This YRC1000 rejects JOB_SELECT (0x87) in REMOTE ("incorrect mode"
            # 0x2080) — selecting a job is a PLAY/TEACH operation — while START
            # (0x86) needs REMOTE. The mode key switch is manual, so Run-on-Robot
            # runs in TWO key-switch phases, chosen from the controller's mode:
            #   key PLAY/TEACH → upload every .JBI + JOB_SELECT the main job (no motion)
            #   key REMOTE     → servo ON + START the already-selected job (⚠ moves)
            try:
                mode = backend.read_controller_status()["mode"]
            except Exception:                            # noqa: BLE001
                mode = "?"                               # default → PHASE 1 (no motion)

            if mode != "REMOTE":
                # ── PHASE 1 (key PLAY/TEACH): upload all + select main. No motion. ──
                for jname, jtext in jobs:
                    if self._hse_stop.is_set():
                        self._signals.status.emit(
                            "Robot: aborted before upload", "warn"); return
                    self._signals.status.emit(
                        f"Robot: FTP uploading '{jname}.JBI'…", "info")
                    try:
                        backend.upload_job(jtext, jname)
                    except JobUploadError as e:
                        # Can't overwrite (active/protected job) → ask to rename.
                        self._signals.run_overwrite_blocked.emit(
                            getattr(e, "job_name", jname) or jname, str(e))
                        return
                self._signals.status.emit(f"Robot: JOB_SELECT '{main_name}'…", "info")
                backend.job_select(main_name)
                self._signals.status.emit(
                    f"Robot: '{main_name}' uploaded & SELECTED (key={mode}). Now turn "
                    f"the key switch to REMOTE and click Run on Robot again to START.",
                    "ok")
                return

            # ── PHASE 2 (key REMOTE): START the job selected in the PLAY phase. ──
            # Do NOT re-upload (FTP 503 on the currently-selected job) or re-select
            # (JOB_SELECT is rejected in REMOTE) — the PLAY phase already did both.
            self._signals.status.emit("Robot: servo ON…", "info")
            try:
                backend.servo_on()
                _time.sleep(1.0)                     # let servos engage before START
            except Exception as e:                   # noqa: BLE001
                self._signals.status.emit(
                    f"Robot: servo-on failed ({e}) — check the TP", "warn")
            # Stop requested during the servo-engage window → abort BEFORE START
            # (otherwise the robot briefly starts moving before the poll loop stops it).
            if self._hse_stop.is_set():
                try: backend.Stop()
                except Exception: pass               # noqa: BLE001
                self._signals.status.emit("Robot: aborted before start", "warn"); return
            self._signals.status.emit(
                f"Robot: START '{main_name}' (job selected in the PLAY step)…", "info")
            backend.job_start()
            # Poll status until idle or stop. On ANY abnormal exit (stop / poll
            # error / timeout) servo-OFF first so a still-moving robot is halted.
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
                    try: backend.Stop()                     # halt — we lost status
                    except Exception: pass                  # noqa: BLE001
                    self._signals.status.emit(
                        f"Robot: status poll error: {e} — servo OFF", "err"); return
                if not running: break
                if _time.monotonic() - t_start > timeout:
                    try: backend.Stop()                     # halt on timeout
                    except Exception: pass                  # noqa: BLE001
                    self._signals.status.emit(
                        f"Robot: timeout {timeout:.0f}s — servo OFF, check TP", "err")
                    return
                _time.sleep(poll_dt)
            # Done — alarm post-check
            code, sub = backend.read_alarm()
            if code != 0:
                self._signals.status.emit(
                    f"Robot: completed WITH ALARM 0x{code:04X}", "warn")
            else:
                self._signals.status.emit(
                    f"Robot: job '{main_name}' completed OK", "ok")
        except Exception as e:                              # noqa: BLE001
            self._signals.status.emit(f"Robot error: {e}", "err")
        finally:
            try:
                backend.disconnect()
            except Exception:                               # noqa: BLE001
                pass

    def _robot_motion_active(self) -> bool:
        """True if the real robot is being commanded by Run-on-Robot, Phase-2 live
        jog, OR a Digital Twin experiment/mirror — used to prevent overlapping
        motion from two sources."""
        run_active = (self._hse_thread is not None and self._hse_thread.is_alive())
        jog_active = (getattr(self, "_live_jog_thread", None) is not None
                      and self._live_jog_thread.is_alive())
        send_busy = bool(getattr(self, "_send_pose_busy", False))
        return bool(run_active or jog_active or send_busy
                    or getattr(self, "_exp_running", False))

    def _on_teach_from_robot(self) -> None:
        """Read the REAL robot's CURRENT joints via HSE (one-shot, READ-ONLY — the
        robot does NOT move) into the app, then re-teach the SELECTED target with
        them. Lets you teach points from the physical robot WITHOUT the live mirror
        (which can't run at the same time as Run on Robot). Jog the robot on the
        pendant first, then click."""
        if not self._hse_ip:
            r = QMessageBox.question(
                self, "Teach from robot",
                "HSE IP not configured. Open Connection settings now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r == QMessageBox.StandardButton.Yes:
                self._on_show_connection_settings()
            return
        if getattr(self, "_model", None) is None:
            self._set_status("Teach: load the GP7 robot first", level="warn"); return
        if self._robot_motion_active():
            self._set_status(
                "Teach: robot busy (Run on Robot / live mirror / jog) — stop it first",
                level="warn"); return
        # One-shot HSE read — READ_POSITION only, no motion command is sent.
        self._set_status("Teach: reading real robot joints via HSE…", level="info")
        backend = MotomanHSEBackend(
            ip=self._hse_ip, timeout_s=2.0, tool_no=self._hse_tool_no)
        try:
            backend.connect()
            joints = backend.Joints()                       # degrees
        except Exception as e:                              # noqa: BLE001
            self._set_status(
                f"Teach: HSE read failed ({e}) — check the connection", level="err")
            return
        finally:
            try:
                backend.disconnect()
            except Exception:                               # noqa: BLE001
                pass
        if not joints or len(joints) < 6:
            self._set_status("Teach: bad joint data from the robot", level="err"); return
        # Load the real pose into the app (renders it), then capture it into a target:
        #   a NAME typed in the target field → teach a NEW target,
        #   else a target selected in the list → re-teach that one.
        self._apply_joints_main([float(q) for q in joints])
        name_edit = getattr(self, "_tgt_name_edit", None)
        name = name_edit.text().strip() if name_edit is not None else ""
        tgt_list = getattr(self, "_tgt_list", None)
        if name:
            self._on_tgt_teach()                            # NEW target from the typed name
        elif (tgt_list is not None and tgt_list.currentRow() >= 0
                and getattr(self, "_targets", None)):
            self._on_tgt_modify()                           # re-teach the SELECTED target
        else:
            self._set_status(
                "Teach: real pose loaded. Type a NAME then click again (new point), "
                "or select a target then click again (re-teach it).", level="info")

    def _on_send_pose_to_robot(self) -> None:
        """Phase-1 direct control (RoboDK-style, discrete): send the app's CURRENT
        joints to the REAL robot via HSE MOVE (0x8B, no job upload). One move per
        click. Robot MUST be in REMOTE mode. ⚠ REAL MOTION."""
        if not self._hse_ip:
            self._set_status("HSE IP not configured — Robot → Connection settings",
                             level="warn"); return
        if getattr(self, "_model", None) is None:
            self._set_status("Robot not loaded", level="warn"); return
        if self._robot_motion_active():
            self._set_status(
                "Robot already busy (Run on Robot / Digital Twin) — stop it first",
                level="warn"); return
        joints = [round(float(q), 3) for q in self._joints]
        speed_pct = min(float(getattr(self, "_pp_max_speed_pct", 10.0)), 10.0)
        # Target in YRC pendant coordinates (BASE frame + TOOL convention) so the
        # operator can cross-check 1:1 with the teach pendant.
        pend = self._pendant_pose() if hasattr(self, "_pendant_pose") else None
        pend_str = ""
        if pend is not None:
            x, y, z, rx, ry, rz = pend
            pend_str = (
                f"YRC pendant frame (should match the TP):<br>"
                f"<code>X {x:.1f}  Y {y:.1f}  Z {z:.1f}  "
                f"Rx {rx:.1f}  Ry {ry:.1f}  Rz {rz:.1f}</code><br><br>")
        r = QMessageBox.warning(
            self, "Send pose to REAL robot",
            f"<b>⚠ THE ROBOT WILL MOVE FOR REAL</b><br><br>"
            f"Target joints (deg):<br><code>{joints}</code><br><br>"
            f"{pend_str}"
            f"Speed: {speed_pct:.0f}%  ·  HSE: <code>{self._hse_ip}</code><br><br>"
            f"Verify: YRC1000 in <b>REMOTE</b> mode, workspace clear, "
            f"hand on the <b>E-stop</b>.<br><br>Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if r != QMessageBox.StandardButton.Yes:
            return
        # Run on a WORKER thread so the GUI stays responsive (no event-loop hang /
        # frozen Stop during servo-engage + move). _send_pose_busy marks the robot
        # busy in _robot_motion_active(); _send_pose_stop lets Stop-all abort it.
        self._send_pose_busy = True
        self._send_pose_stop.clear()
        # Track the handle so closeEvent / Stop-all can JOIN it — otherwise the
        # daemon worker is killed mid-move at exit and its servo-OFF never runs.
        self._send_pose_thread = threading.Thread(
            target=self._send_pose_worker, args=(joints, speed_pct), daemon=True)
        self._send_pose_thread.start()
        self._set_status("Robot: sending pose…", level="info")

    def _send_pose_worker(self, joints: "list[float]", speed_pct: float) -> None:
        """Worker: connect → alarm/running pre-checks → servo on → MOVE → disconnect.
        Status via signals (main thread). ⚠ REAL MOTION."""
        import time as _t
        backend = MotomanHSEBackend(
            ip=self._hse_ip, timeout_s=3.0, tool_no=self._hse_tool_no)
        try:
            backend.connect()
            if not backend.Valid():
                self._signals.status.emit("Robot: HSE not responding", "err"); return
            code, _sub = backend.read_alarm()
            if code != 0:
                self._signals.status.emit(
                    f"Robot: ALARM 0x{code:04X} — reset on TP first", "err"); return
            try:
                if backend.read_status_running():
                    self._signals.status.emit(
                        "Robot is RUNNING a job — stop it on the TP first", "warn")
                    return
            except Exception:                            # noqa: BLE001
                pass
            if self._send_pose_stop.is_set():
                self._signals.status.emit("Send pose: aborted", "warn"); return
            self._signals.status.emit("Robot: servo ON…", "info")
            try:
                backend.servo_on()
            except Exception as e:                       # noqa: BLE001
                self._signals.status.emit(
                    f"servo-on failed ({e}) — check REMOTE mode", "warn")
            for _ in range(10):                          # ~1s servo engage, interruptible
                if self._send_pose_stop.is_set():
                    break
                _t.sleep(0.1)
            if self._send_pose_stop.is_set():
                try: backend.Stop()
                except Exception: pass                   # noqa: BLE001
                self._signals.status.emit("Send pose: aborted — servo OFF", "warn")
                return
            backend.move_joints(joints, speed_pct=speed_pct)
            self._signals.status.emit(
                f"Robot: moving to current pose @ {speed_pct:.0f}%", "ok")
            # The direct MOVE (0x8B) is fire-and-forget (no Running bit), so STAY
            # ALIVE until the robot actually reaches the target — otherwise close /
            # Stop-all joins this thread instantly while the robot is still moving
            # with servos ON. Poll Joints() until within tolerance, the move stalls,
            # or a stop is requested (→ servo OFF). Bounded so it can't hang.
            t_end = _t.monotonic() + 30.0
            read_fails = 0
            while _t.monotonic() < t_end:
                if self._send_pose_stop.is_set():
                    break
                try:
                    cur = backend.Joints()
                    read_fails = 0
                except Exception:                        # noqa: BLE001
                    # A single dropped UDP reply must NOT abandon monitoring (and the
                    # abort path). Tolerate a few; if we PERSISTENTLY can't read the
                    # robot, stop it defensively rather than leave an unmonitored
                    # in-flight move the operator can no longer abort.
                    read_fails += 1
                    if read_fails >= 5:
                        try: backend.Stop()
                        except Exception: pass            # noqa: BLE001
                        self._signals.status.emit(
                            "Send pose: lost monitoring — servo OFF (safety)", "warn")
                        return
                    _t.sleep(0.1)
                    continue
                if max(abs(a - b) for a, b in zip(cur, joints)) < 0.5:
                    break                                # arrived (≤0.5° all axes)
                _t.sleep(0.1)
        except Exception as e:                           # noqa: BLE001
            try: backend.Stop()
            except Exception: pass                       # noqa: BLE001
            self._signals.status.emit(f"Send-pose error: {e}", "err")
        finally:
            # If a stop/close was requested at any point (incl. mid-motion above),
            # servo OFF to halt the in-flight move — the join in closeEvent / Stop-all
            # now actually covers the motion because we waited for arrival above.
            if self._send_pose_stop.is_set():
                try: backend.Stop()
                except Exception: pass                   # noqa: BLE001
                self._signals.status.emit("Send pose: stopped — servo OFF", "warn")
            self._send_pose_busy = False
            try:
                backend.disconnect()
            except Exception:                            # noqa: BLE001
                pass

    # ══════════════════════════════════════════════════════════════════
    # Phase-2 — streaming live jog → REAL robot (RoboDK-style online jog)
    # ══════════════════════════════════════════════════════════════════
    def _on_toggle_live_jog(self, checked: bool) -> None:
        """Toggle Phase-2 live jog. While ON, every user jog streams to the REAL
        robot via HSE MOVE at ~8 Hz (latest-value-wins). ⚠ continuous REAL motion.
        Robot MUST be in REMOTE mode."""
        if not checked:
            if (self._live_jog_thread is not None
                    and self._live_jog_thread.is_alive()):
                self._live_jog_stop.set()
                self._set_status("Robot: live jog stopping (servo OFF)…", level="warn")
            return
        # ── Enabling: guard then confirm ──
        def _refuse(msg: str) -> None:
            self._set_status(msg, level="warn")
            self._act_live_jog.blockSignals(True)
            self._act_live_jog.setChecked(False)
            self._act_live_jog.blockSignals(False)

        if not self._hse_ip:
            return _refuse("HSE IP not configured — Robot → Connection settings")
        if getattr(self, "_model", None) is None:
            return _refuse("Robot not loaded")
        if self._robot_motion_active():
            return _refuse("Robot busy (Run / Mirror / experiment) — stop it first")
        if (self._live_jog_thread is not None
                and self._live_jog_thread.is_alive()):
            return
        speed_pct = min(float(getattr(self, "_pp_max_speed_pct", 10.0)), 10.0)
        r = QMessageBox.warning(
            self, "Live jog on REAL robot",
            f"<b>⚠ THE ROBOT WILL MOVE FOR REAL, CONTINUOUSLY</b><br><br>"
            f"While ON, every jog (Cartesian / dial / joint sliders) streams to the "
            f"real robot via HSE at ~8 Hz, speed <b>{speed_pct:.0f}%</b>.<br><br>"
            f"HSE: <code>{self._hse_ip}</code><br>"
            f"Require: YRC1000 in <b>REMOTE</b>, workspace clear, hand on the "
            f"<b>E-stop</b>.<br><br>"
            f"⚠ Streaming is <b>not yet hardware-verified</b> — start with small, "
            f"slow jogs. Turn the toggle OFF (or Stop) to servo-off.<br><br>"
            f"Enable live jog?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if r != QMessageBox.StandardButton.Yes:
            return _refuse("Live jog cancelled")
        # Seed target = current pose; do NOT move on enable (wait for the first jog).
        with self._live_jog_lock:
            self._live_jog_target = [float(q) for q in self._joints]
            self._live_jog_dirty = False
        self._live_jog_stop.clear()
        self._live_jog_thread = threading.Thread(
            target=self._live_jog_worker, daemon=True)
        self._live_jog_thread.start()

    @staticmethod
    def _live_jog_max_step(target: "list[float]", last_sent: "list[float]") -> float:
        """Largest per-joint move (deg) streaming `target` would command vs the last
        sent pose. The worker BLOCKS the send if this exceeds _LIVE_JOG_MAX_STEP_DEG
        (teleport guard)."""
        return max((abs(t - s) for t, s in zip(target, last_sent)), default=0.0)

    def _stream_live_jog(self) -> None:
        """Push the current jogged joints to the live-jog worker (latest-value-wins,
        coalesced). No-op unless Phase-2 live jog is ON. Called from user jog
        handlers (Cartesian jog + joint sliders) on the main thread."""
        act = getattr(self, "_act_live_jog", None)
        if act is None or not act.isChecked():
            return
        if (self._live_jog_thread is None
                or not self._live_jog_thread.is_alive()):
            return
        with self._live_jog_lock:
            self._live_jog_target = [float(q) for q in self._joints]
            self._live_jog_dirty = True

    def _live_jog_worker(self) -> None:
        """Hold ONE HSE connection + servo on, then chase the latest jogged joints
        at a throttled rate until the toggle is turned off. Sends only when the
        target changed (coalesced) so fast jogging never queues commands. ⚠ REAL
        MOTION. On any abnormal exit: servo-OFF + uncheck the toggle (main thread)."""
        import time as _t
        backend = MotomanHSEBackend(
            ip=self._hse_ip, timeout_s=2.0, tool_no=self._hse_tool_no)
        speed_pct = min(float(getattr(self, "_pp_max_speed_pct", 10.0)), 10.0)
        period = 0.12                    # ~8 Hz send rate
        alarm_every = 12                 # poll alarm ~ every 1.5 s of sends
        max_move_fails = 5               # tolerate transient MOVE rejects before bailing
        n = 0
        fail_streak = 0
        last_sent: "list[float] | None" = None
        try:
            backend.connect()
            if not backend.Valid():
                self._signals.live_jog_off.emit(
                    "Robot: HSE not responding — live jog OFF"); return
            code, sub = backend.read_alarm()
            if code != 0:
                self._signals.live_jog_off.emit(
                    f"Robot: ALARM 0x{code:04X} — reset on TP first; live jog OFF")
                return
            # SAFETY: refuse if a job is already executing on the controller — don't
            # fight TP playback / another motion source.
            try:
                if backend.read_status_running():
                    self._signals.live_jog_off.emit(
                        "Robot: a job is RUNNING on the controller — stop it first; "
                        "live jog OFF")
                    return
            except Exception:                            # noqa: BLE001
                pass
            self._signals.status.emit("Robot: servo ON…", "info")
            try:
                backend.servo_on(); _t.sleep(1.0)        # let servos engage
            except Exception as e:                       # noqa: BLE001
                self._signals.status.emit(
                    f"Robot: servo-on failed ({e}) — check REMOTE mode", "warn")
            if self._live_jog_stop.is_set():
                try: backend.Stop()
                except Exception: pass                   # noqa: BLE001
                self._signals.status.emit("Robot: live jog OFF — servo OFF", "warn")
                return
            # SAFETY (sim↔real sync): the sim pose may differ from where the REAL
            # robot actually is. Read the real joints and SYNC the sim to them so the
            # first jog is a small increment FROM the real pose — never a teleport
            # from an arbitrary sim pose. (RoboDK does the same before online jog.)
            try:
                q_real = [float(q) for q in backend.Joints()]
            except Exception as e:                       # noqa: BLE001
                try: backend.Stop()
                except Exception: pass                   # noqa: BLE001
                self._signals.live_jog_off.emit(
                    f"Robot: can't read current joints ({e}) — live jog OFF")
                return
            last_sent = q_real
            # Drop any stale target a jog set during the connect/servo window so the
            # first streamed move is a fresh increment from the synced real pose, not
            # an old sim-pose delta.
            with self._live_jog_lock:
                self._live_jog_target = list(q_real)
                self._live_jog_dirty = False
            self._signals.joints_update.emit(list(q_real))   # sim → real pose
            self._signals.status.emit(
                f"Robot: LIVE JOG ON @ {speed_pct:.0f}% — synced to real pose; "
                f"jog to drive the robot", "ok")
            while not self._live_jog_stop.is_set():
                target = None
                with self._live_jog_lock:
                    if self._live_jog_dirty and self._live_jog_target is not None:
                        target = list(self._live_jog_target)
                        self._live_jog_dirty = False
                if target is not None and target != last_sent:
                    # SAFETY (max-step): a big jump from the last sent target means
                    # the sim teleported (paste / Home / load target) while live jog
                    # was ON — do NOT stream it as a fast move. Block + ask to re-sync.
                    step = self._live_jog_max_step(target, last_sent)
                    if step > self._LIVE_JOG_MAX_STEP_DEG:
                        self._signals.status.emit(
                            f"Robot: jog step {step:.0f}° > "
                            f"{self._LIVE_JOG_MAX_STEP_DEG:.0f}° BLOCKED (sim out of "
                            f"sync — toggle live jog off/on to re-sync)", "warn")
                        _t.sleep(period)
                        continue
                    try:
                        backend.move_joints(
                            [round(q, 3) for q in target], speed_pct=speed_pct)
                        last_sent = target
                        fail_streak = 0
                    except Exception as e:               # noqa: BLE001
                        # The controller can intermittently reject a streamed MOVE
                        # (busy/timing). Don't kill the whole live-jog on a single
                        # reject — keep the session alive and retry; only servo-OFF
                        # and bail after several CONSECUTIVE failures (real fault).
                        fail_streak += 1
                        self._signals.status.emit(
                            f"Robot: move rejected ({fail_streak}/{max_move_fails}): {e}",
                            "warn")
                        if fail_streak >= max_move_fails:
                            try: backend.Stop()
                            except Exception: pass       # noqa: BLE001
                            self._signals.live_jog_off.emit(
                                f"Robot: {fail_streak} consecutive moves failed — "
                                f"servo OFF, live jog OFF")
                            return
                        # keep this target dirty-equivalent so we retry next tick
                        with self._live_jog_lock:
                            self._live_jog_target = list(target)
                            self._live_jog_dirty = True
                        _t.sleep(period)
                        continue
                    n += 1
                    if n % alarm_every == 0:
                        try:
                            code, _s = backend.read_alarm()
                            if code != 0:
                                backend.Stop()
                                self._signals.live_jog_off.emit(
                                    f"Robot: ALARM 0x{code:04X} during jog — servo OFF")
                                return
                        except Exception:                # noqa: BLE001
                            pass
                _t.sleep(period)
            # Normal stop (toggle OFF / Stop-all / app close).
            try: backend.Stop()
            except Exception: pass                       # noqa: BLE001
            self._signals.status.emit("Robot: live jog OFF — servo OFF", "warn")
        except Exception as e:                           # noqa: BLE001
            try: backend.Stop()
            except Exception: pass                       # noqa: BLE001
            self._signals.live_jog_off.emit(f"Robot live-jog error: {e}")
        finally:
            try: backend.disconnect()
            except Exception: pass                       # noqa: BLE001

    def _on_live_jog_worker_exit(self, msg: str) -> None:
        """Main-thread slot: worker exited abnormally → uncheck toggle + report."""
        act = getattr(self, "_act_live_jog", None)
        if act is not None:
            act.blockSignals(True)
            act.setChecked(False)
            act.blockSignals(False)
        self._set_status(msg, level="err")

    def _on_run_overwrite_blocked(self, job_name: str, message: str) -> None:
        """Main-thread slot: a Run-on-Robot upload could not overwrite an existing job
        on the controller. Explain clearly and — if the blocked job is the active one
        — offer to rename it in the app right away (then the user re-runs)."""
        self._set_status(message, level="err")
        is_active = (self._robot_job_name(self._active_job) == job_name)
        if is_active:
            r = QMessageBox.warning(
                self, "Job already on the controller",
                f"<b>Cannot overwrite job '{job_name}' on the YRC1000.</b><br><br>"
                f"{message}<br><br>"
                f"Rename this job in the app now (then Run on Robot again)?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel)
            if r == QMessageBox.StandardButton.Yes and hasattr(self, "_on_job_rename"):
                self._on_job_rename()
        else:
            QMessageBox.warning(
                self, "Job already on the controller",
                f"<b>Cannot overwrite job '{job_name}' on the YRC1000.</b><br><br>"
                f"{message}<br><br>"
                f"Select job '{job_name}' in the Program panel and click the Rename "
                f"(pencil) button, or delete/deselect it on the teach pendant; then "
                f"Run on Robot again.")

    def _on_stop_all(self) -> None:
        """Stop EVERYTHING that can move the robot: sim playback, Run-on-Robot job,
        Phase-2 live jog, and a Digital Twin experiment (all servo-off / latched)."""
        # Sim stop
        self._on_prog_stop()
        # Robot stop — Run on Robot worker
        if self._hse_thread is not None and self._hse_thread.is_alive():
            self._hse_stop.set()
            self._set_status("Robot: STOP signaled (will servo-off)", level="warn")
        # Robot stop — Phase-2 live jog
        if (getattr(self, "_live_jog_thread", None) is not None
                and self._live_jog_thread.is_alive()):
            self._live_jog_stop.set()
            act = getattr(self, "_act_live_jog", None)
            if act is not None:
                act.blockSignals(True); act.setChecked(False); act.blockSignals(False)
            self._set_status("Robot: live jog STOP (servo-off)", level="warn")
        # Robot stop — Phase-1 discrete send-pose
        if getattr(self, "_send_pose_busy", False):
            self._send_pose_stop.set()
            self._set_status("Robot: send-pose STOP signaled (servo-off)", level="warn")
        # Robot stop — Digital Twin experiment (autonomous motion). Without this,
        # the global Stop silently fails to halt a running experiment.
        if getattr(self, "_exp_running", False) and hasattr(self, "_on_stop_experiment"):
            self._on_stop_experiment()
            self._set_status("Digital Twin: STOP signaled (servo-off)", level="warn")
