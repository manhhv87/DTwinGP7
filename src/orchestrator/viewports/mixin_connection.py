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

from ..backends.motoman_hse import MotomanHSEBackend


class ConnectionMixin:
    """HSE connection + Run on Robot lifecycle (settings, test, upload, stop)."""

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
                    f"Connection FAIL — check HSE Server is enabled",
                    level="err")
                QMessageBox.warning(
                    self, "Connection failed",
                    f"YRC1000 {ip} did not respond to READ_STATUS.\n"
                    "Verify:\n"
                    " • Ping {ip} OK?\n"
                    " • HSE Server function enabled in Maintenance mode?\n"
                    " • PC on the same subnet as the YRC1000?".format(ip=ip))
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
        """Render active job + all CALL JOB sub-jobs → upload every .JBI →
        JOB_SELECT + START the main job → wait_idle.

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
            f"<b>⚠ THE ROBOT WILL MOVE FOR REAL</b><br><br>"
            f"Job: <code>{self._active_job}</code> ({n_steps} instructions)"
            f"{sub_note}<br>"
            f"HSE IP: <code>{self._hse_ip}</code><br>"
            f"Max VJ: {self._pp_max_speed_pct:.0f}%<br><br>"
            f"Before continuing, verify:<br>"
            f"&nbsp;✓ YRC1000 in REMOTE mode<br>"
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
            # Upload the main job + every sub-job so CALL JOB resolves on the
            # controller (main = jobs[0]).
            for jname, jtext in jobs:
                if self._hse_stop.is_set():
                    self._signals.status.emit(
                        "Robot: aborted before upload", "warn"); return
                self._signals.status.emit(
                    f"Robot: FTP uploading '{jname}.JBI'…", "info")
                backend.upload_job(jtext, jname)
            if self._hse_stop.is_set():
                self._signals.status.emit("Robot: aborted before start", "warn"); return
            import time as _time
            # Servo ON (START requires servo power + REMOTE mode). REMOTE must be
            # set on the TP key switch; servo we command here. Warn (don't abort)
            # if it fails — START below will surface the definitive controller error.
            self._signals.status.emit("Robot: servo ON…", "info")
            try:
                backend.servo_on()
                _time.sleep(1.0)                     # let servos engage before START
            except Exception as e:                   # noqa: BLE001
                self._signals.status.emit(
                    f"Robot: servo-on failed ({e}) — check REMOTE mode on the TP",
                    "warn")
            self._signals.status.emit(f"Robot: JOB_SELECT + START '{main_name}'…", "info")
            backend.job_select(main_name)
            backend.job_start()
            # Poll status until idle or stop
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
                    f"Robot: job '{main_name}' completed OK", "ok")
        except Exception as e:                              # noqa: BLE001
            self._signals.status.emit(f"Robot error: {e}", "err")
        finally:
            try:
                backend.disconnect()
            except Exception:                               # noqa: BLE001
                pass

    def _on_send_pose_to_robot(self) -> None:
        """Phase-1 direct control (RoboDK-style, discrete): send the app's CURRENT
        joints to the REAL robot via HSE MOVE (0x8B, no job upload). One move per
        click. Robot MUST be in REMOTE mode. ⚠ REAL MOTION."""
        if not self._hse_ip:
            self._set_status("HSE IP not configured — Robot → Connection settings",
                             level="warn"); return
        if getattr(self, "_model", None) is None:
            self._set_status("Robot not loaded", level="warn"); return
        joints = [round(float(q), 3) for q in self._joints]
        speed_pct = min(float(getattr(self, "_pp_max_speed_pct", 10.0)), 10.0)
        r = QMessageBox.warning(
            self, "Send pose to REAL robot",
            f"<b>⚠ THE ROBOT WILL MOVE FOR REAL</b><br><br>"
            f"Target joints (deg):<br><code>{joints}</code><br>"
            f"Speed: {speed_pct:.0f}%  ·  HSE: <code>{self._hse_ip}</code><br><br>"
            f"Verify: YRC1000 in <b>REMOTE</b> mode, workspace clear, "
            f"hand on the <b>E-stop</b>.<br><br>Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if r != QMessageBox.StandardButton.Yes:
            return
        backend = MotomanHSEBackend(
            ip=self._hse_ip, timeout_s=3.0, tool_no=self._hse_tool_no)
        try:
            backend.connect()
            if not backend.Valid():
                self._set_status("Robot: HSE not responding", level="err"); return
            code, _sub = backend.read_alarm()
            if code != 0:
                self._set_status(
                    f"Robot: ALARM 0x{code:04X} — reset on TP first", level="err")
                return
            self._set_status("Robot: servo ON…", level="info")
            QApplication.processEvents()
            try:
                backend.servo_on()
                import time as _t; _t.sleep(1.0)        # let servos engage
            except Exception as e:                       # noqa: BLE001
                self._set_status(
                    f"servo-on failed ({e}) — check REMOTE mode", level="warn")
            backend.move_joints(joints, speed_pct=speed_pct)
            self._set_status(
                f"Robot: moving to current pose @ {speed_pct:.0f}%", level="ok")
        except Exception as e:                           # noqa: BLE001
            self._set_status(f"Send-pose error: {e}", level="err")
            QMessageBox.critical(self, "Send pose error", str(e))
        finally:
            try:
                backend.disconnect()
            except Exception:                            # noqa: BLE001
                pass

    def _on_stop_all(self) -> None:
        """Dual-purpose stop: sim playback + robot job (servo OFF if HSE is active)."""
        # Sim stop
        self._on_prog_stop()
        # Robot stop
        if self._hse_thread is not None and self._hse_thread.is_alive():
            self._hse_stop.set()
            self._set_status("Robot: STOP signaled (will servo-off)", level="warn")
