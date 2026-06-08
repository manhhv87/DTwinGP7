"""
mixin_job_target.py
───────────────────
JobTargetMixin: Multi-job project (job selector + add/rename/delete) +
target library (Teach/Modify/Delete + Add MoveJ/MoveL to selected target).

Host class (GP7AppQt) must provide:
  attributes: _jobs (dict[str, list[Instruction]]), _active_job, _program,
              _job_combo (QComboBox), _targets, _tgt_list (QListWidget),
              _tgt_name_edit (QLineEdit), _joints
  methods:    _set_status, _refresh_program_list, _current_tool_world
"""
from __future__ import annotations

from PyQt6.QtWidgets import QInputDialog, QLineEdit, QMessageBox

from .control_panel import _matrix_to_xyz_rpy_deg
from .program_model import Instruction


class JobTargetMixin:
    """Multi-job project + target library."""

    # ── Job selector ──────────────────────────────────────────────────
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
        if self._playback_running():
            # A run drives _active_job itself (CALL JOB view-follow). Re-sync the
            # combo to the live job and ignore the user's switch.
            if name != self._active_job:
                self._set_status(
                    "Cannot switch jobs while the program is running", level="warn")
                self._refresh_job_combo()
            return
        self._active_job = name
        self._refresh_program_list()
        self._set_status(
            f"Active job: {name}  ({len(self._program)} steps)", level="info")

    def _on_job_add(self) -> None:
        if self._guard_not_running("add a job"): return
        v, ok = QInputDialog.getText(
            self, "Add job", "New job name:", QLineEdit.EchoMode.Normal, "SUB1")
        if not ok: return
        name = self._safe_job_name(v)
        if not name:
            self._set_status("Invalid job name", level="warn"); return
        if name in self._jobs:
            self._set_status(f"Job '{name}' already exists", level="warn"); return
        self._jobs[name] = []
        self._active_job = name
        self._refresh_job_combo()
        self._refresh_program_list()
        self._set_status(f"Added job '{name}'", level="ok")

    def _on_job_rename(self) -> None:
        if self._guard_not_running("rename a job"): return
        old = self._active_job
        v, ok = QInputDialog.getText(
            self, "Rename job", f"New name for '{old}':",
            QLineEdit.EchoMode.Normal, old)
        if not ok: return
        new = self._safe_job_name(v)
        if not new or new == old: return
        if new in self._jobs:
            self._set_status(f"Job '{new}' already exists", level="warn"); return
        # Preserve dict order
        new_jobs: dict[str, list[Instruction]] = {}
        for k, v_list in self._jobs.items():
            new_jobs[new if k == old else k] = v_list
        self._jobs = new_jobs
        self._active_job = new
        # Update CallJob refs if any (across all jobs). job_name holds the RAW
        # CALL literal (e.g. "SPEED-1"); the project key is its sanitized form
        # ("SPEED1" == old), so match on the sanitized name, not the raw string.
        for job_prog in self._jobs.values():
            for ins in job_prog:
                if (ins.type == "CallJob"
                        and self._safe_job_name(ins.job_name) == old):
                    ins.job_name = new
        self._refresh_job_combo()
        self._refresh_program_list()
        self._set_status(f"Renamed '{old}' → '{new}'", level="ok")

    def _on_job_delete(self) -> None:
        if self._guard_not_running("delete a job"): return
        if len(self._jobs) <= 1:
            self._set_status("At least one job is required", level="warn"); return
        old = self._active_job
        # Check CallJob refs in other jobs
        refs = sum(
            1
            for k, prog in self._jobs.items() if k != old
            for ins in prog
            if ins.type == "CallJob"
            and self._safe_job_name(ins.job_name) == old)
        msg = f"Delete job '{old}' ({len(self._program)} steps)?"
        if refs > 0:
            msg += f"\n\nWarning: {refs} CallJob instruction(s) in other jobs reference it."
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

    # ── Target library (RoboDK-style Teach Target) ────────────────────
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
            # Joints (J) in degrees, labelled + unit so they aren't misread as a
            # Cartesian pose [X,Y,Z,Rx,Ry,Rz]. Stored value keeps full precision;
            # only the display is rounded to whole degrees for compactness.
            vals = "  ".join(f"{q:+.0f}" for q in j)
            self._tgt_list.addItem(f"{name}   J[{vals}] °")

    def _capture_current_pose(self) -> dict | None:
        """Snapshot current joints + tcp_pose from robot state."""
        T = self._current_tool_world()
        if T is None: return None
        x, y, z, rx, ry, rz = _matrix_to_xyz_rpy_deg(T)
        return {
            "joints": list(self._joints),
            "tcp_pose": [x, y, z, rx, ry, rz],
        }

    def _on_tgt_teach(self) -> None:
        if self._guard_not_running("teach a target"): return
        name_raw = self._tgt_name_edit.text().strip()
        if not name_raw:
            self._set_status("Target name empty", level="warn"); return
        name = self._safe_target_name(name_raw)
        if not name:
            self._set_status("Invalid target name", level="warn"); return
        if name in self._targets:
            self._set_status(
                f"Target '{name}' already exists — use Modify (F3) to update it",
                level="warn"); return
        pose = self._capture_current_pose()
        if pose is None: return
        self._targets[name] = pose
        self._refresh_target_list()
        self._tgt_name_edit.clear()
        self._set_status(f"Target '{name}' taught (n={len(self._targets)})",
                          level="ok")

    def _on_tgt_modify(self) -> None:
        """F3 — replace selected target's pose with current pose."""
        if self._guard_not_running("modify a target"): return
        idx = self._tgt_list.currentRow()
        if idx < 0 or not self._targets:
            self._set_status("Select a target to Modify", level="warn"); return
        name = list(self._targets.keys())[idx]
        pose = self._capture_current_pose()
        if pose is None: return
        self._targets[name] = pose
        self._refresh_target_list()
        self._tgt_list.setCurrentRow(idx)
        self._set_status(f"Target '{name}' modified", level="ok")

    def _on_tgt_delete(self) -> None:
        if self._guard_not_running("delete a target"): return
        idx = self._tgt_list.currentRow()
        if idx < 0 or not self._targets: return
        name = list(self._targets.keys())[idx]
        # Targets are project-global — scan EVERY job, not just the active one,
        # so we don't silently orphan a MoveJ/MoveL in another job.
        refs = sum(
            1
            for prog in self._jobs.values()
            for ins in prog
            if ins.type in ("MoveJ", "MoveL") and ins.target_name == name)
        if refs > 0:
            r = QMessageBox.question(
                self, "Delete target",
                f"Target '{name}' is referenced by {refs} instruction(s).\n"
                "Deleting it will make those moves fail to resolve on play/export.\n\n"
                "Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes: return
        del self._targets[name]
        self._refresh_target_list()
        self._set_status(f"Target '{name}' deleted", level="ok")

    def _on_prog_add_move_to_target(self, kind: str) -> None:
        """kind ∈ {'MoveJ','MoveL'}. Append an instruction referencing the
        currently selected target in the list."""
        if self._guard_not_running(f"add {kind}"): return
        idx = self._tgt_list.currentRow()
        if idx < 0 or not self._targets:
            self._set_status(
                "Select a target in the list first (or Teach a new one)", level="warn"); return
        name = list(self._targets.keys())[idx]
        self._program.append(Instruction(type=kind, target_name=name))
        self._refresh_program_list()
        self._set_status(f"Program +{kind} → {name}", level="ok")
