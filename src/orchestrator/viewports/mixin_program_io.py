"""
mixin_program_io.py
───────────────────
ProgramIOMixin: Save/Load project JSON (v1/v2/v3 backward-compat) + Export
INFORM .JBI (single or all jobs).

Host class (GP7AppQt) must provide:
  attributes: _active_job, _jobs, _targets, _program, _saved_signature,
              _pp_max_speed_pct, _pp_default_vj, _pp_default_v_mms
  methods:    _set_status, _safe_job_name, _solve_movel, _project_signature,
              _refresh_job_combo, _refresh_program_list, _refresh_target_list
"""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from ..backends.inform_codegen import InformJobBuilder
from ..backends.inform_parser import ParsedJob, parse_jbi
from ..kinematics.urdf_chain import forward_kinematics_urdf
from .control_panel import _matrix_to_xyz_rpy_deg
from .program_logic import validate_program
from .program_model import Instruction

logger = logging.getLogger(__name__)


class ProgramIOMixin:
    """Save/Load project JSON + Export INFORM .JBI."""

    def _on_prog_save_dlg(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project", "", "Program JSON (*.json)")
        if not path: return
        try:
            # v3 format: project with multiple jobs + global targets.
            doc = {
                "version": 3,
                "active_job": self._active_job,
                "targets": self._targets,
                "jobs": {
                    name: [ins.to_dict() for ins in prog]
                    for name, prog in self._jobs.items()
                },
                # Post-processor settings (modal speed defaults + VJ safety cap).
                # Persisted so a saved project reproduces the SAME default VJ/V on
                # moves before the first SetSpeed (and the same Max VJ cap on export).
                "post_processor": {
                    "max_speed_pct": self._pp_max_speed_pct,
                    "default_vj": self._pp_default_vj,
                    "default_v_mms": self._pp_default_v_mms,
                },
            }
            if self._job_folders:                       # ///FOLDERNAME per job
                doc["job_folders"] = dict(self._job_folders)
            # Preserve verbatim .JBI source + P-var table so a project imported
            # from .JBI still re-exports byte-exact after a save/load round-trip.
            if getattr(self, "_jbi_raw", None):
                doc["jbi_raw"] = self._jbi_raw
            if getattr(self, "_jbi_positions", None):
                doc["jbi_positions"] = self._jbi_positions
            Path(path).write_text(json.dumps(doc, indent=2), encoding="utf-8")
            self._saved_signature = self._project_signature()   # mark clean
            total_steps = sum(len(p) for p in self._jobs.values())
            self._set_status(
                f"Saved {len(self._jobs)} job(s), {total_steps} steps total, "
                f"{len(self._targets)} targets", level="ok")
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"Save failed: {e}", level="err")

    def _on_prog_open_file_dlg(self) -> None:
        """Open a job code file from the Program panel (button beside the Job
        selector). Accepts BOTH Yaskawa INFORM .JBI and the editor's .json
        project, dispatching to the right loader by extension — a convenience
        entry that reuses the same loaders as the File menu."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open job code", "",
            "Job code (*.JBI *.jbi *.json);;INFORM (*.JBI *.jbi);;"
            "Program JSON (*.json);;All files (*)")
        if not path:
            return
        if Path(path).suffix.lower() == ".json":
            self._load_program_file(path)
        else:
            self._load_jbi_file(path)

    def _on_prog_load_dlg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load project", "", "Program JSON (*.json)")
        if not path: return
        self._load_program_file(path)

    def _load_program_file(self, path) -> bool:
        """Parse + load program JSON from `path` into jobs/targets/UI.

        Supports backward compat for 3 formats:
          v1: bare list of instructions → single MAIN job
          v2: {"targets":..., "program":[...]} → single MAIN job
          v3: {"jobs":{name:[...]}, "active_job":..., "targets":...}

        Shared by the Load button (dialog) + `--program` launcher arg.
        Returns True if load succeeded.
        """
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self._job_folders = {}
            # Restored below if the project carried imported-.JBI metadata.
            self._jbi_raw = {}
            self._jbi_positions = {}
            if isinstance(data, dict):
                self._jbi_raw = dict(data.get("jbi_raw", {}) or {})
                self._jbi_positions = {
                    k: list(v)
                    for k, v in (data.get("jbi_positions", {}) or {}).items()}
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
                        # Preserve original .JBI var token (P-var round-trip).
                        **({"jbi_token": str(v["jbi_token"])}
                           if v.get("jbi_token") else {}),
                    } for k, v in tgt_raw.items()
                }
                self._job_folders = {
                    str(k): str(v)
                    for k, v in (data.get("job_folders", {}) or {}).items()}
                # Restore post-processor settings (fallback to current values when a
                # field/the block is absent — e.g. older v1/v2 projects).
                pp = data.get("post_processor") or {}
                if isinstance(pp, dict):
                    self._pp_max_speed_pct = float(
                        pp.get("max_speed_pct", self._pp_max_speed_pct))
                    self._pp_default_vj = float(
                        pp.get("default_vj", self._pp_default_vj))
                    self._pp_default_v_mms = float(
                        pp.get("default_v_mms", self._pp_default_v_mms))
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
            self._saved_signature = self._project_signature()   # loaded = clean
            total_steps = sum(len(p) for p in self._jobs.values())
            self._set_status(
                f"Loaded {len(self._jobs)} job(s), {total_steps} steps, "
                f"{len(self._targets)} targets", level="ok")
            return True
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"Load failed: {e}", level="err")
            return False

    # ─── Import INFORM .JBI (reverse of export) ───
    def _on_prog_import_jbi_dlg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Yaskawa INFORM .JBI", "",
            "INFORM (*.JBI *.jbi);;All files (*)")
        if not path:
            return
        self._load_jbi_file(path)

    def _load_jbi_file(self, path) -> bool:
        """Parse one .JBI file → list[Instruction] → load as the current job.

        Replaces the ENTIRE project (like Load JSON) because .JBI = single job.
        Each //POS var becomes a shared target holding the EXACT original joints;
        MOVJ/MOVL reference it by name (points reused in the job → one target),
        so re-export reproduces the original joints without IK drift. MOVC (no
        target support) still reconstructs a Cartesian pose via FK, which needs
        the robot model loaded. Returns True if OK.
        """
        try:
            main_path = Path(path)
            targets: dict[str, dict] = {}
            jobs: dict[str, list[Instruction]] = {}
            folders: dict[str, str] = {}
            raw_texts: dict[str, str] = {}      # job key → original .JBI text
            all_warnings: list[str] = []
            in_indices: set[int] = set()        # IN#(n) referenced in conditions
            # Global P-var table (canonical key 'P10' → joints_deg), merged across
            # all jobs — controller P-vars are global, used by P[Bxxx] indirect.
            global_pos: dict[str, list[float]] = {}

            # BFS over CALL JOB references — load the master + ALL reachable
            # sub-jobs (sibling .JBI files in the same folder), to any depth. A
            # sub-job may itself CALL further sub-jobs (master → N sub-jobs, even
            # nested). Each job is keyed by the name used to CALL it (= file stem),
            # because on the real controller `CALL JOB:X` resolves to file X.JBI —
            # NOT by the file's //NAME (which may differ). main job keeps //NAME.
            main_job_name = ""
            # Queue items: (path, job_key). main uses its //NAME; subs use the
            # CALL name (file stem) so _exec_call_job lookups always resolve.
            pending: list[tuple[Path, str | None]] = [(main_path, None)]
            seen_files: set[str] = set()
            while pending:
                p, forced_key = pending.pop(0)
                key = str(p.resolve()).lower()
                if key in seen_files:
                    continue
                seen_files.add(key)
                if not p.exists():
                    continue
                raw_text = p.read_text(encoding="utf-8", errors="replace")
                parsed = parse_jbi(raw_text)
                if forced_key is not None:
                    jname = forced_key                       # sub-job: CALL name
                else:
                    jname = self._safe_job_name(parsed.name) or p.stem.upper()
                    main_job_name = jname                    # master: keep //NAME
                instrs = self._jbi_to_instructions(parsed, targets)
                jobs[jname] = instrs
                # Stash original text + the //NAME so an UNEDITED job re-exports
                # byte-for-byte (preserves comments/COMM/IFTHENEXP/format/order).
                raw_texts[jname] = {"text": raw_text, "name": parsed.name}
                if parsed.folder_name:
                    folders[jname] = parsed.folder_name
                global_pos.update(parsed.positions)          # merge P-var table
                all_warnings.extend(parsed.warnings)
                # Collect IN# used in conditions (for sim_io defaulting).
                for ins in instrs:
                    if ins.cond_op:
                        for tok in (ins.cond_lhs, ins.cond_rhs):
                            mm = re.match(r"IN#\((\d+)\)", str(tok),
                                          re.IGNORECASE)
                            if mm:
                                in_indices.add(int(mm.group(1)))
                # Queue every CALL JOB target as a sibling .JBI file, keyed by
                # the CALL name so the sub-job is found at play time.
                for ins in instrs:
                    if ins.type == "CallJob" and ins.job_name:
                        sub_key = self._safe_job_name(ins.job_name) \
                            or ins.job_name.upper()
                        if sub_key in jobs:
                            continue                         # already loaded
                        for cand in (f"{ins.job_name}.JBI", f"{ins.job_name}.jbi"):
                            sib = p.parent / cand
                            if sib.exists():
                                pending.append((sib, sub_key))
                                break

            if not jobs:
                raise ValueError("No job parsed from .JBI")

            self._jobs = jobs
            self._active_job = main_job_name or next(iter(jobs))
            self._targets = targets
            self._job_folders = folders
            # Original .JBI text per job + a snapshot of the imported instruction
            # list, so an UNEDITED job re-exports byte-for-byte (verbatim). Edited
            # jobs (snapshot mismatch) fall back to re-synthesis from the model.
            self._jbi_raw = {
                k: {"text": v["text"], "name": v["name"],
                    "sig": self._job_signature(jobs[k])}
                for k, v in raw_texts.items()}
            # Global P-var table for P[Bxxx] indirect motions (resolved at play).
            self._jbi_positions = global_pos
            # Per user choice: treat referenced inputs (IN#) as ON so gated
            # CALL JOB / IF blocks fire in the sim (e.g. IFTHENEXP IN#(8)=ON).
            self._sim_io = {idx: True for idx in in_indices}
            self._refresh_job_combo()
            self._refresh_program_list()
            self._refresh_target_list()
            self._saved_signature = self._project_signature()   # imported = clean
            for w in all_warnings:
                logger.warning("JBI import: %s", w)
            njobs = len(jobs)
            total = sum(len(v) for v in jobs.values())
            msg = (f"Imported .JBI '{self._active_job}'"
                   + (f" + {njobs-1} sub-job(s)" if njobs > 1 else "")
                   + f": {total} steps")
            if all_warnings:
                msg += f" — {len(all_warnings)} line(s) skipped/warned"
                QMessageBox.warning(
                    self, "Import .JBI — warnings",
                    f"{len(all_warnings)} line(s) outside the supported subset "
                    f"(JUMP/IF/SET/variable/P-var…) were skipped:\n\n"
                    + "\n".join(f"• {w}" for w in all_warnings[:12])
                    + ("\n…" if len(all_warnings) > 12 else ""))
            self._set_status(msg, level=("warn" if all_warnings else "ok"))
            return True
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"Import .JBI failed: {e}", level="err")
            return False

    def _joints_deg_to_tcp_pose(self, joints_deg: list[float]) -> list[float]:
        """Joints (deg) → TCP pose [X,Y,Z mm, Rx,Ry,Rz deg] in WORLD frame.

        Exact inverse of the `_solve_movel` forward path: FK(joints)=T_tool0,
        then apply tool frame T_tcp = T_tool0 @ T_flange_tool. Because
        _solve_movel computes T_target_tool0 = T_target @ inv(T_flange_tool),
        the round-trip cancels → re-export recovers the original joints.
        """
        if self._model is None:
            raise RuntimeError(
                "MOVL/MOVC require the robot model — load GP7 robot before importing")
        T_flange_tool = self._tool_frames[self._tool_idx][1]
        sol_rad = [math.radians(q) for q in joints_deg]
        T_tcp = forward_kinematics_urdf(self._model, sol_rad) @ T_flange_tool
        return list(_matrix_to_xyz_rpy_deg(T_tcp))

    def _jbi_to_instructions(
        self, parsed: ParsedJob, targets: dict[str, dict],
    ) -> list[Instruction]:
        """ParsedJob → list[Instruction] for the editor.

        - MOVJ → MoveJ, MOVL → MoveL: both reference a SHARED target (keyed by
          the //POS var) holding the EXACT original joints. Motions reusing the
          same point share one target — preserving the original structure and
          giving lossless joint round-trip on re-export.
        - MOVC (consecutive pair) → MoveC (mid+end); lone MOVC → degrade to MoveL.
          MoveC has no target support, so it still uses FK→Cartesian pose.
        - Modifiers VJ/V/PL/TL/UF are reconstructed as Set* instructions on change.
        - DOUT→SetDO, TIMER→Wait, WAIT→WaitIO, MSG→ShowMessage, CALL→CallJob.

        `targets` is populated in place with the shared target library.
        """
        def _target_for(p) -> str:
            """Get/create a shared target for parsed motion `p`, return its name.

            Stores the EXACT joints (used for play/export). tcp_pose is best-effort
            metadata: FK if the robot model is loaded, else [] — so PULSE jobs
            import even before a robot is loaded (target moves use joints, not pose).
            """
            joints = list(p.joints_deg or [])
            name = self._safe_target_name(p.pos_key or "") or "P"
            if name not in targets:
                tcp = (self._joints_deg_to_tcp_pose(joints)
                       if self._model is not None else [])
                # jbi_token preserves the original //POS var (kind + index) so
                # re-export reproduces 'P00001' rather than a renumbered C-var.
                targets[name] = {"joints": joints, "tcp_pose": tcp,
                                 "jbi_token": p.pos_key or ""}
            return name

        out: list[Instruction] = []
        cur_vj: float | None = None
        cur_v: float | None = None
        cur_pl: int | None = None
        cur_tl: int | None = None
        cur_uf: int | None = None
        items = parsed.instructions
        n = len(items)
        i = 0
        while i < n:
            p = items[i]
            if p.kind in ("movj", "movl", "movc"):
                # Modal: emit Set* when a modifier changes relative to current state.
                new_vj = p.vj_pct if p.vj_pct is not None else cur_vj
                new_v = p.v_mm_s if p.v_mm_s is not None else cur_v
                if (new_vj, new_v) != (cur_vj, cur_v) and (
                        new_vj is not None or new_v is not None):
                    out.append(Instruction(
                        type="SetSpeed",
                        speed_joint_pct=float(new_vj if new_vj is not None
                                              else 10.0),
                        speed_linear_mm_s=float(new_v if new_v is not None
                                                else 100.0)))
                    cur_vj, cur_v = new_vj, new_v
                if p.tool_no is not None and p.tool_no != cur_tl:
                    out.append(Instruction(type="SetTool", tool_no=int(p.tool_no)))
                    cur_tl = p.tool_no
                if p.user_frame is not None and p.user_frame != cur_uf:
                    out.append(Instruction(type="SetRefFrame",
                                           ref_frame_no=int(p.user_frame)))
                    cur_uf = p.user_frame
                if p.pl is not None and p.pl != cur_pl:
                    out.append(Instruction(type="SetRounding",
                                           rounding_pl=int(p.pl)))
                    cur_pl = p.pl
                # Move itself — MOVJ/MOVL reference a shared target (exact joints).
                if p.kind in ("movj", "movl"):
                    mtype = "MoveJ" if p.kind == "movj" else "MoveL"
                    if p.pos_index_var:
                        # Indirect P[Bxxx] — joints resolved at playback. Keep the
                        # index var + the job's //POS table for runtime lookup.
                        out.append(Instruction(
                            type=mtype, pos_index_var=p.pos_index_var,
                            speed_var=p.speed_var))
                    else:
                        out.append(Instruction(
                            type=mtype, target_name=_target_for(p),
                            speed_var=p.speed_var))
                else:                                       # movc
                    if i + 1 < n and items[i + 1].kind == "movc":
                        out.append(Instruction(
                            type="MoveC",
                            tcp_pose_mid=self._joints_deg_to_tcp_pose(
                                p.joints_deg or []),
                            tcp_pose=self._joints_deg_to_tcp_pose(
                                items[i + 1].joints_deg or [])))
                        i += 1                              # consume pair
                    else:
                        parsed.warnings.append(
                            "Lone MOVC (missing mid+end pair) → treated as MOVL")
                        out.append(Instruction(
                            type="MoveL",
                            tcp_pose=self._joints_deg_to_tcp_pose(
                                p.joints_deg or [])))
            elif p.kind == "dout":
                out.append(Instruction(type="SetDO", do_index=int(p.do_index),
                                       do_state=bool(p.do_on)))
            elif p.kind == "timer":
                out.append(Instruction(type="Wait",
                                       wait_seconds=float(p.seconds)))
            elif p.kind == "wait_in":
                out.append(Instruction(
                    type="WaitIO", io_index=int(p.in_index),
                    io_state=bool(p.in_on),
                    io_timeout_s=float(p.in_timeout_s)))
            elif p.kind == "msg":
                out.append(Instruction(type="ShowMessage", message=p.text))
            elif p.kind == "call":
                out.append(Instruction(type="CallJob", job_name=p.text))
            # ── Extended LG instructions ──
            elif p.kind == "pulse":
                # PULSE OT#(n) — momentary output. Sim treats like a DOUT pulse.
                out.append(Instruction(type="PulseDO", do_index=int(p.do_index)))
            elif p.kind == "clear_stack":
                out.append(Instruction(type="ClearStack"))
            elif p.kind == "clear":
                out.append(Instruction(type="ClearVar", var_name=p.var_name,
                                       clear_count=int(p.clear_count)))
            elif p.kind == "din":
                out.append(Instruction(
                    type="ReadGroupIn", var_name=p.var_name,
                    io_group=int(p.io_group), io_group_kind=p.io_group_kind))
            elif p.kind == "dout_group":
                out.append(Instruction(
                    type="WriteGroupOut", var_name=p.var_name,
                    io_group=int(p.io_group), io_group_kind="OG"))
            # ── Flow control + variables ──
            elif p.kind == "label":
                out.append(Instruction(type="Label", label_name=p.label_name))
            elif p.kind == "jump":
                out.append(Instruction(
                    type="Jump", label_name=p.label_name,
                    cond_lhs=p.cond_lhs, cond_op=p.cond_op, cond_rhs=p.cond_rhs,
                    cond_terms=list(p.cond_terms), cond_join=p.cond_join,
                    cond_exp=p.cond_exp))
            elif p.kind == "setvar":
                out.append(Instruction(
                    type="SetVar", var_name=p.var_name, var_op=p.var_op,
                    var_arg=p.var_arg, var_expr=p.var_expr))
            elif p.kind == "ifthen":
                out.append(Instruction(
                    type="IfThen", cond_lhs=p.cond_lhs, cond_op=p.cond_op,
                    cond_rhs=p.cond_rhs, cond_terms=list(p.cond_terms),
                    cond_join=p.cond_join, cond_exp=p.cond_exp))
            elif p.kind == "elseif":
                out.append(Instruction(
                    type="ElseIf", cond_lhs=p.cond_lhs, cond_op=p.cond_op,
                    cond_rhs=p.cond_rhs, cond_terms=list(p.cond_terms),
                    cond_join=p.cond_join, cond_exp=p.cond_exp))
            elif p.kind == "else":
                out.append(Instruction(type="Else"))
            elif p.kind == "endif":
                out.append(Instruction(type="EndIf"))
            elif p.kind == "while":
                out.append(Instruction(
                    type="While", cond_lhs=p.cond_lhs, cond_op=p.cond_op,
                    cond_rhs=p.cond_rhs, cond_terms=list(p.cond_terms),
                    cond_join=p.cond_join, cond_exp=p.cond_exp))
            elif p.kind == "endwhile":
                out.append(Instruction(type="EndWhile"))
            i += 1
        return out

    def _on_prog_export_dlg(self) -> None:
        # Multi-job: if project has > 1 job, ask the user for the export mode.
        non_empty_jobs = [n for n, p in self._jobs.items() if p]
        if not non_empty_jobs:
            self._set_status("All jobs empty", level="warn"); return
        export_all = False
        if len(non_empty_jobs) > 1:
            r = QMessageBox.question(
                self, "Export INFORM .JBI",
                f"Project has {len(non_empty_jobs)} non-empty jobs.\n\n"
                f"  Yes → Export ALL jobs (separate .JBI files into one folder)\n"
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
            self._export_job_to_path(
                self._program, stem, Path(path),
                folder_name=self._job_folders.get(self._active_job, ""),
                raw_key=self._active_job)
            self._set_status(
                f"Exported '{self._active_job}' → {Path(path).name}", level="ok")
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"Export failed: {e}", level="err")

    def _export_all_jobs(self, job_names: list[str]) -> None:
        out_dir = QFileDialog.getExistingDirectory(
            self, "Export ALL jobs — choose output folder")
        if not out_dir: return
        # Sequential — IK solving (~1.4ms hot, ~10ms cold) per MoveL has GIL
        # contention when parallel; benchmark confirms threading SLOWER for sub-ms
        # numpy ops. Most jobs ≤ 50 instructions → total <200ms is acceptable.
        n_ok = 0; errors: list[str] = []
        for name in job_names:
            try:
                # Prefer the original //NAME (keeps e.g. SPEED-1's dash) for the
                # filename + job name; fall back to the sanitized key.
                raw = getattr(self, "_jbi_raw", {}).get(name)
                orig_name = (raw or {}).get("name", "")
                stem = (orig_name or self._safe_job_name(name))[:32]
                path = Path(out_dir) / f"{stem}.JBI"
                self._export_job_to_path(
                    self._jobs[name], stem, path,
                    folder_name=self._job_folders.get(name, ""), raw_key=name)
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

    @staticmethod
    def _job_signature(program: list[Instruction]) -> str:
        """Compact signature of an instruction list — used to detect whether a
        job was edited since import (→ decide verbatim vs re-synthesis export)."""
        import json as _json
        return _json.dumps([ins.to_dict() for ins in program], sort_keys=True)

    def _export_job_to_path(
        self, program: list[Instruction], job_name: str, path: Path,
        folder_name: str = "", raw_key: str = "",
    ) -> None:
        """Helper: export one job's instruction list to a .JBI file at path.

        If the job came from an imported .JBI and is UNCHANGED since import
        (signature match), write the ORIGINAL text verbatim (byte-for-byte —
        preserves comments/COMM/IFTHENEXP/declaration order/format). Otherwise
        re-synthesise from the editor model.

        Raises on IK failure / invalid CallJob name. Caller catches to show
        status. folder_name re-emits the original ///FOLDERNAME.
        """
        raw = getattr(self, "_jbi_raw", {}).get(raw_key or job_name)
        if raw and raw.get("sig") == self._job_signature(program):
            path.write_bytes(raw["text"].encode("utf-8"))   # verbatim re-export
            return
        errs = validate_program(program)
        if errs:
            raise RuntimeError(
                f"Program logic invalid ({len(errs)} error): {errs[0]}")
        builder = InformJobBuilder(
            name=job_name, max_speed_pct=self._pp_max_speed_pct,
            folder_name=folder_name)
        # Pre-pass: collect referenced targets, add as named C-vars up front
        # (RoboDK convention — multiple references share single C-var).
        target_cvars: dict[str, str] = {}
        for ins in program:
            if ins.type in ("MoveJ", "MoveL") and ins.target_name:
                if ins.target_name not in target_cvars:
                    tgt = self._targets.get(ins.target_name)
                    if tgt is None:
                        raise KeyError(
                            f"Target '{ins.target_name}' does not exist")
                    cvar_name = f"T_{ins.target_name}"[:32]
                    # Preserve the original .JBI var token (e.g. 'P1' → P00001)
                    # if this target came from a .JBI import; else sequential C-var.
                    builder.add_position(
                        cvar_name, tgt["joints"],
                        pos_token=tgt.get("jbi_token") or None)
                    target_cvars[ins.target_name] = cvar_name
        # Modal state — applied to subsequent MOVJ/MOVL/MOVC. Initialized from pp settings.
        cur_vj_pct: float = self._pp_default_vj
        cur_v_mm_s: float = self._pp_default_v_mms
        cur_pl: int | None = None
        cur_tl: int | None = None
        cur_uf: int | None = None
        pos_idx = 0
        for i, ins in enumerate(program):
            t = ins.type
            if t == "MoveJ":
                if ins.pos_index_var:
                    builder.movj_indirect(ins.pos_index_var,
                                          speed_var=ins.speed_var,
                                          speed_pct=cur_vj_pct)
                else:
                    if ins.target_name:
                        pname = target_cvars[ins.target_name]
                    else:
                        pname = f"P{pos_idx:03d}"
                        builder.add_position(pname, list(ins.joints))
                        pos_idx += 1
                    builder.movj(pname, speed_pct=cur_vj_pct,
                                 tool_no=cur_tl, pl=cur_pl, user_frame=cur_uf,
                                 speed_var=ins.speed_var)
            elif t == "MoveL":
                if ins.pos_index_var:
                    builder.movl_indirect(ins.pos_index_var,
                                          speed_var=ins.speed_var,
                                          speed_mm_s=cur_v_mm_s)
                else:
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
                                 tool_no=cur_tl, pl=cur_pl, user_frame=cur_uf,
                                 speed_var=ins.speed_var)
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
                # Legacy — gripper mapped to fixed DOUT #1 (backward-compat with old programs).
                builder.dout(1, on=ins.gripper_close)
            elif t == "SetDO":
                builder.dout(int(ins.do_index), on=bool(ins.do_state))
            elif t == "Wait":
                builder.timer(max(0.0, ins.wait_seconds))
            elif t == "WaitIO":
                builder.wait_in(int(ins.io_index), on=bool(ins.io_state),
                                timeout_s=max(0.0, float(ins.io_timeout_s)))
            elif t == "SetSpeed":
                # Modal only — folded into each subsequent MOV's inline VJ=/V= tag
                # (no separate INFORM line, matching teach-pendant output).
                cur_vj_pct = float(ins.speed_joint_pct)
                cur_v_mm_s = float(ins.speed_linear_mm_s)
            elif t == "SetRounding":
                cur_pl = int(ins.rounding_pl)               # folded into PL= tag
            elif t == "SetTool":
                cur_tl = int(ins.tool_no)                   # folded into TL= tag
            elif t == "SetRefFrame":
                cur_uf = int(ins.ref_frame_no)              # folded into UF#= tag
            elif t == "ShowMessage":
                builder.msg(ins.message)
            elif t == "CallJob":
                builder.call_job(ins.job_name)
            elif t == "SimEvent":
                # Not exported to INFORM (sim-only). Emit a comment for traceability.
                pl = f" — {ins.event_payload}" if ins.event_payload else ""
                builder.comment(f"SimEvent: {ins.event_name}{pl}")
            # ── Extended LG instructions ──
            elif t == "PulseDO":
                builder.pulse(int(ins.do_index))
            elif t == "ClearStack":
                builder.clear_stack()
            elif t == "ClearVar":
                builder.clear_var(ins.var_name, int(ins.clear_count))
            elif t == "ReadGroupIn":
                builder.din(ins.var_name, ins.io_group_kind, int(ins.io_group))
            elif t == "WriteGroupOut":
                builder.dout_group(int(ins.io_group), ins.var_name)
            # ── Flow control + variables (INFORM logic) ──
            elif t == "Label":
                builder.label(ins.label_name)
            elif t == "Jump":
                if ins.cond_terms:
                    builder.jump(ins.label_name, None,
                                 terms=ins.cond_terms, join=ins.cond_join)
                else:
                    cond = ((ins.cond_lhs, ins.cond_op, ins.cond_rhs)
                            if ins.cond_op else None)
                    builder.jump(ins.label_name, cond)
            elif t == "SetVar":
                if ins.var_expr:
                    builder.set_express(ins.var_name, ins.var_expr)
                else:
                    builder.set_var(ins.var_name, ins.var_op, ins.var_arg)
            elif t == "IfThen":
                builder.if_then((ins.cond_lhs, ins.cond_op, ins.cond_rhs),
                                terms=ins.cond_terms or None, join=ins.cond_join,
                                exp=ins.cond_exp)
            elif t == "ElseIf":
                builder.else_if((ins.cond_lhs, ins.cond_op, ins.cond_rhs),
                                terms=ins.cond_terms or None, join=ins.cond_join,
                                exp=ins.cond_exp)
            elif t == "Else":
                builder.else_()
            elif t == "EndIf":
                builder.end_if()
            elif t == "While":
                builder.while_((ins.cond_lhs, ins.cond_op, ins.cond_rhs),
                               terms=ins.cond_terms or None, join=ins.cond_join,
                               exp=ins.cond_exp)
            elif t == "EndWhile":
                builder.end_while()
        path.write_bytes(builder.render().encode("utf-8"))
