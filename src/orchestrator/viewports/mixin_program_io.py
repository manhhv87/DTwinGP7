"""
mixin_program_io.py
───────────────────
ProgramIOMixin: Save/Load project JSON (v1/v2/v3 backward-compat) + Export
INFORM .JBI (single hoặc all jobs).

Host class (GP7AppQt) phải cung cấp:
  attributes: _active_job, _jobs, _targets, _program, _saved_signature,
              _pp_max_speed_pct, _pp_default_vj, _pp_default_v_mms
  methods:    _set_status, _safe_job_name, _solve_movel, _project_signature,
              _refresh_job_combo, _refresh_program_list, _refresh_target_list
"""
from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from ..backends.inform_codegen import InformJobBuilder
from .program_model import Instruction


class ProgramIOMixin:
    """Save/Load project JSON + Export INFORM .JBI."""

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
            self._saved_signature = self._project_signature()   # mark clean
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
        self._load_program_file(path)

    def _load_program_file(self, path) -> bool:
        """Parse + load program JSON từ `path` vào jobs/targets/UI.

        Hỗ trợ backward compat 3 format:
          v1: bare list of instructions → single MAIN job
          v2: {"targets":..., "program":[...]} → single MAIN job
          v3: {"jobs":{name:[...]}, "active_job":..., "targets":...}

        Dùng chung bởi nút Load (dialog) + `--program` launcher arg.
        Return True nếu load OK.
        """
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
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
            self._saved_signature = self._project_signature()   # loaded = clean
            total_steps = sum(len(p) for p in self._jobs.values())
            self._set_status(
                f"Loaded {len(self._jobs)} job(s), {total_steps} steps, "
                f"{len(self._targets)} targets", level="ok")
            return True
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"Load failed: {e}", level="err")
            return False

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
                # Legacy — gripper map cố định DOUT #1 (backward-compat program cũ).
                builder.dout(1, on=ins.gripper_close)
            elif t == "SetDO":
                builder.dout(int(ins.do_index), on=bool(ins.do_state))
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
