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
import logging
import math
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

    # ─── Import INFORM .JBI (reverse of export) ───
    def _on_prog_import_jbi_dlg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Yaskawa INFORM .JBI", "",
            "INFORM (*.JBI *.jbi);;All files (*)")
        if not path:
            return
        self._load_jbi_file(path)

    def _load_jbi_file(self, path) -> bool:
        """Parse 1 file .JBI → list[Instruction] → load thành job hiện tại.

        Thay TOÀN BỘ project (giống Load JSON) vì .JBI = 1 job đơn. MOVL/MOVC
        cần model robot (FK dựng lại Cartesian pose) → báo lỗi nếu chưa load
        robot. Return True nếu OK.
        """
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
            parsed = parse_jbi(text)
            instrs = self._jbi_to_instructions(parsed)
            job_name = self._safe_job_name(parsed.name) or "IMPORTED"
            self._jobs = {job_name: instrs}
            self._active_job = job_name
            self._targets = {}
            self._refresh_job_combo()
            self._refresh_program_list()
            self._refresh_target_list()
            self._saved_signature = self._project_signature()   # imported = clean
            for w in parsed.warnings:
                logger.warning("JBI import: %s", w)
            msg = f"Imported .JBI '{job_name}': {len(instrs)} steps"
            if parsed.warnings:
                msg += f" — {len(parsed.warnings)} line(s) skipped/warned"
                QMessageBox.warning(
                    self, "Import .JBI — warnings",
                    f"{len(parsed.warnings)} dòng không nằm trong subset hỗ trợ "
                    f"(JUMP/IF/SET/biến/P-var…) đã bị bỏ qua:\n\n"
                    + "\n".join(f"• {w}" for w in parsed.warnings[:12])
                    + ("\n…" if len(parsed.warnings) > 12 else ""))
            self._set_status(msg, level=("warn" if parsed.warnings else "ok"))
            return True
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"Import .JBI failed: {e}", level="err")
            return False

    def _joints_deg_to_tcp_pose(self, joints_deg: list[float]) -> list[float]:
        """Joints (deg) → TCP pose [X,Y,Z mm, Rx,Ry,Rz deg] WORLD frame.

        Inverse chính xác của `_solve_movel` forward path: FK(joints)=T_tool0,
        rồi áp tool frame T_tcp = T_tool0 @ T_flange_tool. Vì _solve_movel làm
        T_target_tool0 = T_target @ inv(T_flange_tool), round-trip khử nhau →
        re-export trả về đúng joints ban đầu.
        """
        if self._model is None:
            raise RuntimeError(
                "MOVL/MOVC cần model robot — load robot GP7 trước khi import")
        T_flange_tool = self._tool_frames[self._tool_idx][1]
        sol_rad = [math.radians(q) for q in joints_deg]
        T_tcp = forward_kinematics_urdf(self._model, sol_rad) @ T_flange_tool
        return list(_matrix_to_xyz_rpy_deg(T_tcp))

    def _jbi_to_instructions(self, parsed: ParsedJob) -> list[Instruction]:
        """ParsedJob → list[Instruction] của editor.

        - MOVJ → MoveJ (joints inline, exact).
        - MOVL → MoveL (FK joints → Cartesian pose).
        - MOVC (cặp liên tiếp) → MoveC (mid+end); MOVC lẻ → degrade MoveL.
        - Modifier VJ/V/PL/TL/UF tái tạo thành Set* instruction khi đổi giá trị.
        - DOUT→SetDO, TIMER→Wait, WAIT→WaitIO, MSG→ShowMessage, CALL→CallJob.
        """
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
                # Modal: emit Set* khi modifier đổi so với trạng thái hiện tại.
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
                # Move itself
                if p.kind == "movj":
                    out.append(Instruction(type="MoveJ",
                                           joints=list(p.joints_deg or [])))
                elif p.kind == "movl":
                    out.append(Instruction(
                        type="MoveL",
                        tcp_pose=self._joints_deg_to_tcp_pose(p.joints_deg or [])))
                else:                                       # movc
                    if i + 1 < n and items[i + 1].kind == "movc":
                        out.append(Instruction(
                            type="MoveC",
                            tcp_pose_mid=self._joints_deg_to_tcp_pose(
                                p.joints_deg or []),
                            tcp_pose=self._joints_deg_to_tcp_pose(
                                items[i + 1].joints_deg or [])))
                        i += 1                              # consume cặp
                    else:
                        parsed.warnings.append(
                            "MOVC lẻ (thiếu cặp mid+end) → coi như MOVL")
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
            # ── Flow control + variables ──
            elif p.kind == "label":
                out.append(Instruction(type="Label", label_name=p.label_name))
            elif p.kind == "jump":
                out.append(Instruction(
                    type="Jump", label_name=p.label_name,
                    cond_lhs=p.cond_lhs, cond_op=p.cond_op, cond_rhs=p.cond_rhs))
            elif p.kind == "setvar":
                out.append(Instruction(
                    type="SetVar", var_name=p.var_name, var_op=p.var_op,
                    var_arg=p.var_arg))
            elif p.kind == "ifthen":
                out.append(Instruction(
                    type="IfThen", cond_lhs=p.cond_lhs, cond_op=p.cond_op,
                    cond_rhs=p.cond_rhs))
            elif p.kind == "elseif":
                out.append(Instruction(
                    type="ElseIf", cond_lhs=p.cond_lhs, cond_op=p.cond_op,
                    cond_rhs=p.cond_rhs))
            elif p.kind == "else":
                out.append(Instruction(type="Else"))
            elif p.kind == "endif":
                out.append(Instruction(type="EndIf"))
            elif p.kind == "while":
                out.append(Instruction(
                    type="While", cond_lhs=p.cond_lhs, cond_op=p.cond_op,
                    cond_rhs=p.cond_rhs))
            elif p.kind == "endwhile":
                out.append(Instruction(type="EndWhile"))
            i += 1
        return out

    def _on_prog_export_dlg(self) -> None:
        # Multi-job: nếu project có > 1 job, hỏi user mode export.
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
            self._export_job_to_path(self._program, stem, Path(path))
            self._set_status(
                f"Exported '{self._active_job}' → {Path(path).name}", level="ok")
        except Exception as e:                              # noqa: BLE001
            self._set_status(f"Export failed: {e}", level="err")

    def _export_all_jobs(self, job_names: list[str]) -> None:
        out_dir = QFileDialog.getExistingDirectory(
            self, "Export ALL jobs — choose output folder")
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
        errs = validate_program(program)
        if errs:
            raise RuntimeError(
                f"Program logic invalid ({len(errs)} error): {errs[0]}")
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
                            f"Target '{ins.target_name}' does not exist")
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
            # ── Flow control + variables (INFORM logic) ──
            elif t == "Label":
                builder.label(ins.label_name)
            elif t == "Jump":
                cond = ((ins.cond_lhs, ins.cond_op, ins.cond_rhs)
                        if ins.cond_op else None)
                builder.jump(ins.label_name, cond)
            elif t == "SetVar":
                builder.set_var(ins.var_name, ins.var_op, ins.var_arg)
            elif t == "IfThen":
                builder.if_then((ins.cond_lhs, ins.cond_op, ins.cond_rhs))
            elif t == "ElseIf":
                builder.else_if((ins.cond_lhs, ins.cond_op, ins.cond_rhs))
            elif t == "Else":
                builder.else_()
            elif t == "EndIf":
                builder.end_if()
            elif t == "While":
                builder.while_((ins.cond_lhs, ins.cond_op, ins.cond_rhs))
            elif t == "EndWhile":
                builder.end_while()
        path.write_bytes(builder.render().encode("utf-8"))
