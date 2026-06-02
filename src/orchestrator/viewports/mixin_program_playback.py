"""
mixin_program_playback.py
─────────────────────────
ProgramPlaybackMixin: program list ops (delete/move/clear), play/stop/pause,
play loop worker, MoveL IK solver (Pieper + DLS fallback).

Host class (GP7AppQt) must provide:
  attributes: _program, _prog_list, _btn_pause, _prog_pause, _prog_stop,
              _prog_thread, _jobs, _active_job, _targets, _signals,
              _sim_speed_mult, _model, _joints, _tool_frames, _tool_idx
  methods:    _set_status, _refresh_program_list, _refresh_job_combo,
              _refresh_target_list, _animate_to
"""
from __future__ import annotations

import logging
import math
import threading
import time

import numpy as np
from PyQt6.QtWidgets import QMessageBox

from ..kinematics.inverse_kinematics import inverse_kinematics_seeded
from ..kinematics.pieper_gp7 import inverse_kinematics_pieper_gp7_nearest
from ..kinematics.urdf_chain import forward_kinematics_urdf
from .control_panel import _xyz_rpy_to_matrix
from .program_logic import (
    CONTROL_FLOW,
    VarStore,
    apply_setvar,
    build_block_map,
    next_pc,
    resolve_labels,
    validate_program,
)
from .program_model import Instruction

# Backstop: prevent infinite loops (JUMP/WHILE) from hanging the sim thread.
_LOOP_GUARD_MAX = 200_000

logger = logging.getLogger(__name__)


class ProgramPlaybackMixin:
    """Program list ops + play loop + IK solver for MoveL."""

    # ── Pause / Resume ────────────────────────────────────────────────
    def _on_prog_toggle_pause(self) -> None:
        if self._btn_pause.isChecked():
            self._prog_pause.set()
            self._btn_pause.setText("▶  Resume")
            self._set_status("Program: paused", level="warn")
        else:
            self._prog_pause.clear()
            self._btn_pause.setText("▮▮  Pause")
            self._set_status("Program: resumed", level="ok")

    def _on_prog_delete(self) -> None:
        idx = self._prog_list.currentRow()
        if 0 <= idx < len(self._program):
            del self._program[idx]
            self._refresh_program_list()

    def _on_prog_move_up(self) -> None:
        idx = self._prog_list.currentRow()
        if 0 < idx < len(self._program):
            self._program[idx-1], self._program[idx] = \
                self._program[idx], self._program[idx-1]
            self._refresh_program_list()
            self._prog_list.setCurrentRow(idx - 1)

    def _on_prog_move_down(self) -> None:
        idx = self._prog_list.currentRow()
        if 0 <= idx < len(self._program) - 1:
            self._program[idx+1], self._program[idx] = \
                self._program[idx], self._program[idx+1]
            self._refresh_program_list()
            self._prog_list.setCurrentRow(idx + 1)

    def _on_prog_clear(self) -> None:
        total_steps = sum(len(p) for p in self._jobs.values())
        if total_steps == 0 and not self._targets:
            return
        r = QMessageBox.question(
            self, "Clear all",
            f"Delete all {len(self._jobs)} job(s), {total_steps} instructions, "
            f"and {len(self._targets)} target(s)?\n\n"
            "Reset the project to an empty MAIN job.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if r != QMessageBox.StandardButton.Yes: return
        self._jobs = {"MAIN": []}
        self._active_job = "MAIN"
        self._targets.clear()
        self._refresh_job_combo()
        self._refresh_program_list()
        self._refresh_target_list()
        self._set_status("Project cleared (reset to MAIN/empty)", level="ok")

    def _on_prog_play(self) -> None:
        if self._prog_thread is not None and self._prog_thread.is_alive():
            self._set_status("Program already running", level="warn"); return
        if not self._program:
            self._set_status("Program empty", level="warn"); return
        errs = validate_program(self._program)
        if errs:
            self._set_status(
                f"Program invalid — {len(errs)} error(s): {errs[0]}"
                + (f" (+{len(errs)-1} more)" if len(errs) > 1 else ""),
                level="err")
            return
        self._prog_stop.clear()
        self._set_status(f"Program: running ({len(self._program)} steps)", "ok")
        self._prog_thread = threading.Thread(
            target=self._play_program_loop, daemon=True)
        self._prog_thread.start()

    def _on_prog_stop(self) -> None:
        if self._prog_thread is not None and self._prog_thread.is_alive():
            self._prog_stop.set()
            self._prog_pause.clear()                # unstuck if currently paused
            self._set_status("Program: stopping...", level="warn")

    def _on_program_done(self) -> None:
        # Reset pause UI (program ended → cannot pause anymore).
        if hasattr(self, "_btn_pause"):
            self._btn_pause.setChecked(False)
            self._btn_pause.setText("▮▮  Pause")
        self._prog_pause.clear()

    def _wait_while_paused(self) -> None:
        """Block the player loop while pause is set; exit quickly when stop is set."""
        while self._prog_pause.is_set() and not self._prog_stop.is_set():
            time.sleep(0.05)

    def _resolve_move_target(
        self, ins: Instruction, want: str,
    ) -> list[float] | None:
        """Resolve joints (want='joints') or tcp_pose (want='tcp_pose') for
        MoveJ/MoveL — from the target library if ins.target_name is set, otherwise
        from inline fields. Returns None and emits err if missing."""
        if ins.target_name:
            tgt = self._targets.get(ins.target_name)
            if tgt is None:
                self._signals.status.emit(
                    f"Target '{ins.target_name}' does not exist", "err")
                return None
            return list(tgt[want])
        return list(ins.joints if want == "joints" else ins.tcp_pose)

    def _play_program_loop(self) -> None:
        prog = self._program
        n = len(prog)
        # Modal sim state (scales animation speed; modifiers are metadata only).
        sim_vj_pct: float = 10.0
        # ── Logic infra: interpreter with program counter + variables + labels/blocks ──
        try:
            label_map = resolve_labels(prog)
            block_map = build_block_map(prog)
        except ValueError as e:
            self._signals.status.emit(f"Program logic error: {e}", "err")
            self._signals.program_done.emit()
            return
        store = VarStore()
        sim_io = getattr(self, "_sim_io", {})        # IN# sim (default OFF)
        def io_reader(idx: int) -> bool:
            return bool(sim_io.get(idx, False))
        if_state: dict[int, bool] = {}
        pc = 0
        guard = 0
        try:
            while pc < n:
                guard += 1
                if guard > _LOOP_GUARD_MAX:
                    self._signals.status.emit(
                        f"Program: loop guard tripped (>{_LOOP_GUARD_MAX} "
                        f"steps) — aborted (infinite loop?)", "err")
                    return
                if self._prog_stop.is_set():
                    self._signals.status.emit("Program: stopped", "warn"); return
                self._wait_while_paused()
                if self._prog_stop.is_set():
                    self._signals.status.emit("Program: stopped", "warn"); return
                ins = prog[pc]
                t = ins.type
                self._signals.status.emit(
                    f"Step {pc+1}/{n}: {ins.describe()}", "info")
                # ── Variables: SET/ADD/INC/... — mutate store, no motion ──
                if t == "SetVar":
                    try:
                        apply_setvar(ins.var_name, ins.var_op, ins.var_arg,
                                     store, io_reader)
                        self._signals.status.emit(
                            f"Step {pc+1}: {ins.var_op.upper()} {ins.var_name} "
                            f"→ {store.get(ins.var_name)}", "info")
                    except ValueError as e:
                        self._signals.status.emit(
                            f"Step {pc+1}: var error: {e}", "err"); return
                    pc += 1
                    continue
                # ── Branching: Label/Jump/IfThen/.../EndWhile → pure next_pc ──
                if t in CONTROL_FLOW:
                    try:
                        pc = next_pc(prog, pc, store, io_reader,
                                     block_map, label_map, if_state)
                    except ValueError as e:
                        self._signals.status.emit(
                            f"Step {pc+1}: flow error: {e}", "err"); return
                    continue
                i = pc                               # alias for the block below
                # Animation step count: low VJ% + low sim_speed_mult → slower.
                base_steps = int(40 * 30.0 / max(5.0, sim_vj_pct))
                steps = max(8, int(base_steps / max(0.1, self._sim_speed_mult)))
                if t == "MoveJ":
                    joints = self._resolve_move_target(ins, "joints")
                    if joints is None: return
                    self._animate_to(joints, steps=steps, dt=0.025,
                                      stop_event=self._prog_stop,
                                      pause_event=self._prog_pause)
                elif t == "MoveL":
                    if ins.target_name:
                        # Target stored joints — bypass IK, animate directly.
                        joints = self._resolve_move_target(ins, "joints")
                        if joints is None: return
                        self._animate_to(joints, steps=steps, dt=0.025,
                                          stop_event=self._prog_stop,
                                          pause_event=self._prog_pause)
                    else:
                        sol = self._solve_movel(ins.tcp_pose)
                        if sol is None:
                            self._signals.status.emit(
                                f"Step {i+1}: IK fail, abort", "err"); return
                        self._animate_to(sol, steps=steps, dt=0.025,
                                          stop_event=self._prog_stop,
                                          pause_event=self._prog_pause)
                elif t == "MoveC":
                    # Simplified sim: run MoveL to mid then end (no true circular
                    # interpolation — sufficient to verify sequence, .JBI still MOVC).
                    for pose in (ins.tcp_pose_mid, ins.tcp_pose):
                        sol = self._solve_movel(pose)
                        if sol is None:
                            self._signals.status.emit(
                                f"Step {i+1}: IK fail (MoveC), abort", "err"); return
                        self._animate_to(sol, steps=steps, dt=0.025,
                                          stop_event=self._prog_stop,
                                          pause_event=self._prog_pause)
                elif t == "SetGripper":
                    self._signals.gripper.emit(bool(ins.gripper_close))
                    time.sleep(0.25 / max(0.1, self._sim_speed_mult))
                elif t == "SetDO":
                    # Generic digital output — sim has no real IO, log only.
                    self._signals.status.emit(
                        f"Step {i+1}: (sim) DOUT OT#{ins.do_index} = "
                        f"{'ON' if ins.do_state else 'OFF'}", "info")
                    time.sleep(0.2 / max(0.1, self._sim_speed_mult))
                elif t == "Wait":
                    deadline = time.monotonic() + max(0.0, ins.wait_seconds) \
                        / max(0.1, self._sim_speed_mult)
                    while time.monotonic() < deadline:
                        if self._prog_stop.is_set(): return
                        self._wait_while_paused()
                        if self._prog_stop.is_set(): return
                        time.sleep(0.05)
                elif t == "WaitIO":
                    # Sim has no real IO → log only + short delay.
                    self._signals.status.emit(
                        f"Step {i+1}: (sim) WaitIO IN#{ins.io_index}="
                        f"{'ON' if ins.io_state else 'OFF'} → assumed satisfied",
                        "warn")
                    time.sleep(0.3 / max(0.1, self._sim_speed_mult))
                elif t == "SetSpeed":
                    sim_vj_pct = float(ins.speed_joint_pct)
                elif t == "ShowMessage":
                    self._signals.status.emit(
                        f"Step {i+1}: MSG \"{ins.message[:32]}\"", "info")
                    time.sleep(0.4 / max(0.1, self._sim_speed_mult))
                elif t == "CallJob":
                    # Sim does not execute sub-jobs — log only to confirm order.
                    self._signals.status.emit(
                        f"Step {i+1}: (sim) CALL JOB:{ins.job_name} → skipped",
                        "warn")
                    time.sleep(0.2 / max(0.1, self._sim_speed_mult))
                elif t == "SimEvent":
                    # Sim hook — emit signal + log. event_name 'reset'/'reset_objects'/
                    # 'reset_scene' → return objects to initial positions (for pick-loop demo).
                    ev = (ins.event_name or "").strip().lower()
                    pl = f" — {ins.event_payload}" if ins.event_payload else ""
                    if ev in ("reset", "reset_objects", "reset_scene"):
                        self._signals.sim_reset.emit()
                    self._signals.status.emit(
                        f"⚑ Step {i+1}: SimEvent '{ins.event_name}'{pl}", "info")
                    time.sleep(0.15 / max(0.1, self._sim_speed_mult))
                # SetRounding / SetTool / SetRefFrame: pure metadata for .JBI,
                # sim needs no action — already shown in status.
                pc += 1
            self._signals.status.emit(
                f"Program: done ({guard} steps executed)", "ok")
        except Exception as e:                              # noqa: BLE001
            self._signals.status.emit(f"Program error: {e}", "err")
        finally:
            self._signals.program_done.emit()

    def _solve_movel(self, tcp_pose_6: list[float]) -> list[float] | None:
        """Solve IK via **Pieper analytical** + verify post-FK pose error.

        Strategy:
          1. Pieper closed-form (50-200µs) → all 3-8 solutions
          2. Pick solution nearest current pose (smoothest motion)
          3. Post-FK verify (defensive; Pieper accuracy ~1e-12mm, not strictly
             needed but cheap + catches unreachable edge cases)
          4. Fallback: DLS with analytical Jacobian if Pieper returns empty (e.g.
             pose out of reach — Pieper will skip configs, DLS may return best-effort)
        """
        TOL_POS_MM = 0.5
        TOL_ROT_RAD = 1e-3
        T_target = _xyz_rpy_to_matrix(*tcp_pose_6)
        T_flange_tool = self._tool_frames[self._tool_idx][1]
        T_target_tool0 = T_target @ np.linalg.inv(T_flange_tool)
        q_init = [math.radians(q) for q in self._joints]
        # 1. Pieper analytical IK
        sol = inverse_kinematics_pieper_gp7_nearest(
            self._model, T_target_tool0, q_init)
        if sol is None:
            # 2. Fallback: DLS seeded retry
            sol = inverse_kinematics_seeded(
                self._model, T_target_tool0, q_init,
                tol_mm=TOL_POS_MM, tol_rad=TOL_ROT_RAD, max_iter=100)
            if sol is None:
                return None
        # 3. Post-FK verify
        T_actual = forward_kinematics_urdf(self._model, sol)
        pos_err = float(np.linalg.norm(T_actual[:3, 3] - T_target_tool0[:3, 3]))
        R_err = T_target_tool0[:3, :3] @ T_actual[:3, :3].T
        cos_theta = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
        rot_err_rad = float(np.arccos(cos_theta))
        if pos_err > TOL_POS_MM or rot_err_rad > TOL_ROT_RAD:
            logger.warning(
                "IK convergent but pose error > tol: pos=%.3fmm rot=%.4frad",
                pos_err, rot_err_rad)
            return None
        return [math.degrees(q) for q in sol]
