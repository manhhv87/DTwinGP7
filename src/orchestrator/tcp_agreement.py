"""
tcp_agreement.py
────────────────
The tool point the twin computes against the one the controller reports (E1).

At a pose the operator has jogged to, the joint angles (READ_POSITION 0x75,
instance 1) and the controller's tool centre point (0x75, instance 101: BASE
frame, the tool number the reply carries) are read; the twin's tool point is its
forward kinematics of those joints, the URDF chain the mirror, the planner and the
IK use, composed with the gripper's tool offset from the cell config
(gripper.tcp_offset_xyz_mm / tcp_offset_rpy_deg), in the controller's BASE frame.

Frames, as the app already matches them to the teach pendant
(gp7_app_qt._pendant_pose, commit 63f27fe): position, the tool point in the robot
BASE frame with no offset; orientation, the twin's tool frame times Rz(180 deg)
about the tool Z. The tool Z (approach) is the same in both conventions.

The reads here are the only exchanges: ReadOnlyHSEBackend refuses, before anything
is sent, every command but the position read, so nothing here can move the robot,
switch the servo or start a job.
"""
from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

import numpy as np

from src.cell.pose_utils import make_homogeneous
from src.orchestrator.backends.hse_protocol import Command, Service
from src.orchestrator.backends.motoman_hse import MotomanHSEBackend
from src.orchestrator.frame_convert import (
    matrix_to_rpy_yaskawa,
    rpy_yaskawa_to_matrix,
    world_to_robot_base,
)
from src.orchestrator.kinematics.urdf_chain import forward_kinematics_urdf, gp7_urdf

# The only (command, service) pairs ReadOnlyHSEBackend lets through.
READ_ONLY_EXCHANGES = frozenset({
    (int(Command.READ_POSITION), int(Service.GET_ATTRIBUTE_ALL)),
})

# Twin tool frame -> pendant TOOL frame: Rz(180 deg) about the tool Z (_pendant_pose).
RZ180 = np.diag([-1.0, -1.0, 1.0])

# A pose is read three times, 0.2 s apart, and kept only when it did not change:
# no joint by more than 0.01 deg (5 to 15 pulses, GP7_PULSE_PER_DEG) and the
# controller's tool point by no more than 0.1 mm. A jog at the lowest speed moves
# the tool point by more than that in the 0.4 s the reads span.
N_READS = 3
READ_PAUSE_S = 0.2
JOINT_STILL_DEG = 0.01
TCP_STILL_MM = 0.1

# A tool length implied by the readings that differs from the twin's by more than
# this is the A4 rule of the cell guide (re-measure when off by more than 2 mm).
TOOL_LENGTH_HINT_MM = 2.0

EXPECTED_TOOL_NO = 1


class ReadOnlyHSEBackend(MotomanHSEBackend):
    """MotomanHSEBackend that sends nothing but READ_POSITION reads.

    Any other command, or the position command with a write service, raises
    PermissionError before a byte leaves the socket.
    """

    def _send_request(self, command: int, instance: int = 1, attribute: int = 0,
                      service: int = Service.GET_ATTRIBUTE_ALL, payload: bytes = b""):
        if (int(command), int(service)) not in READ_ONLY_EXCHANGES or payload:
            raise PermissionError(
                f"read-only measurement: HSE command 0x{int(command):02X} "
                f"service 0x{int(service):02X} refused, nothing sent")
        return super()._send_request(command, instance=instance, attribute=attribute,
                                     service=service, payload=payload)


def twin_tool_frames(joints_deg: Sequence[float], cell_config: Any
                     ) -> tuple[np.ndarray, np.ndarray]:
    """(T_base_tool0, T_base_tcp): the twin's flange (tool0) and tool point in the
    controller's BASE frame, from the joint angles in degrees."""
    pose = cell_config.robot.pose
    base_xyz = tuple(float(v) for v in pose.xyz_mm)
    base_rpy = tuple(float(v) for v in pose.rpy_deg)
    model = gp7_urdf(base_xyz_mm=base_xyz,
                     base_rpy_rad=tuple(float(np.deg2rad(d)) for d in base_rpy))
    T_world_tool0 = forward_kinematics_urdf(model, np.deg2rad(np.asarray(joints_deg, float)))
    g = cell_config.gripper
    T_tool0_tcp = make_homogeneous(g.tcp_offset_xyz_mm, g.tcp_offset_rpy_deg)
    T_base_tool0 = world_to_robot_base(T_world_tool0, base_xyz, base_rpy)
    return T_base_tool0, T_base_tool0 @ T_tool0_tcp


def _angle_deg(R: np.ndarray) -> float:
    c = (float(np.trace(R)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def compare(T_base_tool0: np.ndarray, T_base_tcp: np.ndarray, ctrl: dict) -> dict:
    """The twin's tool point against the controller's reading `ctrl`
    (read_cartesian: xyz_mm, rxyz_deg).

    d = twin minus controller, in the BASE frame; d_along_tool_mm, its component
    along the controller's tool Z; rot_err_deg, the angle of the rotation between
    the two orientations (pendant convention); z_axis_err_deg, the angle between
    the two tool Z axes; implied_tool_xyz_mm, the controller's tool point in its
    flange frame (the twin's flange times Rz180), i.e. the X, Y, Z of the
    controller's tool as the readings imply them.
    """
    p_ctrl = np.asarray(ctrl["xyz_mm"], float)
    R_ctrl = rpy_yaskawa_to_matrix(*ctrl["rxyz_deg"])
    p_twin = T_base_tcp[:3, 3]
    R_twin = T_base_tcp[:3, :3] @ RZ180
    d = p_twin - p_ctrl
    t_impl = (T_base_tool0[:3, :3] @ RZ180).T @ (p_ctrl - T_base_tool0[:3, 3])
    rx, ry, rz = matrix_to_rpy_yaskawa(R_twin)
    return {
        "twin_xyz_mm": tuple(float(v) for v in p_twin),
        "twin_rxyz_deg": (rx, ry, rz),
        "d_xyz_mm": tuple(float(v) for v in d),
        "d_mm": float(np.linalg.norm(d)),
        "d_along_tool_mm": float(d @ R_ctrl[:, 2]),
        "rot_err_deg": _angle_deg(R_ctrl.T @ R_twin),
        "z_axis_err_deg": float(np.degrees(np.arccos(np.clip(
            float(R_ctrl[:, 2] @ R_twin[:, 2]), -1.0, 1.0)))),
        "implied_tool_xyz_mm": tuple(float(v) for v in t_impl),
    }


def _spread(rows: list[Sequence[float]]) -> np.ndarray:
    a = np.asarray(rows, float)
    return a.max(axis=0) - a.min(axis=0)


def take_reading(backend: Any, n_reads: int = N_READS, pause_s: float = READ_PAUSE_S,
                 sleep=None) -> dict:
    """Read joints (instance 1) and the controller's tool point (instance 101)
    n_reads times, pause_s apart. Returns the last pair with the spreads over the
    reads and `still`, whether the robot stayed within JOINT_STILL_DEG and
    TCP_STILL_MM."""
    joints, carts = [], []
    for k in range(n_reads):
        if k:
            (sleep or time.sleep)(pause_s)
        joints.append(list(backend.Joints()))
        carts.append(backend.read_cartesian())
    joint_spread = float(_spread(joints).max())
    tcp_spread = float(np.linalg.norm(_spread([c["xyz_mm"] for c in carts])))
    tools = {c["tool_no"] for c in carts}
    return {
        "joints_deg": joints[-1],
        "ctrl": carts[-1],
        "joint_spread_deg": joint_spread,
        "tcp_spread_mm": tcp_spread,
        "tool_changed": len(tools) > 1,
        "still": (joint_spread <= JOINT_STILL_DEG and tcp_spread <= TCP_STILL_MM
                  and len(tools) == 1),
    }


def summarize(rows: Iterable[dict], twin_tool_xyz_mm: Sequence[float],
              expected_tool_no: int = EXPECTED_TOOL_NO) -> dict:
    """Summary over the rows read with the expected tool number. Each row needs
    tool_no, d_mm, d_along_tool_mm, rot_err_deg, z_axis_err_deg and
    implied_tool_x/y/z_mm (the CSV columns of the measurement script)."""
    rows = list(rows)
    use = [r for r in rows if int(r["tool_no"]) == expected_tool_no]
    other = sorted({int(r["tool_no"]) for r in rows if int(r["tool_no"]) != expected_tool_no})
    out: dict = {"n": len(use), "n_other_tool": len(rows) - len(use),
                 "other_tools": other, "expected_tool_no": expected_tool_no}
    if not use:
        return out
    d = np.array([float(r["d_mm"]) for r in use])
    tool = np.array([[float(r[f"implied_tool_{a}_mm"]) for a in "xyz"] for r in use])
    out.update({
        "mean_mm": float(d.mean()),
        "max_mm": float(d.max()),
        "mean_along_tool_mm": float(np.mean([float(r["d_along_tool_mm"]) for r in use])),
        "mean_rot_deg": float(np.mean([float(r["rot_err_deg"]) for r in use])),
        "max_rot_deg": float(np.max([float(r["rot_err_deg"]) for r in use])),
        "max_z_axis_deg": float(np.max([float(r["z_axis_err_deg"]) for r in use])),
        "ctrl_tool_mean_mm": tuple(float(v) for v in tool.mean(axis=0)),
        "ctrl_tool_sd_mm": tuple(float(v) for v in tool.std(axis=0, ddof=1))
        if len(use) > 1 else (0.0, 0.0, 0.0),
        "twin_tool_mm": tuple(float(v) for v in twin_tool_xyz_mm),
    })
    out["tool_length_gap_mm"] = out["ctrl_tool_mean_mm"][2] - float(twin_tool_xyz_mm[2])
    return out


def report_lines(s: dict) -> list[str]:
    """The summary as printed and saved, ASCII only. The first line is the Table 4
    entry (Tool point, twin vs. controller, mean / max)."""
    if not s.get("n"):
        lines = [f"No pose read with TOOL{s.get('expected_tool_no', EXPECTED_TOOL_NO):02d}: "
                 "nothing to report."]
    else:
        tm, tsd, tw = s["ctrl_tool_mean_mm"], s["ctrl_tool_sd_mm"], s["twin_tool_mm"]
        lines = [
            f"Tool point, twin vs. controller, mean / max: "
            f"{s['mean_mm']:.2f}/{s['max_mm']:.2f} mm  (n = {s['n']} poses, "
            f"TOOL{s['expected_tool_no']:02d})",
            f"  along the tool Z, mean (twin minus controller): {s['mean_along_tool_mm']:+.2f} mm",
            f"  orientation, mean / max: {s['mean_rot_deg']:.3f}/{s['max_rot_deg']:.3f} deg; "
            f"tool Z axis, max: {s['max_z_axis_deg']:.3f} deg",
            f"  controller tool point in its flange frame, mean (sd): "
            f"X {tm[0]:.2f} ({tsd[0]:.2f}), Y {tm[1]:.2f} ({tsd[1]:.2f}), "
            f"Z {tm[2]:.2f} ({tsd[2]:.2f}) mm; twin tool: X {tw[0]:.2f}, Y {tw[1]:.2f}, "
            f"Z {tw[2]:.2f} mm",
        ]
        if abs(s["tool_length_gap_mm"]) > TOOL_LENGTH_HINT_MM:
            lines.append(
                f"  CHECK TOOL{s['expected_tool_no']:02d}: the readings imply a tool Z of "
                f"{tm[2]:.1f} mm on the controller, the twin uses {tw[2]:.1f} mm "
                f"(gripper.tcp_offset_xyz_mm), {s['tool_length_gap_mm']:+.1f} mm apart. "
                f"Read X, Y, Z of tool {s['expected_tool_no']} on the pendant "
                f"(ROBOT > TOOL) and photograph the screen.")
    if s.get("n_other_tool"):
        lines.append(f"  {s['n_other_tool']} reading(s) with tool "
                     f"{', '.join(str(t) for t in s['other_tools'])} left out of the summary.")
    return lines
