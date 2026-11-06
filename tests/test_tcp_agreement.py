"""The tool point the twin computes against the controller's (E1, script 30).

A scripted controller behind the socket answers READ_POSITION (0x75): instance 1
with the pulses of its joints, instance 101 with the tool point of its own tool
(in the pendant's TOOL convention, BASE frame, um and 0.0001 deg). It records
every packet, so the tests also check that nothing but that read is ever sent.
"""
from __future__ import annotations

import csv
import importlib.util
import socket
import struct
from pathlib import Path

import numpy as np
import pytest

from src.cell import CellConfig
from src.orchestrator.backends.hse_protocol import (
    ACK_RESPONSE, GP7_PULSE_PER_DEG, HSE_HEADER_SIZE, HSE_IDENTIFIER, HSE_RESERVE1,
    HSE_RESERVE2, Command, HSEDecodeError, Service, parse_cartesian_position_response,
)
from src.orchestrator.frame_convert import matrix_to_rpy_yaskawa
from src.orchestrator import tcp_agreement as ta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CELL = CellConfig.from_yaml(PROJECT_ROOT / "config" / "cell_layout_real.yaml")

# Two poses over the table, wrist pointing down, one tilted and turned.
POSE_A = [0.0, 10.0, -20.0, 0.0, -60.0, 0.0]
POSE_B = [25.0, 30.0, 5.0, 15.0, -70.0, 40.0]


def _reply(payload: bytes, request_id: int) -> bytes:
    header = struct.pack("<4s H H B B B B I 8s B B B B H H", HSE_IDENTIFIER,
                         HSE_HEADER_SIZE, len(payload), HSE_RESERVE1, 0x01,
                         ACK_RESPONSE, request_id, 0, HSE_RESERVE2,
                         0x01, 0, 0, 0, 0, 0)
    return header + payload


class FakeController:
    """Socket stand-in for a YRC1000 holding `joints_deg`, with its own tool
    `tool_xyz_mm` (flange frame, no rotation) selected as number `tool_no`.
    `drift_deg` is added to joint 1 after every pulse read (a robot still moving)."""

    def __init__(self, joints_deg, tool_xyz_mm=(0.0, 0.0, 178.0), tool_no=1, drift_deg=0.0):
        self.joints_deg = list(joints_deg)
        self.tool_xyz_mm = np.asarray(tool_xyz_mm, float)
        self.tool_no, self.drift_deg = tool_no, drift_deg
        self.sent: list[bytes] = []
        self.blocking = True

    def pulses(self):
        return [int(round(q * r)) for q, r in zip(self.joints_deg, GP7_PULSE_PER_DEG)]

    def sendto(self, packet, addr):
        self.sent.append(bytes(packet))

    def setblocking(self, flag):
        self.blocking = flag

    def settimeout(self, s):
        pass

    def close(self):
        pass

    def recvfrom(self, n):
        if not self.blocking:
            raise BlockingIOError
        packet = self.sent[-1]
        command, instance = struct.unpack_from("<HH", packet, 24)
        assert command == Command.READ_POSITION
        if instance == 1:
            payload = struct.pack("<5I 8i", 0, 0, self.tool_no, 0, 0, *self.pulses(), 0, 0)
            self.joints_deg[0] += self.drift_deg
        elif instance == 101:
            # The controller's kinematics are the twin's chain (tested against
            # RoboDK elsewhere) at the joints its encoders hold.
            q = [p / r for p, r in zip(self.pulses(), GP7_PULSE_PER_DEG)]
            T0, _ = ta.twin_tool_frames(q, CELL)
            R_flange = T0[:3, :3] @ ta.RZ180
            p = T0[:3, 3] + R_flange @ self.tool_xyz_mm
            rx, ry, rz = matrix_to_rpy_yaskawa(R_flange)
            payload = struct.pack(
                "<5I 8i", 16, 0, self.tool_no, 0, 0,
                *[int(round(v * 1000.0)) for v in p],
                *[int(round(a * 10000.0)) for a in (rx, ry, rz)], 0, 0)
        else:
            raise AssertionError(f"unexpected instance {instance}")
        return _reply(payload, packet[11]), ("192.168.125.100", 10040)


def _backend(fake):
    b = ta.ReadOnlyHSEBackend(ip="192.168.125.100", timeout_s=0.5)
    b._sock = fake
    return b


def _script():
    path = PROJECT_ROOT / "scripts" / "30_measure_tcp_twin_vs_controller.py"
    spec = importlib.util.spec_from_file_location("measure_tcp_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _no_sleep(monkeypatch):
    monkeypatch.setattr(ta.time, "sleep", lambda s: None)


# ───── packet format and units ─────

def test_cartesian_read_packet_is_0x75_instance_101_get_all():
    fake = FakeController(POSE_A)
    _backend(fake).read_cartesian()
    pkt = fake.sent[-1]
    assert len(pkt) == HSE_HEADER_SIZE                       # no data part
    assert pkt[:4] == b"YERC"
    assert struct.unpack_from("<HH", pkt, 24) == (0x75, 101)
    assert pkt[28] == 0 and pkt[29] == Service.GET_ATTRIBUTE_ALL == 0x01


def test_cartesian_units_um_and_ten_thousandth_degree():
    payload = struct.pack("<5I 8i", 16, 5, 1, 0, 0,
                          612345, -98765, 401002, 1799999, -12345, 900000, 0, 0)
    c = parse_cartesian_position_response(payload)
    assert c["xyz_mm"] == pytest.approx((612.345, -98.765, 401.002))
    assert c["rxyz_deg"] == pytest.approx((179.9999, -1.2345, 90.0))
    assert (c["tool_no"], c["form"], c["data_type"]) == (1, 5, 16)


def test_cartesian_reply_with_pulse_data_type_is_refused():
    payload = struct.pack("<5I 8i", 0, 0, 1, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0)
    with pytest.raises(HSEDecodeError, match="data type 0"):
        parse_cartesian_position_response(payload)


def test_six_axis_reply_without_external_axis_words_parses():
    payload = struct.pack("<5I 6i", 16, 0, 1, 0, 0, 1000, 2000, 3000, 10000, 20000, 30000)
    c = parse_cartesian_position_response(payload)
    assert c["xyz_mm"] == pytest.approx((1.0, 2.0, 3.0))
    assert c["rxyz_deg"] == pytest.approx((1.0, 2.0, 3.0))


# ───── twin kinematics ─────

def test_twin_tool_point_at_zero_joints_from_the_gp7_dimensions():
    # URDF chain: L 40 mm ahead of S, U 445 above L, R 440 ahead and 40 above U,
    # flange 80 ahead of the wrist; tool0 Z points forward at zero joints.
    T0, Tt = ta.twin_tool_frames([0.0] * 6, CELL)
    assert T0[:3, 3] == pytest.approx([560.0, 0.0, 485.0], abs=1e-9)
    assert T0[:3, 2] == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)
    assert Tt[:3, 3] == pytest.approx([560.0 + 178.0, 0.0, 485.0], abs=1e-9)


def test_same_tool_on_both_sides_agrees_to_the_unit_rounding():
    fake = FakeController(POSE_B)
    r = ta.take_reading(_backend(fake), sleep=lambda s: None)
    T0, Tt = ta.twin_tool_frames(r["joints_deg"], CELL)
    c = ta.compare(T0, Tt, r["ctrl"])
    assert c["d_mm"] < 0.002                                   # 1 um per axis
    assert c["rot_err_deg"] < 0.001 and c["z_axis_err_deg"] < 0.001
    assert c["implied_tool_xyz_mm"] == pytest.approx((0.0, 0.0, 178.0), abs=0.002)


def test_short_controller_tool_shows_along_the_tool_axis():
    fake = FakeController(POSE_A, tool_xyz_mm=(0.0, 0.0, 154.1))
    r = ta.take_reading(_backend(fake), sleep=lambda s: None)
    T0, Tt = ta.twin_tool_frames(r["joints_deg"], CELL)
    c = ta.compare(T0, Tt, r["ctrl"])
    assert c["d_mm"] == pytest.approx(23.9, abs=0.002)
    assert c["d_along_tool_mm"] == pytest.approx(23.9, abs=0.002)
    assert c["implied_tool_xyz_mm"][2] == pytest.approx(154.1, abs=0.002)


def test_lateral_tool_offset_is_given_in_the_pendant_tool_axes():
    fake = FakeController(POSE_B, tool_xyz_mm=(3.0, -2.0, 178.0))
    r = ta.take_reading(_backend(fake), sleep=lambda s: None)
    T0, Tt = ta.twin_tool_frames(r["joints_deg"], CELL)
    c = ta.compare(T0, Tt, r["ctrl"])
    assert c["implied_tool_xyz_mm"] == pytest.approx((3.0, -2.0, 178.0), abs=0.002)
    assert c["d_mm"] == pytest.approx(np.hypot(3.0, 2.0), abs=0.002)


# ───── stillness, read-only guard ─────

def test_robot_moving_between_reads_is_not_still():
    fake = FakeController(POSE_A, drift_deg=0.02)
    r = ta.take_reading(_backend(fake), sleep=lambda s: None)
    assert not r["still"] and r["joint_spread_deg"] >= 0.04 - 1e-3


def test_still_robot_is_read_three_times_each_way():
    fake = FakeController(POSE_A)
    r = ta.take_reading(_backend(fake), sleep=lambda s: None)
    assert r["still"] and r["joint_spread_deg"] == 0.0 and r["tcp_spread_mm"] == 0.0
    instances = [struct.unpack_from("<H", p, 26)[0] for p in fake.sent]
    assert instances == [1, 101] * ta.N_READS


@pytest.mark.parametrize("call", [
    lambda b: b.job_start(),
    lambda b: b.servo_on(),
    lambda b: b.servo_off(),
    lambda b: b.move_joints([0.0] * 6),
    lambda b: b.setDO(1, 1),
    lambda b: b.job_select("DTWINRUN"),
    lambda b: b.write_pose_var(p_index=0, T_base=np.eye(4), tool_no=1, form=0),
    lambda b: b.Stop(),
    lambda b: b.Valid(),
])
def test_read_only_backend_sends_nothing_else(call):
    fake = FakeController(POSE_A)
    b = _backend(fake)
    try:
        call(b)
    except Exception:                                       # noqa: BLE001
        pass
    assert fake.sent == []


def test_read_only_guard_refuses_a_write_service_of_the_read_command():
    fake = FakeController(POSE_A)
    with pytest.raises(PermissionError):
        _backend(fake)._send_request(Command.READ_POSITION, instance=101,
                                     service=Service.SET_ATTRIBUTE_ALL, payload=b"\x00" * 4)
    assert fake.sent == []


# ───── summary ─────

def _rows(ds, tool_no=1, tool_z=178.0):
    return [{"tool_no": tool_no, "d_mm": d, "d_along_tool_mm": d, "rot_err_deg": 0.01,
             "z_axis_err_deg": 0.005, "implied_tool_x_mm": 0.0, "implied_tool_y_mm": 0.0,
             "implied_tool_z_mm": tool_z} for d in ds]


def test_summary_mean_max_in_table_format():
    s = ta.summarize(_rows([0.2, 0.4, 0.9]) + _rows([30.0], tool_no=0), (0, 0, 178))
    assert (s["n"], s["n_other_tool"]) == (3, 1)
    lines = ta.report_lines(s)
    assert lines[0].startswith("Tool point, twin vs. controller, mean / max: 0.50/0.90 mm")
    assert "n = 3 poses, TOOL01" in lines[0]
    assert not any("CHECK TOOL01" in line for line in lines)
    assert "1 reading(s) with tool 0 left out" in lines[-1]


def test_summary_hints_at_tool01_when_the_implied_length_is_off():
    s = ta.summarize(_rows([23.9, 23.8], tool_z=154.1), (0, 0, 178))
    text = "\n".join(ta.report_lines(s))
    assert "CHECK TOOL01" in text and "154.1 mm" in text and "-23.9 mm apart" in text


def test_summary_without_any_expected_tool_reading():
    s = ta.summarize(_rows([1.0], tool_no=0), (0, 0, 178))
    assert s["n"] == 0
    assert ta.report_lines(s)[0].startswith("No pose read with TOOL01")


# ───── the script end to end ─────

def test_script_records_poses_and_writes_csv_and_summary(tmp_path, monkeypatch):
    _no_sleep(monkeypatch)
    mod = _script()
    fake = FakeController(POSE_A, tool_xyz_mm=(0.0, 0.0, 154.1))
    answers = iter(["", "pick z high", "q"])
    poses = iter([POSE_B])

    def ask(prompt):
        a = next(answers)
        if a and a != "q":
            fake.joints_deg = list(next(poses))
        return a

    out = tmp_path / "e1_tcp.csv"
    rc = mod.main(["--csv", str(out), "--n-poses", "5"], backend=_backend(fake), input_fn=ask)
    assert rc == 0
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [r["note"] for r in rows] == ["", "pick z high"]
    assert list(rows[0].keys()) == mod.COLUMNS
    assert all(float(r["d_mm"]) == pytest.approx(23.9, abs=0.01) for r in rows)
    summary = (tmp_path / "e1_tcp_tomtat.txt").read_text(encoding="utf-8")
    assert summary.startswith("Tool point, twin vs. controller, mean / max: 23.90/23.90 mm")
    assert "CHECK TOOL01" in summary
    commands = {(struct.unpack_from("<H", p, 24)[0], p[29]) for p in fake.sent}
    assert commands == {(0x75, 0x01)}


def test_script_stops_after_n_poses_and_skips_a_moving_robot(tmp_path, monkeypatch, capsys):
    _no_sleep(monkeypatch)
    mod = _script()
    fake = FakeController(POSE_A, drift_deg=0.05)
    calls = {"n": 0}

    def ask(prompt):
        calls["n"] += 1
        if calls["n"] == 2:
            fake.drift_deg = 0.0                                 # operator let go
        return ""

    out = tmp_path / "t.csv"
    assert mod.main(["--csv", str(out), "--n-poses", "2"], backend=_backend(fake),
                    input_fn=ask) == 0
    assert "NOT STILL" in capsys.readouterr().out
    assert len(list(csv.DictReader(out.open(encoding="utf-8")))) == 2
    assert calls["n"] == 3


def test_script_keeps_a_wrong_tool_row_out_of_the_count(tmp_path, monkeypatch, capsys):
    _no_sleep(monkeypatch)
    mod = _script()
    fake = FakeController(POSE_A, tool_no=0)
    calls = {"n": 0}

    def ask(prompt):
        calls["n"] += 1
        if calls["n"] == 2:
            fake.tool_no = 1                                     # TOOL01 selected
        return ""

    out = tmp_path / "t.csv"
    assert mod.main(["--csv", str(out), "--n-poses", "1"], backend=_backend(fake),
                    input_fn=ask) == 0
    printed = capsys.readouterr().out
    assert "TOOL MISMATCH" in printed and "1 reading(s) with tool 0 left out" in printed
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [r["tool_no"] for r in rows] == ["0", "1"]


def test_from_csv_recomputes_the_same_summary(tmp_path, monkeypatch, capsys):
    _no_sleep(monkeypatch)
    mod = _script()
    fake = FakeController(POSE_B, tool_xyz_mm=(0.0, 0.0, 177.0))
    out = tmp_path / "t.csv"
    answers = iter(["", "", "q"])
    mod.main(["--csv", str(out)], backend=_backend(fake), input_fn=lambda p: next(answers))
    first = (tmp_path / "t_tomtat.txt").read_text(encoding="utf-8").splitlines()[0]
    capsys.readouterr()
    assert mod.main(["--from-csv", str(out)]) == 0
    assert first in capsys.readouterr().out


def test_no_reply_from_the_controller_ends_with_an_error(tmp_path, capsys):
    mod = _script()

    class Silent(FakeController):
        def recvfrom(self, n):
            raise socket.timeout

    out = tmp_path / "t.csv"
    assert mod.main(["--csv", str(out)], backend=_backend(Silent(POSE_A)),
                    input_fn=lambda p: "") == 1
    assert "No position reply" in capsys.readouterr().err
