#!/usr/bin/env python
"""
30_measure_tcp_twin_vs_controller.py
────────────────────────────────────
E1: the tool point the twin computes against the one the controller reports.

READ-ONLY. The PC sends nothing but position reads (HSE 0x75): no motion, no
servo, no job, no I/O. The operator jogs the robot on the teach pendant (TEACH
mode, TOOL01 selected), releases the key and presses ENTER; at each pose the
script reads the joint angles (instance 1) and the controller's tool point
(instance 101, BASE frame, with the tool number) three times 0.2 s apart, keeps
the pose only if nothing moved, computes the twin's tool point from the same
joints (URDF chain + gripper.tcp_offset_xyz_mm of the cell config) and writes one
CSV row. At the end it prints the mean and the maximum distance, the Table 4
entry, and saves the summary next to the CSV.

A reading whose tool number is not --tool-no is written to the CSV with a
warning and left out of the summary: select the tool on the pendant and press
ENTER again at the same pose.

Usage (robot PC, controller in TEACH mode):
    python scripts/30_measure_tcp_twin_vs_controller.py
    python scripts/30_measure_tcp_twin_vs_controller.py --n-poses 20 --hse-ip 192.168.125.100
    python scripts/30_measure_tcp_twin_vs_controller.py --from-csv results/e1_tcp_20261107_0930.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cell import CellConfig  # noqa: E402
from src.orchestrator.tcp_agreement import (  # noqa: E402
    EXPECTED_TOOL_NO,
    ReadOnlyHSEBackend,
    compare,
    report_lines,
    summarize,
    take_reading,
    twin_tool_frames,
)

COLUMNS = (
    ["pose", "time", "note"]
    + [f"j{i}_deg" for i in range(1, 7)]
    + ["tool_no", "form"]
    + [f"ctrl_{a}" for a in ("x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg")]
    + [f"twin_{a}" for a in ("x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg")]
    + ["dx_mm", "dy_mm", "dz_mm", "d_mm", "d_along_tool_mm", "rot_err_deg",
       "z_axis_err_deg"]
    + [f"implied_tool_{a}_mm" for a in "xyz"]
    + [f"twin_tool_{a}_mm" for a in "xyz"]
    + ["joint_spread_deg", "tcp_spread_mm"]
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell-config", default="config/cell_layout_real.yaml")
    p.add_argument("--hse-ip", default=None,
                   help="Controller IP (default robot_connection.ip of the cell config).")
    p.add_argument("--n-poses", type=int, default=20,
                   help="Poses to record with the expected tool (default 20); q ends earlier.")
    p.add_argument("--tool-no", type=int, default=EXPECTED_TOOL_NO,
                   help="Tool number the controller must report (default 1 = TOOL01).")
    p.add_argument("--csv", default=None,
                   help="Output CSV (default results/e1_tcp_<date>_<time>.csv).")
    p.add_argument("--from-csv", default=None,
                   help="Recompute the summary of a recorded CSV; no robot.")
    return p.parse_args(argv)


def _resolve(path: str) -> Path:
    q = Path(path)
    return q if q.is_absolute() else PROJECT_ROOT / q


def _row(pose: int, note: str, reading: dict, cmp: dict, twin_tool) -> dict:
    ctrl = reading["ctrl"]
    r = {"pose": pose, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": note,
         "tool_no": ctrl["tool_no"], "form": ctrl["form"]}
    for i, q in enumerate(reading["joints_deg"], 1):
        r[f"j{i}_deg"] = round(float(q), 4)
    for a, v in zip(("x_mm", "y_mm", "z_mm"), ctrl["xyz_mm"]):
        r[f"ctrl_{a}"] = round(v, 3)
    for a, v in zip(("rx_deg", "ry_deg", "rz_deg"), ctrl["rxyz_deg"]):
        r[f"ctrl_{a}"] = round(v, 4)
    for a, v in zip(("x_mm", "y_mm", "z_mm"), cmp["twin_xyz_mm"]):
        r[f"twin_{a}"] = round(v, 3)
    for a, v in zip(("rx_deg", "ry_deg", "rz_deg"), cmp["twin_rxyz_deg"]):
        r[f"twin_{a}"] = round(v, 4)
    for a, v in zip(("dx_mm", "dy_mm", "dz_mm"), cmp["d_xyz_mm"]):
        r[a] = round(v, 3)
    r["d_mm"] = round(cmp["d_mm"], 3)
    r["d_along_tool_mm"] = round(cmp["d_along_tool_mm"], 3)
    r["rot_err_deg"] = round(cmp["rot_err_deg"], 4)
    r["z_axis_err_deg"] = round(cmp["z_axis_err_deg"], 4)
    for a, v in zip("xyz", cmp["implied_tool_xyz_mm"]):
        r[f"implied_tool_{a}_mm"] = round(v, 3)
    for a, v in zip("xyz", twin_tool):
        r[f"twin_tool_{a}_mm"] = round(float(v), 3)
    r["joint_spread_deg"] = round(reading["joint_spread_deg"], 5)
    r["tcp_spread_mm"] = round(reading["tcp_spread_mm"], 4)
    return r


def _finish(rows: list[dict], twin_tool, tool_no: int, summary_path: Path | None) -> int:
    lines = report_lines(summarize(rows, twin_tool, expected_tool_no=tool_no))
    print("\n== Summary ==")
    for line in lines:
        print(line)
    if summary_path is not None:
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nSummary -> {summary_path}")
    return 0


def from_csv(path: Path, tool_no: int) -> int:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"{path}: no rows.", file=sys.stderr)
        return 1
    twin_tool = [float(rows[0][f"twin_tool_{a}_mm"]) for a in "xyz"]
    return _finish(rows, twin_tool, tool_no, None)


def main(argv=None, backend: Any = None, input_fn: Callable[[str], str] = input) -> int:
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                       # noqa: BLE001
        pass
    if args.from_csv:
        return from_csv(_resolve(args.from_csv), args.tool_no)

    cell = CellConfig.from_yaml(_resolve(args.cell_config))
    if cell.gripper is None:
        print("The cell config has no gripper (tcp_offset_xyz_mm).", file=sys.stderr)
        return 2
    twin_tool = tuple(float(v) for v in cell.gripper.tcp_offset_xyz_mm)
    out = _resolve(args.csv or f"results/e1_tcp_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    own = backend is None
    if own:
        ip = args.hse_ip or cell.robot_connection.ip
        if not ip:
            print("Controller IP required (--hse-ip or robot_connection.ip).", file=sys.stderr)
            return 2
        backend = ReadOnlyHSEBackend(ip=ip)
        backend.connect()
    try:
        try:
            first = backend.read_cartesian()
        except Exception as e:                              # noqa: BLE001
            print(f"No position reply from the controller: {e}", file=sys.stderr)
            return 1
        print("== Tool point, twin vs. controller (E1), READ-ONLY ==")
        print(f"   twin tool (gripper.tcp_offset_xyz_mm): {list(twin_tool)} mm, "
              f"rpy {list(cell.gripper.tcp_offset_rpy_deg)} deg")
        print(f"   controller reports tool {first['tool_no']}; expected {args.tool_no}")
        if first["tool_no"] != args.tool_no:
            print(f"   TOOL MISMATCH: select TOOL{args.tool_no:02d} on the pendant before "
                  "the first pose.")
        print(f"   {args.n_poses} poses across the pick and place regions, several heights "
              "and wrist orientations.\n   At each: jog, release the key, press ENTER "
              "(text before ENTER is saved as a note; q ends).\n")

        rows: list[dict] = []
        n_ok = 0
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            f.flush()
            while n_ok < args.n_poses:
                ans = input_fn(f"[pose {n_ok + 1}/{args.n_poses}] ENTER = read, q = finish: ")
                note = ans.strip()
                if note.lower() == "q":
                    break
                try:
                    reading = take_reading(backend)
                except Exception as e:                      # noqa: BLE001
                    print(f"   read failed ({e}); press ENTER again.")
                    continue
                if not reading["still"]:
                    print(f"   NOT STILL: joints moved {reading['joint_spread_deg']:.4f} deg, "
                          f"tool point {reading['tcp_spread_mm']:.3f} mm over the reads"
                          + (", tool number changed" if reading["tool_changed"] else "")
                          + "; hold still and press ENTER again.")
                    continue
                T0, Tt = twin_tool_frames(reading["joints_deg"], cell)
                cmp = compare(T0, Tt, reading["ctrl"])
                row = _row(len(rows) + 1, note, reading, cmp, twin_tool)
                w.writerow(row)
                f.flush()
                rows.append(row)
                c, d = reading["ctrl"], cmp["d_xyz_mm"]
                print(f"   tool {c['tool_no']}  controller ({c['xyz_mm'][0]:.2f}, "
                      f"{c['xyz_mm'][1]:.2f}, {c['xyz_mm'][2]:.2f})  twin "
                      f"({cmp['twin_xyz_mm'][0]:.2f}, {cmp['twin_xyz_mm'][1]:.2f}, "
                      f"{cmp['twin_xyz_mm'][2]:.2f}) mm")
                print(f"   d = {cmp['d_mm']:.2f} mm (dx {d[0]:+.2f}, dy {d[1]:+.2f}, "
                      f"dz {d[2]:+.2f}; along tool {cmp['d_along_tool_mm']:+.2f})  "
                      f"orientation {cmp['rot_err_deg']:.3f} deg")
                if c["tool_no"] != args.tool_no:
                    print(f"   TOOL MISMATCH: the controller reports tool {c['tool_no']}, "
                          f"expected {args.tool_no}. Select TOOL{args.tool_no:02d} on the "
                          "pendant and press ENTER again at this pose; this row stays in "
                          "the CSV, out of the summary.")
                    continue
                n_ok += 1
    finally:
        if own:
            backend.disconnect()

    print(f"\nCSV -> {out}")
    return _finish(rows, twin_tool, args.tool_no, out.with_name(out.stem + "_tomtat.txt"))


if __name__ == "__main__":
    sys.exit(main())
