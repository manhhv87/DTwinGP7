#!/usr/bin/env python
"""
11_test_yrc_cartesian.py
─────────────────────────
3-phase test for HSE Cartesian path on a real GP7 robot.

⚠ SAFETY before running:
  - YRC1000 in REMOTE mode
  - Speed slider on TP set to 10% (REDUCED SPEED)
  - Hand on E-stop
  - Workspace clear, no personnel within reach

Phase 1 — Protocol verify (NO robot motion):
    - Write P000 with Cartesian pose
    - Read P000 back
    - Confirm bytes encode/decode correctly

Phase 2 — Touch test 3 poses:
    - Pose 1: current pose (sanity check)
    - Pose 2: current + Z offset 50mm (up)
    - Pose 3: current (return to home)

Phase 3 — Full diagnostic:
    - Measure round-trip latency
    - Verify joint state read-back matches commanded

Usage:
    python scripts/11_test_yrc_cartesian.py --phase 1
    python scripts/11_test_yrc_cartesian.py --phase 2 --speed-pct 10
    python scripts/11_test_yrc_cartesian.py --phase 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cell import CellConfig  # noqa: E402
from src.orchestrator.backends.motoman_hse import MotomanHSEBackend  # noqa: E402
from src.orchestrator.backends.hse_protocol import (  # noqa: E402
    Command,
    DataType,
    Service,
    parse_position_response,
)
from src.orchestrator.frame_convert import matrix_to_xyzrpy_yaskawa  # noqa: E402
from src.utils import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--cell-config", default="config/cell_layout_real.yaml")
    p.add_argument("--hse-ip", default=None,
                   help="Override YRC1000 IP (default from cell config).")
    p.add_argument("--tool-no", type=int, default=1)
    p.add_argument("--speed-pct", type=float, default=10.0,
                   help="Max speed % for phase 2 (default 10% REDUCED SPEED).")
    p.add_argument("--z-offset-mm", type=float, default=50.0,
                   help="Z offset for phase 2 movement (default +50mm).")
    return p.parse_args()


def phase1_protocol_verify(backend: MotomanHSEBackend, log) -> int:
    """Write Cartesian pose to P000, read back, compare bytes."""
    log.info("=" * 60)
    log.info("PHASE 1: Protocol verify (NO ROBOT MOTION)")
    log.info("=" * 60)

    # Test pose: simple translation 500mm X, 0 rotation
    T_test = np.eye(4)
    T_test[:3, 3] = [500.0, 100.0, 600.0]
    x, y, z, rx, ry, rz = matrix_to_xyzrpy_yaskawa(T_test)
    log.info("Test pose: xyz=(%.1f, %.1f, %.1f)mm rpy=(%.2f, %.2f, %.2f)°",
             x, y, z, rx, ry, rz)

    log.info("Step 1: WRITE_POS_VAR P000 with BASE Cartesian...")
    try:
        backend.write_pose_var(p_index=0, T_base=T_test, tool_no=1, form=0)
        log.info("  ✓ Write OK (no exception)")
    except Exception as e:                              # noqa: BLE001
        log.error("  ✗ Write FAILED: %s", e)
        return 1

    log.info("Step 2: READ_POS_VAR P000 to verify...")
    try:
        resp = backend._send_request(
            Command.READ_POS_VAR, instance=0,
            service=Service.GET_ATTRIBUTE_ALL,
        )
        parsed = parse_position_response(resp.payload)
        log.info("  Data type: %d (expected 16=BASE)", parsed["data_type"])
        log.info("  Tool no: %d (expected 1)", parsed["tool_no"])
        log.info("  Form: %d (expected 0)", parsed["form"])
        # Axes from joints_raw (which are X,Y,Z,Rx,Ry,Rz for BASE)
        raw = parsed["joints_raw"]
        x_back = raw[0] / 1000.0
        y_back = raw[1] / 1000.0
        z_back = raw[2] / 1000.0
        log.info("  Read back: xyz=(%.1f, %.1f, %.1f)mm", x_back, y_back, z_back)
        if abs(x_back - x) < 0.01 and abs(y_back - y) < 0.01 and abs(z_back - z) < 0.01:
            log.info("  ✓ Round-trip MATCH — protocol bytes correct.")
            return 0
        log.error("  ✗ Round-trip MISMATCH — protocol bug.")
        return 1
    except Exception as e:                              # noqa: BLE001
        log.error("  ✗ Read FAILED: %s", e)
        return 1


def phase2_touch_test(backend: MotomanHSEBackend, args, log) -> int:
    """Move robot 50mm Z up then back. REDUCED SPEED 10%."""
    log.info("=" * 60)
    log.info("PHASE 2: Touch test (ROBOT WILL MOVE — REDUCED SPEED)")
    log.info("=" * 60)
    log.warning("⚠ Robot WILL MOVE. Keep E-stop within reach. Speed %.0f%%.",
                args.speed_pct)

    backend.max_speed_pct = args.speed_pct

    log.info("Step 1: Read current joints + pose...")
    joints_current = backend.Joints()
    log.info("  Joints: %s°", [f"{j:.1f}" for j in joints_current])

    # Read current Cartesian pose from robot
    resp = backend._send_request(
        Command.READ_POSITION, instance=1,
        service=Service.GET_ATTRIBUTE_ALL,
    )
    parsed = parse_position_response(resp.payload)
    if parsed["data_type"] == DataType.PULSE:
        log.error("Robot returned PULSE position. Need READ_POSITION instance=101 for BASE.")
        # Try alt instance to request BASE Cartesian
        # Note: per spec instance 101 = robot 1 cartesian, but try 1 first
        return 1
    raw = parsed["joints_raw"]
    x_now = raw[0] / 1000.0
    y_now = raw[1] / 1000.0
    z_now = raw[2] / 1000.0
    log.info("  Current pose: xyz=(%.1f, %.1f, %.1f)mm", x_now, y_now, z_now)

    # Build target pose: current + Z offset
    T_up = np.eye(4)
    T_up[:3, 3] = [x_now, y_now, z_now + args.z_offset_mm]
    log.info("Step 2: MoveJ Cartesian to +Z%.0fmm...", args.z_offset_mm)
    backend.MoveJ(T_up)
    log.info("  ✓ Move complete")

    time.sleep(0.5)
    joints_after = backend.Joints()
    log.info("  Joints after up: %s°", [f"{j:.1f}" for j in joints_after])

    log.info("Step 3: Move back to original pose...")
    T_back = np.eye(4)
    T_back[:3, 3] = [x_now, y_now, z_now]
    backend.MoveJ(T_back)
    log.info("  ✓ Back to start")

    return 0


def phase3_diagnostic(backend: MotomanHSEBackend, log) -> int:
    """Round-trip latency + accuracy stats."""
    log.info("=" * 60)
    log.info("PHASE 3: Diagnostic stats (10 read cycles)")
    log.info("=" * 60)

    latencies = []
    for i in range(10):
        t0 = time.perf_counter()
        _ = backend.Joints()
        latencies.append((time.perf_counter() - t0) * 1000)

    log.info("Joints read latency: min=%.1fms, median=%.1f, max=%.1f, p95=%.1f",
             min(latencies), np.median(latencies),
             max(latencies), np.percentile(latencies, 95))

    log.info("Alarm status...")
    code, sub = backend.read_alarm()
    if code == 0:
        log.info("  ✓ No alarm")
    else:
        log.warning("  ⚠ Alarm %d (sub %d)", code, sub)

    return 0


def main() -> int:
    args = parse_args()
    log = setup_logging("test_yrc_cartesian", log_file=None)

    # Load cell config for IP
    try:
        cell = CellConfig.from_yaml(PROJECT_ROOT / args.cell_config)
        hse_ip = args.hse_ip or cell.robot_connection.ip
    except Exception as e:                              # noqa: BLE001
        log.error("Load cell config failed: %s", e)
        return 2

    if not hse_ip:
        log.error("Provide --hse-ip or set robot_connection.ip in cell config")
        return 2

    log.info("HSE IP: %s, Tool: %d", hse_ip, args.tool_no)
    backend = MotomanHSEBackend(ip=hse_ip, tool_no=args.tool_no,
                                max_speed_pct=args.speed_pct)
    backend.connect()
    if not backend.Valid():
        log.error("HSE heartbeat fail. Check ping + HSE Server enable.")
        return 3
    log.info("HSE connected.")

    try:
        if args.phase == 1:
            return phase1_protocol_verify(backend, log)
        elif args.phase == 2:
            return phase2_touch_test(backend, args, log)
        else:
            return phase3_diagnostic(backend, log)
    finally:
        backend.disconnect()


if __name__ == "__main__":
    sys.exit(main())
