"""hse_status.py — read & decode the YRC1000 controller status over HSE.

Run this ON THE ROBOT PC (same subnet as the YRC1000). It reads the controller
status (HSE command 0x72) and decodes EXACTLY what mode / cycle / hold / servo /
alarm state the controller is in — so you can see why a job START (0x86) is
rejected (e.g. 0x2080 "incorrect mode" / 0x2100 "command remote not set")
instead of guessing from the pendant.

This is READ-ONLY: it does NOT move the robot, change mode, or start anything.

Usage (default IP = the verified YRC1000):
    python tools/hse_status.py
    python tools/hse_status.py 192.168.125.100

Bit layout (Yaskawa HSE "Controller Status Reading", 0x72):
  Data1: 0 step, 1 1-cycle, 2 auto, 3 running, 4 safeguard/speed, 5 teach,
         6 play, 7 remote
  Data2: 1 hold(PP), 2 hold(external), 3 hold(command), 4 alarming,
         5 error, 6 servo on
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestrator.backends.motoman_hse import MotomanHSEBackend  # noqa: E402
from src.orchestrator.backends.hse_protocol import Command, Service  # noqa: E402

IP = sys.argv[1].strip() if len(sys.argv) > 1 else "192.168.125.100"


def _bit(v: int, n: int) -> bool:
    return bool(v & (1 << n))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                   # noqa: BLE001
        pass

    print(f"== HSE status read -> {IP} ==\n")
    be = MotomanHSEBackend(ip=IP, timeout_s=2.0)
    try:
        be.connect()
    except Exception as e:                              # noqa: BLE001
        print(f"FAIL connect: {e!r}")
        return 1
    try:
        resp = be._send_request(
            Command.READ_STATUS, instance=1,
            service=Service.GET_ATTRIBUTE_ALL,
        )
    except Exception as e:                              # noqa: BLE001
        print(f"FAIL READ_STATUS: {e!r}\n"
              f"=> HSE Server function off / UDP 10040 blocked / wrong IP?")
        return 1
    finally:
        be.disconnect()

    p = resp.payload or b""
    print(f"raw payload ({len(p)} bytes): {p.hex()}\n")
    if len(p) < 1:
        print("Empty payload — controller returned no status data.")
        return 1

    d1 = p[0]
    d2 = p[4] if len(p) >= 5 else 0

    step, cyc1, auto, running, safeg, teach, play, remote = (_bit(d1, i) for i in range(8))
    _r0, hold_pp, hold_ext, hold_cmd, alarming, erroring, servo, _r7 = (_bit(d2, i) for i in range(8))

    mode = "TEACH" if teach else ("PLAY" if play else ("REMOTE" if remote else "UNKNOWN"))
    cycle = "STEP" if step else ("1-CYCLE" if cyc1 else ("AUTO/CONTINUOUS" if auto else "UNKNOWN"))

    print(f"Data1 = 0x{d1:02X}    Data2 = 0x{d2:02X}\n")
    print(f"  MODE       : {mode}        (teach={int(teach)} play={int(play)} remote={int(remote)})")
    print(f"  CYCLE      : {cycle}       (step={int(step)} 1cycle={int(cyc1)} auto={int(auto)})")
    print(f"  RUNNING    : {int(running)}")
    print(f"  SAFEGUARD  : {int(safeg)}")
    print(f"  SERVO ON   : {int(servo)}")
    print(f"  ALARM      : {int(alarming)}     ERROR: {int(erroring)}")
    print(f"  HOLD       : pp={int(hold_pp)} external={int(hold_ext)} command={int(hold_cmd)}")

    print("\n=== Why START (0x86) may be refused ===")
    issues = []
    if not remote:
        issues.append(f"  ✗ NOT in REMOTE mode (controller reports {mode}). The mode KEY "
                      f"switch must be physically in REMOTE — editing #87015 / CMD REMOTE "
                      f"SEL alone does NOT change the reported mode if the key is on "
                      f"TEACH/PLAY. This is the '0x2080 incorrect mode' cause.")
    if not servo:
        issues.append("  ✗ Servo is OFF — turn servo ON.")
    if alarming or erroring:
        issues.append("  ✗ An alarm/error is active — clear it on the pendant (→ 0x2060).")
    if hold_pp or hold_ext or hold_cmd:
        issues.append(f"  ✗ HOLD active (pp={int(hold_pp)} ext={int(hold_ext)} "
                      f"cmd={int(hold_cmd)}) — release HOLD.")
    if running:
        issues.append("  • Already RUNNING a job — START would be ignored/refused.")
    if not issues:
        print("  ✓ Mode=REMOTE, servo ON, no alarm, no hold → START preconditions look OK.\n"
              "    If START is STILL refused, the remaining suspect is the controller's\n"
              "    remote-command FUNCTION config (which interface is the remote source) —\n"
              "    capture the exact added_status and report it.")
    else:
        print("\n".join(issues))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
