"""hse_run_job.py — step-by-step Run-on-Robot HSE sequence with diagnostics.

Replicates the app's start sequence ONE STEP AT A TIME (servo on -> job select
-> job start) and reads the controller status between steps, so you can see
EXACTLY which command the controller rejects (e.g. with status 0x1F /
added_status 0x2080 "incorrect mode") and what mode/servo state it is in at
each point. This isolates whether the 0x2080 comes from JOB_SELECT or START.

SAFETY: by default it STOPS before the actual START (NO motion — it only reads
status, turns servo on, and selects the job). Add --start to ALSO send the
start command — THE ROBOT WILL MOVE. Keep speed <=10%, area clear, hand on
E-stop.

Usage:
    python tools/hse_run_job.py MINHTEST              # safe: up to JOB_SELECT, no motion
    python tools/hse_run_job.py MINHTEST 192.168.125.100
    python tools/hse_run_job.py MINHTEST --start      # also START (robot moves!)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestrator.backends.motoman_hse import MotomanHSEBackend  # noqa: E402
from src.orchestrator.backends.hse_protocol import (  # noqa: E402
    Command, Service, HSEResponseError,
)

_args = [a for a in sys.argv[1:]]
DO_START = "--start" in _args
SKIP_SELECT = "--skip-select" in _args
_args = [a for a in _args if a not in ("--start", "--skip-select")]
JOB = _args[0] if _args else "MINHTEST"
IP = _args[1].strip() if len(_args) > 1 else "192.168.125.100"


def _bit(v: int, n: int) -> bool:
    return bool(v & (1 << n))


def read_status(be: MotomanHSEBackend) -> str:
    """Return a compact one-line decode of HSE status (0x72)."""
    try:
        resp = be._send_request(
            Command.READ_STATUS, instance=1, service=Service.GET_ATTRIBUTE_ALL)
    except Exception as e:                              # noqa: BLE001
        return f"<status read failed: {e!r}>"
    p = resp.payload or b""
    if not p:
        return "<empty status payload>"
    d1 = p[0]
    d2 = p[4] if len(p) >= 5 else 0
    step, cyc1, auto, running, safeg, teach, play, remote = (_bit(d1, i) for i in range(8))
    _r0, hpp, hext, hcmd, alarming, erroring, servo, _r7 = (_bit(d2, i) for i in range(8))
    mode = "REMOTE" if remote else ("PLAY" if play else ("TEACH" if teach else "?"))
    cyc = "STEP" if step else ("1CYCLE" if cyc1 else ("AUTO" if auto else "?"))
    hold = "+".join(n for n, b in (("pp", hpp), ("ext", hext), ("cmd", hcmd)) if b) or "none"
    return (f"D1=0x{d1:02X} D2=0x{d2:02X} | mode={mode} cycle={cyc} "
            f"running={int(running)} servo={int(servo)} alarm={int(alarming)} "
            f"error={int(erroring)} hold={hold}")


def step(be: MotomanHSEBackend, label: str, fn) -> bool:
    """Run one HSE command, printing OK or the exact rejection code."""
    try:
        fn()
        print(f"  {label:<22} -> OK")
        ok = True
    except HSEResponseError as e:
        print(f"  {label:<22} -> REJECTED  status=0x{e.status:02X} "
              f"added_status=0x{e.added_status:04X}")
        ok = False
    except Exception as e:                             # noqa: BLE001
        print(f"  {label:<22} -> ERROR {e!r}")
        ok = False
    print(f"     status after: {read_status(be)}")
    return ok


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                  # noqa: BLE001
        pass

    print(f"== HSE step-by-step run -> {IP}  job='{JOB}'  "
          f"{'(WILL START — robot moves!)' if DO_START else '(no START — safe)'} ==\n")
    be = MotomanHSEBackend(ip=IP, timeout_s=3.0)
    try:
        be.connect()
    except Exception as e:                             # noqa: BLE001
        print(f"FAIL connect: {e!r}")
        return 1

    try:
        print(f"[0] initial status: {read_status(be)}\n")

        print("[1] servo ON (0x83 inst2 data1)")
        step(be, "servo_on", be.servo_on)
        time.sleep(1.0)
        print()

        if SKIP_SELECT:
            print("[2] JOB_SELECT — SKIPPED (--skip-select): the job must already be\n"
                  f"    selected on the pendant (open '{JOB}', cursor at the top).")
            sel_ok = True
            print()
        else:
            print(f"[2] JOB_SELECT '{JOB}' (0x87 inst1 attr0 svc0x02)")
            sel_ok = step(be, f"job_select {JOB}", lambda: be.job_select(JOB))
            print()

        if not sel_ok:
            print(">>> JOB_SELECT is the command being rejected. The 0x2080/added_status\n"
                  "    above is for SELECTING the job in this mode — not START.")
        elif DO_START:
            print("[3] START (0x86 inst1 attr1 svc0x10 data1)  *** robot may MOVE ***")
            step(be, "job_start", be.job_start)
            print()
        else:
            print("[3] START — SKIPPED (no --start). Re-run with --start to test START\n"
                  "    (robot will move).")
    finally:
        be.disconnect()

    print("\nReport the [1]/[2]/[3] result lines + the 'status after' lines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
