"""hse_rtt.py — measure HSE round-trip time to the YRC1000 (READ-ONLY).

Times the joint-position read (HSE command 0x75) in a tight loop and reports the
distribution. This is the platform latency number the paper's E1 table asks for:
the rate at which the digital twin can poll the real robot's state.

This is READ-ONLY: it does NOT move the robot, change mode, start a job or touch
any output. Safe to run with servos off, in TEACH mode, while the robot is idle.

Usage (run on the robot PC, same subnet as the controller):
    python tools/hse_rtt.py 192.168.1.100
    python tools/hse_rtt.py 192.168.1.100 --n 1000 --csv results/e1_hse_rtt.csv

Read the numbers as:
  - p50 / p95 / p99  round-trip of one state read, milliseconds
  - achieved rate    reads per second with no sleep between them, i.e. the ceiling
                     on the mirror's poll rate. The experiment runs at 10 Hz, so a
                     p95 above 100 ms means the telemetry loop cannot keep up.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

# Allow running from the repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestrator.backends.motoman_hse import MotomanHSEBackend  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ip", nargs="?", default="192.168.1.100",
                   help="YRC1000 IP address. Default 192.168.1.100 (cell config).")
    p.add_argument("--n", type=int, default=1000, help="Number of reads. Default 1000.")
    p.add_argument("--warmup", type=int, default=20,
                   help="Discarded reads before timing starts. Default 20.")
    p.add_argument("--csv", default=None,
                   help="Write every sample here (one row per read) for a histogram.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                       # noqa: BLE001
        pass

    backend = MotomanHSEBackend(ip=args.ip)
    backend.connect()
    try:
        if not backend.Valid():
            print(f"No HSE heartbeat from {args.ip}. Check the IP, the subnet, and that "
                  "the High-Speed Ethernet Server function is enabled.", file=sys.stderr)
            return 1
        print(f"== HSE round-trip, joint read -> {args.ip} ==")
        print(f"   warmup {args.warmup}, samples {args.n}, READ-ONLY\n")

        for _ in range(max(0, args.warmup)):
            backend.Joints()

        samples_ms: list[float] = []
        failures = 0
        t_start = time.perf_counter()
        for _ in range(args.n):
            t0 = time.perf_counter()
            try:
                joints = backend.Joints()
            except Exception:                               # noqa: BLE001
                failures += 1
                continue
            dt = (time.perf_counter() - t0) * 1000.0
            if joints is None:
                failures += 1
                continue
            samples_ms.append(dt)
        elapsed = time.perf_counter() - t_start
    finally:
        backend.disconnect()

    if not samples_ms:
        print("Every read failed; nothing to report.", file=sys.stderr)
        return 1

    a = np.asarray(samples_ms)
    p50, p90, p95, p99 = (float(np.percentile(a, q)) for q in (50, 90, 95, 99))
    print(f"   samples    {a.size} ok, {failures} failed")
    print(f"   p50        {p50:.2f} ms")
    print(f"   p90        {p90:.2f} ms")
    print(f"   p95        {p95:.2f} ms")
    print(f"   p99        {p99:.2f} ms")
    print(f"   mean/max   {a.mean():.2f} / {a.max():.2f} ms")
    print(f"   achieved   {a.size / elapsed:.1f} reads/s over {elapsed:.1f} s")
    if p95 > 100.0:
        print("\n   p95 exceeds 100 ms: the 10 Hz telemetry loop cannot hold its rate.")

    if args.csv:
        out = Path(args.csv)
        if not out.is_absolute():
            out = Path(__file__).resolve().parents[1] / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "rtt_ms"])
            w.writerows(enumerate(round(x, 4) for x in samples_ms))
        print(f"\n   samples -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
