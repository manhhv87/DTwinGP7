"""
probe_hse.py — quick connectivity probe for a Yaskawa YRC1000 (HSE).

Usage:
    python tools/probe_hse.py <CONTROLLER_IP>

Checks, in order:
  1. UDP 10040 READ_STATUS  — the channel the app uses to "see" the robot.
  2. TCP 21 (FTP)           — used to upload .JBI on "Run on Robot".
  3. TCP 10040              — sanity (HSE is UDP, so this is expected to fail/refuse).

Prints a clear PASS/FAIL per check so you can tell whether the controller is
answering HSE at all (vs. only ICMP ping).
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path

# Allow running from repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestrator.backends.motoman_hse import MotomanHSEBackend  # noqa: E402


def _tcp_check(ip: str, port: int, timeout: float = 2.0) -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((ip, port))
        return "OPEN"
    except ConnectionRefusedError:
        return "REFUSED (host reachable, port closed)"
    except socket.timeout:
        return "TIMEOUT (no response — firewall/closed)"
    except OSError as e:
        return f"ERROR {e}"
    finally:
        s.close()


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python tools/probe_hse.py <CONTROLLER_IP>")
        return 2
    ip = sys.argv[1].strip()
    # Force UTF-8 stdout so the script never dies on a non-ASCII char under the
    # Windows cp1252 console.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print(f"== HSE probe -> {ip} ==\n")

    # 1. UDP 10040 — the real "can the app see the robot" test.
    print("[1] UDP 10040  READ_STATUS (HSE) …")
    be = MotomanHSEBackend(ip=ip, timeout_s=2.0)
    try:
        be.connect()
        ok = be.Valid()
        if ok:
            try:
                j = be.Joints()
                print("    PASS — controller answered. Joints: ["
                      + ", ".join(f"{q:+.1f}" for q in j) + "]")
            except Exception as e:  # noqa: BLE001
                print(f"    PARTIAL — status OK but Joints() failed: {e}")
        else:
            print("    FAIL — no HSE reply. → HSE Server function likely OFF, "
                  "or UDP 10040 blocked, or wrong NIC.")
    except Exception as e:  # noqa: BLE001
        print(f"    FAIL — {e}")
    finally:
        be.disconnect()

    # 2. TCP 21 — FTP (job upload).
    print("\n[2] TCP 21    FTP (job upload) …")
    print(f"    {_tcp_check(ip, 21)}")

    # 3. TCP 10040 — HSE is UDP; this is just a sanity check.
    print("\n[3] TCP 10040 (sanity; HSE is UDP, refusal is normal) …")
    print(f"    {_tcp_check(ip, 10040)}")

    print("\nInterpretation:")
    print("  • [1] PASS  → app should connect; check REMOTE mode for Run on Robot.")
    print("  • [1] FAIL but ping OK → enable HSE Server on the YRC1000, check the")
    print("    PC firewall, and make sure only the controller-subnet NIC is up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
