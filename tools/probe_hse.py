"""
probe_hse.py — quick connectivity probe for a Yaskawa YRC1000 (HSE).

Usage:
    python tools/probe_hse.py <CONTROLLER_IP> [ftp_user] [ftp_pass] [ftp_dir]

    ftp_user/ftp_pass default to "" (anonymous); ftp_dir defaults to /MPRAM1/JBI.

Checks, in order:
  1. UDP 10040 READ_STATUS  — the channel the app uses to "see" the robot.
  2. FTP 21 login + cwd + list — the actual path used to upload .JBI on
     "Run on Robot" (verifies the credentials AND the job dir, not just the port).
  3. TCP 10040              — sanity (HSE is UDP, so this is expected to fail/refuse).

Prints a clear PASS/FAIL per check so you can tell whether the controller is
answering HSE at all (vs. only ICMP ping).
"""
from __future__ import annotations

import ftplib
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


def _ftp_check(ip: str, user: str, password: str, job_dir: str,
               timeout: float = 10.0) -> str:
    """Real FTP login + cwd(job_dir) + directory listing — verifies the exact
    path 'Run on Robot' uses (credentials AND job dir), not just port 21."""
    # First: does TCP 21 even accept a connection? Separates "port closed" from
    # "port open but no FTP service / banner" (the latter = FTP function off/busy).
    tcp = _tcp_check(ip, 21, timeout=timeout)
    if not tcp.startswith("OPEN"):
        return f"FAIL — TCP 21 {tcp}"
    ftp = ftplib.FTP()
    try:
        ftp.connect(ip, 21, timeout=timeout)   # reads the 220 welcome banner
        ftp.login(user, password)              # "", "" = anonymous
        ftp.cwd(job_dir)
        ftp.set_pasv(True)
        try:
            names = ftp.nlst()
        except Exception:                      # noqa: BLE001 — listing optional
            names = []
        who = user or "anonymous"
        njbi = sum(1 for n in names if n.upper().endswith(".JBI"))
        return (f"PASS — logged in as '{who}', cwd '{job_dir}' OK "
                f"({len(names)} entries, {njbi} .JBI)")
    except ftplib.error_perm as e:
        return f"FAIL — login/cwd refused: {e} (check FTP user/pass + job dir)"
    except (socket.timeout, OSError) as e:
        return (f"FAIL — TCP 21 open but no FTP response ({e}). "
                "→ FTP server function likely DISABLED on the YRC1000 (or banner "
                "slow / a stale FTP session is holding the single slot — power-cycle"
                " or close other FTP clients).")
    except Exception as e:                     # noqa: BLE001
        return f"FAIL — {e}"
    finally:
        try:
            ftp.quit()
        except Exception:                      # noqa: BLE001
            pass


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python tools/probe_hse.py <CONTROLLER_IP> "
              "[ftp_user] [ftp_pass] [ftp_dir]")
        return 2
    ip = sys.argv[1].strip()
    ftp_user = sys.argv[2] if len(sys.argv) > 2 else ""
    ftp_pass = sys.argv[3] if len(sys.argv) > 3 else ""
    ftp_dir = sys.argv[4].strip() if len(sys.argv) > 4 else "/MPRAM1/JBI"
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

    # 2. FTP 21 — full login + cwd + list (what Run-on-Robot actually does).
    print("\n[2] FTP 21    login + cwd + list (job upload path) …")
    print(f"    {_ftp_check(ip, ftp_user, ftp_pass, ftp_dir)}")

    # 3. TCP 10040 — HSE is UDP; this is just a sanity check.
    print("\n[3] TCP 10040 (sanity; HSE is UDP, TIMEOUT/REFUSED both normal) …")
    print(f"    {_tcp_check(ip, 10040)}")

    print("\nInterpretation:")
    print("  • [1] PASS  → app can read the robot; check REMOTE mode for Run on Robot.")
    print("  • [1] FAIL but ping OK → enable HSE Server on the YRC1000, check the")
    print("    PC firewall, and make sure only the controller-subnet NIC is up.")
    print("  • [2] PASS  → 'Run on Robot' upload will work with these FTP creds/dir.")
    print("  • [2] login refused → set the right FTP user/pass (pass as args).")
    print("  • [2] cwd refused   → fix the FTP job dir (default /MPRAM1/JBI).")
    print("  • [3] TIMEOUT/REFUSED is EXPECTED (HSE has no TCP service here).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
