"""ftp_write_test.py — isolate the Run-on-Robot "451 Error closing file [5130]".

Run this ON THE ROBOT PC (same subnet as the YRC1000). It does the EXACT FTP
sequence Run-on-Robot uses (login → cwd → DELE → STOR → list), but with a tiny,
KNOWN-GOOD job under a throwaway name. The point: if even this minimal valid job
cannot be SAVED, the problem is the controller / FTP account — NOT the app's
generated .JBI (which we already proved is byte-exact for PP1.JBI).

Usage (defaults match the verified YRC1000 settings):
    python tools/ftp_write_test.py
    python tools/ftp_write_test.py 192.168.125.100 rcmaster 9999999999999999 /JOB

Read the PASS/FAIL lines at the end:
  - LIST ok but STOR 451/5xx  -> controller refuses to WRITE jobs for this account
    / mode. Check: FTP account job-WRITE permission, operation mode + security
    level (try TEACH mode), teach-pendant edit lock. (App cannot fix this.)
  - STOR ok                   -> the controller CAN save jobs; the app's content
    or sequence is the suspect — capture the exact failing .JBI.
"""
from __future__ import annotations

import ftplib
import io
import sys

IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.125.100"
USER = sys.argv[2] if len(sys.argv) > 2 else "rcmaster"
PW = sys.argv[3] if len(sys.argv) > 3 else "9999999999999999"
JOBDIR = sys.argv[4] if len(sys.argv) > 4 else "/JOB"

# A minimal, syntactically-valid INFORM job (1 P-var, 1 MOVJ). CRLF like a TP export.
TESTNAME = "ZZTEST01"
JOB = (
    "/JOB\r\n//NAME ZZTEST01\r\n//POS\r\n///NPOS 0,0,0,1,0,0\r\n///TOOL 0\r\n"
    "///POSTYPE PULSE\r\n///PULSE\r\nP00000=0,0,0,0,0,0\r\n//INST\r\n"
    "///DATE 2026/01/01 00:00\r\n///ATTR SC,RW\r\n///GROUP1 RB1\r\nNOP\r\n"
    "MOVJ P000 VJ=5.00\r\nEND\r\n"
).encode("ascii")


def main() -> int:
    print(f"FTP {USER}@{IP}{JOBDIR}  (write-test job {TESTNAME}.JBI)\n")
    ftp = ftplib.FTP()
    try:
        ftp.connect(IP, 21, timeout=5.0)
        print("connect : OK", ftp.getwelcome())
        ftp.login(USER, PW)
        print("login   : OK")
        ftp.cwd(JOBDIR)
        print(f"cwd     : OK -> {JOBDIR}")
    except Exception as e:                       # noqa: BLE001
        print(f"FAIL at connect/login/cwd: {e!r}")
        print("\n>>> Cannot even reach /JOB — check IP / account / FTP server enabled.")
        return 1

    try:
        names = ftp.nlst()
        print(f"list    : OK ({len(names)} entries) — READ works")
    except Exception as e:                       # noqa: BLE001
        print(f"list    : FAIL {e!r}")

    # DELE (tolerate 550 = not found), then STOR — same as upload_job.
    try:
        ftp.delete(f"{TESTNAME}.JBI")
        print("delete  : OK (pre-existing test job removed)")
    except ftplib.error_perm as e:
        print(f"delete  : {str(e).strip()}  (550=not found is fine)")

    try:
        ftp.storbinary(f"STOR {TESTNAME}.JBI", io.BytesIO(JOB))
        print("STOR    : OK  <<< controller CAN save jobs >>>")
        verdict = "STOR_OK"
    except ftplib.error_temp as e:               # 4xx, e.g. 451 Error closing file
        print(f"STOR    : FAIL (4xx) {str(e).strip()}")
        verdict = "STOR_451"
    except ftplib.error_perm as e:               # 5xx, e.g. 553/503
        print(f"STOR    : FAIL (5xx) {str(e).strip()}")
        verdict = "STOR_5xx"
    except Exception as e:                        # noqa: BLE001
        print(f"STOR    : FAIL {e!r}")
        verdict = "STOR_ERR"

    try:
        ftp.delete(f"{TESTNAME}.JBI")            # cleanup
    except Exception:                            # noqa: BLE001
        pass
    try:
        ftp.quit()
    except Exception:                            # noqa: BLE001
        pass

    print("\n=== VERDICT ===")
    if verdict == "STOR_OK":
        print("The controller + this FTP account CAN write a minimal valid job.")
        print("=> The 451 is then specific to the uploaded job/sequence — send me")
        print("   the exact .JBI that fails (Program -> Export .JBI) to compare.")
    else:
        print("The controller REFUSED to save even a minimal valid job (same 451).")
        print("=> This is CONTROLLER-SIDE, not the app. Check, in order:")
        print("   1) FTP account job-WRITE permission (rcmaster may be read-only;")
        print("      try the MANAGEMENT-level FTP account/password).")
        print("   2) Operation mode: try TEACH mode on the key switch, then re-run.")
        print("   3) Security/management level + 'job edit prohibit' parameter.")
        print("   4) The job must not be open/edit-locked on the teach pendant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
