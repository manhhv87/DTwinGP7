"""ftp_bisect_pp1.py — pinpoint the Run-on-Robot "451 Error closing file [5130]".

Run this ON THE ROBOT PC (same subnet as the YRC1000), TODAY, AFTER you have
confirmed a minimal job STORs OK (tools/ftp_write_test.py  ->  STOR: OK).

It STORs several controlled VARIANTS of the real job PP1.JBI and reports which
ones the controller SAVES vs rejects with 451. Each variant changes exactly ONE
thing vs the others, so the PASS/FAIL pattern reveals the root cause in a single
run on hardware.

WHY THIS EXISTS
---------------
As of 2026-06-17 a minimal CRLF job (ZZTEST01) STORs fine. That DISPROVES the
earlier "the controller refuses ALL job writes" theory — so the discriminator is
either PP1's CONTENT or HOW the app uploads it.

Strongest code-level suspect: the app's Run-on-Robot path reads the rendered job
with `Path.read_text(encoding="utf-8")` (mixin_connection.py:274). read_text uses
universal-newline mode, which COLLAPSES the file's CRLF line endings to LF before
the bytes are STOR'd. But real teach-pendant .JBI files (and ZZTEST01) use CRLF.
A YRC1000 that expects CRLF record terminators ACCEPTS the byte transfer but
fails to CLOSE/save the job  ->  exactly "451 Error closing file.--[5130]--".
T1 (CRLF) vs T2 (LF) tests this head-on.

VARIANTS  (all STOR'd under THROWAWAY names and deleted afterwards; the optional
real-name test is OFF by default and does NOT delete afterwards):
  A0  ZZTEST01, CRLF                  sanity anchor  (expect: same OK as today)
  T1  PP1 exact on-disk bytes, CRLF   PP1 content as a real pendant file (CRLF)
  T2  PP1 content, LF-only            mimics what the APP actually uploads today
  T3  PP1 content minus FOLDERNAME    isolates the ///FOLDERNAME FOLDER001 line
  T4  PP1 exact bytes under name PP1  isolates name/selection/edit-lock (--real-name)

READING THE RESULT (printed at the end):
  A0 ok, T1 ok, T2 FAIL          -> LINE ENDINGS: controller needs CRLF; the app
                                    sends LF (read_text collapse). FIX = code.
  A0 ok, T1 FAIL, T3 ok          -> FOLDERNAME: drop ///FOLDERNAME on upload.
  A0 ok, T1 ok, (T4 FAIL)        -> NAME/selection/edit-lock on PP1: deselect it
                                    on the teach pendant, then retry.
  A0 ok, T1 ok, T2 ok, T3 ok     -> PP1 content saves fine now; yesterday's 451
                                    was the (now-resolved) global block.
  A0 FAIL                        -> global save-block is back: check FTP account
                                    write permission / operation mode / edit-lock.

Usage (defaults match the verified YRC1000 settings):
    python tools/ftp_bisect_pp1.py
    python tools/ftp_bisect_pp1.py 192.168.125.100 rcmaster 9999999999999999 /JOB
    python tools/ftp_bisect_pp1.py --real-name          # also test under name PP1
    python tools/ftp_bisect_pp1.py --jbi C:\path\to\PP1.JBI
"""
from __future__ import annotations

import ftplib
import io
import sys
from pathlib import Path

# ── argument parsing (flags first, then positional, like ftp_write_test.py) ──
_argv = sys.argv[1:]
TEST_REAL_NAME = "--real-name" in _argv
_argv = [a for a in _argv if a != "--real-name"]
_jbi_override = ""
if "--jbi" in _argv:
    i = _argv.index("--jbi")
    _jbi_override = _argv[i + 1] if i + 1 < len(_argv) else ""
    del _argv[i:i + 2]

IP = _argv[0] if len(_argv) > 0 else "192.168.125.100"
USER = _argv[1] if len(_argv) > 1 else "rcmaster"
PW = _argv[2] if len(_argv) > 2 else "9999999999999999"
JOBDIR = _argv[3] if len(_argv) > 3 else "/JOB"

# ── locate PP1.JBI (default: relative to this script's repo) ──
if _jbi_override:
    PP1_PATH = Path(_jbi_override)
else:
    PP1_PATH = Path(__file__).resolve().parent.parent / "examples" / "MINH" / "PP1.JBI"

# A0: the same minimal CRLF job that STOR'd OK in ftp_write_test.py.
ZZTEST01_CRLF = (
    "/JOB\r\n//NAME ZZTEST01\r\n//POS\r\n///NPOS 0,0,0,1,0,0\r\n///TOOL 0\r\n"
    "///POSTYPE PULSE\r\n///PULSE\r\nP00000=0,0,0,0,0,0\r\n//INST\r\n"
    "///DATE 2026/01/01 00:00\r\n///ATTR SC,RW\r\n///GROUP1 RB1\r\nNOP\r\n"
    "MOVJ P000 VJ=5.00\r\nEND\r\n"
).encode("ascii")


def to_lf(data: bytes) -> bytes:
    """Collapse CRLF/CR to LF — exactly what Path.read_text() does on read."""
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def strip_foldername(crlf: bytes) -> bytes:
    """Remove any ///FOLDERNAME line, preserving CRLF framing."""
    parts = crlf.split(b"\r\n")
    kept = [ln for ln in parts if not ln.startswith(b"///FOLDERNAME")]
    return b"\r\n".join(kept)


def crlf_summary(data: bytes) -> str:
    cr = data.count(b"\r")
    lf = data.count(b"\n")
    crlf = data.count(b"\r\n")
    kind = "CRLF" if crlf and cr == lf == crlf else ("LF-only" if cr == 0 and lf else "MIXED")
    return f"{len(data)}B {kind} (CR={cr} LF={lf} CRLF={crlf})"


def stor_variant(ftp_factory, name: str, data: bytes, cleanup: bool) -> str:
    """Mirror motoman_hse.upload_job's exact FTP sequence for one variant.

    Returns one of: 'OK', '451:<msg>', '4xx:<msg>', '5xx:<msg>', 'ERR:<msg>'.
    """
    filename = f"{name}.JBI"
    ftp = ftp_factory()
    try:
        ftp.connect(IP, 21, timeout=5.0)
        ftp.login(USER, PW)
        ftp.cwd(JOBDIR)
    except Exception as e:                                   # noqa: BLE001
        return f"ERR:connect/login/cwd {e!r}"

    # DELE-before-STOR (550 = not found is fine), then STOR — same as upload_job.
    try:
        ftp.delete(filename)
    except ftplib.error_perm as e:
        if not str(e).strip().startswith("550"):
            try:
                ftp.quit()
            except Exception:                                # noqa: BLE001
                pass
            return f"5xx:DELE-refused {str(e).strip()}"      # selected/locked job
    except Exception:                                        # noqa: BLE001
        pass

    verdict = "OK"
    try:
        ftp.storbinary(f"STOR {filename}", io.BytesIO(data))
    except ftplib.error_temp as e:                           # 4xx, e.g. 451
        msg = str(e).strip()
        verdict = (f"451:{msg}" if msg.startswith(("451", "450", "452"))
                   else f"4xx:{msg}")
    except ftplib.error_perm as e:                           # 5xx
        verdict = f"5xx:{str(e).strip()}"
    except Exception as e:                                   # noqa: BLE001
        verdict = f"ERR:{e!r}"

    if cleanup and verdict == "OK":
        try:
            ftp.delete(filename)
        except Exception:                                    # noqa: BLE001
            pass
    try:
        ftp.quit()
    except Exception:                                        # noqa: BLE001
        pass
    return verdict


def main() -> int:
    print(f"FTP bisect  {USER}@{IP}{JOBDIR}\n")
    if not PP1_PATH.exists():
        print(f"FAIL: cannot find PP1.JBI at {PP1_PATH}\n"
              f"      pass it explicitly:  python tools/ftp_bisect_pp1.py --jbi <path>")
        return 1

    pp1_crlf = PP1_PATH.read_bytes()                         # exact on-disk bytes
    print(f"source PP1.JBI : {PP1_PATH}\n               : {crlf_summary(pp1_crlf)}\n")

    variants = [
        ("A0", "ZZTEST01", ZZTEST01_CRLF, True,
         "minimal job, CRLF (sanity anchor)"),
        ("T1", "ZZBIS1", pp1_crlf, True,
         "PP1 exact bytes, CRLF"),
        ("T2", "ZZBIS2", to_lf(pp1_crlf), True,
         "PP1 content, LF-only (what the app uploads)"),
        ("T3", "ZZBIS3", strip_foldername(pp1_crlf), True,
         "PP1 content minus ///FOLDERNAME, CRLF"),
    ]
    if TEST_REAL_NAME:
        variants.append(
            ("T4", "PP1", pp1_crlf, False,
             "PP1 exact bytes, CRLF, REAL name PP1 (left on controller)"))

    results: dict[str, str] = {}
    print(f"{'id':<3} {'name':<9} {'bytes/kind':<28} desc")
    print("-" * 78)
    for vid, name, data, cleanup, desc in variants:
        print(f"{vid:<3} {name:<9} {crlf_summary(data):<28} {desc}")
        results[vid] = stor_variant(ftplib.FTP, name, data, cleanup)
    print()

    print("=== RESULTS ===")
    for vid, name, *_ in variants:
        r = results[vid]
        tag = "PASS" if r == "OK" else "FAIL"
        print(f"  {vid}  {name:<9} {tag:<4} {('' if r == 'OK' else r)}")
    print()

    def ok(v: str) -> bool:
        return results.get(v) == "OK"

    print("=== VERDICT ===")
    if not ok("A0"):
        print("A0 (minimal CRLF) FAILED — the global save-block is in effect again.")
        print("=> CONTROLLER-SIDE: FTP account job-WRITE permission / operation mode /")
        print("   security level / edit-lock. Re-run tools/ftp_write_test.py to confirm.")
    elif ok("T1") and not ok("T2"):
        print(">>> ROOT CAUSE: LINE ENDINGS. PP1 saves as CRLF (T1 PASS) but FAILS as")
        print("    LF-only (T2 FAIL) — and the app's Run-on-Robot upload sends LF because")
        print("    mixin_connection.py:274 reads the job via Path.read_text() (universal")
        print("    newlines collapse CRLF->LF). FIX (code): read bytes / preserve CRLF on")
        print("    upload, e.g. jbi_path.read_text(newline='') OR read_bytes(), OR have")
        print("    motoman_hse.upload_job normalise '\\n' -> '\\r\\n' before encode('ascii').")
    elif not ok("T1") and ok("T3"):
        print(">>> ROOT CAUSE: ///FOLDERNAME. PP1 with the folder tag FAILS (T1) but saves")
        print("    once it is removed (T3). FIX: drop the ///FOLDERNAME line on Run-on-Robot")
        print("    uploads (folder_name='' in mixin_connection.py:273) or create/enter the")
        print("    folder before STOR.")
    elif ok("T1") and TEST_REAL_NAME and not ok("T4"):
        print(">>> ROOT CAUSE: NAME / SELECTION / EDIT-LOCK on PP1. The content saves under a")
        print("    throwaway name (T1) but not under the real name PP1 (T4). Deselect/CANCEL")
        print("    PP1 on the teach pendant (and clear any edit-lock), then retry.")
    elif ok("T1") and ok("T2") and ok("T3"):
        print("PP1 content SAVES in every form tested (incl. LF-only). Yesterday's 451 was")
        print("the now-resolved global save-block. Run-on-Robot should work today — try it,")
        print("and if --real-name was off, re-run with --real-name to rule out the name lock.")
    else:
        print("Mixed result — see the table above. Report it back with the exact 451 text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
