"""
provenance.py
─────────────
Which code ran: the git commit, branch and uncommitted changes of the checkout.

The run metadata of session 1 of E2 (e2chuan-1, 23 Oct 2026) recorded the config
and the checkpoint but not the code, so which commit produced a CSV could only be
inferred. git_provenance is written into every run's metadata.json by
03_run_experiment.py and into the sealed key file by run_blinded_campaign.py. A
checkout with uncommitted changes to tracked files is marked dirty and the SHA-256
of its diff against the commit is recorded; 03 also saves that diff next to the
run's metadata (save_git_diff), so the code that ran can be rebuilt.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

MAX_LISTED = 50


def _git(repo: str | Path, *args: str, timeout_s: float = 10.0) -> bytes:
    """stdout of a git command run in ``repo``, as bytes; raises on failure."""
    return subprocess.run(
        ["git", "-c", "core.quotepath=off", *args], cwd=str(repo),
        capture_output=True, check=True, timeout=timeout_s,
    ).stdout


def _diff(repo: str | Path, timeout_s: float) -> bytes:
    """The tracked changes against HEAD, binary files included."""
    return _git(repo, "diff", "--binary", "HEAD", timeout_s=timeout_s)


def git_provenance(repo: str | Path, timeout_s: float = 10.0) -> dict[str, Any]:
    """Commit, branch and change state of the git checkout holding ``repo``.

    Returns {"commit", "branch", "dirty", "changed", "untracked"}: dirty is True
    when a tracked file differs from the commit (staged or not); changed lists
    those files and untracked the files git does not track and does not ignore,
    at most MAX_LISTED each, as plain paths (not quoted, non-ASCII kept; the new
    path of a rename). A dirty checkout also gets "diff_sha256", the SHA-256 of
    ``git diff --binary HEAD``. When git is missing or the folder is no checkout,
    {"error": reason}; a run is never stopped for it.
    """
    try:
        commit = _git(repo, "rev-parse", "HEAD", timeout_s=timeout_s).decode(
            "utf-8", errors="replace").strip()
        branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD",
                      timeout_s=timeout_s).decode("utf-8", errors="replace").strip()
        status = _git(repo, "status", "--porcelain=v1", "-z", timeout_s=timeout_s)
    except (OSError, subprocess.SubprocessError) as e:
        detail = getattr(e, "stderr", None) or str(e)
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        return {"error": f"{type(e).__name__}: {str(detail).strip()}"}
    changed, untracked = [], []
    fields = status.decode("utf-8", errors="replace").split("\0")
    i = 0
    while i < len(fields):
        entry = fields[i]
        i += 1
        if len(entry) < 4:
            continue
        code, path = entry[:2], entry[3:]
        if "R" in code or "C" in code:
            i += 1                  # with -z a rename's original path comes next
        (untracked if code == "??" else changed).append(path)
    out: dict[str, Any] = {"commit": commit, "branch": branch, "dirty": bool(changed),
                           "changed": changed[:MAX_LISTED],
                           "untracked": untracked[:MAX_LISTED]}
    if changed:
        try:
            out["diff_sha256"] = hashlib.sha256(_diff(repo, timeout_s)).hexdigest()
        except (OSError, subprocess.SubprocessError) as e:
            out["diff_sha256"] = f"error: {type(e).__name__}"
    return out


def save_git_diff(repo: str | Path, path: str | Path,
                  timeout_s: float = 30.0) -> str | None:
    """Write ``git diff --binary HEAD`` of the checkout to ``path`` and return its
    SHA-256, or None when git fails (nothing is written then)."""
    try:
        diff = _diff(repo, timeout_s)
    except (OSError, subprocess.SubprocessError):
        return None
    Path(path).write_bytes(diff)
    return hashlib.sha256(diff).hexdigest()
