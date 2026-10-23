"""
provenance.py
─────────────
Which code ran: the git commit, branch and uncommitted changes of the checkout.

The run metadata of session 1 of E2 (e2chuan-1, 23 Oct 2026) recorded the config
and the checkpoint but not the code, so which commit produced a CSV could only be
inferred. git_provenance is written into every run's metadata.json by
03_run_experiment.py and into the sealed key file by run_blinded_campaign.py.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

MAX_LISTED = 50


def git_provenance(repo: str | Path, timeout_s: float = 10.0) -> dict[str, Any]:
    """Commit, branch and change state of the git checkout holding ``repo``.

    Returns {"commit", "branch", "dirty", "changed", "untracked"}: dirty is True
    when a tracked file differs from the commit (staged or not); changed lists
    those files and untracked the files git does not track and does not ignore,
    at most MAX_LISTED each. When git is missing or the folder is no checkout,
    {"error": reason}; a run is never stopped for it.
    """
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=str(repo), capture_output=True, check=True,
            timeout=timeout_s, text=True, encoding="utf-8", errors="replace",
        ).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
        status = git("status", "--porcelain")
    except (OSError, subprocess.SubprocessError) as e:
        detail = getattr(e, "stderr", None) or str(e)
        return {"error": f"{type(e).__name__}: {str(detail).strip()}"}
    changed, untracked = [], []
    for line in status.splitlines():
        if len(line) < 4:
            continue
        (untracked if line.startswith("??") else changed).append(line[3:])
    return {"commit": commit, "branch": branch, "dirty": bool(changed),
            "changed": changed[:MAX_LISTED], "untracked": untracked[:MAX_LISTED]}
