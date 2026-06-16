"""
logger.py
─────────
TrialLogger — writes each pick-and-place trial's result to a CSV file.

Each CSV row is one trial; at the end of an experiment use
scripts/04_analyze_results.py for statistical analysis. The summarize() helper
builds a failure-mode matrix (for the paper's Discussion section — doc section 9.3).

Pure stdlib (csv) → no pandas dependency, usable anywhere.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Column order in the CSV.
FIELDNAMES = [
    "timestamp",
    "trial_id",
    "success",
    "class_name",
    "cycle_time_s",
    "failure_reason",
    "final_state",
    "lighting",
    "overlap",
    "mode",
    "ik",
]


class TrialLogger:
    """Logs trials to CSV (append-mode, safe across multiple sessions).

    Args:
        csv_path: Path to the output CSV file.
        extra_context: Dict of fields describing the experiment conditions
            (lighting, overlap, mode) — attached to every row.
    """

    def __init__(
        self,
        csv_path: str | Path,
        extra_context: dict[str, Any] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.context = extra_context or {}
        self._rows: list[dict[str, Any]] = []

        # Write the header if the file does not exist yet.
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    def log_trial(
        self,
        trial_id: int,
        success: bool,
        class_name: str = "",
        cycle_time_s: float = 0.0,
        failure_reason: str = "",
        final_state: str = "",
    ) -> None:
        """Write one trial result row to the CSV."""
        import time

        row = {
            "timestamp": time.time(),
            "trial_id": trial_id,
            "success": int(success),
            "class_name": class_name,
            "cycle_time_s": round(cycle_time_s, 3),
            "failure_reason": failure_reason,
            "final_state": final_state,
            "lighting": self.context.get("lighting", ""),
            "overlap": self.context.get("overlap", ""),
            "mode": self.context.get("mode", ""),
            "ik": self.context.get("ik", ""),     # IK source (client/yrc) per run
        }
        self._rows.append(row)
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

        status = "OK" if success else f"FAIL ({failure_reason})"
        logger.info("Trial %d logged: %s", trial_id, status)

    def summarize(self) -> dict[str, Any]:
        """Summarize the current session's results (trials logged in this process).

        Returns:
            Dict {total, successful, success_rate, failure_modes}.
            failure_modes: {reason: count}.
        """
        total = len(self._rows)
        successful = sum(r["success"] for r in self._rows)
        failure_modes: dict[str, int] = {}
        for r in self._rows:
            if not r["success"] and r["failure_reason"]:
                reason = r["failure_reason"]
                failure_modes[reason] = failure_modes.get(reason, 0) + 1

        return {
            "total": total,
            "successful": successful,
            "success_rate": successful / total if total else 0.0,
            "failure_modes": failure_modes,
        }
