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
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

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

# Failure-context columns (C3 failure-driven loop): detected pose, detector
# confidence, mask area, saved RGB frame, and the pre-drawn pose-list id.
# Appended AFTER the base fields; files created before this change keep their
# original header (see _resolve_fieldnames) so appends stay consistent.
CONTEXT_FIELDNAMES = [
    "det_x_mm",
    "det_y_mm",
    "det_z_mm",
    "det_yaw_deg",
    "confidence",
    "mask_area",
    "frame_path",
    "pose_id",
    # Staging and provenance. The condition and the stacked-support class are
    # announced per trial by the runner; without them in the row there is no way
    # to tell afterwards which trials ran under which staging. session/block/
    # operator make the run order recoverable, so a configuration effect can be
    # separated from a drift-across-the-afternoon effect.
    "condition",
    "stack_on",
    "session_id",
    "block_id",
    "operator_id",
    # C4 depth modes: the configured mode, the estimate that gave z (fusion picks
    # one per detection), the share of mask pixels returning depth, the part height
    # used (resolved per instance for the carton), the carton size-fit IoU, and the
    # height above the table implied by the median depth.
    "depth_mode",
    "depth_used",
    "depth_valid_frac",
    "part_height_mm",
    "size_iou",
    # Top face measured in mm on the part plane, from the mask outline. The
    # motion guard refuses a trial whose rectangle matches no size variant of
    # the detected class, so the numbers it judged on belong in the record.
    "yaw_rect_long_mm",
    "yaw_rect_short_mm",
    "yaw_near_square",
    "h_depth_mm",
    # Height of the median-depth point above the table. With h_depth_mm (the
    # top band) it shows whether the mask lay on the part or spilled beside it.
    "h_median_mm",
    # Share of the mask's pixels returning depth at table height (within 5 mm);
    # from 0.3 up the detection is dropped as off the part, unless what is left
    # still gives the class's known-size fit and passes the shape check
    # (Orchestrator._mask_off_part).
    "table_share",
    # Mask pixels left once those at table height are removed; the outline and the
    # known-size fit read these (perception/depth_modes.py, off_table_mask).
    "part_mask_area",
    # IoU of the class's known top-face rectangle fitted to the mask outline; from
    # 0.5 up its centre and yaw are the grasp pose (perception/depth_modes.py,
    # fit_footprint), below it the mask centroid and rectangle yaw stand.
    "fit_iou",
    # How far that rectangle slides along its long side at the same IoU, mm; over
    # 10 mm the trial is refused before motion (Orchestrator._shape_contradicts_class).
    "fit_plateau_mm",
    # Operator verdict, scored by eye right after the trial (03 --confirm-each-trial).
    # `success` is the machine's: the detect sensor proves a part was in the gripper
    # when it closed, but nothing checks that the part was still held at PLACE or
    # landed where it should. A part dropped in transfer scores success=1 here and
    # human_ok=0. Empty when nobody scored the trial.
    "human_ok",
    "run_id",
    # Called carton size of the target and of the support, mm (80 or 120), from the
    # pose list's carton sidecar; empty for other classes.
    "carton_mm",
    "stack_on_carton_mm",
    # Centre of the part's top face in the base frame, the same in every depth mode
    # (placement_check.top_centre_xy); det_x/y_mm is the commanded point, which in
    # rgbd lies at the median depth and in plane at the class height.
    "top_x_mm",
    "top_y_mm",
    # Placement check before the trial, servo OFF (03 --confirm-each-trial with a
    # pose list, see Orchestrator.run_placement_check). Its captures are never used
    # for the trial, which detects afresh. Empty when the check did not run.
    #   placement_flags       flags of the first check, '+'-joined (class,
    #                         position, stack, carton_size, yaw); empty when none.
    #   placement_recaptures  checks after the first, each after the operator was
    #                         asked to fix the placement (0 to MAX_CHECKS - 1).
    #   placement_override    1 when the trial ran after its last check (the
    #                         MAX_CHECKS-th) was still flagged (limit), else 0.
    #   placement_events      every check as flags:outcome, joined by ';'. flags is
    #                         'none' when nothing was flagged and 'noframe' when the
    #                         detection window received no frame; outcome is clear,
    #                         skip (no frame: nothing checked, the trial goes on),
    #                         fixed (the operator was asked to fix it: checked
    #                         again), limit (the last check, still flagged: the
    #                         trial runs as placed), or abort (the run was stopped
    #                         at the check).
    #   placement_captures    JSON list, one entry per check: capture number,
    #                         frames received, frame path, the judged detection's
    #                         class, top_x_mm, top_y_mm, h_depth_mm and yaw_deg
    #                         (null when absent), its flags, the outcome and, for a
    #                         flagged check, answer_s, the seconds until the
    #                         operator's ENTER.
    #   placement_check_s     seconds from the first check to the motion gate; not
    #                         in cycle_time_s, which starts after the checks.
    "placement_flags",
    "placement_recaptures",
    "placement_override",
    "placement_events",
    "placement_captures",
    "placement_check_s",
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
        run_metadata: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.context = dict(extra_context or {})
        self.run_id = uuid4().hex
        self.metadata_path = self.csv_path.with_name(
            f"{self.csv_path.stem}.{self.run_id}.metadata.json")
        self._metadata_source = run_metadata
        self._created_at = datetime.now(timezone.utc).isoformat()
        self._rows: list[dict[str, Any]] = []
        self._fieldnames = self._resolve_fieldnames()
        with self.csv_path.open(newline="", encoding="utf-8") as stream:
            self._first_data_row = sum(1 for row in csv.reader(stream) if row)
        self.write_run_metadata()

    def write_run_metadata(self) -> None:
        """Refresh runtime settings without recomputing the checkpoint hash.

        A unique sidecar per invocation also preserves provenance when appending
        to a legacy CSV without a run_id column. Data row numbers are one-based.
        """
        if self._metadata_source is None:
            return
        provenance = (self._metadata_source() if callable(self._metadata_source)
                      else self._metadata_source)
        payload = {"schema_version": 1, "run_id": self.run_id,
                   "created_at_utc": self._created_at, "csv": str(self.csv_path.resolve()),
                   "csv_has_run_id": "run_id" in self._fieldnames,
                   "first_data_row": self._first_data_row,
                   "rows_written": len(self._rows), "provenance": provenance}
        temporary = self.metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                        default=str) + "\n", encoding="utf-8")
        temporary.replace(self.metadata_path)

    def _resolve_fieldnames(self) -> list[str]:
        """New files get the full schema; existing files keep their own header
        (appending wider rows to an old file would silently misalign columns)."""
        full = FIELDNAMES + CONTEXT_FIELDNAMES
        if self.csv_path.exists():
            with self.csv_path.open("r", newline="", encoding="utf-8") as f:
                header = f.readline().strip()
            if header:
                return header.split(",")
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=full).writeheader()
        return full

    def log_trial(
        self,
        trial_id: int,
        success: bool,
        class_name: str = "",
        cycle_time_s: float = 0.0,
        failure_reason: str = "",
        final_state: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write one trial result row to the CSV.

        Args:
            extra: Optional failure-context fields (CONTEXT_FIELDNAMES) — the
                orchestrator passes detected pose / confidence / frame path /
                pose_id here. Keys outside the file's schema are dropped.
        """
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
            "depth_mode": self.context.get("depth_mode", ""),   # C4 mode per run
        }
        if extra:
            row.update(extra)
        row["run_id"] = self.run_id
        # Restrict to the file's schema (old files → old columns; unknown keys dropped).
        row = {k: row.get(k, "") for k in self._fieldnames}
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self._fieldnames).writerow(row)
        self._rows.append(row)
        self.write_run_metadata()

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
