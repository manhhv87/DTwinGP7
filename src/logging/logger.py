"""
logger.py
─────────
TrialLogger — ghi kết quả từng trial pick-and-place ra file CSV.

Mỗi dòng CSV là một trial; cuối thí nghiệm dùng scripts/04_analyze_results.py
để phân tích thống kê. Có sẵn hàm summarize() build ma trận failure-mode
(phục vụ Discussion section của paper — mục 9.3 tài liệu).

Pure stdlib (csv) → không phụ thuộc pandas, dùng được mọi nơi.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Thứ tự cột trong CSV.
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
]


class TrialLogger:
    """Ghi log trial ra CSV (append-mode, an toàn khi chạy nhiều phiên).

    Args:
        csv_path: Đường dẫn file CSV output.
        extra_context: Dict các trường mô tả điều kiện thí nghiệm
            (lighting, overlap, mode) — gắn vào mọi dòng.
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

        # Ghi header nếu file chưa tồn tại.
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
        """Ghi một dòng kết quả trial vào CSV."""
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
        }
        self._rows.append(row)
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

        status = "OK" if success else f"FAIL ({failure_reason})"
        logger.info("Trial %d logged: %s", trial_id, status)

    def summarize(self) -> dict[str, Any]:
        """Tổng hợp kết quả phiên hiện tại (các trial đã log trong process này).

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
