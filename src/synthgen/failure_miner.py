"""
failure_miner.py
────────────────
C3 step 1: mine grasp-failure modes from trial CSVs (TrialLogger output).

Default clustering is RULE-BASED BINNING over
    (product class x pick-region cell x lighting condition x failure category)
— interpretable at the small counts a physical campaign produces, and directly
mappable back into sampler conditions. A tiny numpy k-means over the continuous
failure coordinates is provided as a SENSITIVITY CHECK only (paper §3.6).

Consumes the context columns added to TrialLogger (det_x_mm, det_y_mm, ...);
rows from older CSVs without those columns still bin by (class, lighting,
reason) with region cell "na".

Pure stdlib + numpy → fully unit-testable.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Failure categories that exist in the platform (orchestrator/state machine).
# Kept as documentation + validation warning; unknown reasons are still binned.
KNOWN_REASONS = {
    "detection_timeout", "detection_miss", "unreachable",
    "predicted_joint_limit", "predicted_self_collision",
    "gripper_timeout", "grasp_failed", "motion_error", "gripper_slip",
}


def load_trials(csv_paths: list[str | Path]) -> list[dict[str, Any]]:
    """Read one or more trial CSVs → list of row dicts (strings as logged)."""
    rows: list[dict[str, Any]] = []
    for p in csv_paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Trial CSV not found: {path}")
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["_source"] = path.name
                rows.append(row)
    logger.info("Loaded %d trial rows from %d file(s)", len(rows), len(csv_paths))
    return rows


def _region_cell(
    x: float, y: float,
    region: tuple[float, float, float, float],
    grid: tuple[int, int],
) -> tuple[int, int]:
    """Map (x, y) mm → (ix, iy) grid cell index, clamped to the grid."""
    x0, x1, y0, y1 = region
    nx, ny = grid
    ix = int((x - x0) / (x1 - x0) * nx)
    iy = int((y - y0) / (y1 - y0) * ny)
    return min(max(ix, 0), nx - 1), min(max(iy, 0), ny - 1)


def _cell_bounds(
    cell: tuple[int, int],
    region: tuple[float, float, float, float],
    grid: tuple[int, int],
) -> list[float]:
    """Inverse of _region_cell: cell index → [x0, x1, y0, y1] bounds (mm)."""
    x0, x1, y0, y1 = region
    nx, ny = grid
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    ix, iy = cell
    return [x0 + ix * dx, x0 + (ix + 1) * dx, y0 + iy * dy, y0 + (iy + 1) * dy]


def bin_failures(
    rows: list[dict[str, Any]],
    region: tuple[float, float, float, float] = (400.0, 1000.0, -200.0, 200.0),
    grid: tuple[int, int] = (3, 3),
) -> dict[str, Any]:
    """Rule-based failure binning → report dict (JSON-serializable).

    Returns:
        {"n_trials", "n_failures", "region", "grid", "clusters": [
            {"key", "class_name", "region_cell" | None, "region_bounds" | None,
             "lighting", "reason", "count", "frac"} ...]}  sorted by count desc.
    """
    failures = [r for r in rows if str(r.get("success", "")).strip() in ("0", "False", "false")]
    bins: dict[tuple, int] = {}
    for r in failures:
        reason = (r.get("failure_reason") or "unknown").strip()
        if reason not in KNOWN_REASONS:
            logger.debug("Unknown failure_reason '%s' (still binned)", reason)
        cls = (r.get("class_name") or "unknown").strip() or "unknown"
        lighting = (r.get("lighting") or "unlabelled").strip() or "unlabelled"
        try:
            x = float(r.get("det_x_mm", ""))
            y = float(r.get("det_y_mm", ""))
            cell: tuple[int, int] | None = _region_cell(x, y, region, grid)
        except (TypeError, ValueError):
            cell = None  # detection_miss/timeout rows have no coordinates
        bins[(cls, cell, lighting, reason)] = bins.get((cls, cell, lighting, reason), 0) + 1

    n_fail = len(failures)
    clusters = []
    for (cls, cell, lighting, reason), count in sorted(
        bins.items(), key=lambda kv: -kv[1]
    ):
        clusters.append({
            "key": f"{cls}|{'r%d%d' % cell if cell else 'na'}|{lighting}|{reason}",
            "class_name": cls,
            "region_cell": list(cell) if cell else None,
            "region_bounds": _cell_bounds(cell, region, grid) if cell else None,
            "lighting": lighting,
            "reason": reason,
            "count": count,
            "frac": count / n_fail if n_fail else 0.0,
        })
    return {
        "n_trials": len(rows),
        "n_failures": n_fail,
        "region": list(region),
        "grid": list(grid),
        "clusters": clusters,
    }


def kmeans_sensitivity(
    rows: list[dict[str, Any]],
    k: int = 4,
    seed: int = 0,
    n_iter: int = 50,
) -> dict[str, Any] | None:
    """k-means over continuous failure coordinates — SENSITIVITY CHECK only.

    Uses (det_x_mm, det_y_mm) of failed trials that carry coordinates.
    Returns None when there are fewer than k such points. Plain-numpy Lloyd
    iterations with a k-means++-style farthest-point init (no sklearn).
    """
    pts = []
    for r in rows:
        if str(r.get("success", "")).strip() not in ("0", "False", "false"):
            continue
        try:
            pts.append([float(r["det_x_mm"]), float(r["det_y_mm"])])
        except (KeyError, TypeError, ValueError):
            continue
    if len(pts) < k:
        return None

    X = np.asarray(pts, dtype=float)
    rng = np.random.default_rng(seed)
    centers = [X[rng.integers(len(X))]]
    while len(centers) < k:  # farthest-point init
        d2 = np.min(
            [np.sum((X - c) ** 2, axis=1) for c in centers], axis=0
        )
        centers.append(X[int(np.argmax(d2))])
    C = np.asarray(centers)

    assign = np.zeros(len(X), dtype=int)
    for _ in range(n_iter):
        d2 = np.stack([np.sum((X - c) ** 2, axis=1) for c in C], axis=1)
        new_assign = np.argmin(d2, axis=1)
        if np.array_equal(new_assign, assign) and _ > 0:
            break
        assign = new_assign
        for j in range(k):
            members = X[assign == j]
            if len(members):
                C[j] = members.mean(axis=0)

    inertia = float(np.sum((X - C[assign]) ** 2))
    counts = [int(np.sum(assign == j)) for j in range(k)]
    return {
        "k": k,
        "n_points": len(X),
        "counts": counts,
        "centers_mm": C.round(1).tolist(),
        "inertia": round(inertia, 1),
    }


def save_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write a failure report (bin_failures output + extras) to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Failure report saved → %s", p)
    return p


def load_report(path: str | Path) -> dict[str, Any]:
    """Load a failure report JSON."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
