"""
failure_miner.py
────────────────
C3 step 1: mine grasp-failure modes from trial CSVs (TrialLogger output).

Default clustering is RULE-BASED BINNING over
    (staged class x card cell x lighting condition x failure category)
— interpretable at the small counts a physical campaign produces, and directly
mappable back into sampler conditions. A tiny numpy k-means over the staged
failure positions is provided as a SENSITIVITY CHECK only (paper §3.6).

The class and position are those STAGED for the trial, not those the detector
reported: a missed detection has neither, so binning by detector output puts
almost every failure of a hard session into one 'unknown' bin that constrains
lighting alone. stage_trials() joins each trial's pose_id to the pre-drawn pose
list named in the run metadata (03_run_experiment.py --pose-list) and takes the
class from class_hint and the cell from the card's x_mm/y_mm. The staged support
(stack_on) is carried as context; the sampler has no stacking control.

A trial fails by the operator's verdict (human_ok) when one was recorded, and
otherwise by the machine flag (success): a trial refused before motion has no
verdict and counts as a failure, as in the paper's endpoint.

Pure stdlib + numpy → fully unit-testable.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Failure categories that exist in the platform (orchestrator/state machine).
# Kept as documentation + validation warning; unknown reasons are still binned.
# operator_verdict: the operator scored the trial a failure although the machine
# recorded none (for example a part dropped in transfer).
KNOWN_REASONS = {
    "detection_timeout", "detection_miss", "unreachable",
    "predicted_joint_limit", "predicted_self_collision",
    "pick_z_safety_reject", "depth_geometry_reject", "yaw_geometry_reject",
    "gripper_timeout", "grasp_failed", "motion_error", "gripper_slip",
    "operator_verdict",
}

# Poses 221-300 of the hard list are reserved for adaptation trials; poses 1-200
# are the evaluation list and 201-220 the drift-control block (paper §3.7).
ADAPTATION_POSES = (221, 300)

FAILURE_DEFINITION = ("operator verdict human_ok when recorded, otherwise the machine "
                      "flag success; a trial refused before motion has no verdict")
BINNING = ("staged class (pose list class_hint) x card cell (pose list x_mm, y_mm) "
           "x logged lighting x failure category")

_POSE_ID = re.compile(r"^P(\d+)$")
_TRUE = {"1", "1.0", "true", "yes"}
_FALSE = {"0", "0.0", "false", "no"}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _flag(value: Any) -> bool | None:
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return None
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"Unreadable outcome value {value!r}")


def trial_outcome(row: dict[str, Any]) -> tuple[bool, str]:
    """(failed, source): the operator's verdict when recorded, else the machine flag."""
    verdict = _flag(row.get("human_ok"))
    if verdict is not None:
        return not verdict, "human_ok"
    machine = _flag(row.get("success"))
    if machine is None:
        raise ValueError(f"Trial {row.get('trial_id', '?')} of {row.get('_source', '?')} "
                         "has neither an operator verdict nor a machine success flag")
    return not machine, "success"


def pose_number(pose_id: Any) -> int:
    """'P0221' → 221."""
    match = _POSE_ID.match(str(pose_id or "").strip())
    if not match:
        raise ValueError(f"Unreadable pose_id {pose_id!r}; expected P followed by digits")
    return int(match.group(1))


def load_pose_list(path: str | Path) -> dict[str, dict[str, str]]:
    """Pose-list CSV (22_make_pose_lists.py) → {pose_id: row}."""
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"pose_id", "x_mm", "y_mm", "class_hint"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"Pose list {path} needs the columns {sorted(required)}")
    poses: dict[str, dict[str, str]] = {}
    for row in rows:
        pose_id = (row["pose_id"] or "").strip()
        if not pose_id or pose_id in poses:
            raise ValueError(f"Pose list {path}: empty or repeated pose_id {pose_id!r}")
        poses[pose_id] = row
    return poses


def _run_metadata(path: Path, run_ids: set[str]) -> tuple[Path | None, dict[str, Any] | None]:
    """The '<csv stem>.<run_id>.metadata.json' written beside a trial CSV, if any."""
    found = []
    for candidate in sorted(path.parent.glob(glob.escape(path.stem) + ".*.metadata.json")):
        run_id = candidate.name[len(path.stem) + 1:-len(".metadata.json")]
        if not run_ids or run_id in run_ids:
            found.append(candidate)
    if not found:
        return None, None
    if len(found) > 1:
        raise ValueError(f"{path.name}: several metadata files match: {[p.name for p in found]}")
    with found[0].open("r", encoding="utf-8") as f:
        return found[0], json.load(f)


def stage_trials(
    csv_paths: list[str | Path],
    *,
    pose_root: str | Path,
    pose_list: str | Path | None = None,
    poses: tuple[int, int] = ADAPTATION_POSES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read trial CSVs and join every trial to its staged pose.

    The pose list is the one named in each run's metadata, resolved against
    pose_root. pose_list replaces that location (for a CSV without metadata, or a
    list kept elsewhere); its file name must match the one the metadata names,
    since lists of one family can share pose IDs and differ in the card positions.
    The CSV's own condition and stack_on columns must agree with the pose list.
    Every trial must lie within `poses` (inclusive pose numbers): by default the
    adaptation poses, so evaluation and drift-control outcomes cannot feed the loop.

    Returns (rows, inputs): rows carry staged_class, staged_x_mm, staged_y_mm,
    card_id, stack_on and pose_number; inputs record each CSV with its SHA-256,
    metadata and pose list for the report.
    """
    lo, hi = poses
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    outside: list[str] = []
    pose_cache: dict[Path, dict[str, dict[str, str]]] = {}
    for p in csv_paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Trial CSV not found: {path}")
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            table = list(reader)
            columns = set(reader.fieldnames or [])
        run_ids = {(r.get("run_id") or "").strip() for r in table} - {""}
        if len(run_ids) > 1:
            raise ValueError(f"{path.name} mixes runs {sorted(run_ids)}")
        meta_path, meta = _run_metadata(path, run_ids)
        runtime = ((meta or {}).get("provenance") or {}).get("runtime") or {}
        named = runtime.get("pose_list")
        if pose_list is not None:
            list_path = Path(pose_list)
            if named and Path(str(named).replace("\\", "/")).name != list_path.name:
                raise ValueError(f"{path.name}: its metadata names the pose list {named}, "
                                 f"not {list_path.name}")
        elif named:
            list_path = Path(str(named).replace("\\", "/"))
            if not list_path.is_absolute():
                list_path = Path(pose_root) / list_path
        else:
            raise ValueError(f"{path.name}: no run metadata names a pose list; pass the "
                             "pose list explicitly")
        list_path = list_path.resolve()
        if not list_path.is_file():
            raise FileNotFoundError(f"{path.name}: pose list not found: {list_path}")
        if list_path not in pose_cache:
            pose_cache[list_path] = load_pose_list(list_path)
        staged_poses = pose_cache[list_path]

        for row in table:
            pose_id = (row.get("pose_id") or "").strip()
            pose = staged_poses.get(pose_id)
            if pose is None:
                raise ValueError(f"{path.name} trial {row.get('trial_id', '?')}: pose_id "
                                 f"{pose_id!r} is not in {list_path.name}")
            for column in ("condition", "stack_on"):
                if column in columns and (row.get(column) or "").strip() != (pose.get(column) or "").strip():
                    raise ValueError(f"{path.name} {pose_id}: logged {column} "
                                     f"{row.get(column)!r} differs from {list_path.name} "
                                     f"{pose.get(column)!r}; wrong pose list?")
            number = pose_number(pose_id)
            if not lo <= number <= hi:
                outside.append(f"{path.name}:{pose_id}")
            staged = dict(row)
            staged.update({
                "_source": path.name,
                "pose_number": number,
                "staged_class": (pose.get("class_hint") or "").strip() or "unknown",
                "staged_x_mm": float(pose["x_mm"]),
                "staged_y_mm": float(pose["y_mm"]),
                "card_id": (pose.get("card_id") or "").strip(),
                "stack_on": (pose.get("stack_on") or "").strip(),
            })
            rows.append(staged)
        numbers = [pose_number(r.get("pose_id")) for r in table]
        inputs.append({
            "csv": str(path.resolve()),
            "sha256": file_sha256(path),
            "rows": len(table),
            "run_id": next(iter(run_ids), None),
            "pose_ids": [f"P{min(numbers):04d}", f"P{max(numbers):04d}"] if numbers else None,
            "metadata": None if meta_path is None else {
                "path": str(meta_path.resolve()), "sha256": file_sha256(meta_path),
                "pose_list": named, "pose_slice": runtime.get("pose_slice"),
            },
            "pose_list": {"path": str(list_path), "sha256": file_sha256(list_path)},
        })
    if outside:
        raise ValueError(
            f"{len(outside)} trial(s) lie outside poses {lo}-{hi}, e.g. {outside[:3]}; "
            "evaluation and drift-control trials must not feed the loop. Mine only the "
            "adaptation session's CSVs, or override the pose range explicitly.")
    logger.info("Staged %d trial rows from %d file(s)", len(rows), len(inputs))
    return rows, inputs


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

    Bins staged rows (stage_trials) by staged class and card cell. Rows without
    staged fields form explicit unknown class/region bins; detector outputs are
    never used in their place.

    Returns:
        {"n_trials", "n_failures", "failure_definition", "outcome_sources",
         "binning", "region", "grid", "clusters": [
            {"key", "class_name", "region_cell" | None, "region_bounds" | None,
             "lighting", "reason", "count", "frac", "stack_on", "pose_ids"} ...]}
        sorted by count desc (ties keep first-seen order). stack_on counts the
        staged supports of the bin's failures ("none" for a single layer).
    """
    bins: dict[tuple, list[dict[str, Any]]] = {}
    sources = {"human_ok": 0, "success": 0}
    n_fail = 0
    for r in rows:
        failed, source = trial_outcome(r)
        sources[source] += 1
        if not failed:
            continue
        n_fail += 1
        reason = (r.get("failure_reason") or "").strip()
        if not reason:
            reason = "operator_verdict" if source == "human_ok" else "unknown"
        if reason not in KNOWN_REASONS:
            logger.debug("Unknown failure_reason '%s' (still binned)", reason)
        cls = (r.get("staged_class") or "unknown").strip() or "unknown"
        lighting = (r.get("lighting") or "unlabelled").strip() or "unlabelled"
        try:
            x = float(r.get("staged_x_mm", ""))
            y = float(r.get("staged_y_mm", ""))
            cell: tuple[int, int] | None = _region_cell(x, y, region, grid)
        except (TypeError, ValueError):
            cell = None  # not staged: explicit unknown region
        bins.setdefault((cls, cell, lighting, reason), []).append(r)

    clusters = []
    for (cls, cell, lighting, reason), members in sorted(
        bins.items(), key=lambda kv: -len(kv[1])
    ):
        supports: dict[str, int] = {}
        for r in members:
            support = (r.get("stack_on") or "").strip() or "none"
            supports[support] = supports.get(support, 0) + 1
        clusters.append({
            "key": f"{cls}|{'r%d%d' % cell if cell else 'na'}|{lighting}|{reason}",
            "class_name": cls,
            "region_cell": list(cell) if cell else None,
            "region_bounds": _cell_bounds(cell, region, grid) if cell else None,
            "lighting": lighting,
            "reason": reason,
            "count": len(members),
            "frac": len(members) / n_fail,
            "stack_on": dict(sorted(supports.items())),
            "pose_ids": sorted({str(r.get("pose_id", "")) for r in members}),
        })
    return {
        "n_trials": len(rows),
        "n_failures": n_fail,
        "failure_definition": FAILURE_DEFINITION,
        "outcome_sources": sources,
        "binning": BINNING,
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
    """k-means over staged failure positions — SENSITIVITY CHECK only.

    Uses the staged card position (staged_x_mm, staged_y_mm) of failed trials,
    the same positions the binning uses. Returns None when there are fewer than k
    such points. Plain-numpy Lloyd iterations with a k-means++-style
    farthest-point init (no sklearn).
    """
    pts = []
    for r in rows:
        if not trial_outcome(r)[0]:
            continue
        try:
            pts.append([float(r["staged_x_mm"]), float(r["staged_y_mm"])])
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
        "position_source": "staged card position",
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
