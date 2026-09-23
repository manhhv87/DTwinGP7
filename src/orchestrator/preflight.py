"""
preflight.py
────────────
Checks run before any REAL-robot trial. Each guards a setting that, if wrong,
puts the jaws in the wrong place without raising any error:

  * the hand-eye matrix must be a calibration of this cell, not the simulation
    placeholder shipped with the repo (camera at x=700, z=1200);
  * the robot base pose used at calibration must equal the one used now: FK
    runs in the frame of that pose, so T_BC is expressed in it, and grasp
    targets are converted back with the current pose; a later edit of the cell
    layout would shift every grasp by the difference;
  * the table height must be known in that same frame, because it sets the
    floor below which the jaw tips never go. 02_run_calibration.py measures it
    from depth through the freshly solved T_BC, so it cannot be in another
    frame, and records which T_BC it used;
  * the part heights must be configured, since they cap the grasp depth;
  * the plane and fusion depth modes (C4) intersect viewing rays with the table
    plane, so they need that plane as measured with the current calibration, and
    the part sizes must agree with the configured heights.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SIM_PLACEHOLDER_T_BC = np.array([[1.0, 0.0, 0.0, 700.0],
                                 [0.0, -1.0, 0.0, 0.0],
                                 [0.0, 0.0, -1.0, 1200.0],
                                 [0.0, 0.0, 0.0, 1.0]])
META_SUFFIX = "_meta.json"
TABLE_PLANE_FILE = "table_plane.json"
# Invented values the repo ships with, to be replaced by measurements of the cell.
PLACEHOLDER_PLACE_XY = (700.0, 120.0)
PLACEHOLDER_TCP_XYZ = (0.0, 0.0, 100.0)


def is_sim_placeholder(T_BC) -> bool:
    """True if T_BC is the simulation placeholder, i.e. no calibration was done."""
    return bool(np.allclose(np.asarray(T_BC, float), SIM_PLACEHOLDER_T_BC, atol=1e-6))


def meta_path_for(calib_path) -> Path:
    """Where the calibration metadata sits next to the T_BC file."""
    p = Path(calib_path)
    return p.with_name(p.stem + META_SUFFIX)


def check_real_mode(config: dict, calib_path, robot_base_xyz_mm,
                    robot_base_rpy_deg,
                    tcp_offset_xyz_mm=None) -> tuple[list[str], float | None]:
    """Return (problems, table_top_z_mm). An empty problem list means go."""
    from .coord_conv import load_calibration

    problems: list[str] = []
    calib_path = Path(calib_path)
    try:
        T = load_calibration(calib_path)
    except (FileNotFoundError, ValueError) as e:
        return [f"hand-eye calibration unusable: {e}"], None
    if is_sim_placeholder(T):
        problems.append(
            f"{calib_path.name} is the simulation placeholder (camera at x=700, z=1200), "
            "not a calibration of this cell. Run scripts/02_run_calibration.py first.")

    meta_p = meta_path_for(calib_path)
    if not meta_p.exists():
        problems.append(
            f"{meta_p.name} missing: re-run scripts/02_run_calibration.py, which records "
            "the robot base pose the calibration was solved in.")
    else:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        xyz0 = np.asarray(meta.get("robot_base_xyz_mm", [np.nan] * 3), float)
        rpy0 = np.asarray(meta.get("robot_base_rpy_deg", [np.nan] * 3), float)
        xyz1 = np.asarray(robot_base_xyz_mm, float)
        rpy1 = np.asarray(robot_base_rpy_deg, float)
        if not (np.allclose(xyz0, xyz1, atol=0.5) and np.allclose(rpy0, rpy1, atol=0.01)):
            problems.append(
                f"robot base pose differs from calibration: now xyz={xyz1.tolist()} "
                f"rpy={rpy1.tolist()}, calibrated with xyz={xyz0.tolist()} "
                f"rpy={rpy0.tolist()}. Every grasp would shift by the difference; restore "
                "the cell layout or re-calibrate.")

    z = config.get("table_top_z_mm")
    if z is None:
        tp = calib_path.with_name(TABLE_PLANE_FILE)
        if tp.exists():
            plane = json.loads(tp.read_text(encoding="utf-8"))
            z = plane.get("z_top_mm")
            T_plane = plane.get("T_BC")
            if T_plane is not None and not np.allclose(np.asarray(T_plane, float), T, atol=1e-6):
                problems.append(
                    f"{TABLE_PLANE_FILE} was measured with a different T_BC than the "
                    "current calibration; re-measure it with 02_run_calibration.py "
                    "--table-only.")
        if z is None:
            problems.append(
                f"table height unknown: table_top_z_mm is null and {TABLE_PLANE_FILE} is "
                "missing. Measure it with scripts/02_run_calibration.py --table-only.")

    if not (config.get("class_heights_mm") or {}):
        problems.append("class_heights_mm is empty: the grasp depth would ignore part heights.")

    # Two numbers only the cell can give. Shipped as invented placeholders, and both
    # move the robot: the parts would be released over empty floor, and the safety
    # layer would predict the trajectory of a tool the arm does not carry.
    place = [float(v) for v in (config.get("place_position") or [])]
    if place[:2] == list(PLACEHOLDER_PLACE_XY):
        problems.append(
            "place_position is still the placeholder [700, 120, ...] in experiment.yaml: "
            "measure the drop point on the conveyor (guide A7) and write its X and Y.")
    if tcp_offset_xyz_mm is not None and (
            [float(v) for v in tcp_offset_xyz_mm] == list(PLACEHOLDER_TCP_XYZ)):
        problems.append(
            "gripper.tcp_offset_xyz_mm is still the placeholder [0, 0, 100] in the cell "
            "layout: measure flange to jaw tip and write it there and into TOOL01 on the "
            "pendant (guide A4).")
    return problems, (None if z is None else float(z))


def load_table_plane(calib_path) -> tuple[dict | None, list[str]]:
    """Read table_plane.json next to the calibration and check it belongs to it.

    Returns (plane, problems); plane is None whenever there is a problem.
    """
    from .coord_conv import load_calibration

    calib_path = Path(calib_path)
    tp = calib_path.with_name(TABLE_PLANE_FILE)
    if not tp.exists():
        return None, [f"{TABLE_PLANE_FILE} missing: measure the table with "
                      "scripts/02_run_calibration.py --table-only."]
    plane = json.loads(tp.read_text(encoding="utf-8"))
    problems: list[str] = []
    missing = [k for k in ("a", "b", "c") if k not in plane]
    if missing:
        problems.append(f"{TABLE_PLANE_FILE} lacks the plane coefficients {missing}; "
                        "re-measure it with 02_run_calibration.py --table-only.")
    T_plane = plane.get("T_BC")
    try:
        T = load_calibration(calib_path)
    except (FileNotFoundError, ValueError) as e:
        return None, problems + [f"hand-eye calibration unusable: {e}"]
    if T_plane is None:
        problems.append(f"{TABLE_PLANE_FILE} does not record the T_BC it was measured "
                        "with; re-measure it with 02_run_calibration.py --table-only.")
    elif not np.allclose(np.asarray(T_plane, float), T, atol=1e-6):
        problems.append(f"{TABLE_PLANE_FILE} was measured with a different T_BC than the "
                        "current calibration; re-measure it with 02_run_calibration.py "
                        "--table-only.")
    return (None if problems else plane), problems


def check_depth_mode(config: dict, calib_path,
                     depth_mode: str) -> tuple[list[str], dict | None]:
    """Return (problems, table plane) for the chosen depth mode (C4).

    plane and fusion need the table plane of the current calibration. rgbd uses it
    only to tell the carton sizes apart, so there a missing plane is not a problem
    (the plane comes back None and the configured height is used). In every mode
    the part sizes must parse, and the lowest height of each class must equal its
    entry in class_heights_mm, which sets the grasp depth.
    """
    from ..perception.depth_modes import DEPTH_MODES, parse_class_sizes

    if depth_mode not in DEPTH_MODES:
        return [f"unknown depth mode {depth_mode!r}; choose one of {DEPTH_MODES}."], None
    problems: list[str] = []
    try:
        sizes = parse_class_sizes(config.get("class_sizes_mm"))
    except ValueError as e:
        problems.append(f"class_sizes_mm: {e}")
        sizes = {}
    heights = config.get("class_heights_mm") or {}
    for cls, cands in sizes.items():
        h_low = min(c[2] for c in cands)
        if cls in heights and abs(float(heights[cls]) - h_low) > 0.5:
            problems.append(
                f"class_sizes_mm gives {cls} a lowest height of {h_low:g} mm but "
                f"class_heights_mm says {float(heights[cls]):g} mm; one of them is wrong.")
    plane, plane_problems = load_table_plane(calib_path)
    if depth_mode != "rgbd":
        problems += [f"depth mode {depth_mode}: {p}" for p in plane_problems]
    return problems, plane
