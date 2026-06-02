#!/usr/bin/env python
"""
17_compare_fk_ik.py
───────────────────
Compare **FK** and **IK** across 6 solvers to select the algorithm for the thesis:
  1. **Pieper** (analytical closed-form, GP7-specific) — Tier S, default for app
  2. **DLS** (Damped Least Squares, fixed λ²) — baseline numerical
  3. **LM** (Levenberg-Marquardt, adaptive λ) — improved DLS
  4. **SDLS** (Selectively Damped LS, SVD-based) — Buss & Kim 2005
  5. **BFGS** (Quasi-Newton, scipy L-BFGS-B) — second-order convergence
  6. **RoboDK** SolveFK/SolveIK — reference industrial implementation

Contents:
  • FK fidelity: compare TCP position (mm) + angle (°) between methods for same joints
  • IK precision: round-trip FK(q_target) → T → IK(T, q_init) → q_sol;
    measure FK(q_sol) vs T error (mm + rad) + time (ms)
  • Multi-solution: Pieper returns 3-8 configs/pose; others return 1
  • Distribution plot: histogram pos_err + time per method

Output:
  • CSV per-row detail: figures/compare_fk_ik_<ts>.csv
  • PNG plot grid 2×3: figures/compare_fk_ik_<ts>.png

Usage:
    python scripts/17_compare_fk_ik.py                       # 8 fixed + 100 random
    python scripts/17_compare_fk_ik.py --samples 500         # more samples
    python scripts/17_compare_fk_ik.py --no-robodk           # skip RoboDK (free quota)
    python scripts/17_compare_fk_ik.py --methods Pieper,DLS,BFGS  # select subset only
    python scripts/17_compare_fk_ik.py --no-show             # do not open matplotlib window
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestrator.kinematics.inverse_kinematics import (  # noqa: E402
    inverse_kinematics,
    inverse_kinematics_bfgs,
    inverse_kinematics_lm,
    inverse_kinematics_sdls,
    inverse_kinematics_seeded,
)
from src.orchestrator.kinematics.pieper_gp7 import (  # noqa: E402
    inverse_kinematics_pieper_gp7,
    inverse_kinematics_pieper_gp7_nearest,
)
from src.orchestrator.kinematics.urdf_chain import (  # noqa: E402
    forward_kinematics_urdf, gp7_urdf,
)
from src.utils import setup_logging, timestamp  # noqa: E402

# Method registry: name → (callable_fn, color_for_plot)
# Each fn signature: (model, T_target, q_init_deg) → (sol_deg | None, n_solutions, time_ms)
ALL_METHODS = ["Pieper", "DLS", "LM", "SDLS", "BFGS", "RoboDK"]
METHOD_COLORS = {
    "Pieper":  "tab:blue",
    "DLS":     "tab:orange",
    "LM":      "tab:red",
    "SDLS":    "tab:purple",
    "BFGS":    "tab:brown",
    "RoboDK":  "tab:green",
}

DEFAULT_ROBOT_FILE = "C:/RoboDK/Library/Yaskawa-GP7.robot"

# 8 fixed configs — deterministic for the table. Reused from scripts/13_*.py.
FIXED_CONFIGS_DEG = [
    ("zero",  [0, 0, 0, 0, 0, 0]),
    ("J1+30", [30, 0, 0, 0, 0, 0]),
    ("J2+45", [0, 45, 0, 0, 0, 0]),
    ("J3-30", [0, 0, -30, 0, 0, 0]),
    ("J5+90", [0, 0, 0, 0, 90, 0]),
    ("rand1", [15, -30, 45, 10, -20, 30]),
    ("rand2", [-45, 20, -15, 60, 30, -25]),
    ("home",  [1.97, 2.26, -4.99, -181.93, 41.7, -90.35]),
]


# ── CLI ────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", type=int, default=100,
                   help="Number of random poses to add on top of the 8 fixed configs. Default 100.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed RNG cho random samples. Default 42.")
    p.add_argument("--ik-perturb-deg", type=float, default=10.0,
                   help="Perturb q_init away from q_target (deg) for round-trip IK. Default 10°.")
    p.add_argument("--robot-file", default=DEFAULT_ROBOT_FILE,
                   help=f"Path to RoboDK .robot file. Default {DEFAULT_ROBOT_FILE}.")
    p.add_argument("--no-robodk", action="store_true",
                   help="Skip RoboDK (when RoboDK is unavailable or quota is exhausted).")
    p.add_argument("--methods", default="Pieper,DLS,LM,SDLS,BFGS,RoboDK",
                   help="Comma-separated subset of methods. Default: all 6.")
    p.add_argument("--fair", action="store_true",
                   help="Fair mode: DLS uses single-shot (no seeded retry), "
                        "same max_iter for all → pure algorithm comparison. "
                        "Default OFF = production mode (DLS with seeded retry, "
                        "matching how app actually uses each).")
    p.add_argument("--no-plot", action="store_true",
                   help="Do not generate PNG histogram.")
    p.add_argument("--no-show", action="store_true",
                   help="Do not open matplotlib window; save PNG only.")
    p.add_argument("--out-csv", default=None,
                   help="CSV output path. Default figures/compare_fk_ik_<ts>.csv")
    p.add_argument("--out-png", default=None,
                   help="PNG output path. Default figures/compare_fk_ik_<ts>.png")
    return p.parse_args()


# ── RoboDK helpers ─────────────────────────────────────────────────────


def setup_robodk(robot_file: str, log):
    """Connect RoboDK + load GP7 robot. Return robot item or None on failure."""
    try:
        from robodk.robolink import ITEM_TYPE_ROBOT, Robolink
        rdk = Robolink()
        items = rdk.ItemList()
        for it in items:
            try:
                if it.Type() == ITEM_TYPE_ROBOT and "GP7" in it.Name():
                    log.info("RoboDK robot found: %s", it.Name())
                    return rdk, it
            except Exception:
                continue
        # Try loading from file
        import os
        if os.path.exists(robot_file):
            it = rdk.AddFile(robot_file)
            log.info("RoboDK loaded robot from %s", robot_file)
            return rdk, it
        log.warning("RoboDK reachable but no GP7 robot — set --no-robodk")
        return None, None
    except Exception as e:                                  # noqa: BLE001
        log.warning("RoboDK not reachable: %s — use --no-robodk", e)
        return None, None


def rdk_to_np(m) -> np.ndarray:
    """RoboDK Mat 4×4 → numpy."""
    return np.array([[m[i, j] for j in range(4)] for i in range(4)], dtype=float)


def np_to_rdk(arr: np.ndarray):
    from robodk.robomath import Mat
    return Mat([list(arr[i, :]) for i in range(4)])


def rdk_sol_to_list(s) -> list[float]:
    """RoboDK SolveIK output (Mat column 6×1) → list[float] degrees."""
    return [float(s[i, 0]) for i in range(6)]


# ── Error helpers ──────────────────────────────────────────────────────


def pose_diff(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """Return (pos_err_mm, rot_err_rad) between 2 poses 4×4."""
    pos = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    R_err = T_a[:3, :3] @ T_b[:3, :3].T
    cos_t = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
    rot = float(np.arccos(cos_t))
    return pos, rot


# ── Method runners ─────────────────────────────────────────────────────


def run_fk_ours(model, q_deg: list[float]) -> tuple[np.ndarray, float]:
    """FK qua URDF chain. Return (T_4x4, time_ms)."""
    q_rad = np.deg2rad(q_deg)
    t0 = time.perf_counter()
    T = forward_kinematics_urdf(model, q_rad)
    dt = (time.perf_counter() - t0) * 1000
    return T, dt


def run_ik_pieper(model, T_target: np.ndarray, q_init_deg: list[float]
                   ) -> tuple[list[float] | None, int, float]:
    """Pieper analytical IK + pick nearest. Return (sol_deg, n_solutions, time_ms)."""
    q_init_rad = np.deg2rad(q_init_deg)
    t0 = time.perf_counter()
    all_sols = inverse_kinematics_pieper_gp7(model, T_target)
    dt = (time.perf_counter() - t0) * 1000
    if not all_sols:
        return None, 0, dt
    best = None; best_d = float("inf")
    for s in all_sols:
        d = float(np.max(np.abs(np.asarray(s) - q_init_rad)))
        if d < best_d: best_d = d; best = s
    return [math.degrees(q) for q in best], len(all_sols), dt


def _run_ik_numerical(fn, model, T_target, q_init_deg):
    """Common runner for DLS / LM / SDLS / BFGS (single solution methods)."""
    q_init_rad = np.deg2rad(q_init_deg)
    t0 = time.perf_counter()
    sol = fn(model, T_target, q_init_rad.tolist(),
              tol_mm=0.5, tol_rad=1e-3, max_iter=100)
    dt = (time.perf_counter() - t0) * 1000
    if sol is None: return None, 1, dt
    return [math.degrees(q) for q in sol], 1, dt


def run_ik_dls(model, T_target, q_init_deg, fair: bool = False):
    """DLS. Production mode (default): seeded retry (~12 seeds). Fair mode:
    single-shot inverse_kinematics (no retry) — apples-to-apples vs LM/SDLS."""
    fn = inverse_kinematics if fair else inverse_kinematics_seeded
    return _run_ik_numerical(fn, model, T_target, q_init_deg)


def run_ik_lm(model, T_target, q_init_deg):
    return _run_ik_numerical(inverse_kinematics_lm, model, T_target, q_init_deg)


def run_ik_sdls(model, T_target, q_init_deg):
    return _run_ik_numerical(inverse_kinematics_sdls, model, T_target, q_init_deg)


def run_ik_bfgs(model, T_target, q_init_deg):
    return _run_ik_numerical(inverse_kinematics_bfgs, model, T_target, q_init_deg)


def run_ik_robodk(robot, T_target: np.ndarray, q_init_deg: list[float]
                   ) -> tuple[list[float] | None, int, float]:
    """RoboDK SolveIK. Return (sol_deg, n_sols=1, time_ms)."""
    T_rdk = np_to_rdk(T_target)
    t0 = time.perf_counter()
    try:
        s = robot.SolveIK(T_rdk, list(q_init_deg))
        dt = (time.perf_counter() - t0) * 1000
        sol = rdk_sol_to_list(s)
        T_back = rdk_to_np(robot.SolveFK(sol))
        pos, _ = pose_diff(T_back, T_target)
        if pos > 10.0:
            return None, 1, dt
        return sol, 1, dt
    except Exception:
        dt = (time.perf_counter() - t0) * 1000
        return None, 1, dt


# ── Test runner ────────────────────────────────────────────────────────


def evaluate(
    model, robot, configs: list[tuple[str, list[float]]],
    perturb_deg: float, methods: list[str], fair: bool, log,
) -> list[dict]:
    """Run all selected methods on each config, collect per-row results."""
    rows: list[dict] = []
    rng = np.random.RandomState(42)

    method_runners = {
        "Pieper": lambda m, T, q: run_ik_pieper(m, T, q),
        "DLS":    lambda m, T, q: run_ik_dls(m, T, q, fair=fair),
        "LM":     lambda m, T, q: run_ik_lm(m, T, q),
        "SDLS":   lambda m, T, q: run_ik_sdls(m, T, q),
        "BFGS":   lambda m, T, q: run_ik_bfgs(m, T, q),
    }

    for cfg_name, q_deg in configs:
        # FK section
        T_ours, fk_t_ours = run_fk_ours(model, q_deg)
        if robot is not None:
            t0 = time.perf_counter()
            T_rdk = rdk_to_np(robot.SolveFK(list(q_deg)))
            fk_t_rdk = (time.perf_counter() - t0) * 1000
            fk_diff_pos, fk_diff_rot = pose_diff(T_ours, T_rdk)
        else:
            fk_t_rdk = float("nan")
            fk_diff_pos = float("nan"); fk_diff_rot = float("nan")

        # IK: perturb q_init away from q_target
        perturb = rng.uniform(-perturb_deg, perturb_deg, 6)
        q_init_deg = [q + p for q, p in zip(q_deg, perturb)]

        row = {
            "config": cfg_name,
            "fk_diff_pos_mm": fk_diff_pos,
            "fk_diff_rot_deg": (math.degrees(fk_diff_rot) if not math.isnan(fk_diff_rot)
                                else float("nan")),
            "fk_ours_ms": fk_t_ours,
            "fk_rdk_ms": fk_t_rdk,
        }

        for m_name in methods:
            prefix = f"ik_{m_name.lower()}"
            if m_name == "RoboDK":
                if robot is None:
                    row[f"{prefix}_pos_mm"] = float("nan")
                    row[f"{prefix}_rot_arcsec"] = float("nan")
                    row[f"{prefix}_n_sols"] = 0
                    row[f"{prefix}_ms"] = float("nan")
                    continue
                sol, n_sols, t_ms = run_ik_robodk(robot, T_ours, q_init_deg)
            else:
                if m_name not in method_runners:
                    continue
                sol, n_sols, t_ms = method_runners[m_name](model, T_ours, q_init_deg)
            if sol is not None:
                if m_name == "RoboDK":
                    T_back = rdk_to_np(robot.SolveFK(sol))
                else:
                    T_back, _ = run_fk_ours(model, sol)
                pos, rot = pose_diff(T_back, T_ours)
                row[f"{prefix}_pos_mm"] = pos
                row[f"{prefix}_rot_arcsec"] = math.degrees(rot) * 3600
            else:
                row[f"{prefix}_pos_mm"] = float("nan")
                row[f"{prefix}_rot_arcsec"] = float("nan")
            row[f"{prefix}_n_sols"] = n_sols
            row[f"{prefix}_ms"] = t_ms
        rows.append(row)
    return rows


def gen_random_configs(n: int, model, seed: int) -> list[tuple[str, list[float]]]:
    """Generate n random configs within joint limits."""
    rng = np.random.RandomState(seed)
    out: list[tuple[str, list[float]]] = []
    for i in range(n):
        q = [rng.uniform(j.joint_min, j.joint_max) for j in model.joints]
        out.append((f"rand_{i:03d}", [math.degrees(v) for v in q]))
    return out


# ── Reporting ──────────────────────────────────────────────────────────


def print_fixed_table(rows: list[dict], methods: list[str], log) -> None:
    """Print table for 8 fixed configs with one pos err column per method."""
    log.info("=" * 130)
    log.info("FIXED CONFIGS — FK fidelity (vs RoboDK if loaded) + IK round-trip pos err (perturbed q_init)")
    log.info("=" * 130)
    cols = ["Config", "FK diff"] + [f"{m} pos" for m in methods]
    hdr = " | ".join(f"{c:>11s}" for c in cols)
    log.info(hdr)
    log.info("-" * len(hdr))
    n_fixed = 8
    for r in rows[:n_fixed]:
        fk_d = f'{r["fk_diff_pos_mm"]:.2e}' if not math.isnan(r["fk_diff_pos_mm"]) else "—"
        cells = [f'{r["config"]:>11s}', f'{fk_d:>11s}']
        for m in methods:
            prefix = f"ik_{m.lower()}"
            v = r.get(f"{prefix}_pos_mm", float("nan"))
            cells.append(f'{v:.2e}'.rjust(11) if not math.isnan(v) else "FAIL".rjust(11))
        log.info(" | ".join(cells))


def print_distribution_summary(rows: list[dict], methods: list[str], log) -> None:
    """Print distribution statistics summary for all rows (fixed + random)."""
    log.info("")
    log.info("=" * 120)
    log.info(f"DISTRIBUTION SUMMARY (n={len(rows)} poses)")
    log.info("=" * 120)
    log.info(f'{"Method":>10s} | {"Success":>10s} | '
             f'{"pos median":>13s} {"pos p95":>11s} {"pos max":>11s} | '
             f'{"time median":>13s} {"time p95":>11s} {"time max":>11s}')
    log.info("-" * 120)
    for m in methods:
        prefix = f"ik_{m.lower()}"
        pos_arr = np.array([r[f"{prefix}_pos_mm"] for r in rows
                            if not math.isnan(r.get(f"{prefix}_pos_mm", float("nan")))])
        time_arr = np.array([r[f"{prefix}_ms"] for r in rows
                             if not math.isnan(r.get(f"{prefix}_ms", float("nan")))])
        if len(pos_arr) == 0:
            log.info(f'{m:>10s} |   0/{len(rows):<5d} |  all FAIL')
            continue
        log.info(
            f'{m:>10s} | {len(pos_arr):>4d}/{len(rows):<5d} | '
            f'{np.median(pos_arr):>10.3e}mm '
            f'{np.percentile(pos_arr, 95):>8.3e}mm '
            f'{pos_arr.max():>8.3e}mm | '
            f'{np.median(time_arr):>10.3f}ms '
            f'{np.percentile(time_arr, 95):>8.3f}ms '
            f'{time_arr.max():>8.3f}ms')


def save_csv(rows: list[dict], path: Path, log) -> None:
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log.info("CSV saved → %s", path)


def plot_distributions(rows: list[dict], methods: list[str], out_png: Path,
                        show: bool, log) -> None:
    try:
        import matplotlib
        if not show: matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    method_keys = [(m, f"ik_{m.lower()}", METHOD_COLORS.get(m, "gray"))
                   for m in methods]

    # Row 0: pos err histogram (log scale)
    ax = axes[0, 0]
    for name, prefix, color in method_keys:
        arr = np.array([r[f"{prefix}_pos_mm"] for r in rows
                         if not math.isnan(r.get(f"{prefix}_pos_mm", float("nan")))])
        if len(arr) > 0:
            # Log scale: clip min to avoid log(0)
            arr = np.clip(arr, 1e-15, None)
            ax.hist(np.log10(arr), bins=40, alpha=0.5, label=name, color=color)
    ax.set_xlabel("log10(IK pos err, mm)")
    ax.set_ylabel("count")
    ax.set_title("IK position error distribution")
    ax.legend(); ax.grid(alpha=0.3)

    # Row 0 col 1: rot err histogram
    ax = axes[0, 1]
    for name, prefix, color in method_keys:
        arr = np.array([r[f"{prefix}_rot_arcsec"] for r in rows
                         if not math.isnan(r.get(f"{prefix}_rot_arcsec", float("nan")))])
        if len(arr) > 0:
            arr = np.clip(arr, 1e-6, None)
            ax.hist(np.log10(arr), bins=40, alpha=0.5, label=name, color=color)
    ax.set_xlabel("log10(IK rot err, arc-sec)")
    ax.set_ylabel("count")
    ax.set_title("IK rotation error distribution")
    ax.legend(); ax.grid(alpha=0.3)

    # Row 0 col 2: time histogram
    ax = axes[0, 2]
    for name, prefix, color in method_keys:
        arr = np.array([r[f"{prefix}_ms"] for r in rows
                         if not math.isnan(r.get(f"{prefix}_ms", float("nan")))])
        if len(arr) > 0:
            ax.hist(arr, bins=40, alpha=0.5, label=name, color=color)
    ax.set_xlabel("IK time (ms)")
    ax.set_ylabel("count")
    ax.set_title("IK time distribution")
    ax.legend(); ax.grid(alpha=0.3)

    # Row 1: bar chart median/p95/max
    metrics = ["pos median (mm)", "pos p95 (mm)", "time median (ms)"]
    keys = ["pos_mm", "pos_mm", "ms"]
    stats = ["median", "p95", "median"]
    for col, (metric, key, stat) in enumerate(zip(metrics, keys, stats)):
        ax = axes[1, col]
        names = []; vals = []; colors = []
        for name, prefix, color in method_keys:
            arr = np.array([r[f"{prefix}_{key}"] for r in rows
                             if not math.isnan(r.get(f"{prefix}_{key}", float("nan")))])
            if len(arr) > 0:
                val = np.median(arr) if stat == "median" else np.percentile(arr, 95)
                names.append(name); vals.append(val); colors.append(color)
        ax.bar(names, vals, color=colors)
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(alpha=0.3, axis="y")
        # Log scale cho error metrics (huge dynamic range)
        if "pos" in metric:
            ax.set_yscale("log")

    method_list = " vs ".join(methods)
    fig.suptitle(
        f"GP7 IK comparison: {method_list}  (n={len(rows)} poses)",
        fontsize=13)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    log.info("PNG saved → %s", out_png)
    if show: plt.show()
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    log = setup_logging("compare_fk_ik", log_file=None)
    ts = timestamp()

    model = gp7_urdf()

    # Parse method selection
    requested = [m.strip() for m in args.methods.split(",") if m.strip()]
    invalid = [m for m in requested if m not in ALL_METHODS]
    if invalid:
        log.error("Invalid methods: %s. Valid: %s", invalid, ALL_METHODS)
        return 2

    # RoboDK setup (only when RoboDK is in methods and --no-robodk is not set)
    rdk = None; robot = None; have_rdk = False
    if "RoboDK" in requested and not args.no_robodk:
        rdk, robot = setup_robodk(args.robot_file, log)
        have_rdk = robot is not None
        if not have_rdk:
            log.warning("RoboDK requested but not reachable — dropping from methods")
            requested = [m for m in requested if m != "RoboDK"]
    elif "RoboDK" in requested and args.no_robodk:
        requested = [m for m in requested if m != "RoboDK"]

    log.info("Model: GP7 URDF chain (%d joints)", model.num_joints())
    log.info("Methods: %s", requested)
    log.info("Mode: %s", "FAIR (single-shot, no retry)" if args.fair
              else "PRODUCTION (DLS seeded retry, BFGS early-exit at tol)")
    log.info("RoboDK: %s", "ON" if have_rdk else "OFF")
    log.info("Perturb q_init: ±%.1f°", args.ik_perturb_deg)
    log.info("Random samples: %d (seed=%d)", args.samples, args.seed)

    # Configs
    configs = list(FIXED_CONFIGS_DEG)
    if args.samples > 0:
        configs += gen_random_configs(args.samples, model, args.seed)
    log.info("Total configs: %d (%d fixed + %d random)",
             len(configs), len(FIXED_CONFIGS_DEG), args.samples)

    # Evaluate
    rows = evaluate(model, robot, configs, args.ik_perturb_deg,
                     requested, args.fair, log)

    # Reporting
    print_fixed_table(rows, requested, log)
    print_distribution_summary(rows, requested, log)

    # Save CSV
    csv_path = (Path(args.out_csv) if args.out_csv
                else PROJECT_ROOT / "figures" / f"compare_fk_ik_{ts}.csv")
    save_csv(rows, csv_path, log)

    # Plot
    if not args.no_plot:
        png_path = (Path(args.out_png) if args.out_png
                    else PROJECT_ROOT / "figures" / f"compare_fk_ik_{ts}.png")
        plot_distributions(rows, requested, png_path,
                            show=not args.no_show, log=log)

    return 0


if __name__ == "__main__":
    sys.exit(main())
