#!/usr/bin/env python
"""
13_verify_vs_robodk.py
──────────────────────
Validate CLIENT-SIDE kinematics (`gp7_urdf` FK + DLS IK) against RoboDK GP7.

Contents:
  1. FK FIDELITY  — our FK(q) vs RoboDK SolveFK(q): position error (mm) + angle (°).
     Proves "our model == RoboDK model" (expected 0.00 mm).
  2. IK PRECISION — round-trip: generate pose from SolveFK(q), solve IK, measure
     position (mm) + orientation (rad) + time (ms) via FK back-check.
     RoboDK SolveIK (analytical) vs our DLS.
  3. HISTOGRAM (optional) — when --samples ≥ 100 or --histogram, plot error and
     timing distributions as a 2×2 grid → PNG figure for thesis.

Usage:
    python scripts/13_verify_vs_robodk.py                       # fixed config set (table)
    python scripts/13_verify_vs_robodk.py --samples 20          # + 20 random poses
    python scripts/13_verify_vs_robodk.py --samples 500 --histogram   # PNG figure
    python scripts/13_verify_vs_robodk.py --no-robodk           # client-side round-trip only
    python scripts/13_verify_vs_robodk.py --out figures/kin_verify.csv

RoboDK requirement: robot file at C:/RoboDK/Library/Yaskawa-GP7.robot (default). If
RoboDK Free shows "API calls limited" mid-run → close RoboDK and restart (new session
resets the counter). For large --samples (≥500) use --no-robodk (DLS-only) to avoid
hitting the quota.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestrator.kinematics import inverse_kinematics_seeded  # noqa: E402
from src.orchestrator.kinematics.urdf_chain import (  # noqa: E402
    forward_kinematics_urdf,
    gp7_urdf,
)
from src.utils import setup_logging, timestamp  # noqa: E402

DEFAULT_ROBOT_FILE = "C:/RoboDK/Library/Yaskawa-GP7.robot"

# Fixed joint configs (degrees) — deterministic for thesis table.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", type=int, default=0,
                   help="Number of additional random poses (+2 API calls each). Default 0.")
    p.add_argument("--seed", type=int, default=42, help="Seed RNG cho --samples.")
    p.add_argument("--robot-file", default=DEFAULT_ROBOT_FILE,
                   help="Path to .robot file (default %s)." % DEFAULT_ROBOT_FILE)
    p.add_argument("--ik-perturb-deg", type=float, default=10.0,
                   help="Perturb q_init away from q_target for round-trip IK (default 10°).")
    p.add_argument("--no-robodk", action="store_true",
                   help="Skip RoboDK — run client-side FK∘IK round-trip only.")
    p.add_argument("--out", default=None,
                   help="CSV output. Default figures/kin_verify_<ts>.csv")
    p.add_argument("--histogram", action="store_true",
                   help="Plot 2x2 histogram (pos/rot/time + summary) to PNG. "
                        "Auto-enabled when --samples ≥ 100.")
    p.add_argument("--out-png", default=None,
                   help="PNG output cho histogram. Default figures/kin_verify_<ts>.png")
    p.add_argument("--no-show", action="store_true",
                   help="Do not open matplotlib window (save PNG only).")
    return p.parse_args()


def mat_to_np(m) -> np.ndarray:
    """RoboDK Mat 4x4 → numpy (using index [i,j], robust across all versions)."""
    return np.array([[m[i, j] for j in range(4)] for i in range(4)], dtype=float)


def pos_err_mm(Ta: np.ndarray, Tb: np.ndarray) -> float:
    return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))


def rot_err_deg(Ta: np.ndarray, Tb: np.ndarray) -> float:
    Re = Ta[:3, :3] @ Tb[:3, :3].T
    c = np.clip((np.trace(Re) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def build_configs(args) -> list[tuple[str, list[float]]]:
    configs = list(FIXED_CONFIGS_DEG)
    if args.samples > 0:
        model = gp7_urdf()
        rng = np.random.RandomState(args.seed)
        lo = np.array([np.degrees(j.joint_min) for j in model.joints])
        hi = np.array([np.degrees(j.joint_max) for j in model.joints])
        margin = (hi - lo) * 0.15
        for i in range(args.samples):
            q = rng.uniform(lo + margin, hi - margin)
            configs.append((f"rnd{i:02d}", q.tolist()))
    return configs


def connect_robot(robot_file: str, log):
    """Connect to RoboDK and load robot file. Return robot item or None."""
    try:
        from robodk.robolink import Robolink
    except Exception as e:                                  # noqa: BLE001
        log.error("robodk package could not be imported: %s", e)
        return None
    if not Path(robot_file).exists():
        log.error("Robot file not found: %s", robot_file)
        return None
    try:
        from robodk.robolink import ITEM_TYPE_ROBOT
        rdk = Robolink()
        # REUSE an existing GP7 — do NOT wipe the user's open station (the project
        # uses RoboDK as the live viewport; the old blanket ItemList().Delete()
        # destroyed every robot/frame/tool/object unrecoverably).
        robot = None
        for it in (rdk.ItemList() or []):
            try:
                if it.Type() == ITEM_TYPE_ROBOT and "GP7" in it.Name():
                    robot = it
                    break
            except Exception:                               # noqa: BLE001
                continue
        if robot is None:                                   # none present → add it
            robot = rdk.AddFile(robot_file)
        if robot is None or not robot.Valid():
            log.error("No GP7 robot found and AddFile did not yield a valid robot.")
            return None
        robot.setName("Yaskawa GP7")
        log.info("RoboDK robot ready: %s", robot.Name())
        return robot
    except Exception as e:                                  # noqa: BLE001
        log.error("Failed to connect/load RoboDK: %s", e)
        return None


def main() -> int:
    args = parse_args()
    log = setup_logging("verify_vs_robodk", log_file=None)
    log.info("=" * 70)
    log.info("Kinematics verification — gp7_urdf FK + DLS IK vs RoboDK")
    log.info("=" * 70)

    model = gp7_urdf(base_xyz_mm=(0.0, 0.0, 0.0))           # base-relative = SolveFK frame
    configs = build_configs(args)
    robot = None if args.no_robodk else connect_robot(args.robot_file, log)
    use_rdk = robot is not None
    if not use_rdk and not args.no_robodk:
        log.warning("RoboDK unavailable — falling back to client-side round-trip mode.")

    rng = np.random.RandomState(args.seed)
    records: list[dict] = []

    # ── 1. FK FIDELITY ────────────────────────────────────────────────────
    log.info("─" * 70)
    log.info("FK FIDELITY  (ours tool0 vs RoboDK SolveFK, robot-base frame)")
    log.info("%-10s %14s %14s", "config", "pos diff(mm)", "rot diff(deg)")
    fk_pos, fk_rot = [], []
    for name, qd in configs:
        T_ours = forward_kinematics_urdf(model, [np.deg2rad(q) for q in qd])
        rec = {"config": name, "fk_pos_mm": np.nan, "fk_rot_deg": np.nan,
               "rdk_ik_mm": np.nan, "rdk_ik_rot_rad": np.nan, "rdk_ik_ms": np.nan,
               "dls_ik_mm": np.nan, "dls_ik_rot_rad": np.nan, "dls_ik_ms": np.nan}
        if use_rdk:
            try:
                T_rdk = mat_to_np(robot.SolveFK(qd))
                pe, re = pos_err_mm(T_ours, T_rdk), rot_err_deg(T_ours, T_rdk)
                rec["fk_pos_mm"], rec["fk_rot_deg"] = pe, re
                fk_pos.append(pe); fk_rot.append(re)
                log.info("%-10s %14.4f %14.4f", name, pe, re)
            except Exception as e:                          # noqa: BLE001
                log.warning("SolveFK failed (%s): %s", name, e)
        records.append(rec)
    if fk_pos:
        log.info("  → max pos diff=%.4f mm | max rot diff=%.4f deg | %s",
                 max(fk_pos), max(fk_rot),
                 "MATCHES RoboDK" if max(fk_pos) < 0.01 else "DIVERGES — check model")

    # ── 2. IK PRECISION (round-trip) ──────────────────────────────────────
    log.info("─" * 70)
    log.info("IK PRECISION round-trip  (FK(q)->pose ; IK(pose) ; FK back -> err)")
    hdr = "%-10s %18s %18s" % ("config", "RoboDK SolveIK(mm)", "ours DLS(mm)")
    log.info(hdr)
    ik_pdeg = args.ik_perturb_deg
    for rec, (name, qd) in zip(records, configs):
        q = np.array(qd, dtype=float)
        q_init = (q + rng.uniform(-ik_pdeg, ik_pdeg, 6)).tolist()
        # ours
        T_ours = forward_kinematics_urdf(model, [np.deg2rad(x) for x in qd])
        t0 = time.perf_counter()
        sol_o = inverse_kinematics_seeded(model, T_ours, [np.deg2rad(x) for x in q_init])
        rec["dls_ik_ms"] = (time.perf_counter() - t0) * 1000
        if sol_o is not None:
            T_back = forward_kinematics_urdf(model, sol_o)
            rec["dls_ik_mm"] = pos_err_mm(T_back, T_ours)
            rec["dls_ik_rot_rad"] = np.deg2rad(rot_err_deg(T_back, T_ours))
        # RoboDK
        if use_rdk:
            try:
                T_rdk = robot.SolveFK(qd)
                T_rdk_np = mat_to_np(T_rdk)
                t0 = time.perf_counter()
                sol = robot.SolveIK(T_rdk, q_init)
                rec["rdk_ik_ms"] = (time.perf_counter() - t0) * 1000
                sj = sol.list() if hasattr(sol, "list") else list(sol)
                if sj and len(sj) >= 6:
                    T_back_rdk = mat_to_np(robot.SolveFK(list(sj)[:6]))
                    rec["rdk_ik_mm"] = pos_err_mm(T_back_rdk, T_rdk_np)
                    rec["rdk_ik_rot_rad"] = np.deg2rad(rot_err_deg(T_back_rdk, T_rdk_np))
            except Exception as e:                          # noqa: BLE001
                log.debug("SolveIK failed (%s): %s", name, e)
        log.info("%-10s %18s %18s", name,
                 "%.5f" % rec["rdk_ik_mm"] if not np.isnan(rec["rdk_ik_mm"]) else "-",
                 "%.5f" % rec["dls_ik_mm"] if not np.isnan(rec["dls_ik_mm"]) else "FAIL")

    # ── CSV ───────────────────────────────────────────────────────────────
    ts = timestamp()
    out_csv = args.out or str(PROJECT_ROOT / f"figures/kin_verify_{ts}.csv")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)
    log.info("─" * 70)
    log.info("CSV: %s", out_csv)

    # ── 3. HISTOGRAM (optional) ───────────────────────────────────────────
    want_hist = args.histogram or args.samples >= 100
    if want_hist:
        out_png = args.out_png or str(PROJECT_ROOT / f"figures/kin_verify_{ts}.png")
        _plot_histogram(records, out_png, use_rdk, args, log, ts)
    return 0


def _plot_histogram(records, out_png, use_rdk, args, log, ts):
    """Plot 2x2 grid: position/orientation error + compute time + summary."""
    try:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping histogram (CSV still written).")
        return

    dls_pos = np.array([r["dls_ik_mm"] for r in records])
    dls_rot = np.array([r["dls_ik_rot_rad"] for r in records])
    dls_ms = np.array([r["dls_ik_ms"] for r in records])
    dls_pos = dls_pos[~np.isnan(dls_pos)]
    dls_rot = dls_rot[~np.isnan(dls_rot)]
    dls_ms = dls_ms[~np.isnan(dls_ms)]
    if dls_pos.size == 0:
        log.warning("No valid DLS samples — skipping histogram.")
        return

    rdk_pos = np.array([r["rdk_ik_mm"] for r in records])
    rdk_rot = np.array([r["rdk_ik_rot_rad"] for r in records])
    rdk_ms = np.array([r["rdk_ik_ms"] for r in records])
    rdk_pos = rdk_pos[~np.isnan(rdk_pos)]
    rdk_rot = rdk_rot[~np.isnan(rdk_rot)]
    rdk_ms = rdk_ms[~np.isnan(rdk_ms)]

    n = len(records)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"IK Verification — DLS vs RoboDK ({n} configs)", fontsize=13)

    # Position error
    ax = axes[0, 0]
    bins = np.linspace(0, max(dls_pos.max(), 0.2), 50)
    ax.hist(dls_pos, bins=bins, alpha=0.7, label="DLS (client-side)",
            color="#2E7D32", edgecolor="white")
    if use_rdk and rdk_pos.size:
        ax.hist(rdk_pos, bins=bins, alpha=0.5, label="RoboDK",
                color="#1565C0", edgecolor="white")
    ax.axvline(0.1, color="red", linestyle="--", label="DLS tol (0.1mm)")
    ax.set_xlabel("Position error (mm)"); ax.set_ylabel("Count")
    ax.set_title("Position error distribution"); ax.legend(); ax.grid(alpha=0.3)

    # Orientation error
    ax = axes[0, 1]
    rot_max = max(dls_rot.max() * 1e3, 0.5) if dls_rot.size else 0.5
    bins = np.linspace(0, rot_max, 50)
    ax.hist(dls_rot * 1e3, bins=bins, alpha=0.7, label="DLS",
            color="#2E7D32", edgecolor="white")
    if use_rdk and rdk_rot.size:
        ax.hist(rdk_rot * 1e3, bins=bins, alpha=0.5, label="RoboDK",
                color="#1565C0", edgecolor="white")
    ax.axvline(0.1, color="red", linestyle="--", label="DLS tol (1e-4 rad)")
    ax.set_xlabel("Orientation error (×10⁻³ rad)"); ax.set_ylabel("Count")
    ax.set_title("Orientation error distribution"); ax.legend(); ax.grid(alpha=0.3)

    # Compute time
    ax = axes[1, 0]
    ax.hist(dls_ms, bins=50, alpha=0.7, label="DLS",
            color="#2E7D32", edgecolor="white")
    if use_rdk and rdk_ms.size:
        ax.hist(rdk_ms, bins=50, alpha=0.5, label="RoboDK",
                color="#1565C0", edgecolor="white")
    ax.set_xlabel("Time per IK call (ms)"); ax.set_ylabel("Count")
    ax.set_title("Compute time distribution"); ax.legend(); ax.grid(alpha=0.3)

    # Summary
    ax = axes[1, 1]; ax.axis("off")
    lines = [
        f"Configs: {n}  (fixed + samples)",
        f"q_init perturb: ±{args.ik_perturb_deg}°",
        "",
        "─── DLS (client-side, pure Python) ───",
        f"Success rate: {100*dls_pos.size/n:.1f}%",
        f"Position err median: {np.median(dls_pos):.4f} mm",
        f"Position err p95:    {np.percentile(dls_pos, 95):.4f} mm",
        f"Position err max:    {dls_pos.max():.4f} mm",
        f"Time per call (med): {np.median(dls_ms):.2f} ms",
    ]
    if use_rdk and rdk_pos.size:
        lines += [
            "",
            "─── RoboDK SolveIK (analytical) ───",
            f"Success rate: {100*rdk_pos.size/n:.1f}%",
            f"Position err median: {np.median(rdk_pos):.4f} mm",
            f"Position err p95:    {np.percentile(rdk_pos, 95):.4f} mm",
        ]
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes,
            fontsize=10, family="monospace", verticalalignment="top")

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    log.info("Figure: %s", out_png)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    sys.exit(main())
