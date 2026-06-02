#!/usr/bin/env python
"""
05_analyze_telemetry.py
───────────────────────
Analyze + visualize CSV telemetry produced by DigitalTwinMirror.

Input:  `results/telemetry_*.csv` (columns: timestamp, j1..j6, running, alarm)
Output (default to `figures/`):
  - joint_trajectory_<ts>.png   : 6 joint angles vs time
  - joint_velocity_<ts>.png     : Numerical diff to show velocity profile
  - drift_events_<ts>.png       : Drift markers + alarms on timeline
  - cycle_time_<ts>.png         : Histogram of intervals between rest-poses

Usage:
    python scripts/05_analyze_telemetry.py results/telemetry_20260520_223842.csv
    python scripts/05_analyze_telemetry.py latest    # auto-pick the latest file
    python scripts/05_analyze_telemetry.py latest --no-show   # save PNG only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"

JOINT_LABELS = ["S", "L", "U", "R", "B", "T"]
JOINT_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "csv_path", nargs="?", default="latest",
        help="Path to CSV or 'latest' to select the newest file in results/",
    )
    p.add_argument("--no-show", action="store_true", help="Do not open figure window")
    p.add_argument("--out-dir", default=None, help="Output directory for PNGs (default: figures/)")
    p.add_argument(
        "--rest-threshold-deg-s", type=float, default=2.0,
        help="Joint speed threshold considered as rest (deg/s). Default 2.0.",
    )
    return p.parse_args()


def resolve_csv_path(arg: str) -> Path:
    """Resolve 'latest' or an explicit path."""
    if arg == "latest":
        candidates = sorted((PROJECT_ROOT / "results").glob("telemetry_*.csv"))
        if not candidates:
            raise FileNotFoundError("No telemetry_*.csv found in results/")
        latest = candidates[-1]
        print(f"Auto-pick latest: {latest.name}")
        return latest

    p = Path(arg)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p


def load_telemetry(csv_path: Path) -> pd.DataFrame:
    """Load CSV → DataFrame with time_rel column (seconds from first tick)."""
    df = pd.read_csv(csv_path)
    required = {"timestamp", *(f"j{i}" for i in range(1, 7))}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = df.copy()
    df["time_rel"] = df["timestamp"] - df["timestamp"].iloc[0]
    return df


def compute_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute joint velocity (deg/s) via numerical diff."""
    vel = pd.DataFrame(index=df.index)
    vel["time_rel"] = df["time_rel"]
    dt = np.gradient(df["timestamp"].values)
    dt = np.where(dt > 1e-6, dt, 1e-6)            # avoid division by zero
    for i in range(1, 7):
        vel[f"v{i}"] = np.gradient(df[f"j{i}"].values) / dt
    return vel


def plot_joint_trajectory(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, color) in enumerate(zip(JOINT_LABELS, JOINT_COLORS), start=1):
        ax.plot(df["time_rel"], df[f"j{i}"], label=f"J{i} ({label})",
                color=color, linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint angle (°)")
    ax.set_title("Joint Trajectory — Bidirectional Digital Twin Mirror")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  → {out_path}")


def plot_joint_velocity(vel: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, color) in enumerate(zip(JOINT_LABELS, JOINT_COLORS), start=1):
        ax.plot(vel["time_rel"], vel[f"v{i}"], label=f"J{i} ({label})",
                color=color, linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (°/s)")
    ax.set_title("Joint Velocity (numerical diff)")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  → {out_path}")


def plot_drift_events(df: pd.DataFrame, out_path: Path) -> None:
    """Plot alarm + running events on timeline."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3))
    if "running" in df.columns:
        running_int = pd.to_numeric(df["running"], errors="coerce").fillna(0)
        ax.fill_between(df["time_rel"], 0, running_int, step="post",
                        alpha=0.3, color="#4daf4a", label="Running")

    if "alarm" in df.columns:
        alarms = pd.to_numeric(df["alarm"], errors="coerce").fillna(0)
        alarm_mask = alarms != 0
        if alarm_mask.any():
            ax.scatter(df.loc[alarm_mask, "time_rel"],
                       [0.5] * alarm_mask.sum(),
                       color="red", marker="x", s=60, label="Alarm")

    ax.set_ylim(-0.1, 1.5)
    ax.set_xlabel("Time (s)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Idle", "Running"])
    ax.set_title("Robot State + Alarm Events")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  → {out_path}")


def detect_cycle_times(vel: pd.DataFrame, rest_threshold: float) -> list[float]:
    """Detect cycle = interval between two consecutive robot rest entries (all joint vel < threshold).

    Returns list of cycle durations (seconds).
    """
    speed = np.sqrt(sum(vel[f"v{i}"] ** 2 for i in range(1, 7)))
    is_rest = speed < rest_threshold

    # Find transitions rest→moving
    rest_to_move = (~is_rest[1:].reset_index(drop=True)) & (is_rest[:-1].reset_index(drop=True))
    transitions = vel["time_rel"].iloc[1:].reset_index(drop=True)[rest_to_move].tolist()

    if len(transitions) < 2:
        return []
    return [transitions[i + 1] - transitions[i] for i in range(len(transitions) - 1)]


def plot_cycle_time_hist(cycle_times: list[float], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    if cycle_times:
        ax.hist(cycle_times, bins=min(20, max(5, len(cycle_times))),
                color="#377eb8", alpha=0.7, edgecolor="black")
        ax.axvline(np.mean(cycle_times), color="red", linestyle="--",
                   label=f"Mean: {np.mean(cycle_times):.2f}s")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Insufficient data (need ≥ 2 cycles)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_xlabel("Cycle time (s)")
    ax.set_ylabel("Cycle count")
    ax.set_title(f"Cycle Time Distribution (n={len(cycle_times)})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  → {out_path}")


def summarize(df: pd.DataFrame, vel: pd.DataFrame, cycle_times: list[float]) -> None:
    """Print statistics summary to stdout."""
    duration = df["time_rel"].iloc[-1]
    print("\n─── Summary ───────────────────────────────────────")
    print(f"  Telemetry rows:   {len(df)}")
    print(f"  Duration:         {duration:.1f}s")
    if duration > 0:
        print(f"  Sample rate:      {len(df) / duration:.1f} Hz")

    print("\n  Joint range (min..max, ° per axis):")
    for i, label in enumerate(JOINT_LABELS, start=1):
        col = df[f"j{i}"]
        print(f"    J{i} ({label}): {col.min():>7.2f} .. {col.max():>7.2f}  "
              f"(span {col.max() - col.min():>6.2f})")

    print("\n  Max joint velocity (°/s per axis):")
    for i, label in enumerate(JOINT_LABELS, start=1):
        print(f"    J{i} ({label}): {vel[f'v{i}'].abs().max():.2f}")

    if cycle_times:
        print(f"\n  Cycles detected:  {len(cycle_times)}")
        print(f"  Cycle time:       {np.mean(cycle_times):.2f}s mean, "
              f"{np.median(cycle_times):.2f}s median")
    else:
        print("\n  Cycle time:       (insufficient cycles rest→move→rest)")


def main() -> int:
    args = parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    # Windows console defaults to cp1252 which cannot print unicode arrows/non-ASCII.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    csv_path = resolve_csv_path(args.csv_path)
    out_dir = Path(args.out_dir) if args.out_dir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {csv_path}")
    df = load_telemetry(csv_path)
    if len(df) < 2:
        print("CSV has too few rows (< 2) — cannot analyze.", file=sys.stderr)
        return 1

    vel = compute_velocity(df)
    cycle_times = detect_cycle_times(vel, args.rest_threshold_deg_s)

    suffix = csv_path.stem.replace("telemetry_", "")
    print("Saving figures:")
    plot_joint_trajectory(df, out_dir / f"joint_trajectory_{suffix}.png")
    plot_joint_velocity(vel, out_dir / f"joint_velocity_{suffix}.png")
    plot_drift_events(df, out_dir / f"drift_events_{suffix}.png")
    plot_cycle_time_hist(cycle_times, out_dir / f"cycle_time_{suffix}.png")

    summarize(df, vel, cycle_times)

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
