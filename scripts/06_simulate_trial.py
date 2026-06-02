#!/usr/bin/env python
"""
06_simulate_trial.py
────────────────────
UC1 — Offline predictive simulation for a single pick-and-place trial.

Generates a predicted trajectory (pure Python FK, no RoboDK/real robot required):
  - 3D plot TCP path
  - Joint angles timeline
  - Self-collision events
  - Joint limit violations

Usage:
    python scripts/06_simulate_trial.py                    # default GP7 demo trial
    python scripts/06_simulate_trial.py --max-speed 60    # increase joint speed
    python scripts/06_simulate_trial.py --output paper.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cell.cell_models import CellConfig  # noqa: E402
from src.orchestrator.kinematics import (  # noqa: E402
    check_joint_limits,
    check_self_collision_spheres,
    forward_kinematics,
    gp7_default,
    interpolate_joints,
    joint_positions,
)

JOINT_LABELS = ["S", "L", "U", "R", "B", "T"]
JOINT_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--cell-config", default="config/cell_layout.yaml",
        help="Cell config YAML for loading home_joints + base pose.",
    )
    p.add_argument(
        "--max-speed", type=float, default=30.0,
        help="Max joint speed (deg/s) for interpolation. Default 30.",
    )
    p.add_argument(
        "--output", default="figures/predicted_trial.png",
        help="Output PNG path. Default figures/predicted_trial.png.",
    )
    p.add_argument("--no-show", action="store_true", help="Do not open a matplotlib window")
    return p.parse_args()


def build_demo_trajectory() -> list[list[float]]:
    """Build a demo pick-and-place trajectory: home → approach → grasp → place → home.

    Joint values are representative of a typical GP7 trial (degrees, converted to radians).
    Can be replaced with joints from the Orchestrator's planning output.
    """
    deg = np.deg2rad
    waypoints_deg = [
        # home          → approach    → grasp       → lift       → transfer    → place      → retreat    → home
        [0, 30, -60, 0, 60, 0],          # home: elbow-up, wrist down
        [10, 40, -50, 0, 50, 10],        # approach over object
        [10, 50, -40, 0, 45, 10],        # grasp (lower)
        [10, 40, -50, 0, 50, 10],        # lift (back up)
        [-20, 40, -50, 0, 50, -10],      # transfer to place zone
        [-20, 50, -40, 0, 45, -10],      # place (lower)
        [-20, 40, -50, 0, 50, -10],      # retreat (lift)
        [0, 30, -60, 0, 60, 0],          # home
    ]
    return [[deg(d) for d in wp] for wp in waypoints_deg]


def plot_3d_tcp_path(model, samples, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # TCP path
    tcp_xyz = np.array([
        forward_kinematics(model, s.joints_rad)[:3, 3] for s in samples
    ])
    ax.plot(tcp_xyz[:, 0], tcp_xyz[:, 1], tcp_xyz[:, 2],
            "b-", linewidth=2, alpha=0.7, label="TCP path")

    # Skeleton at the first, middle, and last sample
    for idx, color, label in [
        (0, "green", "start"),
        (len(samples) // 2, "orange", "mid"),
        (-1, "red", "end"),
    ]:
        positions = np.array(joint_positions(model, samples[idx].joints_rad))
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                "o-", color=color, linewidth=2, alpha=0.6, label=f"skel {label}")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"Predicted TCP Trajectory — {model.name} ({len(samples)} samples)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  → {out_path}")


def plot_joint_timeline(samples, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    times = [s.t for s in samples]
    for j_idx in range(6):
        joints_deg = [np.rad2deg(s.joints_rad[j_idx]) for s in samples]
        ax.plot(times, joints_deg, label=f"J{j_idx+1} ({JOINT_LABELS[j_idx]})",
                color=JOINT_COLORS[j_idx], linewidth=1.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint angle (°)")
    ax.set_title("Predicted Joint Trajectory")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  → {out_path}")


def main() -> int:
    args = parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    # Build robot DH model (load base offset from cell config if available)
    base_xyz = (0.0, 0.0, 0.0)
    try:
        cfg = CellConfig.from_yaml(PROJECT_ROOT / args.cell_config)
        base_xyz = tuple(cfg.robot.pose.xyz_mm)
        print(f"Cell config: {args.cell_config} → base J1 at {base_xyz}")
    except Exception as e:                          # noqa: BLE001
        print(f"Could not load cell config: {e}. Using base (0,0,0).")

    model = gp7_default(base_xyz_mm=base_xyz)

    # Build trajectory + interpolate
    waypoints = build_demo_trajectory()
    print(f"Waypoints: {len(waypoints)}")
    samples = interpolate_joints(waypoints, dt=0.05,
                                 max_joint_speed_deg_s=args.max_speed)
    print(f"Interpolated to {len(samples)} samples ({samples[-1].t:.2f}s total)")

    # Safety checks
    limit_violations = check_joint_limits(model, samples)
    collision_events = check_self_collision_spheres(model, samples)
    print(f"\nSafety checks:")
    print(f"  Joint limit violations: {len(limit_violations)}")
    if limit_violations:
        for s_idx, j_idx, val in limit_violations[:5]:
            print(f"    Sample {s_idx}, J{j_idx+1} = {np.rad2deg(val):.1f}°")
    print(f"  Self-collision events:  {len(collision_events)}")
    if collision_events:
        for s_idx, i, j, dist in collision_events[:5]:
            print(f"    Sample {s_idx}, joint {i} vs {j}: dist={dist:.1f}mm")

    # Plot
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving figures:")
    suffix = out_path.stem
    plot_3d_tcp_path(model, samples,
                     out_path.parent / f"{suffix}_3d.png")
    plot_joint_timeline(samples,
                        out_path.parent / f"{suffix}_joints.png")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
