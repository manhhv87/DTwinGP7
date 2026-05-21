#!/usr/bin/env python
"""
07_replay_telemetry.py
──────────────────────
UC4 — Replay CSV telemetry thành 3D animation hoặc figure.

Đọc telemetry CSV (sinh bởi DigitalTwinMirror) → forward kinematics mỗi row
→ animate skeleton + TCP path. Export MP4/PNG cho thesis defense.

Usage:
    python scripts/07_replay_telemetry.py latest                 # animate màn hình
    python scripts/07_replay_telemetry.py latest --mp4 out.mp4   # export MP4
    python scripts/07_replay_telemetry.py latest --png frame.png # 1 figure summary
    python scripts/07_replay_telemetry.py latest --speedup 5     # 5x faster playback

CSV format (theo TelemetryLogger.HEADER):
    timestamp, j1..j6 (degrees), running, alarm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cell.cell_models import CellConfig  # noqa: E402
from src.orchestrator.kinematics import (  # noqa: E402
    forward_kinematics,
    gp7_default,
    joint_positions,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "csv_path", nargs="?", default="latest",
        help="Đường dẫn CSV hoặc 'latest'. Default: latest.",
    )
    p.add_argument(
        "--cell-config", default="config/cell_layout.yaml",
        help="Cell config YAML cho base J1 pose.",
    )
    p.add_argument("--mp4", default=None, help="Export MP4 path (cần ffmpeg).")
    p.add_argument("--png", default=None, help="Export 1 PNG summary path.")
    p.add_argument(
        "--speedup", type=float, default=1.0,
        help="Tốc độ playback (1.0 = realtime). Default 1.0.",
    )
    p.add_argument(
        "--max-frames", type=int, default=300,
        help="Subsample xuống N frame max (tránh CSV cực dài). Default 300.",
    )
    return p.parse_args()


def resolve_csv_path(arg: str) -> Path:
    if arg == "latest":
        candidates = sorted((PROJECT_ROOT / "results").glob("telemetry_*.csv"))
        if not candidates:
            raise FileNotFoundError("Không có telemetry_*.csv trong results/")
        return candidates[-1]
    p = Path(arg)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Không tồn tại: {p}")
    return p


def load_and_subsample(csv_path: Path, max_frames: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"timestamp", *(f"j{i}" for i in range(1, 7))}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV thiếu cột: {required - set(df.columns)}")
    df = df.copy()
    df["time_rel"] = df["timestamp"] - df["timestamp"].iloc[0]
    # Subsample uniform để tránh frame quá nhiều
    if len(df) > max_frames:
        idx = np.linspace(0, len(df) - 1, max_frames, dtype=int)
        df = df.iloc[idx].reset_index(drop=True)
    return df


def make_skeleton_artists(ax, model, joints_rad):
    """Tạo line + scatter artists cho 1 frame skeleton."""
    positions = np.array(joint_positions(model, joints_rad))
    (line,) = ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                      "o-", linewidth=2.5, color="#377eb8", markersize=6)
    return line, positions


def compute_workspace_bounds(model, df) -> tuple[np.ndarray, np.ndarray]:
    """Range bounds cho axis limits — visit toàn bộ trajectory + TCP."""
    all_xyz = []
    for _, row in df.iterrows():
        joints = [np.deg2rad(row[f"j{i}"]) for i in range(1, 7)]
        positions = joint_positions(model, joints)
        all_xyz.extend(positions)
    arr = np.array(all_xyz)
    return arr.min(axis=0), arr.max(axis=0)


def replay_animate(model, df, args) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Axis bounds
    lo, hi = compute_workspace_bounds(model, df)
    margin = 100
    ax.set_xlim(lo[0] - margin, hi[0] + margin)
    ax.set_ylim(lo[1] - margin, hi[1] + margin)
    ax.set_zlim(0, hi[2] + margin)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    # Initialize artists
    initial_joints = [np.deg2rad(df.iloc[0][f"j{i}"]) for i in range(1, 7)]
    line, positions = make_skeleton_artists(ax, model, initial_joints)

    tcp_path_x, tcp_path_y, tcp_path_z = [], [], []
    (tcp_line,) = ax.plot([], [], [], "r-", linewidth=1, alpha=0.6, label="TCP path")

    title = ax.set_title(f"Replay {Path(args.csv_path).name} — t=0.00s")
    ax.legend(loc="upper left", fontsize=9)

    def update(frame_idx):
        row = df.iloc[frame_idx]
        joints = [np.deg2rad(row[f"j{i}"]) for i in range(1, 7)]
        positions = np.array(joint_positions(model, joints))
        line.set_data(positions[:, 0], positions[:, 1])
        line.set_3d_properties(positions[:, 2])

        tcp_path_x.append(positions[-1, 0])
        tcp_path_y.append(positions[-1, 1])
        tcp_path_z.append(positions[-1, 2])
        tcp_line.set_data(tcp_path_x, tcp_path_y)
        tcp_line.set_3d_properties(tcp_path_z)

        title.set_text(
            f"Replay {Path(args.csv_path).name} — t={row['time_rel']:.2f}s "
            f"(frame {frame_idx + 1}/{len(df)})"
        )
        return line, tcp_line, title

    # Interval theo speedup
    if len(df) > 1:
        dt_mean = (df["time_rel"].iloc[-1] - df["time_rel"].iloc[0]) / (len(df) - 1)
    else:
        dt_mean = 0.1
    interval_ms = max(20, int(dt_mean * 1000 / args.speedup))

    anim = FuncAnimation(fig, update, frames=len(df),
                         interval=interval_ms, blit=False, repeat=False)

    if args.mp4:
        out = Path(args.mp4)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            anim.save(str(out), writer="ffmpeg", fps=int(1000 / interval_ms))
            print(f"MP4 → {out}")
        except Exception as e:                      # noqa: BLE001
            print(f"MP4 export lỗi (cần ffmpeg trong PATH): {e}")
    elif args.png:
        # Render frame cuối làm summary PNG
        update(len(df) - 1)
        out = Path(args.png)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"PNG summary → {out}")
    else:
        plt.show()


def main() -> int:
    args = parse_args()
    if args.mp4 or args.png:
        matplotlib.use("Agg")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    csv_path = resolve_csv_path(args.csv_path)
    print(f"Loading: {csv_path}")
    df = load_and_subsample(csv_path, args.max_frames)
    print(f"Replay {len(df)} frames (subsampled từ telemetry CSV)")

    base_xyz = (0.0, 0.0, 0.0)
    try:
        cfg = CellConfig.from_yaml(PROJECT_ROOT / args.cell_config)
        base_xyz = tuple(cfg.robot.pose.xyz_mm)
    except Exception:                               # noqa: BLE001
        pass

    model = gp7_default(base_xyz_mm=base_xyz)
    args.csv_path = csv_path                        # cập nhật cho title
    replay_animate(model, df, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
