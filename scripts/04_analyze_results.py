#!/usr/bin/env python
"""
04_analyze_results.py
─────────────────────
Phân tích thống kê kết quả thí nghiệm + sinh figure cho paper (mục 10.3).

Đọc một hoặc nhiều file CSV trial (từ 03_run_experiment.py), tính:
    - Success rate tổng + theo class + theo điều kiện
    - Ma trận failure-mode
    - Paired t-test so sánh RGB-only vs RGB-D (nếu có cột 'mode')
    - Figure tổng hợp → figures/results_summary.png

Usage:
    python scripts/04_analyze_results.py --csv results/experiment_sim_*.csv
    python scripts/04_analyze_results.py --csv results/all_trials.csv
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir, setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="+", required=True,
                        help="File CSV trial (hỗ trợ wildcard)")
    parser.add_argument("--out", default="figures/results_summary.png")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging("analyze")

    import pandas as pd

    # Gom mọi file CSV (mở rộng wildcard).
    paths: list[str] = []
    for pattern in args.csv:
        paths.extend(glob.glob(str(PROJECT_ROOT / pattern)))
        paths.extend(glob.glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        log.error("Không tìm thấy file CSV nào khớp %s", args.csv)
        return 1

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    log.info("Đã nạp %d trial từ %d file", len(df), len(paths))

    # ─── Thống kê cơ bản ───
    log.info("Success rate tổng : %.1f%%", df["success"].mean() * 100)
    log.info("Theo class:\n%s", df.groupby("class_name")["success"].mean())

    if df["lighting"].notna().any() and df["lighting"].astype(str).str.len().gt(0).any():
        log.info("Theo điều kiện:\n%s",
                 df.groupby(["lighting", "overlap"])["success"].mean())

    # ─── Ma trận failure-mode ───
    fails = df[df["success"] == 0]
    if len(fails):
        fm = fails["failure_reason"].value_counts()
        log.info("Failure modes:\n%s", fm)

    # ─── Paired t-test RGB vs RGB-D ───
    if "mode" in df.columns and set(df["mode"].dropna().unique()) >= {"rgb_only", "rgbd"}:
        from scipy import stats as sps

        rgb = df[df["mode"] == "rgb_only"]["success"].to_numpy()
        rgbd = df[df["mode"] == "rgbd"]["success"].to_numpy()
        n = min(len(rgb), len(rgbd))
        if n > 1:
            t, p = sps.ttest_rel(rgb[:n], rgbd[:n])
            log.info("Depth fusion: t=%.3f, p=%.4f (%s)",
                     t, p, "có ý nghĩa" if p < 0.05 else "chưa có ý nghĩa")

    # ─── Figure ───
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        df.groupby("class_name")["success"].mean().plot.bar(
            ax=axes[0], title="Success rate theo class", ylim=(0, 1))
        df.boxplot(column="cycle_time_s", by="class_name", ax=axes[1])
        axes[1].set_title("Cycle time theo class")
        plt.suptitle("")
        out_path = ensure_dir(PROJECT_ROOT / Path(args.out).parent) / Path(args.out).name
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        log.info("Figure lưu tại %s", out_path)
    except Exception as e:  # noqa: BLE001
        log.warning("Không vẽ được figure: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
