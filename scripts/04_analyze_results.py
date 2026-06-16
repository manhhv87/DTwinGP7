#!/usr/bin/env python
"""
04_analyze_results.py
─────────────────────
Statistical analysis of experiment results + figure generation for paper (section 10.3).

Reads one or more trial CSV files (from 03_run_experiment.py) and computes:
    - Overall success rate + by class + by condition
    - Failure-mode matrix
    - Paired t-test comparing RGB-only vs RGB-D (if column 'mode' is present)
    - Summary figure → figures/results_summary.png

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
                        help="Trial CSV file(s) (wildcard supported)")
    parser.add_argument("--out", default="figures/results_summary.png")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging("analyze")

    import pandas as pd

    # Collect all CSV files (expand wildcards). Canonicalize to ABSOLUTE paths
    # before dedup — the same file matched both PROJECT_ROOT/pattern (absolute) and
    # pattern (relative) yields two distinct strings that set() would NOT dedup,
    # reading every CSV twice (inflating trial counts + the t-test n).
    seen: set[str] = set()
    for pattern in args.csv:
        for p in glob.glob(str(PROJECT_ROOT / pattern)) + glob.glob(pattern):
            seen.add(str(Path(p).resolve()))
    paths = sorted(seen)
    if not paths:
        log.error("No CSV files found matching %s", args.csv)
        return 1

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    log.info("Loaded %d trials from %d file(s)", len(df), len(paths))

    # ─── Basic statistics ───
    log.info("Overall success rate: %.1f%%", df["success"].mean() * 100)
    log.info("By class:\n%s", df.groupby("class_name")["success"].mean())

    if df["lighting"].notna().any() and df["lighting"].astype(str).str.len().gt(0).any():
        log.info("By condition:\n%s",
                 df.groupby(["lighting", "overlap"])["success"].mean())

    # ─── Failure-mode matrix ───
    fails = df[df["success"] == 0]
    if len(fails):
        fm = fails["failure_reason"].value_counts()
        log.info("Failure modes:\n%s", fm)

    # ─── Paired t-test RGB vs RGB-D ───
    if "mode" in df.columns and set(df["mode"].dropna().unique()) >= {"rgb_only", "rgbd"}:
        from scipy import stats as sps

        rgb_df = df[df["mode"] == "rgb_only"]
        rgbd_df = df[df["mode"] == "rgbd"]
        # PAIR by trial_id (paired t-test compares the SAME scene under both modes).
        # Positional rgb[:n] vs rgbd[:n] pairs unrelated trials → meaningless stat.
        if "trial_id" in df.columns:
            merged = rgb_df.merge(rgbd_df, on="trial_id", suffixes=("_a", "_b"))
            if len(merged) > 1:
                t, p = sps.ttest_rel(merged["success_a"], merged["success_b"])
                log.info("Depth fusion (paired on trial_id, n=%d): t=%.3f, p=%.4f (%s)",
                         len(merged), t, p,
                         "significant" if p < 0.05 else "not significant")
            else:
                log.warning("No shared trial_id between modes — skipping paired test")
        else:
            # No pairing key → unpaired (independent-samples) test.
            rgb = rgb_df["success"].to_numpy(); rgbd = rgbd_df["success"].to_numpy()
            if len(rgb) > 1 and len(rgbd) > 1:
                t, p = sps.ttest_ind(rgb, rgbd)
                log.info("Depth fusion (UNPAIRED, no trial_id): t=%.3f, p=%.4f (%s)",
                         t, p, "significant" if p < 0.05 else "not significant")

    # ─── Figure ───
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        df.groupby("class_name")["success"].mean().plot.bar(
            ax=axes[0], title="Success rate by class", ylim=(0, 1))
        df.boxplot(column="cycle_time_s", by="class_name", ax=axes[1])
        axes[1].set_title("Cycle time by class")
        plt.suptitle("")
        out_path = ensure_dir(PROJECT_ROOT / Path(args.out).parent) / Path(args.out).name
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        log.info("Figure saved to %s", out_path)
    except Exception as e:  # noqa: BLE001
        log.warning("Could not render figure: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
