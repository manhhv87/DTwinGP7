#!/usr/bin/env python
"""
04_analyze_results.py
─────────────────────
Statistical analysis of experiment results + figure generation for paper (section 10.3).

Reads one or more trial CSV files (from 03_run_experiment.py) and computes:
    - Overall success rate + by class + by condition
    - Failure-mode matrix
    - Paired EXACT McNemar comparisons (grasp outcomes are binary and paired —
      a t-test is the wrong test here):
        (a) RGB-only vs RGB-D when column 'mode' carries both values;
        (b) --paired-with <other.csv>: this run vs another run, paired by
            --pair-key (pose_id from a pre-drawn pose list, or trial_id).
      Reports the discordant counts (n01/n10), exact p, and a paired-bootstrap
      95% CI on the success-rate difference. When comparing >2 configurations
      against a common baseline, feed the p-values to src.utils.stats.holm.
    - Summary figure → figures/results_summary.png

Usage:
    python scripts/04_analyze_results.py --csv results/experiment_sim_*.csv
    python scripts/04_analyze_results.py --csv results/run_anchored.csv \
        --paired-with results/run_blind.csv --pair-key pose_id \
        --label-a anchored --label-b blind
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
    parser.add_argument("--paired-with", default=None,
                        help="Second run CSV — paired exact McNemar vs --csv")
    parser.add_argument("--pair-key", default="pose_id",
                        help="Column pairing trials across runs (pose_id | trial_id)")
    parser.add_argument("--label-a", default="run_a", help="Name of the --csv run")
    parser.add_argument("--label-b", default="run_b", help="Name of the --paired-with run")
    return parser.parse_args()


def _log_mcnemar(log, s: dict) -> None:
    """Log a mcnemar_summary dict in the format the paper tables need."""
    log.info(
        "McNemar %s vs %s (n=%d pairs): %.1f%% vs %.1f%% | "
        "discordant n01=%d n10=%d | exact p=%.4g | delta=%.1f pp, 95%% CI [%.1f, %.1f] (%s)",
        s["label_a"], s["label_b"], s["n_pairs"],
        s["rate_a"] * 100, s["rate_b"] * 100,
        s["n01"], s["n10"], s["p_mcnemar_exact"],
        s["delta_pp"], s["ci95_pp"][0], s["ci95_pp"][1],
        "significant" if s["p_mcnemar_exact"] < 0.05 else "not significant",
    )


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

    # ─── Paired exact McNemar: RGB-only vs RGB-D ───
    # Binary paired outcomes → McNemar on the discordant pairs (paper §3.7).
    # The earlier paired t-test treated 0/1 as continuous — wrong test, removed.
    if "mode" in df.columns and set(df["mode"].dropna().unique()) >= {"rgb_only", "rgbd"}:
        from src.utils.stats import mcnemar_summary

        rgb_df = df[df["mode"] == "rgb_only"]
        rgbd_df = df[df["mode"] == "rgbd"]
        # PAIR by trial_id (compares the SAME scene under both modes).
        if "trial_id" in df.columns:
            merged = rgb_df.merge(rgbd_df, on="trial_id", suffixes=("_a", "_b"))
            if len(merged) > 1:
                _log_mcnemar(log, mcnemar_summary(
                    merged["success_a"], merged["success_b"], "rgb_only", "rgbd"))
            else:
                log.warning("No shared trial_id between modes — skipping paired test")
        else:
            log.warning("No trial_id column — McNemar needs pairing; "
                        "rerun with a shared pose list (03 --pose-list)")

    # ─── Paired exact McNemar: this run vs --paired-with run ───
    if args.paired_with:
        from src.utils.stats import mcnemar_summary

        df_b = pd.read_csv(args.paired_with)
        key = args.pair_key
        if key not in df.columns or key not in df_b.columns:
            log.error("Pair key '%s' missing in one of the runs — cannot pair", key)
        else:
            a = df[df[key].astype(str).str.len() > 0]
            b = df_b[df_b[key].astype(str).str.len() > 0]
            merged = a.merge(b, on=key, suffixes=("_a", "_b"))
            if len(merged) > 1:
                _log_mcnemar(log, mcnemar_summary(
                    merged["success_a"], merged["success_b"],
                    args.label_a, args.label_b))
            else:
                log.error("No shared '%s' values between the two runs — "
                          "both must be run from the SAME pre-drawn pose list", key)

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
