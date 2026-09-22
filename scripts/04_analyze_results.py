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
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir, setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="+", required=True,
                        help="Trial CSV file(s) (wildcard supported)")
    parser.add_argument("--out", default="figures/results_summary.png")
    parser.add_argument("--paired-with", nargs="+", default=None,
                        help="One or more run CSVs, each paired against --csv with exact "
                             "McNemar. Pass the WHOLE family of comparisons in one call: "
                             "with several of them the raw p-values are adjusted with "
                             "Holm, which cannot happen if each comparison is run "
                             "separately and read off on its own.")
    parser.add_argument("--no-holm", action="store_true",
                        help="Report raw p-values only, even with several --paired-with "
                             "runs. Use it when this really is one pre-registered primary "
                             "comparison and the others are exploratory; say so in the "
                             "paper if you do.")
    parser.add_argument("--pair-key", default="pose_id",
                        help="Column pairing trials across runs (pose_id | trial_id)")
    parser.add_argument("--split-col", default=None,
                        help="Split the rows loaded by --csv on this column (typically "
                             "depth_mode) and compare every value against --baseline. "
                             "An interleaved blinded campaign writes one CSV per BLOCK, "
                             "so the arm is a column, not a filename, and there is no "
                             "per-arm file to hand to --paired-with.")
    parser.add_argument("--key", nargs="+", default=None,
                        help="Sealed key file(s) from tools/run_blinded_campaign.py "
                             "(wildcard supported). Adds an 'arm' column from each row's "
                             "block_id, so arms that differ in something the CSV does not "
                             "record (the model, for E3 to E5) can be split with "
                             "--split-col arm. Every loaded row must belong to a keyed "
                             "block: control blocks go in their own folder.")
    parser.add_argument("--only", nargs="+", default=None, metavar="VALUE",
                        help="With --split-col: keep only these values. The comparison "
                             "family is then exactly these arms, e.g. E5's three factors "
                             "against E3's anchored arm without E3's wide arm.")
    parser.add_argument("--baseline", default=None,
                        help="With --split-col: the value that plays A in every "
                             "comparison. Default: the first value alphabetically.")
    parser.add_argument("--score-col", default="success",
                        help="Which column counts as the outcome: 'success' (the "
                             "machine's, from motion + the gripper detect sensor) or "
                             "'human_ok' (what the operator saw, written by 03 "
                             "--confirm-each-trial). They disagree when a part was "
                             "gripped but dropped in transfer or landed off the drop "
                             "point, so the paper must say which one it reports.")
    parser.add_argument("--label-a", default="run_a", help="Name of the --csv run")
    parser.add_argument("--labels-b", nargs="+", default=None,
                        help="Names for the --paired-with runs, in the same order.")
    parser.add_argument("--label-b", default="run_b", help="Name of the --paired-with run")
    return parser.parse_args()



def _decode_arms(log, df, patterns):
    """Add the 'arm' column from sealed campaign keys, matched on block_id."""
    import json

    files = sorted({str(Path(p).resolve()) for pat in patterns
                    for p in glob.glob(str(PROJECT_ROOT / pat)) + glob.glob(pat)})
    if not files:
        log.error("--key %s matched no file", patterns)
        return None
    arm_of: dict[str, str] = {}
    for f in files:
        for entry in json.loads(Path(f).read_text(encoding="utf-8"))["schedule"]:
            if entry["block_id"] in arm_of and arm_of[entry["block_id"]] != entry["arm"]:
                log.error("block_id %s appears in two keys with different arms",
                          entry["block_id"])
                return None
            arm_of[entry["block_id"]] = entry["arm"]
    if "block_id" not in df.columns:
        log.error("--key needs a block_id column in the runs")
        return None
    df = df.copy()
    df["arm"] = df["block_id"].astype(str).map(arm_of)
    stray = df["arm"].isna()
    if stray.any():
        log.error("%d row(s) belong to no keyed block (block_id %s): control blocks and "
                  "other runs must not be loaded with the campaign", int(stray.sum()),
                  sorted(df.loc[stray, "block_id"].astype(str).unique())[:5])
        return None
    log.info("Decoded %d block(s) from %d key file(s)", df["block_id"].nunique(), len(files))
    return df


def _pair(log, a, b, key):
    """Inner-join two runs on `key`, one row against one row.

    Without the one-to-one constraint a duplicated key silently turns the join
    into a cross product: n and both disagreement cells inflate, and McNemar
    returns a p-value that is wrong in the permissive direction. Duplicates are
    not hypothetical — TrialLogger appends to an existing CSV on purpose, and a
    glob can pick up two runs of the same configuration.

    Returns the merged frame, or None after logging why pairing was refused.
    """
    import pandas as pd

    out = {}
    for name, frame in (("A", a), ("B", b)):
        f = frame[frame[key].notna()]
        f = f[f[key].astype(str).str.strip().ne("")]
        f = f[~f[key].astype(str).str.lower().isin({"nan", "none"})]
        dup = f[key].duplicated().sum()
        if dup:
            log.error("Run %s has %d duplicated '%s' values. Two runs of the same "
                      "configuration in one file, or an appended CSV. Pairing refused: "
                      "a duplicated key would inflate the disagreement counts and make "
                      "the p-value look better than it is.", name, dup, key)
            return None
        out[name] = f

    merged = out["A"].merge(out["B"], on=key, suffixes=("_a", "_b"),
                            validate="one_to_one")
    only_a = len(out["A"]) - len(merged)
    only_b = len(out["B"]) - len(merged)
    if only_a or only_b:
        log.warning("Pairing dropped %d row(s) present only in A and %d only in B. "
                    "Both runs must come from the SAME pose list, run to the same "
                    "length; a short run pairs silently against a full one.",
                    only_a, only_b)
    log.info("Paired %d trial(s) on '%s'", len(merged), key)
    return merged


_SCORE_MAP = {"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0}


def _apply_score_col(log, df, col: str, source: str):
    """Make `success` carry the chosen outcome column, or refuse and return None.

    Everything downstream (rates, failure matrix, McNemar, figure) reads
    `success`, so the switch happens once, here. A partly filled column is
    refused rather than silently scored on the rows that happen to be filled:
    that would drop trials from one arm only and bias the comparison.
    """
    if col == "success":
        return df
    if col not in df.columns:
        log.error("Column '%s' is not in %s. That run was recorded before human "
                  "scoring existed, or 03_run_experiment.py ran without "
                  "--confirm-each-trial.", col, source)
        return None
    raw = df[col].astype(str).str.strip().str.lower()
    blank = df[col].isna() | raw.isin({"", "nan", "none"})
    if blank.any():
        log.error("%d of %d row(s) in %s have no '%s' value. Refusing to score on a "
                  "partly filled column.", int(blank.sum()), len(df), source, col)
        return None
    mapped = raw.map(_SCORE_MAP)
    if mapped.isna().any():
        bad = sorted(set(raw[mapped.isna()]))[:5]
        log.error("Column '%s' in %s holds values that are not 0/1: %s", col, source, bad)
        return None
    df = df.copy()
    df["success"] = mapped.astype(int)
    return df


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

    if args.key:
        df = _decode_arms(log, df, args.key)
        if df is None:
            return 1

    if args.only:
        if not args.split_col or args.split_col not in df.columns:
            log.error("--only needs --split-col naming a column of the loaded runs")
            return 1
        values = df[args.split_col].astype(str)
        missing = [v for v in args.only if v not in set(values)]
        if missing:
            log.error("--only %s not found in '%s' (values: %s)", missing, args.split_col,
                      sorted(set(values)))
            return 1
        df = df[values.isin(args.only)].reset_index(drop=True)
        log.info("Kept %d trial(s) of %s = %s", len(df), args.split_col, args.only)

    df = _apply_score_col(log, df, args.score_col, "--csv")
    if df is None:
        return 1
    log.info("Scoring on column '%s'%s", args.score_col,
             " (the operator's verdict, not the machine's)"
             if args.score_col != "success" else "")

    # ─── Basic statistics ───
    log.info("Overall success rate: %.1f%%", df["success"].mean() * 100)
    log.info("By class:\n%s", df.groupby("class_name")["success"].mean())

    if df["lighting"].notna().any() and df["lighting"].astype(str).str.len().gt(0).any():
        log.info("By condition:\n%s",
                 df.groupby(["lighting", "overlap"])["success"].mean())

    # ─── Failure-mode matrix ───
    fails = df[df["success"] == 0]
    if len(fails):
        # Scored on human_ok, a failure can have an empty machine reason: the run
        # thought it went fine. Those rows must show up, not be dropped as NaN.
        fm = fails["failure_reason"].fillna("(no machine failure)").value_counts()
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
            merged = _pair(log, rgb_df, rgbd_df, "trial_id")
            if merged is None:
                return 1
            if len(merged) > 1:
                _log_mcnemar(log, mcnemar_summary(
                    merged["success_a"], merged["success_b"], "rgb_only", "rgbd"))
            else:
                log.warning("No shared trial_id between modes — skipping paired test")
        else:
            log.warning("No trial_id column — McNemar needs pairing; "
                        "rerun with a shared pose list (03 --pose-list)")

    # ─── Paired exact McNemar ───
    # Two ways to say which rows are which arm. --split-col splits the rows
    # already loaded on a column: an interleaved blinded campaign writes one CSV
    # per BLOCK, so the arm is a column, never a filename. --paired-with names
    # one file (or glob) per arm, for runs kept in separate files.
    pairs: list[tuple[str, Any, str, Any]] = []

    if args.split_col:
        if args.split_col not in df.columns:
            log.error("--split-col '%s' is not a column of the loaded runs",
                      args.split_col)
            return 1
        groups = {str(v): g for v, g in df.groupby(args.split_col) if str(v).strip()}
        if len(groups) < 2:
            log.error("--split-col '%s' has %d value(s) in the loaded rows; a paired "
                      "comparison needs at least 2.", args.split_col, len(groups))
            return 1
        base = args.baseline or sorted(groups)[0]
        if base not in groups:
            log.error("--baseline '%s' is not one of %s", base, sorted(groups))
            return 1
        log.info("Split on '%s': %s, baseline '%s'", args.split_col,
                 {k: len(v) for k, v in sorted(groups.items())}, base)
        for name in sorted(k for k in groups if k != base):
            pairs.append((base, groups[base], name, groups[name]))

    if args.paired_with:
        labels_b = args.labels_b or []
        if labels_b and len(labels_b) != len(args.paired_with):
            log.error("--labels-b has %d name(s) for %d run(s)",
                      len(labels_b), len(args.paired_with))
            return 1
        for i, other in enumerate(args.paired_with):
            name_b = labels_b[i] if labels_b else (
                args.label_b if len(args.paired_with) == 1 else Path(other).stem)
            files = sorted({str(Path(p).resolve())
                            for p in glob.glob(str(PROJECT_ROOT / other)) + glob.glob(other)})
            if not files:
                log.error("--paired-with '%s' matched no file", other)
                return 1
            df_b = _apply_score_col(
                log, pd.concat([pd.read_csv(f) for f in files], ignore_index=True),
                args.score_col, other)
            if df_b is None:
                return 1
            pairs.append((args.label_a, df, name_b, df_b))

    if pairs:
        from src.utils.stats import holm, mcnemar_summary

        key = args.pair_key
        results = []
        for name_a, frame_a, name_b, frame_b in pairs:
            if key not in frame_a.columns or key not in frame_b.columns:
                log.error("Pair key '%s' missing in '%s' — cannot pair", key, name_b)
                return 1
            merged = _pair(log, frame_a, frame_b, key)
            if merged is None:
                return 1
            if len(merged) <= 1:
                log.error("No shared '%s' values between '%s' and '%s' — both must come "
                          "from the SAME pre-drawn pose list", key, name_a, name_b)
                return 1
            summary = mcnemar_summary(merged["success_a"], merged["success_b"],
                                      name_a, name_b)
            _log_mcnemar(log, summary)
            results.append(summary)

        if len(results) > 1 and not args.no_holm:
            adj = holm([r["p_mcnemar_exact"] for r in results])
            log.info("─" * 60)
            log.info("Holm-adjusted over this family of %d comparison(s):", len(results))
            for r, pa in zip(results, adj):
                log.info("  %-28s raw p=%.4g → adjusted p=%.4g %s",
                         f"{r['label_a']} vs {r['label_b']}", r["p_mcnemar_exact"], pa,
                         "(significant)" if pa < 0.05 else "(not significant)")
            log.info("Adjustment is over the comparisons passed in THIS call. If the paper "
                     "reports more comparisons than these, the family is larger and the "
                     "adjustment here is too lenient.")
        elif len(results) > 1:
            log.warning("%d comparisons reported WITHOUT Holm adjustment (--no-holm). "
                        "At alpha 0.05 the chance of at least one false positive across "
                        "%d independent tests is %.0f%%.",
                        len(results), len(results),
                        (1 - 0.95 ** len(results)) * 100)

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
