#!/usr/bin/env python
"""
21_mine_failures.py
───────────────────
C3 step: mine failure modes from trial CSVs and allocate the next rendering
budget (Algorithm 1 of the paper: cluster failures → reallocate budget).

Usage:
    python scripts/21_mine_failures.py --runs "results/experiment_real_*.csv" \
        --n-budget 1000 --out results/failure_modes.json

Then generate the loop iteration's synthetic batch:
    python scripts/20_generate_synth.py --from-failures results/failure_modes.json \
        --out data/synth/loop_k1 --seed 1000

The CONTROL condition of E4 (same budget, no failure guidance) is simply:
    python scripts/20_generate_synth.py --mode anchored --n <same total> \
        --out data/synth/loop_k1_control --seed 2000
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.synthgen import (  # noqa: E402
    SynthgenConfig,
    allocate,
    bin_failures,
    kmeans_sensitivity,
    load_trials,
    save_report,
)
from src.utils import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Trial CSV file(s)/wildcards from 03_run_experiment.py")
    parser.add_argument("--grid", type=int, nargs=2, default=[3, 3],
                        metavar=("NX", "NY"), help="Pick-region binning grid")
    parser.add_argument("--n-budget", type=int, default=1000,
                        help="Rendering budget N_k for the next loop iteration")
    parser.add_argument("--floor-frac", type=float, default=0.05,
                        help="Minimum budget share per failure mode")
    parser.add_argument("--kmeans-k", type=int, default=4,
                        help="k for the k-means sensitivity check (0 = skip)")
    parser.add_argument("--config", default="config/synthgen.yaml",
                        help="Synthgen config (supplies the pick-region bounds)")
    parser.add_argument("--out", default="results/failure_modes.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging("mine_failures")

    paths: list[str] = []
    for pattern in args.runs:
        paths += glob.glob(str(PROJECT_ROOT / pattern)) + glob.glob(pattern)
    paths = sorted({str(Path(p).resolve()) for p in paths})
    if not paths:
        log.error("No trial CSVs match %s", args.runs)
        return 1

    cfg = SynthgenConfig.from_yaml(PROJECT_ROOT / args.config)
    region = (
        cfg.objects.region_x_mm[0], cfg.objects.region_x_mm[1],
        cfg.objects.region_y_mm[0], cfg.objects.region_y_mm[1],
    )

    rows = load_trials(paths)
    report = bin_failures(rows, region=region, grid=tuple(args.grid))
    log.info("Trials: %d | failures: %d | modes: %d",
             report["n_trials"], report["n_failures"], len(report["clusters"]))
    if report["n_failures"] == 0:
        log.warning("No failures found — nothing to mine (loop stopping rule?)")
        return 0

    header = f"{'mode':<46} {'count':>5} {'frac':>6}  budget"
    log.info("%s", header)
    log.info("%s", "─" * len(header))

    allocation = allocate(report["clusters"], args.n_budget,
                          floor_frac=args.floor_frac)
    for entry in allocation:
        log.info("%-46s %5d %5.1f%%  %5d img",
                 entry["key"], entry["count"], entry["frac"] * 100,
                 entry["n_images"])
    report["allocation"] = allocation
    report["n_budget"] = args.n_budget

    if args.kmeans_k > 0:
        km = kmeans_sensitivity(rows, k=args.kmeans_k)
        report["kmeans_sensitivity"] = km
        if km:
            log.info("k-means sensitivity (k=%d, n=%d): counts=%s centers=%s",
                     km["k"], km["n_points"], km["counts"], km["centers_mm"])
        else:
            log.info("k-means sensitivity skipped (too few failures with coordinates)")

    out = save_report(report, PROJECT_ROOT / args.out)
    log.info("Next: python scripts/20_generate_synth.py --from-failures %s "
             "--out data/synth/loop_k<K> --seed <new seed>", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
