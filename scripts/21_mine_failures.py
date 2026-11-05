#!/usr/bin/env python
"""
21_mine_failures.py
───────────────────
C3 step: mine failure modes from trial CSVs and allocate the next rendering
budget (Algorithm 1 of the paper: cluster failures → reallocate budget).

Failures are binned by the STAGED class and card cell: each trial's pose_id is
joined to the pose list its run metadata names (resolved under --pose-root, the
repo root by default). Only adaptation poses 221-300 are accepted unless --poses
overrides the range. A failure is the operator's verdict (human_ok) when one was
recorded, else the machine flag. The report records the SHA-256 of every mined
CSV and is written even when the round has no failures (then with an empty
allocation: no failure-guided update is defined).

Usage (adaptation session of E4 update 1; mine its main CSVs, not doichung/):
    python scripts/21_mine_failures.py --runs "results/e4adapt-1/experiment_real_*.csv" \
        --n-budget 1000 --guided-seed 10000 --control-seed 20000 \
        --out results/adaptation/e4_k1/failure_modes.json

Then generate the loop iteration's synthetic batch and the CONTROL batch of E4
(same budget, no failure guidance) with the seeds this report checked:
    python scripts/20_generate_synth.py --from-failures results/adaptation/e4_k1/failure_modes.json \
        --kappa <selected kappa> --out data/synth/loop_k1 --seed 10000
    python scripts/20_generate_synth.py --control-for results/adaptation/e4_k1/failure_modes.json \
        --kappa <selected kappa> --out data/synth/control_k1 --seed 20000
Both refuse a --seed or --n other than the report's plan, and a report without
that arm's checked seed; without --seed and --n they take the plan's values.

Update 2 uses seeds 30000 (guided) and 40000 (control). Scene i of a run draws
from SeedSequence([seed, i]) (spec_version 2), so batches with different seeds
share no scene, while a seed reused reuses its scenes from index 0; E3 and E5 use
seed 0. --guided-seed/--control-seed check the planned streams against the scene
manifests already written (--used-specs) and each other, and refuse an overlap.
"""
from __future__ import annotations

import argparse
import glob
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.synthgen import (  # noqa: E402
    SynthgenConfig,
    allocate,
    bin_failures,
    kmeans_sensitivity,
    save_report,
)
from src.synthgen.failure_miner import ADAPTATION_POSES, file_sha256, stage_trials  # noqa: E402
from src.synthgen.scene_sampler import SPEC_VERSION  # noqa: E402
from src.synthgen.seed_ranges import (  # noqa: E402
    STREAM_RULE,
    manifest_streams,
    planned_streams,
    shared_streams,
)
from src.utils import setup_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Trial CSV file(s)/wildcards from 03_run_experiment.py")
    parser.add_argument("--poses", type=int, nargs=2, default=list(ADAPTATION_POSES),
                        metavar=("FIRST", "LAST"),
                        help="Accepted pose numbers, inclusive (default: the adaptation "
                             "poses 221-300). Any other trial makes the run refuse.")
    parser.add_argument("--pose-root", default=str(PROJECT_ROOT),
                        help="Directory the pose-list path in the run metadata is relative to")
    parser.add_argument("--pose-list", default=None,
                        help="Pose list to use instead of the metadata's location; its "
                             "file name must match the one the metadata names")
    parser.add_argument("--grid", type=int, nargs=2, default=[3, 3],
                        metavar=("NX", "NY"), help="Pick-region binning grid")
    parser.add_argument("--n-budget", type=int, default=1000,
                        help="Rendering budget N_k for the next loop iteration")
    parser.add_argument("--floor-frac", type=float, default=0.05,
                        help="Minimum budget share per failure mode")
    parser.add_argument("--kmeans-k", type=int, default=4,
                        help="k for the k-means sensitivity check (0 = skip)")
    parser.add_argument("--guided-seed", type=int, default=None,
                        help="Seed planned for the guided batch (checked, recorded)")
    parser.add_argument("--control-seed", type=int, default=None,
                        help="Seed planned for the equal-budget control batch (checked, recorded)")
    parser.add_argument("--used-specs", nargs="+", default=["data/synth/*/specs/manifest.json"],
                        help="Scene manifests whose seed/index streams are already used")
    parser.add_argument("--config", default="config/synthgen.yaml",
                        help="Synthgen config (supplies the pick-region bounds)")
    parser.add_argument("--out", default="results/failure_modes.json")
    return parser.parse_args(argv)


def _expand(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        paths += glob.glob(str(PROJECT_ROOT / pattern)) + glob.glob(pattern)
    return sorted({str(Path(p).resolve()) for p in paths})


def generation_plan(guided_seed: int | None, control_seed: int | None, n_images: int,
                    used_specs: list[str]) -> dict | None:
    """Check the streams of the planned guided/control batches; refuse an overlap.

    Each batch is scenes 0..n_images-1 of its seed, as 20_generate_synth.py writes
    them with the current sampler (SPEC_VERSION), which the plan records.
    """
    planned = {}
    if guided_seed is not None:
        planned["guided"] = planned_streams(guided_seed, n_images, spec_version=SPEC_VERSION)
    if control_seed is not None:
        planned["control"] = planned_streams(control_seed, n_images, spec_version=SPEC_VERSION)
    if not planned:
        return None
    checked, unchecked = [], []
    for path in _expand(used_specs):
        info = manifest_streams(path)
        info["sha256"] = file_sha256(path)
        (unchecked if info["streams"] is None else checked).append(info)
    conflicts = [
        f"{name} {ranges} shares {shared_streams(ranges, info['streams'])} stream(s) with "
        f"{info['path']} (seed {info['seed']})"
        for name, ranges in planned.items() for info in checked
        if shared_streams(ranges, info["streams"])
    ]
    if len(planned) == 2 and shared_streams(planned["guided"], planned["control"]):
        conflicts.append("the guided and control ranges overlap each other")
    if conflicts:
        raise ValueError("Planned generation streams are already used: " + "; ".join(conflicts))
    return {"spec_version": SPEC_VERSION, "rule": STREAM_RULE[SPEC_VERSION], "n_images": n_images,
            "guided_seed": guided_seed, "control_seed": control_seed,
            "planned_streams": {name: [[list(first), list(second)] for first, second in blocks]
                                for name, blocks in planned.items()},
            "checked_against": [{key: info[key] for key in
                                 ("path", "sha256", "spec_version", "seed", "streams")}
                                for info in checked],
            "unchecked": [{key: info[key] for key in ("path", "sha256", "unchecked")}
                          for info in unchecked]}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("mine_failures")

    paths = _expand(args.runs)
    if not paths:
        log.error("No trial CSVs match %s", args.runs)
        return 1

    config_path = PROJECT_ROOT / args.config
    cfg = SynthgenConfig.from_yaml(config_path)
    region = (
        cfg.objects.region_x_mm[0], cfg.objects.region_x_mm[1],
        cfg.objects.region_y_mm[0], cfg.objects.region_y_mm[1],
    )
    poses = (args.poses[0], args.poses[1])
    try:
        rows, inputs = stage_trials(paths, pose_root=args.pose_root,
                                    pose_list=args.pose_list, poses=poses)
        plan = generation_plan(args.guided_seed, args.control_seed, args.n_budget,
                               args.used_specs)
    except (ValueError, FileNotFoundError) as exc:
        log.error("Refused: %s", exc)
        return 2
    if poses != ADAPTATION_POSES:
        log.warning("Pose range %d-%d overrides the adaptation poses %d-%d: these "
                    "outcomes must not be evaluation trials of the same comparison",
                    *poses, *ADAPTATION_POSES)
    if plan is None:
        log.warning("No --guided-seed/--control-seed given: generation streams unchecked, "
                    "and 20_generate_synth.py refuses a report without a checked seed")
    elif not plan["checked_against"]:
        log.warning("No scene manifest matches %s with a known stream rule: the planned "
                    "ranges were checked only against each other", args.used_specs)

    report = bin_failures(rows, region=region, grid=tuple(args.grid))
    report.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "poses": list(poses),
        "pose_range_override": poses != ADAPTATION_POSES,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "n_budget": args.n_budget,
        "floor_frac": args.floor_frac,
        "generation_plan": plan,
    })
    log.info("Trials: %d | failures: %d | modes: %d",
             report["n_trials"], report["n_failures"], len(report["clusters"]))
    for item in inputs:
        log.info("  %s  sha256 %s  %s", Path(item["csv"]).name, item["sha256"],
                 Path(item["pose_list"]["path"]).name)

    if report["n_failures"] == 0:
        report["allocation"] = []
        report["update"] = "none: no failed trials, so no failure-guided update is defined"
        log.warning("No failures found — record that no update is defined and stop the loop")
        save_report(report, PROJECT_ROOT / args.out)
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

    if args.kmeans_k > 0:
        km = kmeans_sensitivity(rows, k=args.kmeans_k)
        report["kmeans_sensitivity"] = km
        if km:
            log.info("k-means sensitivity (k=%d, n=%d): counts=%s centers=%s",
                     km["k"], km["n_points"], km["counts"], km["centers_mm"])
        else:
            log.info("k-means sensitivity skipped (too few failures with positions)")

    out = save_report(report, PROJECT_ROOT / args.out)
    if args.guided_seed is not None:
        log.info("Next: python scripts/20_generate_synth.py --from-failures %s "
                 "--kappa <selected kappa> --out data/synth/loop_k<K> --seed %d",
                 out, args.guided_seed)
    if args.control_seed is not None:
        log.info("Control: python scripts/20_generate_synth.py --control-for %s "
                 "--kappa <selected kappa> --out data/synth/control_k<K> --seed %d",
                 out, args.control_seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
