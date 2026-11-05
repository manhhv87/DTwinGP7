#!/usr/bin/env python
"""
20_generate_synth.py
────────────────────
C2 entry point: sample scene specs for anchored/blind domain randomization,
then (separately) render with BlenderProc2 and convert to a YOLO-seg dataset.

Three-step flow (rendering runs in the blenderproc env, NOT the repo venv):

  1) Sample specs (this script, repo venv):
       python scripts/20_generate_synth.py --mode anchored --kappa 2 --n 3000 \
           --seed 0 --out data/synth/anchored_k2
       python scripts/20_generate_synth.py --mode blind --n 3000 \
           --seed 0 --out data/synth/blind
     C3 loop iteration (budget and seed from 21_mine_failures.py) and its E4 control
     (as many anchored scenes, with the control seed the report checked, unconditioned
     but for the session lighting the report's allocation names, so that both arms
     render the same light and exposure distributions):
       python scripts/20_generate_synth.py --from-failures results/failure_modes.json \
           --kappa 2 --out data/synth/loop_k1 --seed 10000
       python scripts/20_generate_synth.py --control-for results/failure_modes.json \
           --kappa 2 --out data/synth/control_k1 --seed 20000
     E5 single-factor intervention (same seed/index as the full anchored recipe):
       python scripts/20_generate_synth.py --mode anchored --ablation camera \
           --kappa 2 --n 3000 --seed 0 --out data/synth/e5_camera

  2) Render (any env with `pip install blenderproc`):
       blenderproc run src/synthgen/render_blenderproc.py -- \
           --scenes data/synth/anchored_k2/specs --out data/synth/anchored_k2/render

  3) Convert to YOLO-seg dataset (this script, repo venv):
       python scripts/20_generate_synth.py --make-labels --out data/synth/anchored_k2 \
           --val-frac 0

Paper training budgets use --val-frac 0 (the default) and the separately locked
real validation set. --kappa has no default and is required for anchored scenes. Scene i of a run draws from SeedSequence([seed, i]): a batch with
another seed shares no scene with this one, whatever its index range, while one
seed reused must not reuse an index range. Give every batch its own seed.
--from-failures and --control-for take the seed and scene count that the report's
generation plan checked (21_mine_failures.py --guided-seed/--control-seed) and
refuse a --seed or --n that differs, or a report without a checked seed. Both refuse
a report whose allocation names a lighting label the hard profile does not render
(anchored_config.LIGHTING_CONDITION_SCALE), and the control refuses one that names
more than one.

Anchored sampling needs config/calibration/T_base_camera_sigma.json (02_run_
calibration.py --bootstrap 200). Without it the run stops; --allow-sigma-fallback
runs on the YAML fallback widths instead, for development only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.synthgen import (  # noqa: E402
    LIGHTING_CONDITION_SCALE, SceneSampler, SynthgenConfig, load_sigma_json)
from src.synthgen.scene_sampler import ABLATION_FACTORS, SPEC_VERSION  # noqa: E402
from src.utils import setup_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["anchored", "blind"], default="anchored")
    parser.add_argument("--ablation", choices=ABLATION_FACTORS, default="none",
                        help="E5: fix/remove one factor of the unconditional anchored "
                             "recipe; see docs/E5_ABLATION_PROTOCOL.md")
    parser.add_argument("--kappa", type=float, default=None,
                        help="Width multiplier of Eq. (1); E3 selects among {1,2,4}, E4 and E5 "
                             "use the selected one. Required for anchored scenes (also with "
                             "--ablation, --from-failures, --control-for): there is no default, "
                             "so a forgotten kappa cannot fill a batch with kappa 2")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of scenes (default 3000; with a failure report, "
                             "the count its plan checked)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run seed (default 0; with a failure report, the seed its "
                             "plan checked, and no other)")
    parser.add_argument("--out", required=True,
                        help="Run dir; specs → <out>/specs, render → <out>/render, "
                             "dataset → <out>/dataset")
    parser.add_argument("--config", default="config/synthgen.yaml")
    parser.add_argument("--calib", default="config/calibration/T_base_camera.npy")
    parser.add_argument("--sigma", default="config/calibration/T_base_camera_sigma.json",
                        help="Bootstrap sigma JSON (02_run_calibration.py --bootstrap). "
                             "Anchored sampling stops when it is missing, unless "
                             "--allow-sigma-fallback is given.")
    parser.add_argument("--allow-sigma-fallback", action="store_true",
                        help="Development only: sample anchored scenes on the YAML "
                             "fallback widths when the bootstrap sigma file is missing. "
                             "The manifest then records sigma_source yaml_fallback; "
                             "paper runs must not use it.")
    parser.add_argument("--from-failures", default=None,
                        help="failure_modes.json from 21_mine_failures.py — generates "
                             "per-mode batches per its 'allocation' (C3 loop)")
    parser.add_argument("--control-for", default=None,
                        help="failure_modes.json from 21_mine_failures.py — generates the "
                             "equal-budget E4 control batch: unconditioned anchored scenes "
                             "with the control seed its plan checked")
    parser.add_argument("--make-labels", action="store_true",
                        help="Convert <out>/render → <out>/dataset (after step 2)")
    parser.add_argument("--val-frac", type=float, default=0.0,
                        help="Synthetic validation fraction (default 0: every paper budget "
                             "trains on all frames and validates on real images only)")
    args = parser.parse_args(argv)
    if args.ablation != "none":
        if args.mode != "anchored":
            parser.error("--ablation requires --mode anchored")
        if args.from_failures or args.control_for:
            parser.error("--ablation cannot be combined with --from-failures or --control-for")
        if args.make_labels:
            parser.error("--ablation applies when sampling; omit it for --make-labels")
    if args.from_failures and args.control_for:
        parser.error("--from-failures and --control-for generate different E4 arms; give one")
    if args.control_for and args.mode != "anchored":
        parser.error("--control-for samples anchored scenes")
    if not 0.0 <= args.val_frac < 1.0:
        parser.error("--val-frac must be in [0, 1)")
    # --from-failures always samples anchored scenes, whatever --mode says.
    if not args.make_labels and (args.mode == "anchored" or args.from_failures) and args.kappa is None:
        parser.error("--kappa is required for anchored scenes (E3: 1, 2 or 4; E4 and E5: the kappa "
                     "selected in E3, results/selection/e3_kappa.json)")
    if args.kappa is not None and args.kappa <= 0:
        parser.error("--kappa must be positive")
    return args


def planned_batch(report: dict, arm: str, seed: int | None, n: int | None) -> tuple[int, int]:
    """Seed and scene count of an E4 arm's batch as the miner's report planned them.

    21_mine_failures.py checked scenes 0..n_images-1 of the guided and control
    seeds against the batches already written, for the sampler's spec version.
    The guided arm writes the allocation, the control arm as many unconditioned
    scenes. A --seed or --n other than the plan's, a plan without this arm's seed,
    or one made for another spec version is refused: the streams written are the
    streams checked.
    """
    plan = report.get("generation_plan") or {}
    checked = plan.get(f"{arm}_seed")
    if checked is None:
        raise ValueError(f"the report records no checked {arm} seed; re-run "
                         f"21_mine_failures.py with --{arm}-seed")
    if plan.get("spec_version") != SPEC_VERSION:
        raise ValueError(f"the plan was checked for spec_version {plan.get('spec_version')!r}, "
                         f"but the sampler writes {SPEC_VERSION}; re-run 21_mine_failures.py")
    if seed is not None and seed != checked:
        raise ValueError(f"--seed {seed} is not the {arm} seed {checked} the report checked")
    planned = int(plan["n_images"])
    if n is not None and n != planned:
        raise ValueError(f"--n {n} is not the {planned} scenes the report checked")
    allocated = sum(max(0, int(entry["n_images"])) for entry in report["allocation"])
    if allocated != planned:
        raise ValueError(f"the allocation holds {allocated} scenes, the plan checked {planned}")
    return int(checked), planned


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("synthgen")
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out

    # ─── Step 3: labels only ───
    if args.make_labels:
        from src.synthgen import convert_render_output

        stats = convert_render_output(
            out / "render", out / "dataset", val_frac=args.val_frac,
            seed=0 if args.seed is None else args.seed)
        log.info("Dataset: %s", stats)
        log.info("Train on the GPU machine: see scripts/23_launch_retrain.py")
        return 0

    # ─── Step 1: sample specs ───
    config = SynthgenConfig.from_yaml(PROJECT_ROOT / args.config)
    no_material = sorted(set(config.objects.classes) - set(config.objects.materials))
    if no_material:
        log.warning("No material configured for %s: those classes render in neutral "
                    "grey and teach the detector nothing about their colour. Add them "
                    "under objects.materials in %s.", no_material, args.config)

    import numpy as np
    calib_path = PROJECT_ROOT / args.calib
    if not calib_path.exists():
        log.error("Calibration not found: %s — run 02_run_calibration.py first "
                  "(anchoring NEEDS the measured T_BC)", calib_path)
        return 1
    T_BC = np.load(calib_path)

    sigma = load_sigma_json(PROJECT_ROOT / args.sigma)
    # --from-failures always samples anchored scenes, whatever --mode says; the
    # control arm (--control-for) needs --mode anchored, the default.
    if (args.mode == "anchored" or args.from_failures) and sigma is None:
        if not args.allow_sigma_fallback:
            log.error(
                "No bootstrap sigma at %s: anchored widths would come from the YAML "
                "fallback, not from the calibration. Run 02_run_calibration.py "
                "--bootstrap 200 first, or pass --allow-sigma-fallback for a "
                "development run.", args.sigma)
            return 1
        log.warning(
            "No bootstrap sigma at %s → using YAML fallback sigmas "
            "(--allow-sigma-fallback). Not for paper runs.", args.sigma)

    specs_dir = out / "specs"
    seed = 0 if args.seed is None else args.seed
    n = 3000 if args.n is None else args.n

    report_path = args.from_failures or args.control_for
    if report_path:
        with (PROJECT_ROOT / report_path).open("r", encoding="utf-8") as f:
            report = json.load(f)
        allocation = report.get("allocation")
        if not allocation:
            log.error("%s has no 'allocation' — run 21_mine_failures.py with "
                      "--n-budget", report_path)
            return 1
        try:
            seed, n = planned_batch(report, "guided" if args.from_failures else "control",
                                    args.seed, args.n)
        except ValueError as exc:
            log.error("Refused: %s", exc)
            return 1
        # Every bin of a hard-condition session carries its lighting label; the profile
        # renders only the labels it was fitted for, and refuses the rest up front.
        labels = {(e.get("condition") or {}).get("lighting") for e in allocation} - {None}
        unknown = sorted(labels - set(LIGHTING_CONDITION_SCALE))
        if unknown:
            log.error("%s holds lighting labels %s the hard profile does not render (%s)",
                      report_path, unknown, sorted(LIGHTING_CONDITION_SCALE))
            return 1
        if args.control_for and len(labels) > 1:
            log.error("%s mixes lighting labels %s; the control batch renders one session "
                      "condition", report_path, sorted(labels))
            return 1

    if args.from_failures:
        sampler = SceneSampler(config, T_BC, mode="anchored", kappa=args.kappa,
                               sigma=sigma, seed=seed)
        start = 0
        for entry in allocation:
            n_img = int(entry["n_images"])
            if n_img <= 0:
                continue
            sampler.write_specs(specs_dir, n_img, condition=entry["condition"],
                                start_index=start)
            log.info("Mode %-40s → %d specs (start %d)",
                     entry["key"], n_img, start)
            start += n_img
        total = start
    else:
        # Blind sampling ignores kappa (its manifest records none).
        sampler = SceneSampler(config, T_BC, mode=args.mode,
                               kappa=1.0 if args.kappa is None else args.kappa,
                               sigma=sigma, seed=seed, ablation=args.ablation)
        # The control batch renders the session's lighting, as the guided arm does,
        # and nothing else of the failure bins.
        condition = None
        if args.control_for and labels:
            condition = {"class_name": None, "region": None, "lighting": sorted(labels)[0]}
        sampler.write_specs(specs_dir, n, condition=condition)
        total = n

    log.info("Wrote %d scene specs → %s (seed %d; sigma source: %s; ablation: %s; "
             "exposure gain log-uniform in %s)", total, specs_dir, seed, sampler.sigma_source,
             sampler.ablation, config.exposure.gain_range)
    log.info("Next step — render in the blenderproc env:")
    log.info("  blenderproc run src/synthgen/render_blenderproc.py -- "
             "--scenes %s --out %s", specs_dir, out / "render")
    log.info("Then (paper training budget): python scripts/20_generate_synth.py "
             "--make-labels --val-frac 0 --out %s",
             args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
