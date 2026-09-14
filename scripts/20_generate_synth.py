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
     C3 loop iteration (budget from 21_mine_failures.py):
       python scripts/20_generate_synth.py --from-failures results/failure_modes.json \
           --out data/synth/loop_k1 --seed 1000

  2) Render (any env with `pip install blenderproc`):
       blenderproc run src/synthgen/render_blenderproc.py -- \
           --scenes data/synth/anchored_k2/specs --out data/synth/anchored_k2/render

  3) Convert to YOLO-seg dataset (this script, repo venv):
       python scripts/20_generate_synth.py --make-labels --out data/synth/anchored_k2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.synthgen import SceneSampler, SynthgenConfig, load_sigma_json  # noqa: E402
from src.utils import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["anchored", "blind"], default="anchored")
    parser.add_argument("--kappa", type=float, default=2.0,
                        help="Width multiplier of Eq. (1); E3 selects among {1,2,4}")
    parser.add_argument("--n", type=int, default=3000, help="Number of scenes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True,
                        help="Run dir; specs → <out>/specs, render → <out>/render, "
                             "dataset → <out>/dataset")
    parser.add_argument("--config", default="config/synthgen.yaml")
    parser.add_argument("--calib", default="config/calibration/T_base_camera.npy")
    parser.add_argument("--sigma", default="config/calibration/T_base_camera_sigma.json",
                        help="Bootstrap sigma JSON (02_run_calibration.py --bootstrap). "
                             "Missing file → YAML fallback sigmas (warned).")
    parser.add_argument("--from-failures", default=None,
                        help="failure_modes.json from 21_mine_failures.py — generates "
                             "per-mode batches per its 'allocation' (C3 loop)")
    parser.add_argument("--make-labels", action="store_true",
                        help="Convert <out>/render → <out>/dataset (after step 2)")
    parser.add_argument("--val-frac", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging("synthgen")
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out

    # ─── Step 3: labels only ───
    if args.make_labels:
        from src.synthgen import convert_render_output

        stats = convert_render_output(
            out / "render", out / "dataset", val_frac=args.val_frac, seed=args.seed)
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
    if args.mode == "anchored" and sigma is None:
        log.warning(
            "No bootstrap sigma at %s → using YAML fallback sigmas. For the "
            "paper run 02_run_calibration.py --bootstrap 200 first.", args.sigma)

    specs_dir = out / "specs"

    if args.from_failures:
        with (PROJECT_ROOT / args.from_failures).open("r", encoding="utf-8") as f:
            report = json.load(f)
        allocation = report.get("allocation")
        if not allocation:
            log.error("%s has no 'allocation' — run 21_mine_failures.py with "
                      "--n-budget", args.from_failures)
            return 1
        sampler = SceneSampler(config, T_BC, mode="anchored", kappa=args.kappa,
                               sigma=sigma, seed=args.seed)
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
        sampler = SceneSampler(config, T_BC, mode=args.mode, kappa=args.kappa,
                               sigma=sigma, seed=args.seed)
        sampler.write_specs(specs_dir, args.n)
        total = args.n

    log.info("Wrote %d scene specs → %s (sigma source: %s)",
             total, specs_dir, sampler.sigma_source)
    log.info("Next step — render in the blenderproc env:")
    log.info("  blenderproc run src/synthgen/render_blenderproc.py -- "
             "--scenes %s --out %s", specs_dir, out / "render")
    log.info("Then: python scripts/20_generate_synth.py --make-labels --out %s",
             args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
