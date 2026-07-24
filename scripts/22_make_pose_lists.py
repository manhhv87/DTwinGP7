#!/usr/bin/env python
"""
22_make_pose_lists.py
─────────────────────
Generate PRE-DRAWN pose lists for the paired trial protocol (paper §3.7).

Each pose = a numbered template card position + a yaw. The SAME list is run
under every training configuration being compared, so grasp outcomes pair
pose-by-pose and McNemar's exact test applies. The operator places the part
on the printed card WITHOUT looking at the detector output.

Usage:
    # Standard test set, 300 trials:
    python scripts/22_make_pose_lists.py --n 300 --seed 42 \
        --out config/pose_lists/std_v1.csv

    # Hard test set (cycles unseen conditions; build the physical variations
    # per WORKFLOW_CODE_TO_PAPER.md §5 / paper §3.7):
    python scripts/22_make_pose_lists.py --n 300 --seed 43 --hard \
        --out config/pose_lists/hard_v1.csv

Also writes <out>.cards.txt — the card grid layout to print and lay on the table.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.synthgen import SynthgenConfig  # noqa: E402
from src.utils import setup_logging  # noqa: E402

# Hard-set condition labels cycled over trials with --hard. They must match
# the physical setups you build AND the --lighting labels used in 03 runs.
DEFAULT_HARD_CONDITIONS = ["dim", "side_light", "novel_background"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=300, help="Number of trials")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cards", type=int, nargs=2, default=[5, 4],
                        metavar=("NX", "NY"),
                        help="Template card grid over the pick region (default 5x4=20)")
    parser.add_argument("--margin-mm", type=float, default=40.0,
                        help="Keep cards this far from the region edge")
    parser.add_argument("--config", default="config/synthgen.yaml",
                        help="Supplies the pick-region bounds (single source of truth)")
    parser.add_argument("--class-hints", default="carton,plastic_box,wood_box,metal_box",
                        help="Comma list cycled per trial for class balance "
                             "('' = no hint, operator's choice)")
    parser.add_argument("--hard", action="store_true",
                        help="Cycle hard-set condition labels per trial")
    parser.add_argument("--conditions", default=",".join(DEFAULT_HARD_CONDITIONS),
                        help="Hard condition labels to cycle (with --hard)")
    parser.add_argument("--out", required=True, help="Output CSV path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging("pose_lists")

    cfg = SynthgenConfig.from_yaml(PROJECT_ROOT / args.config)
    x0, x1 = cfg.objects.region_x_mm
    y0, y1 = cfg.objects.region_y_mm
    yaw0, yaw1 = cfg.objects.yaw_range_deg
    m = args.margin_mm
    nx, ny = args.cards

    # Card grid: cell centers over the margined region, card_id row-major 1..nx*ny.
    xs = np.linspace(x0 + m, x1 - m, nx)
    ys = np.linspace(y0 + m, y1 - m, ny)
    cards = [(ix + iy * nx + 1, float(xs[ix]), float(ys[iy]))
             for iy in range(ny) for ix in range(nx)]

    classes = [c for c in args.class_hints.split(",") if c]
    for c in classes:
        if c not in cfg.objects.classes:
            log.error("class hint '%s' not in synthgen classes %s",
                      c, sorted(cfg.objects.classes))
            return 1
    conditions = [c for c in args.conditions.split(",") if c] if args.hard else []

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pose_id", "card_id", "x_mm", "y_mm", "yaw_deg",
                         "class_hint", "condition"])
        for i in range(args.n):
            card_id, x, y = cards[int(rng.integers(0, len(cards)))]
            yaw = float(rng.uniform(yaw0, yaw1))
            writer.writerow([
                f"P{i + 1:04d}", card_id, round(x, 1), round(y, 1), round(yaw, 1),
                classes[i % len(classes)] if classes else "",
                conditions[i % len(conditions)] if conditions else "",
            ])

    cards_txt = out.with_suffix(out.suffix + ".cards.txt")
    with cards_txt.open("w", encoding="utf-8") as f:
        f.write(f"Template card layout {nx}x{ny} over pick region "
                f"x[{x0},{x1}] y[{y0},{y1}] mm (margin {m} mm)\n")
        f.write("Print, cut, and fix the cards at these BASE-frame coordinates:\n\n")
        for card_id, x, y in cards:
            f.write(f"  card {card_id:>2}: x={x:7.1f} mm  y={y:7.1f} mm\n")

    log.info("Pose list: %d trials, %d cards, seed %d → %s",
             args.n, len(cards), args.seed, out)
    if conditions:
        log.info("Hard conditions cycled: %s", conditions)
    log.info("Card layout for printing → %s", cards_txt)
    log.info("Run every configuration under comparison with THIS list: "
             "03_run_experiment.py --pose-list %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
