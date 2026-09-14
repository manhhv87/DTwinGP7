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

# Hard-set condition labels for --hard. The paper's primary hard set is the
# TWO-FACTOR family (novel background under dimmed light), so that is the default:
# one condition, staged once, held for the whole campaign. The single-factor
# families are secondary descriptors scored on images, not on grasp trials — pass
# them explicitly if you really want grasp data on them, and note that every extra
# label means the operator must restage the cell between blocks.
DEFAULT_HARD_CONDITIONS = ["novel_background_dim"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=300, help="Number of trials")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cards", type=int, nargs=2, default=[5, 4],
                        metavar=("NX", "NY"),
                        help="Template card grid over the pick region (default 5x4=20)")
    parser.add_argument("--yaw-step-deg", type=float, default=5.0,
                        help="Round each yaw to this step so the operator can set it against "
                             "the printed tick ring on the card, and repeat the SAME angle in "
                             "every configuration being compared. 0 disables rounding.")
    parser.add_argument("--margin-mm", type=float, default=40.0,
                        help="Keep cards this far from the region edge")
    parser.add_argument("--config", default="config/synthgen.yaml",
                        help="Supplies the pick-region bounds (single source of truth)")
    parser.add_argument("--class-hints",
                        default="carton,plastic_box,wood_box,metal_box,inox_box",
                        help="Comma list cycled per trial for class balance "
                             "('' = no hint, operator's choice). The inox case is the "
                             "decisive class of the depth-mode comparison, so leaving it "
                             "out produces a campaign that cannot answer C4.")
    parser.add_argument("--class-weights", default="",
                        help="Comma list of weights matching --class-hints, e.g. "
                             "'1,1,1,1,4' to spend 40 percent of the trials on the last "
                             "class. McNemar power is set by the number of pairs in the "
                             "deciding cell, not by the total, so enrich that cell rather "
                             "than splitting the budget evenly. Empty = equal shares.")
    parser.add_argument("--stack-frac", type=float, default=0.0,
                        help="Fraction of trials staged as a stacked scene: the operator "
                             "puts the class named in the stack_on column on the card "
                             "first, then the target part on top of it. The one-layer "
                             "assumption of the plane depth mode is exactly what these "
                             "trials probe, so the stacked rows of the depth-mode table "
                             "have no source without them. 0 = none.")
    parser.add_argument("--hard", action="store_true",
                        help="Attach hard-set condition labels")
    parser.add_argument("--conditions", default=",".join(DEFAULT_HARD_CONDITIONS),
                        help="Hard condition labels (with --hard). Emitted in CONTIGUOUS "
                             "BLOCKS, never alternated per trial: restaging the background "
                             "and the lighting every third trial is not something anyone "
                             "does 200 times.")
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

    # Class order: cycled so the balance is exact rather than approximate, and
    # weighted so the deciding class can be given more of the budget.
    if classes:
        if args.class_weights:
            w = [int(x) for x in args.class_weights.split(",")]
            if len(w) != len(classes):
                log.error("--class-weights has %d entries but --class-hints has %d",
                          len(w), len(classes))
                return 1
            cycle = [c for c, k in zip(classes, w) for _ in range(k)]
        else:
            cycle = list(classes)
    else:
        cycle = [""]

    # Stacked trials: spread evenly through the list rather than clustered, so a
    # half-finished run still holds a representative share of them.
    n_stack = int(round(args.stack_frac * args.n))
    stack_at = set(np.linspace(0, args.n - 1, n_stack).round().astype(int).tolist()) \
        if n_stack else set()

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pose_id", "card_id", "x_mm", "y_mm", "yaw_deg",
                         "class_hint", "condition", "stack_on"])
        for i in range(args.n):
            card_id, x, y = cards[int(rng.integers(0, len(cards)))]
            yaw = float(rng.uniform(yaw0, yaw1))
            if args.yaw_step_deg > 0:
                # The operator sets the angle by eye against a printed tick ring; a yaw of
                # 85.6 deg cannot be placed, let alone repeated in the next run.
                yaw = round(yaw / args.yaw_step_deg) * args.yaw_step_deg
            cls = cycle[i % len(cycle)]
            # Contiguous blocks, not per-trial alternation: the operator restages the
            # cell once per block.
            cond = conditions[(i * len(conditions)) // args.n] if conditions else ""
            if i in stack_at and classes:
                # Support box: any class other than the target, cycled for variety.
                others = [c for c in classes if c != cls] or [cls]
                stack_on = others[i % len(others)]
            else:
                stack_on = ""
            writer.writerow([
                f"P{i + 1:04d}", card_id, round(x, 1), round(y, 1), round(yaw, 1),
                cls, cond, stack_on,
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
