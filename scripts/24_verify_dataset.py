#!/usr/bin/env python
"""
24_verify_dataset.py
────────────────────
Verify (and repair) a YOLO-seg dataset before it ever reaches training.

Catches the classic Roboflow trap: exports ordered alphabetically
(carton, metal_box, plastic_box, wood_box) while the paper's canonical order is
(carton, plastic_box, wood_box, metal_box). Training on shifted ids silently swaps classes.

Usage:
    # after assembling data/real_dataset from the Roboflow export:
    python scripts/24_verify_dataset.py --data data/real_dataset
    # if it reports a class-order MISMATCH, repair in place (labels are
    # backed up to labels_orig/ first) and re-verify:
    python scripts/24_verify_dataset.py --data data/real_dataset --fix

Full capture-to-training walkthrough: DATASET_DESIGN (paper repo) +
docs/DATASET_DESIGN.md (spec).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logging  # noqa: E402
from src.utils.dataset_check import (  # noqa: E402
    EXPECTED_NAMES,
    format_report,
    remap_labels,
    verify_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True,
                        help="Dataset root (contains dataset.yaml/data.yaml, images/, labels/)")
    parser.add_argument("--splits", default="train,val",
                        help="Comma list of splits to check (default train,val)")
    parser.add_argument("--expect", default=",".join(EXPECTED_NAMES),
                        help="Canonical class order (default: platform order)")
    parser.add_argument("--fix", action="store_true",
                        help="Remap label ids to the canonical order + write "
                             "canonical dataset.yaml (originals backed up)")
    parser.add_argument("--min-per-class", type=int, default=0,
                        help="Fail if any class has fewer instances (use 180 "
                             "for the train split target)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging("verify_dataset")
    root = Path(args.data)
    if not root.is_absolute():
        root = PROJECT_ROOT / args.data
    splits = tuple(s for s in args.splits.split(",") if s)
    expected = [s for s in args.expect.split(",") if s]

    if args.fix:
        result = remap_labels(root, expected_names=expected, splits=splits)
        if result["noop"]:
            log.info("Class order already canonical — dataset.yaml normalized only")
        else:
            log.info("Remapped ids %s across %d file(s); originals in labels_orig/",
                     result["mapping"], result["changed_files"])

    report = verify_dataset(root, expected_names=expected, splits=splits)
    for line in format_report(report).splitlines():
        log.info("%s", line)

    if args.min_per_class > 0:
        thin = {k: v for k, v in report["class_counts"].items()
                if v < args.min_per_class}
        if thin:
            log.error("Below --min-per-class %d: %s — capture more of these classes",
                      args.min_per_class, thin)
            return 1

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
