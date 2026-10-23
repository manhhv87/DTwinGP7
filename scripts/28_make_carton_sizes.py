#!/usr/bin/env python
"""
28_make_carton_sizes.py
───────────────────────
Fix the carton size of every carton slot of a pose list, in a sidecar file.

The cell has two cartons, 180 x 120 x 120 and 180 x 100 x 80 mm, and the pose lists
name the class only. In session 1 of E2 (e2chuan-1, 23 Oct 2026) the operator chose
the carton each time, and on seven poses (P0021, P0033, P0065, P0073, P0081, P0089,
P0097) the arms did not get the same one, so those pairs did not see the same scene.
The sidecar <list>_carton.csv, next to the pose list, names 80 or 120 for every
carton target and every carton support. 03_run_experiment.py reads it: the prompt
shows the size, the CSV logs it, and the placement check compares the measured top
with it.

Columns: pose_id, carton_mm (size of a carton target), stack_on_carton_mm (size of
a carton support). Only poses with a carton are listed; a blank cell means that role
is not a carton.

Balance. The slots are grouped by role (carton alone, carton on a support, carton
under another part) and by block of --block-rows rows, one sitting of a campaign
(run_blinded_campaign.py --first-pose). Within each group half the slots get each
size, in an order drawn with --seed; a group with an odd count gives its extra slot
to 80 and 120 in turn across groups, so the whole list differs by at most one.

Usage (the committed sidecars were written with the default seed):
    python scripts/28_make_carton_sizes.py --pose-list config/pose_lists/std_v2.csv
    python scripts/28_make_carton_sizes.py --pose-list config/pose_lists/hard_v2.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestrator.placement_check import (  # noqa: E402
    CARTON, CARTON_TOP_MM, carton_sizes_path,
)

SIZES = tuple(sorted(CARTON_TOP_MM))       # (80, 120)
DEFAULT_SEED = 2026


def assign_sizes(rows: list[dict], seed: int = DEFAULT_SEED,
                 block_rows: int = 100) -> dict[str, dict[str, int]]:
    """{pose_id: {"carton_mm": size, "stack_on_carton_mm": size}} for every carton
    target and support of ``rows`` (in pose-list order), balanced as described in
    the module docstring."""
    groups: dict[tuple[int, str], list[tuple[str, str]]] = defaultdict(list)
    for i, row in enumerate(rows):
        pid = str(row["pose_id"])
        cls = str(row.get("class_hint", "") or "").strip().lower()
        support = str(row.get("stack_on", "") or "").strip().lower()
        block = i // block_rows
        if cls == CARTON:
            role = "carton_on_support" if support else "carton_alone"
            groups[(block, role)].append((pid, "carton_mm"))
        if support == CARTON:
            groups[(block, "carton_support")].append((pid, "stack_on_carton_mm"))
    rng = random.Random(seed)
    odd = 0
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for key in sorted(groups):
        slots = groups[key]
        sizes = list(SIZES) * (len(slots) // 2)
        if len(slots) % 2:
            sizes.append(SIZES[odd % 2])
            odd += 1
        rng.shuffle(sizes)
        for (pid, field), size in zip(slots, sizes):
            out[pid][field] = size
    return dict(out)


def write_sidecar(rows: list[dict], sizes: dict[str, dict[str, int]], path: Path) -> int:
    """Write the sidecar in pose-list order; returns the number of rows written."""
    n = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pose_id", "carton_mm", "stack_on_carton_mm"])
        for row in rows:
            pid = str(row["pose_id"])
            if pid in sizes:
                s = sizes[pid]
                w.writerow([pid, s.get("carton_mm", ""), s.get("stack_on_carton_mm", "")])
                n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pose-list", required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--block-rows", type=int, default=100,
                   help="Rows per sitting; each sitting is balanced on its own. "
                        "Default 100.")
    p.add_argument("--out", default=None,
                   help="Default: <pose list stem>_carton.csv next to the pose list.")
    args = p.parse_args()

    pose_list = Path(args.pose_list)
    if not pose_list.is_absolute():
        pose_list = PROJECT_ROOT / pose_list
    with pose_list.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sizes = assign_sizes(rows, args.seed, args.block_rows)
    out = Path(args.out) if args.out else carton_sizes_path(pose_list)
    n = write_sidecar(rows, sizes, out)
    counts = defaultdict(int)
    for s in sizes.values():
        for v in s.values():
            counts[v] += 1
    print(f"{out}: {n} poses, carton slots per size {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
