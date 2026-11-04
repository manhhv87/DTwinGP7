#!/usr/bin/env python
"""Apply the Methods selection rules to returned training packages; write a hashed record.

Reads runs/seed<n>/run_record.json and results.csv of each package (as left by its
train.py), re-checks every run, and writes the decision once to --out, with the
record's SHA-256 in <out>.sha256. Nothing is trained or evaluated here, and no
physical or test-image outcome enters any choice.

E3 kappa (one package per kappa, all declared seeds complete):
    python scripts/select_trained_models.py kappa --candidate 1=data/packages/e3_k1/dsv1 \
        --candidate 2=data/packages/e3_k2/dsv1 --candidate 4=data/packages/e3_k4/dsv1 \
        --out results/selection/e3_kappa.json
  The deployed seed of the selected kappa is f0 (e3_anchored.pt).

Deployed seed of one configuration (E2, E3 wide-range, an E5 factor):
    python scripts/select_trained_models.py deploy --role e3_wide \
        --package data/packages/e3_wide/dsv1 --out results/selection/e3_wide.json

E4, the update-1 loop model that runs the second adaptation block (e4_guided1.pt):
    python scripts/select_trained_models.py e4-adaptation \
        --package data/packages/e4_loop_u1/dsv1 --out results/selection/e4_adaptation2.json

E4, the final checkpoint of each arm (update-2 packages, one per seed):
    python scripts/select_trained_models.py e4-final \
        --loop data/packages/e4_loop_u2_s0/dsv1 data/packages/e4_loop_u2_s1/dsv1 data/packages/e4_loop_u2_s2/dsv1 \
        --control data/packages/e4_control_u2_s0/dsv1 data/packages/e4_control_u2_s1/dsv1 data/packages/e4_control_u2_s2/dsv1 \
        --out results/selection/e4_final.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.selection import (  # noqa: E402
    select_deployed,
    select_e4_adaptation,
    select_e4_final,
    select_kappa,
    write_record,
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _candidate(value: str) -> tuple[float, Path]:
    kappa, separator, package = value.partition("=")
    try:
        if not separator:
            raise ValueError
        return float(kappa), _resolve(package)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected KAPPA=PACKAGE, got {value!r}") from None


def parser_for_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    kappa = commands.add_parser("kappa", help="E3: choose kappa, then its deployed seed")
    kappa.add_argument("--candidate", type=_candidate, action="append", required=True, metavar="KAPPA=PACKAGE")
    deploy = commands.add_parser("deploy", help="Deployed seed of one configuration")
    deploy.add_argument("--role", required=True, help="Name of the configuration, e.g. e3_wide")
    deploy.add_argument("--package", nargs="+", required=True)
    adaptation = commands.add_parser("e4-adaptation", help="E4 update-1 loop model for adaptation block 2")
    adaptation.add_argument("--package", nargs="+", required=True)
    final = commands.add_parser("e4-final", help="E4 final checkpoint of each arm")
    final.add_argument("--loop", nargs="+", required=True)
    final.add_argument("--control", nargs="+", required=True)
    for command in (kappa, deploy, adaptation, final):
        command.add_argument("--out", required=True, help="Selection record to create (never overwritten)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = parser_for_cli()
    args = parser.parse_args(argv)
    try:
        if args.command == "kappa":
            candidates = dict(args.candidate)
            if len(candidates) != len(args.candidate):
                raise ValueError("Each kappa may be given once")
            selection = select_kappa(candidates)
        elif args.command == "deploy":
            selection = select_deployed([_resolve(path) for path in args.package], args.role)
        elif args.command == "e4-adaptation":
            selection = select_e4_adaptation([_resolve(path) for path in args.package])
        else:
            selection = select_e4_final([_resolve(path) for path in args.loop],
                                        [_resolve(path) for path in args.control])
        out = _resolve(args.out)
        digest = write_record(selection, out)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Selection refused: {exc}\n")
    chosen = selection.get("deployed") or {name: arm["deployed"] for name, arm in selection["arms"].items()}
    print(json.dumps({"record": str(out), "sha256": digest,
                      "selected_kappa": selection.get("selected_kappa"), "deployed": chosen},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
