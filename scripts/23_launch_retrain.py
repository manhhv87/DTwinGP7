#!/usr/bin/env python
"""Build a verified portable package for Linux training without training here.

Example (freeze counts against the actual capture manifest first):
    python scripts/23_launch_retrain.py --real data/real_dataset \
        --synth data/synth/anchored_k2/dataset --experiment E3 --version 1 \
        --expected-synth-train 3000 --zip

Validation is always real-only. Source images and labels are never changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.package import build_package  # noqa: E402


def parser_for_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", required=True, help="Real dataset root with images/{train,val} and labels/{train,val}")
    parser.add_argument("--synth", nargs="*", default=[], help="Synthetic roots; only their train splits are used")
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--experiment", choices=["E2", "E3", "E4", "E5"], default="E3")
    parser.add_argument("--out", default="data/packages", help="Parent directory; creates a new dsvN directory")
    parser.add_argument("--model", default="yolov8s-seg.pt", help="E4 only: selected parent checkpoint path")
    parser.add_argument("--epochs", type=int, help="Default 130 for E2/E3/E5; explicit frozen budget required for E4")
    seed_options = parser.add_mutually_exclusive_group()
    seed_options.add_argument("--seeds", type=int, nargs="+", help="Default 0..4 for E2/E3 and 0..2 for E4/E5")
    seed_options.add_argument("--seed", type=int, help="Alias for packaging one declared training seed")
    parser.add_argument("--real-train-list", help="Explicit real train allowlist; raw capture names flattened with __ are supported")
    parser.add_argument("--expected-real-train", type=int)
    parser.add_argument("--expected-synth-train", type=int, help="Total across all synthetic source train splits")
    parser.add_argument("--inspect-only", action="store_true", help="Check and print a summary; do not copy or train")
    parser.add_argument("--zip", action="store_true", help="Also create one portable dsvN.zip")
    return parser


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = parser_for_cli()
    args = parser.parse_args(argv)
    destination = _resolve(args.out) / f"dsv{args.version}"
    archive = destination.with_suffix(".zip")
    if args.zip and args.inspect_only:
        parser.error("--zip cannot be combined with --inspect-only")
    if args.zip and archive.exists():
        parser.error(f"Refusing to overwrite existing archive: {archive}")
    try:
        report = build_package(
            _resolve(args.real), [_resolve(value) for value in args.synth], destination,
            experiment=args.experiment, version=args.version,
            model=args.model if args.model == "yolov8s-seg.pt" else _resolve(args.model),
            epochs=args.epochs, seeds=[args.seed] if args.seed is not None else args.seeds,
            real_train_list=_resolve(args.real_train_list) if args.real_train_list else None,
            expected_real_train=args.expected_real_train,
            expected_synth_train=args.expected_synth_train, inspect_only=args.inspect_only,
        )
        if args.inspect_only:
            # The API returns full hashes; terminal output remains compact.
            report["manifest"].pop("images", None)
        if args.zip:
            shutil.make_archive(str(destination), "zip", root_dir=destination.parent, base_dir=destination.name)
            report["archive"] = str(archive)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Package refused: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
