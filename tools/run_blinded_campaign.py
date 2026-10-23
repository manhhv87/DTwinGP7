#!/usr/bin/env python
"""run_blinded_campaign.py: run several configurations blinded and interleaved.

Two problems this solves, both of which a reviewer will raise.

Unblinding. The person placing the parts is the same person who scores the place
tolerance by eye, and in a hand-run campaign they also type the command, so they
know which configuration is on trial. That is an unblinded assessor on the primary
endpoint. Here the operator sees only an arm code (A, B, C ...) and the placement
prompts; which arm is which lives in a sealed key file that nobody opens until the
analysis.

Confounding with time. Run each configuration as one uninterrupted stretch and every
drift over the session -- calibration creep, the operator getting quicker and then
tired, room light through the afternoon -- lands in the discordant cells of the paired
test looking exactly like a configuration effect. Here the arms are interleaved in
short blocks whose order is drawn per round.

The seed. Which code is which arm, and the block order, follow from one seed. A seed
that anyone can read (a guide, the command line, the shell history) lets them rerun
this script and recompute the key: session 1 of E2 printed S=1 in the operator guide,
and the mapping was reproduced from it; the S=2 printed for session 2 gives the same
mapping again. Without --seed a secret seed is drawn from the operating system
(secrets) and written only to the sealed key file, never to the console. A live
campaign refuses --seed unless --public-seed says it is knowingly not blinded (a
rehearsal); --dry-run accepts it.

The placement check (03 --confirm-each-trial with a pose list) compares detections
with the call. It stays on only when the arms differ in --depth-mode alone, so that
every arm sees the same detections and the same flags; otherwise (the models of E3
to E5, or any other difference) --no-placement-check is added for every arm. The
decision is printed and written to the key.

Usage:
    python tools/run_blinded_campaign.py \\
        --arm rgbd   "--depth-mode rgbd" \\
        --arm plane  "--depth-mode plane" \\
        --arm fusion "--depth-mode fusion" \\
        --pose-list config/pose_lists/std_v2.csv --trials 200 --block 25 \\
        --session 2026-09-20-morning --operator AN \\
        --common "--mode real --ik-source yrc --tool-no 1 --confirm-each-trial --no-viewport-mirror"

Dry run first (prints the schedule, starts nothing; it draws its own secret seed, so
the real run will show a different order):
    ... --dry-run

The key file is written before the first block, so an interrupted campaign is still
decodable. Do not open it until the runs are finished and scored.
"""
from __future__ import annotations

import argparse
import json
import secrets
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
CODES = "ABCDEFGHJKLMNPQRSTUVWXYZ"          # no I, no O


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", nargs=2, action="append", metavar=("NAME", "ARGS"),
                   required=True,
                   help="Repeatable. NAME is the true configuration name, kept out of "
                        "sight; ARGS are the extra flags for it, quoted.")
    p.add_argument("--common", default="",
                   help="Flags shared by every arm, quoted.")
    p.add_argument("--pose-list", required=True)
    p.add_argument("--trials", type=int, required=True,
                   help="Trials per arm. Must divide evenly by --block.")
    p.add_argument("--first-pose", type=int, default=0,
                   help="First pose-list row (0-based) of this sitting. A campaign too long "
                        "for one sitting is split into sittings that each cover their own "
                        "rows with every arm: --first-pose 0 --trials 100, then "
                        "--first-pose 100 --trials 100, each with its own --session. "
                        "Must be a multiple of --block. Default 0.")
    p.add_argument("--block", type=int, default=25,
                   help="Trials per block. Short blocks mix the arms through the "
                        "session; long ones cost less restaging. Default 25.")
    p.add_argument("--session", required=True, help="Session tag, e.g. 2026-09-20-morning")
    p.add_argument("--operator", default="", help="Who is placing the parts")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed for the code assignment and the block order. Anyone who knows "
                        "it can recompute which code is which arm, so a live campaign "
                        "refuses it unless --public-seed is given too; --dry-run accepts it. "
                        "Without it a secret seed is drawn from the operating system and "
                        "written only to the sealed key file.")
    p.add_argument("--public-seed", action="store_true",
                   help="Let a live campaign run with --seed. The code-to-arm mapping is "
                        "then public and the campaign is not blinded: for rehearsals and "
                        "tests only. The key records seed_source 'cli-public'.")
    p.add_argument("--key-out", default="results/blinding_keys",
                   help="Where the sealed key file is written")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the schedule and stop. Nothing is run, no key written.")
    return p.parse_args()


def _split_options(args: str) -> list[str]:
    """shlex tokens of an argument string, each --opt=value split into two."""
    out: list[str] = []
    for tok in shlex.split(args):
        out.extend(tok.split("=", 1) if tok.startswith("--") and "=" in tok else [tok])
    return out


def _without_depth_mode(tokens: list[str]) -> list[str]:
    """The tokens with every --depth-mode and its value left out."""
    out: list[str] = []
    skip = False
    for tok in tokens:
        if skip:
            skip = False
        elif tok == "--depth-mode":
            skip = True
        else:
            out.append(tok)
    return out


ARMS_DIFFER = "the arms differ in more than --depth-mode"


def placement_check_decision(arms, common: str) -> tuple[bool, str]:
    """(on, reason) for 03's placement check over the whole campaign.

    On only with --confirm-each-trial in --common, without --no-placement-check
    there, and with arms whose arguments are the same once --depth-mode is left
    out (in E2 the arms share the detector, so they see the same detections).
    With reason ARMS_DIFFER the caller adds --no-placement-check for every arm.
    """
    common_tokens = _split_options(common)
    if "--no-placement-check" in common_tokens:
        return False, "--no-placement-check in --common"
    if "--confirm-each-trial" not in common_tokens:
        return False, "no --confirm-each-trial in --common"
    if len(arms) == 1:
        return True, "one arm"
    rest = {tuple(_without_depth_mode(_split_options(extra))) for _, extra in arms}
    if len(rest) > 1:
        return False, ARMS_DIFFER
    return True, "the arms differ in --depth-mode only"


def main() -> int:
    # A console or a pipe that cannot print a character must not stop a run:
    # piped through Tee-Object, Windows hands Python a cp1252 stream.
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(errors="replace")
    args = parse_args()
    arms = args.arm
    if args.seed is not None and not args.dry_run and not args.public_seed:
        print("--seed fixes which code is which arm, and anyone who knows it can "
              "recompute the mapping with this script, so a blinded campaign cannot run "
              "with it. Leave --seed out (a secret seed is drawn and written only to the "
              "sealed key file), or add --public-seed for a rehearsal that is not "
              "blinded. Nothing was started and no key was written.", file=sys.stderr)
        return 2
    for name, extra in arms:
        if "--no-placement-check" in _split_options(extra):
            print(f"--no-placement-check in the arguments of arm {name} would switch the "
                  f"placement check off for that arm alone, and the flags would tell "
                  f"the arms apart. Put it in --common to switch it off for every arm.",
                  file=sys.stderr)
            return 2
    if len(arms) > len(CODES):
        print(f"At most {len(CODES)} arms.", file=sys.stderr)
        return 2
    if args.trials % args.block:
        print(f"--trials {args.trials} is not a whole number of --block {args.block} "
              f"blocks; the arms would not get equal coverage of the pose list.",
              file=sys.stderr)
        return 2
    if args.first_pose < 0 or args.first_pose % args.block:
        print(f"--first-pose {args.first_pose} must be a non-negative multiple of --block "
              f"{args.block}, so sittings line up with whole blocks.", file=sys.stderr)
        return 2
    last = args.first_pose + args.trials
    pose_list = Path(args.pose_list)
    if not pose_list.is_absolute():
        pose_list = PROJECT_ROOT / pose_list
    if pose_list.exists():
        with pose_list.open(encoding="utf-8") as f:
            n_rows = sum(1 for line in f if line.strip()) - 1
        if last > n_rows:
            print(f"Rows {args.first_pose}:{last} run past the {n_rows} poses of "
                  f"{args.pose_list}.", file=sys.stderr)
            return 2

    import random

    from src.utils.provenance import git_provenance

    seed_source = "cli-public" if args.seed is not None else "secrets"
    seed = args.seed if args.seed is not None else secrets.randbits(64)
    rng = random.Random(seed)
    if args.seed is not None and not args.dry_run:
        print("WARNING: --public-seed: the code-to-arm mapping follows from the seed on "
              "the command line; this campaign is not blinded.", file=sys.stderr)
    code = git_provenance(PROJECT_ROOT)
    if code.get("dirty"):
        print(f"WARNING: the checkout has uncommitted changes in "
              f"{len(code.get('changed', []))} file(s); the key records the SHA-256 of "
              f"their diff and every block saves the diff next to its metadata.",
              file=sys.stderr)

    # Codes are assigned at random, so "A" is not the first arm on the command line.
    codes = list(CODES[:len(arms)])
    rng.shuffle(codes)
    code_of = {name: codes[i] for i, (name, _) in enumerate(arms)}
    args_of = {name: extra for name, extra in arms}

    # Schedule: one round per pose-list block, arm order drawn fresh each round, so
    # no arm is systematically first (first-of-session and last-of-session differ).
    schedule = []
    for lo in range(args.first_pose, last, args.block):
        order = [name for name, _ in arms]
        rng.shuffle(order)
        for name in order:
            schedule.append((name, lo, lo + args.block))

    key = {
        "session": args.session,
        "operator": args.operator,
        "seed": seed,
        "seed_source": seed_source,
        "created": datetime.now().isoformat(timespec="seconds"),
        "code": code,
        "pose_list": args.pose_list,
        "block": args.block,
        "trials_per_arm": args.trials,
        "first_pose": args.first_pose,
        "arm_of_code": {code_of[n]: n for n, _ in arms},
        "schedule": [{"block_id": f"{args.session}-b{i + 1:03d}",
                      "code": code_of[n], "arm": n, "slice": f"{lo}:{hi}"}
                     for i, (n, lo, hi) in enumerate(schedule)],
    }

    # The placement check of 03 compares detections with the call. Arms that differ
    # in anything but the depth mode (the model of E3 to E5, a config) may detect
    # differently, and then the flags would differ by arm: the check is switched off
    # for every arm of such a campaign.
    common = shlex.split(args.common)
    check_on, why = placement_check_decision(arms, args.common)
    if why == ARMS_DIFFER:
        common.append("--no-placement-check")
    key["placement_check"] = {"on": check_on, "reason": why}
    print(f"Placement check: {'on' if check_on else 'off'} for every arm ({why}).")

    print(f"{len(arms)} arm(s), {len(schedule)} block(s) of {args.block} trials.")
    print("Order of arm CODES (the true names are sealed):")
    for i, (n, lo, hi) in enumerate(schedule):
        print(f"  block {i + 1:3d}  arm {code_of[n]}  poses {lo}:{hi}")

    if args.dry_run:
        print("\nDry run: nothing started, no key written.")
        return 0

    key_path = Path(args.key_out)
    if not key_path.is_absolute():
        key_path = PROJECT_ROOT / key_path
    key_path.mkdir(parents=True, exist_ok=True)
    key_file = key_path / f"key_{args.session}.json"
    if key_file.exists():
        print(f"{key_file} already exists. Refusing to overwrite a key: without it the "
              f"runs it describes cannot be decoded.", file=sys.stderr)
        return 2
    key_file.write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"\nSealed key: {key_file}")
    print("Do not open it until every block is run and scored.\n")

    for i, (name, lo, hi) in enumerate(schedule):
        block_id = f"{args.session}-b{i + 1:03d}"
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "03_run_experiment.py"),
               "--pose-list", args.pose_list,
               "--pose-slice", f"{lo}:{hi}",
               "--trials", str(args.block),
               "--session-id", args.session,
               "--block-id", block_id,
               "--operator-id", args.operator,
               "--blind-label", code_of[name],
               *common, *shlex.split(args_of[name])]
        print(f"\n{'=' * 60}\n  Block {i + 1}/{len(schedule)}, arm {code_of[name]}"
              f"\n{'=' * 60}", flush=True)
        # stdin/stdout are inherited: the operator answers the per-trial prompt.
        rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT))
        if rc != 0:
            print(f"\nBlock {i + 1} exited with {rc}. Campaign stopped here; the key "
                  f"already on disk covers the blocks that did run.", file=sys.stderr)
            return rc

    print(f"\nAll {len(schedule)} block(s) done. Decode with {key_file} when scoring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
