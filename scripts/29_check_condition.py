"""Check the image condition at the start of a session (step 4 of B6).

Run with the table set up as for the session but without parts, the robot jogged
out of the camera's view:

    python scripts/29_check_condition.py --session $B --condition hard

It captures from the cell camera, measures the pick and place tables separately
(src/perception/condition_check.py, thresholds in config/condition_check.yaml) and
writes results/do_sang_<session>.png (the frame), results/do_sang_<session>_kiem.png
(what was measured) and results/do_sang_<session>.txt (the numbers and the verdict).
For --condition hard it prints PASS or FAIL with what to change and exits 0 or 1;
for --condition std it only records the numbers. --image evaluates a saved frame
instead of capturing one.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.condition_check import (  # noqa: E402
    AREAS, judge_hard, load_criteria, measure, overlay, report_lines)


def capture(n_warmup: int = 20, n_frames: int = 5) -> list[np.ndarray]:
    """BGR frames from the cell camera after the auto exposure has settled."""
    from src.perception.camera import D455Camera
    cam = D455Camera()
    try:
        for _ in range(n_warmup):
            cam.get_frame()
        frames = []
        while len(frames) < n_frames:
            bgr, _ = cam.get_frame()
            if bgr is not None:
                frames.append(np.asarray(bgr).copy())
        return frames
    finally:
        cam.stop()


def median_measure(frames: list[np.ndarray], criteria: dict) -> dict:
    """Per-value median over the frames, against single-frame noise."""
    ms = [measure(f, criteria) for f in frames]
    out = {"mean_brightness": float(np.median([m["mean_brightness"] for m in ms]))}
    for area in AREAS:
        out[area] = {k: float(np.median([m[area][k] for m in ms])) for k in ms[0][area]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", required=True, help="session name, e.g. e2kho-1")
    ap.add_argument("--condition", choices=("hard", "std"), required=True)
    ap.add_argument("--image", help="evaluate this saved BGR frame instead of capturing")
    ap.add_argument("--criteria", default=str(ROOT / "config" / "condition_check.yaml"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    args = ap.parse_args(argv)

    criteria = load_criteria(args.criteria)
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Cannot read {args.image}")
            return 2
        frames = [frame]
    else:
        frames = capture()
    m = median_measure(frames, criteria)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"do_sang_{args.session}"
    cv2.imwrite(str(stem) + ".png", frames[-1])
    cv2.imwrite(str(stem) + "_kiem.png", overlay(frames[-1], criteria))

    lines = [f"Session: {args.session}", f"Time: {datetime.now().isoformat(timespec='seconds')}",
             f"Condition: {args.condition}"] + report_lines(m)
    code = 0
    if args.condition == "hard":
        problems = judge_hard(m, criteria)
        if problems:
            lines.append("HARD CONDITION: FAIL")
            lines += [f"  - {p}" for p in problems]
            code = 1
        else:
            lines.append("HARD CONDITION: PASS")
    else:
        lines.append("Standard condition: numbers recorded, no threshold.")
    text = "\n".join(lines)
    print(text)
    Path(str(stem) + ".txt").write_text(text + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
