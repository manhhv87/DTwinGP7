"""Check the image condition at the start of a session (step 4 of B6).

Run with the table set up as for the session but without parts, the robot jogged
out of the camera's view:

    python scripts/29_check_condition.py --session $B --condition hard

It captures from the cell camera, measures the upper and lower parts of the table
separately as they appear in the image (src/perception/condition_check.py,
thresholds in config/condition_check.yaml) and writes results/do_sang_<session>.png
(the frame), results/do_sang_<session>_kiem.png
(what was measured) and results/do_sang_<session>.txt (the numbers and the verdict).
For --condition hard it prints PASS or FAIL with what to change and exits 0 or 1;
for --condition std it only records the numbers. --image evaluates a saved frame
instead of capturing one.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception.condition_check import (  # noqa: E402
    judge_hard, load_criteria, measure, overlay, report_lines)


def capture(warmup_s: float = 5.0, n_frames: int = 10) -> np.ndarray:
    """One BGR frame from the cell camera: the mean of n_frames, taken once the auto
    exposure has settled, which in dim light takes seconds after the camera starts."""
    from src.perception.camera import D455Camera
    cam = D455Camera()
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < warmup_s:
            cam.get_frame()
        frames: list[np.ndarray] = []
        while len(frames) < n_frames:
            bgr, _ = cam.get_frame()
            if bgr is not None:
                frames.append(np.asarray(bgr, dtype=np.float32))
        return np.clip(np.mean(frames, axis=0) + 0.5, 0, 255).astype(np.uint8)
    finally:
        cam.stop()


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
    else:
        frame = capture()
    m = measure(frame, criteria)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"do_sang_{args.session}"
    cv2.imwrite(str(stem) + ".png", frame)
    cv2.imwrite(str(stem) + "_kiem.png", overlay(frame, criteria))

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
