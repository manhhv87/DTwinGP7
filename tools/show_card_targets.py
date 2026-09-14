#!/usr/bin/env python
"""show_card_targets.py — where each position card goes, drawn on the live camera (READ-ONLY).

The card coordinates are in the robot BASE frame, but the cards have to end up on a
table. Jogging the robot to all 20 of them costs a session. After Phase 1 there is a
cheaper way: the hand-eye result already says where the camera sits relative to the
base, and the table plane says how high the table is, so every card centre can be
projected straight into the camera image. Lay a card down, look at the screen, slide
it until its centre cross sits on the marker, tape it.

Reads the camera and the calibration files. Sends nothing to the robot.

Usage:
    python tools/show_card_targets.py
    python tools/show_card_targets.py --cards-file config/pose_lists/hard_v1.csv.cards.txt
    python tools/show_card_targets.py --snapshot results/card_targets.png

Keys: s = save a snapshot, q or Esc = quit.

Accuracy comes from the calibration, so check one card by hand before trusting the
other nineteen: jog the robot to that card's coordinate with the teach pendant and
see whether the tool centre lands on the cross you taped down.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.depth_modes import TablePlane, project_points  # noqa: E402

CARD_RE = re.compile(r"card\s+(\d+):\s*x=\s*([-\d.]+)\s*mm\s+y=\s*([-\d.]+)\s*mm")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cards-file", default="config/pose_lists/std_v1.csv.cards.txt")
    p.add_argument("--calib-dir", default="config/calibration")
    p.add_argument("--snapshot", default=None,
                   help="Save one frame here and exit, instead of opening a window.")
    return p.parse_args()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def draw_targets(img, intr, T_BC, plane, cards):
    """Cross, ring and number at every card centre. Returns how many landed in frame."""
    h, w = img.shape[:2]
    pts = np.array([[x, y, float(plane.z_at(x, y))] for _, x, y in cards])
    uv = project_points(intr, T_BC, pts)
    if uv is None:
        cv2.putText(img, "cards project behind the camera - wrong calibration file?",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return 0

    inside = 0
    for (n, x, y), (u, v) in zip(cards, uv):
        u, v = int(round(u)), int(round(v))
        if not (0 <= u < w and 0 <= v < h):
            continue
        inside += 1
        cv2.circle(img, (u, v), 16, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.line(img, (u - 26, v), (u + 26, v), (0, 165, 255), 1, cv2.LINE_AA)
        cv2.line(img, (u, v - 26), (u, v + 26), (0, 165, 255), 1, cv2.LINE_AA)
        cv2.circle(img, (u, v), 2, (0, 165, 255), -1, cv2.LINE_AA)
        cv2.putText(img, str(n), (u + 20, v - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, str(n), (u + 20, v - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 165, 255), 2, cv2.LINE_AA)
    return inside


def main() -> int:
    args = parse_args()

    cards_path = resolve(args.cards_file)
    if not cards_path.exists():
        print(f"{cards_path} not found. Run 22_make_pose_lists.py first.", file=sys.stderr)
        return 2
    cards = [(int(m.group(1)), float(m.group(2)), float(m.group(3)))
             for line in cards_path.read_text(encoding="utf-8").splitlines()
             if (m := CARD_RE.search(line))]
    if not cards:
        print(f"No card lines in {cards_path}.", file=sys.stderr)
        return 2

    calib = resolve(args.calib_dir)
    t_path, plane_path = calib / "T_base_camera.npy", calib / "table_plane.json"
    for f in (t_path, plane_path):
        if not f.exists():
            print(f"{f} not found. Run Phase 1 (02_run_calibration.py) first: without the "
                  f"hand-eye result and the table plane there is nothing to project with.",
                  file=sys.stderr)
            return 2
    T_BC = np.load(t_path)
    plane = TablePlane.from_dict(json.loads(plane_path.read_text(encoding="utf-8")))

    from src.perception.camera import D455Camera

    cam = D455Camera()
    try:
        intr = cam.intrinsics
        if args.snapshot:
            color, _ = cam.get_frame()
            if color is None:
                print("No frame from the camera.", file=sys.stderr)
                return 1
            n_in = draw_targets(color, intr, T_BC, plane, cards)
            out = resolve(args.snapshot)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), color)
            print(f"{n_in}/{len(cards)} card targets in frame → {out}")
            return 0

        print(f"{len(cards)} card targets. s = snapshot, q = quit.")
        while True:
            color, _ = cam.get_frame()
            if color is None:
                continue
            n_in = draw_targets(color, intr, T_BC, plane, cards)
            cv2.putText(color, f"{n_in}/{len(cards)} targets in frame",
                        (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.imshow("card targets (read-only)", color)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                out = resolve("results/card_targets.png")
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), color)
                print(f"saved {out}")
    finally:
        cam.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
