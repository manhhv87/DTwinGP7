#!/usr/bin/env python
"""
01_collect_dataset.py
─────────────────────
Systematic dataset capture tool using RealSense D455.

Displays a live view; hotkeys change scene metadata, SPACE captures a frame.
Saves: data/raw/{class}_{lighting}_{angle}_{overlap}_{bg}_{ts}_{NNNN}_rgb.png
       + _depth.npy

Hotkeys:
    SPACE      capture 1 image (RGB + depth)
    1/2/3      class = bottle / cup / bolt
    b/m/d      lighting = bright / medium / dim
    n/l/o      overlap  = none / light / medium
    g/u/r      background = gray / blue / brown
    q          quit

Usage:
    python scripts/01_collect_dataset.py
    python scripts/01_collect_dataset.py --output data/raw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir, setup_logging, timestamp  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="data/raw", help="Output directory for saved images")
    parser.add_argument("--color-w", type=int, default=1280)
    parser.add_argument("--color-h", type=int, default=720)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging("collect_dataset", log_file=PROJECT_ROOT / "logs/collect.log")

    try:
        import cv2
        import numpy as np
        import pyrealsense2 as rs
    except ImportError as e:
        log.error("Missing dependency: %s. Install: pip install opencv-python pyrealsense2", e)
        return 1

    out_dir = ensure_dir(PROJECT_ROOT / args.output)

    # ─── Initialize RealSense ───
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.color_w, args.color_h, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    # Depth scale (z16 units → metres). Saved depth must be float METRES to match
    # D455Camera.get_frame() / the in-app capture, which both write metres into the
    # SAME data/raw/*_depth.npy — raw uint16 here would be ~1000× off for any metres-
    # assuming consumer (labeling / pose verification / training).
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    log.info("RealSense started (depth_scale=%.6f m/unit). Output → %s",
             depth_scale, out_dir)

    meta = {"class": "bottle", "lighting": "bright",
            "angle": "0", "overlap": "none", "bg": "gray"}
    counter = 0

    key_map = {
        # Class hotkeys MUST cover every detector class (DEFAULT_CLASS_NAMES =
        # tray/bottle/cup/bolt) — 'tray' (index 0) was previously uncapturable.
        ord("0"): ("class", "tray"),
        ord("1"): ("class", "bottle"), ord("2"): ("class", "cup"),
        ord("3"): ("class", "bolt"),
        ord("b"): ("lighting", "bright"), ord("m"): ("lighting", "medium"),
        ord("d"): ("lighting", "dim"),
        ord("n"): ("overlap", "none"), ord("l"): ("overlap", "light"),
        ord("o"): ("overlap", "medium"),
        ord("g"): ("bg", "gray"), ord("u"): ("bg", "blue"),
        ord("r"): ("bg", "brown"),
    }

    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

            overlay = color.copy()
            txt = (f"{meta['class']} | {meta['lighting']} | overlap:{meta['overlap']} "
                   f"| bg:{meta['bg']} | count:{counter}")
            cv2.putText(overlay, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            cv2.imshow("Capture (SPACE=capture, q=quit)", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                ts = timestamp("%H%M%S")
                name = (f"{meta['class']}_{meta['lighting']}_{meta['angle']}_"
                        f"{meta['overlap']}_{meta['bg']}_{ts}_{counter:04d}")
                cv2.imwrite(str(out_dir / f"{name}_rgb.png"), color)
                np.save(str(out_dir / f"{name}_depth.npy"), depth)
                counter += 1
                log.info("Saved %s (total %d)", name, counter)
            elif key in key_map:
                field, value = key_map[key]
                meta[field] = value
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        log.info("Done. Total %d image(s) captured.", counter)

    return 0


if __name__ == "__main__":
    sys.exit(main())
