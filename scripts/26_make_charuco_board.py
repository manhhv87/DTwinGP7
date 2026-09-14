#!/usr/bin/env python
"""
26_make_charuco_board.py
────────────────────────
Draw the ChArUco board used by the hand-eye calibration, at exact print scale.

Generate the board with THIS script rather than downloading one: the board is
built by the same OpenCV that will detect it, so the marker layout matches. Boards
made by OpenCV before 4.6, and most board generators on the web, use the legacy
layout, which prints to something that looks identical but is read wrong, or not
read at all, by the detector in src/calibration.

The sheet carries its own parameters and a 100 mm ruler, because the number that
matters is not what you asked for but what came out of the printer.

Usage:
    python scripts/26_make_charuco_board.py
    python scripts/26_make_charuco_board.py --squares 7 5 --square-mm 45 --marker-mm 34
    python scripts/26_make_charuco_board.py --paper a4

Then: print at 100% scale ("actual size"), NOT "fit to page". Measure a square and
the black part of one marker with calipers, and pass those measured numbers to
02_run_calibration.py — a printer that quietly scales by 3% moves the camera
position by 3% with no error message anywhere.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PAPER_MM = {"a4": (297.0, 210.0), "a3": (420.0, 297.0)}   # landscape


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--squares", type=int, nargs=2, default=[7, 5], metavar=("COLS", "ROWS"),
                   help="Squares across and down. Default 7 5.")
    p.add_argument("--square-mm", type=float, default=40.0,
                   help="Side of one chessboard square, mm. Default 40.")
    p.add_argument("--marker-mm", type=float, default=30.0,
                   help="Side of the black ArUco square inside a white square, mm. "
                        "Must be smaller than --square-mm; 0.75 of it is usual. Default 30.")
    p.add_argument("--dict", default="DICT_4X4_50", help="ArUco dictionary. Default DICT_4X4_50.")
    p.add_argument("--dpi", type=int, default=300, help="Print resolution. Default 300.")
    p.add_argument("--paper", choices=sorted(PAPER_MM), default="a3",
                   help="Sheet the board must fit on, landscape. Default a3.")
    p.add_argument("--out", default="figures/charuco_board.png")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cols, rows = args.squares
    if args.marker_mm >= args.square_mm:
        print("--marker-mm must be smaller than --square-mm: the marker is printed "
              "inside a white square.", file=sys.stderr)
        return 2

    pattern_mm = (cols * args.square_mm, rows * args.square_mm)
    sheet_mm = PAPER_MM[args.paper]
    margin_mm = 12.0
    caption_mm = 22.0
    need = (pattern_mm[0] + 2 * margin_mm, pattern_mm[1] + 2 * margin_mm + caption_mm)
    if need[0] > sheet_mm[0] or need[1] > sheet_mm[1]:
        print(f"The board needs {need[0]:.0f} x {need[1]:.0f} mm including margins, which does "
              f"not fit on {args.paper.upper()} landscape ({sheet_mm[0]:.0f} x {sheet_mm[1]:.0f} "
              f"mm). Use --paper a3, fewer squares, or a smaller --square-mm.", file=sys.stderr)
        return 2

    px_per_mm = args.dpi / 25.4
    to_px = lambda mm: int(round(mm * px_per_mm))                       # noqa: E731

    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    except AttributeError:
        print(f"Unknown ArUco dictionary '{args.dict}'.", file=sys.stderr)
        return 2
    # Metres here only to mirror the estimator's units; the drawing is in pixels.
    board = cv2.aruco.CharucoBoard((cols, rows), args.square_mm / 1000.0,
                                   args.marker_mm / 1000.0, aruco_dict)
    # One square must be a whole number of pixels: OpenCV slices the image square by
    # square, and a fractional square overruns the last row on the right or bottom.
    square_px = to_px(args.square_mm)
    pattern = board.generateImage((cols * square_px, rows * square_px))

    sheet = np.full((to_px(sheet_mm[1]), to_px(sheet_mm[0])), 255, np.uint8)
    x0, y0 = to_px(margin_mm), to_px(margin_mm)
    sheet[y0:y0 + pattern.shape[0], x0:x0 + pattern.shape[1]] = pattern

    # Caption + a ruler to catch a printer that rescaled the sheet.
    text_y = y0 + pattern.shape[0] + to_px(9.0)
    cv2.putText(sheet, f"ChArUco {cols}x{rows} squares | square {args.square_mm:g} mm | "
                       f"marker {args.marker_mm:g} mm | {args.dict}",
                (x0, text_y), cv2.FONT_HERSHEY_SIMPLEX, args.dpi / 300.0 * 0.6, 0,
                max(1, args.dpi // 300), cv2.LINE_AA)
    ruler_y = text_y + to_px(7.0)
    cv2.line(sheet, (x0, ruler_y), (x0 + to_px(100.0), ruler_y), 0, max(2, args.dpi // 150))
    for t in (0.0, 50.0, 100.0):
        cv2.line(sheet, (x0 + to_px(t), ruler_y - to_px(2.0)),
                 (x0 + to_px(t), ruler_y + to_px(2.0)), 0, max(2, args.dpi // 150))
    cv2.putText(sheet, "this line must measure 100 mm on the print",
                (x0 + to_px(104.0), ruler_y + to_px(2.0)), cv2.FONT_HERSHEY_SIMPLEX,
                args.dpi / 300.0 * 0.5, 0, max(1, args.dpi // 300), cv2.LINE_AA)

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)

    # PDF as well: it carries the page size, so the print dialog has no reason to
    # rescale, which is the failure this whole sheet is designed to catch.
    from PIL import Image

    pdf = out.with_suffix(".pdf")
    # 1-bit, no dithering: the pattern is already pure black and white, and a bilevel
    # page prints crisper and compresses to a fraction of the size.
    Image.fromarray(sheet).convert("1", dither=Image.NONE).save(
        pdf, resolution=float(args.dpi), title=f"ChArUco {cols}x{rows} {args.square_mm:g}mm")

    print(f"Board pattern {pattern_mm[0]:.0f} x {pattern_mm[1]:.0f} mm on "
          f"{args.paper.upper()} landscape, {args.dpi} dpi")
    print(f"  {out}")
    print(f"  {pdf}   <- print this one")
    print(f"Squares {cols} x {rows}, square {args.square_mm:g} mm, marker {args.marker_mm:g} mm, "
          f"{args.dict}")
    print("Print at 100% scale (not 'fit to page'), glue it flat onto a rigid plate, then "
          "measure a square and a marker with calipers and calibrate with THOSE numbers:")
    print(f"  python scripts/02_run_calibration.py --hse-ip 192.168.1.100 --squares {cols} {rows} "
          f"--square-mm <measured> --marker-mm <measured> --dict {args.dict} "
          f"--method park --bootstrap 200")
    return 0


if __name__ == "__main__":
    sys.exit(main())
