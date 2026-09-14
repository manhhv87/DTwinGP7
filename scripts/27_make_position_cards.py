#!/usr/bin/env python
"""
27_make_position_cards.py
─────────────────────────
Draw the numbered position cards that go on the table for the paired trial
protocol, at exact print scale.

One card per grid position. During a run, 03_run_experiment.py calls out
"card=14 ... yaw=85 deg"; the operator centres the part on card 14 and turns it
to that angle. The SAME list is replayed under every configuration compared, so
the trials pair up and McNemar's exact test applies — which only holds if the
part really goes back to the same place and angle each time.

Coordinates are read from the .cards.txt that 22_make_pose_lists.py wrote, so
the printed cards cannot drift out of step with the pose list.

Card size is capped by the grid pitch: the columns here sit 51 mm apart, so a
card wider than that would overlap its neighbour. The default 48 mm square fits.

Usage:
    python scripts/27_make_position_cards.py
    python scripts/27_make_position_cards.py --cards-file config/pose_lists/hard_v1.csv.cards.txt

Then: print at 100% scale ("actual size"), NOT "fit to page". Check the 50 mm
line on the sheet with a ruler, cut along the borders, and fix each card with
its centre cross on the coordinate printed on it.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CARD_RE = re.compile(r"card\s+(\d+):\s*x=\s*([-\d.]+)\s*mm\s+y=\s*([-\d.]+)\s*mm")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cards-file", default="config/pose_lists/std_v1.csv.cards.txt",
                   help="The .cards.txt written by 22_make_pose_lists.py.")
    p.add_argument("--card-mm", type=float, default=48.0,
                   help="Card side, mm. Must be smaller than the grid pitch or cards "
                        "overlap. Default 48.")
    p.add_argument("--tick-step-deg", type=float, default=5.0,
                   help="Angle tick spacing. Must match --yaw-step-deg of 22_make_pose_lists.py. "
                        "Default 5.")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--out", default="figures/position_cards.png")
    return p.parse_args()


def read_cards(path: Path) -> list[tuple[int, float, float]]:
    cards = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = CARD_RE.search(line)
        if m:
            cards.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return cards


def pitch_mm(cards: list[tuple[int, float, float]]) -> float:
    """Smallest gap between two card centres, over either axis."""
    xs = sorted({c[1] for c in cards})
    ys = sorted({c[2] for c in cards})
    gaps = [b - a for s in (xs, ys) for a, b in zip(s, s[1:])]
    return min(gaps) if gaps else float("inf")


def draw_card(img, ox, oy, px_mm, card_px, n, x_mm, y_mm, tick_step):
    """One card: cut border, centre cross, angle tick ring, number, coordinates."""
    thin = max(1, int(round(px_mm * 0.15)))
    thick = max(2, int(round(px_mm * 0.35)))
    cx, cy = ox + card_px // 2, oy + card_px // 2
    fs = px_mm / 3.4                                  # cv2 font scale at this dpi

    cv2.rectangle(img, (ox, oy), (ox + card_px, oy + card_px), 128, thin)

    r_out = int(round(19 * px_mm))
    for a in np.arange(-180.0, 180.0, tick_step):
        quarter = abs(a % 45.0) < 1e-6
        long_tick = abs(a % 15.0) < 1e-6
        r0 = int(round((12.0 if quarter else 14.0 if long_tick else 16.5) * px_mm))
        t = math.radians(a)
        cv2.line(img,
                 (int(cx + r0 * math.cos(t)), int(cy - r0 * math.sin(t))),
                 (int(cx + r_out * math.cos(t)), int(cy - r_out * math.sin(t))),
                 0, thick if quarter else thin if not long_tick else max(thin, thick - 1),
                 cv2.LINE_AA)

    arm = int(round(9 * px_mm))
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), 0, thick, cv2.LINE_AA)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), 0, thick, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(2, int(round(0.6 * px_mm))), 0, -1, cv2.LINE_AA)

    # 0 deg points along +x, i.e. away from the robot. Arrowhead so the card
    # cannot be laid down rotated by 180 deg without it being obvious.
    tipx = cx + r_out + int(round(2.2 * px_mm))
    cv2.arrowedLine(img, (cx + arm + int(px_mm), cy), (tipx, cy), 0, thick, cv2.LINE_AA,
                    tipLength=0.3)
    cv2.putText(img, "0", (cx + int(11 * px_mm), cy - int(1.6 * px_mm)),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.36, 0, thin, cv2.LINE_AA)

    num = str(n)
    (tw, th), _ = cv2.getTextSize(num, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
    cv2.putText(img, num, (ox + int(2.5 * px_mm), oy + th + int(2 * px_mm)),
                cv2.FONT_HERSHEY_SIMPLEX, fs, 0, thick, cv2.LINE_AA)

    coord = f"x={x_mm:.1f}  y={y_mm:.1f}"
    (tw, _), _ = cv2.getTextSize(coord, cv2.FONT_HERSHEY_SIMPLEX, fs * 0.26, thin)
    cv2.putText(img, coord, (cx - tw // 2, oy + card_px - int(1.8 * px_mm)),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.26, 0, thin, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    src = Path(args.cards_file)
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    if not src.exists():
        print(f"{src} not found. Run 22_make_pose_lists.py first.", file=sys.stderr)
        return 2

    cards = read_cards(src)
    if not cards:
        print(f"No card lines found in {src}.", file=sys.stderr)
        return 2

    gap = pitch_mm(cards)
    if args.card_mm >= gap:
        print(f"--card-mm {args.card_mm:g} is not smaller than the {gap:.1f} mm grid pitch, "
              f"so neighbouring cards would overlap on the table. Use --card-mm "
              f"{gap - 3:.0f} or less.", file=sys.stderr)
        return 2

    px_mm = args.dpi / 25.4
    to_px = lambda mm: int(round(mm * px_mm))          # noqa: E731
    card_px = to_px(args.card_mm)

    sheet_mm = (210.0, 297.0)                          # A4 portrait
    margin_mm, caption_mm = 8.0, 16.0
    ncol = int((sheet_mm[0] - 2 * margin_mm) // args.card_mm)
    nrow = int((sheet_mm[1] - 2 * margin_mm - caption_mm) // args.card_mm)
    per_sheet = ncol * nrow
    if per_sheet < 1:
        print("Card does not fit on A4 at this size.", file=sys.stderr)
        return 2

    pages = []
    for start in range(0, len(cards), per_sheet):
        sheet = np.full((to_px(sheet_mm[1]), to_px(sheet_mm[0])), 255, np.uint8)
        for k, (n, x, y) in enumerate(cards[start:start + per_sheet]):
            r, c = divmod(k, ncol)
            draw_card(sheet, to_px(margin_mm) + c * card_px,
                      to_px(margin_mm) + r * card_px, px_mm, card_px, n, x, y,
                      args.tick_step_deg)
        # Ruler: the only way to notice the print dialog quietly rescaled the sheet.
        ry = to_px(sheet_mm[1] - caption_mm + 4.0)
        rx = to_px(margin_mm)
        cv2.line(sheet, (rx, ry), (rx + to_px(50.0), ry), 0, max(2, args.dpi // 300))
        for t in (0.0, 25.0, 50.0):
            cv2.line(sheet, (rx + to_px(t), ry - to_px(1.8)),
                     (rx + to_px(t), ry + to_px(1.8)), 0, max(2, args.dpi // 300))
        cv2.putText(sheet, "this line must measure 50 mm on the print",
                    (rx + to_px(54.0), ry + to_px(1.8)), cv2.FONT_HERSHEY_SIMPLEX,
                    args.dpi / 600.0 * 0.9, 0, max(1, args.dpi // 400), cv2.LINE_AA)
        pages.append(sheet)

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    for i, page in enumerate(pages):
        name = out if len(pages) == 1 else out.with_stem(f"{out.stem}_{i + 1}")
        cv2.imwrite(str(name), page)

    from PIL import Image

    pdf = out.with_suffix(".pdf")
    imgs = [Image.fromarray(p).convert("1", dither=Image.NONE) for p in pages]
    imgs[0].save(pdf, save_all=True, append_images=imgs[1:], resolution=float(args.dpi),
                 title=f"Position cards {args.card_mm:g} mm")

    print(f"{len(cards)} cards, {args.card_mm:g} mm square, grid pitch {gap:.1f} mm, "
          f"{ncol} x {nrow} per A4 page, {len(pages)} page(s)")
    print(f"  {pdf}   <- print this one, 100% scale")
    print("Cut along the borders. Lay each card with its centre cross on the coordinate "
          "printed on it, arrow pointing away from the robot (+x). Fix with clear tape "
          "over the whole card so it cannot creep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
