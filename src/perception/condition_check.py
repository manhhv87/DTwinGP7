"""Whether the cell looks like the image condition a session is meant to run in.

The hard condition of the experiments is the dataset family novelbg_dim: a green
sheet covering the whole table under dimmed light. Mean image brightness alone does
not establish it: on 29 Oct 2026 a near-black sheet over the upper part of the table,
with the lower part left bare, read 25/255, inside the old 15 to 35 window, because
the bare part and the white cards lifted the mean. The check therefore measures the
upper and lower parts of the one table separately, as they appear in the image, from
one camera frame of the empty table, with the card block left out.

The measures are taken on square blocks, not on pixels. In this dim light a pixel's
hue is mostly sensor noise, and a pixel count near its threshold changed from run to
run (2 Nov 2026), whereas a block mean averages over about 1000 pixels. Each part
gives the median green share of the colour, G/(R+G+B), the median brightness, and
the share of its blocks as bright as bare table, judged against the white cards in
the same frame so that the exposure does not enter. The thresholds and the ranges
they come from are in config/condition_check.yaml. Frames are BGR, as the camera
stream and cv2.imread deliver them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

AREAS = ("upper_area", "lower_area")
LABELS = {"upper_area": "TABLE, UPPER PART", "lower_area": "TABLE, LOWER PART"}


def load_criteria(path: str | Path) -> dict[str, Any]:
    """The areas and hard-condition thresholds from the YAML file."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _blocks(bgr: np.ndarray, box, px: int, cut=None):
    """Mean BGR of the px-square blocks of an area, which of them to keep (those
    clear of cut), and their top-left corners."""
    x0, x1, y0, y1 = (int(v) for v in box)
    nx, ny = (x1 - x0) // px, (y1 - y0) // px
    crop = bgr[y0:y0 + ny * px, x0:x0 + nx * px].astype(np.float32)
    means = cv2.resize(crop, (nx, ny), interpolation=cv2.INTER_AREA)
    xs, ys = x0 + px * np.arange(nx), y0 + px * np.arange(ny)
    keep = np.ones((ny, nx), bool)
    if cut is not None:
        cx0, cx1, cy0, cy1 = (int(v) for v in cut)
        keep &= ~(((xs + px > cx0) & (xs < cx1))[None, :]
                  & ((ys + px > cy0) & (ys < cy1))[:, None])
    return means, keep, xs, ys


def _luminance(means: np.ndarray) -> np.ndarray:
    return 0.114 * means[..., 0] + 0.587 * means[..., 1] + 0.299 * means[..., 2]


def _green_share(means: np.ndarray) -> np.ndarray:
    return means[..., 1] / (means.sum(axis=-1) + 1e-6)


def _cut(area: str, criteria: dict[str, Any]):
    return criteria.get("cards") if area == "upper_area" else None


def _card_brightness(bgr: np.ndarray, criteria: dict[str, Any]) -> float:
    means, _, _, _ = _blocks(bgr, criteria["cards"], int(criteria["hard"]["block_px"]))
    return float(np.median(_luminance(means)))


def measure(bgr: np.ndarray, criteria: dict[str, Any]) -> dict[str, Any]:
    """Mean brightness of the frame, brightness of the cards and, per part of the
    table, median green share, median brightness and share of bare blocks."""
    hard = criteria["hard"]
    px = int(hard["block_px"])
    cards = _card_brightness(bgr, criteria)
    bare_above = float(hard["bare_relative_brightness"]) * cards
    out: dict[str, Any] = {"mean_brightness": float(bgr.astype(float).mean()),
                           "card_brightness": cards}
    for area in AREAS:
        means, keep, _, _ = _blocks(bgr, criteria[area], px, _cut(area, criteria))
        lum = _luminance(means)[keep]
        out[area] = {"green_share": float(np.median(_green_share(means)[keep])),
                     "brightness": float(np.median(lum)),
                     "bare_share": float((lum > bare_above).mean())}
    return out


def judge_hard(m: dict[str, Any], criteria: dict[str, Any]) -> list[str]:
    """Why the frame is not the hard condition; empty when it is."""
    hard = criteria["hard"]
    lo, hi = (float(v) for v in hard["luminance"])
    problems: list[str] = []
    for area in AREAS:
        a, name = m[area], LABELS[area]
        need = float(hard["min_green_share"][area])
        if a["bare_share"] > float(hard["max_bare_share"]):
            problems.append(f"{name}: {100 * a['bare_share']:.0f}% of it is as bright as the "
                            "white cards: bare table, cover it with the green sheet")
        if a["green_share"] < need:
            problems.append(f"{name}: not green enough (green share {a['green_share']:.3f}, "
                            f"need {need:.3f}): check the sheet")
        if a["brightness"] < lo:
            problems.append(f"{name}: too dark ({a['brightness']:.1f}, allowed {lo:.0f} to "
                            f"{hi:.0f}): raise the light a little")
        elif a["brightness"] > hi:
            problems.append(f"{name}: too bright ({a['brightness']:.1f}, allowed {lo:.0f} to "
                            f"{hi:.0f}): dim the light")
    return problems


def report_lines(m: dict[str, Any]) -> list[str]:
    """The measured values, one line per part of the table."""
    lines = [f"Brightness, whole frame: {m['mean_brightness']:.1f}/255, "
             f"cards: {m['card_brightness']:.1f}"]
    for area in AREAS:
        a = m[area]
        lines.append(f"{LABELS[area]:17}: green share {a['green_share']:.3f}  brightness "
                     f"{a['brightness']:.1f}  bare {100 * a['bare_share']:.0f}%")
    return lines


def overlay(bgr: np.ndarray, criteria: dict[str, Any]) -> np.ndarray:
    """The frame with each block tinted, green when it reaches the part's green share,
    red when it is as bright as bare table, and the card block crossed out, for the
    operator to see what was measured."""
    hard = criteria["hard"]
    px = int(hard["block_px"])
    bare_above = float(hard["bare_relative_brightness"]) * _card_brightness(bgr, criteria)
    tint = bgr.copy()
    for area in AREAS:
        need = float(hard["min_green_share"][area])
        means, keep, xs, ys = _blocks(bgr, criteria[area], px, _cut(area, criteria))
        lum, share = _luminance(means), _green_share(means)
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                if not keep[j, i]:
                    continue
                if lum[j, i] > bare_above:
                    tint[y:y + px, x:x + px] = (0, 0, 255)
                elif share[j, i] >= need:
                    tint[y:y + px, x:x + px] = (0, 255, 0)
    out = cv2.addWeighted(tint, 0.35, bgr, 0.65, 0)
    for area, colour in (("upper_area", (255, 200, 0)), ("lower_area", (0, 200, 255))):
        x0, x1, y0, y1 = (int(v) for v in criteria[area])
        cv2.rectangle(out, (x0, y0), (x1, y1), colour, 2)
        cv2.putText(out, LABELS[area], (x0 + 6, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    colour, 2)
    if criteria.get("cards") is not None:
        x0, x1, y0, y1 = (int(v) for v in criteria["cards"])
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 1)
        cv2.line(out, (x0, y0), (x1, y1), (0, 0, 255), 1)
        cv2.line(out, (x0, y1), (x1, y0), (0, 0, 255), 1)
    return out
