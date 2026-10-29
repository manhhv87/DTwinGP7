"""Whether the cell looks like the image condition a session is meant to run in.

The hard condition of the experiments is the dataset family novelbg_dim: a green
sheet covering the whole table under dimmed light. Mean image brightness alone does
not establish it: on 29 Oct 2026 a near-black sheet over the upper part of the table,
with the lower part left bare, read 25/255, inside the old 15 to 35 window, because
the bare part and the white cards lifted the mean. The check therefore measures the
upper and lower parts of the one table separately, as they appear in the image, from
one camera frame of the empty table, with the card block
left out because the reference images predate the cards: the share of green pixels
(hue, saturation and value bounds), the median green share of the colour,
G/(R+G+B), and the median brightness. The thresholds and the ranges they come from
are in config/condition_check.yaml. Frames are BGR, as the camera stream and
cv2.imread deliver them.
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


def _area_mask(shape, box, cut=None) -> np.ndarray:
    h, w = shape[:2]
    m = np.zeros((h, w), bool)
    x0, x1, y0, y1 = (int(v) for v in box)
    m[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = True
    if cut is not None:
        x0, x1, y0, y1 = (int(v) for v in cut)
        m[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = False
    return m


def green_pixels(bgr: np.ndarray, hard: dict[str, Any]) -> np.ndarray:
    """Pixels counted as the green sheet, on a lightly blurred frame."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (7, 7), 0), cv2.COLOR_BGR2HSV)
    lo, hi = (float(v) / 2.0 for v in hard["green_hue_deg"])   # OpenCV hue is 0..180
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h >= lo) & (h <= hi) & (s >= int(hard["green_min_saturation"]))
            & (v >= int(hard["green_min_value"])))


def measure(bgr: np.ndarray, criteria: dict[str, Any]) -> dict[str, Any]:
    """Mean brightness of the frame and, per part of the table, green cover, green
    share and median brightness."""
    f = cv2.GaussianBlur(bgr, (7, 7), 0).astype(float)
    total = f.sum(axis=2) + 1e-6
    chroma = f[..., 1] / total
    lum = 0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]
    green = green_pixels(bgr, criteria["hard"])
    out: dict[str, Any] = {"mean_brightness": float(bgr.astype(float).mean())}
    for area in AREAS:
        m = _area_mask(bgr.shape, criteria[area],
                       criteria.get("cards") if area == "upper_area" else None)
        out[area] = {"green_cover": float(green[m].mean()),
                     "green_chroma": float(np.median(chroma[m])),
                     "brightness": float(np.median(lum[m]))}
    return out


def judge_hard(m: dict[str, Any], criteria: dict[str, Any]) -> list[str]:
    """Why the frame is not the hard condition; empty when it is."""
    hard = criteria["hard"]
    lo, hi = (float(v) for v in hard["luminance"])
    problems: list[str] = []
    for area in AREAS:
        a, name = m[area], LABELS[area]
        if (a["green_cover"] < float(hard["min_green_cover"])
                or a["green_chroma"] < float(hard["min_green_chroma"])):
            problems.append(
                f"{name}: not covered by the green sheet (green {100 * a['green_cover']:.0f}%, "
                f"need {100 * float(hard['min_green_cover']):.0f}%; green share "
                f"{a['green_chroma']:.2f}, need {float(hard['min_green_chroma']):.2f})")
        if a["brightness"] < lo:
            problems.append(f"{name}: too dark ({a['brightness']:.1f}, allowed {lo:.0f} to "
                            f"{hi:.0f}): raise the light a little")
        elif a["brightness"] > hi:
            problems.append(f"{name}: too bright ({a['brightness']:.1f}, allowed {lo:.0f} to "
                            f"{hi:.0f}): dim the light")
    return problems


def report_lines(m: dict[str, Any]) -> list[str]:
    """The measured values, one line per part of the table."""
    lines = [f"Brightness, whole frame: {m['mean_brightness']:.1f}/255"]
    for area in AREAS:
        a = m[area]
        lines.append(f"{LABELS[area]:17}: green {100 * a['green_cover']:3.0f}%  green share "
                     f"{a['green_chroma']:.2f}  brightness {a['brightness']:.1f}")
    return lines


def overlay(bgr: np.ndarray, criteria: dict[str, Any]) -> np.ndarray:
    """The frame with the two parts of the table framed, the card block crossed out and the
    pixels counted as green tinted, for the operator to see what was measured."""
    out = bgr.copy()
    green = green_pixels(bgr, criteria["hard"])
    tint = out.copy()
    tint[green] = (0, 255, 0)
    out = cv2.addWeighted(tint, 0.35, out, 0.65, 0)
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
