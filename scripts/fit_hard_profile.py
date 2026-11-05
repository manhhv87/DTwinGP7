#!/usr/bin/env python
"""
fit_hard_profile.py
───────────────────
Fit the hard-condition scene profile of the synthetic generator (hard_profile,
sensor and background sections of config/synthgen.yaml, lighting.anchor_intensity_w)
to the empty-table frames of the cell's condition check, and score rendered frames
with the same check.

Only the frames taken for the condition check are used, never a trial frame or a
test image: results/e2kho-1/do_sang_e2kho-1.png and results/e2kho-2/do_sang_e2kho-2.png
(each the mean of 10 camera frames, empty table, hard condition), the overlays
results/e2kho-2/dieu_kien/lan_1.png and lan_2.png (the two earlier checks of
e2kho-2; their untinted parts are camera values, and the difference of two
checks isolates the sensor noise), and, for one ratio only, the standard-condition
check results/e2chuan-2/do_sang_e2chuan-2.png, which shows the white cards on the
bare table whose colour the object colours of the config are measured against.

The fit needs one reference render, because every element of the static scene is a
flat Lambertian surface on (or just below) the table plane: none lights another, so
the scene-linear image is exactly albedo times irradiance, and the irradiance of
the lamps comes from one render with all albedos 1. The room (a uniform world)
adds its radiance times the albedo on top.

Steps (paths of the paper folder and of the cell branch's condition_check.yaml):
    python scripts/fit_hard_profile.py measure --results <paper folder>/results \\
        --criteria <cell repo>/config/condition_check.yaml --out fit/measure.json
    # copy the geometry it prints into hard_profile, then:
    python scripts/fit_hard_profile.py reference --out fit/ref
    blenderproc run --custom-blender-path <blender> src/synthgen/render_blenderproc.py -- \\
        --scenes fit/ref/specs --out fit/ref/render --config fit/ref/reference.yaml --save-linear
    python scripts/fit_hard_profile.py fit --measure fit/measure.json --reference fit/ref/render \\
        --criteria <cell repo>/config/condition_check.yaml --out fit/fitted.json
    python scripts/fit_hard_profile.py noise --measure fit/measure.json --fitted fit/fitted.json \\
        --out fit/noise.json
    # copy the fitted numbers into config/synthgen.yaml, render, then:
    python scripts/fit_hard_profile.py score --renders <render dir> --criteria ...

The fitted profile lives in config/synthgen.yaml; rendering needs none of this
script's inputs. Run on 05/11/2026 (Blender 4.2.1, BlenderProc 2.8.0): the fit
reproduces the card brightness and the median block colour of both parts of the
table to 0.1/255, and the noise of the sheet to 0.01.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.synthgen.hard_profile import load_render_config, static_elements  # noqa: E402
from src.synthgen.sensor_model import (  # noqa: E402
    apply_sensor, highpass_noise_std, linear_to_srgb, luma601, srgb_to_linear)

REF_LAMP_W = 10.0          # lamp power of the reference render
CARD_MM = 48.0             # card side of the standard-condition frame (27_make_position_cards.py)
FRAMES = {"e2kho-1": "e2kho-1/do_sang_e2kho-1.png", "e2kho-2": "e2kho-2/do_sang_e2kho-2.png"}


# ────────────────────────────────────────────────────────────
# The cell's condition check (src/perception/condition_check.py, fix/b3-review)
# ────────────────────────────────────────────────────────────

def _blocks(img: np.ndarray, box, px: int, cut=None):
    """Mean colour of the px-square blocks of an area and which to keep (clear of cut)."""
    import cv2

    x0, x1, y0, y1 = (int(v) for v in box)
    nx, ny = (x1 - x0) // px, (y1 - y0) // px
    crop = img[y0:y0 + ny * px, x0:x0 + nx * px].astype(np.float32)
    means = cv2.resize(crop, (nx, ny), interpolation=cv2.INTER_AREA)
    xs, ys = x0 + px * np.arange(nx), y0 + px * np.arange(ny)
    keep = np.ones((ny, nx), bool)
    if cut is not None:
        cx0, cx1, cy0, cy1 = (int(v) for v in cut)
        keep &= ~(((xs + px > cx0) & (xs < cx1))[None, :] & ((ys + px > cy0) & (ys < cy1))[:, None])
    return means, keep


def condition_measure(rgb: np.ndarray, crit: dict) -> dict:
    """The cell's measure on an RGB frame (0-255): card brightness and, per part of the
    table, median green share, median brightness, share of bare blocks, and the
    per-channel median block colour (RGB) that the fit needs on top."""
    px = int(crit["hard"]["block_px"])
    cmeans, _ = _blocks(rgb, crit["cards"], px)
    cards = float(np.median(luma601(cmeans)))
    out = {"card_brightness": cards,
           "card_rgb": np.median(cmeans.reshape(-1, 3), axis=0).tolist(),
           "mean_brightness": float(rgb.astype(float).mean())}
    for area in ("upper_area", "lower_area"):
        means, keep = _blocks(rgb, crit[area], px, crit["cards"] if area == "upper_area" else None)
        m = means[keep]
        lum = luma601(m)
        out[area] = {"green_share": float(np.median(m[:, 1] / (m.sum(axis=1) + 1e-6))),
                     "brightness": float(np.median(lum)),
                     "bare_share": float((lum > float(crit["hard"]["bare_relative_brightness"]) * cards).mean()),
                     "rgb": np.median(m, axis=0).tolist()}
    return out


def judge_hard(m: dict, crit: dict) -> list[str]:
    """Why the frame is not the hard condition; empty when it is (condition_check.judge_hard)."""
    hard = crit["hard"]
    lo, hi = (float(v) for v in hard["luminance"])
    problems = []
    for area in ("upper_area", "lower_area"):
        a = m[area]
        if a["bare_share"] > float(hard["max_bare_share"]):
            problems.append(f"{area}: bare table")
        if a["green_share"] < float(hard["min_green_share"][area]):
            problems.append(f"{area}: green share {a['green_share']:.3f}")
        if not lo <= a["brightness"] <= hi:
            problems.append(f"{area}: brightness {a['brightness']:.1f}")
    return problems


# ────────────────────────────────────────────────────────────
# Camera geometry
# ────────────────────────────────────────────────────────────

class Camera:
    """Pinhole camera of the calibration (the hand-eye fit solved with zero distortion)."""

    def __init__(self, T_BC: np.ndarray, intr: dict, z_mm: float) -> None:
        self.T, self.z = np.asarray(T_BC, float), float(z_mm)
        self.fx, self.fy, self.cx, self.cy = (float(intr[k]) for k in ("fx", "fy", "ppx", "ppy"))

    def to_table(self, u: float, v: float, dz: float = 0.0) -> np.ndarray:
        d = self.T[:3, :3] @ np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
        s = (self.z + dz - self.T[2, 3]) / d[2]
        return self.T[:3, 3] + s * d

    def to_image(self, p) -> np.ndarray:
        pc = self.T[:3, :3].T @ (np.asarray(p, float) - self.T[:3, 3])
        return np.array([self.fx * pc[0] / pc[2] + self.cx, self.fy * pc[1] / pc[2] + self.cy])


def _robust_line(t, s) -> np.ndarray:
    t, s = np.asarray(t, float), np.asarray(s, float)
    keep = np.ones(len(t), bool)
    for _ in range(5):
        c = np.polyfit(t[keep], s[keep], 1)
        r = s - np.polyval(c, t)
        keep = np.abs(r) < 3.0 * (1.4826 * np.median(np.abs(r[keep])) + 0.3)
    return c


def _edge(profile: np.ndarray, lo: int, hi: int) -> tuple[float, float]:
    """Sub-pixel position of the steepest step of a 1-D profile within [lo, hi)."""
    g = np.abs(np.gradient(profile))[lo:hi]
    i = int(np.argmax(g))
    d = 0.0
    if 0 < i < len(g) - 1:
        den = g[i - 1] - 2 * g[i] + g[i + 1]
        d = 0.5 * (g[i - 1] - g[i + 1]) / den if den != 0 else 0.0
    return lo + i + d, float(g.max())


def measure_geometry(rgb: np.ndarray, cam: Camera) -> dict:
    """Table edges, black strip and card columns of one empty-table frame, base frame mm."""
    import cv2

    lum = cv2.GaussianBlur(luma601(rgb.astype(np.float32)).astype(np.float32), (0, 0), 1.2)
    lines = {}
    for name, ts, lo, hi, along_rows in (("left", range(30, 715, 5), 186, 214, True),
                                         ("right", range(30, 715, 5), 1094, 1116, True),
                                         ("near", range(240, 1080, 10), 4, 32, False)):
        t_ok, s_ok = [], []
        for t in ts:
            pos, gmax = _edge(lum[t, :] if along_rows else lum[:, t], lo, hi)
            if gmax > 2.0:
                t_ok.append(t)
                s_ok.append(pos)
        lines[name] = _robust_line(t_ok, s_ok)
    # The strip: per column, the dark band between the two sheet levels; its centre
    # is the middle of the two half-depth crossings.
    us, centres, widths = [], [], []
    for u in range(240, 1080, 20):
        col = cv2.GaussianBlur(lum[:, u].reshape(-1, 1), (1, 5), 1.0).ravel()
        seg = col[450:500]
        i = int(np.argmin(seg))
        top_lvl, bot_lvl = np.median(col[430:455]), np.median(col[490:515])
        top = 450 + np.where(seg[:i + 1] >= 0.5 * (top_lvl + seg[i]))[0].max()
        bot = 450 + i + np.where(seg[i:] >= 0.5 * (bot_lvl + seg[i]))[0].min()
        us.append(u)
        centres.append(0.5 * (top + bot))
        widths.append(bot - top)
    strip_line = np.polyfit(us, centres, 1)
    strip_px = float(np.median(widths))
    # Card columns: connected bright components inside the card block.
    thr = 0.5 * (np.median(lum[240:380, 450:820]) + np.median(lum[150:200, 450:820]))
    m = (lum > thr).astype(np.uint8)
    m[:200], m[410:], m[:, :420], m[:, 860:] = 0, 0, 0, 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    cols = []
    for i in range(1, n):
        x0, y0, w, h, area = st[i]
        if area < 2000 or w > 70:           # merged neighbours give no clean width
            continue
        mm = lab == i
        occ_u, occ_v = mm[y0 + 10:y0 + h - 10].mean(0), mm[:, x0 + 8:x0 + w - 8].mean(1)
        uL, uR = np.where(occ_u > 0.5)[0][[0, -1]] + np.array([-0.5, 0.5])
        vT, vB = np.where(occ_v > 0.5)[0][[0, -1]] + np.array([-0.5, 0.5])
        c = [cam.to_table(u, v, 0.2) for u, v in ((uL, vT), (uR, vT), (uL, vB), (uR, vB))]
        cols.append({"width_y_mm": 0.5 * ((c[1][1] - c[0][1]) + (c[3][1] - c[2][1])),
                     "length_x_mm": 0.5 * ((c[2][0] - c[0][0]) + (c[3][0] - c[1][0])),
                     "centre_mm": (0.25 * sum(c))[:2].tolist()})

    # Table corners on the table plane: the near edge meets the left and right edges;
    # the far corners lie 100 mm beyond the bottom image row (the far edge is not seen).
    def pt_left(v):
        return cam.to_table(np.polyval(lines["left"], v), v)

    def pt_right(v):
        return cam.to_table(np.polyval(lines["right"], v), v)

    def pt_near(u):
        return cam.to_table(u, np.polyval(lines["near"], u))

    def intersect(p1, p2, q1, q2):
        a = np.array([[p2[0] - p1[0], -(q2[0] - q1[0])], [p2[1] - p1[1], -(q2[1] - q1[1])]])
        t = np.linalg.solve(a, np.array([q1[0] - p1[0], q1[1] - p1[1]]))[0]
        return p1[:2] + t * (p2[:2] - p1[:2])

    nl = intersect(pt_left(40), pt_left(700), pt_near(260), pt_near(1060))
    nr = intersect(pt_right(40), pt_right(700), pt_near(260), pt_near(1060))
    far_x = cam.to_table(640, 719)[0] + 100.0
    fl = intersect(pt_left(40), pt_left(700), np.array([far_x, -2000.0]), np.array([far_x, 2000.0]))
    fr = intersect(pt_right(40), pt_right(700), np.array([far_x, -2000.0]), np.array([far_x, 2000.0]))
    s0 = cam.to_table(240, np.polyval(strip_line, 240))
    s1 = cam.to_table(1040, np.polyval(strip_line, 1040))
    sc = cam.to_table(640, np.polyval(strip_line, 640))
    width_mm = float(cam.to_table(640, np.polyval(strip_line, 640) + strip_px / 2)[0]
                     - cam.to_table(640, np.polyval(strip_line, 640) - strip_px / 2)[0])
    return {"table_corners_mm": [p.tolist() for p in (nl, nr, fr, fl)],
            "strip_centre_line_mm": [s0[:2].tolist(), s1[:2].tolist()], "strip_centre_mm": sc[:2].tolist(),
            "strip_width_mm": width_mm, "strip_line_px": strip_line.tolist(),
            "edge_lines_px": {k: v.tolist() for k, v in lines.items()}, "card_columns": cols}


# ────────────────────────────────────────────────────────────
# Appearance targets
# ────────────────────────────────────────────────────────────

def region_masks(shape, geo: dict) -> dict[str, np.ndarray]:
    """Image regions of the strip core and of the three surround sides."""
    h, w = shape[:2]
    vv, uu = np.mgrid[0:h, 0:w]
    left = np.polyval(geo["edge_lines_px"]["left"], vv)
    right = np.polyval(geo["edge_lines_px"]["right"], vv)
    near = np.polyval(geo["edge_lines_px"]["near"], uu)
    strip = np.polyval(geo["strip_line_px"], uu)
    return {"strip": (np.abs(vv - strip) <= 2.0) & (uu >= 240) & (uu <= 1040),
            "surround_left": uu < left - 8,
            "surround_right": uu > right + 8,
            "surround_near": (vv < near - 4) & (uu > left + 8) & (uu < right - 8),
            "surround_all": (uu < left - 8) | (uu > right + 8) | ((vv < near - 4) & (uu > left) & (uu < right))}


def appearance(rgb: np.ndarray, crit: dict, masks: dict) -> dict:
    """Condition measures plus the median colour of the strip core and surround sides."""
    out = condition_measure(rgb, crit)
    for name in ("strip", "surround_left", "surround_right", "surround_near", "surround_all"):
        out[name] = np.median(rgb[masks[name]].astype(float), axis=0).tolist()
    return out


def lin_mean(values: list) -> list[float]:
    """Mean of sRGB 0-255 colours taken in linear light, returned as sRGB 0-255."""
    lin = np.mean([srgb_to_linear(np.asarray(v, float) / 255.0) for v in values], axis=0)
    return (linear_to_srgb(lin) * 255.0).tolist()


def card_ratio_standard(rgb: np.ndarray, cam: Camera, cards: list) -> tuple[list, list]:
    """Linear colour of the cards over the bare table on the standard-condition frame.

    Each card is read inside a 36 mm square around its centre (the 48 mm card less
    its cut border), the table on the bare top above and beside the card block.
    """
    lin = srgb_to_linear(rgb.astype(float) / 255.0)
    card_means = []
    for _, x, y in cards:
        u0, v0 = cam.to_image([x - 18.0, y - 18.0, cam.z + 0.2])
        u1, v1 = cam.to_image([x + 18.0, y + 18.0, cam.z + 0.2])
        ua, ub = sorted((int(round(u0)), int(round(u1))))
        va, vb = sorted((int(round(v0)), int(round(v1))))
        card_means.append(lin[va:vb, ua:ub].reshape(-1, 3).mean(axis=0))
    card = np.median(card_means, axis=0)
    table = np.median(np.concatenate([lin[60:200, 450:820].reshape(-1, 3),
                                      lin[150:450, 850:1050].reshape(-1, 3)]), axis=0)
    return (card / table).tolist(), (linear_to_srgb(card) * 255).tolist()


def noise_targets(results: Path, crit: dict) -> dict:
    """Single-frame sensor noise from the difference of two condition checks.

    Each check is the mean of 10 frames, so its noise is the camera's over sqrt(10),
    plus the texture of the sheet, which does not average out. The difference of two
    checks taken 10 s apart cancels the texture; half its variance, less the 8-bit
    rounding, times 10, is the variance of one frame. Only pixels that the overlay
    left untinted and that are flat in the raw frame are used, binned by brightness.
    """
    import cv2

    raw = cv2.imread(str(results / FRAMES["e2kho-2"])).astype(np.float32)[..., ::-1]
    ov3 = cv2.imread(str(results / "e2kho-2/dieu_kien/lan_3.png")).astype(np.float32)[..., ::-1]
    untinted = np.all(ov3 == raw, axis=-1)
    l3 = luma601(raw).astype(np.float32)
    hp = lambda img: img - cv2.blur(img, (7, 7))             # noqa: E731
    flat = cv2.erode((np.abs(hp(l3)) < 2.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    q = float(np.sum(np.array([0.299, 0.587, 0.114]) ** 2) / 12.0) * 48.0 / 49.0
    bins = {"sheet": (15.0, 30.0), "cards": (100.0, 125.0)}
    out = {b: [] for b in bins}
    acorr = []
    for name in ("lan_1", "lan_2"):
        ov = cv2.imread(str(results / f"e2kho-2/dieu_kien/{name}.png")).astype(np.float32)[..., ::-1]
        ok = untinted & np.all(np.abs(ov - raw) < 15, axis=-1)
        ok = cv2.erode(ok.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & flat
        d = hp((luma601(ov) - l3).astype(np.float32))
        for b, (lo, hi) in bins.items():
            m = ok & (l3 >= lo) & (l3 < hi)
            out[b].append((float(np.median(l3[m])), float(np.sqrt(10.0 * (d[m].var() / 2.0 - q)))))
        # Lag-1 correlation of the noise itself: the rounding of the two checks adds
        # white variance q to the difference, which dilutes the measured correlation.
        dm = np.where(ok, d, 0.0)
        okf = ok.astype(float)
        num = (dm[:, :-1] * dm[:, 1:]).sum() / (okf[:, :-1] * okf[:, 1:]).sum()
        acorr.append(float(num / (d[ok].var() - 2.0 * q)))
    return {"levels": {b: [float(np.mean([v[0] for v in out[b]])), float(np.mean([v[1] for v in out[b]]))]
                       for b in bins},
            "acorr_lag1": float(np.mean(acorr))}


def cmd_measure(args) -> int:
    import cv2
    import yaml

    results = Path(args.results)
    crit = yaml.safe_load(open(args.criteria, encoding="utf-8"))
    T_BC = np.load(PROJECT_ROOT / "config/calibration/T_base_camera.npy")
    syn = yaml.safe_load(open(PROJECT_ROOT / "config/synthgen.yaml", encoding="utf-8"))
    cam = Camera(T_BC, syn["camera"]["intrinsics"], syn["objects"]["table_z_mm"])
    frames = {k: cv2.imread(str(results / p))[..., ::-1].copy() for k, p in FRAMES.items()}
    geo = {k: measure_geometry(f, cam) for k, f in frames.items()}
    g = geo["e2kho-2"]
    masks = region_masks(frames["e2kho-2"].shape, g)
    app = {k: appearance(f, crit, masks) for k, f in frames.items()}
    from src.synthgen.hard_profile import read_cards
    cards = read_cards(PROJECT_ROOT / "config/pose_lists/hard_v2.csv.cards.txt")
    std = cv2.imread(str(results / "e2chuan-2/do_sang_e2chuan-2.png"))[..., ::-1].copy()
    ratio, card_std = card_ratio_standard(std, cam, cards)
    out = {"geometry": geo, "appearance": app, "card_over_table_standard": ratio,
           "card_standard_srgb": card_std, "noise": noise_targets(results, crit)}
    cols = [c for gg in geo.values() for c in gg["card_columns"]]
    out["card_size_mm"] = [float(np.median([c["length_x_mm"] for c in cols])),
                           float(np.median([c["width_y_mm"] for c in cols]))]
    out["card_centre_x_mm"] = float(np.median([c["centre_mm"][0] for c in cols]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k in FRAMES:
        a = app[k]
        print(f"{k}: cards {a['card_brightness']:.1f}  upper {a['upper_area']['green_share']:.3f}/"
              f"{a['upper_area']['brightness']:.1f}  lower {a['lower_area']['green_share']:.3f}/"
              f"{a['lower_area']['brightness']:.1f}  strip {np.round(a['strip'], 1)}")
        print(f"  table corners {np.round(geo[k]['table_corners_mm'], 1).tolist()}")
        print(f"  strip {np.round(geo[k]['strip_centre_line_mm'], 1).tolist()} width {geo[k]['strip_width_mm']:.1f}")
    print(f"cards {np.round(out['card_size_mm'], 1)} mm, centre x {out['card_centre_x_mm']:.1f}; "
          f"card/table (standard, linear) {np.round(ratio, 3)}; noise {out['noise']}")
    return 0


def cmd_reference(args) -> int:
    """Spec and config of the reference render: every albedo 1, the anchor lamps at
    REF_LAMP_W each, no room light, no sensor effects."""
    import yaml

    out = Path(args.out)
    (out / "specs").mkdir(parents=True, exist_ok=True)
    syn = yaml.safe_load(open(PROJECT_ROOT / "config/synthgen.yaml", encoding="utf-8"))
    prof = syn["hard_profile"]
    prof["table"]["lower_sheet_ratio"] = [1.0, 1.0, 1.0]
    prof["strip"]["albedo_rgb"] = [1.0, 1.0, 1.0]
    prof["cards"]["albedo_rgb"] = [1.0, 1.0, 1.0]
    for side in ("left", "right", "near"):
        prof["surround"][f"{side}_albedo_rgb"] = [1.0, 1.0, 1.0]
    prof["environment"]["world_radiance_rgb"] = [0.0, 0.0, 0.0]
    syn["sensor"].update({"gain": 1.0, "white_balance_rgb": [1.0, 1.0, 1.0],
                          "shot_noise": 0.0, "read_noise": 0.0})
    (out / "reference.yaml").write_text(yaml.safe_dump(syn, sort_keys=False), encoding="utf-8")
    T_BC = np.load(PROJECT_ROOT / "config/calibration/T_base_camera.npy")
    lit = syn["lighting"]
    spec = {"spec_version": 1, "index": 0, "seed": 0, "mode": "reference",
            "camera": {"T_BC_mm": T_BC.tolist(), "intrinsics": syn["camera"]["intrinsics"]},
            "lights": [{"type": "point", "position_mm": p, "energy_w": REF_LAMP_W,
                        "color_temp_k": lit["anchor_color_temp_k"]} for p in lit["anchor_positions_mm"]],
            "background": {"kind": "plane", "z_mm": syn["objects"]["table_z_mm"],
                           "albedo_rgb": [1.0, 1.0, 1.0], "roughness": 1.0},
            "objects": [], "distractors": []}
    (out / "specs" / "scene_000000.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(f"wrote {out / 'specs'} and {out / 'reference.yaml'}")
    return 0


def element_map(cfg: dict, shape, cam: Camera) -> dict[str, np.ndarray]:
    """Which static element each pixel sees, by ray-casting the profile's polygons."""
    from matplotlib.path import Path as MplPath

    h, w = shape[:2]
    vv, uu = np.mgrid[0:h, 0:w]
    rays = np.stack([(uu - cam.cx) / cam.fx, (vv - cam.cy) / cam.fy, np.ones_like(uu, float)], -1)
    d = rays @ cam.T[:3, :3].T
    els = static_elements(cfg, [1.0, 1.0, 1.0], cfg["table_z_mm"])
    owner = np.full((h, w), -1, int)
    best = np.full((h, w), -np.inf)
    for k, el in enumerate(els):
        s = (el["z_mm"] - cam.T[2, 3]) / d[..., 2]
        xy = cam.T[:2, 3] + s[..., None] * d[..., :2]
        inside = MplPath(np.asarray(el["polygon_mm"])).contains_points(xy.reshape(-1, 2)).reshape(h, w)
        top = inside & (el["z_mm"] > best)
        owner[top], best[top] = k, el["z_mm"]
    return {el["name"]: owner == k for k, el in enumerate(els)}


def cmd_fit(args) -> int:
    import yaml
    from scipy.optimize import brentq

    meas = json.loads(Path(args.measure).read_text(encoding="utf-8"))
    crit = yaml.safe_load(open(args.criteria, encoding="utf-8"))
    cfg = load_render_config(PROJECT_ROOT / "config/synthgen.yaml")
    syn = yaml.safe_load(open(PROJECT_ROOT / "config/synthgen.yaml", encoding="utf-8"))
    I_l = np.load(Path(args.reference) / "frame_000000_linear.npy").astype(np.float64)
    T_BC = np.load(PROJECT_ROOT / "config/calibration/T_base_camera.npy")
    cam = Camera(T_BC, syn["camera"]["intrinsics"], cfg["table_z_mm"])
    owner = element_map(cfg, I_l.shape, cam)
    masks = region_masks(I_l.shape, meas["geometry"]["e2kho-2"])
    app = meas["appearance"]
    # Targets: the two sessions averaged in linear light.
    tgt = {"card_rgb": lin_mean([a["card_rgb"] for a in app.values()]),
           "upper": lin_mean([a["upper_area"]["rgb"] for a in app.values()]),
           "lower": lin_mean([a["lower_area"]["rgb"] for a in app.values()]),
           "strip": lin_mean([a["strip"] for a in app.values()]),
           "surround_all": lin_mean([a["surround_all"] for a in app.values()])}
    for side in ("left", "right", "near"):
        tgt[side] = lin_mean([a[f"surround_{side}"] for a in app.values()])
    tgt_card_l = float(luma601(np.asarray(tgt["card_rgb"])))
    # The card in the albedo scale of the object colours: the bare standard table
    # (background.standard_table_rgb) times the card/table ratio of the standard check.
    rho_card = (srgb_to_linear(np.asarray(syn["background"]["standard_table_rgb"]))
                * np.asarray(meas["card_over_table_standard"]))

    names = list(owner)
    rho = {n: np.ones(3) for n in names}
    for n in names:
        if n.startswith("card_"):
            rho[n] = rho_card.copy()
    scale, wb, world = 1.0, np.ones(3), np.zeros(3)

    def image(sc, wbg, wrl, albedo):
        alb = np.zeros(I_l.shape)
        for n in names:
            alb[owner[n]] = albedo[n]
        lin = alb * (sc * I_l + wrl)
        return linear_to_srgb(lin * wbg) * 255.0

    def set_part(prefix, rgb):
        for n in names:
            if n == prefix or (prefix == "card" and n.startswith("card_")):
                rho[n] = np.asarray(rgb, float)

    sides = {"left": "surround_left", "right": "surround_right", "near": "surround_near"}
    for it in range(4):
        # Room radiance: the median off-table colour, in the units the render produces
        # before the white balance.
        world = srgb_to_linear(np.asarray(tgt["surround_all"]) / 255.0) / wb
        # Light level from the card brightness of the check, white balance from the card colour.
        f = lambda s: condition_measure(image(s, wb, world, rho), crit)["card_brightness"] - tgt_card_l  # noqa: E731
        scale = brentq(f, 1e-4, 1e3, xtol=1e-6)
        m = condition_measure(image(scale, wb, world, rho), crit)
        got = srgb_to_linear(np.asarray(m["card_rgb"]) / 255.0)
        want = srgb_to_linear(np.asarray(tgt["card_rgb"]) / 255.0)
        wb = wb * (want / got) / (want / got)[1]
        # Albedos, channel by channel, so the median block colour of each part matches.
        for part, key, area in (("sheet_upper", "upper", "upper_area"), ("sheet_lower", "lower", "lower_area")):
            for c in range(3):
                def g(a, part=part, c=c, area=area, key=key):
                    r = rho[part].copy()
                    r[c] = a
                    alb = dict(rho)
                    alb[part] = r
                    return condition_measure(image(scale, wb, world, alb), crit)[area]["rgb"][c] - tgt[key][c]
                rho[part][c] = brentq(g, 1e-5, 1.0, xtol=1e-7)
        for part, key, mask in [("strip", "strip", masks["strip"])] + \
                [(f"surround_{s}", s, masks[v]) for s, v in sides.items()]:
            for c in range(3):
                def h(a, part=part, c=c, key=key, mask=mask):
                    alb = dict(rho)
                    r = rho[part].copy()
                    r[c] = a
                    alb[part] = r
                    return float(np.median(image(scale, wb, world, alb)[..., c][mask])) - tgt[key][c]
                lo, hi = h(1e-5), h(1.0)
                rho[part][c] = 1e-5 if lo > 0 else (1.0 if hi < 0 else brentq(h, 1e-5, 1.0, xtol=1e-7))
        print(f"iteration {it}: lamp scale {scale:.4f}, wb {np.round(wb, 4)}, world {np.round(world, 5)}")

    final = image(scale, wb, world, rho)
    m = condition_measure(final, crit)
    to_srgb = lambda r: [round(float(v), 4) for v in linear_to_srgb(r)]   # noqa: E731
    fitted = {
        "lighting.anchor_intensity_w": round(REF_LAMP_W * scale, 3),
        "background.anchor_albedo_rgb": to_srgb(rho["sheet_upper"]),
        "hard_profile.table.lower_sheet_ratio": [round(float(v), 4) for v in rho["sheet_lower"] / rho["sheet_upper"]],
        "hard_profile.strip.albedo_rgb": to_srgb(rho["strip"]),
        "hard_profile.cards.albedo_rgb": to_srgb(rho_card),
        **{f"hard_profile.surround.{s}_albedo_rgb": to_srgb(rho[f"surround_{s}"]) for s in sides},
        "hard_profile.environment.world_radiance_rgb": [round(float(v), 5) for v in world],
        "sensor.white_balance_rgb": [round(float(v), 4) for v in wb],
    }
    print(json.dumps(fitted, indent=1))
    print("fitted frame (no noise):", json.dumps({k: m[k] for k in ("card_brightness", "upper_area", "lower_area")}))
    print("targets: cards %.1f, upper %s, lower %s" % (tgt_card_l, np.round(tgt["upper"], 1), np.round(tgt["lower"], 1)))
    # Linear colour (before the white balance) of the two noise levels: the upper sheet
    # and a card, as the fitted profile renders them.
    lin_of = lambda part: np.median((rho[part] * (scale * I_l + world))[owner[part]], axis=0)  # noqa: E731
    levels = {"sheet": lin_of("sheet_upper").tolist(), "cards": lin_of("card_04").tolist()}
    Path(args.out).write_text(json.dumps({"fitted": fitted, "targets": tgt, "noise_levels_linear": levels},
                                         indent=1), encoding="utf-8")
    return 0


def cmd_noise(args) -> int:
    """Sensor noise: grain from the lag-1 correlation, then the shot noise that gives the
    sheet of the fitted profile the single-frame noise of the condition checks.

    The sheet is the brightness of the hard condition, so the noise is matched there.
    A free read-noise term does not help: fitted jointly on the sheet and the cards it
    goes to zero, because the real noise does not fall towards the shadows (2.13 at
    luma 23, 2.37 at 111) while sRGB-encoded photon noise is almost flat; the card
    level is then reported, not matched. Every trial uses the same random field
    (common random numbers), so the search sees a smooth function."""
    import cv2
    from scipy.optimize import brentq

    noise = json.loads(Path(args.measure).read_text(encoding="utf-8"))["noise"]
    fit = json.loads(Path(args.fitted).read_text(encoding="utf-8"))
    wb = fit["fitted"]["sensor.white_balance_rgb"]
    levels = fit["noise_levels_linear"]

    def lag1(sigma):
        sensor = {"gain": 1.0, "white_balance_rgb": [1, 1, 1], "shot_noise": 0.0,
                  "read_noise": 1e-4, "grain_sigma_px": sigma}
        img = apply_sensor(np.full((300, 300, 3), 0.2), sensor, np.random.default_rng(0)).astype(np.float32)
        lum = luma601(img).astype(np.float32)
        hp = (lum - cv2.blur(lum, (7, 7)))[4:-4, 4:-4]
        hp = hp - hp.mean()
        return float((hp[:, :-1] * hp[:, 1:]).mean() / hp.var())

    sigma = brentq(lambda s: lag1(s) - noise["acorr_lag1"], 0.3, 3.0, xtol=1e-3)

    def std_at(name, a, gain=1.0):
        sensor = {"gain": gain, "white_balance_rgb": wb, "shot_noise": a, "read_noise": 0.0,
                  "grain_sigma_px": sigma}
        img = apply_sensor(np.broadcast_to(np.asarray(levels[name]), (300, 300, 3)), sensor,
                           np.random.default_rng(1))
        return highpass_noise_std(luma601(img)), float(np.median(luma601(img)))

    a = brentq(lambda a: std_at("sheet", a)[0] - noise["levels"]["sheet"][1], 1e-6, 1e-2, xtol=1e-8)
    out = {"sensor.grain_sigma_px": round(float(sigma), 3), "sensor.shot_noise": float(f"{a:.3g}"),
           "sensor.read_noise": 0.0, "targets": noise,
           "model": {name: [round(v, 2) for v in std_at(name, a)] for name in levels}}
    print(json.dumps(out, indent=1))
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


def class_brightness(rgb: np.ndarray, inst: np.ndarray, meta: dict) -> dict[str, list[float]]:
    """Median luma inside each labelled instance (eroded 2 px), by class id."""
    import cv2

    out: dict[str, list[float]] = {}
    lum = luma601(rgb.astype(float))
    for k, cls in meta["instances"].items():
        m = cv2.erode((inst == int(k)).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if m.sum() >= 50:
            out.setdefault(str(cls), []).append(float(np.median(lum[m])))
    return out


def cmd_score(args) -> int:
    """Condition check, noise and class brightness of every rendered frame of a dir."""
    import cv2
    import yaml

    crit = yaml.safe_load(open(args.criteria, encoding="utf-8"))
    rows, classes = [], {}
    for p in sorted(Path(args.renders).glob("frame_*_rgb.png")):
        rgb = cv2.imread(str(p))[..., ::-1].copy()
        m = condition_measure(rgb, crit)
        problems = judge_hard(m, crit)
        lum = luma601(rgb.astype(np.float32)).astype(np.float32)
        row = {"frame": p.name, "pass": not problems, "problems": problems,
               "card_brightness": round(m["card_brightness"], 1),
               "upper": [round(m["upper_area"]["green_share"], 3), round(m["upper_area"]["brightness"], 1)],
               "lower": [round(m["lower_area"]["green_share"], 3), round(m["lower_area"]["brightness"], 1)],
               "noise_lower_sheet": round(highpass_noise_std(lum[560:660, 600:700]), 2),
               "noise_upper_sheet": round(highpass_noise_std(lum[60:160, 300:400]), 2)}
        inst_path = p.with_name(p.name.replace("_rgb.png", "_inst.png"))
        meta_path = p.with_name(p.name.replace("_rgb.png", "_meta.json"))
        if inst_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            inst = cv2.imread(str(inst_path), cv2.IMREAD_UNCHANGED)
            cb = class_brightness(rgb, inst, meta)
            row["class_luma"] = {k: [round(v, 1) for v in vs] for k, vs in cb.items()}
            for k, vs in cb.items():
                classes.setdefault(k, []).extend(v / m["upper_area"]["brightness"] for v in vs)
        rows.append(row)
        print(json.dumps(row))
    summary = {"frames": len(rows), "pass": sum(r["pass"] for r in rows),
               "class_over_upper_sheet": {k: round(float(np.median(v)), 2) for k, v in sorted(classes.items())}}
    print(json.dumps(summary))
    if args.out:
        Path(args.out).write_text(json.dumps({"rows": rows, "summary": summary}, indent=1), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--results", required=True, help="results/ folder of the paper")
    m.add_argument("--criteria", required=True, help="condition_check.yaml of the cell branch")
    m.add_argument("--out", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--measure", required=True)
    f.add_argument("--reference", required=True, help="render dir of the reference render")
    f.add_argument("--criteria", required=True)
    f.add_argument("--out", required=True)
    nz = sub.add_parser("noise")
    nz.add_argument("--measure", required=True)
    nz.add_argument("--fitted", required=True, help="output of the fit step")
    nz.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--renders", required=True)
    s.add_argument("--criteria", required=True)
    s.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    return {"measure": cmd_measure, "reference": cmd_reference, "fit": cmd_fit, "noise": cmd_noise,
            "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
