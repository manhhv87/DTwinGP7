"""
hard_profile.py
───────────────
The image condition every E3-E5 trial runs in, as the synthetic generator draws
it, and the render settings that go with it.

The hard condition is a green sheet over the table under dimmed light, with the 21
white position cards on it, the black strip across the table, and the table's edges
with the cell around them: about 29% of the image width is off the table. Before
November 2026 the generator rendered one uniform grey plane that filled the frame,
lit to the standard condition, and no recipe at any kappa produced a frame that the
cell's own condition check passes (0 of 200+ renders).

This module reads the render-side sections of config/synthgen.yaml (render, sensor,
hard_profile), turns the profile into the flat scene elements the renderer builds,
and writes and compares the render manifest. Both recipes use the profile: the
sampler draws the upper sheet's colour (spec background.albedo_rgb) and the light
around the profile's level, and the static elements stay as measured.

The light is a set of 600 mm area luminaires above the cell plus a uniform overhead
(the world), as the mirror top of the stainless case shows them in the training
images; luminaire_lights() turns a drawn layout and level into the spec's lights and
environment, so the sampler and the profile fit (scripts/fit_hard_profile.py) share
one definition of the level: the irradiance at the card block's centre.

Every number of the profile is fitted on the empty-table frames of the condition
check (scripts/fit_hard_profile.py); the config comments say which.

Pure Python + numpy. render_blenderproc.py imports it inside Blender, where it is a
top-level module next to sensor_model.py (src/synthgen is put on sys.path), hence
the fallback import below.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .sensor_model import LUMINANCE_709, apply_sensor, linear_to_srgb, luma601, srgb_to_linear
except ImportError:  # loaded as a top-level module by render_blenderproc.py
    from sensor_model import (  # type: ignore[no-redef]
        LUMINANCE_709, apply_sensor, linear_to_srgb, luma601, srgb_to_linear)

RENDER_SECTIONS = ("render", "sensor", "hard_profile")
RENDER_KEYS = ("samples", "denoiser", "diffuse_bounces", "glossy_bounces", "max_bounces",
               "transmission_bounces", "transparent_max_bounces", "view_transform",
               "noise_threshold", "reflectance_scale")
MANIFEST_NAME = "render_manifest.json"
MANIFEST_VERSION = 1
# What has to be equal for two batches to count as rendered the same way. Two arms
# of a comparison (E3 recipes, E5 ablations, E4 rounds) that differ here differ in
# more than the factor under test: E5 once allowed 64 samples where E3 used 24.
MATCH_KEYS = ("blender_version", "blenderproc_version", "device", "render", "profile_sha256")


def _rgb(value: Any, name: str) -> list[float]:
    rgb = [float(v) for v in value]
    if len(rgb) != 3 or not all(0.0 <= v <= 1.0 for v in rgb):
        raise ValueError(f"{name} must be three sRGB-encoded values in [0, 1]")
    return rgb


def _xy(value: Any, name: str) -> list[float]:
    xy = [float(v) for v in value]
    if len(xy) != 2:
        raise ValueError(f"{name} must be [x, y] in mm")
    return xy


def check_render_config(cfg: dict[str, Any]) -> None:
    """Raise ValueError when the render or hard_profile section is incomplete.

    The sensor section is checked by sensor_model.check_sensor.
    """
    render = cfg["render"]
    missing = [k for k in RENDER_KEYS if k not in render]
    if missing:
        raise ValueError(f"render section misses {missing}")
    if int(render["samples"]) < 1:
        raise ValueError("render.samples must be at least 1")
    if int(render["max_bounces"]) < int(render["glossy_bounces"]):
        raise ValueError("render.max_bounces caps every bounce type, so it must be at "
                         "least render.glossy_bounces or the mirror case renders black")
    if render["view_transform"] != "Standard":
        raise ValueError("render.view_transform must be Standard: the sensor model applies "
                         "the plain sRGB curve, and Filmic/AgX would tone-map twice")
    if not 0.0 < float(render["reflectance_scale"]) <= 1.0:
        raise ValueError("render.reflectance_scale must lie in (0, 1]: it converts physical "
                         "specular reflectances into the albedo units of the config")
    prof = cfg["hard_profile"]
    for key in ("table", "strip", "cards", "surround"):
        if key not in prof:
            raise ValueError(f"hard_profile misses '{key}'")
    corners = prof["table"]["corners_mm"]
    if len(corners) != 4:
        raise ValueError("hard_profile.table.corners_mm must list the 4 table corners")
    for c in corners:
        _xy(c, "hard_profile.table.corners_mm")
    ratio = [float(v) for v in prof["table"]["lower_sheet_ratio"]]
    if len(ratio) != 3 or not all(v > 0.0 for v in ratio):
        raise ValueError("hard_profile.table.lower_sheet_ratio must be 3 positive factors")
    strip = prof["strip"]
    if len(strip["centre_line_mm"]) != 2 or float(strip["width_mm"]) <= 0.0:
        raise ValueError("hard_profile.strip needs two centre-line points and a width")
    _rgb(strip["albedo_rgb"], "hard_profile.strip.albedo_rgb")
    cards = prof["cards"]
    if len(cards["size_mm"]) != 2 or not cards["centres_mm"]:
        raise ValueError("hard_profile.cards needs size_mm [along x, along y] and centres_mm")
    _rgb(cards["albedo_rgb"], "hard_profile.cards.albedo_rgb")
    for side in ("left", "right", "near"):
        _rgb(prof["surround"][f"{side}_albedo_rgb"], f"hard_profile.surround.{side}_albedo_rgb")


def load_render_config(path: str | Path) -> dict[str, Any]:
    """The render-side sections of config/synthgen.yaml, checked.

    Also carries the two numbers of other sections the renderer needs: the lamp
    temperature the camera's white is balanced to (lighting.anchor_color_temp_k) and
    the table height (objects.table_z_mm).
    """
    import yaml  # lazy import

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    missing = [s for s in RENDER_SECTIONS if s not in data]
    if missing:
        raise ValueError(f"{path} has no {missing} section(s)")
    cfg = {s: data[s] for s in RENDER_SECTIONS}
    cfg["white_k"] = float(data["lighting"]["anchor_color_temp_k"])
    cfg["table_z_mm"] = float(data["objects"]["table_z_mm"])
    check_render_config(cfg)
    return cfg


def profile_sha256(cfg: dict[str, Any]) -> str:
    """Hash of everything the renderer takes from the config, key order ignored."""
    keep = {k: cfg[k] for k in (*RENDER_SECTIONS, "white_k", "table_z_mm")}
    text = json.dumps(keep, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ────────────────────────────────────────────────────────────
# Scene geometry (flat polygons in the base frame, mm)
# ────────────────────────────────────────────────────────────

def _ccw(points: Any) -> np.ndarray:
    """Polygon as an (n, 2) array, counter-clockwise seen from above (+z)."""
    p = np.asarray(points, dtype=float)
    area = 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
    return p if area > 0 else p[::-1].copy()


def clip_halfplane(poly: np.ndarray, normal: Any, offset: float) -> np.ndarray:
    """Part of a convex polygon where normal . p <= offset (Sutherland-Hodgman)."""
    n = np.asarray(normal, dtype=float)
    out: list[np.ndarray] = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        da, db = float(n @ a) - offset, float(n @ b) - offset
        if da <= 0.0:
            out.append(a)
        if da * db < 0.0:
            out.append(a + (b - a) * (da / (da - db)))
    return np.asarray(out, dtype=float).reshape(-1, 2)


def _edge_outward(poly: np.ndarray, i: int) -> tuple[np.ndarray, float]:
    """Outward normal and offset of edge i of a CCW polygon: outside is n.p > off."""
    a, b = poly[i], poly[(i + 1) % len(poly)]
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([d[1], -d[0]])
    return n, float(n @ a)


def table_corners(cfg: dict[str, Any]) -> np.ndarray:
    """Table top corners, CCW, ordered near-left first (smallest x, then smallest y)."""
    p = _ccw(cfg["hard_profile"]["table"]["corners_mm"])
    start = int(np.lexsort((p[:, 1], p[:, 0]))[0])
    return np.roll(p, -start, axis=0)


def lower_sheet_albedo(upper_srgb: Any, cfg: dict[str, Any]) -> list[float]:
    """Albedo of the sheet beyond the strip, from the upper sheet's (spec) albedo.

    The two parts of the table are the same kind of sheet, yet the part beyond the
    black strip images less green and brighter (green share 0.40 against 0.45 on the
    condition frames). The fitted ratio is applied in linear light, so the sampler's
    draw of the upper sheet carries over to the lower one.
    """
    ratio = np.asarray(cfg["hard_profile"]["table"]["lower_sheet_ratio"], dtype=float)
    lin = np.clip(srgb_to_linear(upper_srgb) * ratio, 0.0, 1.0)
    return [float(v) for v in linear_to_srgb(lin)]


def static_elements(cfg: dict[str, Any], upper_srgb: Any, z_mm: float) -> list[dict[str, Any]]:
    """Flat scene elements of the profile: name, CCW polygon (mm), z (mm), sRGB albedo.

    The table splits along the strip into the upper sheet (robot side, where the
    parts go), the strip and the lower sheet; the cards lie on the upper sheet; the
    surround lies just below the table top on the three sides the camera sees past
    the table's edges (the far edge is outside the image).
    """
    prof = cfg["hard_profile"]
    table = table_corners(cfg)
    strip = prof["strip"]
    p0, p1 = (np.asarray(_xy(p, "strip.centre_line_mm"), dtype=float)
              for p in strip["centre_line_mm"])
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-d[1], d[0]])
    if n[0] < 0.0:              # orient the strip normal away from the robot (+x)
        n = -n
    c, hw = float(n @ p0), 0.5 * float(strip["width_mm"])
    upper = clip_halfplane(table, n, c - hw)
    lower = clip_halfplane(table, -n, -(c + hw))
    band = clip_halfplane(clip_halfplane(table, n, c + hw), -n, -(c - hw))
    z = float(z_mm)
    els = [
        {"name": "sheet_upper", "polygon_mm": upper, "z_mm": z, "albedo_rgb": _rgb(upper_srgb, "sheet")},
        {"name": "sheet_lower", "polygon_mm": lower, "z_mm": z,
         "albedo_rgb": lower_sheet_albedo(upper_srgb, cfg)},
        {"name": "strip", "polygon_mm": band, "z_mm": z, "albedo_rgb": _rgb(strip["albedo_rgb"], "strip")},
    ]
    cards = prof["cards"]
    hx, hy = 0.5 * float(cards["size_mm"][0]), 0.5 * float(cards["size_mm"][1])
    for k, (cx, cy) in enumerate(cards["centres_mm"], start=1):
        rect = np.array([[cx - hx, cy - hy], [cx + hx, cy - hy], [cx + hx, cy + hy], [cx - hx, cy + hy]])
        # A card is paper on the sheet: lift it clear of the sheet so the two never
        # share a plane, which a ray tracer resolves at random from sample to sample.
        els.append({"name": f"card_{k:02d}", "polygon_mm": rect, "z_mm": z + 0.2,
                    "albedo_rgb": _rgb(cards["albedo_rgb"], "cards")})
    sur = prof["surround"]
    half = float(sur["extent_mm"])
    cx, cy = table.mean(axis=0)
    square = _ccw([[cx - half, cy - half], [cx + half, cy - half],
                   [cx + half, cy + half], [cx - half, cy + half]])
    # The table's edges, told apart by their outward normals rather than by index.
    edges = [_edge_outward(table, i) for i in range(4)]
    near = min(edges, key=lambda e: e[0][0])        # outward normal towards -x
    left = min(edges, key=lambda e: e[0][1])        # towards -y
    right = max(edges, key=lambda e: e[0][1])       # towards +y
    zs = z - float(sur["depth_mm"])
    left_poly = clip_halfplane(square, -left[0], -left[1])
    right_poly = clip_halfplane(square, -right[0], -right[1])
    near_poly = clip_halfplane(clip_halfplane(clip_halfplane(square, -near[0], -near[1]),
                                              left[0], left[1]), right[0], right[1])
    for name, poly in (("left", left_poly), ("right", right_poly), ("near", near_poly)):
        els.append({"name": f"surround_{name}", "polygon_mm": poly, "z_mm": zs,
                    "albedo_rgb": _rgb(sur[f"{name}_albedo_rgb"], f"surround.{name}")})
    return els


def read_cards(path: str | Path) -> list[tuple[int, float, float]]:
    """(card, x mm, y mm) from a .cards.txt written by 22_make_pose_lists.py."""
    import re

    pat = re.compile(r"card\s+(\d+):\s*x=\s*([-\d.]+)\s*mm\s+y=\s*([-\d.]+)\s*mm")
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = pat.search(line)
        if m:
            out.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return out


# ────────────────────────────────────────────────────────────
# Light: area luminaires and the uniform overhead
# ────────────────────────────────────────────────────────────

def panel_irradiance_factor(centre_mm: Any, size_mm: Any, point_mm: Any, n: int = 20) -> float:
    """Irradiance at an upward-facing point per unit radiance of a horizontal,
    downward-facing rectangular Lambertian panel: the integral of cos * cos / d^2
    over the panel (midpoint rule on n x n cells), dimensionless. A panel covering
    the whole sky gives pi, the irradiance of a uniform hemisphere of radiance 1."""
    c = np.asarray(centre_mm, dtype=float)
    p = np.asarray(point_mm, dtype=float)
    dz = float(c[2] - p[2])
    if dz <= 0.0:
        raise ValueError("a luminaire must hang above the point it lights")
    sx, sy = (float(v) for v in size_mm)
    g = (np.arange(n) + 0.5) / n - 0.5
    qx = c[0] + g[:, None] * sx - p[0]
    qy = c[1] + g[None, :] * sy - p[1]
    d2 = qx ** 2 + qy ** 2 + dz ** 2
    return float((dz ** 2 / d2 ** 2).sum() * sx * sy / n ** 2)


def area_light_power(radiance: float, size_mm: Any) -> float:
    """Blender power (W) of an area light of the given size whose face has the given
    radiance (render units, luminance): a Lambertian face emits pi * area * radiance.
    scripts/fit_hard_profile.py checks this on its reference render (the cards of an
    albedo-1 table lit to irradiance 1 must render at 1 / pi)."""
    sx, sy = (float(v) * 1e-3 for v in size_mm)
    return float(np.pi * float(radiance) * sx * sy)


def luminaire_lights(centres_mm: list, weights: Any, overhead_share: float, irradiance: float,
                     temps_k: list, tint_rgb: Any, size_mm: Any,
                     reference_mm: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Spec lights and environment for a luminaire layout and a light level.

    irradiance is the luminance-weighted irradiance at reference_mm (the centre of the
    card block on the table) in render units; the uniform overhead carries
    overhead_share of it, each luminaire weights[i] of the rest. A luminaire's colour
    (color_temp_k, as sensor_model.light_rgb renders it) has unit luminance, and so has
    the overhead's tint, so these shares hold at the reference point whatever the colours.
    """
    share = float(overhead_share)
    if not 0.0 <= share <= 1.0:
        raise ValueError("overhead_share must lie in [0, 1]")
    tint = np.asarray(tint_rgb, dtype=float)
    tint = tint / float(LUMINANCE_709 @ tint)
    lights = []
    for centre, w, temp in zip(centres_mm, weights, temps_k):
        g = panel_irradiance_factor(centre, size_mm, reference_mm)
        radiance = float(w) * (1.0 - share) * float(irradiance) / g
        lights.append({"type": "area", "position_mm": [float(v) for v in centre],
                       "size_mm": [float(v) for v in size_mm],
                       "energy_w": area_light_power(radiance, size_mm),
                       "color_temp_k": float(temp), "irradiance_share": float(w)})
    environment = {"overhead_share": share,
                   "radiance_rgb": (share * float(irradiance) / np.pi * tint).tolist()}
    return lights, environment


# ────────────────────────────────────────────────────────────
# Session exposure
# ────────────────────────────────────────────────────────────

def admitted_gain_range(sensor: dict[str, Any], upper_srgb: Any, lower_srgb: Any,
                        luminance: Any = (10.0, 45.0), seed: int = 0) -> tuple[float, float]:
    """Gains over which both table parts meet the check's luminance bounds.

    Each part's colour at the fit gain is put through the sensor model as a flat
    192 x 256 patch, noise on; the statistic is the median of 32 px block means of
    Rec. 601 luma, as the condition check takes it. The upper (darker) part sets the
    low end, the lower part the high end.
    """
    wb = np.asarray(sensor["white_balance_rgb"], dtype=float)
    gains = np.exp(np.linspace(np.log(0.05), np.log(20.0), 241))

    def level(srgb: Any, g: float) -> float:
        x = np.broadcast_to(srgb_to_linear(np.asarray(srgb, dtype=float)) / wb / float(sensor["gain"]),
                            (192, 256, 3)).astype(np.float32)
        y = luma601(apply_sensor(x, sensor, np.random.default_rng(seed), exposure_gain=g).astype(float))
        return float(np.median(y.reshape(6, 32, 8, 32).mean(axis=(1, 3))))

    def gain_at(srgb: Any, target: float) -> float:
        lv = np.array([level(srgb, g) for g in gains])
        return float(np.exp(np.interp(np.log(target), np.log(lv), np.log(gains))))

    return gain_at(upper_srgb, float(luminance[0])), gain_at(lower_srgb, float(luminance[1]))


# ────────────────────────────────────────────────────────────
# Render manifest
# ────────────────────────────────────────────────────────────

def render_device(kind: str, compute_type: str, names: list[str]) -> str:
    """The manifest's device label: Cycles' device, the GPU backend and every device used.

    BlenderProc renders a GPU batch on all GPUs of the first backend it finds AND on the
    CPU (hybrid rendering), so the label names both, sorted: two machines with the same
    backend but another GPU or CPU get different labels, and check_arms_match refuses to
    compare their batches. Arms rendered on one machine get one label.
    """
    head = f"GPU:{compute_type}" if kind == "GPU" else str(kind)
    used = sorted({str(name).strip() for name in names if str(name).strip()})
    return f"{head} [{', '.join(used)}]" if used else head


def build_manifest(cfg: dict[str, Any], render_settings: dict[str, Any], *, device: str,
                   blender_version: str, blenderproc_version: str, config_path: str,
                   scenes_dir: str, frames: list[str]) -> dict[str, Any]:
    """Render manifest of one batch: how it was rendered and from which profile."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "blender_version": blender_version,
        "blenderproc_version": blenderproc_version,
        "device": device,
        "render": dict(render_settings),
        "profile_sha256": profile_sha256(cfg),
        "profile": {k: cfg[k] for k in (*RENDER_SECTIONS, "white_k", "table_z_mm")},
        "config_path": config_path,
        "scenes_dir": scenes_dir,
        "frames": sorted(frames),
    }


def manifest_mismatches(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Names of the MATCH_KEYS on which two render manifests differ."""
    return [k for k in MATCH_KEYS if a.get(k) != b.get(k)]


def merge_manifest(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Add a batch to an output directory's manifest; refuse a batch rendered differently.

    Raises ValueError naming the differing settings, so a resumed or extended run
    cannot mix, say, 24-sample and 64-sample frames in one arm.
    """
    bad = manifest_mismatches(old, new)
    if bad:
        raise ValueError(f"render settings differ from the existing manifest in {bad}; "
                         "use a new output directory")
    merged = dict(new)
    merged["frames"] = sorted(set(old.get("frames", [])) | set(new["frames"]))
    return merged


def check_arms_match(manifests: dict[str, dict[str, Any]]) -> None:
    """Raise ValueError unless every arm's render manifest matches the first one.

    For packaging: the synthetic arms compared in one experiment must have been
    rendered with the same settings and the same scene profile.
    """
    names = list(manifests)
    for name in names[1:]:
        bad = manifest_mismatches(manifests[names[0]], manifests[name])
        if bad:
            raise ValueError(f"arm '{name}' was rendered differently from '{names[0]}': {bad}")
