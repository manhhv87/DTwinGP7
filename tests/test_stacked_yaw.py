"""
test_stacked_yaw.py
───────────────────
Grasp yaw of a part standing on a support.

B4, 20 Oct 2026, trial 14 (pose P0214): an inox box (180 x 85 x 40 mm, mirror top)
resting on the plastic box (180 x 140 x 30 mm), rgbd mode. The mask also covered part
of the plastic, the rectangle fitted to the whole mask measured 194.0 x 134.6 mm, its
long side lay about 12 deg off the yaw the pose list called for the inox, and the
jaws came down misaligned. On a support the rectangle is now fitted to the top face
alone (depth_modes.top_face_mask). A support is inferred only from the part's own
height, so a part whose top face is not raised above it, and a carton without a
trusted size fit, keep the whole mask, with the rectangle identical to before.

Since 22 Oct 2026 the grasp yaw and pixel come from the class's known top-face
rectangle fitted to that same outline (depth_modes.fit_footprint, tested in
test_known_size_fit.py); the minimum-area rectangle is still recorded. The tests
below check that the fit read the outline the rectangle read.

Scenes are ray-cast through the cell calibration in config/calibration (T_BC, the
colour intrinsics at 1280 x 720 and the fitted table plane): every box stands on the
table plane, a stacked box on that plane lifted by its support's height. The extractor
gets the class tables of the real block of config/experiment.yaml, as
03_run_experiment builds it.

The pose list gives the yaw of a long side 90 deg away from yaw_base (mod 180), so in
trial 14 the plastic, set at 0 deg, had its long side along base y and the inox, set
at 75 deg, lay at -15 deg in the base frame.
"""
from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

import src.perception.postprocess as postprocess
from src.perception.depth_modes import (
    MIN_SIZE_IOU,
    TOP_FACE_BAND_MM,
    TablePlane,
    fit_footprint,
    parse_class_sizes,
    top_face_mask,
)
from src.perception.postprocess import PoseExtractor, _metric_edge_yaw_base

ROOT = Path(__file__).resolve().parent.parent
CALIB = ROOT / "config" / "calibration"

T = np.load(CALIB / "T_base_camera.npy")
PLANE = TablePlane.from_dict(json.loads((CALIB / "table_plane.json").read_text(encoding="utf-8")))
INTR = json.loads((CALIB / "T_base_camera_meta.json").read_text(encoding="utf-8"))["intrinsics"]
SHAPE = (int(INTR["height"]), int(INTR["width"]))

REAL = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text(encoding="utf-8"))["real"]
SIZES = parse_class_sizes(REAL["class_sizes_mm"])
HEIGHTS = REAL["class_heights_mm"]
LARGE_CARTON, SMALL_CARTON = sorted(SIZES["carton"], key=lambda s: -s[2])
PLASTIC = SIZES["plastic_box"][0]
WOOD = SIZES["wood_box"][0]
METAL = SIZES["metal_box"][0]
INOX = SIZES["inox_box"][0]

CARD_7 = (525.0, -40.0)       # where P0214 put the stack
CARD_13 = (525.0, 80.0)       # where P0209, a carton on the plastic box, put it


@functools.lru_cache(maxsize=None)
def _table():
    """Viewing rays (base frame, camera-frame z = 1) and their depth on the table (mm)."""
    h, w = SHAPE
    R, t = T[:3, :3], T[:3, 3]
    vv, uu = np.mgrid[0:h, 0:w]
    r = np.stack([(uu - INTR["ppx"]) / INTR["fx"], (vv - INTR["ppy"]) / INTR["fy"],
                  np.ones((h, w))], axis=-1) @ R.T
    a, b, c = PLANE.a, PLANE.b, PLANE.c
    depth = (a * t[0] + b * t[1] + c - t[2]) / (r[..., 2] - a * r[..., 0] - b * r[..., 1])
    r.flags.writeable = False
    depth.flags.writeable = False
    return r, depth


@functools.lru_cache(maxsize=None)
def render(boxes):
    """Ray-cast boxes standing on the table plane: (labels, depth in metres).

    boxes: tuple of (centre_xy, yaw_deg, (L, W, H), lift_mm[, pitch_deg]). The L side
    lies along yaw_deg in the base frame and the box stands on the table plane lifted
    by lift_mm along its normal, that is on a support that high; centre_xy and
    lift_mm place the centre of its bottom face. pitch_deg, 0 by default, tilts the L
    side up about the W side through that centre, for a part leaning on a support.
    labels is 0 on the table and i where the i-th box (from 1) is the nearest
    surface. Each ray is intersected with the box's own slabs, only inside the box's
    projected bounding rectangle. The arrays are cached and read-only.
    """
    h, w = SHAPE
    R, t = T[:3, :3], T[:3, 3]
    n = PLANE.normal
    r, table = _table()
    depth = table.copy()
    labels = np.zeros((h, w), np.int32)
    for i, box in enumerate(boxes, start=1):
        cxy, yaw_deg, (L, W, H), lift = box[:4]
        p = math.radians(box[4]) if len(box) > 4 else 0.0
        d = np.array([math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg)), 0.0])
        ex = d - d.dot(n) * n
        ex /= np.linalg.norm(ex)
        M = np.stack([math.cos(p) * ex + math.sin(p) * n, np.cross(n, ex),
                      -math.sin(p) * ex + math.cos(p) * n])    # rows: box axes in base
        o0 = np.array([cxy[0], cxy[1], PLANE.z_at(cxy[0], cxy[1]) + lift / n[2]])
        local = np.array([[sx * L / 2, sy * W / 2, sz * H]
                          for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)])
        pc = (o0 + local @ M - t) @ R                          # corners, camera frame
        px = INTR["fx"] * pc[:, 0] / pc[:, 2] + INTR["ppx"]
        py = INTR["fy"] * pc[:, 1] / pc[:, 2] + INTR["ppy"]
        u0, u1 = max(0, int(px.min()) - 2), min(w, int(math.ceil(px.max())) + 3)
        v0, v1 = max(0, int(py.min()) - 2), min(h, int(math.ceil(py.max())) + 3)
        rl = r[v0:v1, u0:u1] @ M.T
        rl = np.where(np.abs(rl) < 1e-12, 1e-12, rl)
        o = M @ (t - o0)
        lo, hi = np.array([-L / 2, -W / 2, 0.0]), np.array([L / 2, W / 2, H])
        t1, t2 = (lo - o) / rl, (hi - o) / rl
        tmin = np.max(np.minimum(t1, t2), axis=-1)
        tmax = np.min(np.maximum(t1, t2), axis=-1)
        win = depth[v0:v1, u0:u1]
        hit = (tmax >= tmin) & (tmin > 0) & (tmin < win)
        win[hit] = tmin[hit]
        labels[v0:v1, u0:u1][hit] = i
    depth_m = (depth / 1000.0).astype(np.float32)
    labels.flags.writeable = False
    depth_m.flags.writeable = False
    return labels, depth_m


def heights(depth_m):
    """Height above the table plane of every pixel's depth (mm)."""
    h, w = SHAPE
    vv, uu = np.mgrid[0:h, 0:w]
    z = depth_m.astype(float) * 1000.0
    p = np.stack([(uu - INTR["ppx"]) * z / INTR["fx"], (vv - INTR["ppy"]) * z / INTR["fy"], z],
                 axis=-1)
    return PLANE.height_above(p @ T[:3, :3].T + T[:3, 3])


def stack(support, support_yaw, part, part_yaw, center=CARD_7):
    return render(((center, support_yaw, support, 0.0), (center, part_yaw, part, support[2])))


def leaning(part, support, part_yaw, center=CARD_7):
    """part with its low end on the table and its high end on support, its L side
    tilted up; the support stands across that end, its near edge 5 mm under it."""
    L = part[0]
    pitch = math.asin(support[2] / L)
    d = np.array([math.cos(math.radians(part_yaw)), math.sin(math.radians(part_yaw))])
    sc = np.asarray(center) + d * (L / 2 * math.cos(pitch) + support[1] / 2 - 5.0)
    return render(((tuple(sc), part_yaw + 90.0, support, 0.0),
                   (center, part_yaw, part, L / 2 * math.sin(pitch), math.degrees(pitch))))


def spill(mask, width_px=40):
    """The mask and a strip of table beside it, the way a detector mask spills."""
    ys, xs = np.nonzero(mask)
    out = mask.copy()
    out[ys.min():ys.max() + 1, xs.max() + 1:xs.max() + 1 + width_px] = True
    return out


def extractor(mode="rgbd"):
    return PoseExtractor(INTR, depth_mode=mode, T_BC_mm=T, table_plane=PLANE,
                         class_heights_mm=HEIGHTS, class_sizes_mm=REAL["class_sizes_mm"])


def extract(mask, depth, cls, mode="rgbd"):
    # The detector hands over the class height from class_heights_mm.
    return extractor(mode).extract(
        {"mask": mask.astype(np.uint8), "class_name": cls, "height_mm": HEIGHTS[cls]}, depth)


def axis_err_deg(yaw_rad, axis_deg):
    """Angle between the axis at yaw_rad and the axis at axis_deg, in [0, 90] deg."""
    return abs((math.degrees(yaw_rad) - axis_deg + 90.0) % 180.0 - 90.0)


def top_face_fit(mask, depth, size, h_depth):
    """The known-size fit extract makes on a part standing on a support: on the
    outline cut half the part's height below the top face, seeded with that
    outline's rectangle, read down to the support's top."""
    outline = top_face_mask(depth, mask, INTR, T, PLANE, h_depth, 0.5 * size[2])
    rect = _metric_edge_yaw_base(outline, INTR, T, PLANE, h_depth)
    return fit_footprint(outline, INTR, T, PLANE, h_depth, size[:2], rect["yaw_base"],
                         bottom_height_mm=h_depth - size[2])


def test_support_bar_is_half_the_thinnest_part():
    # The plastic box, 30 mm, is the thinnest part in the tables.
    assert extractor().min_support_mm == 15.0


# Trial 14 as staged, and the same pair at 0 and 75 deg in the base frame, where the
# rectangle of the union is nearly square.
ORIENTATIONS = [pytest.param(90.0, -15.0, id="trial14"), pytest.param(0.0, 75.0, id="base_0_75")]


class TestInoxOnPlastic:
    @pytest.mark.parametrize("mode", ["rgbd", "plane", "fusion"])
    @pytest.mark.parametrize("plastic_yaw,inox_yaw", ORIENTATIONS)
    def test_yaw_follows_the_inox(self, plastic_yaw, inox_yaw, mode):
        labels, depth = stack(PLASTIC, plastic_yaw, INOX, inox_yaw)
        mask = labels > 0                                      # both silhouettes
        det = extract(mask, depth, "inox_box", mode)
        assert det["h_depth_mm"] == pytest.approx(PLASTIC[2] + INOX[2], abs=2.0)
        # Was "metric_base_top_face", the rectangle of the cut outline: the yaw is
        # now the known-size fit on that same outline.
        assert det["yaw_source"] == "known_size_fit"
        fit = top_face_fit(mask, depth, INOX, det["h_depth_mm"])
        assert det["yaw_base"] == fit.yaw_base
        assert det["pixel_uv"] == fit.pixel_uv
        assert axis_err_deg(det["yaw_base"], inox_yaw) < 2.0
        # The rectangle the long-side check reads is still the cut outline's.
        outline = top_face_mask(depth, mask, INTR, T, PLANE, det["h_depth_mm"], 0.5 * INOX[2])
        rect = _metric_edge_yaw_base(outline, INTR, T, PLANE, det["h_depth_mm"])
        assert det["yaw_rect_long_mm"] == rect["long_mm"]
        # The defect: the rectangle of the whole mask.
        whole = _metric_edge_yaw_base(mask, INTR, T, PLANE, det["h_depth_mm"])
        assert axis_err_deg(whole["yaw_base"], inox_yaw) > 5.0

    @pytest.mark.parametrize("plastic_yaw,inox_yaw", ORIENTATIONS)
    def test_mirror_holes_in_the_inox_top(self, plastic_yaw, inox_yaw):
        labels, depth = stack(PLASTIC, plastic_yaw, INOX, inox_yaw)
        top = (labels == 2) & (heights(depth) > PLASTIC[2] + INOX[2] - 0.5)
        vv, uu = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
        holes = top & ((vv // 16 + uu // 16) % 2 == 0)        # 16 px blocks
        assert 0.4 < holes.sum() / top.sum() < 0.6
        depth = depth.copy()
        depth[holes] = 0.0
        det = extract(labels > 0, depth, "inox_box")
        assert det["depth_valid_frac"] >= 0.3
        # Was "metric_base_top_face": the fit now reads the same cut outline.
        assert det["yaw_source"] == "known_size_fit"
        assert det["yaw_base"] == top_face_fit(labels > 0, depth, INOX, det["h_depth_mm"]).yaw_base
        assert axis_err_deg(det["yaw_base"], inox_yaw) < 5.0


def test_leaning_part_keeps_its_outline():
    # A metal box with its high end on the plastic box and its low end on the
    # table: its top tilts 9.1 deg along the long side and falls 30 mm, less than
    # the cut half its 85 mm height down. The mask also covers the plastic.
    labels, depth = leaning(METAL, PLASTIC, -65.0)
    mask = labels > 0
    det = extract(mask, depth, "metal_box")
    # Was "metric_base_top_face": the fit now reads the same cut outline.
    assert det["yaw_source"] == "known_size_fit"
    assert det["yaw_base"] == top_face_fit(mask, depth, METAL, det["h_depth_mm"]).yaw_base
    assert axis_err_deg(det["yaw_base"], -65.0) < 3.0
    # Why the cut is not the 15 mm band of top_face_height: that keeps only the
    # high end of the top, the rectangle comes out nearly square, and its yaw
    # folds onto the short side.
    band = top_face_mask(depth, mask, INTR, T, PLANE, det["h_depth_mm"], TOP_FACE_BAND_MM)
    old = _metric_edge_yaw_base(band, INTR, T, PLANE, det["h_depth_mm"])
    assert axis_err_deg(old["yaw_base"], -65.0) > 45.0


SINGLES = [  # class, size, card centre, yaw_deg
    ("carton", LARGE_CARTON, (575.0, 140.0), 25.0),
    ("carton", SMALL_CARTON, (625.0, -160.0), -40.0),
    ("plastic_box", PLASTIC, (575.0, -100.0), 65.0),
    ("wood_box", WOOD, CARD_7, 20.0),
    ("metal_box", METAL, (625.0, 20.0), -50.0),
    ("inox_box", INOX, (575.0, 20.0), 10.0),
]


class TestSingleLayerWholeMask:
    """Single parts: the whole mask, at the trusted top face. Its rectangle is
    recorded exactly as before; the grasp yaw and pixel, which were that rectangle's
    yaw and the mask centroid, are now the known-size fit on the same mask, read
    down to the table (no support)."""

    def _check(self, mask, depth, cls):
        det = extract(mask, depth, cls)
        assert det["yaw_top_height_mm"] == det["h_depth_mm"]
        before = _metric_edge_yaw_base(mask, INTR, T, PLANE, det["h_depth_mm"])
        assert det["yaw_rect_long_mm"] == before["long_mm"]
        assert det["yaw_rect_short_mm"] == before["short_mm"]
        size = det.get("size_mm") or SIZES[cls][0]
        fit = fit_footprint(mask, INTR, T, PLANE, det["h_depth_mm"], size[:2],
                            before["yaw_base"], bottom_height_mm=0.0)
        assert det["yaw_source"] == "known_size_fit"
        assert det["fit_iou"] == fit.iou
        assert det["yaw_base"] == fit.yaw_base
        assert det["pixel_uv"] == fit.pixel_uv
        return det

    @pytest.mark.parametrize("cls,size,center,yaw", SINGLES,
                             ids=[f"{c}_{s[2]:g}" for c, s, *_ in SINGLES])
    def test_every_class(self, cls, size, center, yaw):
        labels, depth = render(((center, yaw, size, 0.0),))
        self._check(labels > 0, depth, cls)

    def test_inox_with_depth_holes(self):
        # About half the mask without depth, as on the single inox boxes of B4.
        labels, depth = render(((CARD_7, -15.0, INOX, 0.0),))
        mask = labels > 0
        vv, uu = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
        depth = depth.copy()
        depth[mask & ((vv // 16 + uu // 16) % 2 == 0)] = 0.0
        det = self._check(mask, depth, "inox_box")
        assert 0.3 <= det["depth_valid_frac"] <= 0.7

    def test_carton_without_a_trusted_size_fit(self):
        # A mask spilling onto the table spoils the size fit, the lowest carton
        # height stands in, and the 120 mm top face alone would pass the bar.
        labels, depth = render(((CARD_7, -65.0, LARGE_CARTON, 0.0),))
        det = self._check(spill(labels > 0), depth, "carton")
        assert det["size_iou"] < MIN_SIZE_IOU
        assert det["h_depth_mm"] - det["height_mm"] >= extractor().min_support_mm


def test_carton_on_plastic_with_its_mask_on_the_carton():
    # P0209 staged: plastic set at 0 deg, the 120 mm carton at -25 deg, so 90 and
    # 65 deg in the base frame. The size fit is trusted, so the support is
    # inferred; only the side faces leave the outline.
    labels, depth = stack(PLASTIC, 90.0, LARGE_CARTON, 65.0, center=CARD_13)
    mask = labels == 2
    det = extract(mask, depth, "carton")
    assert det["size_iou"] >= MIN_SIZE_IOU
    assert det["height_mm"] == LARGE_CARTON[2]
    # Was "metric_base_top_face" with the yaw of the whole mask's rectangle within
    # 1 deg: the fit with the resolved size now reads the same cut outline.
    assert det["yaw_source"] == "known_size_fit"
    assert det["yaw_base"] == top_face_fit(mask, depth, LARGE_CARTON, det["h_depth_mm"]).yaw_base
    assert axis_err_deg(det["yaw_base"], 65.0) < 1.0


def test_carton_without_a_size_fit_keeps_the_whole_mask(monkeypatch):
    # No fit at all: the height is the detector's, not the part's own, so no
    # support is inferred even on the plastic box.
    monkeypatch.setattr(postprocess, "resolve_size", lambda *args, **kwargs: None)
    labels, depth = stack(PLASTIC, 90.0, LARGE_CARTON, 65.0, center=CARD_13)
    mask = labels == 2
    det = extract(mask, depth, "carton")
    assert "size_iou" not in det
    assert det["h_depth_mm"] - det["height_mm"] >= extractor().min_support_mm
    # Without a size there is no known footprint either: no fit.
    assert "fit_iou" not in det
    assert det["yaw_source"] == "metric_base_minAreaRect"
    whole = _metric_edge_yaw_base(mask, INTR, T, PLANE, det["h_depth_mm"])
    assert det["yaw_base"] == whole["yaw_base"]


def test_top_face_mask_drops_only_pixels_with_depth_below_the_band():
    labels, depth = stack(PLASTIC, 90.0, INOX, -15.0)
    mask = labels > 0
    depth = depth.copy()
    blind = (labels == 1) & (np.arange(SHAPE[1])[None, :] % 2 == 0)
    depth[blind] = 0.0                                         # plastic without depth
    hgt = heights(depth)
    top_mm = PLASTIC[2] + INOX[2]
    out = top_face_mask(depth, mask, INTR, T, PLANE, top_mm)
    assert out.dtype == np.uint8 and not out[~mask].any()
    assert out[blind].all()
    assert not out[(labels == 1) & ~blind].any()
    assert out[(labels == 2) & (hgt > top_mm - 14.0)].all()
    assert not out[(labels == 2) & (hgt < top_mm - 16.0)].any()
