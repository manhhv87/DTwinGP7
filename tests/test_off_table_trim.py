"""The outline and the known-size fit read the mask less its pixels at table height
(depth_modes.off_table_mask); a detection over the share limit is kept when what is left
still holds the class's top (Orchestrator._mask_off_part)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.orchestrator.orchestrator import Orchestrator
from src.perception.depth_modes import (
    TablePlane,
    off_table_mask,
    table_share,
)
from src.perception.postprocess import PoseExtractor, _metric_edge_yaw_base
from tests.test_known_size_fit import corner_spill, end_spill, xy_err
from tests.test_stacked_yaw import (
    HEIGHTS,
    INOX,
    INTR,
    LARGE_CARTON,
    METAL,
    PLANE,
    PLASTIC,
    REAL,
    SMALL_CARTON,
    T,
    WOOD,
    extract,
    heights,
    render,
)

MODES = ["rgbd", "plane", "fusion"]
FIX = Path(__file__).resolve().parent / "fixtures" / "spill_20261023"
SIZES = REAL["class_sizes_mm"]


def _orch(**cfg):
    o = Orchestrator.__new__(Orchestrator)
    o.config = {"class_sizes_mm": SIZES, "class_heights_mm": HEIGHTS, **cfg}
    return o


class TestOffTableMask:
    def test_it_removes_exactly_the_pixels_table_share_counts(self):
        labels, depth = render((((575.0, 20.0), 25.0, INOX, 0.0),))
        mask = end_spill(labels > 0, (575.0, 20.0), 25.0, INOX)
        kept = off_table_mask(depth, mask, INTR, T, PLANE)
        n = int(np.count_nonzero(mask))
        assert n - int(np.count_nonzero(kept)) == round(table_share(depth, mask, INTR, T, PLANE) * n)
        assert not np.any((kept > 0) & ~mask)

    @pytest.mark.parametrize("size", [LARGE_CARTON, SMALL_CARTON, PLASTIC, WOOD, METAL, INOX])
    def test_a_part_keeps_everything_above_the_table_band(self, size):
        labels, depth = render((((450.0, 0.0), -50.0, size, 0.0),))
        mask = labels > 0
        kept = off_table_mask(depth, mask, INTR, T, PLANE) > 0
        h = heights(depth)
        assert np.all(kept[mask & (h >= 5.0)])
        # what goes is the foot of the side faces the camera sees, nothing of the top
        assert np.count_nonzero(mask & ~kept) < 0.02 * np.count_nonzero(mask)

    def test_a_stacked_part_and_its_support_lose_nothing(self):
        labels, depth = render((((525.0, -40.0), 90.0, PLASTIC, 0.0),
                                ((525.0, -40.0), 65.0, INOX, PLASTIC[2])))
        mask = labels == 2
        assert np.array_equal(off_table_mask(depth, mask, INTR, T, PLANE) > 0, mask)

    def test_a_mirror_top_keeps_its_outline(self):
        labels, depth = render((((575.0, 20.0), 25.0, INOX, 0.0),))
        mask = labels > 0
        depth = depth.copy()
        vv, uu = np.nonzero(mask)
        depth[vv, uu] = 0.0                       # no return over most of the top
        deep = vv % 3 == 0
        depth[vv[deep], uu[deep]] = 1.3           # reflections far below the table
        assert np.array_equal(off_table_mask(depth, mask, INTR, T, PLANE) > 0, mask)

    def test_no_depth_and_empty_mask(self):
        labels, _ = render((((575.0, 20.0), 25.0, INOX, 0.0),))
        none = np.zeros(labels.shape, np.float32)
        assert np.array_equal(off_table_mask(none, labels > 0, INTR, T, PLANE) > 0, labels > 0)
        assert not off_table_mask(none, np.zeros(labels.shape, bool), INTR, T, PLANE).any()


class TestSpillOverTheCards:
    @pytest.mark.parametrize("mode", MODES)
    def test_an_end_spill_no_longer_lengthens_the_rectangle(self, mode):
        """22 Oct 2026 (kiemtra-2210, trial 3) and 23 Oct (e2chuan-1, P0046, P0079): the
        mask ran about 50 mm past the box's end over the cards."""
        center, yaw = (575.0, 20.0), 25.0
        labels, depth = render(((center, yaw, INOX, 0.0),))
        mask = end_spill(labels > 0, center, yaw, INOX)
        whole = _metric_edge_yaw_base(mask, INTR, T, PLANE, INOX[2])
        assert whole["long_mm"] > 220.0                 # refused before the change
        det = extract(mask, depth, "inox_box", mode)
        assert abs(det["yaw_rect_long_mm"] - 180.0) < 10.0
        assert det["fit_plateau_mm"] <= 10.0
        assert _orch()._shape_contradicts_class(det) is None
        assert xy_err(det, center) < 3.0

    def test_a_thin_protrusion_over_the_table_goes(self):
        """23 Oct 2026 (P0056, P0080): a thin arm of the mask over the table lengthened
        the rectangle to 220 to 237 mm while the grasp sat within 2 mm of the part."""
        center, yaw = (450.0, 0.0), -50.0
        labels, depth = render(((center, yaw, INOX, 0.0),))
        mask = corner_spill(labels > 0, center, yaw, INOX, semi=(45.0, 3.0), reach=40.0, angle_deg=0.0)
        assert _metric_edge_yaw_base(mask, INTR, T, PLANE, INOX[2])["long_mm"] > 220.0
        det = extract(mask, depth, "inox_box")
        assert _orch()._shape_contradicts_class(det) is None
        assert xy_err(det, center) < 3.0

    def test_a_spill_that_returns_no_depth_is_still_refused(self):
        """Where the spill returns no depth there is nothing to remove: the long-side
        guard refuses as before."""
        center, yaw = (575.0, 20.0), 25.0
        labels, depth = render(((center, yaw, INOX, 0.0),))
        mask = end_spill(labels > 0, center, yaw, INOX)
        depth = depth.copy()
        depth[mask & (labels == 0)] = 0.0
        det = extract(mask, depth, "inox_box")
        assert det["part_mask_area"] == det["mask_area"]
        assert _orch()._shape_contradicts_class(det) is not None

    @pytest.mark.parametrize("center,yaw", [((575.0, 20.0), 25.0), ((450.0, 0.0), -50.0)])
    def test_the_three_modes_get_the_same_pixel_yaw_and_verdicts(self, center, yaw):
        labels, depth = render(((center, yaw, INOX, 0.0),))
        mask = end_spill(labels > 0, center, yaw, INOX)
        dets = [extract(mask, depth, "inox_box", m) for m in MODES]
        for d in dets[1:]:
            for key in ("pixel_uv", "yaw_base", "fit_iou", "fit_plateau_mm", "yaw_rect_long_mm",
                        "part_mask_area", "table_share", "yaw_source"):
                assert d[key] == dets[0][key]
            assert _orch()._mask_off_part(d) == _orch()._mask_off_part(dets[0])
            assert _orch()._shape_contradicts_class(d) == _orch()._shape_contradicts_class(dets[0])


class TestDropRule:
    @staticmethod
    def _det(share, source="metric_base_minAreaRect"):
        return {"class_name": "inox_box", "table_share": share, "yaw_source": source}

    def test_a_spill_whose_rest_holds_the_part_is_kept(self):
        """23 Oct 2026, P0038 and P0096: the inox mask covered a row of cards, 0.38 to
        0.48 of it at table height, and the part was never picked."""
        assert _orch()._mask_off_part(self._det(0.44, "known_size_fit")) is None

    def test_a_piece_of_a_part_over_the_limit_stays_dropped(self):
        """23 Oct 2026, P0029: a second inox mask over half the box and the cards; what
        is left measures 130 mm and would be refused if it were chosen."""
        det = dict(self._det(0.55, "known_size_fit"), yaw_rect_long_mm=130.0, fit_plateau_mm=2.0)
        assert _orch()._mask_off_part(det) is not None

    def test_a_flat_detection_on_the_cards_is_still_dropped(self):
        reason = _orch()._mask_off_part(self._det(0.97))
        assert reason is not None and reason.startswith("on the table")

    def test_below_the_limit_nothing_changes(self):
        assert _orch()._mask_off_part(self._det(0.2)) is None

    def test_flat_card_detection_end_to_end(self):
        """26 Oct 2026, Trial 6: inox detections on the cards read 1.00 at table height."""
        labels, depth = render(())                 # the bare table and its cards
        cards = end_spill(np.zeros(labels.shape, bool), (575.0, 20.0), 25.0, INOX, beyond=120.0)
        det = extract(cards, depth, "inox_box")
        assert det["table_share"] > 0.99 and det["part_mask_area"] < 0.01 * det["mask_area"]
        assert det["yaw_source"] != "known_size_fit"
        assert _orch()._mask_off_part(det) is not None


def _render_case(case, frame):
    """Depth of a 23 Oct frame, rendered: the table plane, and the part's box at its
    edge-fitted pose (the cards are flat on the table)."""
    K = case["intrinsics"]
    Tm = np.asarray(frame["T_BC"], float)
    plane = TablePlane.from_dict(frame["table_plane"])
    h, w = 720, 1280
    vv, uu = np.mgrid[0:h, 0:w]
    r = np.stack([(uu - K["ppx"]) / K["fx"], (vv - K["ppy"]) / K["fy"], np.ones((h, w))], -1) @ Tm[:3, :3].T
    t = Tm[:3, 3]
    depth = (plane.a * t[0] + plane.b * t[1] + plane.c - t[2]) / (r[..., 2] - plane.a * r[..., 0] - plane.b * r[..., 1])
    (cx, cy), yaw = case["part"]["center_xy"], math.radians(case["part"]["yaw_deg"])
    L, W, H = case["part"]["size_mm"]
    n = plane.normal
    ex = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    ex -= ex.dot(n) * n
    ex /= np.linalg.norm(ex)
    M = np.stack([ex, np.cross(n, ex), n])
    o0 = np.array([cx, cy, plane.z_at(cx, cy)])
    rl = r @ M.T
    rl = np.where(np.abs(rl) < 1e-12, 1e-12, rl)
    o = M @ (t - o0)
    lo, hi = np.array([-L / 2, -W / 2, 0.0]), np.array([L / 2, W / 2, H])
    t1, t2 = (lo - o) / rl, (hi - o) / rl
    tmin, tmax = np.max(np.minimum(t1, t2), -1), np.min(np.maximum(t1, t2), -1)
    hit = (tmax >= tmin) & (tmin > 0) & (tmin < depth)
    depth[hit] = tmin[hit]
    return K, Tm, plane, (depth / 1000.0).astype(np.float32)


@pytest.mark.parametrize("name", ["b010_t4", "b004_t21", "b008_t6", "b006_t13"])
@pytest.mark.parametrize("mode", MODES)
def test_the_masks_of_23_october_now_grasp_the_part(name, mode):
    """The detector's own masks of four refused or missed trials of 23 Oct 2026, on
    rendered depth (that day's depth was not saved): the part is grasped within 5 mm
    of its edge-fitted centre, with the jaws clear of it."""
    cases = json.loads((FIX / "cases.json").read_text(encoding="utf-8"))
    case, frame = cases[name], cases["_frame"]
    z = np.load(FIX / f"{name}.npz")
    mask = np.unpackbits(z["mask"], axis=-1)[:, :int(z["shape"][1])].astype(np.uint8)
    K, Tm, plane, depth = _render_case(case, frame)
    ex = PoseExtractor(K, depth_mode=mode, T_BC_mm=Tm, table_plane=plane,
                       class_heights_mm=HEIGHTS, class_sizes_mm=SIZES)
    det = ex.extract({"class_name": "inox_box", "mask": mask, "confidence": 0.9}, depth)
    o = _orch(table_share_max=0.3)
    assert o._mask_off_part(det) is None
    assert o._shape_contradicts_class(det) is None
    top = case["part"]["size_mm"][2]
    from src.perception.depth_modes import plane_homography
    Hm = plane_homography(K, Tm, plane, top)
    q = np.linalg.inv(Hm) @ np.array([*det["pixel_uv"], 1.0])
    g = q[:2] / q[2]
    yaw = math.radians(case["part"]["yaw_deg"])
    e = np.array([math.cos(yaw), math.sin(yaw)])
    off = g - np.asarray(case["part"]["center_xy"])
    assert abs(off @ e) < 5.0 and abs(off @ np.array([-e[1], e[0]])) < 5.0
    # jaw margin: half the 216 mm opening less the part's farthest corner along the jaw axis
    eg = np.array([math.cos(det["yaw_base"]), math.sin(det["yaw_base"])])
    c = np.asarray(case["part"]["center_xy"])
    f = np.array([-e[1], e[0]])
    corners = [c + sx * 90.0 * e + sy * 42.5 * f for sx in (-1, 1) for sy in (-1, 1)]
    assert 108.0 - max(abs((p - g) @ eg) for p in corners) > 10.0
