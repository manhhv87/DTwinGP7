"""
depth_modes.py
──────────────
Contribution C4: lift a 2D detection to 3D with or without reading depth.

The three modes return the camera-frame depth (metres) at the mask centroid, so
each drops into deproject_pixel() in place of masked_depth():

    rgbd    median depth over the mask's valid pixels (the baseline);
    plane   the viewing ray through the centroid meets the measured table plane
            lifted by the part height. No depth is read, so dropouts on a mirror
            finish cannot corrupt it; it assumes a single layer of parts;
    fusion  the plane estimate is the prior. The median depth replaces it only
            when enough of the mask returns depth and that depth disagrees with
            the prior by more than a tolerance while staying physically possible,
            i.e. not below the lowest height the part can stand at. A part resting
            on another part, or a carton of the other size, is corrected this way;
            a depth that sinks the top face into the table is dropout or edge
            bleed, and the prior stands.

One class, the carton, covers two sizes. The size is resolved per instance from
the mask alone: each candidate box is stood on the table plane at the mask's yaw,
projected into the image, and the candidate whose silhouette overlaps the mask
best (IoU) is kept. The silhouette includes the side faces, so the comparison
holds away from the image centre, where perspective widens the mask.

Frames: T_BC maps camera to base coordinates (mm), and the table plane
z = a*x + b*y + c is expressed in that same frame, which is how
02_run_calibration.py measures it (through T_BC).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

DEPTH_MODES = ("rgbd", "plane", "fusion")

# Fusion: share of mask pixels that must return depth before the depth may
# override the plane prior, and the disagreement below which the prior stands.
# 20 mm is under the lowest part (30 mm), so any stacking exceeds it, and half
# the 40 mm between the two carton sizes.
MIN_VALID_FRAC = 0.3
DISAGREE_TOL_MM = 20.0
# Measuring the top face from depth: the mask is eroded by this many pixels so its
# edge, where depth bleeds onto the table, cannot pull the estimate down; the top
# face is the cluster of heights within TOP_FACE_BAND_MM of the 90th percentile, and
# that cluster must hold this share of the valid pixels to be believed. Taking the
# cluster's median rather than an upper percentile keeps the estimate off the depth
# noise, worth about half a millimetre on this camera. A plain median over the whole
# mask is not used: it also covers the side faces the camera sees, which drags it
# down once they take up much of the mask.
TOP_FACE_ERODE_PX = 7
TOP_FACE_BAND_MM = 15.0
MIN_TOP_FACE_FRAC = 0.15
# A depth return within this many millimetres of the table plane counts as table
# in table_share and off_table_mask.
TABLE_SHARE_TOL_MM = 5.0
# Below this silhouette IoU the size decision is not trusted: on the labelled set a
# correct decision scores 0.96 at the median and a wrong one 0.66, and every wrong
# one had another part hiding a piece of this one. An occluded part is then stood at
# its lowest candidate height, because assuming a part is taller than it is puts the
# jaws above it, which never grips.
MIN_SIZE_IOU = 0.80
# A rectangle whose long side is shorter than this many times its short side is
# treated as square: its yaw is folded modulo 90 deg. PoseExtractor's minimum-area
# rectangle (_metric_edge_yaw_base) and fit_footprint share it.
NEAR_SQUARE_RATIO = 1.15
# fit_footprint: the known-size fit replaces the mask centroid and the rectangle yaw
# only from this IoU up (see PoseExtractor.extract for the measurements behind it).
MIN_FOOTPRINT_IOU = 0.5

Size = tuple[float, float, float]


def parse_class_sizes(raw: Mapping[str, Any] | None) -> dict[str, list[Size]]:
    """Normalise {class: [L, W, H] or [[L, W, H], ...]} (mm) to lists of triples.

    Raises:
        ValueError: an entry is not one or more positive (L, W, H) triples.
    """
    out: dict[str, list[Size]] = {}
    for cls, v in (raw or {}).items():
        try:
            arr = np.asarray(v, dtype=float)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{cls}: expected [L, W, H] or a list of them, got {v!r}") from e
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
            raise ValueError(f"{cls}: expected [L, W, H] or a list of them, got {v!r}")
        if np.any(arr <= 0):
            raise ValueError(f"{cls}: sizes must be positive, got {v!r}")
        out[str(cls)] = [(float(r[0]), float(r[1]), float(r[2])) for r in arr]
    return out


@dataclass(frozen=True)
class TablePlane:
    """Table top z = a*x + b*y + c in the T_BC (base) frame, mm."""

    a: float
    b: float
    c: float

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TablePlane":
        """From the content of table_plane.json (keys a, b, c)."""
        return cls(float(d["a"]), float(d["b"]), float(d["c"]))

    @property
    def normal(self) -> np.ndarray:
        """Unit normal, pointing up."""
        n = np.array([-self.a, -self.b, 1.0])
        return n / np.linalg.norm(n)

    def z_at(self, x: Any, y: Any) -> Any:
        return self.a * np.asarray(x, float) + self.b * np.asarray(y, float) + self.c

    def height_above(self, p_xyz: Any) -> Any:
        """Distance of point(s) (..., 3) above the plane, along its normal (mm)."""
        p = np.asarray(p_xyz, float)
        return (p[..., 2] - self.z_at(p[..., 0], p[..., 1])) * self.normal[2]


def _ray(intrinsics: Mapping[str, float], u: float, v: float,
         T_BC_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Origin and direction (base frame) of the viewing ray through pixel (u, v).

    The direction has camera-frame z = 1, so the ray parameter is the camera-frame
    depth in mm.
    """
    T = np.asarray(T_BC_mm, float)
    d = np.array([(u - intrinsics["ppx"]) / intrinsics["fx"],
                  (v - intrinsics["ppy"]) / intrinsics["fy"], 1.0])
    return T[:3, 3], T[:3, :3] @ d


def ray_plane_depth(intrinsics: Mapping[str, float], u: float, v: float,
                    T_BC_mm: np.ndarray, plane: TablePlane,
                    height_mm: float = 0.0) -> float | None:
    """Camera-frame depth (m) where the ray through (u, v) meets the plane lifted by height_mm.

    Generalises z_from_ground_plane() to a tilted table: the top face of a part of
    height h standing on the table lies on the parallel plane h above it, measured
    along the normal. None if the ray is parallel to that plane or meets it behind
    the camera.
    """
    t, r = _ray(intrinsics, u, v, T_BC_mm)
    c_top = plane.c + height_mm / plane.normal[2]
    denom = r[2] - plane.a * r[0] - plane.b * r[1]
    if abs(denom) < 1e-9:
        return None
    s = (plane.a * t[0] + plane.b * t[1] + c_top - t[2]) / denom
    if s <= 0:
        return None
    return float(s) / 1000.0


def image_yaw_to_base(yaw_img_rad: float, T_BC_mm: np.ndarray) -> float:
    """Base-frame yaw of an image-plane direction.

    Same transform as coord_conv.camera_yaw_to_base, kept here so the perception
    package does not import the orchestrator; a test pins the two together.
    """
    R = np.asarray(T_BC_mm, float)[:3, :3]
    d = R @ np.array([math.cos(yaw_img_rad), math.sin(yaw_img_rad), 0.0])
    return float(math.atan2(d[1], d[0]))


def valid_depth_fraction(depth_m: np.ndarray, mask: np.ndarray) -> float:
    """Share of mask pixels that return a depth (> 0)."""
    sel = mask > 0
    n = int(np.count_nonzero(sel))
    return float(np.count_nonzero(depth_m[sel] > 0)) / n if n else 0.0


def project_points(intrinsics: Mapping[str, float], T_BC_mm: np.ndarray,
                   P_base: np.ndarray) -> np.ndarray | None:
    """Pixel coordinates (N, 2) of base-frame points (mm); None if any lies behind the camera."""
    T_CB = np.linalg.inv(np.asarray(T_BC_mm, float))
    Pc = np.asarray(P_base, float) @ T_CB[:3, :3].T + T_CB[:3, 3]
    if np.any(Pc[:, 2] <= 1e-6):
        return None
    return np.column_stack([intrinsics["fx"] * Pc[:, 0] / Pc[:, 2] + intrinsics["ppx"],
                            intrinsics["fy"] * Pc[:, 1] / Pc[:, 2] + intrinsics["ppy"]])


def box_corners(center_xy: Sequence[float], yaw_rad: float, size_mm: Sequence[float],
                plane: TablePlane) -> np.ndarray:
    """Eight corners (base frame, mm) of a box (L, W, H) standing on the plane.

    The footprint is centred at center_xy with the L side along yaw_rad.
    """
    L, W, H = (float(x) for x in size_mm)
    ex = np.array([math.cos(yaw_rad), math.sin(yaw_rad)])
    ey = np.array([-ex[1], ex[0]])
    c = np.asarray(center_xy, float)
    foot = np.array([c + sx * L / 2.0 * ex + sy * W / 2.0 * ey
                     for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)])
    bottom = np.column_stack([foot, plane.z_at(foot[:, 0], foot[:, 1])])
    return np.vstack([bottom, bottom + H * plane.normal])


def _convex_hull(px: np.ndarray) -> np.ndarray:
    import cv2  # lazy import, as elsewhere in the perception package

    return cv2.convexHull(px.astype(np.float32)).reshape(-1, 2).astype(float)


def _polygon_centroid(poly: np.ndarray) -> tuple[float, float]:
    """Area centroid of a simple polygon (pixel coordinates)."""
    x, y = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-9:
        return float(x.mean()), float(y.mean())
    return (float(((x + x1) * cross).sum() / (6.0 * area)),
            float(((y + y1) * cross).sum() / (6.0 * area)))


def _clip_to_image(poly: np.ndarray, shape: tuple[int, int]) -> np.ndarray | None:
    """Convex polygon clipped to the image rectangle, or None if nothing is left.

    A mask is cut off by the frame edge and its centroid is cut off with it, so the
    modelled silhouette has to be cut the same way before the two are compared.
    """
    import cv2

    h, w = shape
    rect = np.array([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
                    np.float32)
    area, inter = cv2.intersectConvexConvex(poly.astype(np.float32), rect)
    if inter is None or area <= 0 or len(inter) < 3:
        return None
    return inter.reshape(-1, 2).astype(float)


_SUBPIX_BITS = 4  # fractional bits handed to cv2.fillPoly


def _silhouette_iou(poly: np.ndarray, mask: np.ndarray) -> float:
    """IoU between a filled polygon (pixel coordinates) and a binary mask."""
    import cv2

    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    x0 = max(0, int(math.floor(min(xs.min(), poly[:, 0].min()))) - 1)
    y0 = max(0, int(math.floor(min(ys.min(), poly[:, 1].min()))) - 1)
    x1 = min(w, int(math.ceil(max(xs.max(), poly[:, 0].max()))) + 2)
    y1 = min(h, int(math.ceil(max(ys.max(), poly[:, 1].max()))) + 2)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    sil = np.zeros((y1 - y0, x1 - x0), np.uint8)
    pts = np.round((poly - [x0, y0]) * (1 << _SUBPIX_BITS)).astype(np.int32)
    cv2.fillPoly(sil, [pts], 1, lineType=cv2.LINE_8, shift=_SUBPIX_BITS)
    m = mask[y0:y1, x0:x1] > 0
    s = sil > 0
    union = int(np.count_nonzero(m | s))
    return float(np.count_nonzero(m & s)) / union if union else 0.0


@dataclass(frozen=True)
class BoxFit:
    """A candidate box placed on the table so that its silhouette matches a mask."""

    size: Size                        # (L, W, H), mm
    iou: float                        # silhouette against the mask
    center_xy: tuple[float, float]    # footprint centre on the table, base frame, mm
    yaw_base: float                   # direction of the L side, base frame, rad


def fit_box(mask: np.ndarray, intrinsics: Mapping[str, float], T_BC_mm: np.ndarray,
            plane: TablePlane, size_mm: Sequence[float], yaw_base_rad: float,
            iters: int = 6, tol_px: float = 0.05) -> BoxFit | None:
    """Stand a box of size_mm on the plane so that its silhouette centroid matches
    the mask's, and score the overlap.

    The ray through the current pixel meets the plane lifted by half the box height
    to place the box centre; the pixel then moves by the offset between the two
    centroids, which settles in two or three rounds. The modelled silhouette is
    clipped to the image first, so a part cut off by the frame edge is compared
    against what the camera actually sees of it. None if the box cannot be projected.
    """
    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        return None
    mu, mv = float(xs.mean()), float(ys.mean())
    half = float(size_mm[2]) / 2.0

    def place(pu: float, pv: float):
        """(footprint centre, visible silhouette) of the box hung on the ray through (pu, pv)."""
        s = ray_plane_depth(intrinsics, pu, pv, T_BC_mm, plane, half)
        if s is None:
            return None
        t, r = _ray(intrinsics, pu, pv, T_BC_mm)
        foot = t + s * 1000.0 * r - half * plane.normal     # box centre dropped onto the table
        px = project_points(intrinsics, T_BC_mm,
                            box_corners(foot[:2], yaw_base_rad, size_mm, plane))
        if px is None:
            return None
        vis = _clip_to_image(_convex_hull(px), mask.shape)
        return None if vis is None else (foot, vis)

    pu, pv = mu, mv
    for _ in range(max(1, int(iters))):
        got = place(pu, pv)
        if got is None:
            return None
        foot, vis = got
        su, sv = _polygon_centroid(vis)
        du, dv = mu - su, mv - sv
        if math.hypot(du, dv) < tol_px:
            break
        pu, pv = pu + du, pv + dv
    else:                                    # never settled: place it at the last pixel
        got = place(pu, pv)
        if got is None:
            return None
        foot, vis = got
    return BoxFit((float(size_mm[0]), float(size_mm[1]), float(size_mm[2])),
                  _silhouette_iou(vis, mask), (float(foot[0]), float(foot[1])),
                  float(yaw_base_rad))


def resolve_size(mask: np.ndarray, intrinsics: Mapping[str, float], T_BC_mm: np.ndarray,
                 plane: TablePlane, candidates: Sequence[Sequence[float]],
                 yaw_img_rad: float) -> tuple[BoxFit, list[float]] | None:
    """Best-matching candidate size for one instance, and every candidate's IoU.

    Each candidate is tried at the mask's major-axis yaw and at right angles to it:
    away from the image centre the side faces can make a 180 x 120 mm box look
    square, and the major axis then flips.
    """
    yaw_b = image_yaw_to_base(yaw_img_rad, T_BC_mm)
    best: BoxFit | None = None
    ious: list[float] = []
    for size in candidates:
        cand: BoxFit | None = None
        for dyaw in (0.0, math.pi / 2.0):
            f = fit_box(mask, intrinsics, T_BC_mm, plane, size, yaw_b + dyaw)
            if f is not None and (cand is None or f.iou > cand.iou):
                cand = f
        ious.append(cand.iou if cand is not None else 0.0)
        if cand is not None and (best is None or cand.iou > best.iou):
            best = cand
    return None if best is None else (best, ious)


# fit_footprint search. Coarse: the yaw every 2 deg within 20 deg of the seed, over the
# whole 90 deg for a square footprint (180 deg for a near-square one), centres every
# 4 mm anywhere over the mask. Fine: the yaw every 0.25 deg within 1.5 deg of the
# coarse optimum, and further while the IoU stays near its best, centres every 2 mm
# within 4 mm of the best centre at the neighbouring yaw. Last: the centre on 1 mm
# cells within 25 mm. Each grid samples the mask averaged over blocks about one cell
# wide.
_FIT_COARSE_HALF_DEG = 20.0
_FIT_COARSE_STEP_DEG = 2.0
_FIT_COARSE_CELL_MM = 4.0
_FIT_FINE_HALF_DEG = 1.5
_FIT_FINE_STEP_DEG = 0.25
_FIT_FINE_CELL_MM = 2.0
_FIT_FINE_WINDOW_MM = 4.0
_FIT_LAST_WINDOW_MM = 25.0
# Centres whose IoU comes within this of the best one form the near-optimal set. On a
# real inox mask (21 Oct 2026, P0207) the IoU along such a set varied by 3e-4, while it
# fell by 9e-3 per mm past the set's sharp edge.
_FIT_PLATEAU_TOL = 0.001


@dataclass(frozen=True)
class FootprintFit:
    """A class's L x W top face fitted to a mask on the plane of the part's top."""

    center_xy: tuple[float, float]    # centre on the top plane, base frame, mm
    yaw_base: float                   # long side, base frame, rad, in (-pi/2, pi/2];
                                      # (-pi/4, pi/4] for a near-square footprint
    iou: float                        # rectangle against the mask on the top plane
    pixel_uv: tuple[float, float]     # the centre projected into the image
    plateau_mm: float = 0.0           # how far the centre slides along the long side
                                      # with the IoU within 0.001 of its best, mm


def plane_homography(intrinsics: Mapping[str, float], T_BC_mm: np.ndarray,
                     plane: TablePlane, height_mm: float) -> np.ndarray:
    """3x3 homography from base (x, y) on the plane lifted by height_mm to pixels.

    The lifted plane is the one ray_plane_depth intersects.
    """
    K = np.array([[intrinsics["fx"], 0.0, intrinsics["ppx"]],
                  [0.0, intrinsics["fy"], intrinsics["ppy"]], [0.0, 0.0, 1.0]])
    T_CB = np.linalg.inv(np.asarray(T_BC_mm, float))
    c_top = plane.c + float(height_mm) / plane.normal[2]
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [plane.a, plane.b, c_top]])
    return K @ (T_CB[:3, :3] @ M + np.outer(T_CB[:3, 3], [0.0, 0.0, 1.0]))


def _vertex(f0: float, f1: float, f2: float) -> tuple[float, float]:
    """Offset (in steps, within +-0.5) and height of the parabola through three
    equally spaced samples whose middle one is the largest."""
    den = f0 - 2.0 * f1 + f2
    if den >= 0.0:
        return 0.0, f1
    d = min(0.5, max(-0.5, 0.5 * (f0 - f2) / den))
    return d, f1 - 0.25 * (f0 - f2) * d


class _TopPlaneMask:
    """Readings of one mask, cropped from the image, on the top plane.

    The readings are filled 0/1 crops of one shape. Stacked as channels, one warp serves
    all of them. _level(k, i) is reading i (all of them stacked for None) averaged over
    k x k blocks, with the homography from the top plane to that averaged crop.
    """

    def __init__(self, readings: Sequence[np.ndarray], x0: int, y0: int,
                 H: np.ndarray) -> None:
        import cv2

        Hi = np.linalg.inv(H)
        self.areas = []
        q = 1.0
        for r in readings:
            mo = cv2.moments(r, True)
            if mo["m00"] < 25:
                self.areas.append(0.0)
                continue
            # A pixel (u, v) covers |det Hi| / q^3 mm^2 of the top plane, q = Hi[2].(u, v, 1).
            # q is affine in the pixel, so its mean over a reading is its value at the centroid.
            q = (Hi[2, 0] * (mo["m10"] / mo["m00"] + x0)
                 + Hi[2, 1] * (mo["m01"] / mo["m00"] + y0) + Hi[2, 2])
            self.areas.append(float(mo["m00"] * abs(np.linalg.det(Hi)) / abs(q) ** 3))
        union = np.bitwise_or.reduce(np.stack(readings), axis=0)
        outline = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        pts = np.vstack([c.reshape(-1, 2) for c in outline]).astype(float)
        p = Hi @ np.vstack([pts[:, 0] + x0, pts[:, 1] + y0, np.ones(len(pts))])
        self.ok = max(self.areas) > 0.0 and bool(np.all(p[2] * q > 0.0))  # not past the horizon
        self.outline_xy = (p[:2] / p[2]).T          # extent of the readings in any frame
        self._H = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]]) @ H
        self._readings = [r.astype(np.float32) for r in readings]
        self._cache: dict[tuple[int, int | None], tuple[np.ndarray, np.ndarray]] = {}

    def _level(self, k: int, i: int | None) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        if (k, i) not in self._cache:
            src = (np.dstack(self._readings) if i is None and len(self._readings) > 1
                   else self._readings[0 if i is None else i])
            Hs = self._H
            if k > 1:                                # exact block means: sides divide by k
                h, w = src.shape[:2]
                src = cv2.resize(src, (w // k, h // k), interpolation=cv2.INTER_AREA)
                o = (k - 1) / (2.0 * k)              # block j is centred on pixel k*j + o*k
                Hs = np.array([[1.0 / k, 0.0, -o], [0.0, 1.0 / k, -o], [0.0, 0.0, 1.0]]) @ Hs
            self._cache[(k, i)] = (src, Hs)
        return self._cache[(k, i)]

    def ious(self, theta: float, block: int, cell: float, a_rng: tuple[float, float],
             b_rng: tuple[float, float], whole: bool, L: float, W: float,
             reading: int | None = None):
        """IoU of the L x W rectangle at yaw theta for every centre of a grid.

        Centres lie every step_a along the long side (a) and step_b across it (b) from
        (a_rng[0], b_rng[0]) to past (a_rng[1], b_rng[1]); entry (r, c, i) of the map is
        the centre a_rng[0] + c * step_a, b_rng[0] + r * step_b on reading i, or on the
        one reading asked for (i = 0). The mask is resampled on a grid aligned with the
        rectangle, whose cells divide L and W exactly, and the overlap of every placement
        is a box sum over it. whole: the grid spans the window only and boxes reaching
        past it read zeros, right when the window holds the whole mask; otherwise the
        grid spans every box of the window.

        Returns (IoU map, step_a, step_b).
        """
        import cv2

        src, Hs = self._level(block, reading)
        areas = np.array(self.areas if reading is None else self.areas[reading:reading + 1])
        nL = max(1, int(round(L / cell)))
        nW = max(1, int(round(W / cell)))
        sL, sW = L / nL, W / nW
        ct, st = math.cos(theta), math.sin(theta)
        na = int(math.ceil((a_rng[1] - a_rng[0]) / sL - 1e-9)) + 1
        nb = int(math.ceil((b_rng[1] - b_rng[0]) / sW - 1e-9)) + 1
        if whole:
            # boxFilter centres a box of even size half a cell before its anchor cell
            a0 = a_rng[0] + 0.5 * sL * (1 - nL % 2)
            b0 = b_rng[0] + 0.5 * sW * (1 - nW % 2)
            ncols, nrows = na, nb
        else:
            a0 = a_rng[0] - 0.5 * (nL - 1) * sL
            b0 = b_rng[0] - 0.5 * (nW - 1) * sW
            ncols, nrows = na + nL - 1, nb + nW - 1
        G = np.array([[sL * ct, -sW * st, a0 * ct - b0 * st],
                      [sL * st, sW * ct, a0 * st + b0 * ct],
                      [0.0, 0.0, 1.0]])                       # grid (col, row) -> base (x, y)
        grid = cv2.warpPerspective(src, Hs @ G, (ncols, nrows),
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if whole:
            O = cv2.boxFilter(grid, -1, (nL, nW), normalize=False,
                              borderType=cv2.BORDER_CONSTANT)
        else:
            S = cv2.integral(grid)
            O = S[nW:, nL:] - S[:-nW, nL:] - S[nW:, :-nL] + S[:-nW, :-nL]
        O = O.reshape(O.shape[0], O.shape[1], -1) * (sL * sW)
        return O / (areas + L * W - O), sL, sW

    def best(self, theta: float, block: int, cell: float, a_rng: tuple[float, float],
             b_rng: tuple[float, float], whole: bool, L: float, W: float,
             reading: int | None = None) -> list[tuple[float, np.ndarray]]:
        """Per reading (or for the one asked for), the largest IoU over the grid of ious,
        refined by a parabola in each axis, and its centre (base x, y)."""
        iou, sL, sW = self.ious(theta, block, cell, a_rng, b_rng, whole, L, W, reading)
        ct, st = math.cos(theta), math.sin(theta)
        out = []
        for i in range(iou.shape[2]):
            m = iou[:, :, i]
            r, c = divmod(int(np.argmax(m)), m.shape[1])
            v = float(m[r, c])
            dr = dc = gain = 0.0
            if 0 < r < m.shape[0] - 1:
                dr, vr = _vertex(float(m[r - 1, c]), v, float(m[r + 1, c]))
                gain += vr - v
            if 0 < c < m.shape[1] - 1:
                dc, vc = _vertex(float(m[r, c - 1]), v, float(m[r, c + 1]))
                gain += vc - v
            a, b = a_rng[0] + (c + dc) * sL, b_rng[0] + (r + dr) * sW
            out.append((v + gain, np.array([a * ct - b * st, a * st + b * ct])))
        return out


def _best_yaw(thetas: np.ndarray, vals: Sequence[float], step: float,
              cyclic: bool) -> tuple[float, int]:
    """Yaw of the largest of vals, refined by a parabola, and its index."""
    j = int(np.argmax(vals))
    th = float(thetas[j])
    if cyclic:                                  # the yaws close on themselves
        th += _vertex(vals[j - 1], vals[j], vals[(j + 1) % len(vals)])[0] * step
    elif 0 < j < len(vals) - 1:
        th += _vertex(vals[j - 1], vals[j], vals[j + 1])[0] * step
    return th, j


def fit_footprint(mask: np.ndarray, intrinsics: Mapping[str, float], T_BC_mm: np.ndarray,
                  plane: TablePlane, top_height_mm: float, footprint_mm: Sequence[float],
                  seed_yaw_rad: float,
                  bottom_height_mm: float | None = None) -> FootprintFit | None:
    """The L x W rectangle that overlaps the mask best (IoU), on the plane of the top.

    The mask's largest outer outline, filled, is read on the table plane lifted by
    top_height_mm, through the same pinhole and T_BC as ray_plane_depth. The rectangle
    of the class's known top size (footprint_mm, its first two values) is placed at
    every centre of a grid and at a series of yaws, and the placement with the largest
    IoU is kept. The yaw is searched within 20 deg of seed_yaw_rad (the rectangle yaw of
    the outline, _metric_edge_yaw_base), over the whole 90 deg for a square footprint
    and 180 deg for a near-square one (L/W under NEAR_SQUARE_RATIO). It returns the long
    side's yaw in (-pi/2, pi/2], folded into (-pi/4, pi/4] for a near-square footprint,
    the convention of _metric_edge_yaw_base.

    Why: a mask that spills off the part pulls its centroid and turns its minimum-area
    rectangle. On 21 Oct 2026 (kiemtra-2110, trial 3, pose P0216) the detector's mask of
    the mirror-finish inox box (180 x 85 x 40 mm) also covered the paper position cards
    beside it; the rectangle measured 212.5 x 130.5 mm, its yaw was 6.3 deg off and the
    centroid 10.5 mm along and 15.3 mm across the long side off an edge-fitted reference,
    and one jaw came down on the box corner. A rectangle of the known size cannot follow
    the spill: on that frame this fit is 1.6 deg and 1.1 mm along off the reference.

    The visible side faces. Seen from above and off the camera's nadir, a box shows the
    side faces that face the nadir; the detector's masks cover them (0.8 to 1.0 of their
    area on 22 of the 24 real detections where they show, 20 and 21 Oct 2026), and on
    the top plane they form a band beyond the top face on the nadir side, about as wide
    as the distance from the nadir times the part height over the camera height. Inside
    a mask that long the rectangle slides along the band with almost no change of IoU.
    With bottom_height_mm, the height of the part's bottom above the table plane, the
    mask is also read on that plane, where the band falls beyond the far side of the
    bottom face instead, and the two readings are intersected: for a box this keeps
    exactly its footprint. A mask that leaves the side faces out (a metal box on 21 Oct,
    0.09 of its band covered) loses a strip of its top face that way, so the
    intersection and the plain reading are both searched, and the one whose best
    rectangle overlaps it more is kept. On ray-cast boxes of every class up to 262 mm
    from the nadir the fit stayed within 0.2 deg and 0.9 mm; the plain reading alone was
    up to 1.8 deg and 6.0 mm off.

    Where the IoU still stays flat, within 0.001 of its best, over a set of centres or
    a range of yaws (a mask larger than the part, as with a spill, lets the rectangle
    slide and turn inside it: over 4.75 to 5.75 deg of yaw on the spilled inox masks
    of P0213 and P0216, 21 Oct), the centroid of the set and the middle of the range
    are returned rather than an arbitrary member of them. The extent of that set of
    centres along the long side is returned as plateau_mm (searched within 25 mm of
    the best centre, so it saturates at 50 mm): a long plateau means the mask does not
    pin the rectangle down along the axis the jaws close on.

    Returns None when the mask is empty or the plane cannot be projected.
    """
    import cv2

    L, W = sorted((float(x) for x in footprint_mm[:2]), reverse=True)
    if not W > 0.0:
        return None
    near_square = L / W < NEAR_SQUARE_RATIO
    m = np.asarray(mask)
    m8 = (m.view(np.uint8) if m.dtype == np.bool_
          else m if m.dtype == np.uint8 else (m > 0).astype(np.uint8))
    bx, by, bw, bh = cv2.boundingRect(m8)
    if bw == 0 or bh == 0:
        return None
    sub = (m8[by:by + bh, bx:bx + bw] > 0).astype(np.uint8)
    contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    H = plane_homography(intrinsics, T_BC_mm, plane, top_height_mm)
    Hi = np.linalg.inv(H)
    cx, cy, w, h = cv2.boundingRect(contour)
    # Blocks about one cell wide, from the size of a pixel on the top plane.
    q = float(Hi[2] @ np.array([bx + cx + 0.5 * w, by + cy + 0.5 * h, 1.0]))
    if q == 0.0:
        return None
    px_mm = math.sqrt(abs(np.linalg.det(Hi)) / abs(q) ** 3)
    kc = max(1, int(round(_FIT_COARSE_CELL_MM / px_mm)))
    kf = max(1, int(round(_FIT_FINE_CELL_MM / px_mm)))
    k_all = kc * kf // math.gcd(kc, kf)
    x0, y0 = bx + cx - 2, by + cy - 2
    w = int(math.ceil((w + 4) / k_all)) * k_all
    h = int(math.ceil((h + 4) / k_all)) * k_all
    crop = np.zeros((h, w), np.uint8)
    cv2.drawContours(crop, [contour - [cx - 2, cy - 2]], -1, 1, thickness=cv2.FILLED)
    readings = [crop]
    if bottom_height_mm is not None:
        Tc = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
        Hb = plane_homography(intrinsics, T_BC_mm, plane, bottom_height_mm)
        below = cv2.warpPerspective(crop, Tc @ Hb @ Hi @ np.linalg.inv(Tc), (w, h),
                                    flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        readings.append(crop & below)
    top = _TopPlaneMask(readings, x0, y0, H)
    if not top.ok:
        return None
    use = [i for i, a in enumerate(top.areas) if a > 0.0]

    # Coarse: the yaw over its window, centres over the whole mask, every reading.
    step = math.radians(_FIT_COARSE_STEP_DEG)
    if near_square:
        span = math.pi / 2.0 if L - W < 1e-9 else math.pi
        thetas = -0.5 * span + step * np.arange(int(round(span / step)))
    else:
        n = int(round(_FIT_COARSE_HALF_DEG / _FIT_COARSE_STEP_DEG))
        thetas = float(seed_yaw_rad) + step * np.arange(-n, n + 1)
    P = top.outline_xy
    coarse = []
    for th in thetas:
        ct, st = math.cos(th), math.sin(th)
        a = P[:, 0] * ct + P[:, 1] * st
        b = P[:, 1] * ct - P[:, 0] * st
        coarse.append(top.best(th, kc, _FIT_COARSE_CELL_MM, (a.min(), a.max()),
                               (b.min(), b.max()), True, L, W))

    # Fine, per reading: the yaw every 0.25 deg near its coarse optimum, centres near
    # the best centre of the neighbouring yaw. A spill lets the rectangle turn inside
    # the mask as well: on 21 Oct 2026 the IoU stayed within 0.001 of its best over 4.75
    # to 5.75 deg of yaw on the spilled inox masks of P0213 and P0216. While the run of
    # yaws within _FIT_PLATEAU_TOL of the best reaches an end of the samples, they are
    # extended (up to 20 deg), and the middle of the run is taken; a single best sample
    # is refined by a parabola.
    n = int(round(_FIT_FINE_HALF_DEG / _FIT_FINE_STEP_DEG))
    n_max = int(round(_FIT_COARSE_HALF_DEG / _FIT_FINE_STEP_DEG))
    fstep = math.radians(_FIT_FINE_STEP_DEG)
    R = _FIT_FINE_WINDOW_MM

    def fine_at(i: int, th: float, cxy: np.ndarray) -> tuple[float, np.ndarray]:
        ct, st = math.cos(th), math.sin(th)
        a, b = cxy[0] * ct + cxy[1] * st, cxy[1] * ct - cxy[0] * st
        return top.best(th, kf, _FIT_FINE_CELL_MM, (a - R, a + R), (b - R, b + R),
                        False, L, W, i)[0]

    best = None
    for i in use:
        th_c, j = _best_yaw(thetas, [c[i][0] for c in coarse], step, near_square)
        fine = {k: fine_at(i, th_c + k * fstep, coarse[j][i][1]) for k in range(-n, n + 1)}
        lo, hi = -n, n
        while True:
            v = max(f[0] for f in fine.values())
            if fine[lo][0] >= v - _FIT_PLATEAU_TOL and lo > -n_max:
                lo -= 1
                fine[lo] = fine_at(i, th_c + lo * fstep, fine[lo + 1][1])
            elif fine[hi][0] >= v - _FIT_PLATEAU_TOL and hi < n_max:
                hi += 1
                fine[hi] = fine_at(i, th_c + hi * fstep, fine[hi - 1][1])
            else:
                break
        kb = max(fine, key=lambda k: fine[k][0])
        v = fine[kb][0]
        k0 = k1 = kb
        while k0 > lo and fine[k0 - 1][0] >= v - _FIT_PLATEAU_TOL:
            k0 -= 1
        while k1 < hi and fine[k1 + 1][0] >= v - _FIT_PLATEAU_TOL:
            k1 += 1
        if k0 == k1 and lo < kb < hi:
            mid = kb + _vertex(fine[kb - 1][0], v, fine[kb + 1][0])[0]
        else:
            mid = 0.5 * (k0 + k1)
        if best is None or v > best[0]:
            best = (v, i, th_c + mid * fstep, fine[int(round(mid))][1])
    _, i, th, cxy = best

    # Last: the centre at that yaw, the centroid of the near-optimal set around the best.
    ct, st = math.cos(th), math.sin(th)
    a, b = cxy[0] * ct + cxy[1] * st, cxy[1] * ct - cxy[0] * st
    R = _FIT_LAST_WINDOW_MM
    iou, sL, sW = top.ious(th, 1, 1.0, (a - R, a + R), (b - R, b + R), False, L, W, i)
    iou = iou[:, :, 0]
    r0, c0 = divmod(int(np.argmax(iou)), iou.shape[1])
    v = float(iou[r0, c0])
    _, lab = cv2.connectedComponents((iou >= v - _FIT_PLATEAU_TOL).astype(np.uint8),
                                     connectivity=8)
    rr, kk = np.nonzero(lab == lab[r0, c0])
    plateau_mm = float(kk.max() - kk.min()) * sL
    a, b = a - R + float(kk.mean()) * sL, b - R + float(rr.mean()) * sW
    x, y = a * ct - b * st, a * st + b * ct

    yaw = (th + math.pi / 2.0) % math.pi - math.pi / 2.0
    if yaw <= -math.pi / 2.0:
        yaw += math.pi
    if near_square:
        yaw = (yaw + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0
        if yaw <= -math.pi / 4.0:
            yaw += math.pi / 2.0
    p = H @ np.array([x, y, 1.0])
    return FootprintFit((float(x), float(y)), float(yaw), v,
                        (float(p[0] / p[2]), float(p[1] / p[2])), plateau_mm)


def top_face_height(depth_m: np.ndarray, mask: np.ndarray, intrinsics: Mapping[str, float],
                    T_BC_mm: np.ndarray, plane: TablePlane,
                    band_mm: float = TOP_FACE_BAND_MM,
                    min_frac: float = MIN_TOP_FACE_FRAC) -> float | None:
    """Height of the part's top face above the table (mm), measured from depth.

    None when too little of the mask returns depth, or when the depths do not form a
    top face at all: on a mirror finish the returns scatter, and a scattered cloud
    must not be allowed to override the plane prior.
    """
    import cv2

    m = cv2.erode((mask > 0).astype(np.uint8), np.ones((TOP_FACE_ERODE_PX,) * 2, np.uint8))
    vv, uu = np.nonzero((m > 0) & (depth_m > 0))
    if uu.size < 200:
        return None
    z = depth_m[vv, uu] * 1000.0
    x = (uu - intrinsics["ppx"]) * z / intrinsics["fx"]
    y = (vv - intrinsics["ppy"]) * z / intrinsics["fy"]
    T = np.asarray(T_BC_mm, float)
    h = plane.height_above(np.column_stack([x, y, z]) @ T[:3, :3].T + T[:3, 3])
    face = h[h > np.percentile(h, 90) - band_mm]
    if face.size < min_frac * h.size:
        return None
    return float(np.median(face))


def top_face_mask(depth_m: np.ndarray, mask: np.ndarray, intrinsics: Mapping[str, float],
                  T_BC_mm: np.ndarray, plane: TablePlane, top_mm: float,
                  band_mm: float = TOP_FACE_BAND_MM) -> np.ndarray:
    """The mask less the pixels whose depth puts them below the top face, as 0/1 uint8.

    A pixel is dropped when it returns depth and that depth places it less than
    top_mm - band_mm above the table; pixels without depth stay. The default
    band_mm is the band top_face_height takes as the top face. The deprojection is
    that of top_face_height, without its erosion: here the outline itself is
    wanted.

    It serves the yaw of a part standing on a support. On 20 Oct 2026 (B4, trial
    14, pose P0214) the mask of an inox box resting on the plastic box also covered
    part of the plastic. The rectangle fitted to the whole mask measured 194.0 x
    134.6 mm, against 188.0 to 193.4 x 90.6 to 109.7 mm for six of the seven single
    inox boxes of that run, and its long side lay about 12 deg off the yaw the pose
    list called for the inox; the jaws came down misaligned and the cycle failed.

    PoseExtractor cuts half a part height below the top face (band_mm = height / 2).
    On a flat stack the support's top lies at about top_mm - height, half a part
    height below the cut, so it is removed. A part leaning with one end on a
    support has a tilted top that falls from its high end by the support's height,
    so most of it stays above the cut, all of it while the support is lower than
    about half the part, and the outline stays the part's. The default 15 mm band
    keeps only the high end of such a top, and the rectangle turns: on the C4
    labelled dataset (1512 real images, the deployed detector), 37 of 151 leaning
    stacked instances came out more than 10 deg off with that band, against 1 with
    the whole mask. With the half-height cut 2 of the 151 did, and 9 of all 648
    matched stacked instances against 12 with the whole mask. A support taller than
    half the part tilts the top past the cut and its low end is lost: in a ray-cast
    metal box with one end on an 85 mm box the rectangle turned 90 deg in 2 of 3
    poses, where the whole mask stayed within 3 deg.
    """
    m = np.asarray(mask) > 0
    out = m.astype(np.uint8)
    vv, uu = np.nonzero(m & (depth_m > 0))
    if uu.size == 0:
        return out
    z = depth_m[vv, uu] * 1000.0
    x = (uu - intrinsics["ppx"]) * z / intrinsics["fx"]
    y = (vv - intrinsics["ppy"]) * z / intrinsics["fy"]
    T = np.asarray(T_BC_mm, float)
    h = plane.height_above(np.column_stack([x, y, z]) @ T[:3, :3].T + T[:3, 3])
    low = h < float(top_mm) - band_mm
    out[vv[low], uu[low]] = 0
    return out


def table_share(depth_m: np.ndarray, mask: np.ndarray, intrinsics: Mapping[str, float],
                T_BC_mm: np.ndarray, plane: TablePlane,
                tol_mm: float = TABLE_SHARE_TOL_MM) -> float | None:
    """Share of the mask's pixels whose depth puts them on the table plane.

    The count of pixels returning a depth within tol_mm of the table, over all
    pixels of the mask, with or without depth. A mask spilling beside a part, or a
    false detection on the paper cards, holds a piece of matte table that returns
    depth at table height. A part does not: its top stands above the table, and a
    mirror top's wrong returns are holes or lie far below the table, so on a real
    part only the outline band reads table height. None for an empty mask.
    """
    m = np.asarray(mask) > 0
    n = int(np.count_nonzero(m))
    if n == 0:
        return None
    vv, uu = np.nonzero(m & (depth_m > 0))
    if uu.size == 0:
        return 0.0
    z = depth_m[vv, uu] * 1000.0
    x = (uu - intrinsics["ppx"]) * z / intrinsics["fx"]
    y = (vv - intrinsics["ppy"]) * z / intrinsics["fy"]
    T = np.asarray(T_BC_mm, float)
    h = plane.height_above(np.column_stack([x, y, z]) @ T[:3, :3].T + T[:3, 3])
    return float(np.count_nonzero(np.abs(h) < tol_mm)) / n


def off_table_mask(depth_m: np.ndarray, mask: np.ndarray, intrinsics: Mapping[str, float],
                   T_BC_mm: np.ndarray, plane: TablePlane,
                   tol_mm: float = TABLE_SHARE_TOL_MM) -> np.ndarray:
    """The mask less the pixels whose depth puts them on the table plane, as 0/1 uint8.

    The pixels removed are exactly the ones table_share counts: a depth return within
    tol_mm of the table. Pixels without depth stay, and so do returns far from the
    table, so a mirror top that returns holes or reflections far below the table
    keeps its outline. What goes is matte paper and table under the mask.
    """
    m = np.asarray(mask) > 0
    out = m.astype(np.uint8)
    vv, uu = np.nonzero(m & (depth_m > 0))
    if uu.size == 0:
        return out
    z = depth_m[vv, uu] * 1000.0
    x = (uu - intrinsics["ppx"]) * z / intrinsics["fx"]
    y = (vv - intrinsics["ppy"]) * z / intrinsics["fy"]
    T = np.asarray(T_BC_mm, float)
    h = plane.height_above(np.column_stack([x, y, z]) @ T[:3, :3].T + T[:3, 3])
    on = np.abs(h) < tol_mm
    out[vv[on], uu[on]] = 0
    return out


def fuse_height(h_depth_mm: float | None, h_prior_mm: float, h_min_mm: float,
                valid_frac: float, min_valid_frac: float = MIN_VALID_FRAC,
                tol_mm: float = DISAGREE_TOL_MM) -> tuple[float, str]:
    """Fusion rule: the height the top face is placed at, and where it came from.

    The plane prior stands unless the depth is worth believing (enough of the mask
    returns it, and it forms a top face), is physically possible (not below the
    lowest height the part can stand at) and disagrees by more than the tolerance.
    A part resting on another part, or a carton whose size was read wrong, is
    corrected; dropout and edge bleed are not.

    Args:
        h_depth_mm: Top face height measured from depth, or None.
        h_prior_mm: Height the plane estimate assumes.
        h_min_mm: Lowest height the part can stand at (the smaller carton, for the carton).
        valid_frac: Share of mask pixels returning depth.
    """
    if (h_depth_mm is not None and valid_frac >= min_valid_frac
            and h_depth_mm >= h_min_mm - tol_mm and abs(h_depth_mm - h_prior_mm) > tol_mm):
        return float(h_depth_mm), "rgbd"
    return float(h_prior_mm), "plane"
