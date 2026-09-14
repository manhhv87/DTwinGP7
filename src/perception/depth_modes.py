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
# Below this silhouette IoU the size decision is not trusted: on the labelled set a
# correct decision scores 0.96 at the median and a wrong one 0.66, and every wrong
# one had another part hiding a piece of this one. An occluded part is then stood at
# its lowest candidate height, because assuming a part is taller than it is puts the
# jaws above it, which never grips.
MIN_SIZE_IOU = 0.80

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
