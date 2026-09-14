"""
table_plane.py
──────────────
Measure the feeding-table plane in the hand-eye frame from a depth image.

The grasp floor (table_top_z_mm + table_safety_margin_mm in the orchestrator)
must be expressed in exactly the frame of T_BC, and that frame carries whatever
robot base offset the cell layout declared at calibration time. A height typed
by hand invites a frame mix-up; one derived from depth through the same T_BC
cannot have one.
"""
from __future__ import annotations

import numpy as np


def depth_to_points(depth_m: np.ndarray, intrinsics: dict, T_BC: np.ndarray,
                    stride: int = 4, roi: float = 0.5) -> np.ndarray:
    """Deproject the central ROI of a depth image into the T_BC frame (mm).

    Args:
        depth_m: Depth image in metres, aligned to the colour stream.
        intrinsics: Colour intrinsics dict (fx, fy, ppx, ppy).
        T_BC: Camera-to-base matrix (mm).
        stride: Pixel step, to keep the point count modest.
        roi: Fraction of width and height kept around the image centre, where the
            table dominates the view.
    """
    d = np.asarray(depth_m, float)
    h, w = d.shape
    y0, y1 = int(h * (1 - roi) / 2), int(h * (1 + roi) / 2)
    x0, x1 = int(w * (1 - roi) / 2), int(w * (1 + roi) / 2)
    vv, uu = np.mgrid[y0:y1:stride, x0:x1:stride]
    z = d[vv, uu] * 1000.0
    ok = z > 0
    uu, vv, z = uu[ok].astype(float), vv[ok].astype(float), z[ok]
    x = (uu - intrinsics["ppx"]) * z / intrinsics["fx"]
    y = (vv - intrinsics["ppy"]) * z / intrinsics["fy"]
    pts = np.column_stack([x, y, z, np.ones_like(z)])
    return (np.asarray(T_BC, float) @ pts.T).T[:, :3]


def fit_table_plane(points_mm: np.ndarray, band_mm: float = 15.0,
                    inlier_mm: float = 4.0, iters: int = 4) -> dict:
    """Robust fit z = a*x + b*y + c of the dominant near-horizontal plane.

    Starts from points within ``band_mm`` of the median height (the table fills
    most of the view), then refits on inliers within ``inlier_mm``. Reports the
    95th percentile of inlier heights as ``z_top_mm``: a floor derived from it
    stays above the table even where the plane tilts up.
    """
    P = np.asarray(points_mm, float)
    if len(P) < 100:
        raise ValueError(f"too few points for a plane fit: {len(P)}")
    sel = np.abs(P[:, 2] - np.median(P[:, 2])) < band_mm
    coef = np.zeros(3)
    res = np.zeros(len(P))
    for _ in range(iters):
        if sel.sum() < 100:
            raise ValueError("plane fit lost its inliers: is the table clear and in view?")
        A = np.column_stack([P[sel, 0], P[sel, 1], np.ones(int(sel.sum()))])
        coef, *_ = np.linalg.lstsq(A, P[sel, 2], rcond=None)
        res = P[:, 2] - (coef[0] * P[:, 0] + coef[1] * P[:, 1] + coef[2])
        sel = np.abs(res) < inlier_mm
    if sel.sum() < 100:
        raise ValueError("plane fit lost its inliers: is the table clear and in view?")
    a, b, c = (float(v) for v in coef)
    zi = P[sel, 2]
    return {
        "a": a, "b": b, "c": c,
        "z_mean_mm": float(zi.mean()),
        "z_top_mm": float(np.percentile(zi, 95)),
        "rms_mm": float(np.sqrt(np.mean(res[sel] ** 2))),
        "tilt_deg": float(np.degrees(np.arctan(np.hypot(a, b)))),
        "n_inliers": int(sel.sum()),
        "n_points": int(len(P)),
    }
