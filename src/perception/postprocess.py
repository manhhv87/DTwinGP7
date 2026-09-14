"""
postprocess.py
──────────────
Extract 3D pose of an object from a segmentation mask + depth image.

Pipeline for each detection:
    mask (2D)  →  centroid (u, v)
    depth+mask →  z (median, noise-robust), or, in the C4 depth modes, the
                  viewing ray meeting the table plane (see depth_modes.py)
    (u, v, z)  →  deproject → (x, y, z) in camera frame
    mask (2D)  →  PCA → yaw

Entirely pure-numpy → testable without a camera. `cv2` is used only for
mask resizing and polygon geometry, and is lazy-imported.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .depth_modes import (
    DEPTH_MODES,
    DISAGREE_TOL_MM,
    MIN_SIZE_IOU,
    MIN_VALID_FRAC,
    TablePlane,
    fuse_height,
    parse_class_sizes,
    ray_plane_depth,
    resolve_size,
    top_face_height,
    valid_depth_fraction,
)

logger = logging.getLogger(__name__)

# Minimum number of mask pixels to consider a detection valid.
MIN_MASK_PIXELS = 25


def resize_mask(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask to the target (height, width) using nearest-neighbor interpolation.

    YOLOv8 returns masks at a different resolution than the original image,
    so they must be resized before combining with the depth image.
    """
    if mask.shape == target_hw:
        return mask
    import cv2  # lazy import

    resized = cv2.resize(
        mask.astype(np.uint8),
        (target_hw[1], target_hw[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized


def mask_centroid(mask: np.ndarray) -> tuple[int, int] | None:
    """Compute the centroid (u, v) of a binary mask.

    Returns:
        (u, v) pixel coordinate, or None if the mask is too small.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < MIN_MASK_PIXELS:
        return None
    return int(round(xs.mean())), int(round(ys.mean()))


def masked_depth(
    depth_image: np.ndarray,
    mask: np.ndarray,
    method: str = "median",
) -> float | None:
    """Return a representative depth value over the masked region.

    Args:
        depth_image: Depth image (in metres), shape (H, W).
        mask: Binary mask with the same shape.
        method: "median" (default, noise-robust) or "mean".

    Returns:
        Depth (metres), or None if there are not enough valid depth pixels.
    """
    if depth_image.shape != mask.shape:
        mask = resize_mask(mask, depth_image.shape)

    values = depth_image[mask > 0]
    values = values[values > 0]  # discard pixels with no depth
    if len(values) < 5:
        return None
    return float(np.median(values) if method == "median" else np.mean(values))


# A mask this full of its minimum-area rectangle is a square part; a disk gives pi/4.
SQUARE_FILL_MIN = 0.90


def _square_edge_yaw(mask: np.ndarray) -> float | None:
    """Edge direction of a square-like mask, folded into (-pi/4, pi/4].

    Returns None when the mask does not fill its minimum-area rectangle (a round
    part), where every yaw is equivalent.
    """
    import cv2  # lazy import, as elsewhere in the perception package

    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    w, h = rect[1]
    if w * h <= 0 or float(m.sum()) / (w * h) < SQUARE_FILL_MIN:
        return None
    p = cv2.boxPoints(rect)
    a = float(np.arctan2(p[1][1] - p[0][1], p[1][0] - p[0][0]))
    # A square looks the same every 90 degrees: fold into (-pi/4, pi/4].
    a = (a + np.pi / 4) % (np.pi / 2) - np.pi / 4
    if a <= -np.pi / 4:
        a += np.pi / 2
    return a


def mask_pca_yaw(mask: np.ndarray) -> float:
    """Estimate the object's yaw via PCA on mask pixels.

    The principal component (largest eigenvalue) aligns with the object's long axis.

    Returns:
        Yaw (radians), in the range (-pi/2, pi/2].
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < MIN_MASK_PIXELS:
        return 0.0

    pts = np.column_stack([xs, ys]).astype(float)
    pts -= pts.mean(axis=0)

    # 2x2 covariance matrix → eigenvector corresponding to the largest eigenvalue.
    cov = np.cov(pts, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    lo, hi = float(np.min(eigvals)), float(np.max(eigvals))
    if hi <= 1e-9:
        return 0.0
    if (lo / hi) > 0.85:
        # Near-isotropic mask: the principal axis is arbitrary and flips frame to
        # frame. A SQUARE part (the wooden box, 190 x 190 mm) still has well-defined
        # edges, and the jaws must close across a pair of faces: with a yaw of zero,
        # a 216 mm jaw opening straddles its 269 mm diagonal only when the box lies
        # within about 8.5 degrees of alignment. Use the edge direction of the
        # minimum-area rectangle for square masks; keep 0.0 for round ones.
        edge = _square_edge_yaw(mask)
        return 0.0 if edge is None else edge
    major_axis = eigvecs[:, np.argmax(eigvals)]

    yaw = np.arctan2(major_axis[1], major_axis[0])
    # Normalise to (-pi/2, pi/2] because an axis has 180° symmetry.
    if yaw > np.pi / 2:
        yaw -= np.pi
    elif yaw <= -np.pi / 2:
        yaw += np.pi
    return float(yaw)


def deproject_pixel(
    intrinsics: dict[str, float],
    u: float,
    v: float,
    z_m: float,
) -> np.ndarray:
    """Deproject pixel (u, v) + depth → 3D point in camera frame.

    Uses the pinhole model. Equivalent to rs2_deproject_pixel_to_point but
    pure-numpy → testable without the RealSense SDK.

    Args:
        intrinsics: Dict {fx, fy, ppx, ppy} (pixels).
        u, v: Pixel coordinates.
        z_m: Depth at that pixel (metres).

    Returns:
        Point (x, y, z) in camera frame, in mm.
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    ppx, ppy = intrinsics["ppx"], intrinsics["ppy"]
    x = (u - ppx) * z_m / fx
    y = (v - ppy) * z_m / fy
    return np.array([x, y, z_m]) * 1000.0  # metres → mm


def z_from_ground_plane(
    intrinsics: dict[str, float],
    u: float,
    v: float,
    T_BC_mm: np.ndarray,
    plane_z_base_mm: float,
) -> float | None:
    """Camera-frame depth z (metres) of pixel (u,v) by ray--plane intersection.

    Contribution C4 (depth-free localisation). Instead of reading depth, intersect
    the viewing ray through (u,v) with a horizontal plane Z=plane_z_base_mm in the
    BASE frame (e.g. the conveyor top, or top face of a known-height box). Returns
    the camera-frame z (metres) so it is a drop-in for masked_depth() feeding
    deproject_pixel(). None if the ray is parallel to the plane.

    Immune to depth dropouts on specular/dark parts (the D455 failure mode). Assumes
    the pixel lies on a known plane — i.e. a single layer of parts of known height.

    Args:
        intrinsics: {fx, fy, ppx, ppy}.
        u, v: pixel coordinates.
        T_BC_mm: 4x4 camera->base transform, translation in mm.
        plane_z_base_mm: plane height in base frame (mm), e.g. Z_conveyor + h_class.
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    ppx, ppy = intrinsics["ppx"], intrinsics["ppy"]
    a = (u - ppx) / fx
    b = (v - ppy) / fy
    R = np.asarray(T_BC_mm, dtype=float)[:3, :3]
    tz = float(T_BC_mm[2, 3])
    # base-frame Z of a camera-ray point at camera depth z (mm) = z*denom + tz
    denom = R[2, 0] * a + R[2, 1] * b + R[2, 2]
    if abs(denom) < 1e-9:
        return None                      # ray parallel to the plane
    z_mm = (plane_z_base_mm - tz) / denom
    if z_mm <= 0:
        return None                      # plane behind the camera
    return z_mm / 1000.0                  # → metres, matches masked_depth()


class PoseExtractor:
    """Combine the above helpers to extract a 3D pose from a single detection.

    Pure logic — accepts intrinsics as a dict; holds no camera reference.

    Args:
        intrinsics: {fx, fy, ppx, ppy} of the colour stream the depth is aligned to.
        depth_mode: "rgbd" (default: median depth under the mask), "plane" or
            "fusion" (contribution C4, see depth_modes.py). The last two need
            T_BC_mm and table_plane.
        T_BC_mm: Camera-to-base matrix of this cell (mm).
        table_plane: Table plane in the T_BC frame: a TablePlane, or the dict read
            from table_plane.json (keys a, b, c).
        class_heights_mm: Part height per class, for detections that carry none.
        class_sizes_mm: Candidate (L, W, H) sizes per class. A class with two or
            more (the carton) has its size resolved per instance from the
            silhouette whenever T_BC_mm and table_plane are given, in every mode,
            and the resolved height replaces the detection's.
        min_valid_frac, tol_mm: Fusion thresholds (see depth_modes.fuse_depth).
    """

    def __init__(
        self,
        intrinsics: dict[str, float],
        depth_mode: str = "rgbd",
        T_BC_mm: np.ndarray | None = None,
        table_plane: TablePlane | dict[str, Any] | None = None,
        class_heights_mm: dict[str, float] | None = None,
        class_sizes_mm: dict[str, Any] | None = None,
        min_valid_frac: float = MIN_VALID_FRAC,
        tol_mm: float = DISAGREE_TOL_MM,
    ) -> None:
        if depth_mode not in DEPTH_MODES:
            raise ValueError(f"depth_mode must be one of {DEPTH_MODES}, got {depth_mode!r}")
        if table_plane is not None and not isinstance(table_plane, TablePlane):
            table_plane = TablePlane.from_dict(table_plane)
        if depth_mode != "rgbd" and (T_BC_mm is None or table_plane is None):
            raise ValueError(f"depth_mode={depth_mode!r} intersects viewing rays with the "
                             "table plane: it needs T_BC_mm and table_plane")
        self.intrinsics = intrinsics
        self.depth_mode = depth_mode
        self.T_BC = None if T_BC_mm is None else np.asarray(T_BC_mm, dtype=float)
        self.plane = table_plane
        self.class_heights = {str(k): float(v) for k, v in (class_heights_mm or {}).items()}
        self.class_sizes = parse_class_sizes(class_sizes_mm)
        self.min_valid_frac = float(min_valid_frac)
        self.tol_mm = float(tol_mm)
        self._warned: set[str] = set()

    @property
    def has_geometry(self) -> bool:
        """True when rays can be intersected with the table (T_BC and plane known)."""
        return self.T_BC is not None and self.plane is not None

    def _heights(
        self, detection: dict[str, Any], mask: np.ndarray, yaw: float,
    ) -> tuple[float | None, float | None]:
        """(height used, lowest height the part can stand at) for this instance, mm."""
        name = detection.get("class_name")
        cands = self.class_sizes.get(name, [])
        if self.has_geometry and len(cands) > 1:
            res = resolve_size(mask, self.intrinsics, self.T_BC, self.plane, cands, yaw)
            if res is not None:
                fit = res[0]
                detection["size_iou"] = fit.iou
                h_low = min(c[2] for c in cands)
                if fit.iou < MIN_SIZE_IOU:      # occluded: the match means little
                    detection["size_mm"] = min(cands, key=lambda c: c[2])
                    return h_low, h_low
                detection["size_mm"] = fit.size
                return fit.size[2], h_low
        h = detection.get("height_mm")
        if h is None:
            h = self.class_heights.get(name)
        if h is None and cands:
            h = min(c[2] for c in cands)
        if h is None:
            return None, None
        return float(h), min([float(h)] + [c[2] for c in cands])

    def _measured_height(self, detection: dict[str, Any], h_use: float) -> None:
        """Record a height the depth corrected.

        A measured height that matches another candidate size means the size was read
        wrong; anything else means the part is standing on something, and its own
        height, which caps the grasp depth, has not changed.
        """
        for cand in self.class_sizes.get(detection.get("class_name"), []):
            if abs(cand[2] - h_use) <= self.tol_mm:
                detection["height_mm"] = cand[2]
                detection["size_mm"] = cand
                return

    def _warn_once(self, name: Any) -> None:
        key = str(name)
        if key not in self._warned:
            self._warned.add(key)
            logger.warning("No height for class %r: depth mode %s cannot place its top "
                           "face, so its detections are dropped.", key, self.depth_mode)

    def extract(
        self,
        detection: dict[str, Any],
        depth_image: np.ndarray,
    ) -> dict[str, Any] | None:
        """Augment a detection with the 'pose_camera' = (x, y, z, yaw) field.

        Also sets pixel_uv, mask_area, depth_used (which estimate gave z: "rgbd" or
        "plane"), depth_valid_frac, h_depth_mm (the top face measured from depth,
        when the table plane is known) and, for a class resolved by size, size_mm,
        size_iou and height_mm.

        Returns:
            Detection dict with pose added, or None if the pose cannot be extracted.
        """
        mask = detection["mask"]
        if mask.shape != depth_image.shape:
            mask = resize_mask(mask, depth_image.shape)

        centroid = mask_centroid(mask)
        if centroid is None:
            return None
        u, v = centroid
        yaw = mask_pca_yaw(mask)
        z_depth = masked_depth(depth_image, mask)
        valid_frac = valid_depth_fraction(depth_image, mask)

        height, h_min = self._heights(detection, mask, yaw)
        if height is not None:
            detection["height_mm"] = height

        h_depth = None
        if self.has_geometry and z_depth is not None:
            h_depth = top_face_height(depth_image, mask, self.intrinsics, self.T_BC,
                                      self.plane)

        if self.depth_mode == "rgbd":
            z_m, source = z_depth, "rgbd"
        elif height is None:
            # Nothing to stand on the table; fusion still has the median depth.
            self._warn_once(detection.get("class_name"))
            z_m, source = ((z_depth, "rgbd") if self.depth_mode == "fusion"
                           else (None, "plane"))
        elif self.depth_mode == "plane":
            z_m = ray_plane_depth(self.intrinsics, u, v, self.T_BC, self.plane, height)
            source = "plane"
        else:
            h_use, source = fuse_height(h_depth, height,
                                        height if h_min is None else h_min,
                                        valid_frac, self.min_valid_frac, self.tol_mm)
            z_m = ray_plane_depth(self.intrinsics, u, v, self.T_BC, self.plane, h_use)
            if source == "rgbd":
                self._measured_height(detection, h_use)
        if z_m is None:
            return None

        xyz_cam = deproject_pixel(self.intrinsics, u, v, z_m)
        detection["pose_camera"] = (
            float(xyz_cam[0]), float(xyz_cam[1]), float(xyz_cam[2]), yaw,
        )
        detection["pixel_uv"] = (u, v)
        detection["mask_area"] = int(np.sum(mask > 0))
        detection["depth_used"] = source
        detection["depth_valid_frac"] = valid_frac
        detection["h_depth_mm"] = h_depth
        return detection
