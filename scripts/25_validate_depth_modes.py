#!/usr/bin/env python
"""
25_validate_depth_modes.py
──────────────────────────
Check the C4 depth modes on the labelled dataset, before the physical grasp campaign.

For every human-labelled instance that has depth:
  * the table plane is fitted from the depth around the parts, image by image;
  * the part's top face is measured from depth (the dataset audit's rule, which
    agrees with the caliper heights to about 2 mm);
  * the carton's size is resolved from its silhouette and checked against that
    measurement, which is what the method section promises to validate;
  * the three depth modes are compared against the depth-derived centre of the top
    face, which is the point the jaws have to reach;
  * the branch fusion takes is counted, by class and by layout.

The camera looks straight down, so the frames come from the dataset itself: the
base frame is the camera frame with y and z flipped, and the table plane is
measured in it. Every distance reported is relative to that plane, which is what
all three modes use, so no hand-eye calibration is needed to compare them. The
focal length is the one calibrated from the table edge in this dataset (655 px at
1280 x 720); --focal overrides it.

Depth cannot referee itself on the mirror-finish inox case, whose depth is the
failure C4 exists for: its rows are reported but carry that caveat.

Run:
    python scripts/25_validate_depth_modes.py
    python scripts/25_validate_depth_modes.py --limit 200      # quick pass
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.table_plane import fit_table_plane  # noqa: E402
from src.perception.depth_modes import (  # noqa: E402
    DISAGREE_TOL_MM, MIN_VALID_FRAC, TablePlane, fit_box, fuse_height, image_yaw_to_base,
    ray_plane_depth, resolve_size, top_face_height, valid_depth_fraction)
from src.perception.postprocess import (  # noqa: E402
    deproject_pixel, mask_centroid, mask_pca_yaw, masked_depth)

W, H = 1280, 720
# Caliper sizes (length, width, height) in mm; the carton comes in two.
SIZES = {
    "carton": [(180.0, 120.0, 120.0), (180.0, 100.0, 80.0)],
    "plastic_box": [(180.0, 140.0, 30.0)],
    "wood_box": [(190.0, 190.0, 150.0)],
    "metal_box": [(190.0, 135.0, 85.0)],
    "inox_box": [(180.0, 85.0, 40.0)],
}
CLASSES = list(SIZES)
# Camera looking straight down: base = camera frame with y and z flipped.
T_BC = np.array([[1.0, 0.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]])
K_TABLE = np.ones((31, 31), np.uint8)      # keeps the parts out of the table fit
K_ERODE = np.ones((7, 7), np.uint8)        # keeps mask edges out of the height measure
K_TOUCH = np.ones((21, 21), np.uint8)      # "next to" for the support test
ROI = (20, 710, 200, 1100)                 # table region of the frame
RAISE_TOL, SUPPORT_TOL = 20.0, 25.0        # the audit's stacking rule
TOP_BAND_MM = 8.0                          # how close to the top face a pixel must be
TOP_FACE_BAND_MM = 15.0                    # thickness of the top-face height cluster


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=r"D:\Scientific\Dataset\DigitalTwin")
    p.add_argument("--labels", default=None, help="Default: <dataset>/_work/labels_human")
    p.add_argument("--out", default=None, help="Default: <dataset>/_work/reports")
    p.add_argument("--focal", type=float, default=655.0, help="fx = fy in pixels")
    p.add_argument("--limit", type=int, default=0, help="Stop after N images (0 = all)")
    p.add_argument("--stride", type=int, default=4, help="Pixel step of the table fit")
    return p.parse_args()


def polygons(label_path: Path) -> list[tuple[str, np.ndarray]]:
    """(class, mask) per instance from a YOLO-seg polygon label file."""
    out = []
    for row in label_path.read_text(encoding="utf-8").splitlines():
        f = row.split()
        if len(f) < 7:
            continue
        pts = (np.array(f[1:], float).reshape(-1, 2) * [W, H]).astype(np.int32)
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [pts], 1)
        if m.sum() >= 200:
            out.append((CLASSES[int(f[0])], m))
    return out


def to_base(intr: dict, uu, vv, z_mm) -> np.ndarray:
    """Deproject pixels with their depths (mm) into the base frame."""
    x = (np.asarray(uu, float) - intr["ppx"]) * z_mm / intr["fx"]
    y = (np.asarray(vv, float) - intr["ppy"]) * z_mm / intr["fy"]
    return np.column_stack([x, y, z_mm]) @ T_BC[:3, :3].T + T_BC[:3, 3]


def table_plane(depth_mm: np.ndarray, masks: list[np.ndarray], intr: dict,
                stride: int) -> TablePlane | None:
    """Fit the table from the depth around the parts, which are dilated away."""
    union = np.zeros((H, W), np.uint8)
    if masks:
        union = cv2.dilate(np.clip(np.sum(masks, axis=0), 0, 1).astype(np.uint8), K_TABLE)
    keep = np.zeros((H, W), bool)
    keep[ROI[0]:ROI[1], ROI[2]:ROI[3]] = True
    vv, uu = np.nonzero(keep & (union == 0) & (depth_mm > 0))
    if uu.size < 5000:
        return None
    vv, uu = vv[::stride], uu[::stride]
    try:
        fit = fit_table_plane(to_base(intr, uu, vv, depth_mm[vv, uu]))
    except ValueError:
        return None
    return TablePlane(fit["a"], fit["b"], fit["c"])


def top_height(depth_mm: np.ndarray, mask: np.ndarray, plane: TablePlane,
               intr: dict) -> tuple[float | None, np.ndarray | None, float | None]:
    """Height of the part's top face above the table (mm), its centre in the base
    frame, and the upper-percentile height the dataset audit used.

    The mask is eroded first, so that its edge, where depth bleeds onto the table,
    cannot pull the measurement down. The top face is then the cluster of heights
    near the top, and its median is taken; an upper percentile on its own rides the
    depth noise, which on this camera puts it about half a millimetre high.
    """
    e = cv2.erode(mask, K_ERODE)
    vv, uu = np.nonzero((e > 0) & (depth_mm > 0))
    if uu.size < 200 or uu.size < 0.3 * np.count_nonzero(e):
        return None, None, None
    hgt = plane.height_above(to_base(intr, uu, vv, depth_mm[vv, uu]))
    h_p80 = float(np.percentile(hgt, 80))
    h_top = float(np.median(hgt[hgt > np.percentile(hgt, 90) - TOP_FACE_BAND_MM]))
    # Centre of the top face: the pixels that sit on it, over the whole mask.
    vv2, uu2 = np.nonzero((mask > 0) & (depth_mm > 0))
    P = to_base(intr, uu2, vv2, depth_mm[vv2, uu2])
    on_top = np.abs(plane.height_above(P) - h_top) < TOP_BAND_MM
    centre = P[on_top][:, :2].mean(axis=0) if np.count_nonzero(on_top) >= 200 else None
    return h_top, centre, h_p80


def layout(idx: int, cls: list[str], hs: list[float | None],
           masks: list[np.ndarray]) -> str:
    """"single", "stacked" or "raised" for one instance (the dataset audit's rule)."""
    h = hs[idx]
    if h is None:
        return "unknown"
    nominal = [s[2] for s in SIZES[cls[idx]]]
    if h <= max(nominal) + RAISE_TOL:
        return "single"
    touch = cv2.dilate(masks[idx], K_TOUCH)
    for j, hj in enumerate(hs):
        if j == idx or hj is None or not np.any(touch & masks[j]):
            continue
        if any(abs((h - hc) - hj) < SUPPORT_TOL for hc in nominal):
            return "stacked"
    return "raised"


def instance_row(cls: str, mask: np.ndarray, depth_m: np.ndarray, depth_mm: np.ndarray,
                 plane: TablePlane, intr: dict, h_top: float | None,
                 centre: np.ndarray | None, lay: str) -> dict:
    """One instance through all three depth modes, scored against the depth reference."""
    r: dict = {"class": cls, "layout": lay, "h_top_mm": h_top,
               "valid_frac": round(valid_depth_fraction(depth_m, mask), 4)}
    uv = mask_centroid(mask)
    if uv is None:
        return r
    u, v = uv
    yaw = mask_pca_yaw(mask)
    cands = SIZES[cls]

    # ── size, as the runtime would resolve it ──────────────────────────────
    if len(cands) > 1:
        res = resolve_size(mask, intr, T_BC, plane, cands, yaw)
        if res is None:
            return r
        fit, ious = res
        order = sorted(ious, reverse=True)
        r["size_pred_mm"] = fit.size[2]
        r["iou_best"] = round(fit.iou, 4)
        r["iou_margin"] = round(order[0] - order[1], 4)
        r["yaw_flipped"] = int(abs(((fit.yaw_base - image_yaw_to_base(yaw, T_BC))
                                    + np.pi / 4) % (np.pi / 2) - np.pi / 4) > 0.5)
        if h_top is not None and lay == "single":
            near = min(cands, key=lambda s: abs(s[2] - h_top))
            if abs(near[2] - h_top) < RAISE_TOL:
                r["size_true_mm"] = near[2]
    else:
        fit = fit_box(mask, intr, T_BC, plane, cands[0], image_yaw_to_base(yaw, T_BC))
        if fit is None:
            return r
        alt = fit_box(mask, intr, T_BC, plane, cands[0],
                      image_yaw_to_base(yaw, T_BC) + np.pi / 2)
        if alt is not None and alt.iou > fit.iou:
            fit = alt
        r["size_pred_mm"] = fit.size[2]
        r["iou_best"] = round(fit.iou, 4)

    h_pred = float(r.get("size_pred_mm", cands[0][2]))

    # ── the three depth modes ──────────────────────────────────────────────
    z_depth = masked_depth(depth_m, mask)
    z_plane = ray_plane_depth(intr, u, v, T_BC, plane, h_pred)
    h_depth = None
    if z_depth is not None:
        p_rgbd = to_base(intr, [u], [v], [z_depth * 1000.0])[0]
        h_depth = float(plane.height_above(p_rgbd))
        r["h_depth_mm"] = round(h_depth, 1)
        r["z_err_rgbd_mm"] = None if h_top is None else round(h_depth - h_top, 1)
    h_face = top_face_height(depth_m, mask, intr, T_BC, plane)
    if h_face is not None:
        r["h_face_mm"] = round(h_face, 1)
    h_use, source = fuse_height(h_face, h_pred, min(s[2] for s in cands), r["valid_frac"],
                                MIN_VALID_FRAC, DISAGREE_TOL_MM)
    z_fused = ray_plane_depth(intr, u, v, T_BC, plane, h_use)
    r["fusion_used"] = source
    r["z_err_plane_mm"] = None if h_top is None else round(h_pred - h_top, 1)
    r["z_err_fusion_mm"] = None if h_top is None else round(h_use - h_top, 1)

    # ── grasp point: each estimate against the measured top-face centre ────
    if centre is not None:
        for name, z in (("rgbd", z_depth), ("plane", z_plane), ("fusion", z_fused)):
            if z is None:
                continue
            p = to_base(intr, [u], [v], [z * 1000.0])[0]
            r[f"xy_err_{name}_mm"] = round(float(np.hypot(*(p[:2] - centre))), 1)
        r["xy_err_fit_mm"] = round(
            float(np.hypot(*(np.asarray(fit.center_xy) - centre))), 1)
    return r


def med_p90(vals) -> str:
    v = np.asarray([x for x in vals if x is not None], float)
    return "n/a" if v.size == 0 else "%.1f / %.1f" % (np.median(v), np.percentile(v, 90))


def write_report(rows: list[dict], stats: dict, out_dir: Path, intr: dict) -> Path:
    L: list[str] = []
    A = L.append
    A("# C4 depth modes on the labelled dataset\n")
    A("Generated by `scripts/25_validate_depth_modes.py`. Focal length %.0f px, "
      "principal point (%.0f, %.0f); the table plane is fitted per image from the "
      "depth around the parts.\n" % (intr["fx"], intr["ppx"], intr["ppy"]))
    A("Images with labels and depth: **%d** (table fit failed on %d). "
      "Instances: **%d**, of which %d have no usable depth of their own.\n"
      % (stats["images"], stats["plane_failed"], len(rows),
         sum(1 for r in rows if r.get("h_top_mm") is None)))

    by_cls = collections.defaultdict(list)
    for r in rows:
        by_cls[r["class"]].append(r)

    A("\n## 1. Depth truth cross-check (single-layer instances)\n")
    A("Height measured from depth minus the caliper height. This is the reference the "
      "rest of the report leans on, so it is shown first.\n")
    A("| class | instances | median | IQR | share over 20 mm out | median, audit rule |")
    A("|---|---:|---:|---|---:|---:|")
    for c in CLASSES:
        g = [r for r in by_cls[c] if r.get("h_top_mm") is not None
             and r.get("size_pred_mm") and r["layout"] == "single"]
        if not g:
            continue
        a = np.asarray([r["h_top_mm"] - r["size_pred_mm"] for r in g])
        p80 = [r["h_top_p80_mm"] - r["size_pred_mm"] for r in g
               if r.get("h_top_p80_mm") is not None]
        A("| %s | %d | %.1f | %.1f .. %.1f | %.1f%% | %s |"
          % (c, len(a), np.median(a), *np.percentile(a, [25, 75]),
             100.0 * np.mean(np.abs(a) > 20.0),
             "n/a" if not p80 else "%.1f" % np.median(p80)))

    A("\n## 2. Carton size resolved from the silhouette\n")
    car = [r for r in by_cls["carton"] if r.get("size_true_mm") and r.get("size_pred_mm")]
    if car:
        ok = sum(1 for r in car if r["size_true_mm"] == r["size_pred_mm"])
        A("Instances with an unambiguous depth truth: **%d**. Correct: **%d (%.1f%%)**.\n"
          % (len(car), ok, 100.0 * ok / len(car)))
        A("| true \\ predicted | 80 mm | 120 mm |")
        A("|---|---:|---:|")
        for t in (80.0, 120.0):
            A("| %.0f mm | %d | %d |" % (t, sum(1 for r in car if r["size_true_mm"] == t
                                                and r["size_pred_mm"] == 80.0),
                                         sum(1 for r in car if r["size_true_mm"] == t
                                             and r["size_pred_mm"] == 120.0)))
        A("\nBy condition, and by how clearly the winning size beat the other:\n")
        A("| split | instances | correct |")
        A("|---|---:|---:|")
        groups = [(c, [r for r in car if r["condition"] == c])
                  for c in sorted({r["condition"] for r in car})]
        groups += [("IoU margin under 0.02", [r for r in car if r["iou_margin"] < 0.02]),
                   ("IoU margin 0.02 and over", [r for r in car if r["iou_margin"] >= 0.02])]
        for name, g in groups:
            if g:
                A("| %s | %d | %.1f%% |" % (name, len(g), 100.0 * np.mean(
                    [r["size_true_mm"] == r["size_pred_mm"] for r in g])))
        A("\nThe mask's major axis was not the part's long axis in %.1f%% of cartons; "
          "the fit tries both hypotheses.\n"
          % (100.0 * np.mean([r.get("yaw_flipped", 0) for r in by_cls["carton"]])))

    A("\n## 3. Grasp point: distance from the measured top-face centre (mm, median / 90th)\n")
    A("Single-layer instances only. The RGB-D and plane modes both send the ray through "
      "the mask centroid, which the visible side faces pull towards the image centre; "
      "the box fit models those faces instead.\n")
    A("| class | instances | RGB-D | plane | fusion | box fit |")
    A("|---|---:|---:|---:|---:|---:|")
    for c in CLASSES:
        g = [r for r in by_cls[c] if r["layout"] == "single" and "xy_err_plane_mm" in r]
        if g:
            A("| %s | %d | %s | %s | %s | %s |"
              % (c, len(g), med_p90([r.get("xy_err_rgbd_mm") for r in g]),
                 med_p90([r.get("xy_err_plane_mm") for r in g]),
                 med_p90([r.get("xy_err_fusion_mm") for r in g]),
                 med_p90([r.get("xy_err_fit_mm") for r in g])))

    A("\n## 4. Height error of each mode (mm, median / 90th of the absolute error)\n")
    A("Measured against the same depth-derived top face, so the RGB-D and fusion columns "
      "are not independent of it: on the stacked rows they show that the fusion decision "
      "fires, not that its value was confirmed by anything else. The plane column is "
      "independent, because it reads no depth at all.\n")
    A("| class | layout | instances | valid depth (%) | RGB-D | plane | fusion |")
    A("|---|---|---:|---:|---:|---:|---:|")
    for c in CLASSES:
        for lay in ("single", "stacked", "raised"):
            g = [r for r in by_cls[c] if r["layout"] == lay]
            if not g:
                continue
            A("| %s | %s | %d | %.1f | %s | %s | %s |"
              % (c, lay, len(g), 100.0 * float(np.median([r["valid_frac"] for r in g])),
                 med_p90([abs(r["z_err_rgbd_mm"]) for r in g
                          if r.get("z_err_rgbd_mm") is not None]),
                 med_p90([abs(r["z_err_plane_mm"]) for r in g
                          if r.get("z_err_plane_mm") is not None]),
                 med_p90([abs(r["z_err_fusion_mm"]) for r in g
                          if r.get("z_err_fusion_mm") is not None])))

    A("\n## 5. Which branch fusion takes\n")
    A("| class | layout | instances | keeps the plane prior | takes the depth |")
    A("|---|---|---:|---:|---:|")
    for c in CLASSES:
        for lay in ("single", "stacked", "raised"):
            g = [r for r in by_cls[c] if r["layout"] == lay and r.get("fusion_used")]
            if g:
                pl = sum(1 for r in g if r["fusion_used"] == "plane")
                A("| %s | %s | %d | %.1f%% | %.1f%% |"
                  % (c, lay, len(g), 100.0 * pl / len(g), 100.0 * (len(g) - pl) / len(g)))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "depth_modes.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    root = Path(args.dataset)
    lab_root = Path(args.labels) if args.labels else root / "_work" / "labels_human"
    out_dir = Path(args.out) if args.out else root / "_work" / "reports"
    intr = {"fx": args.focal, "fy": args.focal, "ppx": W / 2.0, "ppy": H / 2.0}

    labels = sorted(p for p in lab_root.rglob("*.txt") if p.parent != lab_root)
    rows: list[dict] = []
    stats = {"images": 0, "plane_failed": 0}
    for n, lp in enumerate(labels):
        if args.limit and stats["images"] >= args.limit:
            break
        rel = lp.parent.relative_to(lab_root)
        dp = root / rel / (lp.stem + "_depth.npy")
        if not dp.exists():
            continue
        inst = polygons(lp)
        if not inst:
            continue
        depth_m = np.load(dp)
        depth_mm = depth_m * 1000.0
        cls = [c for c, _ in inst]
        masks = [m for _, m in inst]
        plane = table_plane(depth_mm, masks, intr, args.stride)
        if plane is None:
            stats["plane_failed"] += 1
            continue
        stats["images"] += 1
        tops = [top_height(depth_mm, m, plane, intr) for m in masks]
        hs = [t[0] for t in tops]
        for i, (c, m) in enumerate(inst):
            r = instance_row(c, m, depth_m, depth_mm, plane, intr, hs[i], tops[i][1],
                             layout(i, cls, hs, masks))
            r["h_top_p80_mm"] = None if tops[i][2] is None else round(tops[i][2], 1)
            r["image"] = str(rel / lp.stem).replace("\\", "/")
            r["condition"] = rel.parts[0]
            rows.append(r)
        if stats["images"] % 200 == 0:
            print("  %d images, %d instances" % (stats["images"], len(rows)))

    if not rows:
        print("no labelled instance with depth found under", root)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r})
    csv_path = out_dir / "depth_modes_instances.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "condition", "class", "layout"]
                           + [c for c in cols if c not in
                              ("image", "condition", "class", "layout")])
        w.writeheader()
        w.writerows(rows)
    report = write_report(rows, stats, out_dir, intr)
    print("instances -> %s" % csv_path)
    print("report    -> %s" % report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
