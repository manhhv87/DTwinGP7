#!/usr/bin/env python3
"""
make_top_textures.py
────────────────────
Cut the top-face textures of the dielectric products (carton, plastic, wood) out of
the TRAIN split of the real dataset, for the anchored recipe (src/synthgen/
top_texture.py says what a texture is and how it is stored).

Source: <yolo>/images/train and <yolo>/labels/train only (human YOLO-seg labels, the
class order of objects.class_ids). No validation, test, trial (E2-E5) or
condition-check image is read.

Per candidate (an image of the class's own single-product folder, std__<class>__*,
holding exactly one labelled instance):
  fit      the caliper box of each mesh of the class (carton: both sizes) standing on
           the table (objects.table_z_mm) at (x, y, yaw), projected with the calibrated
           camera (config/calibration/T_base_camera.npy, camera.intrinsics; pinhole,
           distortion 0, as the hand-eye fit); its silhouette (hull of the 8 corners)
           against the hull of the label polygon, IoU maximised (Nelder-Mead from the
           label projected onto the top plane, its minimum-area rectangle, cut to the
           caliper size on the nadir side, where the side faces project). The mesh of
           the higher IoU is the instance's mesh; it tells the 80 mm carton from the
           120 mm one, since the two heights project the label to different sizes.
  keep     label solidity (area over hull area) >= --min-solidity (nothing lies on or
           in front of the part), IoU >= --min-iou (a single box standing flat on the
           table: a stacked, tilted or occluded part does not fit), the label clear of
           the image border by --border px, the sampled face (below) wholly inside the
           label, under --max-saturated of its pixels with a channel at 250 or more,
           and no specular highlight: at most --max-highlight of the face inset by
           --highlight-inset of its sides (clear of the plastic box's rim and screws)
           brighter than 1.5 times that region's median luminance (the wrinkled film on
           the plastic box mirrors the cell luminaires in the bright-light images, which
           an albedo must not carry; carton and wood never reach the limit).
  rank     horizontal distance of the top-face centre from the camera nadir (fewest
           side-face pixels near the face), ascending; the first --per-mesh of each mesh.
  texture  the top face, inset by --inset-mm on every side (the label edge mixes the
           face with the bright table), resampled at --px-per-mm in the mesh's axes
           (columns along mesh +x, the long side; row 0 at mesh +y) from the linearised
           image (bilinear), divided by its mean in each linear channel, stored with
           top_texture.encode at its own scale (top_texture.scale_for).
Writes <out>/<mesh stem>/<mesh stem>_<nn>.png, <out>/manifest.json (per mesh: file,
SHA-256, source image and its SHA-256, label line, fit, mean linear colour of the face
in the source image), <out>/candidates.json (every candidate and why it was left out)
and, with --montage, one image of the kept textures and their fits.

Usage (from the repo root, repo venv):
    python scripts/make_top_textures.py --yolo /path/to/dataset/_work/yolo \
        --out models/textures --montage <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src.synthgen import top_texture as tt  # noqa: E402
from src.synthgen.sensor_model import srgb_to_linear  # noqa: E402

IMG_W, IMG_H = 1280, 720
LUMINANCE = np.array([0.2126, 0.7152, 0.0722])


def project(T: np.ndarray, intr: dict, p: np.ndarray) -> np.ndarray:
    """Base-frame points (n, 3) mm to pixel coordinates (n, 2)."""
    pc = (np.asarray(p, float) - T[:3, 3]) @ T[:3, :3]
    return np.stack([intr["fx"] * pc[:, 0] / pc[:, 2] + intr["ppx"],
                     intr["fy"] * pc[:, 1] / pc[:, 2] + intr["ppy"]], axis=1)


def to_plane(T: np.ndarray, intr: dict, uv: np.ndarray, z: float) -> np.ndarray:
    """Pixels (n, 2) to their rays' points on the plane z (base frame, mm), (n, 3)."""
    uv = np.asarray(uv, float)
    rays = np.stack([(uv[:, 0] - intr["ppx"]) / intr["fx"], (uv[:, 1] - intr["ppy"]) / intr["fy"],
                     np.ones(len(uv))], axis=1) @ T[:3, :3].T
    s = (z - T[2, 3]) / rays[:, 2]
    return T[:3, 3] + s[:, None] * rays


def axes(yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.radians(yaw_deg)
    return np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])


def corners(x: float, y: float, yaw: float, size: list[float], z: float) -> np.ndarray:
    """The 4 corners (x-y+, x+y+, x+y-, x-y-) of the L x W rectangle at height z."""
    ex, ey = axes(yaw)
    L, W = size[0] / 2.0, size[1] / 2.0
    c = np.array([x, y])
    xy = [c - L * ex + W * ey, c + L * ex + W * ey, c + L * ex - W * ey, c - L * ex - W * ey]
    return np.array([[p[0], p[1], z] for p in xy])


def silhouette(T, intr, x, y, yaw, size, z0) -> np.ndarray:
    pts = np.vstack([corners(x, y, yaw, size, z0), corners(x, y, yaw, size, z0 + size[2])])
    uv = project(T, intr, pts).astype(np.float32)
    return cv2.convexHull(uv)[:, 0, :]


def convex_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.ascontiguousarray(a, np.float32), np.ascontiguousarray(b, np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / union) if union > 0 else 0.0


def initial_pose(T, intr, poly, size, z0) -> tuple[float, float, float]:
    """Label projected onto the top plane, its minimum-area rectangle, each side cut to
    the caliper size on the side towards the nadir."""
    q = to_plane(T, intr, poly, z0 + size[2])[:, :2].astype(np.float32)
    (cx, cy), (w, h), ang = cv2.minAreaRect(q)
    long_first = w >= h
    yaw = ang if long_first else ang + 90.0
    ex, ey = axes(yaw)
    ext = [max(w, h), min(w, h)]
    c = np.array([cx, cy])
    nadir = T[:2, 3]
    for e, extent, true in ((ex, ext[0], size[0]), (ey, ext[1], size[1])):
        excess = extent - true
        if excess > 0:
            s = np.sign(e @ (c - nadir)) or 1.0
            c = c + e * s * excess / 2.0
    return float(c[0]), float(c[1]), float(yaw)


def fit_box(T, intr, poly, size, z0) -> dict:
    hull = cv2.convexHull(np.asarray(poly, np.float32))[:, 0, :]
    x0 = np.array(initial_pose(T, intr, poly, size, z0))

    def cost(p):
        return 1.0 - convex_iou(silhouette(T, intr, p[0], p[1], p[2], size, z0), hull)

    best = None
    for dyaw in (0.0, 90.0) if abs(size[0] - size[1]) < 1e-6 else (0.0,):
        start = x0 + [0.0, 0.0, dyaw]
        simplex = np.array([start, start + [4.0, 0, 0], start + [0, 4.0, 0], start + [0, 0, 4.0]])
        res = minimize(cost, start, method="Nelder-Mead",
                       options={"initial_simplex": simplex, "xatol": 0.02, "fatol": 1e-6,
                                "maxiter": 800})
        if best is None or res.fun < best.fun:
            best = res
    x, y, yaw = (float(v) for v in best.x)
    return {"x": x, "y": y, "yaw": yaw, "iou": 1.0 - float(best.fun)}


def face_maps(T, intr, fit, size, z_top, px_mm, inset):
    """Source pixel coordinates of every texture pixel, and the texture size."""
    L, W = size[0], size[1]
    wt, ht = int(round(L * px_mm)), int(round(W * px_mm))
    ex, ey = axes(fit["yaw"])
    xs = -L / 2 + inset + (np.arange(wt) + 0.5) / wt * (L - 2 * inset)
    ys = W / 2 - inset - (np.arange(ht) + 0.5) / ht * (W - 2 * inset)
    gx, gy = np.meshgrid(xs, ys)
    c = np.array([fit["x"], fit["y"]])
    p = c[None, None, :] + gx[..., None] * ex + gy[..., None] * ey
    pts = np.concatenate([p.reshape(-1, 2), np.full((p.shape[0] * p.shape[1], 1), z_top)], axis=1)
    uv = project(T, intr, pts).reshape(ht, wt, 2).astype(np.float32)
    return uv[..., 0], uv[..., 1]


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def face_texture(lin, T, intr, fit, size, z0, a):
    """The inset top face resampled from the linear image, (ht, wt, 3) float32."""
    mx, my = face_maps(T, intr, fit, size, z0 + size[2], a.px_per_mm, a.inset_mm)
    return cv2.remap(lin, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def examine(img_path, lines, names, meshes, sizes, T, intr, z0, a) -> dict:
    """One candidate: the fit and every test of the docstring; 'out' names the first
    test it fails (None: kept so far), 'face' holds the resampled top face."""
    rec = {"label_line": 0}
    cls = names[int(lines[0][0])]
    rec["cls"] = cls
    poly = np.array(lines[0][1:], float).reshape(-1, 2) * [IMG_W, IMG_H]
    hull = cv2.convexHull(poly.astype(np.float32))[:, 0, :]
    rec["solidity"] = float(cv2.contourArea(poly.astype(np.float32)) / cv2.contourArea(hull))
    rec["border_px"] = float(min(poly[:, 0].min(), poly[:, 1].min(), IMG_W - 1 - poly[:, 0].max(),
                                 IMG_H - 1 - poly[:, 1].max()))
    fits = {m: fit_box(T, intr, poly, sizes[m], z0) for m in meshes[cls]}
    mesh = max(fits, key=lambda m: fits[m]["iou"])
    fit = fits[mesh]
    size = sizes[mesh]
    rec.update(mesh=mesh, fit=fit, iou_by_mesh={m: f["iou"] for m, f in fits.items()})
    z_top = z0 + size[2]
    rec["top_uv"] = project(T, intr, corners(fit["x"], fit["y"], fit["yaw"], size, z_top)).tolist()
    rec["nadir_mm"] = float(np.hypot(fit["x"] - T[0, 3], fit["y"] - T[1, 3]))
    # The sampled region: the top face inset by inset_mm on every side.
    inner = [size[0] - 2 * a.inset_mm, size[1] - 2 * a.inset_mm, size[2]]
    quad = project(T, intr, corners(fit["x"], fit["y"], fit["yaw"], inner, z_top))
    mask = np.zeros((IMG_H, IMG_W), np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    region = np.zeros_like(mask)
    cv2.fillPoly(region, [np.round(quad).astype(np.int32)], 1)
    rec["sampled_inside"] = float((mask[region > 0] > 0).mean()) if region.any() else 0.0
    bgr = cv2.imread(str(img_path))
    rec["saturated"] = float((bgr[region > 0].max(axis=1) >= 250).mean())
    lin = srgb_to_linear(bgr[..., ::-1] / 255.0).astype(np.float32)
    face = face_texture(lin, T, intr, fit, size, z0, a)
    h, w = face.shape[:2]
    k0, k1 = int(round(a.highlight_inset * h)), int(round(a.highlight_inset * w))
    lum = face[k0:h - k0, k1:w - k1] @ LUMINANCE
    rec["highlight"] = float((lum > 1.5 * np.median(lum)).mean())
    if rec["solidity"] < a.min_solidity:
        rec["out"] = "label not convex (occluded or touching something)"
    elif rec["border_px"] < a.border:
        rec["out"] = "label at the image border"
    elif fit["iou"] < a.min_iou:
        rec["out"] = "box does not fit the label (stacked, tilted or occluded)"
    elif rec["sampled_inside"] < 0.999:
        rec["out"] = "sampled top face leaves the label"
    elif rec["saturated"] > a.max_saturated:
        rec["out"] = "saturated top face"
    elif rec["highlight"] > a.max_highlight:
        rec["out"] = "specular highlight on the top face"
    else:
        rec["out"] = None
    rec["face"] = face
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo", required=True, help="dataset/_work/yolo (images/train, labels/train are read)")
    ap.add_argument("--config", default="config/synthgen.yaml")
    ap.add_argument("--calib", default="config/calibration/T_base_camera.npy")
    ap.add_argument("--out", default="models/textures", help="relative to the repo root")
    ap.add_argument("--per-mesh", type=int, default=24)
    ap.add_argument("--px-per-mm", type=float, default=1.25)
    ap.add_argument("--inset-mm", type=float, default=3.0)
    ap.add_argument("--min-iou", type=float, default=0.95)
    ap.add_argument("--min-solidity", type=float, default=0.97)
    ap.add_argument("--border", type=float, default=5.0)
    ap.add_argument("--max-saturated", type=float, default=0.002)
    ap.add_argument("--highlight-inset", type=float, default=0.12)
    ap.add_argument("--max-highlight", type=float, default=0.011)
    ap.add_argument("--montage", help="directory for the check images")
    a = ap.parse_args()

    cfg = yaml.safe_load((REPO / a.config).read_text(encoding="utf-8"))
    T = np.load(REPO / a.calib)
    intr = cfg["camera"]["intrinsics"]
    obj = cfg["objects"]
    z0 = float(obj["table_z_mm"])
    sizes = {m: [float(v) for v in s] for m, s in obj["mesh_sizes_mm"].items()}
    names = {int(v): k for k, v in obj["class_ids"].items()}
    textured = [c for c, m in obj["materials"].items() if "albedo_rgb" in m]
    meshes = {c: (obj["classes"][c] if isinstance(obj["classes"][c], list) else [obj["classes"][c]])
              for c in textured}

    yolo = Path(a.yolo)
    img_dir, lab_dir = yolo / "images" / "train", yolo / "labels" / "train"
    cands = []
    for img_path in sorted(img_dir.glob("*.png")):
        parts = img_path.name.split("__")
        folder = parts[1] if len(parts) >= 3 else None
        if folder not in textured:
            continue
        lines = [ln.split() for ln in (lab_dir / (img_path.stem + ".txt")).read_text().splitlines()
                 if ln.strip()]
        rec = {"image": f"images/train/{img_path.name}", "folder_class": folder, "n_labels": len(lines)}
        if len(lines) != 1:
            rec["out"] = "not one labelled instance"
        elif names[int(lines[0][0])] != folder:
            rec["out"] = "label class differs from the folder"
        else:
            rec.update(examine(img_path, lines, names, meshes, sizes, T, intr, z0, a))
        cands.append(rec)

    rel_out = Path(a.out)
    out = REPO / rel_out
    manifest = {"manifest_version": tt.MANIFEST_VERSION, "encoding": tt.ENCODING,
                "made_by": "scripts/make_top_textures.py",
                "settings": {k: getattr(a, k) for k in ("per_mesh", "px_per_mm", "inset_mm", "min_iou",
                                                         "min_solidity", "border", "max_saturated",
                                                         "highlight_inset", "max_highlight")},
                "source": "dataset/_work/yolo, train split only (images/train, labels/train)",
                "camera": {"calib": a.calib, "intrinsics": intr, "table_z_mm": z0},
                "textures": {}}
    kept_by_mesh = {}
    for cls in textured:
        for mesh in meshes[cls]:
            ok = sorted([r for r in cands if r.get("mesh") == mesh and r.get("out") is None],
                        key=lambda r: r["nadir_mm"])
            for r in ok[a.per_mesh:]:
                r["out"] = f"beyond the {a.per_mesh} nearest the nadir"
            kept_by_mesh[mesh] = ok[: a.per_mesh]

    for mesh, kept in kept_by_mesh.items():
        stem = Path(mesh).stem
        (out / stem).mkdir(parents=True, exist_ok=True)
        for old in (out / stem).glob(f"{stem}_*.png"):
            old.unlink()
        entries = []
        for k, r in enumerate(kept):
            mean = r["face"].reshape(-1, 3).mean(axis=0)
            t = r["face"] / mean
            scale = tt.scale_for(t)
            name = f"{stem}_{k:02d}.png"
            path = out / stem / name
            cv2.imwrite(str(path), tt.encode(t, scale)[..., ::-1])
            back = tt.decode(tt.read_png(path), scale)
            r["texture"] = (rel_out / stem / name).as_posix()
            entries.append({"file": r["texture"], "sha256": sha_file(path), "scale": scale,
                            "size_px": [int(t.shape[1]), int(t.shape[0])],
                            "source_image": r["image"], "source_sha256": sha_file(yolo / r["image"]),
                            "label_line": r["label_line"], "fit": r["fit"], "nadir_mm": r["nadir_mm"],
                            "face_mean_linear_rgb": [float(v) for v in mean],
                            "decoded_mean": [float(v) for v in back.reshape(-1, 3).mean(axis=0)]})
        manifest["textures"][mesh] = entries
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    (out / "candidates.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "face"} for r in cands], indent=1) + "\n",
        encoding="utf-8")

    reasons = {}
    for r in cands:
        key = (r.get("mesh") or r["folder_class"], r.get("out") or "kept")
        reasons[key] = reasons.get(key, 0) + 1
    for (m, why), n in sorted(reasons.items()):
        print(f"{m:34s} {why:58s} {n}")
    for mesh, kept in kept_by_mesh.items():
        print(f"{mesh}: {len(kept)} textures, nadir {kept[0]['nadir_mm']:.0f}-{kept[-1]['nadir_mm']:.0f} mm, "
              f"IoU {min(r['fit']['iou'] for r in kept):.3f}-{max(r['fit']['iou'] for r in kept):.3f}"
              if kept else f"{mesh}: none")

    if a.montage:
        mdir = Path(a.montage)
        mdir.mkdir(parents=True, exist_ok=True)
        for mesh, kept in kept_by_mesh.items():
            tiles = []
            for r in kept:
                img = cv2.imread(str(yolo / r["image"]))
                sil = silhouette(T, intr, r["fit"]["x"], r["fit"]["y"], r["fit"]["yaw"], sizes[mesh], z0)
                cv2.polylines(img, [np.round(sil).astype(np.int32)], True, (0, 0, 255), 1)
                cv2.polylines(img, [np.round(np.array(r["top_uv"])).astype(np.int32)], True, (0, 255, 0), 1)
                c = np.array(r["top_uv"]).mean(axis=0)
                x0 = int(np.clip(c[0] - 160, 0, IMG_W - 320))
                y0 = int(np.clip(c[1] - 160, 0, IMG_H - 320))
                crop = img[y0:y0 + 320, x0:x0 + 320]
                tex = cv2.imread(str(REPO / r["texture"]))
                s = 320.0 / max(tex.shape[:2])
                tex = cv2.resize(tex, (int(tex.shape[1] * s), int(tex.shape[0] * s)),
                                 interpolation=cv2.INTER_NEAREST)
                pad = np.full((320, 320, 3), 255, np.uint8)
                pad[: tex.shape[0], : tex.shape[1]] = tex
                tile = np.hstack([crop, pad])
                cv2.putText(tile, f"{Path(r['texture']).stem} IoU {r['fit']['iou']:.3f}", (4, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
                tiles.append(tile)
            blank = np.full((320, 640, 3), 255, np.uint8)
            rows = [np.hstack(tiles[i:i + 3] + [blank] * (3 - len(tiles[i:i + 3])))
                    for i in range(0, len(tiles), 3)]
            if rows:
                cv2.imwrite(str(mdir / f"textures_{Path(mesh).stem}.jpg"), np.vstack(rows),
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
    return 0


if __name__ == "__main__":
    sys.exit(main())
