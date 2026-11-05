"""
auto_label.py
─────────────
Convert renderer instance-segmentation output → YOLOv8-seg training labels.

Contract with render_blenderproc.py (per frame, in the render output dir):
    frame_000123_rgb.png    RGB image (W x H)
    frame_000123_inst.png   instance map (uint8/uint16; 0 = background)
    frame_000123_meta.json  {"instances": {"<inst_id>": <class_id>, ...},
                             "scene": "scene_000123.json"}

Output: ultralytics dataset layout
    <out>/images/{train,val}/frame_*.png    hard links to the rendered images
    <out>/labels/{train,val}/frame_*.txt     one line per instance:
                                             class_id x0 y0 x1 y1 ... (normalized polygon)
    <out>/dataset.yaml

Polygon extraction uses cv2.findContours on each instance mask; the external
contours of a mask in several parts are joined into one polygon. Pure logic + cv2 →
testable with synthetic instance maps.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Instances smaller than this many pixels are dropped (matches the detector-side
# MIN_MASK_PIXELS spirit — tiny slivers make degenerate polygons).
MIN_INSTANCE_PIXELS = 50

# Paper dataset class order (5 box-shaped classes) — must match config/synthgen.yaml
# class_ids and dataset_check.EXPECTED_NAMES. (Sim/demo uses a separate set in
# detector.DEFAULT_CLASS_NAMES; the synthetic pipeline here is the paper dataset.)
# inox_box is the mirror-finish class contribution C4 rests on: it costs the D455
# 8.9% of its depth pixels at the median and 25.8% at the 90th percentile, an order
# of magnitude worse than the matte metal case (2.9%).
DEFAULT_CLASS_NAMES = ["carton", "plastic_box", "wood_box", "metal_box", "inox_box"]


def _join_contours(contours: list[np.ndarray]) -> np.ndarray:
    """Join external contours into one closed polygon by zero-width bridges.

    Each further contour is cut in at the pair of points nearest to the polygon
    built so far: the path runs out along the bridge, round the contour and back
    along the same bridge, so a polygon fill covers every part (plus the one-pixel
    bridge line). Ultralytics joins the parts of a multi-part COCO mask the same
    way (merge_multi_segment). Contours are taken largest first, so the result does
    not depend on the order findContours returns them in.
    """
    import cv2  # lazy import

    parts = sorted((c.reshape(-1, 2) for c in contours), key=cv2.contourArea, reverse=True)
    out = parts[0]
    for c in parts[1:]:
        d = ((out[:, None, :].astype(np.int64) - c[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        c = np.roll(c, -int(j), axis=0)
        out = np.concatenate([out[: i + 1], c, c[:1], out[i:]], axis=0)
    return out


def mask_to_polygon(
    mask: np.ndarray, min_area_px: int = MIN_INSTANCE_PIXELS
) -> list[float] | None:
    """Binary mask → normalized flat polygon [x0, y0, x1, y1, ...] or None.

    A mask in several parts is ONE polygon over all of them (_join_contours). The
    visible part of a box under a stacked box, turned against it, breaks into two to
    four pieces round the top box; keeping only the largest piece lost a median 38 %
    of the visible pixels of such boxes and shrank their box label (median IoU 0.35),
    in about one scene in six. Holes are not kept (RETR_EXTERNAL): the box under a
    stack is labelled over the top box too, which wins there, because Ultralytics
    draws the smaller mask last. Coordinates are normalized by image width/height.
    """
    import cv2  # lazy import

    m = (np.asarray(mask) > 0).astype(np.uint8)
    if int(m.sum()) < min_area_px:
        return None

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # A part of one or two pixels has no area as a contour; dropping it loses at
    # most those pixels.
    contours = [c for c in contours if len(c) >= 3]
    if not contours:
        return None
    if sum(cv2.contourArea(c) for c in contours) < min_area_px:
        return None
    pts = _join_contours(contours).astype(float)

    h, w = m.shape
    pts[:, 0] /= float(w)
    pts[:, 1] /= float(h)
    return [float(v) for v in pts.flatten()]


def count_parts(mask: np.ndarray) -> int:
    """Number of external contours (of three points or more) of a binary mask."""
    import cv2  # lazy import

    m = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in contours if len(c) >= 3)


def instance_map_to_yolo_lines(
    inst_map: np.ndarray,
    inst_to_class: dict[int, int],
    min_area_px: int = MIN_INSTANCE_PIXELS,
) -> list[str]:
    """Instance map + id→class mapping → YOLOv8-seg label lines.

    Instance ids present in the map but absent from the mapping (e.g.
    distractors, background plane) are skipped silently — that is exactly how
    distractors stay unlabelled.
    """
    lines: list[str] = []
    ids = [int(i) for i in np.unique(inst_map) if int(i) != 0]
    for inst_id in ids:
        if inst_id not in inst_to_class:
            continue
        poly = mask_to_polygon(inst_map == inst_id, min_area_px=min_area_px)
        if poly is None:
            logger.debug("Instance %d dropped (below %d px or degenerate)",
                         inst_id, min_area_px)
            continue
        coords = " ".join(f"{v:.6f}" for v in poly)
        lines.append(f"{inst_to_class[inst_id]} {coords}")
    return lines


def link_or_copy(source: str | Path, target: str | Path) -> str:
    """Hard-link source to target, or copy it when a link is impossible (another file
    system). An existing target is removed first, so a file that other hard links
    share is never written into. Returns "link" or "copy"."""
    target = Path(target)
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
        return "link"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def convert_render_output(
    render_dir: str | Path,
    out_dir: str | Path,
    val_frac: float = 0.1,
    seed: int = 0,
    class_names: list[str] | None = None,
) -> dict[str, int]:
    """Convert a full render directory into an ultralytics YOLO-seg dataset.

    Returns:
        Dict {frames, train, val, skipped, multipart, copied} with counts; multipart
        is the number of labelled instances whose mask came in two or more parts
        (joined into one polygon by mask_to_polygon), copied the number of images
        copied because a hard link to the render was impossible (0 on one disk).
    """
    import cv2  # lazy import

    render = Path(render_dir)
    out = Path(out_dir)
    names = class_names or DEFAULT_CLASS_NAMES

    rgb_files = sorted(render.glob("*_rgb.png"))
    if not rgb_files:
        raise FileNotFoundError(f"No *_rgb.png frames found in {render}")

    rng = np.random.default_rng(seed)
    n_val = int(round(len(rgb_files) * val_frac))
    val_idx = set(rng.choice(len(rgb_files), size=n_val, replace=False).tolist())

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"frames": 0, "train": 0, "val": 0, "skipped": 0, "multipart": 0, "copied": 0}
    for i, rgb_path in enumerate(rgb_files):
        stem = rgb_path.name[: -len("_rgb.png")]
        inst_path = render / f"{stem}_inst.png"
        meta_path = render / f"{stem}_meta.json"
        if not inst_path.exists() or not meta_path.exists():
            logger.warning("Frame %s missing inst/meta — skipped", stem)
            stats["skipped"] += 1
            continue

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta.get("instances"), dict):
            # Without the mapping every part would go unlabelled: a negative image.
            logger.warning("Frame %s: meta has no 'instances' — skipped", stem)
            stats["skipped"] += 1
            continue
        inst_to_class = {int(k): int(v) for k, v in meta["instances"].items()}
        inst_map = cv2.imread(str(inst_path), cv2.IMREAD_UNCHANGED)
        if inst_map is None:
            logger.warning("Frame %s: unreadable instance map — skipped", stem)
            stats["skipped"] += 1
            continue
        if inst_map.ndim == 3:  # safety: some writers save 3-channel
            inst_map = inst_map[:, :, 0]

        lines = instance_map_to_yolo_lines(inst_map, inst_to_class)
        stats["multipart"] += sum(
            1 for k in inst_to_class
            if (inst_map == k).sum() >= MIN_INSTANCE_PIXELS and count_parts(inst_map == k) >= 2)
        split = "val" if i in val_idx else "train"
        # A hard link, not a copy: the image exists once on disk however many
        # datasets and packages hold it (1.2 MB a frame, 3000 frames an arm).
        stats["copied"] += link_or_copy(rgb_path, out / "images" / split / f"{stem}.png") == "copy"
        label_path = out / "labels" / split / f"{stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""),
                              encoding="utf-8")
        stats["frames"] += 1
        stats[split] += 1

    # The render manifest travels with the dataset, so that packaging can check that
    # the synthetic arms it combines or compares were rendered the same way
    # (hard_profile.check_arms_match).
    manifest = render / "render_manifest.json"
    if manifest.exists():
        shutil.copy2(manifest, out / "render_manifest.json")
    else:
        logger.warning("%s has no render_manifest.json: the dataset cannot show how it "
                       "was rendered, and packaging cannot compare it with other arms", render)

    yaml_text = (
        f"# Auto-generated by src/synthgen/auto_label.py\n"
        f"path: {out.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    )
    (out / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    logger.info("Dataset written to %s (%d train / %d val, %d skipped)",
                out, stats["train"], stats["val"], stats["skipped"])
    return stats
