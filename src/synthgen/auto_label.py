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
    <out>/images/{train,val}/frame_*.png
    <out>/labels/{train,val}/frame_*.txt     one line per instance:
                                             class_id x0 y0 x1 y1 ... (normalized polygon)
    <out>/dataset.yaml

Polygon extraction uses cv2.findContours on each instance mask (largest
external contour). Pure logic + cv2 → testable with synthetic instance maps.
"""
from __future__ import annotations

import json
import logging
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


def mask_to_polygon(
    mask: np.ndarray, min_area_px: int = MIN_INSTANCE_PIXELS
) -> list[float] | None:
    """Binary mask → normalized flat polygon [x0, y0, x1, y1, ...] or None.

    Takes the largest external contour; fragmented instances therefore yield
    the dominant fragment (a warning is logged by the caller when fragments
    are dropped). Coordinates are normalized by image width/height.
    """
    import cv2  # lazy import

    m = (np.asarray(mask) > 0).astype(np.uint8)
    if int(m.sum()) < min_area_px:
        return None

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area_px:
        return None
    if len(contour) < 3:
        return None

    h, w = m.shape
    pts = contour.reshape(-1, 2).astype(float)
    pts[:, 0] /= float(w)
    pts[:, 1] /= float(h)
    return [float(v) for v in pts.flatten()]


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


def convert_render_output(
    render_dir: str | Path,
    out_dir: str | Path,
    val_frac: float = 0.1,
    seed: int = 0,
    class_names: list[str] | None = None,
) -> dict[str, int]:
    """Convert a full render directory into an ultralytics YOLO-seg dataset.

    Returns:
        Dict {frames, train, val, skipped} with counts.
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

    stats = {"frames": 0, "train": 0, "val": 0, "skipped": 0}
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
        inst_to_class = {int(k): int(v) for k, v in meta.get("instances", {}).items()}
        inst_map = cv2.imread(str(inst_path), cv2.IMREAD_UNCHANGED)
        if inst_map is None:
            logger.warning("Frame %s: unreadable instance map — skipped", stem)
            stats["skipped"] += 1
            continue
        if inst_map.ndim == 3:  # safety: some writers save 3-channel
            inst_map = inst_map[:, :, 0]

        lines = instance_map_to_yolo_lines(inst_map, inst_to_class)
        split = "val" if i in val_idx else "train"
        shutil.copy2(rgb_path, out / "images" / split / f"{stem}.png")
        label_path = out / "labels" / split / f"{stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""),
                              encoding="utf-8")
        stats["frames"] += 1
        stats[split] += 1

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
