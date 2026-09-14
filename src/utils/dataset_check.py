"""
dataset_check.py
────────────────
Verify (and repair) an ultralytics YOLO-seg dataset before training.

Why this exists: the single most damaging dataset bug is a SILENT class-id
mismatch — annotation tools order classes alphabetically
(Carton, Inox, Metal, Plastic, Wood) while the paper's canonical order is
(carton, plastic_box, wood_box, metal_box, inox_box) = EXPECTED_NAMES.
A model trained on shifted ids "works" but swaps classes at run time.

This is not hypothetical: the first Roboflow export of this dataset used exactly
that alphabetical order, so mapping by id would have mislabelled 4 of the 5
classes with no error raised anywhere. Always map by NAME, never by index.

Checks performed:
  - dataset.yaml / data.yaml present; class names read (list or dict form);
    compared against the canonical order;
  - image <-> label pairing per split (missing labels counted; empty label
    files = intentional negatives);
  - label syntax: class id in range, coordinates normalized to [0, 1],
    polygon lines have 1 + 2k tokens with k >= 3;
  - per-class instance histogram (checked against the >=180/class target
    by the caller);
  - filename leakage across splits (same stem in train AND val).

`remap_labels` rewrites every label file from the yaml's order to the
canonical order (backing up the original labels once) and writes a clean
canonical dataset.yaml.

Pure stdlib + yaml (+ optional cv2 for image-size sampling) → unit-testable.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from .helpers import load_yaml

logger = logging.getLogger(__name__)

# Canonical class order for the PAPER dataset (5 box-shaped classes) — must match
# config/synthgen.yaml objects.class_ids and auto_label.DEFAULT_CLASS_NAMES.
EXPECTED_NAMES = ["carton", "plastic_box", "wood_box", "metal_box", "inox_box"]

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _find_yaml(root: Path) -> Path | None:
    for name in ("dataset.yaml", "data.yaml"):
        if (root / name).exists():
            return root / name
    return None


def load_names(yaml_path: str | Path) -> list[str]:
    """Read the class-name list from a dataset yaml (list or {id: name} dict)."""
    data = load_yaml(yaml_path)
    names = data.get("names")
    if names is None:
        raise ValueError(f"{yaml_path}: no 'names' key")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    return [str(n) for n in names]


def parse_label_file(path: Path, n_classes: int) -> tuple[list[int], list[str]]:
    """Parse one YOLO-seg label file.

    Returns:
        (class_ids, problems) — one id per valid line; human-readable problem
        strings for every malformed line.
    """
    ids: list[int] = []
    problems: list[str] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ids, problems                    # empty = negative image, valid
    for ln, line in enumerate(text.splitlines(), start=1):
        tok = line.split()
        if len(tok) < 7 or len(tok) % 2 == 0:
            problems.append(f"{path.name}:{ln}: expected 'id x0 y0 x1 y1 x2 y2 ...' "
                            f"(odd token count >= 7), got {len(tok)} tokens")
            continue
        try:
            cid = int(tok[0])
            coords = [float(v) for v in tok[1:]]
        except ValueError:
            problems.append(f"{path.name}:{ln}: non-numeric token")
            continue
        if not (0 <= cid < n_classes):
            problems.append(f"{path.name}:{ln}: class id {cid} outside [0, {n_classes - 1}]")
            continue
        if any(v < -1e-6 or v > 1 + 1e-6 for v in coords):
            problems.append(f"{path.name}:{ln}: coordinate outside [0, 1] "
                            f"(forgot normalization?)")
            continue
        ids.append(cid)
    return ids, problems


def verify_dataset(
    root: str | Path,
    expected_names: list[str] | None = None,
    splits: tuple[str, ...] = ("train", "val"),
    max_label_problems: int = 20,
) -> dict[str, Any]:
    """Run all checks. Returns a JSON-able report; report['ok'] is the verdict."""
    root = Path(root)
    expected = expected_names or EXPECTED_NAMES
    report: dict[str, Any] = {
        "root": str(root), "errors": [], "warnings": [],
        "splits": {}, "class_counts": {}, "ok": False,
    }

    yaml_path = _find_yaml(root)
    if yaml_path is None:
        report["errors"].append("no dataset.yaml / data.yaml in dataset root")
        return report
    try:
        names = load_names(yaml_path)
    except Exception as e:  # noqa: BLE001
        report["errors"].append(f"cannot read names from {yaml_path.name}: {e}")
        return report
    report["yaml"] = yaml_path.name
    report["names"] = names
    report["names_match"] = names == expected
    if not report["names_match"]:
        report["errors"].append(
            f"class order in {yaml_path.name} is {names}, canonical order is "
            f"{expected} — run remap (24_verify_dataset.py --fix) BEFORE training"
        )

    stems_per_split: dict[str, set[str]] = {}
    counts = {n: 0 for n in names}
    n_problems = 0
    for split in splits:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        info = {"images": 0, "labelled": 0, "negatives": 0, "missing_label": 0}
        if not img_dir.exists():
            report["errors"].append(f"missing directory images/{split}")
            report["splits"][split] = info
            continue
        stems = set()
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            info["images"] += 1
            stems.add(img.stem)
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                info["missing_label"] += 1
                continue
            ids, problems = parse_label_file(lbl, n_classes=len(names))
            for p in problems:
                n_problems += 1
                if n_problems <= max_label_problems:
                    report["errors"].append(f"labels/{split}/{p}")
            if ids:
                info["labelled"] += 1
                for cid in ids:
                    counts[names[cid]] += 1
            else:
                info["negatives"] += 1
        stems_per_split[split] = stems
        if info["missing_label"]:
            report["warnings"].append(
                f"{split}: {info['missing_label']} image(s) without a label file "
                f"(YOLO treats them as background — intended?)")
        report["splits"][split] = info
    if n_problems > max_label_problems:
        report["errors"].append(
            f"... and {n_problems - max_label_problems} more label problems")

    # Cross-split leakage by filename stem.
    split_list = [s for s in splits if s in stems_per_split]
    for i in range(len(split_list)):
        for j in range(i + 1, len(split_list)):
            common = stems_per_split[split_list[i]] & stems_per_split[split_list[j]]
            if common:
                report["errors"].append(
                    f"LEAKAGE: {len(common)} filename(s) in both "
                    f"{split_list[i]} and {split_list[j]}, e.g. {sorted(common)[:3]}")

    report["class_counts"] = counts
    report["ok"] = not report["errors"]
    return report


def remap_labels(
    root: str | Path,
    expected_names: list[str] | None = None,
    splits: tuple[str, ...] = ("train", "val"),
) -> dict[str, Any]:
    """Rewrite label class ids from the yaml's order to the canonical order.

    Backs up the original labels/ tree to labels_orig/ (only once), rewrites
    every .txt in place, and writes a canonical dataset.yaml. No-op when the
    yaml order already matches.
    """
    root = Path(root)
    expected = expected_names or EXPECTED_NAMES
    yaml_path = _find_yaml(root)
    if yaml_path is None:
        raise FileNotFoundError("no dataset.yaml / data.yaml in dataset root")
    names = load_names(yaml_path)
    missing = set(names) ^ set(expected)
    if missing:
        raise ValueError(f"class-name sets differ: {sorted(missing)} — "
                         f"remap only reorders identical name sets")

    result: dict[str, Any] = {"mapping": {}, "changed_files": 0, "noop": False}
    if names == expected:
        result["noop"] = True
    else:
        mapping = {old_id: expected.index(name) for old_id, name in enumerate(names)}
        result["mapping"] = {names[k]: v for k, v in mapping.items()}

        backup = root / "labels_orig"
        if not backup.exists() and (root / "labels").exists():
            shutil.copytree(root / "labels", backup)
            logger.info("Original labels backed up to %s", backup)

        for split in splits:
            lbl_dir = root / "labels" / split
            if not lbl_dir.exists():
                continue
            for txt in sorted(lbl_dir.glob("*.txt")):
                lines_out = []
                changed = False
                for line in txt.read_text(encoding="utf-8").splitlines():
                    tok = line.split()
                    if tok and tok[0].isdigit() and int(tok[0]) in mapping:
                        new_id = mapping[int(tok[0])]
                        if new_id != int(tok[0]):
                            changed = True
                        tok[0] = str(new_id)
                    lines_out.append(" ".join(tok))
                if changed:
                    txt.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
                    result["changed_files"] += 1

    # Canonical dataset.yaml (also normalizes a Roboflow data.yaml).
    yaml_text = (
        "# Canonical dataset.yaml — written by src/utils/dataset_check.py\n"
        f"path: {root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(expected))
    )
    (root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    logger.info("Canonical dataset.yaml written (%s)",
                "order unchanged" if result["noop"] else f"remapped {result['mapping']}")
    return result


def format_report(report: dict[str, Any]) -> str:
    """Human-readable multi-line summary for CLI printing."""
    lines = [f"Dataset: {report['root']}"]
    if "names" in report:
        mark = "OK" if report.get("names_match") else "MISMATCH"
        lines.append(f"Classes ({report.get('yaml')}): {report['names']}  [{mark}]")
    for split, info in report.get("splits", {}).items():
        lines.append(
            f"  {split:<6} images={info['images']:<5} labelled={info['labelled']:<5} "
            f"negatives={info['negatives']:<4} missing_label={info['missing_label']}")
    if report.get("class_counts"):
        lines.append("  instances/class: "
                     + ", ".join(f"{k}={v}" for k, v in report["class_counts"].items()))
    for w in report.get("warnings", []):
        lines.append(f"  WARN  {w}")
    for e in report.get("errors", []):
        lines.append(f"  ERROR {e}")
    lines.append("VERDICT: " + ("OK — ready for training" if report.get("ok")
                                else "NOT READY — fix the errors above"))
    return "\n".join(lines)
