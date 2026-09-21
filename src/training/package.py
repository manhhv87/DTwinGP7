"""Validate and package the paper's training inputs without starting training.

The package contains relative image/label paths, real-only validation, and
content hashes. Hash checks detect exact duplicates; they cannot establish
independence of capture sessions or detect visually similar frames.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import yaml

from src.utils.dataset_check import EXPECTED_NAMES, IMAGE_EXTS


ULTRALYTICS_VERSION = "8.4.66"
RECIPE: dict[str, Any] = {
    "epochs": 130, "imgsz": 1280, "batch": 8, "nbs": 64, "patience": 0,
    "optimizer": "AdamW", "lr0": 0.001111, "lrf": 0.01,
    "momentum": 0.9, "weight_decay": 0.0005,
    "warmup_epochs": 3.0, "warmup_momentum": 0.8, "warmup_bias_lr": 0.0,
    "cos_lr": False, "close_mosaic": 10, "mosaic": 1.0,
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    "translate": 0.1, "scale": 0.5, "fliplr": 0.5, "flipud": 0.0,
    "degrees": 0.0, "shear": 0.0, "perspective": 0.0,
    "mixup": 0.0, "copy_paste": 0.0,
    "deterministic": True, "save": True, "val": True,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _names(root: Path) -> list[dict[str, str]]:
    """Require the five canonical IDs, including in every available YAML."""
    paths = [root / name for name in ("dataset.yaml", "data.yaml")
             if (root / name).is_file()]
    if not paths:
        raise ValueError(f"No dataset.yaml or data.yaml in {root}")
    result = []
    for path in paths:
        try:
            metadata = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        except yaml.YAMLError as exc:
            raise ValueError(f"{path}: invalid dataset YAML: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"{path}: dataset metadata must be a mapping")
        names = metadata.get("names")
        if isinstance(names, dict):
            keyed: dict[int, Any] = {}
            for key, value in names.items():
                if isinstance(key, bool):
                    raise ValueError(f"{path}: class IDs must be consecutive integers")
                try:
                    integer = int(key)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path}: invalid class ID {key!r}") from exc
                if str(integer) != str(key) or integer in keyed:
                    raise ValueError(f"{path}: invalid or repeated class ID {key!r}")
                keyed[integer] = value
            if set(keyed) != set(range(len(EXPECTED_NAMES))):
                raise ValueError(f"{path}: class IDs must be exactly 0 through 4")
            names = [keyed[index] for index in range(len(EXPECTED_NAMES))]
        if names != EXPECTED_NAMES:
            raise ValueError(f"{path}: names must be in canonical order {EXPECTED_NAMES!r}; got {names!r}")
        if "nc" in metadata and metadata["nc"] != len(EXPECTED_NAMES):
            raise ValueError(f"{path}: nc must be {len(EXPECTED_NAMES)}")
        result.append({"path": str(path), "sha256": sha256(path)})
    return result


def _labels(path: Path) -> list[int]:
    if not path.is_file():
        raise ValueError(f"Missing label {path}; use an empty label file for an intentional negative")
    classes: list[int] = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        tokens = line.split()
        if not tokens:
            continue
        prefix = f"{path}:{number}"
        if len(tokens) < 7 or (len(tokens) - 1) % 2:
            raise ValueError(f"{prefix}: a segmentation polygon requires at least three x/y pairs")
        try:
            class_id = int(tokens[0])
            coordinates = [float(token) for token in tokens[1:]]
        except ValueError as exc:
            raise ValueError(f"{prefix}: invalid numeric polygon label") from exc
        if class_id not in range(len(EXPECTED_NAMES)):
            raise ValueError(f"{prefix}: class ID must be between 0 and 4")
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"{prefix}: coordinates must be finite and normalized to [0, 1]")
        classes.append(class_id)
    return classes


def _allowlist(path: Path, available: list[Path], image_root: Path) -> list[Path]:
    """Resolve exported paths or the raw capture split's slash-to-__ names."""
    aliases: dict[str, set[Path]] = {}
    for image in available:
        relative = image.relative_to(image_root).as_posix()
        for alias in (relative, image.name):
            aliases.setdefault(alias.casefold(), set()).add(image)
    selected = []
    seen: set[Path] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        entry = line.strip().replace("\\", "/")
        if not entry or entry.startswith("#"):
            continue
        candidates = aliases.get(entry.casefold(), set()) | aliases.get(entry.replace("/", "__").casefold(), set())
        if len(candidates) != 1:
            raise ValueError(f"{path}:{number}: allowlist entry is missing or ambiguous: {entry}")
        image = next(iter(candidates))
        if image in seen:
            raise ValueError(f"{path}:{number}: duplicate allowlist entry: {entry}")
        seen.add(image)
        selected.append(image)
    if not selected:
        raise ValueError(f"{path}: empty training allowlist")
    return sorted(selected, key=lambda item: item.as_posix().casefold())


def inspect_sources(
    real: Path,
    synthetic: list[Path],
    *,
    real_train_list: Path | None = None,
    expected_real_train: int | None = None,
    expected_synth_train: int | None = None,
) -> dict[str, Any]:
    """Inspect inputs without modifying them or constructing a package."""
    real = Path(real).resolve()
    synthetic = [Path(path).resolve() for path in synthetic]
    roots = [real, *synthetic]
    if len(set(roots)) != len(roots):
        raise ValueError("Each real/synthetic source must be a distinct directory")
    for expected in (expected_real_train, expected_synth_train):
        if expected is not None and expected < 0:
            raise ValueError("Expected image counts cannot be negative")
    records: list[dict[str, Any]] = []
    sources = []
    content_seen: dict[str, Path] = {}
    counts = {"real_train": 0, "real_val": 0, "synthetic_train": 0, "train": 0, "val": 0}
    for index, root in enumerate(roots):
        name = "real" if index == 0 else f"synth{index:02d}"
        source = {"name": name, "root": str(root), "metadata": _names(root), "counts": {}}
        for split in (["train", "val"] if index == 0 else ["train"]):
            image_root = root / "images" / split
            label_root = root / "labels" / split
            if not image_root.is_dir():
                raise ValueError(f"Missing image directory: {image_root}")
            images = sorted((path for path in image_root.rglob("*")
                             if path.is_file() and path.suffix.lower() in IMAGE_EXTS),
                            key=lambda item: item.as_posix().casefold())
            available_count = len(images)
            if index == 0 and split == "train" and real_train_list is not None:
                images = _allowlist(Path(real_train_list), images, image_root)
            if not images:
                raise ValueError(f"No images selected in {image_root}")
            label_targets: set[str] = set()
            image_targets: set[str] = set()
            for image in images:
                relative = image.relative_to(image_root)
                if not image.resolve().is_relative_to(image_root.resolve()):
                    raise ValueError(f"Image symlink escapes the source image directory: {image}")
                label = label_root / relative.with_suffix(".txt")
                if not label.resolve().is_relative_to(label_root.resolve()):
                    raise ValueError(f"Label symlink escapes the source label directory: {label}")
                image_key = relative.as_posix().casefold()
                label_key = relative.with_suffix(".txt").as_posix().casefold()
                if image_key in image_targets or label_key in label_targets:
                    raise ValueError(f"Image/label filename collision in {image_root}: {relative}")
                image_targets.add(image_key)
                label_targets.add(label_key)
                classes = _labels(label)
                image_hash = sha256(image)
                if image_hash in content_seen:
                    raise ValueError(f"Duplicate image content (possible leakage or repeated budget): {image} and {content_seen[image_hash]}")
                content_seen[image_hash] = image
                records.append({
                    "source": name, "split": split,
                    "source_image": str(image), "source_label": str(label),
                    "image": (Path("images") / split / name / relative).as_posix(),
                    "label": (Path("labels") / split / name / relative.with_suffix(".txt")).as_posix(),
                    "image_sha256": image_hash, "label_sha256": sha256(label),
                    "class_ids": classes,
                })
            count = len(images)
            source["counts"][split] = {"available": available_count, "selected": count}
            counts[split] += count
            counts[f"real_{split}" if index == 0 else "synthetic_train"] += count
        sources.append(source)
    for key, expected in (("real_train", expected_real_train), ("synthetic_train", expected_synth_train)):
        if expected is not None and counts[key] != expected:
            raise ValueError(f"Expected {expected} {key} images, found {counts[key]}; reconcile the source manifest before packaging")
    histogram = {split: dict(Counter(class_id for row in records if row["split"] == split
                                     for class_id in row["class_ids"])) for split in ("train", "val")}
    return {
        "schema_version": 1, "classes": list(EXPECTED_NAMES),
        "counts": counts, "sources": sources, "class_instances": histogram,
        "real_train_allowlist": None if real_train_list is None else {
            "path": str(Path(real_train_list).resolve()), "sha256": sha256(Path(real_train_list)),
        },
        "validation_role": "model_selection_only; not an independent final test set",
        "independence_check": "Exact image-content duplicates rejected; session separation and near-duplicate checks still require a frozen capture manifest.",
        "images": records,
    }


def build_package(
    real: Path,
    synthetic: list[Path],
    destination: Path,
    *,
    experiment: str,
    version: int,
    model: str | Path = "yolov8s-seg.pt",
    epochs: int | None = None,
    seeds: list[int] | None = None,
    real_train_list: Path | None = None,
    expected_real_train: int | None = None,
    expected_synth_train: int | None = None,
    inspect_only: bool = False,
    runner_source: Path | None = None,
) -> dict[str, Any]:
    """Copy verified inputs to a new portable directory; never overwrite one."""
    experiment = experiment.upper()
    if experiment not in {"E2", "E3", "E4", "E5"} or version < 0:
        raise ValueError("Use experiment E2/E3/E4/E5 and a nonnegative dataset version")
    if experiment == "E2" and synthetic:
        raise ValueError("E2 is real-only; use E3/E4/E5 for synthetic training sources")
    if experiment != "E2" and not synthetic:
        raise ValueError(f"{experiment} requires synthetic training sources")
    if seeds is None:
        seeds = list(range(5 if experiment in {"E2", "E3"} else 3))
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Training seeds must be unique nonnegative integers")
    if epochs is not None and (type(epochs) is not int or epochs < 1):
        raise ValueError("Epoch count must be a positive integer")
    initializer = None
    if experiment == "E4":
        if str(model) == "yolov8s-seg.pt" or epochs is None:
            raise ValueError("E4 requires --model pointing to the selected parent checkpoint and explicit --epochs for the frozen fine-tuning budget")
        initializer = Path(model).resolve()
        if not initializer.is_file():
            raise ValueError(f"Parent checkpoint does not exist: {initializer}")
        initialization = {"model": "initial.pt", "sha256": sha256(initializer), "source": str(initializer)}
    else:
        if str(model) != "yolov8s-seg.pt":
            raise ValueError(f"{experiment} starts from the common COCO yolov8s-seg.pt initializer; custom parent checkpoints belong to E4")
        initialization = {"model": "yolov8s-seg.pt"}
    recipe = dict(RECIPE)
    if epochs is not None:
        recipe["epochs"] = epochs
    training = {
        "schema_version": 1, "experiment": experiment, "version": version,
        "seeds": seeds, "ultralytics_version": ULTRALYTICS_VERSION,
        "recipe": recipe, "initialization": initialization,
        "selection": {"metric": "sum_box_mask_map50_95", "checkpoint_tie": "latest_epoch", "seed_tie": "first_in_declared_seed_order"},
    }
    destination = Path(destination).resolve()
    for root in [Path(real), *[Path(path) for path in synthetic]]:
        if destination.is_relative_to(root.resolve()):
            raise ValueError(f"Package destination must not be inside a source dataset: {root}")
    if destination.exists() and not inspect_only:
        raise ValueError(f"Refusing to overwrite existing package: {destination}")
    manifest = inspect_sources(
        real, synthetic, real_train_list=real_train_list,
        expected_real_train=expected_real_train, expected_synth_train=expected_synth_train,
    )
    manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
    if inspect_only:
        return {"inspect_only": True, "destination": str(destination), "training": training, "manifest": manifest}
    runner_source = Path(runner_source) if runner_source else Path(__file__).resolve().parents[2] / "scripts" / "train_packaged_model.py"
    if not runner_source.is_file():
        raise ValueError(f"Portable training runner is missing: {runner_source}")
    destination.mkdir(parents=True)
    for row in manifest["images"]:
        for kind in ("image", "label"):
            target = destination / row[kind]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row[f"source_{kind}"], target)
            if sha256(target) != row[f"{kind}_sha256"]:
                raise ValueError(f"Source content changed during packaging: {row[f'source_{kind}']}; package is incomplete")
    if initializer is not None:
        shutil.copy2(initializer, destination / "initial.pt")
        if sha256(destination / "initial.pt") != initialization["sha256"]:
            raise ValueError("Parent checkpoint changed during packaging; package is incomplete")
    dataset = {"train": "images/train", "val": "images/val", "nc": len(EXPECTED_NAMES), "names": list(EXPECTED_NAMES)}
    (destination / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    for filename, content in (("manifest.json", manifest), ("training.json", training)):
        (destination / filename).write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shutil.copy2(runner_source, destination / "train.py")
    (destination / "requirements-training.txt").write_text(
        f"ultralytics=={ULTRALYTICS_VERSION}\nPyYAML>=6.0\n", encoding="utf-8")
    (destination / "README.md").write_text(
        "# Portable training package\n\n"
        "Copy this entire directory to the Linux training machine. Install a CUDA-compatible PyTorch build for that machine, then run:\n\n"
        "```sh\npython -m pip install -r requirements-training.txt\npython train.py --dry-run\npython train.py --device 0\n```\n\n"
        "The dry run verifies every image and label hash and does not train. The runner resolves paths after transfer. "
        "Use `--seeds` to run a declared subset. Existing run directories are not overwritten. "
        "Keep manifest.json, training.json and every run record with returned checkpoints. "
        "Validation selects models and is not an independent final test set. "
        "Content hashes do not establish capture-session independence. "
        "The source paths in manifest.json are provenance only; they are not needed on Linux.\n", encoding="utf-8")
    return {"inspect_only": False, "destination": str(destination), "counts": manifest["counts"],
            "manifest_sha256": sha256(destination / "manifest.json"), "training": training}
