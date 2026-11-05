"""Validate and package the paper's training inputs without starting training.

The package contains relative image/label paths, real-only validation, and
content hashes. Hash checks detect exact duplicates; they cannot establish
independence of capture sessions or detect visually similar frames. Synthetic
sources whose scene manifest sits beside them (<batch>/specs/manifest.json next
to <batch>/dataset) must not share generation streams (the seed/index rule of
their spec version, src/synthgen/seed_ranges.py). Synthetic sources that carry the
render manifest of their batch (render_manifest.json, copied by auto_label) must have
been rendered the same way: settings, device, Blender and BlenderProc versions and
scene profile (src/synthgen/hard_profile.check_arms_match), within a package and
against the other arm's package of an E4 update.

An E4 package names the package that trained its parent checkpoint. It must keep
that package's real training and validation images and all of its synthetic
images, and an update-2 package must continue the same arm and seed with the same
schedule. A package of the other arm, when given, must match in schedule, seeds,
real data and image counts.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import yaml

from src.synthgen.hard_profile import MANIFEST_NAME as RENDER_MANIFEST
from src.synthgen.hard_profile import MATCH_KEYS as RENDER_MATCH_KEYS
from src.synthgen.hard_profile import check_arms_match
from src.synthgen.seed_ranges import STREAM_RULE, manifest_streams, overlaps
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
SELECTION = {"metric": "sum_box_mask_map50_95", "checkpoint_tie": "latest_epoch",
             "kappa": "highest_mean_over_seeds", "kappa_tie": "smaller_kappa",
             "seed_tie": "lowest_seed"}
E4_ARMS = ("loop", "control")
E4_MAX_UPDATES = 2


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
        if index:
            specs = root.parent / "specs" / "manifest.json"
            source["generation"] = None
            if specs.is_file():
                source["generation"] = manifest_streams(specs)
                source["generation"]["sha256"] = sha256(specs)
            rendered = root / RENDER_MANIFEST
            source["render"] = None
            if rendered.is_file():
                record = _read_json(rendered)
                source["render"] = {**{key: record.get(key) for key in RENDER_MATCH_KEYS},
                                    "sha256": sha256(rendered)}
        sources.append(source)
    streams = {source["name"]: source["generation"]["streams"] for source in sources[1:]
               if source["generation"] and source["generation"]["streams"] is not None}
    clashes = overlaps(streams)
    if clashes:
        raise ValueError("Synthetic sources share generation streams: " + "; ".join(
            f"{first} and {second}, {count} scene(s)" for first, second, count in clashes))
    renders = {source["name"]: source["render"] for source in sources[1:] if source["render"]}
    check_arms_match(renders)
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
        "expected_real_train": expected_real_train,
        "generation_overlap_check": {
            "rules": {source["name"]: STREAM_RULE[source["generation"]["spec_version"]]
                      for source in sources[1:] if source["name"] in streams},
            "checked": sorted(streams),
            "unchecked": [source["name"] for source in sources[1:] if source["name"] not in streams],
        },
        "render_check": {
            "match_keys": list(RENDER_MATCH_KEYS),
            "checked": sorted(renders),
            "unchecked": [source["name"] for source in sources[1:] if source["name"] not in renders],
        },
        "validation_role": "model_selection_only; not an independent final test set",
        "independence_check": "Exact image-content duplicates rejected; session separation and near-duplicate checks still require a frozen capture manifest.",
        "images": records,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc


def _content(manifest: dict[str, Any], split: str, real: bool) -> set[tuple[str, str]]:
    """Image and label hashes of one split's real or synthetic records."""
    return {(row["image_sha256"], row["label_sha256"]) for row in manifest["images"]
            if row["split"] == split and (row["source"] == "real") == real}


def _parent_lineage(parent: Path, checkpoint_sha256: str) -> dict[str, Any]:
    """The package that trained the parent checkpoint, and that checkpoint's run."""
    parent = Path(parent).resolve()
    training = _read_json(parent / "training.json")
    manifest = _read_json(parent / "manifest.json")
    manifest_sha256 = sha256(parent / "manifest.json")
    runs = [record for record in (_read_json(path) for path in sorted((parent / "runs").glob("seed*/run_record.json")))
            if record.get("status") == "complete" and record.get("checkpoint_sha256") == checkpoint_sha256]
    if len(runs) != 1:
        raise ValueError(f"{parent}: {len(runs)} complete run records hold the parent checkpoint's SHA-256; "
                         "give the package that trained it, with its runs/seed<n>/run_record.json")
    if runs[0].get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"{parent}: the parent run was trained on another manifest.json")
    return {"root": parent, "training": training, "manifest": manifest, "run": runs[0],
            "record": {"package": str(parent), "experiment": training.get("experiment"),
                       "training_sha256": sha256(parent / "training.json"),
                       "manifest_sha256": manifest_sha256, "run_seed": runs[0].get("seed"),
                       "checkpoint_sha256": checkpoint_sha256}}


def _e4_update(lineage: dict[str, Any], arm: str, seeds: list[int], recipe: dict[str, Any]) -> int:
    """Update number of an E4 package; an update continues its own arm, seed and schedule."""
    parent = lineage["training"]
    if parent.get("experiment") == "E3":
        return 1
    if parent.get("experiment") != "E4":
        raise ValueError("E4 starts from the selected E3 checkpoint; the parent package must be E3 or an earlier E4 update")
    previous = parent.get("e4") or {}
    if previous.get("arm") != arm:
        raise ValueError(f"An update continues its own arm: the parent package is arm {previous.get('arm')!r}, not {arm!r}")
    seed = lineage["run"].get("seed")
    if seeds != [seed]:
        raise ValueError(f"After the first update each seed continues its own checkpoint: the parent is seed "
                         f"{seed}; package it with --seeds {seed}")
    if parent.get("recipe") != recipe:
        raise ValueError(f"Every update repeats the same fine-tuning schedule: the parent update used epochs "
                         f"{parent.get('recipe', {}).get('epochs')}, this package {recipe['epochs']}")
    update = int(previous.get("update", 0)) + 1
    if update > E4_MAX_UPDATES:
        raise ValueError(f"The loop permits at most {E4_MAX_UPDATES} guided updates")
    return update


def _check_e4_data(lineage: dict[str, Any], manifest: dict[str, Any]) -> None:
    """E4 keeps the parent's real and synthetic images and adds new synthetic ones."""
    parent = lineage["manifest"]
    for split in ("train", "val"):
        if _content(parent, split, True) != _content(manifest, split, True):
            raise ValueError(f"E4 keeps the parent's real {split} images; this real {split} set differs from {lineage['root']}")
    kept = _content(parent, "train", False)
    missing = kept - _content(manifest, "train", False)
    if missing:
        raise ValueError(f"E4 keeps the parent's synthetic images; {len(missing)} of its {len(kept)} are missing")
    if len(_content(manifest, "train", False)) == len(kept):
        raise ValueError("This E4 update adds no synthetic images to the parent's")


def _kappas(manifest: dict[str, Any]) -> dict[str, Any]:
    """kappa of each synthetic source, from its specs/manifest.json (None when unrecorded)."""
    return {f"{source['name']} ({source['root']})": (source.get("generation") or {}).get("kappa")
            for source in manifest["sources"][1:]}


def _check_e4_kappa(lineage: dict[str, Any], manifest: dict[str, Any], *, require: bool) -> float | None:
    """Every batch of an E4 package, new and kept, is sampled at the kappa of f0.

    The parent's synthetic images are the E3 arm that trained f0 (or the earlier update,
    which kept them), so one recorded kappa over this package and its parent is the
    kappa of f0; a loop or control batch generated with another (or a forgotten) --kappa
    is refused here instead of being trained. require: every source must record one.
    """
    recorded = {**_kappas(lineage["manifest"]), **_kappas(manifest)}
    if require and any(value is None for value in recorded.values()):
        unrecorded = sorted(name for name, value in recorded.items() if value is None)
        raise ValueError(f"E4 batches must be anchored and record their kappa; none for {unrecorded}")
    values = {value for value in recorded.values() if value is not None}
    if len(values) > 1:
        raise ValueError("E4 samples every batch at the kappa of f0: " + "; ".join(
            f"{name} kappa {value}" for name, value in sorted(recorded.items())))
    return next(iter(values)) if values else None


def _check_e4_pair(paired: Path, training: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """The other arm's package of the same update must share schedule, seeds, real data and counts."""
    paired = Path(paired).resolve()
    other = _read_json(paired / "training.json")
    other_manifest = _read_json(paired / "manifest.json")
    mine, theirs = training["e4"], other.get("e4") or {}
    if other.get("experiment") != "E4" or theirs.get("arm") == mine["arm"] or theirs.get("update") != mine["update"]:
        raise ValueError(f"{paired} is not the other arm's package of update {mine['update']}")
    if other.get("recipe") != training["recipe"]:
        raise ValueError(f"Both arms use the same fine-tuning schedule: epochs {other.get('recipe', {}).get('epochs')} "
                         f"in {paired}, {training['recipe']['epochs']} here")
    if other.get("seeds") != training["seeds"]:
        raise ValueError(f"Both arms declare the same training seeds: {other.get('seeds')} in {paired}, {training['seeds']} here")
    for split in ("train", "val"):
        if _content(other_manifest, split, True) != _content(manifest, split, True):
            raise ValueError(f"Both arms use the same real {split} images")
    if other_manifest["counts"]["synthetic_train"] != manifest["counts"]["synthetic_train"]:
        raise ValueError("Both arms hold the same number of synthetic images")
    if mine.get("kappa") is not None and theirs.get("kappa") is not None and mine["kappa"] != theirs["kappa"]:
        raise ValueError(f"Both arms sample at one kappa: {theirs['kappa']} in {paired}, {mine['kappa']} here")
    renders = {source["name"]: source["render"] for source in manifest["sources"][1:]
               if source.get("render")}
    renders.update({f"{paired.name}/{source['name']}": source["render"]
                    for source in other_manifest["sources"][1:] if source.get("render")})
    check_arms_match(renders)
    if mine["update"] == 1 and (other.get("initialization") or {}).get("sha256") != training["initialization"]["sha256"]:
        raise ValueError("Both arms of update 1 start from the same f0 checkpoint")
    return {"package": str(paired), "training_sha256": sha256(paired / "training.json"),
            "manifest_sha256": sha256(paired / "manifest.json")}


def _link_or_copy(source: Path, target: Path) -> str:
    """Hard-link an image into the package, or copy it across file systems."""
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _require_provenance(manifest: dict[str, Any]) -> None:
    """Every synthetic source of E3-E5 carries both manifests of its batch.

    Without specs/manifest.json beside the dataset the generation streams cannot be
    checked for overlap, and without render_manifest.json the arms cannot be compared
    for how they were rendered; the inspection records such a source as 'unchecked'.
    """
    missing = []
    for source in manifest["sources"][1:]:
        root = Path(source["root"])
        if source.get("generation") is None:
            missing.append(f"{source['name']} ({root}): no {root.parent / 'specs' / 'manifest.json'}")
        if source.get("render") is None:
            missing.append(f"{source['name']} ({root}): no {root / RENDER_MANIFEST}")
        elif not source["render"].get("generator_sha256") or not source["render"].get("code_sha256"):
            missing.append(f"{source['name']} ({root}): {RENDER_MANIFEST} records no generator or "
                           "code digest (rendered before they were recorded, or without specs/manifest.json)")
    if missing:
        raise ValueError("synthetic sources lack the records their checks need: " + "; ".join(missing))


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
    parent_package: Path | None = None,
    arm: str | None = None,
    paired_package: Path | None = None,
    require_provenance: bool = False,
) -> dict[str, Any]:
    """Copy verified inputs to a new portable directory; never overwrite one.

    Images are hard-linked into the package (copied only across file systems), labels
    copied; the content hashes are checked either way. require_provenance (set by
    scripts/23_launch_retrain.py for E3-E5) refuses a synthetic source without its
    spec and render manifests instead of recording it as unchecked.

    E4 also needs parent_package (the package whose run produced `model`) and arm
    ('loop' or 'control'); paired_package names the other arm's package of the
    same update when it exists.
    """
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
    recipe = dict(RECIPE)
    if epochs is not None:
        recipe["epochs"] = epochs
    initializer = None
    lineage = None
    if experiment == "E4":
        if str(model) == "yolov8s-seg.pt" or epochs is None or parent_package is None or arm not in E4_ARMS:
            raise ValueError("E4 requires --model pointing to the selected parent checkpoint, --parent-package "
                             "naming the package that trained it, --arm loop or control, and explicit --epochs "
                             "for the frozen fine-tuning budget")
        initializer = Path(model).resolve()
        if not initializer.is_file():
            raise ValueError(f"Parent checkpoint does not exist: {initializer}")
        initialization = {"model": "initial.pt", "sha256": sha256(initializer), "source": str(initializer)}
        lineage = _parent_lineage(parent_package, initialization["sha256"])
        update = _e4_update(lineage, arm, seeds, recipe)
    else:
        if str(model) != "yolov8s-seg.pt":
            raise ValueError(f"{experiment} starts from the common COCO yolov8s-seg.pt initializer; custom parent checkpoints belong to E4")
        if parent_package is not None or arm is not None or paired_package is not None:
            raise ValueError("Parent, arm and paired packages belong to E4")
        initialization = {"model": "yolov8s-seg.pt"}
    training = {
        "schema_version": 1, "experiment": experiment, "version": version,
        "seeds": seeds, "ultralytics_version": ULTRALYTICS_VERSION,
        "recipe": recipe, "initialization": initialization,
        "selection": dict(SELECTION),
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
    if require_provenance:
        _require_provenance(manifest)
    manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
    if lineage is not None:
        _check_e4_data(lineage, manifest)
        kappa = _check_e4_kappa(lineage, manifest, require=require_provenance)
        training["e4"] = {"arm": arm, "update": update, "kappa": kappa, "parent": lineage["record"],
                          "paired_package": None}
        if paired_package is not None:
            training["e4"]["paired_package"] = _check_e4_pair(paired_package, training, manifest)
    if inspect_only:
        return {"inspect_only": True, "destination": str(destination), "training": training, "manifest": manifest}
    runner_source = Path(runner_source) if runner_source else Path(__file__).resolve().parents[2] / "scripts" / "train_packaged_model.py"
    if not runner_source.is_file():
        raise ValueError(f"Portable training runner is missing: {runner_source}")
    destination.mkdir(parents=True)
    placed = {"hardlink": 0, "copy": 0}
    for row in manifest["images"]:
        for kind in ("image", "label"):
            target = destination / row[kind]
            target.parent.mkdir(parents=True, exist_ok=True)
            if kind == "image":
                placed[_link_or_copy(Path(row["source_image"]), target)] += 1
            else:
                shutil.copy2(row["source_label"], target)
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
        "The source paths in manifest.json are provenance only; they are not needed on Linux. "
        "Images are hard links to the source datasets where both sit on one disk: never edit an "
        "image in place here, it would change the source too (deleting the package is safe).\n", encoding="utf-8")
    return {"inspect_only": False, "destination": str(destination), "counts": manifest["counts"],
            "images_placed": placed,
            "manifest_sha256": sha256(destination / "manifest.json"), "training": training}
