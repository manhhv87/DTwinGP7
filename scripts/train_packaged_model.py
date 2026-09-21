#!/usr/bin/env python
"""Standalone runner copied into a portable dataset package; dry-run needs only Python."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CLASSES = ["carton", "plastic_box", "wood_box", "metal_box", "inox_box"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Package path escapes its root: {relative}")
    return path


def label_cache_paths(root: Path, manifest: dict) -> set[Path]:
    # Ultralytics places the cache beside the parent of the first label file.
    return {checked_path(root, record["label"]).parent.with_suffix(".cache")
            for record in manifest["images"]}


def verify_package(root: Path) -> tuple[dict, dict]:
    config = json.loads((root / "training.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("classes") != CLASSES:
        raise ValueError("Package class order does not match the five-class protocol")
    counts = {"real_train": 0, "real_val": 0, "synthetic_train": 0, "train": 0, "val": 0}
    known = set()
    images_seen = set()
    for record in manifest["images"]:
        split, source = record["split"], record["source"]
        if split not in {"train", "val"} or (split == "val" and source != "real"):
            raise ValueError("Validation must contain only real images")
        counts[split] += 1
        counts[("real_" + split) if source == "real" else "synthetic_train"] += 1
        for kind in ("image", "label"):
            relative = record[kind]
            expected_prefix = ("images" if kind == "image" else "labels") + "/" + split + "/"
            if not relative.startswith(expected_prefix) or relative.casefold() in known:
                raise ValueError(f"Invalid or duplicate package path: {relative}")
            known.add(relative.casefold())
            if sha256(checked_path(root, relative)) != record[kind + "_sha256"]:
                raise ValueError(f"Content changed since packaging: {relative}")
        if record["image_sha256"] in images_seen:
            raise ValueError("Duplicate image content in package")
        images_seen.add(record["image_sha256"])
    if counts != manifest["counts"] or not counts["real_train"] or not counts["real_val"]:
        raise ValueError("Manifest counts do not match its image records")
    # The runtime loader scans directories, so reject extra files as well as missing ones.
    caches = label_cache_paths(root, manifest)
    for directory in ("images", "labels"):
        for path in (root / directory).rglob("*"):
            if path.is_file() and path.relative_to(root).as_posix().casefold() not in known:
                if path in caches and path.resolve().is_relative_to(root.resolve()):
                    continue
                raise ValueError(f"Unmanifested dataset file: {path.relative_to(root)}")
    initial = config["initialization"]
    if initial.get("sha256"):
        if sha256(checked_path(root, initial["model"])) != initial["sha256"]:
            raise ValueError("Initial checkpoint changed since packaging")
    elif initial["model"] != "yolov8s-seg.pt":
        raise ValueError("An explicit initial checkpoint requires a recorded SHA-256")
    return config, manifest


def validation_record(trainer) -> dict:
    """Capture unrounded scores at checkpoint-save time, before final re-validation."""
    box = float(trainer.metrics["metrics/mAP50-95(B)"])
    mask = float(trainer.metrics["metrics/mAP50-95(M)"])
    fitness = float(trainer.fitness)
    if not all(math.isfinite(x) for x in (box, mask, fitness)):
        raise ValueError("Non-finite validation score")
    if not (0 <= box <= 1 and 0 <= mask <= 1) or not math.isclose(fitness, box + mask, abs_tol=1e-12):
        raise ValueError("Trainer fitness differs from the declared box-plus-mask criterion")
    return {"epoch": int(trainer.epoch) + 1, "box_map50_95": box,
            "mask_map50_95": mask, "score": fitness}


def assert_recipe(trainer, recipe: dict) -> None:
    for key, expected in recipe.items():
        actual = getattr(trainer.args, key)
        if actual != expected:
            raise ValueError(f"Effective {key}={actual!r}, requested {expected!r}; run is not comparable")
    if trainer.batch_size != recipe["batch"]:
        raise ValueError("The trainer changed batch size")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n", encoding="utf-8")


def run_training(root: Path, config: dict, manifest: dict, seeds: list[int], device: str) -> None:
    from importlib.metadata import version
    installed = version("ultralytics")
    if installed != config["ultralytics_version"]:
        raise ValueError(f"Requires ultralytics=={config['ultralytics_version']}; installed {installed}")
    import torch
    import yaml
    from ultralytics import YOLO

    if device not in {"cpu", "mps"} and not device.isdecimal():
        raise ValueError("Use one explicit device (e.g. 0 or cpu); multi-GPU requires a separate validated protocol")
    for seed in seeds:
        if (root / "runs" / f"seed{seed}").exists():
            raise ValueError(f"Run seed{seed} already exists; use --seeds for the remaining declared seeds")
    # Cache validation uses file sizes rather than our content hashes. Rebuild the
    # exact generated cache files after verification to avoid stale parsed labels.
    for cache in label_cache_paths(root, manifest):
        if cache.exists():
            checked_path(root, cache.relative_to(root).as_posix()).unlink()
    runtime_yaml = root / "dataset.runtime.yaml"
    runtime_yaml.write_text(yaml.safe_dump({"path": root.as_posix(), "train": "images/train",
                                           "val": "images/val", "names": CLASSES, "nc": len(CLASSES)},
                                          sort_keys=False), encoding="utf-8")
    # Keep any automatic COCO download in the package, then record its actual bytes.
    initial = config["initialization"]
    initial_path = checked_path(root, initial["model"])
    for seed in seeds:
        output = root / "runs" / f"seed{seed}"
        output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        record = {"status": "running", "experiment": config["experiment"], "version": config["version"],
                  "seed": seed, "started_at": datetime.now(timezone.utc).isoformat(),
                  "manifest_sha256": sha256(root / "manifest.json"),
                  "training_sha256": sha256(root / "training.json"),
                  "runner_sha256": sha256(Path(__file__).resolve()),
                  "requested_recipe": config["recipe"], "data_counts": manifest["counts"],
                  "runtime": {"python": sys.version, "platform": platform.platform(),
                              "torch": torch.__version__, "ultralytics": installed, "device": device,
                              "cuda": torch.version.cuda,
                              "available_gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
                  "optimizer_updates": 0, "validation_history": []}
        write_json(output / "run_record.json", record)
        hook = None
        try:
            model = YOLO(str(initial_path))
            if model.task != "segment":
                raise ValueError("Initial checkpoint is not an instance-segmentation model")
            if initial.get("sha256"):
                names = model.names
                ordered = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
                if ordered != CLASSES:
                    raise ValueError("Fine-tuning parent must use the canonical five-class order")
            record["initial_checkpoint_sha256"] = sha256(initial_path)
            if initial.get("sha256") and record["initial_checkpoint_sha256"] != initial["sha256"]:
                raise ValueError("Initial checkpoint hash changed before training")

            def count_update(*_args):
                record["optimizer_updates"] += 1

            def on_start(trainer):
                nonlocal hook
                assert_recipe(trainer, config["recipe"])
                if Path(trainer.save_dir).resolve() != output:
                    raise ValueError("Unexpected trainer output directory")
                if (len(trainer.train_loader.dataset) != manifest["counts"]["train"] or
                        len(trainer.test_loader.dataset) != manifest["counts"]["val"]):
                    raise ValueError("The loader omitted images; the declared image budget was not loaded")
                # Reject silent image repair or any other loader-time content change.
                verify_package(root)
                # Post-hook counts executed optimizer steps; AMP-skipped steps are excluded.
                hook = trainer.optimizer.register_step_post_hook(count_update)

            def on_batch_start(trainer):
                # Also detects automatic OOM batch reduction before training can continue.
                if trainer.batch_size != config["recipe"]["batch"]:
                    raise ValueError("Automatic batch reduction detected; revise the whole comparison explicitly")

            def on_save(trainer):
                entry = validation_record(trainer)
                record["validation_history"].append(entry)
                if trainer.fitness == trainer.best_fitness:
                    record["selected_validation"] = entry
                write_json(output / "run_record.json", record)

            model.add_callback("on_train_start", on_start)
            model.add_callback("on_train_batch_start", on_batch_start)
            model.add_callback("on_model_save", on_save)
            model.train(**config["recipe"], data=str(runtime_yaml), seed=seed, device=device,
                        project=str(root / "runs"), name=f"seed{seed}", exist_ok=True)
            trainer = model.trainer
            assert_recipe(trainer, config["recipe"])
            record["effective_args"] = vars(trainer.args)
            with (output / "results.csv").open(encoding="utf-8", newline="") as stream:
                epochs = len(list(csv.DictReader(stream)))
            if epochs != config["recipe"]["epochs"] or len(record["validation_history"]) != epochs:
                raise ValueError(f"Incomplete or repeated epochs: CSV={epochs}, saved={len(record['validation_history'])}")
            if "selected_validation" not in record or record["optimizer_updates"] < 1:
                raise ValueError("Training lacks checkpoint-selection or optimizer-step evidence")
            best = output / "weights" / "best.pt"
            record["checkpoint"] = best.relative_to(root).as_posix()
            record["checkpoint_sha256"] = sha256(best)
            # Model.train reloads best.pt; compare its unrounded saved fitness to the callback.
            saved_fitness = float(model.ckpt["train_metrics"]["fitness"])
            if not math.isclose(saved_fitness, record["selected_validation"]["score"], abs_tol=1e-12):
                raise ValueError("Saved best checkpoint does not match the recorded selection score")
            record["status"] = "complete"
        except BaseException as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if hook is not None:
                hook.remove()
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            record["wall_seconds"] = time.monotonic() - started
            write_json(output / "run_record.json", record)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dry-run", action="store_true", help="Verify every file without importing torch or training")
    parser.add_argument("--device", default="0")
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args(argv)
    root = args.package.resolve()
    try:
        config, manifest = verify_package(root)
        seeds = args.seeds if args.seeds is not None else config["seeds"]
        if not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(config["seeds"]):
            raise ValueError("Seeds must be a non-empty, unique subset of the packaged seed list")
        if args.dry_run:
            print(json.dumps({"verified": True, "counts": manifest["counts"], "training": config,
                              "requested_seeds": seeds}, indent=2))
        else:
            run_training(root, config, manifest, seeds, args.device)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Training refused: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
