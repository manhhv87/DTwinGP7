"""Resolve the selected detector and describe it without loading a second model.

Metadata is written to the run sidecar, never printed: checkpoint identity may be
unblinding information. Hashing happens once when a detector is loaded, while
runtime inference settings can be refreshed after the first prediction.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from copy import deepcopy
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_path(config: dict, project_root: str | Path) -> Path:
    """Require the configured artifact; never search for an arbitrary checkpoint."""
    value = config.get("model_path", "models/yolov8s-seg_best.pt")
    if not value or not isinstance(value, (str, Path)):
        raise ValueError("model_path must name a local checkpoint file")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(project_root) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configured model_path does not exist: {path}")
    return path


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_configured_detector(config: dict, project_root: str | Path, detector_cls=None):
    """Load exactly one selected checkpoint and cache its identity on the detector."""
    path = resolve_model_path(config, project_root)
    before = path.stat()
    checkpoint_hash = file_sha256(path)
    if detector_cls is None:
        from .detector import ObjectDetector
        detector_cls = ObjectDetector
    detector = detector_cls(
        model_path=str(path), conf=config.get("conf_threshold", 0.5),
        iou=config.get("iou_threshold", 0.45),
        class_heights_mm=config.get("class_heights_mm") or {},
    )
    after = path.stat()
    # A changed file must not associate the in-memory model with another artifact.
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size,
                             stat.st_mtime_ns, stat.st_ctime_ns)
    if identity(before) != identity(after):
        raise ValueError("Configured checkpoint changed while it was being loaded")
    detector._model_artifact = {
        "kind": "checkpoint", "path": str(path),
        "sha256": checkpoint_hash, "size_bytes": after.st_size,
        "classes": list(detector.class_names),
        "framework": {"ultralytics": _version("ultralytics"), "torch": _version("torch")},
    }
    return detector


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def detector_metadata(detector) -> dict:
    """Snapshot loaded identity and observed prediction options; never guess imgsz.

    Before first inference, imgsz follows model overrides, then the installed
    framework default. Other implicit prediction options remain unknown until
    predictor.args exists: predict() may override the framework defaults. The
    stride-adjusted target size and selected device are recorded separately.
    """
    cached = getattr(detector, "_model_artifact", None)
    if cached is None:
        return {"kind": "mock" if type(detector).__name__ == "MockDetector" else "untracked",
                "detector_type": type(detector).__name__,
                "classes": list(getattr(detector, "class_names", []))}
    result = deepcopy(cached)
    model = detector.model
    predictor = getattr(model, "predictor", None)
    defaults = getattr(sys.modules.get("ultralytics.utils"), "DEFAULT_CFG_DICT", {})
    overrides = getattr(model, "overrides", {}) or {}
    runtime_args = getattr(predictor, "args", None)
    runtime = runtime_args if isinstance(runtime_args, dict) else vars(runtime_args) if runtime_args else {}
    keys = ("imgsz", "device", "half", "max_det", "agnostic_nms", "classes", "rect", "batch", "task")
    inference = {key: runtime.get(key) for key in keys}
    inference["imgsz"] = runtime.get("imgsz", overrides.get("imgsz", defaults.get("imgsz")))
    inference["task"] = runtime.get("task", overrides.get("task"))
    inference.update({"conf": detector.conf, "iou": detector.iou,
                      "retina_masks": True, "verbose": False})
    inference["imgsz_source"] = ("predictor.args" if "imgsz" in runtime else
                                 "model.overrides" if "imgsz" in overrides else
                                 "framework_default" if "imgsz" in defaults else "unknown")
    inference["predictor_initialized"] = predictor is not None
    inference["predictor_imgsz"] = getattr(predictor, "imgsz", None)
    device = getattr(predictor, "device", None)
    inference["runtime_device"] = str(device) if device is not None else None
    result["inference"] = inference
    return json.loads(json.dumps(result, default=_json_value))


def capture_config_files(paths) -> list[dict]:
    """Capture input-file identity once when the configuration is assembled.

    Later sidecar updates refresh detector runtime settings, not these inputs.
    The resolved config snapshot remains the authority for actual run settings.
    """
    records = []
    for value in paths:
        path = Path(value).resolve()
        record = {"path": str(path), "exists": path.is_file()}
        if record["exists"]:
            record.update(sha256=file_sha256(path), size_bytes=path.stat().st_size)
        records.append(record)
    return records


def run_metadata(config: dict, detector, *, entrypoint: str, runtime: dict | None = None,
                 config_files=()) -> dict:
    """Keep the supplied resolved config and inputs with the selected model."""
    snapshot = json.loads(json.dumps(config, default=_json_value))
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return {
        "entrypoint": entrypoint,
        "config": snapshot,
        "config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "config_files": deepcopy(list(config_files)),
        "model": detector_metadata(detector),
        "runtime": runtime or {},
    }
