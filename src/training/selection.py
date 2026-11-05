"""Model selection from returned training runs, by the rules fixed in Methods.

Within a run the kept checkpoint is the one with the highest sum of box and
mask mAP@0.5:0.95 on real validation, the latest epoch on ties (Ultralytics
rewrites best.pt when a later epoch equals the best score). kappa is chosen by
the highest mean of that score over the declared seeds, ties to the smaller
kappa; within a configuration the deployed seed is the run with the highest
score, ties to the lowest seed. Scores are compared unrounded, and means as
exact fractions of the recorded floats, so no rounding can create or break a tie.

Each run is re-checked against its own record before it is used: complete, the
selected epoch equal to the best of its validation history, that history equal
to results.csv to the CSV's six significant digits, and the record bound to the
package's manifest.json and training.json by SHA-256. Candidates compared with
one another must share recipe, seeds, real images and synthetic image counts.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any

# results.csv stores scores with six significant digits (Ultralytics "%.6g").
CSV_TOLERANCE = 1e-5
RULES = {
    "score": "box mAP50-95 + mask mAP50-95 on real validation, unrounded",
    "checkpoint": "highest score within the run; latest epoch on ties",
    "kappa": "highest mean score over the declared seeds, exact; smaller kappa on ties",
    "deployed_seed": "highest score within the configuration; lowest seed on ties",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc


def load_run(package: Path, seed: int) -> dict[str, Any]:
    """One complete run of a package, checked against its history and results.csv."""
    run_dir = Path(package) / "runs" / f"seed{seed}"
    record_path = run_dir / "run_record.json"
    record = _read_json(record_path)
    where = f"{package} seed {seed}"
    if record.get("status") != "complete" or record.get("seed") != seed:
        raise ValueError(f"{where}: the run is not complete (status {record.get('status')!r})")
    if record.get("manifest_sha256") != sha256(Path(package) / "manifest.json") or \
            record.get("training_sha256") != sha256(Path(package) / "training.json"):
        raise ValueError(f"{where}: the run record belongs to another manifest.json or training.json")
    history = record.get("validation_history") or []
    if not history:
        raise ValueError(f"{where}: no validation history")
    best = max(history, key=lambda entry: (entry["score"], entry["epoch"]))
    selected = record.get("selected_validation") or {}
    if (selected.get("epoch"), selected.get("score")) != (best["epoch"], best["score"]):
        raise ValueError(f"{where}: the kept checkpoint (epoch {selected.get('epoch')}) is not the best "
                         f"validation epoch {best['epoch']}")
    with (run_dir / "results.csv").open(encoding="utf-8", newline="") as stream:
        rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(stream)]
    if len(rows) != len(history):
        raise ValueError(f"{where}: results.csv has {len(rows)} epochs, the record {len(history)}")
    for row, entry in zip(rows, history):
        try:
            epoch = int(float(row["epoch"]))
            score = float(row["metrics/mAP50-95(B)"]) + float(row["metrics/mAP50-95(M)"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{where}: unreadable results.csv row: {exc}") from exc
        if epoch != entry["epoch"] or abs(score - entry["score"]) > CSV_TOLERANCE:
            raise ValueError(f"{where}: results.csv epoch {epoch} disagrees with the run record")
    return {
        "package": str(Path(package).resolve()), "seed": seed,
        "score": best["score"], "epoch": best["epoch"],
        "box_map50_95": best["box_map50_95"], "mask_map50_95": best["mask_map50_95"],
        "run_record_sha256": sha256(record_path),
        "checkpoint": record.get("checkpoint"), "checkpoint_sha256": record.get("checkpoint_sha256"),
    }


def load_package(package: Path) -> dict[str, Any]:
    """A package's configuration and every declared run, all of which must be complete."""
    package = Path(package).resolve()
    training = _read_json(package / "training.json")
    manifest = _read_json(package / "manifest.json")
    return {"root": package, "training": training, "manifest": manifest,
            "training_sha256": sha256(package / "training.json"),
            "manifest_sha256": sha256(package / "manifest.json"),
            "runs": [load_run(package, seed) for seed in training["seeds"]]}


def _real(manifest: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {(row["split"], row["image_sha256"], row["label_sha256"])
            for row in manifest["images"] if row["source"] == "real"}


def _comparable(packages: list[dict[str, Any]], *, same_seeds: bool) -> None:
    """Candidates differ only in what is being compared."""
    first = packages[0]
    for other in packages[1:]:
        where = f"{first['root']} and {other['root']}"
        if other["training"]["recipe"] != first["training"]["recipe"]:
            raise ValueError(f"{where} use different training recipes")
        if _real(other["manifest"]) != _real(first["manifest"]):
            raise ValueError(f"{where} use different real images")
        if other["manifest"]["counts"]["synthetic_train"] != first["manifest"]["counts"]["synthetic_train"]:
            raise ValueError(f"{where} hold different numbers of synthetic images")
        if same_seeds and sorted(other["training"]["seeds"]) != sorted(first["training"]["seeds"]):
            raise ValueError(f"{where} declare different seeds")


def best_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Highest score; the lowest seed on ties."""
    seeds = [run["seed"] for run in runs]
    if not runs or len(set(seeds)) != len(seeds):
        raise ValueError(f"Deployment needs runs with distinct seeds, got {seeds}")
    return min(runs, key=lambda run: (-Fraction(run["score"]), run["seed"]))


def _deployed(run: dict[str, Any]) -> dict[str, Any]:
    """The chosen run, with its checkpoint file present and unchanged."""
    if not run["checkpoint"]:
        raise ValueError(f"{run['package']} seed {run['seed']}: the record names no checkpoint")
    path = Path(run["package"]) / run["checkpoint"]
    if not path.is_file() or sha256(path) != run["checkpoint_sha256"]:
        raise ValueError(f"{path}: the deployed checkpoint is missing or differs from its run record")
    return {**run, "checkpoint_path": str(path)}


def _summary(package: dict[str, Any]) -> dict[str, Any]:
    training = package["training"]
    return {"package": str(package["root"]), "experiment": training.get("experiment"),
            "e4": training.get("e4"), "training_sha256": package["training_sha256"],
            "manifest_sha256": package["manifest_sha256"], "epochs": training["recipe"]["epochs"],
            "runs": package["runs"]}


def select_kappa(candidates: dict[float, Path]) -> dict[str, Any]:
    """E3: kappa by the highest exact mean score over seeds, then its deployed seed."""
    if len(candidates) < 2:
        raise ValueError("A kappa choice needs at least two candidates")
    loaded = {float(kappa): load_package(path) for kappa, path in candidates.items()}
    for kappa, package in loaded.items():
        if package["training"].get("experiment") != "E3":
            raise ValueError(f"{package['root']}: kappa candidates are E3 packages")
        recorded = {source["generation"].get("kappa") for source in package["manifest"]["sources"][1:]
                    if source.get("generation")}
        if recorded and recorded != {kappa}:
            raise ValueError(f"{package['root']}: its scene manifests record kappa {sorted(recorded)}, not {kappa}")
    _comparable(list(loaded.values()), same_seeds=True)
    means = {kappa: sum(Fraction(run["score"]) for run in package["runs"]) / len(package["runs"])
             for kappa, package in loaded.items()}
    chosen = min(means, key=lambda kappa: (-means[kappa], kappa))
    return {
        "decision": "kappa",
        "candidates": [{"kappa": kappa, **_summary(package), "mean_score": float(means[kappa]),
                        "mean_score_exact": str(means[kappa])}
                       for kappa, package in sorted(loaded.items())],
        "selected_kappa": chosen,
        "deployed": _deployed(best_run(loaded[chosen]["runs"])),
    }


def _arm(packages: list[Path], arm: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    loaded = [load_package(path) for path in packages]
    if not loaded:
        raise ValueError("No package given")
    experiments = {package["training"].get("experiment") for package in loaded}
    if len(experiments) != 1:
        raise ValueError(f"One configuration comes from one experiment, got {sorted(experiments)}")
    if experiments == {"E4"}:
        stages = {(package["training"]["e4"]["arm"], package["training"]["e4"]["update"]) for package in loaded}
        if len(stages) != 1 or (arm is not None and next(iter(stages))[0] != arm):
            raise ValueError(f"Packages of one E4 arm and update expected{f' ({arm})' if arm else ''}, got {sorted(stages)}")
    elif arm is not None:
        raise ValueError("E4 arms come from E4 packages")
    _comparable(loaded, same_seeds=False)
    runs = [run for package in loaded for run in package["runs"]]
    return loaded, _deployed(best_run(runs))


def select_deployed(packages: list[Path], role: str) -> dict[str, Any]:
    """The deployed seed of one configuration, from one or several packages."""
    loaded, deployed = _arm(packages)
    return {"decision": "deployed_seed", "role": role,
            "packages": [_summary(package) for package in loaded], "deployed": deployed}


def select_e4_adaptation(packages: list[Path]) -> dict[str, Any]:
    """The update-1 loop model that runs the second adaptation block."""
    loaded, deployed = _arm(packages, "loop")
    if loaded[0]["training"]["e4"]["update"] != 1:
        raise ValueError("The second adaptation block runs the loop model of update 1")
    return {"decision": "e4_adaptation_model", "role": "loop update 1, runs adaptation block 2",
            "packages": [_summary(package) for package in loaded], "deployed": deployed}


def select_e4_final(loop: list[Path], control: list[Path]) -> dict[str, Any]:
    """The final checkpoint of each E4 arm; both arms must share update, schedule and data."""
    arms = {"loop": _arm(loop, "loop"), "control": _arm(control, "control")}
    first, second = arms["loop"][0][0], arms["control"][0][0]
    if first["training"]["e4"]["update"] != second["training"]["e4"]["update"]:
        raise ValueError("Both arms are evaluated after the same number of updates")
    _comparable([first, second], same_seeds=False)
    seeds = {name: sorted(run["seed"] for package in loaded for run in package["runs"])
             for name, (loaded, _) in arms.items()}
    if seeds["loop"] != seeds["control"]:
        raise ValueError(f"Both arms train the same seeds, got {seeds}")
    return {"decision": "e4_final_models", "update": first["training"]["e4"]["update"],
            "epochs": first["training"]["recipe"]["epochs"],
            "arms": {name: {"packages": [_summary(package) for package in loaded], "deployed": deployed}
                     for name, (loaded, deployed) in arms.items()}}


def write_record(selection: dict[str, Any], path: Path) -> str:
    """Write the selection record once, with its SHA-256 beside it; returns the hash."""
    path = Path(path)
    if path.exists():
        raise ValueError(f"Refusing to overwrite the selection record {path}")
    record = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "rules": RULES, **selection}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    digest = sha256(path)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest
