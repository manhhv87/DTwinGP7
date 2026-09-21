"""Input/provenance and portability checks; no model or GPU is used."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from src.training.package import RECIPE, build_package, inspect_sources, sha256
from src.utils.dataset_check import EXPECTED_NAMES


def make_source(root: Path, *, synthetic: bool = False) -> Path:
    root.mkdir()
    (root / "dataset.yaml").write_text(yaml.safe_dump({"names": EXPECTED_NAMES, "nc": 5}), encoding="utf-8")
    for split in (["train"] if synthetic else ["train", "val"]):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        # Unique contents are sufficient: these tests validate packaging, not decoding.
        (root / "images" / split / "sample.png").write_bytes(f"{root.name}/{split}".encode())
        (root / "labels" / split / "sample.txt").write_text("4 0.1 0.1 0.7 0.1 0.4 0.8\n", encoding="utf-8")
    return root


@pytest.fixture
def real(tmp_path: Path) -> Path:
    return make_source(tmp_path / "real")


@pytest.fixture
def runner(tmp_path: Path) -> Path:
    path = tmp_path / "runner.py"
    path.write_text("# standalone training runner fixture\n", encoding="utf-8")
    return path


def test_portable_package_retains_five_classes_real_validation_and_namespaces(real: Path, runner: Path, tmp_path: Path) -> None:
    synth = make_source(tmp_path / "synthetic", synthetic=True)
    # Synthetic validation is deliberately absent and must not be required.
    destination = tmp_path / "dsv1"
    result = build_package(real, [synth], destination, experiment="E3", version=1, runner_source=runner)
    assert result["counts"] == {"real_train": 1, "real_val": 1, "synthetic_train": 1, "train": 2, "val": 1}
    manifest = json.loads((destination / "manifest.json").read_text())
    assert {row["source"] for row in manifest["images"] if row["split"] == "val"} == {"real"}
    assert all(row["class_ids"] == [4] for row in manifest["images"])
    assert (destination / "images/train/real/sample.png").exists()
    assert (destination / "images/train/synth01/sample.png").exists()
    data = yaml.safe_load((destination / "dataset.yaml").read_text())
    assert data["names"] == EXPECTED_NAMES
    assert data["nc"] == 5
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert (destination / "train.py").read_bytes() == runner.read_bytes()
    moved = tmp_path / "transferred"
    destination.rename(moved)
    # A transfer does not require the original Windows source paths.
    shutil.rmtree(real)
    shutil.rmtree(synth)
    for row in manifest["images"]:
        assert sha256(moved / row["image"]) == row["image_sha256"]
        assert sha256(moved / row["label"]) == row["label_sha256"]


@pytest.mark.parametrize("names", [EXPECTED_NAMES[:4], sorted(EXPECTED_NAMES), {1: name for name in EXPECTED_NAMES}, {str(i + 1): name for i, name in enumerate(EXPECTED_NAMES)}])
def test_bad_class_mapping_is_refused(real: Path, names: object) -> None:
    (real / "dataset.yaml").write_text(yaml.safe_dump({"names": names}), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical order|class IDs"):
        inspect_sources(real, [])


def test_dict_classes_are_supported_and_conflicting_second_yaml_refused(real: Path) -> None:
    (real / "dataset.yaml").write_text(yaml.safe_dump({"names": dict(enumerate(EXPECTED_NAMES))}), encoding="utf-8")
    assert inspect_sources(real, [])["classes"] == EXPECTED_NAMES
    (real / "data.yaml").write_text(yaml.safe_dump({"names": EXPECTED_NAMES[:4]}), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical order"):
        inspect_sources(real, [])


@pytest.mark.parametrize("label", [
    "4 nan 0.1 0.7 0.1 0.4 0.8", "4 inf 0.1 0.7 0.1 0.4 0.8",
    "4 -0.1 0.1 0.7 0.1 0.4 0.8", "4 1.1 0.1 0.7 0.1 0.4 0.8",
    "5 0.1 0.1 0.7 0.1 0.4 0.8", "1.0 0.1 0.1 0.7 0.1 0.4 0.8",
    "4 0.1 0.1 0.7 0.1", "4 0.1 0.1 0.7 0.1 0.4 0.8 0.5",
])
def test_malformed_polygon_is_refused(real: Path, label: str) -> None:
    (real / "labels/train/sample.txt").write_text(label)
    with pytest.raises(ValueError):
        inspect_sources(real, [])


def test_empty_negative_is_valid_missing_label_is_not(real: Path) -> None:
    label = real / "labels/train/sample.txt"
    label.write_text("")
    inspected = inspect_sources(real, [])
    assert inspected["images"][0]["class_ids"] == []
    label.unlink()
    with pytest.raises(ValueError, match="Missing label"):
        inspect_sources(real, [])


def test_train_validation_duplicate_content_is_refused(real: Path) -> None:
    (real / "images/val/sample.png").write_bytes((real / "images/train/sample.png").read_bytes())
    with pytest.raises(ValueError, match="Duplicate image content"):
        inspect_sources(real, [])


def test_repeated_synthetic_images_cannot_inflate_budget(real: Path, tmp_path: Path) -> None:
    synth = make_source(tmp_path / "synth", synthetic=True)
    (synth / "images/train/sample.png").write_bytes((real / "images/train/sample.png").read_bytes())
    with pytest.raises(ValueError, match="Duplicate image content"):
        inspect_sources(real, [synth])


def test_image_extensions_cannot_share_a_label(real: Path) -> None:
    (real / "images/train/sample.jpg").write_bytes(b"other image")
    with pytest.raises(ValueError, match="filename collision"):
        inspect_sources(real, [])


def test_allowlist_is_explicit_and_accepts_raw_capture_paths(real: Path, tmp_path: Path) -> None:
    (real / "images/train/sample.png").rename(real / "images/train/std__carton__sample.png")
    (real / "labels/train/sample.txt").rename(real / "labels/train/std__carton__sample.txt")
    (real / "images/train/negative.png").write_bytes(b"intentional negative")
    (real / "labels/train/negative.txt").write_text("")
    allowlist = tmp_path / "train.txt"
    allowlist.write_text("std/carton/sample.png\n")
    assert inspect_sources(real, [])["counts"]["real_train"] == 2
    selected = inspect_sources(real, [], real_train_list=allowlist, expected_real_train=1)
    assert selected["counts"]["real_train"] == 1
    assert selected["sources"][0]["counts"]["train"] == {"available": 2, "selected": 1}
    assert selected["real_train_allowlist"]["sha256"] == sha256(allowlist)
    with pytest.raises(ValueError, match="Expected 1 real_train images, found 2"):
        inspect_sources(real, [], expected_real_train=1)
    allowlist.write_text("std/carton/sample.png\nstd/carton/sample.png\n")
    with pytest.raises(ValueError, match="duplicate allowlist"):
        inspect_sources(real, [], real_train_list=allowlist)
    allowlist.write_text("not_present.png\n")
    with pytest.raises(ValueError, match="missing or ambiguous"):
        inspect_sources(real, [], real_train_list=allowlist)


def test_expected_synthetic_count_is_total_over_sources(real: Path, tmp_path: Path) -> None:
    synthetic = [make_source(tmp_path / name, synthetic=True) for name in ("s1", "s2")]
    assert inspect_sources(real, synthetic, expected_synth_train=2)["counts"]["synthetic_train"] == 2
    with pytest.raises(ValueError, match="Expected 3 synthetic_train"):
        inspect_sources(real, synthetic, expected_synth_train=3)


def test_dry_inspection_never_creates_destination(real: Path, tmp_path: Path) -> None:
    destination = tmp_path / "do_not_create"
    result = build_package(real, [], destination, experiment="E2", version=0, inspect_only=True)
    assert not destination.exists()
    assert result["training"]["seeds"] == list(range(5))
    assert result["training"]["recipe"] == RECIPE
    assert result["training"]["ultralytics_version"] == "8.4.66"


def test_package_cannot_overwrite_or_pollute_sources(real: Path, runner: Path, tmp_path: Path) -> None:
    destination = tmp_path / "already_there"
    destination.mkdir()
    with pytest.raises(ValueError, match="overwrite"):
        build_package(real, [], destination, experiment="E2", version=0, runner_source=runner)
    with pytest.raises(ValueError, match="inside a source"):
        build_package(real, [], real / "nested", experiment="E2", version=0, runner_source=runner)
    with pytest.raises(ValueError, match="distinct directory"):
        inspect_sources(real, [real])


def test_e4_parent_is_copied_with_hash_and_requires_explicit_budget(real: Path, runner: Path, tmp_path: Path) -> None:
    synth = make_source(tmp_path / "synth", synthetic=True)
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"fake checkpoint for copying test")
    with pytest.raises(ValueError, match="E4 requires"):
        build_package(real, [synth], tmp_path / "bad1", experiment="E4", version=1, model=parent, runner_source=runner)
    with pytest.raises(ValueError, match="E4 requires"):
        build_package(real, [synth], tmp_path / "bad2", experiment="E4", version=1, epochs=30, runner_source=runner)
    destination = tmp_path / "e4"
    result = build_package(real, [synth], destination, experiment="E4", version=1, model=parent, epochs=30, runner_source=runner)
    assert result["training"]["initialization"]["sha256"] == sha256(destination / "initial.pt") == sha256(parent)
    assert result["training"]["initialization"]["model"] == "initial.pt"
    assert result["training"]["recipe"]["epochs"] == 30
    assert result["training"]["seeds"] == [0, 1, 2]


@pytest.mark.parametrize("experiment", ["E3", "E4", "E5"])
def test_synthetic_experiments_require_synthetic_data(real: Path, tmp_path: Path, experiment: str) -> None:
    with pytest.raises(ValueError, match="requires synthetic"):
        build_package(real, [], tmp_path / "bad", experiment=experiment, version=0, inspect_only=True)


def test_e2_cannot_include_synthetic_and_e5_cannot_change_parent(real: Path, tmp_path: Path) -> None:
    synth = make_source(tmp_path / "synth", synthetic=True)
    with pytest.raises(ValueError, match="E2 is real-only"):
        build_package(real, [synth], tmp_path / "bad", experiment="E2", version=0, inspect_only=True)
    with pytest.raises(ValueError, match="common COCO"):
        build_package(real, [synth], tmp_path / "bad", experiment="E5", version=0, model="custom.pt", inspect_only=True)
    result = build_package(real, [synth], tmp_path / "e5", experiment="E5", version=0, inspect_only=True)
    assert result["training"]["seeds"] == [0, 1, 2]


def test_cli_inspect_only_accepts_explicit_count_guards(real: Path, tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/23_launch_retrain.py"
    completed = subprocess.run([
        sys.executable, str(script), "--real", str(real), "--experiment", "E2", "--version", "0",
        "--out", str(tmp_path / "packages"), "--expected-real-train", "1", "--inspect-only",
    ], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["manifest"]["counts"]["real_train"] == 1
    assert not (tmp_path / "packages").exists()


def load_portable_runner():
    path = Path(__file__).resolve().parents[1] / "scripts/train_packaged_model.py"
    specification = importlib.util.spec_from_file_location("portable_training_runner", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_real_runner_dry_run_works_after_transfer_and_detects_tampering(real: Path, tmp_path: Path) -> None:
    package = tmp_path / "original"
    build_package(real, [], package, experiment="E2", version=0)
    moved = tmp_path / "transferred"
    package.rename(moved)
    shutil.rmtree(real)
    completed = subprocess.run([sys.executable, str(moved / "train.py"), "--dry-run"], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "real_train" in completed.stdout
    (moved / "labels/train/real/sample.txt").write_text("")
    completed = subprocess.run([sys.executable, str(moved / "train.py"), "--dry-run"], capture_output=True, text=True)
    assert completed.returncode != 0
    assert "Content changed since packaging" in completed.stderr


def test_generated_ultralytics_cache_is_allowed_but_unmanifested_labels_are_not(real: Path, tmp_path: Path) -> None:
    package = tmp_path / "package"
    build_package(real, [], package, experiment="E2", version=0)
    module = load_portable_runner()
    for split in ("train", "val"):
        (package / "labels" / split / "real.cache").write_bytes(b"generated Ultralytics cache")
    config, manifest = module.verify_package(package)
    assert manifest["counts"]["real_train"] == 1
    assert config["selection"]["seed_tie"] == "first_in_declared_seed_order"
    completed = subprocess.run([sys.executable, str(package / "train.py"), "--dry-run"], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    (package / "labels/train/bogus.txt").write_text("")
    with pytest.raises(ValueError, match="Unmanifested dataset file"):
        module.verify_package(package)
