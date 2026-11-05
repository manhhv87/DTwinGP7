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

from src.synthgen.seed_ranges import STREAM_RULE
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


def finish_runs(package: Path, tmp_path: Path, seeds: list[int]) -> dict[int, Path]:
    """Run records as train.py leaves them, each with its own fake best checkpoint."""
    checkpoints = {}
    for seed in seeds:
        checkpoint = tmp_path / f"{package.name}_seed{seed}.pt"
        checkpoint.write_bytes(f"fake checkpoint {package.name} {seed}".encode())
        run = package / "runs" / f"seed{seed}"
        run.mkdir(parents=True)
        (run / "run_record.json").write_text(json.dumps({
            "status": "complete", "seed": seed, "checkpoint_sha256": sha256(checkpoint),
            "manifest_sha256": sha256(package / "manifest.json")}), encoding="utf-8")
        checkpoints[seed] = checkpoint
    return checkpoints


@pytest.fixture
def f0(real: Path, runner: Path, tmp_path: Path) -> tuple[Path, Path, Path]:
    """The selected E3 package (real + its synthetic batch) and its seed-0 checkpoint."""
    initial = make_source(tmp_path / "anchored", synthetic=True)
    package = tmp_path / "e3"
    build_package(real, [initial], package, experiment="E3", version=1, runner_source=runner)
    return package, initial, finish_runs(package, tmp_path, [0])[0]


def test_e4_parent_is_copied_with_hash_and_lineage(real: Path, runner: Path, tmp_path: Path, f0) -> None:
    parent_package, initial, parent = f0
    loop1 = make_source(tmp_path / "loop1", synthetic=True)
    sources = [initial, loop1]
    complete = dict(model=parent, epochs=30, parent_package=parent_package, arm="loop", runner_source=runner)
    for missing in ("model", "epochs", "parent_package", "arm"):
        partial = {key: value for key, value in complete.items() if key != missing}
        with pytest.raises(ValueError, match="E4 requires"):
            build_package(real, sources, tmp_path / f"bad_{missing}", experiment="E4", version=1, **partial)
    destination = tmp_path / "e4"
    result = build_package(real, sources, destination, experiment="E4", version=1, **complete)
    assert result["training"]["initialization"]["sha256"] == sha256(destination / "initial.pt") == sha256(parent)
    assert result["training"]["initialization"]["model"] == "initial.pt"
    assert result["training"]["recipe"]["epochs"] == 30
    assert result["training"]["seeds"] == [0, 1, 2]
    lineage = result["training"]["e4"]
    assert (lineage["arm"], lineage["update"], lineage["parent"]["run_seed"]) == ("loop", 1, 0)
    assert lineage["parent"]["manifest_sha256"] == sha256(parent_package / "manifest.json")


def test_e4_keeps_the_parent_real_and_synthetic_images(real: Path, runner: Path, tmp_path: Path, f0) -> None:
    parent_package, initial, parent = f0
    loop1 = make_source(tmp_path / "loop1", synthetic=True)
    options = dict(experiment="E4", version=1, model=parent, epochs=30, parent_package=parent_package,
                   arm="loop", inspect_only=True)
    with pytest.raises(ValueError, match="1 of its 1 are missing"):
        build_package(real, [loop1], tmp_path / "x", **options)
    with pytest.raises(ValueError, match="adds no synthetic images"):
        build_package(real, [initial], tmp_path / "x", **options)
    (real / "images/train/extra.png").write_bytes(b"one more real image")
    (real / "labels/train/extra.txt").write_text("")
    with pytest.raises(ValueError, match="real train set differs"):
        build_package(real, [initial, loop1], tmp_path / "x", **options)
    stranger = tmp_path / "other.pt"
    stranger.write_bytes(b"not trained by the parent package")
    with pytest.raises(ValueError, match="0 complete run records"):
        build_package(real, [initial, loop1], tmp_path / "x", **{**options, "model": stranger})


def test_e4_update_two_continues_its_own_arm_seed_and_schedule(real: Path, runner: Path, tmp_path: Path, f0) -> None:
    parent_package, initial, parent = f0
    loop1, loop2, loop3 = (make_source(tmp_path / name, synthetic=True) for name in ("loop1", "loop2", "loop3"))
    update1 = tmp_path / "loop_u1"
    build_package(real, [initial, loop1], update1, experiment="E4", version=1, model=parent, epochs=30,
                  parent_package=parent_package, arm="loop", runner_source=runner)
    checkpoints = finish_runs(update1, tmp_path, [0, 1, 2])
    sources = [initial, loop1, loop2]
    options = dict(experiment="E4", version=2, model=checkpoints[1], epochs=30, parent_package=update1,
                   arm="loop", seeds=[1], inspect_only=True)
    with pytest.raises(ValueError, match="--seeds 1"):
        build_package(real, sources, tmp_path / "x", **{**options, "seeds": None})
    with pytest.raises(ValueError, match="own arm"):
        build_package(real, sources, tmp_path / "x", **{**options, "arm": "control"})
    with pytest.raises(ValueError, match="same fine-tuning schedule"):
        build_package(real, sources, tmp_path / "x", **{**options, "epochs": 40})
    update2 = tmp_path / "loop_u2_seed1"
    result = build_package(real, sources, update2, **{**options, "inspect_only": False, "runner_source": runner})
    assert result["training"]["e4"]["update"] == 2 and result["training"]["e4"]["parent"]["run_seed"] == 1
    third = finish_runs(update2, tmp_path, [1])[1]
    with pytest.raises(ValueError, match="at most 2"):
        build_package(real, [*sources, loop3], tmp_path / "x", **{**options, "model": third, "parent_package": update2})


def test_e4_arms_are_checked_against_each_other(real: Path, runner: Path, tmp_path: Path, f0) -> None:
    parent_package, initial, parent = f0
    loop1, control1 = (make_source(tmp_path / name, synthetic=True) for name in ("loop1", "control1"))
    loop_package = tmp_path / "loop_u1"
    build_package(real, [initial, loop1], loop_package, experiment="E4", version=1, model=parent, epochs=30,
                  parent_package=parent_package, arm="loop", runner_source=runner)
    options = dict(experiment="E4", version=1, model=parent, parent_package=parent_package, arm="control",
                   paired_package=loop_package, inspect_only=True)
    with pytest.raises(ValueError, match="same fine-tuning schedule: epochs 30"):
        build_package(real, [initial, control1], tmp_path / "x", epochs=20, **options)
    with pytest.raises(ValueError, match="same training seeds"):
        build_package(real, [initial, control1], tmp_path / "x", epochs=30, seeds=[0, 1], **options)
    with pytest.raises(ValueError, match="not the other arm"):
        build_package(real, [initial, control1], tmp_path / "x", epochs=30, **{**options, "arm": "loop"})
    result = build_package(real, [initial, control1], tmp_path / "x", epochs=30, **options)
    assert result["training"]["e4"]["paired_package"]["training_sha256"] == sha256(loop_package / "training.json")
    with pytest.raises(ValueError, match="belong to E4"):
        build_package(real, [initial], tmp_path / "x", experiment="E3", version=1, arm="loop", inspect_only=True)


RENDERED = {"manifest_version": 1, "blender_version": "4.2.1 LTS", "blenderproc_version": "2.8.0",
            "device": "GPU:OPTIX", "render": {"samples": 24, "reflectance_scale": 0.46},
            "profile_sha256": "a" * 64, "generator_sha256": "c" * 64, "code_sha256": "d" * 64,
            "frames": ["frame_000000"]}


def rendered(dataset: Path, **change: object) -> Path:
    """The render manifest auto_label copies into a dataset, optionally changed."""
    (dataset / "render_manifest.json").write_text(json.dumps({**RENDERED, **change}), encoding="utf-8")
    return dataset


def test_synthetic_sources_must_be_rendered_alike(real: Path, tmp_path: Path) -> None:
    a = rendered(make_source(tmp_path / "anchored_k2", synthetic=True))
    b = rendered(make_source(tmp_path / "blind", synthetic=True))
    loose = make_source(tmp_path / "loose", synthetic=True)
    other_samples = rendered(make_source(tmp_path / "samples64", synthetic=True),
                             render={"samples": 64, "reflectance_scale": 0.46})
    other_profile = rendered(make_source(tmp_path / "old_profile", synthetic=True),
                             profile_sha256="b" * 64)
    with pytest.raises(ValueError, match="synth02.*rendered differently.*render"):
        inspect_sources(real, [a, other_samples])
    with pytest.raises(ValueError, match="profile_sha256"):
        inspect_sources(real, [a, other_profile])
    # Another sampler config (materials, light, exposure) or other generator code: refused.
    other_generator = rendered(make_source(tmp_path / "refit", synthetic=True), generator_sha256="e" * 64)
    with pytest.raises(ValueError, match="generator_sha256"):
        inspect_sources(real, [a, other_generator])
    other_code = rendered(make_source(tmp_path / "newcode", synthetic=True), code_sha256="f" * 64)
    with pytest.raises(ValueError, match="code_sha256"):
        inspect_sources(real, [a, other_code])
    inspected = inspect_sources(real, [a, b, loose])
    check = inspected["render_check"]
    assert check["checked"] == ["synth01", "synth02"] and check["unchecked"] == ["synth03"]
    assert inspected["sources"][1]["render"]["device"] == "GPU:OPTIX"
    assert inspected["sources"][3]["render"] is None


def test_e4_arms_must_be_rendered_alike(real: Path, runner: Path, tmp_path: Path, f0) -> None:
    parent_package, initial, parent = f0
    loop1 = rendered(make_source(tmp_path / "loop1", synthetic=True))
    control1 = rendered(make_source(tmp_path / "control1", synthetic=True), device="CPU")
    loop_package = tmp_path / "loop_u1"
    build_package(real, [initial, loop1], loop_package, experiment="E4", version=1, model=parent, epochs=30,
                  parent_package=parent_package, arm="loop", runner_source=runner)
    with pytest.raises(ValueError, match="rendered differently.*device"):
        build_package(real, [initial, control1], tmp_path / "x", experiment="E4", version=1, model=parent,
                      epochs=30, parent_package=parent_package, arm="control", paired_package=loop_package,
                      inspect_only=True)


def make_batch(tmp_path: Path, name: str, seed: int, indices: range, spec_version: int = 2,
               kappa: float | None = 2.0) -> Path:
    """A generated batch as 20_generate_synth.py lays it out: specs/ beside dataset/."""
    dataset = tmp_path / name / "dataset"
    dataset.parent.mkdir()
    make_source(dataset, synthetic=True)
    (dataset / "images/train/sample.png").write_bytes(name.encode())
    specs = tmp_path / name / "specs"
    specs.mkdir()
    (specs / "manifest.json").write_text(json.dumps({
        "spec_version": spec_version, "seed": seed, "mode": "anchored", "kappa": kappa,
        "files": [f"scene_{i:06d}.json" for i in indices]}), encoding="utf-8")
    return dataset


def test_synthetic_batches_must_not_share_generation_streams(real: Path, tmp_path: Path) -> None:
    e3 = make_batch(tmp_path, "anchored_k2", 0, range(3000))
    reused = make_batch(tmp_path, "control_bad", 0, range(2000, 3000))
    other_seed = make_batch(tmp_path, "control_k1", 2000, range(1000))
    unrecorded = make_source(tmp_path / "loose", synthetic=True)
    with pytest.raises(ValueError, match="synth01 and synth02, 1000 scene"):
        inspect_sources(real, [e3, reused])
    # Version 1 drew from seed + index, so seed 2000 then reused E3 scenes 2000..2999.
    old_e3 = make_batch(tmp_path, "old_k2", 0, range(3000), spec_version=1)
    old_control = make_batch(tmp_path, "old_control", 2000, range(1000), spec_version=1)
    with pytest.raises(ValueError, match="synth01 and synth02, 1000 scene"):
        inspect_sources(real, [old_e3, old_control])
    inspected = inspect_sources(real, [e3, other_seed, unrecorded])
    check = inspected["generation_overlap_check"]
    assert check["checked"] == ["synth01", "synth02"] and check["unchecked"] == ["synth03"]
    assert check["rules"] == {"synth01": STREAM_RULE[2], "synth02": STREAM_RULE[2]}
    assert inspected["sources"][1]["generation"]["kappa"] == 2.0
    assert inspected["sources"][2]["generation"]["streams"] == [[[2000, 2001], [0, 1000]]]


def test_images_are_hard_links_and_labels_copies(real: Path, runner: Path, tmp_path: Path) -> None:
    import os
    synth = make_source(tmp_path / "synthetic", synthetic=True)
    result = build_package(real, [synth], tmp_path / "dsv1", experiment="E3", version=1, runner_source=runner)
    assert result["images_placed"] == {"hardlink": 3, "copy": 0}
    image = tmp_path / "dsv1/images/train/synth01/sample.png"
    assert os.stat(image).st_ino == os.stat(synth / "images/train/sample.png").st_ino
    assert os.stat(tmp_path / "dsv1/images/val/real/sample.png").st_ino == \
        os.stat(real / "images/val/sample.png").st_ino
    label = tmp_path / "dsv1/labels/train/synth01/sample.txt"
    assert os.stat(label).st_ino != os.stat(synth / "labels/train/sample.txt").st_ino


def test_images_are_copied_across_file_systems(real: Path, runner: Path, tmp_path: Path, monkeypatch) -> None:
    import os
    synth = make_source(tmp_path / "synthetic", synthetic=True)

    def no_link(*_args):
        raise OSError(18, "Invalid cross-device link")
    monkeypatch.setattr(os, "link", no_link)
    result = build_package(real, [synth], tmp_path / "dsv1", experiment="E3", version=1, runner_source=runner)
    assert result["images_placed"] == {"hardlink": 0, "copy": 3}
    assert (tmp_path / "dsv1/images/train/synth01/sample.png").read_bytes() == \
        (synth / "images/train/sample.png").read_bytes()


def test_synthetic_sources_without_their_manifests_are_refused(real: Path, runner: Path, tmp_path: Path) -> None:
    """For E3-E5 the CLI refuses what the inspection would only record as unchecked."""
    loose = make_source(tmp_path / "loose", synthetic=True)
    with pytest.raises(ValueError, match="loose.*specs.*manifest.json"):
        build_package(real, [loose], tmp_path / "x", experiment="E3", version=1, inspect_only=True,
                      require_provenance=True)
    specs_only = make_batch(tmp_path, "specs_only", 0, range(1))
    with pytest.raises(ValueError, match="specs_only.*render_manifest.json"):
        build_package(real, [specs_only], tmp_path / "x", experiment="E3", version=1, inspect_only=True,
                      require_provenance=True)
    old_render = rendered(make_batch(tmp_path, "old_render", 0, range(1)), code_sha256=None)
    with pytest.raises(ValueError, match="old_render.*no generator or code digest"):
        build_package(real, [old_render], tmp_path / "x", experiment="E3", version=1, inspect_only=True,
                      require_provenance=True)
    both = rendered(make_batch(tmp_path, "anchored_k2", 0, range(1)))
    build_package(real, [both], tmp_path / "x", experiment="E3", version=1, inspect_only=True,
                  require_provenance=True)
    script = Path(__file__).resolve().parents[1] / "scripts/23_launch_retrain.py"
    completed = subprocess.run([
        sys.executable, str(script), "--real", str(real), "--synth", str(loose), "--experiment", "E3",
        "--version", "1", "--out", str(tmp_path / "packages"), "--expected-real-train", "1"],
        capture_output=True, text=True)
    assert completed.returncode == 2 and "lack the records" in completed.stderr
    assert not (tmp_path / "packages").exists()


def test_e4_batches_are_sampled_at_the_kappa_of_f0(real: Path, runner: Path, tmp_path: Path) -> None:
    """A loop or control batch generated at another kappa than the E3 arm of f0 (a
    forgotten --kappa) is refused; both arms record the kappa and must agree."""
    e3_batch = rendered(make_batch(tmp_path, "anchored_k2", 0, range(3)))
    e3 = tmp_path / "e3_k2"
    build_package(real, [e3_batch], e3, experiment="E3", version=1, runner_source=runner,
                  require_provenance=True)
    parent = finish_runs(e3, tmp_path, [0])[0]
    options = dict(experiment="E4", version=1, model=parent, epochs=30, parent_package=e3,
                   runner_source=runner, require_provenance=True)
    wrong = rendered(make_batch(tmp_path, "loop_k1_at_1", 10000, range(2), kappa=1.0))
    with pytest.raises(ValueError, match="kappa of f0.*loop_k1_at_1.*kappa 1.0"):
        build_package(real, [e3_batch, wrong], tmp_path / "x", arm="loop", inspect_only=True, **options)
    loop = rendered(make_batch(tmp_path, "loop_k1", 20000, range(2)))
    loop_package = tmp_path / "loop_u1"
    result = build_package(real, [e3_batch, loop], loop_package, arm="loop", **options)
    assert result["training"]["e4"]["kappa"] == 2.0
    control = rendered(make_batch(tmp_path, "control_k1", 30000, range(2)))
    paired = build_package(real, [e3_batch, control], tmp_path / "control_u1", arm="control",
                           paired_package=loop_package, inspect_only=True, **options)
    assert paired["training"]["e4"]["kappa"] == 2.0
    blind = rendered(make_batch(tmp_path, "blind_like", 40000, range(2), kappa=None))
    with pytest.raises(ValueError, match="record their kappa.*blind_like"):
        build_package(real, [e3_batch, blind], tmp_path / "y", arm="loop", inspect_only=True, **options)


def test_cli_requires_a_declared_real_training_set(real: Path, tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/23_launch_retrain.py"
    command = [sys.executable, str(script), "--real", str(real), "--experiment", "E2", "--version", "0",
               "--out", str(tmp_path / "packages")]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 2
    assert "--real-train-list, --expected-real-train" in completed.stderr
    assert not (tmp_path / "packages").exists()
    # Looking at the data first creates nothing and needs no declaration.
    completed = subprocess.run([*command, "--inspect-only"], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["manifest"]["expected_real_train"] is None


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
    assert config["selection"]["seed_tie"] == "lowest_seed"
    completed = subprocess.run([sys.executable, str(package / "train.py"), "--dry-run"], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    (package / "labels/train/bogus.txt").write_text("")
    with pytest.raises(ValueError, match="Unmanifested dataset file"):
        module.verify_package(package)
