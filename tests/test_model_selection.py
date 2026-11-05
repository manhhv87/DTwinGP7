"""Selection rules applied to returned run records; no model or GPU is used."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from src.training.package import build_package, sha256
from src.training.selection import (
    best_run,
    load_run,
    select_deployed,
    select_e4_adaptation,
    select_e4_final,
    select_kappa,
    write_record,
)
from src.utils.dataset_check import EXPECTED_NAMES


def make_source(root: Path, *, synthetic: bool = False, kappa: float | None = None) -> Path:
    root.mkdir(parents=True)
    (root / "dataset.yaml").write_text(yaml.safe_dump({"names": EXPECTED_NAMES, "nc": 5}), encoding="utf-8")
    for split in (["train"] if synthetic else ["train", "val"]):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "images" / split / "sample.png").write_bytes(f"{root.parent.name}/{root.name}/{split}".encode())
        (root / "labels" / split / "sample.txt").write_text("4 0.1 0.1 0.7 0.1 0.4 0.8\n", encoding="utf-8")
    if kappa is not None:
        specs = root.parent / "specs"
        specs.mkdir()
        (specs / "manifest.json").write_text(json.dumps({
            "spec_version": 1, "seed": 0, "mode": "anchored", "kappa": kappa,
            "files": ["scene_000000.json"]}), encoding="utf-8")
    return root


def finish(package: Path, scores: dict[int, list[tuple[float, float]]], *, selected_epoch: int | None = None,
           pad: bool = True) -> None:
    """Complete runs as train.py leaves them: per seed, (box, mask) for each epoch. pad
    fills the remaining epochs of the package's recipe with zero scores, which never
    become the best epoch, so a run holds every epoch as train.py requires."""
    recipe_epochs = json.loads((package / "training.json").read_text())["recipe"]["epochs"]
    for seed, epochs in scores.items():
        if pad:
            epochs = list(epochs) + [(0.0, 0.0)] * (recipe_epochs - len(epochs))
        run = package / "runs" / f"seed{seed}"
        (run / "weights").mkdir(parents=True)
        (run / "weights/best.pt").write_bytes(f"{package.name} seed {seed}".encode())
        history = [{"epoch": i + 1, "box_map50_95": box, "mask_map50_95": mask, "score": box + mask}
                   for i, (box, mask) in enumerate(epochs)]
        best = max(history, key=lambda entry: (entry["score"], entry["epoch"]))
        if selected_epoch is not None:
            best = history[selected_epoch - 1]
        (run / "run_record.json").write_text(json.dumps({
            "status": "complete", "seed": seed,
            "manifest_sha256": sha256(package / "manifest.json"),
            "training_sha256": sha256(package / "training.json"),
            "validation_history": history, "selected_validation": best,
            "checkpoint": f"runs/seed{seed}/weights/best.pt",
            "checkpoint_sha256": sha256(run / "weights/best.pt")}), encoding="utf-8")
        lines = ["epoch,time,metrics/mAP50-95(B),metrics/mAP50-95(M)"]
        lines += [f"{e['epoch']},1.0,{e['box_map50_95']:.6g},{e['mask_map50_95']:.6g}" for e in history]
        (run / "results.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def real(tmp_path: Path) -> Path:
    return make_source(tmp_path / "real" / "yolo")


@pytest.fixture
def runner(tmp_path: Path) -> Path:
    path = tmp_path / "runner.py"
    path.write_text("# standalone training runner fixture\n", encoding="utf-8")
    return path


def e3_package(tmp_path: Path, real: Path, runner: Path, name: str, kappa: float | None = None, **options) -> Path:
    synth = make_source(tmp_path / name / "dataset", synthetic=True, kappa=kappa)
    package = tmp_path / "packages" / name
    build_package(real, [synth], package, experiment="E3", version=1, runner_source=runner, **options)
    return package


# Box and mask per run; sums 1.50, 1.52, 1.51, 1.50 and 1.54, the best one unique.
SCORES = [(0.80, 0.70), (0.82, 0.70), (0.79, 0.72), (0.81, 0.69), (0.80, 0.74)]


def test_kappa_by_exact_mean_with_ties_to_the_smaller_kappa(tmp_path, real, runner):
    k2, k4, k1 = (e3_package(tmp_path, real, runner, name) for name in ("k2", "k4", "k1"))
    finish(k2, {seed: [(0.5, 0.5), SCORES[seed]] for seed in range(5)})
    # The same scores on other seeds: an exact tie of the means goes to the smaller kappa.
    finish(k4, {seed: [SCORES[(seed + 2) % 5]] for seed in range(5)})
    # 1e-12 below kappa 2 on one seed: any rounding would tie and pick kappa 1.
    lower = [(box, mask - (1e-12 if seed == 0 else 0.0)) for seed, (box, mask) in enumerate(SCORES)]
    finish(k1, {seed: [lower[seed]] for seed in range(5)})
    chosen = select_kappa({4: k4, 2: k2})
    assert chosen["selected_kappa"] == 2.0
    assert chosen["candidates"][0]["mean_score"] == chosen["candidates"][1]["mean_score"]
    assert select_kappa({1: k1, 2: k2})["selected_kappa"] == 2.0
    deployed = chosen["deployed"]
    assert (deployed["seed"], deployed["epoch"]) == (4, 2)   # 0.80 + 0.74 is the best run of kappa 2
    assert deployed["checkpoint_sha256"] == sha256(k2 / "runs/seed4/weights/best.pt")


def test_tied_runs_deploy_the_lowest_seed():
    runs = [{"seed": 2, "score": 1.5}, {"seed": 0, "score": 1.5}, {"seed": 1, "score": 1.4}]
    assert best_run(runs)["seed"] == 0
    with pytest.raises(ValueError, match="distinct seeds"):
        best_run([{"seed": 0, "score": 1.0}, {"seed": 0, "score": 1.2}])


def test_each_run_is_rechecked_against_its_history_and_csv(tmp_path, real, runner):
    package = e3_package(tmp_path, real, runner, "k2", seeds=[0], epochs=3)
    finish(package, {0: [(0.7, 0.7), (0.8, 0.7), (0.7, 0.8)]}, selected_epoch=2)
    # Epochs 2 and 3 tie at 1.5; the kept checkpoint must be the later one.
    with pytest.raises(ValueError, match="not the best validation epoch 3"):
        load_run(package, 0)
    run = package / "runs/seed0"
    record = json.loads((run / "run_record.json").read_text())
    record["selected_validation"] = record["validation_history"][2]
    (run / "run_record.json").write_text(json.dumps(record))
    assert load_run(package, 0)["epoch"] == 3
    (run / "results.csv").write_text("epoch,time,metrics/mAP50-95(B),metrics/mAP50-95(M)\n"
                                     "1,1,0.7,0.7\n2,1,0.8,0.7\n3,1,0.7,0.81\n")
    with pytest.raises(ValueError, match="results.csv epoch 3 disagrees"):
        load_run(package, 0)
    record["status"] = "failed"
    (run / "run_record.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="not complete"):
        load_run(package, 0)


def test_a_run_shorter_than_the_recipe_is_refused(tmp_path, real, runner):
    """A complete-looking record of 3 epochs in a 130-epoch package does not compete."""
    package = e3_package(tmp_path, real, runner, "k2", seeds=[0])
    finish(package, {0: [(0.7, 0.7), (0.8, 0.7), (0.7, 0.8)]}, pad=False)
    assert load_run(package, 0)["epoch"] == 3                 # the record itself is consistent
    with pytest.raises(ValueError, match="holds 3 epochs, the package's recipe 130"):
        load_run(package, 0, 130)
    with pytest.raises(ValueError, match="holds 3 epochs"):
        select_deployed([package], "e3_wide")


def test_candidates_must_be_comparable_and_correctly_labelled(tmp_path, real, runner):
    k1 = e3_package(tmp_path, real, runner, "k1", kappa=1.0)
    k2 = e3_package(tmp_path, real, runner, "k2", kappa=2.0)
    longer = e3_package(tmp_path, real, runner, "k4", kappa=4.0, epochs=140)
    for package in (k1, k2, longer):
        finish(package, {seed: [SCORES[seed]] for seed in range(5)})
    with pytest.raises(ValueError, match="record kappa \\[1.0\\], not 2.0"):
        select_kappa({2: k1, 1: k2})
    with pytest.raises(ValueError, match="different training recipes"):
        select_kappa({1: k1, 4: longer})
    assert select_kappa({1: k1, 2: k2})["selected_kappa"] == 1.0
    (k1 / "runs/seed4/weights/best.pt").write_bytes(b"changed after training")
    with pytest.raises(ValueError, match="differs from its run record"):
        select_kappa({1: k1, 2: k2})


@pytest.fixture
def e4(tmp_path, real, runner):
    """f0 (E3, one seed), then update 1 of both arms with seeds 0..2."""
    f0 = e3_package(tmp_path, real, runner, "anchored", seeds=[0])
    finish(f0, {0: [SCORES[0]]})
    initial = tmp_path / "anchored/dataset"
    packages = {}
    for arm, epochs in (("loop", 30), ("control", 30), ("control_long", 40)):
        batch = make_source(tmp_path / f"{arm}1" / "dataset", synthetic=True)
        packages[arm] = tmp_path / "packages" / f"{arm}_u1"
        build_package(real, [initial, batch], packages[arm], experiment="E4", version=1,
                      model=f0 / "runs/seed0/weights/best.pt", epochs=epochs, parent_package=f0,
                      arm=arm.split("_")[0], runner_source=runner)
    finish(packages["loop"], {0: [SCORES[0]], 1: [SCORES[4]], 2: [SCORES[1]]})
    finish(packages["control"], {0: [SCORES[1]], 1: [SCORES[1]], 2: [SCORES[2]]})
    finish(packages["control_long"], {seed: [SCORES[seed]] for seed in range(3)})
    return packages


def test_e4_adaptation_model_is_the_best_update_one_loop_run(e4):
    chosen = select_e4_adaptation([e4["loop"]])
    assert chosen["deployed"]["seed"] == 1
    with pytest.raises(ValueError, match="E4 arm and update expected \\(loop\\)"):
        select_e4_adaptation([e4["control"]])


def test_e4_final_models_need_the_same_schedule_in_both_arms(e4):
    final = select_e4_final([e4["loop"]], [e4["control"]])
    assert (final["update"], final["epochs"]) == (1, 30)
    assert final["arms"]["loop"]["deployed"]["seed"] == 1
    assert final["arms"]["control"]["deployed"]["seed"] == 0   # seeds 0 and 1 tie
    with pytest.raises(ValueError, match="different training recipes"):
        select_e4_final([e4["loop"]], [e4["control_long"]])
    with pytest.raises(ValueError, match="arm and update expected"):
        select_e4_final([e4["control"]], [e4["control_long"]])


def test_update_two_per_seed_packages_form_one_arm(tmp_path, real, runner, e4):
    batches = [tmp_path / "anchored/dataset", tmp_path / "loop1/dataset",
               make_source(tmp_path / "loop2" / "dataset", synthetic=True)]
    per_seed = []
    for seed in range(3):
        package = tmp_path / "packages" / f"loop_u2_s{seed}"
        build_package(real, batches, package, experiment="E4", version=2, seeds=[seed], epochs=30,
                      model=e4["loop"] / f"runs/seed{seed}/weights/best.pt", parent_package=e4["loop"],
                      arm="loop", runner_source=runner)
        finish(package, {seed: [SCORES[seed + 2]]})
        per_seed.append(package)
    chosen = select_deployed(per_seed, "e4_loop_final")
    assert chosen["deployed"]["seed"] == 2   # 0.80 + 0.74
    with pytest.raises(ValueError, match="update 1"):
        select_e4_adaptation(per_seed)
    with pytest.raises(ValueError, match="same number of updates"):
        select_e4_final(per_seed, [e4["control"]])


def test_cli_writes_the_record_once_with_its_hash(tmp_path, real, runner):
    k1, k2 = (e3_package(tmp_path, real, runner, name) for name in ("k1", "k2"))
    for package in (k1, k2):
        finish(package, {seed: [SCORES[seed]] for seed in range(5)})
    path = Path(__file__).resolve().parents[1] / "scripts/select_trained_models.py"
    spec = importlib.util.spec_from_file_location("select_trained_models", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    out = tmp_path / "selection/e3_kappa.json"
    argv = ["kappa", "--candidate", f"1={k1}", "--candidate", f"2={k2}", "--out", str(out)]
    assert cli.main(argv) == 0
    record = json.loads(out.read_text())
    assert record["selected_kappa"] == 1.0 and record["deployed"]["seed"] == 4
    assert (tmp_path / "selection/e3_kappa.json.sha256").read_text().split()[0] == sha256(out)
    with pytest.raises(SystemExit) as refused:
        cli.main(argv)
    assert refused.value.code == 2
    with pytest.raises(ValueError, match="overwrite"):
        write_record({"decision": "kappa"}, out)
