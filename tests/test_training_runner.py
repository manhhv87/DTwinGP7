"""Offline checks of run records and comparison guards; no model training."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "packaged_runner", Path(__file__).resolve().parents[1] / "scripts/train_packaged_model.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_validation_uses_unrounded_composite_and_epoch():
    trainer = SimpleNamespace(metrics={"metrics/mAP50-95(B)": 0.900000001,
                                       "metrics/mAP50-95(M)": 0.800000009},
                              fitness=1.700000010, epoch=7)
    result = runner.validation_record(trainer)
    assert result["epoch"] == 8
    assert result["score"] == 1.700000010
    trainer.fitness = 0.8
    with pytest.raises(ValueError, match="criterion"):
        runner.validation_record(trainer)
    trainer.metrics["metrics/mAP50-95(B)"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        runner.validation_record(trainer)


def test_recipe_refuses_automatic_batch_changes():
    trainer = SimpleNamespace(args=SimpleNamespace(batch=8), batch_size=4)
    with pytest.raises(ValueError, match="batch size"):
        runner.assert_recipe(trainer, {"batch": 8})


@pytest.mark.parametrize("actual_epochs,expected_status", [(2, "complete"), (1, "failed")])
def test_runner_records_selection_steps_and_incomplete_runs(tmp_path, monkeypatch, actual_epochs, expected_status):
    # A minimal trainer double exercises callbacks and accounting, not model accuracy.
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "8.4.66")
    fake_torch = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda=None),
                                 cuda=SimpleNamespace(device_count=lambda: 0))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class Optimizer:
        def register_step_post_hook(self, callback):
            self.callback = callback
            return SimpleNamespace(remove=lambda: None)

    class Model:
        def __init__(self, path):
            assert Path(path).is_file()
            self.callbacks = {}
            self.task = "segment"
            self.names = runner.CLASSES

        def add_callback(self, event, callback):
            self.callbacks[event] = callback

        def train(self, **kwargs):
            output = Path(kwargs["project"]) / kwargs["name"]
            self.trainer = trainer = SimpleNamespace(args=SimpleNamespace(**kwargs),
                                                      batch_size=kwargs["batch"], save_dir=output,
                                                      optimizer=Optimizer(),
                                                      train_loader=SimpleNamespace(dataset=[1, 2]),
                                                      test_loader=SimpleNamespace(dataset=[1]))
            self.callbacks["on_train_start"](trainer)
            rows = ["epoch,metric"]
            for epoch in range(actual_epochs):
                self.callbacks["on_train_batch_start"](trainer)
                trainer.optimizer.callback()  # one real step in the double per epoch
                trainer.epoch = epoch
                trainer.metrics = {"metrics/mAP50-95(B)": .7, "metrics/mAP50-95(M)": .8}
                trainer.fitness = trainer.best_fitness = 1.5
                self.callbacks["on_model_save"](trainer)
                rows.append(f"{epoch + 1},1.5")
            (output / "results.csv").write_text("\n".join(rows) + "\n")
            (output / "weights").mkdir()
            (output / "weights/best.pt").write_bytes(b"fake checkpoint for offline test")
            self.ckpt = {"train_metrics": {"fitness": 1.5}}

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    monkeypatch.setattr(runner, "verify_package", lambda _: None)
    (tmp_path / "initial.pt").write_bytes(b"fake initial checkpoint")
    config = {"ultralytics_version": "8.4.66", "experiment": "E4", "version": 1,
              "initialization": {"model": "initial.pt", "sha256": runner.sha256(tmp_path / "initial.pt")},
              "recipe": {"epochs": 2, "batch": 8}}
    manifest = {"images": [], "counts": {"real_train": 1, "real_val": 1, "synthetic_train": 1, "train": 2, "val": 1}}
    runner.write_json(tmp_path / "training.json", config)
    runner.write_json(tmp_path / "manifest.json", manifest)
    if expected_status == "failed":
        with pytest.raises(ValueError, match="Incomplete"):
            runner.run_training(tmp_path, config, manifest, [0], "cpu")
    else:
        runner.run_training(tmp_path, config, manifest, [0], "cpu")
    record = json.loads((tmp_path / "runs/seed0/run_record.json").read_text())
    assert record["status"] == expected_status
    assert record["optimizer_updates"] == actual_epochs
    assert record["selected_validation"]["epoch"] == actual_epochs  # latest tie, full precision
    assert len(record["validation_history"]) == actual_epochs
    assert record["wall_seconds"] >= 0
    with pytest.raises(ValueError, match="already exists"):
        runner.run_training(tmp_path, config, manifest, [0], "cpu")
