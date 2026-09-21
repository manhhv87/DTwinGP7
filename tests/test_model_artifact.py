"""Checkpoint selection/provenance regressions; no downloaded model or hardware."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.logging.logger import FIELDNAMES, TrialLogger
from src.perception import MockDetector
from src.perception.model_artifact import (
    capture_config_files, detector_metadata, load_configured_detector,
    resolve_model_path, run_metadata,
)


class StubDetector:
    def __init__(self, model_path, conf, iou, class_heights_mm):
        self.path = model_path
        self.conf, self.iou = conf, iou
        self.class_names = ["carton", "plastic_box", "wood_box", "metal_box", "inox_box"]
        self.model = SimpleNamespace(overrides={"imgsz": 1280, "task": "segment"}, predictor=None)


@pytest.fixture
def checkpoint(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "aaa_unrelated.pt").write_bytes(b"wrong model")
    selected = models / "selected.pt"
    selected.write_bytes(b"selected checkpoint")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "experiment.yaml").write_text("model_path: models/selected.pt\n", encoding="utf-8")
    return selected


def test_explicit_selection_independent_of_cwd(checkpoint, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path.parent)
    det = load_configured_detector({"model_path": "models/selected.pt"}, tmp_path, StubDetector)
    assert Path(det.path) == checkpoint
    assert detector_metadata(det)["classes"][-1] == "inox_box"


def test_missing_configured_checkpoint_does_not_fallback_or_load(checkpoint, tmp_path):
    def must_not_load(**kwargs):
        pytest.fail("Must reject the path before constructing a model")
    with pytest.raises(FileNotFoundError, match="missing.pt"):
        load_configured_detector({"model_path": "models/missing.pt"}, tmp_path, must_not_load)
    with pytest.raises(ValueError, match="model_path"):
        resolve_model_path({"model_path": ""}, tmp_path)


def test_hash_cached_and_imgsz_follows_loaded_model(checkpoint, tmp_path, monkeypatch):
    import src.perception.model_artifact as artifacts
    calls = []
    original_hash = artifacts.file_sha256
    def count_hash(path):
        calls.append(Path(path))
        return original_hash(path)
    monkeypatch.setattr(artifacts, "file_sha256", count_hash)
    det = load_configured_detector({"model_path": checkpoint}, tmp_path, StubDetector)
    initial = detector_metadata(det)
    assert initial["sha256"] == hashlib.sha256(b"selected checkpoint").hexdigest()
    assert initial["inference"]["imgsz"] == 1280
    assert initial["inference"]["imgsz_source"] == "model.overrides"
    assert initial["inference"]["batch"] is None
    assert initial["inference"]["rect"] is None
    det.model.predictor = SimpleNamespace(
        args=SimpleNamespace(imgsz=1280, half=False, device="cpu"),
        device="cpu", imgsz=[1280, 1280])
    observed = detector_metadata(det)
    assert observed["inference"]["imgsz_source"] == "predictor.args"
    assert observed["inference"]["runtime_device"] == "cpu"
    assert calls == [checkpoint]


def test_unknown_imgsz_is_not_invented(checkpoint, tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "ultralytics.utils", SimpleNamespace())
    det = load_configured_detector({"model_path": checkpoint}, tmp_path, StubDetector)
    det.model.overrides = {}
    assert detector_metadata(det)["inference"]["imgsz"] is None
    assert detector_metadata(det)["inference"]["imgsz_source"] == "unknown"


def test_sidecar_tracks_run_and_refreshes_runtime(checkpoint, tmp_path, capsys):
    det = load_configured_detector({"model_path": checkpoint}, tmp_path, StubDetector)
    config = {"model_path": str(checkpoint), "conf_threshold": 0.5}
    log = TrialLogger(tmp_path / "trials.csv", run_metadata=lambda: run_metadata(
        config, det, entrypoint="test", runtime={"blind_label": "A"}))
    det.model.predictor = SimpleNamespace(args={"imgsz": 1280}, device="cpu", imgsz=[1280, 1280])
    log.log_trial(1, True, extra={"run_id": "cannot override identity"})
    meta = json.loads(log.metadata_path.read_text(encoding="utf-8"))
    with log.csv_path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["run_id"] == meta["run_id"] == log.run_id
    assert meta["rows_written"] == 1
    assert meta["first_data_row"] == 1
    assert meta["provenance"]["model"]["inference"]["runtime_device"] == "cpu"
    assert meta["provenance"]["config"] == config
    assert capsys.readouterr().out == ""


def test_old_csv_keeps_header_and_distinct_append_metadata(tmp_path):
    path = tmp_path / "old.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=FIELDNAMES).writeheader()
    first = TrialLogger(path, run_metadata={"arm": "A"})
    first.log_trial(1, True)
    second = TrialLogger(path, run_metadata={"arm": "B"})
    second.log_trial(1, False)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == FIELDNAMES
    assert all(len(row) == len(FIELDNAMES) for row in rows)
    assert first.metadata_path != second.metadata_path
    meta = json.loads(second.metadata_path.read_text(encoding="utf-8"))
    assert meta["first_data_row"] == 2 and meta["rows_written"] == 1
    assert meta["csv_has_run_id"] is False


def test_camera_gui_uses_config_and_mock_is_explicit(checkpoint, tmp_path):
    from src.orchestrator.viewports.mixin_camera import CameraMixin
    host = SimpleNamespace(_project_root=tmp_path, _cam_source="D455")
    det = CameraMixin._open_detector(host, StubDetector, MockDetector)
    assert Path(det.path) == checkpoint
    checkpoint.unlink()
    host._cam_source = "Auto (D455→Mock)"
    with pytest.raises(FileNotFoundError):
        CameraMixin._open_detector(host, StubDetector, MockDetector)
    host._cam_source = "Mock"
    assert isinstance(CameraMixin._open_detector(host, StubDetector, MockDetector), MockDetector)


def test_cli_and_experiment_gui_share_selected_model(checkpoint, tmp_path, monkeypatch):
    import src.perception as perception
    from src.orchestrator.viewports.mixin_experiment import ExperimentMixin
    monkeypatch.setattr(perception, "ObjectDetector", StubDetector)
    monkeypatch.setattr(perception, "D455Camera", lambda: SimpleNamespace())
    monkeypatch.setattr(perception, "PerceptionNode", lambda camera, detector, queue:
                        SimpleNamespace(detector=detector))
    path = Path(__file__).resolve().parents[1] / "scripts" / "03_run_experiment.py"
    spec = importlib.util.spec_from_file_location("runner_provenance_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.PROJECT_ROOT = tmp_path
    config = {"model_path": "models/selected.pt", "class_heights_mm": {
        name: 80 for name in ["carton", "plastic_box", "wood_box", "metal_box", "inox_box"]}}
    _, cli = runner.build_perception("real", config)
    node, threaded = ExperimentMixin._build_experiment_perception(
        SimpleNamespace(_project_root=tmp_path), config, None, "real", 1)
    assert threaded is True
    assert detector_metadata(cli)["sha256"] == detector_metadata(node.detector)["sha256"]
    assert cli.path == node.detector.path == str(checkpoint)


def test_checkpoint_replaced_during_loading_is_rejected(checkpoint, tmp_path):
    class ReplacingDetector(StubDetector):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            Path(self.path).write_bytes(b"replacement with different size")

    with pytest.raises(ValueError, match="changed while"):
        load_configured_detector({"model_path": checkpoint}, tmp_path, ReplacingDetector)


def test_config_file_identity_stays_with_loaded_configuration(checkpoint, tmp_path):
    config_file = tmp_path / "config" / "experiment.yaml"
    captured = capture_config_files([config_file])
    loaded = {"model_path": "models/selected.pt"}
    expected_hash = hashlib.sha256(config_file.read_bytes()).hexdigest()
    config_file.write_text("model_path: a_new_model.pt\n", encoding="utf-8")
    result = run_metadata(loaded, MockDetector(), entrypoint="test", config_files=captured)
    assert result["config_files"][0]["sha256"] == expected_hash
    assert result["config"] == loaded
    result["config_files"][0]["sha256"] = "tampered return value"
    assert captured[0]["sha256"] == expected_hash


def test_camera_state_is_reset_even_if_cleanup_also_fails():
    from src.orchestrator.viewports.mixin_camera import CameraMixin

    def fail(*args):
        raise RuntimeError("not available")

    updates = []
    host = SimpleNamespace(
        _cam_source="D455", _cam_color_size=(1280, 720), _cam_fps=30,
        _cam_use_detector=True, _cam_running=True,
        _open_camera=lambda *args: (SimpleNamespace(stop=fail), "D455"),
        _open_detector=fail,
        _signals=SimpleNamespace(status=SimpleNamespace(emit=lambda *args: None),
                                 camera_result=SimpleNamespace(emit=updates.append)))
    CameraMixin._camera_loop(host)
    assert host._cam_running is False
    assert updates == [{"stopped": True}]


def test_blind_setup_error_keeps_checkpoint_identity_out_of_console(tmp_path, monkeypatch, capsys):
    import io
    import logging
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "03_run_experiment.py"
    spec = importlib.util.spec_from_file_location("runner_blind_error_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.PROJECT_ROOT = tmp_path
    monkeypatch.setattr(sys, "argv", [str(path), "--mode", "sim", "--blind-label", "A"])
    monkeypatch.setattr(runner, "load_yaml", lambda path: {"calibration_path": "unused.npy"})
    monkeypatch.setattr(runner.CellConfig, "from_yaml", lambda path:
                        SimpleNamespace(robot=SimpleNamespace(home_joints_deg=[0] * 6)))
    def fail(*args):
        raise RuntimeError("Cannot load secret_E3_kappa4_seed3.pt")
    monkeypatch.setattr(runner, "build_perception", fail)
    stream = io.StringIO()
    console = logging.StreamHandler(stream)
    log_file = tmp_path / "private.log"
    disk = logging.FileHandler(log_file, encoding="utf-8")
    root = logging.getLogger()
    old_handlers, old_level = root.handlers[:], root.level
    try:
        root.handlers = [console, disk]
        root.setLevel(logging.INFO)
        monkeypatch.setattr(runner, "setup_logging", lambda *args, **kwargs:
                            logging.getLogger("experiment"))
        assert runner.main() == 5
        public = stream.getvalue() + capsys.readouterr().out
        assert "secret_E3" not in public
        assert "Perception setup failed" in public
        disk.flush()
        assert "secret_E3_kappa4_seed3.pt" in log_file.read_text(encoding="utf-8")
    finally:
        console.close()
        disk.close()
        root.handlers = old_handlers
        root.setLevel(old_level)
