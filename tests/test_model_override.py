"""
test_model_override.py
──────────────────────
--model-path lets each arm of one blinded campaign run its own checkpoint (E3, E4, E5
compare models, not depth modes). It must replace model_path from the YAML, including
one set under real:, and leave it alone when not given.

Run:
    pytest tests/test_model_override.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _runner():
    path = PROJECT_ROOT / "scripts" / "03_run_experiment.py"
    spec = importlib.util.spec_from_file_location("runner_model_override_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_override_replaces_yaml_value():
    runner = _runner()
    config = {"model_path": "models/e2_seed1_best.pt"}
    config.update({"model_path": "models/from_real_block.pt"})  # as real: would
    runner.apply_model_override(config, "models/e3_anchored.pt")
    assert config["model_path"] == "models/e3_anchored.pt"


def test_no_override_keeps_yaml_value():
    runner = _runner()
    config = {"model_path": "models/e2_seed1_best.pt"}
    runner.apply_model_override(config, None)
    assert config["model_path"] == "models/e2_seed1_best.pt"


def test_flag_is_parsed(monkeypatch):
    runner = _runner()
    monkeypatch.setattr(sys, "argv", ["03_run_experiment.py", "--mode", "sim",
                                      "--model-path", "models/e3_wide.pt"])
    assert runner.parse_args().model_path == "models/e3_wide.pt"
