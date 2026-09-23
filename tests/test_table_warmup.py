"""
test_table_warmup.py
────────────────────
A just-started depth stream hands out holes and an unsettled temporal filter. The
median of those frames is a cloud no plane fits, and the operator sees "plane fit
lost its inliers" while looking at a perfectly clear table. The table measurement
therefore discards the first frames.

Run:
    pytest tests/test_table_warmup.py -v
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger("test_table_warmup")
EYE = np.eye(4)


def _calib_script():
    path = PROJECT_ROOT / "scripts" / "02_run_calibration.py"
    spec = importlib.util.spec_from_file_location("calib_warmup_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ColdCamera:
    """Depth at 700 mm once warm; before that, half the pixels drop out."""

    def __init__(self, cold_frames: int, shape=(720, 1280)):
        self.cold_frames = cold_frames
        self.shape = shape
        self.calls = 0
        self.intrinsics = {"fx": 645.0, "fy": 644.2, "ppx": 647.3, "ppy": 368.8}

    def get_frame(self):
        self.calls += 1
        depth = np.full(self.shape, 0.7)
        if self.calls <= self.cold_frames:
            # Holes and a wildly wrong reading in the rest: nothing 15 mm-consistent.
            depth[::2, :] = 0.0
            depth[1::2, :] = np.random.default_rng(self.calls).uniform(
                0.3, 1.1, size=(self.shape[0] // 2, self.shape[1]))
        return None, depth


def test_warmup_lets_the_fit_succeed(tmp_path):
    mod = _calib_script()
    cam = ColdCamera(cold_frames=25)
    rc = mod._measure_table(cam, EYE, tmp_path / "T_base_camera.npy", 10, log)
    assert rc == 0
    plane = json.loads((tmp_path / "table_plane.json").read_text(encoding="utf-8"))
    assert plane["z_top_mm"] == pytest.approx(700.0, abs=0.5)
    assert plane["n_inliers"] == plane["n_points"]
    assert cam.calls >= 25 + 10


def test_without_warmup_the_cold_stream_is_what_fails(tmp_path):
    """The operator's report: the fit, not the table, is what went wrong."""
    mod = _calib_script()
    cam = ColdCamera(cold_frames=25)
    rc = mod._measure_table(cam, EYE, tmp_path / "T_base_camera.npy", 10, log,
                            warmup_frames=0)
    assert rc == 1
    assert not (tmp_path / "table_plane.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
