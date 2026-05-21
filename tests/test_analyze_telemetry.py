"""
test_analyze_telemetry.py
─────────────────────────
Smoke test cho 05_analyze_telemetry.py — chạy được mà không lỗi với 1 CSV synthetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def synthetic_csv(tmp_path):
    """Sinh telemetry CSV giả: 10s sinusoidal motion ở 10Hz."""
    n = 100
    t = np.linspace(0, 10, n)
    df = pd.DataFrame({
        "timestamp": t + 1700000000.0,            # synthetic UNIX time
        "j1": 30 * np.sin(t * 0.5),
        "j2": 20 * np.cos(t * 0.5),
        "j3": 10 * np.sin(t * 1.0),
        "j4": np.zeros(n),
        "j5": 15 * np.cos(t * 0.8),
        "j6": -10 * np.sin(t * 0.3),
        "running": [1 if 1 < ti < 9 else 0 for ti in t],
        "alarm": [0] * n,
    })
    path = tmp_path / "telemetry_test.csv"
    df.to_csv(path, index=False)
    return path


class TestAnalyzeTelemetry:
    def test_load_telemetry_adds_time_rel(self, synthetic_csv):
        from importlib import import_module
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        mod = import_module("05_analyze_telemetry")
        df = mod.load_telemetry(synthetic_csv)
        assert "time_rel" in df.columns
        assert df["time_rel"].iloc[0] == 0
        assert df["time_rel"].iloc[-1] > 9

    def test_load_missing_columns_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("timestamp,j1\n1.0,0.0\n")
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from importlib import import_module
        mod = import_module("05_analyze_telemetry")
        with pytest.raises(ValueError, match="thiếu cột"):
            mod.load_telemetry(bad)

    def test_compute_velocity_shape(self, synthetic_csv):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from importlib import import_module
        mod = import_module("05_analyze_telemetry")
        df = mod.load_telemetry(synthetic_csv)
        vel = mod.compute_velocity(df)
        assert len(vel) == len(df)
        assert all(f"v{i}" in vel.columns for i in range(1, 7))

    def test_main_writes_figures(self, synthetic_csv, tmp_path, monkeypatch):
        """End-to-end smoke test: gọi main() → check 4 PNG ra figures dir."""
        out_dir = tmp_path / "figs"
        monkeypatch.setattr(sys, "argv", [
            "05_analyze_telemetry.py", str(synthetic_csv),
            "--no-show", "--out-dir", str(out_dir),
        ])
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from importlib import import_module, reload
        mod = import_module("05_analyze_telemetry")
        reload(mod)                                 # tránh module-level state cache
        rc = mod.main()
        assert rc == 0
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) == 4                       # 4 figures
        names = {p.name for p in pngs}
        assert any("joint_trajectory" in n for n in names)
        assert any("joint_velocity" in n for n in names)
        assert any("drift_events" in n for n in names)
        assert any("cycle_time" in n for n in names)
