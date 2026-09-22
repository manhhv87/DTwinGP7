"""
test_touch_test.py
──────────────────
The touch-test tool inverts the camera model: pixel → ray → table plane → base X, Y.
Checked against project_points (the forward model) on the measured calibration of
2026-09-18, and the ChArUco detection is checked on the printed board artwork.

Run:
    pytest tests/test_touch_test.py -v
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from src.perception.depth_modes import TablePlane, project_points  # noqa: E402

CALIB = PROJECT_ROOT / "config" / "calibration"


@pytest.fixture(scope="module")
def cell():
    T = np.load(CALIB / "T_base_camera.npy")
    plane = TablePlane.from_dict(json.loads((CALIB / "table_plane.json").read_text(encoding="utf-8")))
    intr = json.loads((CALIB / "T_base_camera_meta.json").read_text(encoding="utf-8"))["intrinsics"]
    return T, plane, intr


def test_pixel_to_base_inverts_projection(cell):
    from touch_test import pixel_to_base
    T, plane, intr = cell
    for x, y, h in [(525., -160., 0.), (625., 200., 3.), (575., 20., 3.), (400., -300., 0.)]:
        p = np.array([x, y, plane.z_at(x, y) + h / plane.normal[2]])
        u, v = project_points(intr, T, p[None, :])[0]
        back = pixel_to_base(intr, T, plane, u, v, height_mm=h)
        assert back is not None
        assert np.allclose(back[:2], (x, y), atol=0.05), (x, y, back)


def test_pick_spread_covers_the_board():
    from touch_test import pick_spread
    ids = np.arange(24)
    pts = np.array([[c * 100.0, r * 100.0] for r in range(4) for c in range(6)])
    sel = pick_spread(ids, pts, 8)
    assert len(sel) == 8 and len(set(sel)) == 8
    chosen = pts[sel]
    # the four extreme corners of a 6x4 grid must all be in the selection
    for corner in ([0., 0.], [500., 0.], [0., 300.], [500., 300.]):
        assert any(np.allclose(c, corner) for c in chosen)


def test_board_artwork_is_detected(tmp_path):
    """The printed A3 board (7x5, 45/34 mm, DICT_4X4_50) is what the tool has to detect."""
    fitz = pytest.importorskip("fitz")
    pdf = PROJECT_ROOT / "docs" / "charuco_a3_o45mm.pdf"
    img = tmp_path / "board.png"
    fitz.open(str(pdf))[0].get_pixmap(dpi=60).save(str(img))
    # Any intrinsics will do for detection; base coordinates are not checked here.
    r = subprocess.run([sys.executable, str(PROJECT_ROOT / "tools" / "touch_test.py"),
                        "--square-mm", "45", "--marker-mm", "34", "--image", str(img),
                        "--no-prompt", "--out-dir", str(tmp_path)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "8 chosen" in r.stdout
    assert list(tmp_path.glob("touch_test_*.png"))
