"""
test_score_renders.py
─────────────────────
scripts/fit_hard_profile.py score on what production renders write (rgb, instance map,
meta and the scene specs; no linear arrays): the per-frame measures, the set summary in
the units of the real references, and the --run shorthand for a rendered set folder.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.synthgen.sensor_model import LUMINANCE_709, srgb_to_linear

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("fit_hard_profile", ROOT / "scripts" / "fit_hard_profile.py")
fhp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fhp)

# The cell's condition check criteria (config/condition_check.yaml of the cell branch).
CRITERIA = {"upper_area": [230, 1080, 30, 455], "lower_area": [230, 1080, 495, 700],
            "cards": [430, 845, 220, 400],
            "hard": {"block_px": 32, "min_green_share": {"upper_area": 0.38, "lower_area": 0.345},
                     "luminance": [10.0, 45.0], "bare_relative_brightness": 0.5, "max_bare_share": 0.05}}
SHEET, LOWER, CARD, PART = (14, 24, 16), (20, 27, 21), (97, 99, 93), (60, 62, 58)


def row(**over) -> dict:
    base = {"frame": "f", "pass": True, "problems": [], "card_brightness": 97.0,
            "upper": [0.45, 20.0], "lower": [0.40, 24.0], "noise_upper_sheet": 2.0,
            "noise_lower_sheet": 2.1}
    return {**base, **over}


def test_rule_of_names_the_rule_not_the_value():
    assert fhp.rule_of("upper_area: brightness 54.6") == "upper_area: brightness"
    assert fhp.rule_of("lower_area: green share 0.331") == "lower_area: green share"
    assert fhp.rule_of("upper_area: bare table") == "upper_area: bare table"


def test_summary_counts_the_check_and_keeps_only_clean_noise_patches():
    rows = [row(),
            row(**{"pass": False, "problems": ["upper_area: bare table"],
                   "noise_upper_sheet": 9.0, "noise_clean": {"upper": False, "lower": True}}),
            row(**{"pass": False, "problems": ["upper_area: bare table", "lower_area: brightness 46.3"],
                   "box_over_upper_sheet": [["inox_box", 5.0], ["metal_box", 7.0]]}),
            row(**{"box_over_upper_sheet": [["inox_box", 6.0]], "exposure_gain": 1.5})]
    s = fhp.score_summary(rows)
    assert (s["frames"], s["pass"], s["pass_fraction"]) == (4, 2, 0.5)
    assert s["pass_without_bare_rule"] == 3
    assert s["fails_by_rule"] == {"lower_area: brightness": 1, "upper_area: bare table": 2}
    assert s["noise_upper_sheet"]["n"] == 3 and s["noise_upper_sheet"]["p95"] == 2.0
    assert s["noise_lower_sheet"]["n"] == 4
    assert s["exposure_gain"] == {"n": 1, **{f"p{q}": 1.5 for q in fhp.QUANTILES}}
    inox = s["box_over_upper_sheet"]["inox_box"]
    assert inox["n"] == 2 and inox["median"] == 5.5 and inox["p25"] == 5.25 and inox["range"] == [5.0, 6.0]
    text = fhp.format_summary(s, "set")
    assert "2/4 pass (0.5)" in text and "3/4 without the bare-table rule" in text
    assert "inox_box" in text and "noise upper sheet" in text


def _project(spec: dict, top_mm) -> tuple[int, int]:
    cam = spec["camera"]
    T, intr = np.asarray(cam["T_BC_mm"], float), cam["intrinsics"]
    pc = T[:3, :3].T @ (np.asarray(top_mm, float) - T[:3, 3])
    return (int(round(intr["fx"] * pc[0] / pc[2] + intr["ppx"])),
            int(round(intr["fy"] * pc[1] / pc[2] + intr["ppy"])))


@pytest.fixture()
def rendered_set(tmp_path: Path) -> Path:
    """One frame as the renderer writes it: an inox case on the green sheet."""
    import cv2

    syn = yaml.safe_load((ROOT / "config" / "synthgen.yaml").read_text(encoding="utf-8"))
    z = syn["objects"]["table_z_mm"]
    spec = {"camera": {"T_BC_mm": np.load(ROOT / "config/calibration/T_base_camera.npy").tolist(),
                       "intrinsics": syn["camera"]["intrinsics"]},
            "objects": [{"class_name": "inox_box", "class_id": 4, "mesh": "models/objects/inox_box.stl",
                         "xyz_mm": [575.0, 20.0, z], "yaw_deg": 0.0}]}
    u, v = _project(spec, [575.0, 20.0, z + syn["objects"]["mesh_sizes_mm"]["models/objects/inox_box.stl"][2]])
    rgb = np.zeros((720, 1280, 3), np.uint8)
    rgb[:, :] = SHEET
    rgb[495:, :] = LOWER
    rgb[220:400, 430:845] = CARD
    inst = np.ones((720, 1280), np.uint16)
    rgb[v - 20:v + 21, u - 20:u + 21] = PART
    inst[v - 20:v + 21, u - 20:u + 21] = 14
    run = tmp_path / "anchored_k1"
    (run / "render").mkdir(parents=True)
    (run / "specs").mkdir()
    cv2.imwrite(str(run / "render/frame_000000_rgb.png"), rgb[..., ::-1])
    cv2.imwrite(str(run / "render/frame_000000_inst.png"), inst)
    (run / "render/frame_000000_meta.json").write_text(json.dumps(
        {"instances": {"14": 4}, "scene": "scene_000000.json", "exposure_gain": 1.2}), encoding="utf-8")
    (run / "specs/scene_000000.json").write_text(json.dumps(spec), encoding="utf-8")
    (tmp_path / "criteria.yaml").write_text(yaml.safe_dump(CRITERIA), encoding="utf-8")
    return run


def test_score_a_rendered_set_folder(rendered_set: Path, capsys) -> None:
    criteria = rendered_set.parent / "criteria.yaml"
    assert fhp.main(["score", "--run", str(rendered_set), "--criteria", str(criteria), "--quiet"]) == 0
    result = json.loads((rendered_set / "score.json").read_text(encoding="utf-8"))
    frame, summary = result["rows"][0], result["summary"]
    assert frame["noise_clean"] == {"upper": True, "lower": True} and frame["exposure_gain"] == 1.2
    lum = lambda c: float(srgb_to_linear(np.asarray(c) / 255.0) @ LUMINANCE_709)  # noqa: E731
    want = round(lum(PART) / lum(SHEET), 2)
    assert frame["box_over_upper_sheet"] == [["inox_box", want]]
    assert summary["box_over_upper_sheet"]["inox_box"]["p50"] == want
    assert summary["frames"] == 1 and summary["card_brightness"]["n"] == 1
    text = (rendered_set / "score.txt").read_text(encoding="utf-8")
    assert "Condition check" in text and "inox_box" in text
    assert "inox_box" in capsys.readouterr().out
