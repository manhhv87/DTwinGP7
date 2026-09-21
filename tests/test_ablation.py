"""E5 interventions preserve paired nuisance draws and record a reproducible recipe."""
from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.synthgen import SceneSampler, SynthgenConfig

ROOT = Path(__file__).resolve().parent.parent
FACTORS = ("illumination", "background", "distractors", "camera", "pose")
T_BC = np.array([[1., 0., 0., 419.5], [0., -1., 0., 0.],
                 [0., 0., -1., 1200.], [0., 0., 0., 1.]])


@pytest.fixture
def config():
    return SynthgenConfig.from_yaml(ROOT / "config/synthgen.yaml")


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location(
        "generate_synth_cli", ROOT / "scripts/20_generate_synth.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("factor", FACTORS)
def test_only_selected_factor_changes(config, factor):
    """Ablating an early factor must not shift RNG draws for later factors."""
    full = SceneSampler(config, T_BC, kappa=2, seed=17)
    ablated = SceneSampler(config, T_BC, kappa=2, seed=17, ablation=factor)
    changed = False
    for index in range(40):
        before, after = full.sample(index), ablated.sample(index)
        assert after == ablated.sample(index), "same scene must regenerate exactly"
        assert after.pop("ablation")["factor"] == factor
        before.pop("ablation")
        if factor == "pose":
            assert len(before["objects"]) == len(after["objects"])
            for original, fixed in zip(before["objects"], after["objects"]):
                changed |= original["xyz_mm"] != fixed["xyz_mm"]
                for key in ("xyz_mm", "yaw_deg"):
                    original.pop(key)
                    fixed.pop(key)
        else:
            key = "lights" if factor == "illumination" else factor
            changed |= before.pop(key) != after.pop(key)
        assert before == after, "an unrelated factor changed in the paired scene"
    assert changed, "intervention made no change across the sampled scenes"


@pytest.mark.parametrize("mode", ["anchored", "blind"])
def test_default_retains_full_recipe(config, mode):
    implicit = SceneSampler(config, T_BC, mode=mode, seed=17)
    explicit = SceneSampler(config, T_BC, mode=mode, seed=17, ablation="none")
    for index in range(10):
        assert implicit.sample(index) == explicit.sample(index)


def test_fixed_settings_use_supplied_anchors(config):
    camera = SceneSampler(config, T_BC, ablation="camera").sample(0)["camera"]
    assert camera == {"T_BC_mm": T_BC.tolist(),
                      "intrinsics": config.camera.intrinsics}
    lights = SceneSampler(config, T_BC, ablation="illumination").sample(0)["lights"]
    assert lights == [{"type": "point", "position_mm": pos,
                       "energy_w": config.lighting.anchor_intensity_w,
                       "color_temp_k": config.lighting.anchor_color_temp_k}
                      for pos in config.lighting.anchor_positions_mm]
    bg = SceneSampler(config, T_BC, ablation="background").sample(0)["background"]
    assert bg == {"kind": "plane", "z_mm": config.objects.table_z_mm,
                  "albedo_rgb": config.background.anchor_albedo_rgb,
                  "roughness": config.background.anchor_roughness}
    assert SceneSampler(config, T_BC, ablation="distractors").sample(0)["distractors"] == []


@pytest.mark.parametrize("count", [1, 2, 3])
def test_pose_reference_conditional_on_count(config, count):
    # Expected slots follow from the configured envelope, not from copied numbers:
    # the envelope is re-measured whenever the cell is re-calibrated.
    config.objects.count_range = [count, count]
    (x0, x1), (y0, y1) = config.objects.region_x_mm, config.objects.region_y_mm
    assert y1 - y0 > x1 - x0, "this case exercises the long-Y branch"
    z = config.objects.table_z_mm
    expected = [[(x0 + x1) / 2, y0 + (i + 0.5) * (y1 - y0) / count, z] for i in range(count)]
    yaw = sum(config.objects.yaw_range_deg) / 2
    sampler = SceneSampler(config, T_BC, seed=17, ablation="pose")
    for index in range(10):
        spec = sampler.sample(index)
        got = [v for o in spec["objects"] for v in o["xyz_mm"]]
        assert got == pytest.approx([v for xyz in expected for v in xyz])
        assert [o["yaw_deg"] for o in spec["objects"]] == pytest.approx([yaw] * count)
        assert spec["ablation"]["reference"]["axis"] == "y"


def test_pose_long_axis_tie_uses_x_and_yaw_midpoint(config):
    config.objects.count_range = [2, 2]
    config.objects.region_x_mm = [0., 400.]
    config.objects.region_y_mm = [-200., 200.]
    config.objects.yaw_range_deg = [-30., 90.]
    spec = SceneSampler(config, T_BC, ablation="pose").sample(0)
    z = config.objects.table_z_mm
    assert [o["xyz_mm"] for o in spec["objects"]] == [[100., 0., z], [300., 0., z]]
    assert [o["yaw_deg"] for o in spec["objects"]] == [30., 30.]


def test_impossible_pose_spacing_is_rejected(config):
    config.objects.count_range = [1, 10]
    with pytest.raises(ValueError, match="min_separation"):
        SceneSampler(config, T_BC, ablation="pose")


@pytest.mark.parametrize("factor", FACTORS)
def test_ablation_not_allowed_for_blind_or_failure_conditioning(config, tmp_path, factor):
    with pytest.raises(ValueError, match="anchored"):
        SceneSampler(config, T_BC, mode="blind", ablation=factor)
    sampler = SceneSampler(config, T_BC, ablation=factor)
    with pytest.raises(ValueError, match="failure conditioning"):
        sampler.sample(0, condition={"lighting": "dim"})
    destination = tmp_path / "should_not_exist"
    with pytest.raises(ValueError, match="failure conditioning"):
        sampler.write_specs(destination, 1, condition={"class_name": "carton"})
    assert not destination.exists()


def test_unknown_factor_rejected(config):
    with pytest.raises(ValueError, match="ablation must be"):
        SceneSampler(config, T_BC, ablation="failure_conditioning")


def test_manifest_stores_effective_profile_and_independent_scene_metadata(config, tmp_path):
    sigma = {"sigma_trans_mm": [1., 3., 5.], "sigma_rot_deg": [.2, .3, .4],
             "source": "bootstrap"}
    sampler = SceneSampler(config, T_BC, sigma=sigma, ablation="pose", seed=11)
    manifest_path = sampler.write_specs(tmp_path, 2)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["generator_profile"] == {
        "config": config.model_dump(mode="json"), "T_BC_mm": T_BC.tolist(),
        "sigma_trans_mm": sigma["sigma_trans_mm"], "sigma_rot_deg": sigma["sigma_rot_deg"]}
    scene = json.loads((tmp_path / manifest["files"][0]).read_text())
    assert scene["ablation"] == manifest["ablation"]
    pristine = deepcopy(sampler.ablation_protocol)
    sampler.sample(0)["ablation"]["reference"]["poses_by_count"]["1"][0]["yaw_deg"] = 999
    assert sampler.ablation_protocol == pristine


@pytest.mark.parametrize("change", ["factor", "configuration", "seed", "sigma"])
def test_manifest_rejects_mixing_treatments_before_writing(config, tmp_path, change):
    SceneSampler(config, T_BC, ablation="camera").write_specs(tmp_path, 2)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    kwargs = {"ablation": "camera"}
    if change == "factor":
        kwargs["ablation"] = "background"
    elif change == "configuration":
        config.lighting.anchor_intensity_w += 1
    elif change == "seed":
        kwargs["seed"] = 20
    else:
        kwargs["sigma"] = {"sigma_trans_mm": [3., 3., 3.], "sigma_rot_deg": [1., 1., 1.]}
    with pytest.raises(ValueError, match="Existing manifest differs"):
        SceneSampler(config, T_BC, **kwargs).write_specs(tmp_path, 2, start_index=2)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_manifest_rejects_duplicate_indices_and_can_append(config, tmp_path):
    sampler = SceneSampler(config, T_BC, ablation="background")
    sampler.write_specs(tmp_path, 2)
    original = (tmp_path / "scene_000001.json").read_bytes()
    with pytest.raises(ValueError, match="indices already exist"):
        sampler.write_specs(tmp_path, 2, start_index=1)
    assert (tmp_path / "scene_000001.json").read_bytes() == original
    assert not (tmp_path / "scene_000002.json").exists()
    sampler.write_specs(tmp_path, 2, start_index=2)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["start_index"] == 0
    assert manifest["n"] == 4
    assert [b["start_index"] for b in manifest["batches"]] == [0, 2]


@pytest.mark.parametrize("arguments", [
    ["--ablation", "camera", "--mode", "blind"],
    ["--ablation", "pose", "--from-failures", "unused.json"],
    ["--ablation", "background", "--make-labels"],
    ["--ablation", "failure_conditioning"],
    ["--val-frac", "1.0"],
    ["--val-frac", "-0.1"],
])
def test_cli_rejects_confounded_or_invalid_options(cli, arguments):
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["--out", "unused", *arguments])
    assert exc.value.code == 2


@pytest.mark.parametrize("factor", ("none", *FACTORS))
def test_cli_writes_requested_treatment_without_rendering(cli, config, tmp_path, monkeypatch, factor):
    calib = tmp_path / "T_BC.npy"
    np.save(calib, T_BC)
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["20_generate_synth.py", "--out", str(out),
                        "--config", str(ROOT / "config/synthgen.yaml"),
                        "--calib", str(calib), "--sigma", str(tmp_path / "absent.json"),
                        "--ablation", factor, "--n", "2", "--seed", "17"])
    assert cli.main() == 0
    manifest = json.loads((out / "specs/manifest.json").read_text())
    assert manifest["ablation"]["factor"] == factor
    assert len(manifest["files"]) == 2
    assert not (out / "render").exists()
