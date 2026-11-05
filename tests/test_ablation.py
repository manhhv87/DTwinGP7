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
from src.synthgen.hard_profile import luminaire_lights
from src.synthgen.scene_sampler import footprint, footprints_overlap

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


def _footprint(config, obj):
    length, width, _ = config.objects.mesh_sizes_mm[obj["mesh"]]
    return footprint(obj["xyz_mm"][:2], length, width, obj["yaw_deg"])


def _square(d):
    return footprint(d["xyz_mm"][:2], d["size_mm"], d["size_mm"], d["yaw_deg"])


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
            # Distractors the slot poses would cut through move (test below); every
            # one keeps its shape, size, yaw and colour, in order.
            looks = [{k: v for k, v in d.items() if k != "xyz_mm"}
                     for d in before.pop("distractors")]
            moved = [{k: v for k, v in d.items() if k != "xyz_mm"}
                     for d in after.pop("distractors")]
            rest = iter(looks)
            assert all(any(m == d for d in rest) for m in moved)
        elif factor == "illumination":
            # The measured light factors move to the profile; the free ones (where the
            # luminaires hang, how they share the light, the overhead share and the
            # exposure) stay as drawn.
            lb, la = before.pop("lights"), after.pop("lights")
            eb, ea = before.pop("environment"), after.pop("environment")
            assert [(li["position_mm"], li["irradiance_share"]) for li in lb] == \
                [(li["position_mm"], li["irradiance_share"]) for li in la]
            assert eb["overhead_share"] == ea["overhead_share"]
            changed |= lb != la or eb != ea
        else:
            changed |= before.pop(factor) != after.pop(factor)
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
    lit = config.lighting
    full = SceneSampler(config, T_BC).sample(0)
    fixed = SceneSampler(config, T_BC, ablation="illumination").sample(0)
    lights, environment = luminaire_lights(
        [li["position_mm"] for li in full["lights"]], [li["irradiance_share"] for li in full["lights"]],
        full["environment"]["overhead_share"], lit.anchor_irradiance,
        [lit.anchor_color_temp_k] * len(full["lights"]), lit.overhead_tint_rgb, lit.luminaire_size_mm,
        [*lit.reference_point_mm, config.objects.table_z_mm])
    assert fixed["lights"] == lights and fixed["environment"] == environment
    assert fixed["exposure"] == full["exposure"]
    bg = SceneSampler(config, T_BC, ablation="background").sample(0)["background"]
    assert bg == {"kind": "plane", "z_mm": config.objects.table_z_mm,
                  "albedo_rgb": config.background.anchor_albedo_rgb,
                  "roughness": config.background.anchor_roughness}
    assert SceneSampler(config, T_BC, ablation="distractors").sample(0)["distractors"] == []


def _on_table(spec):
    return [o for o in spec["objects"] if "stack_on_index" not in o]


def _check_stacked_follow_support(config, spec):
    for obj in spec["objects"]:
        if "stack_on_index" in obj:
            support = spec["objects"][obj["stack_on_index"]]
            x, y, z = support["xyz_mm"]
            height = config.objects.mesh_sizes_mm[support["mesh"]][2]
            assert obj["xyz_mm"] == pytest.approx([x, y, z + height])
            assert obj["yaw_deg"] == support["yaw_deg"]


@pytest.mark.parametrize("count", [1, 2])
def test_pose_reference_conditional_on_count(config, count):
    # Expected slots follow from the configured envelope, not from copied numbers:
    # the envelope is re-measured whenever the cell is re-calibrated. Slots hold the
    # placements on the table; a stacked product sits on its support's slot.
    config.objects.count_range = [count, count]
    (x0, x1), (y0, y1) = config.objects.region_x_mm, config.objects.region_y_mm
    assert y1 - y0 > x1 - x0, "this case exercises the long-Y branch"
    z = config.objects.table_z_mm
    expected = [[(x0 + x1) / 2, y0 + (i + 0.5) * (y1 - y0) / count, z] for i in range(count)]
    yaw = sum(config.objects.yaw_range_deg) / 2
    sampler = SceneSampler(config, T_BC, seed=17, ablation="pose")
    stacked = 0
    for index in range(30):
        spec = sampler.sample(index)
        on_table = _on_table(spec)
        got = [v for o in on_table for v in o["xyz_mm"]]
        assert got == pytest.approx([v for xyz in expected[:len(on_table)] for v in xyz])
        assert [o["yaw_deg"] for o in on_table] == pytest.approx([yaw] * len(on_table))
        _check_stacked_follow_support(config, spec)
        stacked += len(spec["objects"]) > len(on_table)
        assert spec["ablation"]["reference"]["axis"] == "y"
    assert stacked > 0


def test_pose_long_axis_tie_uses_x_and_yaw_midpoint(config):
    config.objects.count_range = [2, 2]
    config.objects.region_x_mm = [0., 600.]
    config.objects.region_y_mm = [-300., 300.]
    config.objects.yaw_range_deg = [-30., 90.]
    sampler = SceneSampler(config, T_BC, ablation="pose")
    z = config.objects.table_z_mm
    for index in range(10):
        spec = sampler.sample(index)
        on_table = _on_table(spec)
        assert [o["xyz_mm"] for o in on_table] == [[150., 0., z], [450., 0., z]][:len(on_table)]
        assert [o["yaw_deg"] for o in on_table] == [30.] * len(on_table)
        _check_stacked_follow_support(config, spec)


@pytest.mark.parametrize("counts", [[1, 3], [1, 10]])
def test_impossible_pose_spacing_is_rejected(config, counts):
    # Three slots of 130 mm cannot hold the 190 mm wood box without overlap.
    config.objects.count_range = counts
    with pytest.raises(ValueError, match="slots overlap"):
        SceneSampler(config, T_BC, ablation="pose")


def test_pose_arm_moves_only_the_distractors_in_the_way(config):
    full = SceneSampler(config, T_BC, kappa=2, seed=17)
    ablated = SceneSampler(config, T_BC, kappa=2, seed=17, ablation="pose")
    moved = 0
    for index in range(60):
        before, after = full.sample(index), ablated.sample(index)
        products = [_footprint(config, o) for o in after["objects"]]
        for i in range(len(products)):
            for j in range(i + 1, len(products)):
                a, b = after["objects"][i], after["objects"][j]
                if a.get("stack_on_index") == j or b.get("stack_on_index") == i:
                    continue
                assert not footprints_overlap(products[i], products[j])
        squares = [_square(d) for d in after["distractors"]]
        for k, sq in enumerate(squares):
            assert not any(footprints_overlap(sq, p) for p in products)
            assert not any(footprints_overlap(sq, other) for other in squares[k + 1:])
        # A distractor clear of the slot poses stays exactly where it was.
        for d in before["distractors"]:
            if not any(footprints_overlap(_square(d), p) for p in products):
                assert d in after["distractors"]
        moved += sum(d not in before["distractors"] for d in after["distractors"])
    assert moved > 0


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
        "sigma_trans_mm": sigma["sigma_trans_mm"], "sigma_rot_deg": sigma["sigma_rot_deg"],
        "top_textures": sampler.top_textures}
    # The texture list (files with their SHA-256) is part of the profile, hence of its digest.
    assert sampler.top_textures and all(e["sha256"] for es in sampler.top_textures.values() for e in es)
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
        config.lighting.anchor_irradiance += 1
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


@pytest.mark.parametrize("arguments", [
    [], ["--ablation", "camera"], ["--from-failures", "f.json"], ["--control-for", "f.json"],
    ["--from-failures", "f.json", "--mode", "blind"],
])
def test_cli_requires_kappa_for_anchored_scenes(cli, arguments, capsys):
    """No default: a forgotten --kappa in E4 or E5 would otherwise sample at 2."""
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["--out", "unused", *arguments])
    assert exc.value.code == 2 and "--kappa is required" in capsys.readouterr().err
    assert cli.parse_args(["--out", "unused", "--kappa", "4", *arguments]).kappa == 4.0


def test_cli_blind_and_labels_need_no_kappa_and_no_val_split(cli):
    assert cli.parse_args(["--out", "unused", "--mode", "blind"]).kappa is None
    args = cli.parse_args(["--out", "unused", "--make-labels"])
    assert args.kappa is None and args.val_frac == 0.0


@pytest.mark.parametrize("factor", ("none", *FACTORS))
def test_cli_writes_requested_treatment_without_rendering(cli, config, tmp_path, monkeypatch, factor):
    calib = tmp_path / "T_BC.npy"
    np.save(calib, T_BC)
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["20_generate_synth.py", "--out", str(out),
                        "--config", str(ROOT / "config/synthgen.yaml"),
                        "--calib", str(calib), "--sigma", str(tmp_path / "absent.json"),
                        "--allow-sigma-fallback",
                        "--ablation", factor, "--kappa", "2", "--n", "2", "--seed", "17"])
    assert cli.main() == 0
    manifest = json.loads((out / "specs/manifest.json").read_text())
    assert manifest["ablation"]["factor"] == factor
    assert manifest["sigma_source"] == "yaml_fallback"
    assert len(manifest["files"]) == 2
    assert not (out / "render").exists()


@pytest.mark.parametrize("extra", [[], ["--ablation", "camera"], ["--from-failures", "f.json"]])
def test_cli_stops_without_bootstrap_sigma(cli, tmp_path, monkeypatch, extra):
    # The fallback widths are configured, not measured: a paper run must not slip
    # onto them because a file is missing.
    calib = tmp_path / "T_BC.npy"
    np.save(calib, T_BC)
    report = tmp_path / "f.json"
    report.write_text(json.dumps({"allocation": [
        {"key": "k", "n_images": 2, "condition": {"class_name": "carton"}}]}))
    extra = [str(report) if a == "f.json" else a for a in extra]
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["20_generate_synth.py", "--out", str(out),
                        "--config", str(ROOT / "config/synthgen.yaml"),
                        "--calib", str(calib), "--sigma", str(tmp_path / "absent.json"),
                        "--kappa", "2", "--n", "2", *extra])
    assert cli.main() == 1
    assert not out.exists()


def test_cli_blind_needs_no_sigma(cli, tmp_path, monkeypatch):
    calib = tmp_path / "T_BC.npy"
    np.save(calib, T_BC)
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["20_generate_synth.py", "--out", str(out),
                        "--config", str(ROOT / "config/synthgen.yaml"),
                        "--calib", str(calib), "--sigma", str(tmp_path / "absent.json"),
                        "--mode", "blind", "--n", "2"])
    assert cli.main() == 0
    assert len(json.loads((out / "specs/manifest.json").read_text())["files"]) == 2
