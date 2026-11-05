"""
test_hard_profile.py
────────────────────
The render side of the synthetic generator, without Blender: the colour convention,
the sensor model, the hard-condition scene profile and the render manifest.

The renderer itself only runs under `blenderproc run`; these tests cover what it
takes from src/synthgen/sensor_model.py and src/synthgen/hard_profile.py (the
luminaires and the overhead, the session exposure), plus source checks of the rules
that cannot run without Blender: the placement of stacked parts, flat shading, the
reflectance scale and the refusal of specs without an exposure draw.
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.synthgen import hard_profile as hp
from src.synthgen import sensor_model as sm

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "synthgen.yaml"
CLASSES = ["carton", "plastic_box", "wood_box"]     # the dielectric classes: a measured colour
QUIET = {"gain": 1.0, "white_balance_rgb": [1.0, 1.0, 1.0], "shot_noise": 0.0,
         "read_noise": 0.0, "grain_sigma_px": 0.0}


@pytest.fixture(scope="module")
def cfg() -> dict:
    return hp.load_render_config(CONFIG)


@pytest.fixture(scope="module")
def raw() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


class TestColourSpace:
    def test_known_values(self):
        assert sm.srgb_to_linear(0.5) == pytest.approx(0.21404, abs=1e-5)
        assert sm.linear_to_srgb(0.18) == pytest.approx(0.46135, abs=1e-5)
        assert sm.srgb_to_linear([0.0, 1.0]) == pytest.approx([0.0, 1.0])

    def test_round_trip_and_continuity(self):
        c = np.linspace(0.0, 1.0, 1001)
        assert sm.linear_to_srgb(sm.srgb_to_linear(c)) == pytest.approx(c, abs=1e-9)
        # the two branches of the curve meet at the threshold
        assert sm.srgb_to_linear(0.04045 - 1e-9) == pytest.approx(sm.srgb_to_linear(0.04045 + 1e-9), abs=1e-7)

    @pytest.mark.parametrize("cls", CLASSES)
    def test_class_colour_survives_the_render_chain(self, raw, cls):
        """Per class: a matte patch of the class's configured colour, lit so that the
        bare standard table renders at its measured (150, 151, 150), comes out of the
        sensor model at the class's measured 8-bit colour. Every colour of the config
        is measured against that table, so this is the chain the renderer must keep."""
        measured = np.asarray(raw["objects"]["materials"][cls]["albedo_rgb"])
        table = np.asarray(raw["background"]["standard_table_rgb"])
        # Lambertian: radiance = albedo * irradiance / pi. Pick the irradiance at which
        # the table, at its own linearised colour, renders back at the measured value.
        e_over_pi = 1.0
        assert np.abs(sm.apply_sensor(np.broadcast_to(sm.srgb_to_linear(table) * e_over_pi, (2, 2, 3)),
                                      QUIET, None)[0, 0].astype(int) - np.round(table * 255)).max() <= 1
        patch = np.broadcast_to(sm.srgb_to_linear(measured) * e_over_pi, (2, 2, 3))
        out = sm.apply_sensor(patch, QUIET, None)[0, 0]
        assert np.abs(out.astype(int) - np.round(measured * 255).astype(int)).max() <= 1

    def test_metals_render_their_physical_f0_times_the_scale(self, raw):
        """A metal's Base Color is its physical F0 times render.reflectance_scale: the
        config's albedos sit about 0.46 below physical (bare table 0.305 against 0.66),
        so an unscaled F0 would be 2.2 times too strong against the diffuse classes."""
        s = float(raw["render"]["reflectance_scale"])
        assert s == pytest.approx(0.305 / (0.85 / 1.29), abs=0.01)
        mats = raw["objects"]["materials"]
        assert mats["inox_box"]["f0_rgb"][0] * s == pytest.approx(0.262, abs=0.002)
        assert mats["metal_box"]["f0_rgb"][0] * s == pytest.approx(0.156, abs=0.002)
        table = float(sm.srgb_to_linear(raw["background"]["standard_table_rgb"]) @ sm.LUMINANCE_709)
        assert table == pytest.approx(0.305, abs=0.005)

    def test_the_old_bug_would_fail(self, raw):
        # Writing the sRGB value into Base Color as if it were linear: plastic came out
        # at (43, 83, 100) against a measured (8, 38, 57). The chain test catches it.
        measured = np.asarray(raw["objects"]["materials"]["plastic_box"]["albedo_rgb"])
        wrong = sm.apply_sensor(np.broadcast_to(measured, (2, 2, 3)), QUIET, None)[0, 0]
        assert np.abs(wrong.astype(int) - np.round(measured * 255)).max() > 20

    def test_every_profile_colour_is_srgb_in_range(self, cfg):
        prof = cfg["hard_profile"]
        colours = [prof["strip"]["albedo_rgb"], prof["cards"]["albedo_rgb"]] + \
            [prof["surround"][f"{s}_albedo_rgb"] for s in ("left", "right", "near")]
        for c in colours:
            assert len(c) == 3 and all(0.0 <= v <= 1.0 for v in c)


class TestLights:
    def test_reference_lamp_is_neutral(self):
        assert sm.light_rgb(5000.0, 5000.0) == pytest.approx([1.0, 1.0, 1.0])

    def test_unit_luminance_and_tint(self):
        warm, cool = sm.light_rgb(3000.0, 5000.0), sm.light_rgb(8000.0, 5000.0)
        for c in (warm, cool):
            assert float(sm.LUMINANCE_709 @ c) == pytest.approx(1.0)
        assert warm[0] / warm[2] > 1.0 > cool[0] / cool[2]


class TestSensorModel:
    def test_no_rng_is_the_noise_free_expectation(self):
        x = np.full((8, 8, 3), 0.01)
        assert np.all(sm.apply_sensor(x, {**QUIET, "shot_noise": 1e-3}, None)
                      == sm.apply_sensor(x, QUIET, None))

    def test_variance_is_poisson_gaussian(self):
        a, b, level = 2e-4, 1e-6, 0.05
        sensor = {**QUIET, "shot_noise": a, "read_noise": b}
        rng = np.random.default_rng(1)
        y = np.full((400, 400, 3), level, np.float32)
        noise = rng.standard_normal(y.shape).astype(np.float32)
        # same arithmetic as apply_sensor, before the curve
        var = a * y + b
        assert float((np.sqrt(var) * noise).var()) == pytest.approx(a * level + b, rel=0.02)
        # and the 8-bit output is unbiased to within the curve's curvature
        out = sm.apply_sensor(y, sensor, np.random.default_rng(2)).astype(float) / 255.0
        assert float(sm.srgb_to_linear(out).mean()) == pytest.approx(level, rel=0.02)

    def test_gain_scales_signal_and_noise(self):
        sensor = {**QUIET, "shot_noise": 2e-4, "read_noise": 1e-6, "gain": 0.5}
        y = np.full((300, 300, 3), 0.04, np.float32)
        out = sm.srgb_to_linear(sm.apply_sensor(y, sensor, np.random.default_rng(3)) / 255.0)
        assert float(out.mean()) == pytest.approx(0.02, rel=0.03)
        expect = np.sqrt(0.5 * 2e-4 * 0.02 + 0.25 * 1e-6)
        assert float(out.std()) == pytest.approx(expect, rel=0.1)

    def test_grain_has_unit_variance_and_correlation(self):
        rng = np.random.default_rng(4)
        white = sm.unit_grain((300, 300), 0.0, rng)
        grain = sm.unit_grain((300, 300), 0.9, rng)
        assert float(grain.var()) == pytest.approx(1.0, rel=0.1)
        lag1 = lambda g: float((g[:, :-1] * g[:, 1:]).mean() / g.var())   # noqa: E731
        assert abs(lag1(white)) < 0.05 and lag1(grain) > 0.5

    def test_deterministic_per_seed(self):
        sensor = {**QUIET, "shot_noise": 2e-4, "read_noise": 1e-6, "grain_sigma_px": 0.8}
        x = np.full((20, 20, 3), 0.01, np.float32)
        a = sm.apply_sensor(x, sensor, np.random.default_rng([0, 7, 1]))
        b = sm.apply_sensor(x, sensor, np.random.default_rng([0, 7, 1]))
        c = sm.apply_sensor(x, sensor, np.random.default_rng([0, 8, 1]))
        assert np.array_equal(a, b) and not np.array_equal(a, c)

    @pytest.mark.parametrize("bad", [{"gain": 0.0}, {"white_balance_rgb": [1.0, 0.0, 1.0]},
                                     {"shot_noise": -1.0}, {"grain_sigma_px": -0.5}])
    def test_rejects_bad_sensor(self, bad):
        with pytest.raises(ValueError):
            sm.check_sensor({**QUIET, **bad})

    def test_shipped_noise_matches_the_hard_condition(self, cfg):
        """At the brightness of the green sheet, the shipped sensor gives the noise of
        the real camera: 1.6 to 1.9 in the trial frames at luma 15 (their gain was lower
        than in the condition checks the model was fitted on, which gave 2.1 at luma 23),
        far from the 0.39 of the denoised renders before the sensor model."""
        sensor = cfg["sensor"]
        target = sm.srgb_to_linear(np.array([15.0, 15.0, 15.0]) / 255.0) / np.asarray(sensor["white_balance_rgb"])
        img = sm.apply_sensor(np.broadcast_to(target, (200, 200, 3)), sensor, np.random.default_rng(5))
        std = sm.highpass_noise_std(sm.luma601(img))
        assert 1.4 < std < 2.6

    def test_exposure_gain_is_a_gain(self):
        """exposure_gain multiplies sensor.gain: signal and noise alike."""
        sensor = {**QUIET, "shot_noise": 2e-4, "read_noise": 1e-6, "gain": 1.0}
        y = np.full((100, 100, 3), 0.03, np.float32)
        for g in (0.43, 2.9):
            a = sm.apply_sensor(y, sensor, np.random.default_rng(7), exposure_gain=g)
            b = sm.apply_sensor(y, {**sensor, "gain": g}, np.random.default_rng(7))
            assert np.array_equal(a, b)
        with pytest.raises(ValueError, match="exposure_gain"):
            sm.apply_sensor(y, sensor, None, exposure_gain=0.0)

    def test_admitted_gain_range_is_the_configured_one(self, cfg, raw):
        """exposure.gain_range is where both table parts of the profile meet the
        check's luminance bounds; it has to be redone after every profile fit."""
        e = raw["exposure"]
        lo, hi = hp.admitted_gain_range(cfg["sensor"], e["reference_upper_rgb"],
                                        e["reference_lower_rgb"], e["check_luminance"])
        assert lo == pytest.approx(e["gain_range"][0], rel=0.02)
        assert hi == pytest.approx(e["gain_range"][1], rel=0.02)


class TestLuminaires:
    REF = [572.7, 20.0, 593.9]

    def test_a_panel_over_the_whole_sky_gives_pi(self):
        assert hp.panel_irradiance_factor([0, 0, 1000], [1e6, 1e6], [0, 0, 0], n=2000) == \
            pytest.approx(np.pi, rel=1e-3)
        far = hp.panel_irradiance_factor([1950, -50, 2443.9], [600, 600], self.REF)
        d2 = (1950 - 572.7) ** 2 + (-50 - 20) ** 2 + 1850 ** 2
        assert far == pytest.approx(1850 ** 2 / d2 ** 2 * 360000, rel=0.01)   # small-source limit
        with pytest.raises(ValueError):
            hp.panel_irradiance_factor([0, 0, 500], [600, 600], self.REF)

    def test_power_is_pi_area_radiance(self):
        # Checked on the reference render of the profile fit: an albedo-1 card lit to
        # irradiance 1 rendered at 0.9996 / pi.
        assert hp.area_light_power(2.0, [600, 600]) == pytest.approx(np.pi * 2.0 * 0.36)

    def test_lights_give_the_level_and_share(self):
        centres = [[2000.0, 100.0, 2500.0], [300.0, -900.0, 2400.0]]
        lights, env = hp.luminaire_lights(centres, [0.3, 0.7], 0.4, 1.5, [5000.0, 4000.0],
                                          [0.886, 1.0, 0.852], [600.0, 600.0], self.REF)
        e_lamps = sum(li["energy_w"] / (np.pi * 0.36) * hp.panel_irradiance_factor(
            li["position_mm"], li["size_mm"], self.REF) for li in lights)
        e_over = np.pi * float(sm.LUMINANCE_709 @ np.asarray(env["radiance_rgb"]))
        assert e_lamps + e_over == pytest.approx(1.5)
        assert e_over / 1.5 == pytest.approx(0.4) and env["overhead_share"] == 0.4
        rgb = np.asarray(env["radiance_rgb"])
        assert rgb / rgb[1] == pytest.approx([0.886, 1.0, 0.852])
        assert [li["irradiance_share"] for li in lights] == [0.3, 0.7]
        assert all(li["type"] == "area" for li in lights)


class TestProfile:
    def test_render_settings(self, cfg):
        r = cfg["render"]
        assert r["view_transform"] == "Standard"
        assert r["glossy_bounces"] >= 4 and r["max_bounces"] >= r["glossy_bounces"]

    def test_config_errors(self, raw, tmp_path):
        for change in (lambda d: d.pop("hard_profile"),
                       lambda d: d["render"].update(view_transform="Filmic"),
                       lambda d: d["render"].update(max_bounces=2),
                       lambda d: d["render"].update(reflectance_scale=0.0),
                       lambda d: d["render"].pop("reflectance_scale"),
                       lambda d: d["hard_profile"]["cards"].update(albedo_rgb=[1.5, 0.0, 0.0])):
            d = copy.deepcopy(raw)
            change(d)
            p = tmp_path / "bad.yaml"
            p.write_text(yaml.safe_dump(d), encoding="utf-8")
            with pytest.raises(ValueError):
                hp.load_render_config(p)

    def test_cards_are_the_hard_pose_list_cards(self, cfg):
        cards = hp.read_cards(ROOT / "config" / "pose_lists" / "hard_v2.csv.cards.txt")
        assert len(cards) == 21
        ys = sorted({y for _, _, y in cards})
        centres = cfg["hard_profile"]["cards"]["centres_mm"]
        assert sorted(c[1] for c in centres) == ys
        mean_x = float(np.mean([x for _, x, _ in cards]))
        assert all(abs(c[0] - mean_x) < 3.0 for c in centres)
        # each column covers its three cards (x 525, 575, 625)
        half_x = 0.5 * cfg["hard_profile"]["cards"]["size_mm"][0]
        assert all(c[0] - half_x < 525.0 - 20.0 and c[0] + half_x > 625.0 + 20.0 for c in centres)

    def test_elements_partition_the_table(self, cfg):
        els = {e["name"]: e for e in hp.static_elements(cfg, [0.1, 0.2, 0.1], cfg["table_z_mm"])}
        assert len(els) == 3 + 7 + 3
        area = lambda p: 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))  # noqa: E731
        for e in els.values():
            assert area(np.asarray(e["polygon_mm"])) > 0.0          # CCW: normal up
        table = area(hp.table_corners(cfg))
        parts = sum(area(els[n]["polygon_mm"]) for n in ("sheet_upper", "sheet_lower", "strip"))
        assert parts == pytest.approx(table, rel=1e-9)
        # the strip is as wide as configured, and the cards lie on the upper sheet
        assert area(els["strip"]["polygon_mm"]) / 1000.0 == pytest.approx(
            cfg["hard_profile"]["strip"]["width_mm"], rel=0.02)
        upper = np.asarray(els["sheet_upper"]["polygon_mm"])
        assert all(np.asarray(els[f"card_{k:02d}"]["polygon_mm"])[:, 0].max() < upper[:, 0].max()
                   for k in range(1, 8))
        assert all(e["z_mm"] < cfg["table_z_mm"] for n, e in els.items() if n.startswith("surround"))

    def test_lower_sheet_ratio_in_linear_light(self, cfg):
        upper = [0.1, 0.2, 0.1]
        lower = hp.lower_sheet_albedo(upper, cfg)
        ratio = sm.srgb_to_linear(lower) / sm.srgb_to_linear(upper)
        assert ratio == pytest.approx(cfg["hard_profile"]["table"]["lower_sheet_ratio"], rel=1e-6)

    def test_clip_halfplane(self):
        square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        half = hp.clip_halfplane(square, [1.0, 0.0], 1.0)
        assert half[:, 0].max() == pytest.approx(1.0) and len(half) == 4

    def test_shipped_anchor_is_green_and_dim(self, raw):
        """The anchored recipe samples around the hard condition, not the grey table."""
        a = sm.srgb_to_linear(raw["background"]["anchor_albedo_rgb"])
        assert a[1] / a.sum() > 0.40 and a.mean() < 0.05


class TestManifest:
    def make(self, cfg, **over):
        settings = {"samples": 24, "denoiser": "INTEL"}
        kw = dict(device="GPU:OPTIX", blender_version="4.2.1 LTS", blenderproc_version="2.8.0",
                  config_path="config/synthgen.yaml", scenes_dir="specs", frames=["frame_000000"])
        kw.update(over.pop("kw", {}))
        settings.update(over)
        return hp.build_manifest(cfg, settings, **kw)

    def test_hash_ignores_key_order_and_sees_numbers(self, cfg):
        shuffled = json.loads(json.dumps(cfg, sort_keys=True))
        assert hp.profile_sha256(shuffled) == hp.profile_sha256(cfg)
        changed = copy.deepcopy(cfg)
        changed["sensor"]["shot_noise"] = float(changed["sensor"]["shot_noise"]) * 2 + 1e-6
        assert hp.profile_sha256(changed) != hp.profile_sha256(cfg)

    def test_merge_unions_frames_and_refuses_other_settings(self, cfg):
        a = self.make(cfg)
        b = self.make(cfg, kw={"frames": ["frame_000001"]})
        assert hp.merge_manifest(a, b)["frames"] == ["frame_000000", "frame_000001"]
        with pytest.raises(ValueError, match="render"):
            hp.merge_manifest(a, self.make(cfg, samples=64))
        with pytest.raises(ValueError, match="device"):
            hp.merge_manifest(a, self.make(cfg, kw={"device": "CPU"}))

    def test_arms_must_match(self, cfg):
        other = copy.deepcopy(cfg)
        other["hard_profile"]["strip"]["width_mm"] = 20.0
        hp.check_arms_match({"anchored": self.make(cfg), "wide": self.make(cfg)})
        with pytest.raises(ValueError, match="profile_sha256"):
            hp.check_arms_match({"anchored": self.make(cfg), "wide": self.make(other)})

    def test_device_label_names_every_device_used(self):
        """BlenderProc renders a GPU batch on the GPU and the CPU: both names go in."""
        label = hp.render_device("GPU", "OPTIX", ["NVIDIA GeForce RTX 4090", "AMD Ryzen 9 7950X"])
        assert label == "GPU:OPTIX [AMD Ryzen 9 7950X, NVIDIA GeForce RTX 4090]"
        two_cards = ["AMD Ryzen 9 7950X", "NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"]
        assert hp.render_device("GPU", "OPTIX", two_cards) == label
        assert hp.render_device("CPU", "NONE", ["Intel Core i7-8650U CPU @ 1.90GHz"]) == \
            "CPU [Intel Core i7-8650U CPU @ 1.90GHz]"
        assert hp.render_device("GPU", "CUDA", []) == "GPU:CUDA"

    def test_arms_of_one_machine_match_and_a_mix_does_not(self, cfg):
        """Every arm on one Linux machine passes; one arm from another machine with the
        same backend fails, and so does a manifest that recorded the backend alone."""
        linux = hp.render_device("GPU", "OPTIX", ["NVIDIA GeForce RTX 4090", "AMD Ryzen 9 7950X"])
        laptop = hp.render_device("GPU", "OPTIX", ["NVIDIA GeForce MX150",
                                                   "Intel Core i7-8650U CPU @ 1.90GHz"])
        arms = {name: self.make(cfg, kw={"device": linux, "scenes_dir": f"{name}/specs"})
                for name in ("anchored_k1", "anchored_k2", "anchored_k4", "blind")}
        hp.check_arms_match(arms)
        with pytest.raises(ValueError, match="'blind'.*device"):
            hp.check_arms_match({**arms, "blind": self.make(cfg, kw={"device": laptop})})
        with pytest.raises(ValueError, match="device"):
            hp.check_arms_match({"anchored_k1": arms["anchored_k1"],
                                 "old": self.make(cfg, kw={"device": "GPU:OPTIX"})})

    def test_manifest_travels_with_the_dataset(self, cfg, tmp_path):
        import cv2

        from src.synthgen.auto_label import convert_render_output
        render = tmp_path / "render"
        render.mkdir()
        cv2.imwrite(str(render / "frame_000000_rgb.png"), np.zeros((8, 8, 3), np.uint8))
        cv2.imwrite(str(render / "frame_000000_inst.png"), np.zeros((8, 8), np.uint16))
        (render / "frame_000000_meta.json").write_text('{"instances": {}}', encoding="utf-8")
        (render / hp.MANIFEST_NAME).write_text(json.dumps(self.make(cfg)), encoding="utf-8")
        convert_render_output(render, tmp_path / "dataset", val_frac=0.0)
        copied = json.loads((tmp_path / "dataset" / hp.MANIFEST_NAME).read_text(encoding="utf-8"))
        assert copied["profile_sha256"] == hp.profile_sha256(cfg)


def _renderer_calls(name: str) -> list[ast.Call]:
    src = (ROOT / "src" / "synthgen" / "render_blenderproc.py").read_text(encoding="utf-8")
    return [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
            and (getattr(n.func, "id", None) == name or getattr(n.func, "attr", None) == name)]


def _renderer_source() -> str:
    return (ROOT / "src" / "synthgen" / "render_blenderproc.py").read_text(encoding="utf-8")


def _shader_values(socket: str) -> list[str]:
    return [ast.unparse(c.args[1]) for c in _renderer_calls("set_principled_shader_value")
            if isinstance(c.args[0], ast.Constant) and c.args[0].value == socket]


def test_renderer_linearises_every_base_colour():
    """Every Base Color the renderer sets goes through srgb_to_linear, except a metal's
    physical F0, which is linear already and scaled by render.reflectance_scale
    (source check)."""
    calls = _shader_values("Base Color")
    assert len(calls) == 3           # static elements, workpieces, distractors
    direct = [c for c in calls if "srgb_to_linear" in c]
    assert len(direct) == 2 and calls[1] == "base + [1.0]"
    src = _renderer_source()
    assert 'base = [float(v) * reflectance_scale for v in ms["f0_rgb"]]' in src
    assert 'base = [float(v) for v in sm.srgb_to_linear(ms["albedo_rgb"])]' in src


def test_renderer_scales_every_physical_specular_reflectance():
    """Dielectric specular (workpieces and distractors) and the clear coat's weight
    carry render.reflectance_scale; the static elements stay pure Lambertian."""
    levels = _shader_values("Specular IOR Level")
    assert levels.count("DIELECTRIC_SPECULAR_LEVEL * reflectance_scale") == 2
    assert levels.count("0.0") == 1
    assert _shader_values("Coat Weight") == ["reflectance_scale"]


def test_renderer_shades_parts_flat_and_lights_with_luminaires():
    src = _renderer_source()
    assert "poly.use_smooth = False" in src
    assert 'light.set_type("AREA")' in src and 'light.set_type("POINT")' not in src
    assert 'environment["radiance_rgb"]' in src


def test_renderer_applies_the_exposure_and_refuses_specs_without_it():
    src = _renderer_source()
    calls = [ast.unparse(c) for c in _renderer_calls("apply_sensor")]
    assert calls == ["sm.apply_sensor(linear, cfg['sensor'], noise_rng, exposure_gain=exposure_gain)"]
    assert 'spec.get("mode") != "reference" and (exposure is None or environment is None)' in src
    assert "no exposure draw" in src


def test_renderer_records_every_device_it_used():
    """The manifest's device is the backend plus the name of each device used, so that
    arms from two machines with the same backend do not pass check_arms_match."""
    calls = [ast.unparse(c) for c in _renderer_calls("render_device")]
    assert calls == ["hp.render_device(kind, compute, used)"]
    assert "get_devices_for_type(compute) if d.use" in _renderer_source()


def test_renderer_places_every_part_at_its_spec_height():
    """A stacked part rests on the part below it: the sampler writes the support's top
    face as xyz_mm[2]. Dropping every part onto the table, as the renderer once did,
    would sink the top part into its support. Checked on the source, since the
    renderer runs only under Blender."""
    calls = _renderer_calls("place_on_surface")
    assert len(calls) == 2           # workpieces and distractors
    for call in calls:
        assert "xyz_mm" in ast.unparse(call.args[1])
