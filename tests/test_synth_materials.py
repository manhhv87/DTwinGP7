"""
test_synth_materials.py
───────────────────────
Per-class appearance in the synthetic generator (C2, and the "texture" factor
that E5 ablates).

Without it every product renders in the same default grey: the anchored and the
blind arm then differ only in lighting and background, and no detector trained on
that data can tell a brown carton from a blue plastic box.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from src.synthgen import SceneSampler, SynthgenConfig
from src.synthgen.anchored_config import MaterialFactors

ROOT = Path(__file__).resolve().parent.parent
PAPER_CLASSES = ["carton", "plastic_box", "wood_box", "metal_box", "inox_box"]

T_BC = np.array([[1.0, 0.0, 0.0, 700.0],
                 [0.0, -1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0, 1200.0],
                 [0.0, 0.0, 0.0, 1.0]])


@pytest.fixture(scope="module")
def config() -> SynthgenConfig:
    return SynthgenConfig.from_yaml(ROOT / "config" / "synthgen.yaml")


def materials_in(sampler: SceneSampler, n: int = 60) -> list[tuple[str, dict]]:
    """(class, material) of every workpiece over n sampled scenes."""
    out = []
    for i in range(n):
        for o in sampler.sample(i)["objects"]:
            if "material" in o:
                out.append((o["class_name"], o["material"]))
    return out


class TestShippedConfig:
    def test_every_paper_class_has_a_material(self, config):
        assert sorted(config.objects.materials) == sorted(PAPER_CLASSES)

    def test_mirror_class_is_the_most_reflective(self, config):
        mats = config.objects.materials
        assert mats["inox_box"].metallic == 1.0
        assert mats["inox_box"].roughness == min(m.roughness for m in mats.values())

    def test_measured_spread_is_widest_for_the_mirror_class(self, config):
        # The mirror finish reflects its surroundings, so its measured colour varies
        # far more between images than any matte class does.
        mats = config.objects.materials
        widest = max(m.sigma_albedo_rel for m in mats.values())
        assert mats["inox_box"].sigma_albedo_rel == widest
        assert mats["inox_box"].sigma_albedo_rel > 3 * mats["carton"].sigma_albedo_rel

    def test_carton_is_brown_and_plastic_is_blue(self, config):
        r, g, b = config.objects.materials["carton"].albedo_rgb
        assert r > g > b                      # paperboard
        r, g, b = config.objects.materials["plastic_box"].albedo_rgb
        assert b > g > r                      # blue box


class TestAnchoredSampling:
    def test_every_workpiece_carries_a_material(self, config):
        s = SceneSampler(config, T_BC, mode="anchored", kappa=2.0, seed=7)
        for i in range(20):
            for o in s.sample(i)["objects"]:
                assert "material" in o, o["class_name"]

    def test_colour_keeps_its_hue(self, config):
        # What varies between images is brightness, not hue: a carton has to stay
        # brown and a plastic box blue, or the class stops being recognisable.
        # Independent per-channel noise at the measured width fails this.
        s = SceneSampler(config, T_BC, mode="anchored", kappa=2.0, seed=1)
        order = {"carton": (0, 1, 2), "plastic_box": (2, 1, 0), "wood_box": (0, 1, 2)}
        kept = {c: [0, 0] for c in order}
        for cls, mat in materials_in(s, n=140):
            if cls not in order:
                continue
            a = np.asarray(mat["albedo_rgb"])
            hi, mid, lo = order[cls]
            kept[cls][1] += 1
            kept[cls][0] += int(a[hi] >= a[mid] >= a[lo])
        for cls, (ok, total) in kept.items():
            assert total >= 10, f"too few {cls} samples"
            assert ok / total >= 0.9, f"{cls}: hue kept only {ok}/{total}"

    def test_brightness_spread_tracks_the_measurement(self, config):
        s = SceneSampler(config, T_BC, mode="anchored", kappa=1.0, seed=4)
        per: dict[str, list[float]] = {}
        for cls, mat in materials_in(s, n=220):
            per.setdefault(cls, []).append(float(np.mean(mat["albedo_rgb"])))
        rel = {c: float(np.std(v) / np.mean(v)) for c, v in per.items() if len(v) > 20}
        assert rel["inox_box"] > 2 * rel["carton"]

    def test_reflectance_follows_the_class(self, config):
        s = SceneSampler(config, T_BC, mode="anchored", kappa=2.0, seed=2)
        for cls, mat in materials_in(s):
            assert mat["metallic"] == pytest.approx(config.objects.materials[cls].metallic)
            assert 0.02 <= mat["roughness"] <= 1.0

    def test_kappa_widens_the_colour_spread(self, config):
        def spread(k):
            s = SceneSampler(config, T_BC, mode="anchored", kappa=k, seed=3)
            per = {}
            for cls, mat in materials_in(s, n=80):
                per.setdefault(cls, []).append(mat["albedo_rgb"])
            return float(np.mean([np.std(np.asarray(v)) for v in per.values()]))
        assert spread(4.0) > spread(1.0)


class TestBlindSampling:
    def test_colour_ignores_the_measurement(self, config):
        s = SceneSampler(config, T_BC, mode="blind", seed=5)
        vals = np.asarray([m["albedo_rgb"] for _, m in materials_in(s, n=80)])
        assert vals.min() < 0.2 and vals.max() > 0.8      # the wide hand-set range
        assert 0.35 < float(vals.mean()) < 0.65           # centred on nothing measured

    def test_reflectance_is_random_too(self, config):
        s = SceneSampler(config, T_BC, mode="blind", seed=6)
        mats = materials_in(s, n=60)
        assert len({round(m["metallic"], 3) for _, m in mats}) > 5


class TestBackgroundColour:
    def test_table_stays_neutral_grey(self, config):
        # The photographed table is neutral (150, 151, 150). An anchored table that
        # comes out pink or green defeats the point of anchoring.
        s = SceneSampler(config, T_BC, mode="anchored", kappa=2.0, seed=8)
        worst = 0.0
        for i in range(60):
            a = np.asarray(s.sample(i)["background"]["albedo_rgb"])
            worst = max(worst, float((a.max() - a.min()) / max(a.mean(), 1e-6)))
        assert worst < 0.25, f"table hue drifted by {worst:.2f} of its brightness"

    def test_table_brightness_tracks_the_measurement(self, config):
        s = SceneSampler(config, T_BC, mode="anchored", kappa=1.0, seed=9)
        vals = [float(np.mean(s.sample(i)["background"]["albedo_rgb"])) for i in range(120)]
        anchor = float(np.mean(config.background.anchor_albedo_rgb))
        assert abs(float(np.mean(vals)) - anchor) < 0.03
        assert float(np.std(vals)) < 3 * config.background.sigma_albedo

    def test_blind_table_is_any_colour(self, config):
        s = SceneSampler(config, T_BC, mode="blind", seed=10)
        arr = np.asarray([s.sample(i)["background"]["albedo_rgb"] for i in range(60)])
        assert arr.min() < 0.2 and arr.max() > 0.8
        spread = float(np.mean((arr.max(axis=1) - arr.min(axis=1)) / arr.mean(axis=1)))
        assert spread > 0.3          # hue all over the place, by design


class TestMissingMaterial:
    def test_class_without_material_gets_no_material_key(self, config):
        cfg = config.model_copy(deep=True)
        cfg.objects.materials = {k: v for k, v in cfg.objects.materials.items()
                                 if k != "wood_box"}
        s = SceneSampler(cfg, T_BC, mode="anchored", seed=11)
        seen = {o["class_name"]: ("material" in o) for i in range(40)
                for o in s.sample(i)["objects"]}
        assert seen.get("wood_box") is False
        assert all(v for k, v in seen.items() if k != "wood_box")


class TestValidation:
    @pytest.mark.parametrize("bad", [
        {"albedo_rgb": [0.4, 0.3]},
        {"albedo_rgb": [1.4, 0.3, 0.2]},
        {"albedo_rgb": [0.4, 0.3, 0.2], "roughness": 1.5},
        {"albedo_rgb": [0.4, 0.3, 0.2], "metallic": -0.1},
        {"albedo_rgb": [0.4, 0.3, 0.2], "sigma_albedo_rel": -0.05},
    ])
    def test_bad_material_rejected(self, bad):
        with pytest.raises(ValidationError):
            MaterialFactors(**bad)

    def test_material_for_unknown_class_rejected(self, config):
        raw = config.model_dump()
        raw["objects"]["materials"]["khong_co_lop_nay"] = {"albedo_rgb": [0.5, 0.5, 0.5]}
        with pytest.raises(ValidationError):
            SynthgenConfig.model_validate(raw)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
