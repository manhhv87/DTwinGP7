"""
test_scene_sampler.py
─────────────────────
Unit tests for the anchored/blind scene sampler (C2, paper Eq. (1)).

Verifies: determinism per (seed, index); anchored distributions centred on the
calibrated anchor with width kappa*sigma; blind sampling confined to the blind
ranges; task variables (object pose) inside the pick region under BOTH modes;
C3 condition constraints (class / region / lighting) honoured; spec files +
manifest written correctly.

Run:
    pytest tests/test_scene_sampler.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.cell.pose_utils import make_homogeneous
from src.synthgen import SceneSampler, SynthgenConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The camera anchor used across tests: overhead, looking straight down.
T_BC = make_homogeneous([700.0, 0.0, 1200.0], [180.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def config() -> SynthgenConfig:
    """The SHIPPED config — the test doubles as validation of synthgen.yaml."""
    return SynthgenConfig.from_yaml(PROJECT_ROOT / "config" / "synthgen.yaml")


def make_sampler(config, mode="anchored", kappa=1.0, seed=0, sigma=None):
    return SceneSampler(config, T_BC, mode=mode, kappa=kappa, sigma=sigma, seed=seed)


class TestDeterminism:
    def test_same_seed_same_scene(self, config):
        a = make_sampler(config).sample(5)
        b = make_sampler(config).sample(5)
        assert a == b

    def test_different_index_different_scene(self, config):
        s = make_sampler(config)
        assert s.sample(1) != s.sample(2)


class TestAnchoredSampling:
    def test_camera_centred_on_anchor(self, config):
        s = make_sampler(config, kappa=1.0)
        trans = np.array([
            np.asarray(s.sample(i)["camera"]["T_BC_mm"])[:3, 3] for i in range(200)
        ])
        # Mean within 5 standard errors of the anchor; spread ~ kappa*sigma.
        sigma = np.asarray(config.camera.sigma_trans_mm)
        se = sigma / np.sqrt(len(trans))
        np.testing.assert_allclose(trans.mean(axis=0), T_BC[:3, 3], atol=5 * se.max() + 1e-9)
        assert np.all(trans.std(axis=0) > 0.5 * sigma)
        assert np.all(trans.std(axis=0) < 2.0 * sigma)

    def test_kappa_scales_spread(self, config):
        narrow = make_sampler(config, kappa=1.0)
        wide = make_sampler(config, kappa=4.0, seed=1)
        t_narrow = np.array([
            np.asarray(narrow.sample(i)["camera"]["T_BC_mm"])[:3, 3] for i in range(150)
        ])
        t_wide = np.array([
            np.asarray(wide.sample(i)["camera"]["T_BC_mm"])[:3, 3] for i in range(150)
        ])
        assert np.all(t_wide.std(axis=0) > 2.0 * t_narrow.std(axis=0))

    def test_sigma_json_overrides_yaml(self, config):
        sigma = {"sigma_trans_mm": [9.0, 9.0, 9.0], "sigma_rot_deg": [1.0, 1.0, 1.0],
                 "source": "bootstrap"}
        s = make_sampler(config, kappa=1.0, sigma=sigma)
        assert s.sigma_source == "bootstrap"
        trans = np.array([
            np.asarray(s.sample(i)["camera"]["T_BC_mm"])[:3, 3] for i in range(200)
        ])
        assert np.all(trans.std(axis=0) > 4.5)   # ~9mm sigma, not the 2mm fallback

    def test_light_count_matches_anchor(self, config):
        spec = make_sampler(config).sample(0)
        assert len(spec["lights"]) == len(config.lighting.anchor_positions_mm)


class TestBlindSampling:
    def test_camera_within_blind_bounds(self, config):
        s = make_sampler(config, mode="blind")
        for i in range(100):
            T = np.asarray(s.sample(i)["camera"]["T_BC_mm"])
            d = np.abs(T[:3, 3] - T_BC[:3, 3])
            assert np.all(d <= config.camera.blind_trans_mm + 1e-9)

    def test_kappa_ignored_in_blind(self, config):
        a = make_sampler(config, mode="blind", kappa=1.0).sample(3)
        b = make_sampler(config, mode="blind", kappa=4.0).sample(3)
        assert a["camera"] == b["camera"]
        assert a["kappa"] is None


class TestTaskVariables:
    @pytest.mark.parametrize("mode", ["anchored", "blind"])
    def test_objects_inside_region(self, config, mode):
        s = make_sampler(config, mode=mode)
        x0, x1 = config.objects.region_x_mm
        y0, y1 = config.objects.region_y_mm
        for i in range(50):
            for obj in s.sample(i)["objects"]:
                x, y, z = obj["xyz_mm"]
                assert x0 <= x <= x1 and y0 <= y <= y1
                assert z == config.objects.table_z_mm
                assert -90.0 <= obj["yaw_deg"] <= 90.0

    def test_min_separation(self, config):
        s = make_sampler(config)
        # min_separation may be halved on crowded scenes; never below 1/4 nominal.
        floor = config.objects.min_separation_mm / 4.0
        for i in range(50):
            objs = s.sample(i)["objects"]
            for a in range(len(objs)):
                for b in range(a + 1, len(objs)):
                    d = np.hypot(
                        objs[a]["xyz_mm"][0] - objs[b]["xyz_mm"][0],
                        objs[a]["xyz_mm"][1] - objs[b]["xyz_mm"][1],
                    )
                    assert d >= floor

    def test_class_ids_consistent(self, config):
        s = make_sampler(config)
        for i in range(20):
            for obj in s.sample(i)["objects"]:
                assert obj["class_id"] == config.objects.class_ids[obj["class_name"]]


class TestConditions:
    def test_condition_forces_class_and_region(self, config):
        cond = {"class_name": "carton", "region": [450.0, 500.0, -50.0, 0.0],
                "lighting": None}
        s = make_sampler(config)
        for i in range(30):
            first = s.sample(i, condition=cond)["objects"][0]
            assert first["class_name"] == "carton"
            x, y, _ = first["xyz_mm"]
            assert 450.0 <= x <= 500.0 and -50.0 <= y <= 0.0

    def test_condition_unknown_class_raises(self, config):
        s = make_sampler(config)
        with pytest.raises(ValueError, match="not in objects.classes"):
            s.sample(0, condition={"class_name": "widget", "region": None,
                                   "lighting": None})

    def test_dim_lighting_scales_energy(self, config):
        s = make_sampler(config)
        e_none = np.mean([
            li["energy_w"] for i in range(100) for li in s.sample(i)["lights"]
        ])
        e_dim = np.mean([
            li["energy_w"] for i in range(100)
            for li in s.sample(i, condition={"class_name": None, "region": None,
                                             "lighting": "dim"})["lights"]
        ])
        assert e_dim < 0.6 * e_none


class TestWriteSpecs:
    def test_files_and_manifest(self, config, tmp_path):
        s = make_sampler(config)
        manifest_path = s.write_specs(tmp_path, n=5)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["n"] == 5
        for name in manifest["files"]:
            spec = json.loads((tmp_path / name).read_text(encoding="utf-8"))
            assert spec["mode"] == "anchored"

    def test_multi_batch_merges_manifest(self, config, tmp_path):
        s = make_sampler(config)
        s.write_specs(tmp_path, n=3)
        cond = {"class_name": "metal_box", "region": None, "lighting": "dim"}
        s.write_specs(tmp_path, n=2, condition=cond, start_index=3)
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["n"] == 5
        assert len(manifest["files"]) == 5
        assert manifest["batches"][-1]["condition"] == cond


class TestValidation:
    def test_bad_mode_raises(self, config):
        with pytest.raises(ValueError, match="anchored.*blind"):
            SceneSampler(config, T_BC, mode="wild")

    def test_bad_T_shape_raises(self, config):
        with pytest.raises(ValueError, match="4x4"):
            SceneSampler(config, np.eye(3))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
