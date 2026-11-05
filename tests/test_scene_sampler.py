"""
test_scene_sampler.py
─────────────────────
Unit tests for the anchored/blind scene sampler (C2, paper Eq. (1)).

Verifies: determinism per (seed, index), and no shared scenes between seeds;
anchored distributions centred on the calibrated anchor with width kappa*sigma,
and positive quantities (albedo, light power, light height) kept positive at
every kappa; blind sampling confined to the blind ranges, with light power a
multiple of the anchor; task variables inside the pick region under BOTH modes,
with no overlapping footprints and distractors off the products; the first
product's class and stacking as in the physical pose lists; C3 condition
constraints (class / region / lighting) honoured; spec files + manifest written
correctly.

Run:
    pytest tests/test_scene_sampler.py -v
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from src.cell.pose_utils import make_homogeneous
from src.synthgen import LIGHTING_CONDITION_SCALE, SceneSampler, SynthgenConfig
from src.synthgen.scene_sampler import SPEC_VERSION, footprint, footprints_overlap
from src.synthgen.seed_ranges import scene_streams

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The camera anchor used across tests: overhead, looking straight down.
T_BC = make_homogeneous([700.0, 0.0, 1200.0], [180.0, 0.0, 0.0])
N_SCENES = 500


@pytest.fixture(scope="module")
def config() -> SynthgenConfig:
    """The SHIPPED config — the test doubles as validation of synthgen.yaml."""
    return SynthgenConfig.from_yaml(PROJECT_ROOT / "config" / "synthgen.yaml")


def make_sampler(config, mode="anchored", kappa=1.0, seed=0, sigma=None):
    return SceneSampler(config, T_BC, mode=mode, kappa=kappa, sigma=sigma, seed=seed)


@pytest.fixture(scope="module")
def scenes(config) -> dict[str, list[dict]]:
    """N_SCENES scenes per recipe, shared by the distribution tests."""
    recipes = {"k1": {"kappa": 1.0}, "k4": {"kappa": 4.0}, "blind": {"mode": "blind"}}
    return {name: [make_sampler(config, **kw).sample(i) for i in range(N_SCENES)]
            for name, kw in recipes.items()}


def footprint_of(config, obj) -> np.ndarray:
    length, width, _ = config.objects.mesh_sizes_mm[obj["mesh"]]
    return footprint(obj["xyz_mm"][:2], length, width, obj["yaw_deg"])


def irradiance_ratio(config, lights) -> float:
    """Irradiance at the centre of the placement envelope, relative to the anchored
    reference lights (point sources, cosine law)."""
    o = config.objects
    centre = np.array([np.mean(o.region_x_mm), np.mean(o.region_y_mm), o.table_z_mm])

    def irradiance(ls):
        e = 0.0
        for li in ls:
            d = np.asarray(li["position_mm"], dtype=float) - centre
            r = float(np.linalg.norm(d))
            e += max(0.0, d[2] / r) * li["energy_w"] / (4.0 * np.pi * r * r)
        return e

    lit = config.lighting
    ref = [{"position_mm": p, "energy_w": lit.anchor_intensity_w}
           for p in lit.anchor_positions_mm]
    return irradiance(lights) / irradiance(ref)


def pose_list_rows(name: str) -> list[dict]:
    with (PROJECT_ROOT / "config" / "pose_lists" / name).open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if int(r["pose_id"][1:]) <= 200]


class TestDeterminism:
    def test_same_seed_same_scene(self, config):
        a = make_sampler(config).sample(5)
        b = make_sampler(config).sample(5)
        assert a == b

    def test_different_index_different_scene(self, config):
        s = make_sampler(config)
        assert s.sample(1) != s.sample(2)

    def test_seed_and_index_do_not_trade_off(self, config):
        # With seed + index, seed 1 / index 16 was seed 0 / index 17, scene for scene.
        a = make_sampler(config, seed=0).sample(17)
        b = make_sampler(config, seed=1).sample(16)
        assert a["camera"] != b["camera"]
        assert a["objects"] != b["objects"]

    @pytest.mark.parametrize("other", [1, 17, 2000])
    def test_batches_with_different_seeds_share_no_scene(self, config, other):
        def cameras(seed):
            s = make_sampler(config, seed=seed)
            return {json.dumps(s.sample(i)["camera"]["T_BC_mm"]) for i in range(150)}
        assert not cameras(0) & cameras(other)

    def test_kappa_arms_share_the_task_variables(self, config):
        # The kappa arms differ in the nuisance widths only: layouts and distractors
        # are the same draws, so the arms compare like with like.
        narrow, wide = make_sampler(config, kappa=1.0), make_sampler(config, kappa=4.0)
        keys = ("class_name", "mesh", "xyz_mm", "yaw_deg", "stack_on_index")
        for i in range(60):
            a, b = narrow.sample(i), wide.sample(i)
            assert [[o.get(k) for k in keys] for o in a["objects"]] == \
                   [[o.get(k) for k in keys] for o in b["objects"]]
            assert a["distractors"] == b["distractors"]

    @pytest.mark.parametrize("value", [-1, 2 ** 32])
    def test_seed_or_index_outside_one_word_rejected(self, config, value):
        # numpy splits a value of 2**32 or more into two words, and the scene's
        # stream could then be another seed's.
        with pytest.raises(ValueError, match="seed"):
            make_sampler(config, seed=value)
        with pytest.raises(ValueError, match="index"):
            make_sampler(config).sample(value)

    def test_scene_stream_is_the_one_the_overlap_checker_names(self, config, monkeypatch):
        # seed_ranges names a scene's stream by its spec version's rule; seeding
        # it otherwise needs a SPEC_VERSION bump and the new rule there.
        seen = []
        default_rng = np.random.default_rng
        monkeypatch.setattr(np.random, "default_rng", lambda s: seen.append(s) or default_rng(s))
        make_sampler(config, seed=7).sample(16)
        [(first, second)] = scene_streams(SPEC_VERSION, 7, [16])
        named = np.random.SeedSequence([first[0], second[0]])
        assert np.array_equal(seen[0].generate_state(4), named.generate_state(4))


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

    def test_positive_factors_stay_positive_at_kappa_4(self, config, scenes):
        # max(0, 1 + eta) rendered 32% of inox boxes black at kappa 4, left a light
        # at the 1 W floor in 23% of scenes and put 2% of lights under the table.
        z_table = config.objects.table_z_mm
        specs = scenes["k4"]
        albedo = np.array([o["material"]["albedo_rgb"] for s in specs for o in s["objects"]])
        assert albedo.min() > 0.0
        assert min(min(s["background"]["albedo_rgb"]) for s in specs) > 0.0
        lights = [li for s in specs for li in s["lights"]]
        assert min(li["position_mm"][2] for li in lights) > z_table
        energy = np.array([li["energy_w"] for li in lights])
        assert energy.min() > 0.0
        assert np.unique(energy).size == energy.size        # no floor value repeats

    def test_light_height_spread_near_the_anchor(self, config, scenes):
        # At kappa 1 the log-scale height moves like the configured Gaussian.
        z = np.array([li["position_mm"][2] for s in scenes["k1"] for li in s["lights"]])
        z_anchor = config.lighting.anchor_positions_mm[0][2]
        sd = config.lighting.sigma_position_mm
        assert abs(float(np.median(z)) - z_anchor) < 0.15 * sd
        assert 0.85 * sd < float(z.std()) < 1.15 * sd

    def test_median_colour_stays_at_the_measurement(self, config):
        # exp(eta) has median 1: the brightest draws clip at 1, but the median colour
        # stays the measured one, with no draw at zero.
        s = make_sampler(config, kappa=2.0)
        cond = {"class_name": "inox_box", "region": None, "lighting": None}
        anchor = np.asarray(config.objects.materials["inox_box"].albedo_rgb)
        factors = np.array([s.sample(i, condition=cond)["objects"][0]["material"]["albedo_rgb"]
                            for i in range(300)]) / anchor
        assert factors.min() > 0.0
        assert np.all(np.abs(np.log(np.median(factors, axis=0))) < 0.3)


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

    def test_light_power_is_a_multiple_of_the_anchor(self, config, scenes):
        lit = config.lighting
        lo, hi = lit.blind_intensity_rel_range
        rel = np.array([li["energy_w"] for s in scenes["blind"] for li in s["lights"]])
        rel /= lit.anchor_intensity_w
        assert rel.min() >= lo - 1e-12 and rel.max() <= hi + 1e-12
        # Log-uniform: the log of the factor is uniform, so its median sits midway.
        assert abs(float(np.median(np.log(rel))) - np.log(lo * hi) / 2) < 0.25

    def test_refitted_anchor_moves_the_wide_range_power(self, config):
        refit = config.model_copy(deep=True)
        refit.lighting.anchor_intensity_w *= 3.0
        a = make_sampler(config, mode="blind").sample(7)["lights"]
        b = make_sampler(refit, mode="blind").sample(7)["lights"]
        assert [li["energy_w"] * 3.0 for li in a] == pytest.approx([li["energy_w"] for li in b])

    def test_light_level_centred_on_the_anchor_and_wider(self, config, scenes):
        # The 10-200 W range, set for a 60 W anchor, lit the pick region at 1.6x,
        # 11.5x and 38.8x the 19 W anchor (P5/P50/P95): brighter, not wider.
        blind = np.array([irradiance_ratio(config, s["lights"]) for s in scenes["blind"]])
        k4 = np.array([irradiance_ratio(config, s["lights"]) for s in scenes["k4"]])
        p5, p50, p95 = np.percentile(blind, [5, 50, 95])
        assert 0.7 < p50 < 1.4
        assert p5 < np.percentile(k4, 5) and p95 > np.percentile(k4, 95)
        assert (blind <= LIGHTING_CONDITION_SCALE["dim"]).mean() > 0.02


class TestTaskVariables:
    @pytest.mark.parametrize("recipe", ["k1", "blind"])
    def test_objects_inside_region(self, config, scenes, recipe):
        o = config.objects
        (x0, x1), (y0, y1) = o.region_x_mm, o.region_y_mm
        for spec in scenes[recipe]:
            for obj in spec["objects"]:
                assert o.yaw_range_deg[0] <= obj["yaw_deg"] <= o.yaw_range_deg[1]
                if "stack_on_index" in obj:
                    continue                      # TestPoseListMix checks the stack
                x, y, z = obj["xyz_mm"]
                assert x0 <= x <= x1 and y0 <= y <= y1
                assert z == o.table_z_mm

    @pytest.mark.parametrize("recipe", ["k1", "k4", "blind"])
    def test_footprints_never_overlap(self, config, scenes, recipe):
        # The old 120 mm centre distance let 48% of scenes interpenetrate.
        pairs = 0
        for spec in scenes[recipe]:
            objs = spec["objects"]
            fps = [footprint_of(config, ob) for ob in objs]
            for i in range(len(objs)):
                for j in range(i + 1, len(objs)):
                    if objs[i].get("stack_on_index") == j or objs[j].get("stack_on_index") == i:
                        continue                  # a product resting on its support
                    pairs += 1
                    assert not footprints_overlap(fps[i], fps[j]), (spec["index"], i, j)
        assert pairs > 100

    @pytest.mark.parametrize("recipe", ["k1", "blind"])
    def test_distractors_clear_of_products_and_each_other(self, config, scenes, recipe):
        # The old sampler put a distractor inside a product in 53% of scenes.
        n = 0
        for spec in scenes[recipe]:
            taken = [footprint_of(config, ob) for ob in spec["objects"]]
            for d in spec["distractors"]:
                sq = footprint(d["xyz_mm"][:2], d["size_mm"], d["size_mm"], d["yaw_deg"])
                assert not any(footprints_overlap(sq, t) for t in taken), spec["index"]
                taken.append(sq)
                n += 1
        assert n > 300

    def test_counts_follow_the_ranges(self, config, scenes):
        # Placements on the table: count_range; products: one more for a stack.
        # Rejection sampling rarely gives up a second placement or a distractor.
        lo, hi = config.objects.count_range
        placements = Counter(sum("stack_on_index" not in ob for ob in s["objects"])
                             for s in scenes["k1"])
        assert set(placements) <= set(range(lo, hi + 1))
        assert placements[hi] / N_SCENES > 0.4
        d_lo, d_hi = config.distractors.count_range
        mean_d = np.mean([len(s["distractors"]) for s in scenes["k1"]])
        assert 0.9 * (d_lo + d_hi) / 2 < mean_d <= d_hi

    def test_first_placement_is_uniform(self, config, scenes):
        # Drawn before anything else, the first placement sees no rejection.
        o = config.objects
        first = [s["objects"][1] if "stack_on_index" in s["objects"][0] else s["objects"][0]
                 for s in scenes["k1"]]
        x = np.array([ob["xyz_mm"][0] for ob in first])
        y = np.array([ob["xyz_mm"][1] for ob in first])
        for v, (a, b) in ((x, o.region_x_mm), (y, o.region_y_mm)):
            se = (b - a) / np.sqrt(12 * len(v))
            assert abs(float(v.mean()) - (a + b) / 2) < 4 * se
            assert abs(float(v.std()) - (b - a) / np.sqrt(12)) < 0.1 * (b - a) / np.sqrt(12)

    def test_class_ids_consistent(self, config):
        s = make_sampler(config)
        for i in range(20):
            for obj in s.sample(i)["objects"]:
                assert obj["class_id"] == config.objects.class_ids[obj["class_name"]]


class TestFootprintOverlap:
    def test_rectangles(self):
        a = footprint((0.0, 0.0), 100.0, 40.0, 0.0)
        assert footprints_overlap(a, footprint((50.0, 0.0), 100.0, 40.0, 0.0))
        # Touching along an edge is not overlapping.
        assert not footprints_overlap(a, footprint((100.0, 0.0), 100.0, 40.0, 0.0))
        assert not footprints_overlap(a, footprint((0.0, 45.0), 100.0, 40.0, 0.0))
        # A cross: no corner of either lies inside the other, yet they overlap.
        assert footprints_overlap(a, footprint((0.0, 0.0), 100.0, 40.0, 90.0))
        # Turned by 45 deg, a square's corner reaches 70.7 mm from its centre.
        sq = footprint((0.0, 0.0), 100.0, 100.0, 45.0)
        assert footprints_overlap(sq, footprint((0.0, 115.0), 100.0, 100.0, 0.0))
        assert not footprints_overlap(sq, footprint((0.0, 125.0), 100.0, 100.0, 0.0))

    def test_yaw_turns_the_mesh_length(self):
        fp = footprint((10.0, 20.0), 180.0, 85.0, 90.0)
        assert np.ptp(fp[:, 0]) == pytest.approx(85.0)
        assert np.ptp(fp[:, 1]) == pytest.approx(180.0)
        assert fp.mean(axis=0) == pytest.approx([10.0, 20.0])


class TestPoseListMix:
    def test_mix_matches_the_pose_lists(self, config):
        # Recounted from the files: the configuration must not drift from the
        # pose lists the physical trials run.
        counts = []
        for name in ("hard_v2.csv", "std_v2.csv"):
            rows = pose_list_rows(name)
            assert len(rows) == 200
            counts.append(Counter((r["class_hint"], r["stack_on"] or None) for r in rows))
        assert counts[0] == counts[1]
        mix = config.objects.first_object_mix
        configured = Counter()
        for cls, m in mix.items():
            configured[(cls, None)] += m.single
            for support, k in m.stacked_on.items():
                configured[(cls, support)] += k
        assert +configured == +counts[0]

    def test_mesh_sizes_match_the_stl_files(self, config):
        record = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        for mesh, size in config.objects.mesh_sizes_mm.items():
            data = (PROJECT_ROOT / mesh).read_bytes()
            n = int(np.frombuffer(data, "<u4", count=1, offset=80)[0])
            v = np.frombuffer(data, record, count=n, offset=84)["v"].reshape(-1, 3)
            assert np.ptp(v, axis=0) == pytest.approx(size, abs=0.01), mesh
            assert v[:, 2].min() == pytest.approx(0.0, abs=0.01), mesh

    @pytest.mark.parametrize("recipe", ["k1", "blind"])
    def test_first_class_and_stacking_shares(self, config, scenes, recipe):
        # Pose lists, poses 1-200: inox 100, the other classes 25 each; 40 stacked.
        specs = scenes[recipe]
        share = Counter(s["objects"][0]["class_name"] for s in specs)
        assert abs(share["inox_box"] / len(specs) - 0.5) < 0.08
        for cls in ("carton", "plastic_box", "wood_box", "metal_box"):
            assert abs(share[cls] / len(specs) - 0.125) < 0.06
        stacked = [s for s in specs if "stack_on_index" in s["objects"][0]]
        assert abs(len(stacked) / len(specs) - 0.2) < 0.07
        mix = config.objects.first_object_mix
        allowed = {(c, s) for c, m in mix.items() for s, k in m.stacked_on.items() if k}
        seen = {(s["objects"][0]["class_name"], s["objects"][1]["class_name"]) for s in stacked}
        assert seen <= allowed and len(seen) >= len(allowed) - 2
        # Only the first product may rest on another.
        assert all("stack_on_index" not in ob for s in specs for ob in s["objects"][1:])

    @pytest.mark.parametrize("recipe", ["k1", "blind"])
    def test_stacked_product_rests_on_its_support(self, config, scenes, recipe):
        o = config.objects
        n = 0
        for spec in scenes[recipe]:
            top = spec["objects"][0]
            if "stack_on_index" not in top:
                continue
            support = spec["objects"][top["stack_on_index"]]
            assert support["xyz_mm"][2] == o.table_z_mm
            assert top["xyz_mm"][2] == pytest.approx(
                o.table_z_mm + o.mesh_sizes_mm[support["mesh"]][2])
            offset = np.subtract(top["xyz_mm"][:2], support["xyz_mm"][:2])
            assert np.all(np.abs(offset) <= o.stack_offset_mm)
            # Its centre lies over the support's footprint.
            rel = footprint_of(config, support) - top["xyz_mm"][:2]
            edge = np.roll(rel, -1, axis=0) - rel
            cross = edge[:, 1] * rel[:, 0] - edge[:, 0] * rel[:, 1]
            assert np.all(cross > 0) or np.all(cross < 0)
            n += 1
        assert n > 50

    def test_forced_class_keeps_its_stacking(self, config):
        # A failure condition on carton draws from the 25 carton rows: 4 rest on a
        # plastic box, 21 stand alone.
        s = make_sampler(config)
        cond = {"class_name": "carton", "region": None, "lighting": None}
        firsts = [s.sample(i, condition=cond)["objects"][:2] for i in range(400)]
        assert all(f[0]["class_name"] == "carton" for f in firsts)
        supports = [f[1]["class_name"] for f in firsts if "stack_on_index" in f[0]]
        assert set(supports) == {"plastic_box"}
        assert abs(len(supports) / 400 - 4 / 25) < 0.06

    def test_carton_sizes_balanced(self, config):
        # As in the <list>_carton.csv sidecars: each size in half of the slots.
        s = make_sampler(config)
        meshes = config.objects.meshes_of("carton")
        roles = {"alone": Counter(), "top": Counter(), "support": Counter()}
        for cls, n in (("carton", 400), ("inox_box", 600)):
            cond = {"class_name": cls, "region": None, "lighting": None}
            for i in range(n):
                objs = s.sample(i, condition=cond)["objects"]
                if cls == "carton":
                    role = "top" if "stack_on_index" in objs[0] else "alone"
                    roles[role][objs[0]["mesh"]] += 1
                elif "stack_on_index" in objs[0] and objs[1]["class_name"] == "carton":
                    roles["support"][objs[1]["mesh"]] += 1
        for role, c in roles.items():
            assert set(c) == set(meshes), role
            assert 0.25 < c[meshes[0]] / sum(c.values()) < 0.75, role


class TestConditions:
    def test_condition_forces_class_and_region(self, config):
        cond = {"class_name": "carton", "region": [450.0, 500.0, -50.0, 0.0],
                "lighting": None}
        s = make_sampler(config)
        for i in range(30):
            objs = s.sample(i, condition=cond)["objects"]
            assert objs[0]["class_name"] == "carton"
            # The placement: the product itself, or the support it rests on.
            placed = objs[1] if "stack_on_index" in objs[0] else objs[0]
            x, y, _ = placed["xyz_mm"]
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

    @pytest.mark.parametrize("section, key, value, message", [
        ("lighting", "blind_intensity_rel_range", [1.5, 3.0], "bracket the anchor"),
        ("lighting", "blind_intensity_rel_range", [0.0, 3.0], "bracket the anchor"),
        ("lighting", "anchor_positions_mm", [[700.0, 0.0, 500.0]], "above"),
        ("objects", "count_range", [0, 2], "count_range"),
        ("objects", "stack_offset_mm", 40.0, "overhang"),
        ("objects", "first_object_mix", {"widget": {"single": 1}}, "unknown classes"),
        ("objects", "first_object_mix", {"carton": {"stacked_on": {"carton": 2}}}, "itself"),
    ])
    def test_bad_config_rejected(self, config, section, key, value, message):
        raw = config.model_dump()
        raw[section][key] = value
        with pytest.raises(ValidationError, match=message):
            SynthgenConfig.model_validate(raw)

    def test_every_mesh_needs_a_size(self, config):
        raw = config.model_dump()
        del raw["objects"]["mesh_sizes_mm"]["models/objects/inox_box.stl"]
        with pytest.raises(ValidationError, match="no size"):
            SynthgenConfig.model_validate(raw)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
