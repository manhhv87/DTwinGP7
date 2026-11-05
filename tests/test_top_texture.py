"""
test_top_texture.py
───────────────────
Measured top-face textures of the anchored recipe (src/synthgen/top_texture.py,
scripts/make_top_textures.py, the sampler's texture draw).

Verifies: the stored encoding round-trips and never clips; the shipped manifest
names only the dielectric meshes, 20-40 textures each, every file matching its
SHA-256, of mean one per linear channel, sized to the caliper face, cut from the
train split only; a changed or missing file is refused; the extraction's box fit
recovers a known pose and the carton's size, and its texture axes are the
renderer's (column 0 at mesh -x, row 0 at mesh +y); the sampler textures exactly
the anchored products of those meshes, from a stream of its own, so the other
draws of a scene, the kappa arms and the E5 arms are unchanged by it; a metal
mesh in a manifest is refused.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.cell.pose_utils import make_homogeneous
from src.synthgen import SceneSampler, SynthgenConfig
from src.synthgen import scene_sampler as ss
from src.synthgen import top_texture as tt
from src.synthgen.scene_sampler import ABLATION_FACTORS, SPEC_VERSION
from src.synthgen.seed_ranges import STREAM_RULE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
T_BC = make_homogeneous([700.0, 0.0, 1200.0], [180.0, 0.0, 0.0])
TEXTURED = {"models/objects/carton_large.stl", "models/objects/carton_small.stl",
            "models/objects/plastic_box.stl", "models/objects/wood_box.stl"}


@pytest.fixture(scope="module")
def config() -> SynthgenConfig:
    return SynthgenConfig.from_yaml(PROJECT_ROOT / "config" / "synthgen.yaml")


@pytest.fixture(scope="module")
def manifest(config) -> dict:
    return json.loads((PROJECT_ROOT / config.objects.top_textures).read_text(encoding="utf-8"))


def script():
    path = PROJECT_ROOT / "scripts" / "make_top_textures.py"
    spec = importlib.util.spec_from_file_location("make_top_textures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestEncoding:
    def test_round_trip_within_half_a_code(self):
        rng = np.random.default_rng(0)
        t = rng.uniform(0.02, 1.9, size=(40, 50, 3))
        codes = tt.encode(t, 2.0)
        back = tt.decode(codes, 2.0)
        assert np.array_equal(tt.encode(back, 2.0), codes)
        # half an 8-bit sRGB step, relative: 2.4 x 0.5 / 255 / (s + 0.055) at code value s
        s = codes / 255.0
        assert np.all(np.abs(back - t) <= 2.4 * 0.5 / 255.0 / (s + 0.055) * t * 1.05 + 1e-6)

    def test_scale_holds_the_peak(self):
        assert tt.scale_for(np.array([0.5, 1.7])) == 2.0
        assert tt.scale_for(np.array([1.0, 2.0])) == 2.0
        assert tt.scale_for(np.array([1.0, 21.9])) == 32.0
        with pytest.raises(ValueError, match="above its scale"):
            tt.encode(np.array([[[5.0, 1.0, 1.0]]]), 4.0)


class TestShippedTextures:
    def test_config_names_the_manifest(self, config):
        assert config.objects.top_textures == "models/textures/manifest.json"

    def test_only_dielectric_meshes_20_to_40_each(self, config, manifest):
        assert set(manifest["textures"]) == TEXTURED
        for mesh, entries in manifest["textures"].items():
            assert 20 <= len(entries) <= 40, mesh
        metals = {m for c, mat in config.objects.materials.items() if mat.f0_rgb is not None
                  for m in config.objects.meshes_of(c)}
        assert not metals & set(manifest["textures"])

    def test_files_match_and_are_mean_one(self, config, manifest):
        loaded = tt.load_manifest(PROJECT_ROOT, config.objects.top_textures)
        px = manifest["settings"]["px_per_mm"]
        for mesh, entries in manifest["textures"].items():
            L, W, _ = config.objects.mesh_sizes_mm[mesh]
            assert [e["sha256"] for e in loaded[mesh]] == [e["sha256"] for e in entries]
            for e in entries:
                t = tt.decode(tt.read_png(PROJECT_ROOT / e["file"]), e["scale"])
                assert t.shape == (round(W * px), round(L * px), 3), e["file"]
                assert np.allclose(t.reshape(-1, 3).mean(axis=0), 1.0, atol=0.01), e["file"]
                assert t.max() <= e["scale"] + 1e-6

    def test_sources_are_train_images_only(self, manifest):
        for entries in manifest["textures"].values():
            for e in entries:
                assert e["source_image"].startswith("images/train/"), e["source_image"]
        sources = [e["source_image"] for es in manifest["textures"].values() for e in es]
        assert len(sources) == len(set(sources))

    def test_changed_or_missing_file_refused(self, manifest, tmp_path):
        root = tmp_path / "repo"
        entry = manifest["textures"]["models/objects/wood_box.stl"][0]
        (root / Path(entry["file"]).parent).mkdir(parents=True)
        shutil.copy(PROJECT_ROOT / entry["file"], root / entry["file"])
        small = dict(manifest, textures={"models/objects/wood_box.stl": [entry]})
        (root / "m.json").write_text(json.dumps(small))
        assert tt.load_manifest(root, "m.json")["models/objects/wood_box.stl"][0]["sha256"] == entry["sha256"]
        data = bytearray((root / entry["file"]).read_bytes())
        data[-20] ^= 0xFF
        (root / entry["file"]).write_bytes(bytes(data))
        with pytest.raises(ValueError, match="SHA-256"):
            tt.load_manifest(root, "m.json")
        (root / entry["file"]).unlink()
        with pytest.raises(ValueError, match="missing"):
            tt.load_manifest(root, "m.json")


class TestExtractionGeometry:
    @pytest.fixture(scope="class")
    def cam(self):
        import yaml
        cfg = yaml.safe_load((PROJECT_ROOT / "config" / "synthgen.yaml").read_text(encoding="utf-8"))
        return (np.load(PROJECT_ROOT / "config" / "calibration" / "T_base_camera.npy"),
                cfg["camera"]["intrinsics"], float(cfg["objects"]["table_z_mm"]),
                {m: [float(v) for v in s] for m, s in cfg["objects"]["mesh_sizes_mm"].items()})

    def test_fit_recovers_pose_and_carton_size(self, cam):
        m = script()
        T, intr, z0, sizes = cam
        small, large = sizes["models/objects/carton_small.stl"], sizes["models/objects/carton_large.stl"]
        for x, y, yaw in ((560.0, -120.0, 25.0), (620.0, 150.0, -60.0)):
            label = m.silhouette(T, intr, x, y, yaw, small, z0)
            fit_s, fit_l = m.fit_box(T, intr, label, small, z0), m.fit_box(T, intr, label, large, z0)
            assert fit_s["iou"] > 0.995 and fit_s["iou"] - fit_l["iou"] > 0.1
            assert abs(fit_s["x"] - x) < 0.5 and abs(fit_s["y"] - y) < 0.5
            assert abs((fit_s["yaw"] - yaw + 90.0) % 180.0 - 90.0) < 0.3

    def test_texture_axes_are_the_renderers(self, cam):
        """Column 0 lies at mesh -x and row 0 at mesh +y: the renderer maps u = x and
        v = y over the face and puts the PNG's first row at v = 1."""
        m = script()
        T, intr, z0, sizes = cam
        size = sizes["models/objects/plastic_box.stl"]
        fit = {"x": 580.0, "y": 30.0, "yaw": 35.0}
        args = type("A", (), {"px_per_mm": 1.25, "inset_mm": 0.0})
        mx, my = m.face_maps(T, intr, fit, size, z0 + size[2], args.px_per_mm, args.inset_mm)
        ex, ey = m.axes(fit["yaw"])
        c = np.array([fit["x"], fit["y"]])
        half = 0.5 / args.px_per_mm
        expect = {(0, 0): c + (-size[0] / 2 + half) * ex + (size[1] / 2 - half) * ey,
                  (0, -1): c + (size[0] / 2 - half) * ex + (size[1] / 2 - half) * ey,
                  (-1, 0): c + (-size[0] / 2 + half) * ex - (size[1] / 2 - half) * ey}
        for (r, k), p in expect.items():
            uv = m.project(T, intr, np.array([[p[0], p[1], z0 + size[2]]]))[0]
            assert np.allclose([mx[r, k], my[r, k]], uv, atol=1e-3)


class TestSamplerTextures:
    def test_anchored_products_of_textured_meshes_only(self, config):
        s = SceneSampler(config, T_BC, mode="anchored", kappa=2.0, seed=4)
        listed = {m: {e["file"] for e in es} for m, es in s.top_textures.items()}
        n = 0
        for i in range(200):
            spec = s.sample(i)
            assert spec["spec_version"] == SPEC_VERSION == 4
            for o in spec["objects"]:
                tex = o["material"].get("top_texture")
                if o["mesh"] in TEXTURED:
                    assert tex["file"] in listed[o["mesh"]]
                    n += 1
                else:
                    assert tex is None
        assert n > 100
        blind = SceneSampler(config, T_BC, mode="blind", seed=4)
        assert not any("top_texture" in o["material"] for i in range(100) for o in blind.sample(i)["objects"])

    def test_own_stream_leaves_every_other_draw(self, config):
        plain = config.model_copy(deep=True)
        plain.objects.top_textures = None
        a = SceneSampler(config, T_BC, mode="anchored", kappa=2.0, seed=9)
        b = SceneSampler(plain, T_BC, mode="anchored", kappa=2.0, seed=9)
        for i in range(60):
            sa, sb = a.sample(i), b.sample(i)
            for o in sa["objects"]:
                o["material"].pop("top_texture", None)
            assert sa == sb

    def test_kappa_and_e5_arms_share_textures(self, config):
        def files(sampler, i):
            return [o["material"].get("top_texture", {}).get("file") for o in sampler.sample(i)["objects"]]

        ref = SceneSampler(config, T_BC, mode="anchored", kappa=2.0, seed=3)
        arms = [SceneSampler(config, T_BC, mode="anchored", kappa=k, seed=3) for k in (1.0, 4.0)]
        arms += [SceneSampler(config, T_BC, mode="anchored", kappa=2.0, seed=3, ablation=f)
                 for f in ABLATION_FACTORS if f != "none"]
        for i in range(40):
            want = files(ref, i)
            for arm in arms:
                assert files(arm, i) == want

    def test_every_texture_is_drawn(self, config):
        s = SceneSampler(config, T_BC, mode="anchored", kappa=1.0, seed=0)
        used = {}
        for i in range(1500):
            for o in s.sample(i)["objects"]:
                tex = o["material"].get("top_texture")
                if tex:
                    used.setdefault(o["mesh"], set()).add(tex["file"])
        for mesh, entries in s.top_textures.items():
            assert used[mesh] == {e["file"] for e in entries}, mesh

    def test_stream_rule_and_profile(self, config):
        assert SPEC_VERSION in STREAM_RULE and "SeedSequence([seed, index, 3])" in STREAM_RULE[SPEC_VERSION]
        s = SceneSampler(config, T_BC, mode="blind", seed=0)
        assert s._generator_profile()["top_textures"] == tt.load_manifest(PROJECT_ROOT, config.objects.top_textures)

    def test_metal_mesh_refused(self, config, manifest, tmp_path, monkeypatch):
        entry = manifest["textures"]["models/objects/wood_box.stl"][0]
        (tmp_path / Path(entry["file"]).parent).mkdir(parents=True)
        shutil.copy(PROJECT_ROOT / entry["file"], tmp_path / entry["file"])
        bad = dict(manifest, textures={"models/objects/inox_box.stl": [entry]})
        (tmp_path / "m.json").write_text(json.dumps(bad))
        cfg = config.model_copy(deep=True)
        cfg.objects.top_textures = "m.json"
        monkeypatch.setattr(ss, "REPO_ROOT", tmp_path)
        with pytest.raises(ValueError, match="metal"):
            SceneSampler(cfg, T_BC, mode="anchored", seed=0)
