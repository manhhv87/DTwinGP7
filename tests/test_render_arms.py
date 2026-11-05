"""
test_render_arms.py
───────────────────
scripts/check_render_arms.py: the E3 recipes and the E5 arms are packaged one by one, so
their render manifests are compared here, not by packaging. Arms rendered on one
machine with one config pass; an arm from another machine, with other settings, or with
frames missing fails.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.synthgen import hard_profile as hp

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("check_render_arms",
                                               ROOT / "scripts" / "check_render_arms.py")
cra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cra)

LINUX = hp.render_device("GPU", "OPTIX", ["NVIDIA GeForce RTX 4090", "AMD Ryzen 9 7950X"])
LAPTOP = hp.render_device("GPU", "OPTIX", ["NVIDIA GeForce MX150", "Intel Core i7-8650U CPU @ 1.90GHz"])


@pytest.fixture(scope="module")
def cfg() -> dict:
    return hp.load_render_config(ROOT / "config" / "synthgen.yaml")


def batch(root: Path, name: str, cfg: dict, *, device: str = LINUX, frames: int = 3,
          specs: int = 3, where: str = "dataset", samples: int = 24) -> Path:
    """A run directory as 20_generate_synth.py lays it out, with its render manifest."""
    run = root / name
    (run / where).mkdir(parents=True)
    manifest = hp.build_manifest(cfg, {"samples": samples, "denoiser": "INTEL"}, device=device,
                                 blender_version="4.2.1 LTS", blenderproc_version="2.8.0",
                                 config_path="config/synthgen.yaml", scenes_dir=f"{name}/specs",
                                 frames=[f"frame_{i:06d}" for i in range(frames)])
    (run / where / hp.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    (run / "specs").mkdir()
    (run / "specs" / "manifest.json").write_text(
        json.dumps({"files": [f"scene_{i:06d}.json" for i in range(specs)]}), encoding="utf-8")
    return run


def test_e3_arms_of_one_machine_match(tmp_path: Path, cfg: dict, capsys) -> None:
    runs = [batch(tmp_path, name, cfg) for name in ("anchored_k1", "anchored_k2", "anchored_k4", "blind")]
    assert cra.main([str(r) for r in runs]) == 0
    out = capsys.readouterr().out
    assert "MATCH: 4 arms" in out and LINUX in out and "3/3 frames" in out


def test_an_arm_from_another_machine_fails(tmp_path: Path, cfg: dict, capsys) -> None:
    full = batch(tmp_path, "anchored_k2", cfg)
    other = batch(tmp_path, "e5_camera", cfg, device=LAPTOP)
    assert cra.main([str(full / "dataset"), str(other)]) == 2
    out = capsys.readouterr().out
    assert "MISMATCH" in out and "'e5_camera'" in out and "device" in out


def test_other_settings_and_incomplete_batches_fail(tmp_path: Path, cfg: dict) -> None:
    a = cra.read_arm(batch(tmp_path, "anchored_k2", cfg))
    b = cra.read_arm(batch(tmp_path, "blind", cfg, samples=16))
    with pytest.raises(ValueError, match="render"):
        cra.check([a, b])
    c = cra.read_arm(batch(tmp_path, "anchored_k4", cfg, frames=2))
    with pytest.raises(ValueError, match="incomplete.*anchored_k4"):
        cra.check([a, c])


def test_render_manifest_read_before_labels(tmp_path: Path, cfg: dict) -> None:
    run = batch(tmp_path, "e5_pose", cfg, where="render")
    arm = cra.read_arm(run / "render")
    assert arm["arm"] == "e5_pose" and arm["frames"] == 3 and arm["specs"] == 3


def test_a_batch_without_manifest_is_refused(tmp_path: Path, cfg: dict, capsys) -> None:
    (tmp_path / "empty" / "dataset").mkdir(parents=True)
    other = batch(tmp_path, "anchored_k2", cfg)
    assert cra.main([str(tmp_path / "empty"), str(other)]) == 2
    assert "REFUSED" in capsys.readouterr().out


def test_a_batch_without_its_scene_list_is_refused(tmp_path: Path, cfg: dict, capsys) -> None:
    """Without specs/manifest.json the frame count cannot be checked; that is an error."""
    run = batch(tmp_path, "anchored_k1", cfg)
    (run / "specs" / "manifest.json").unlink()
    other = batch(tmp_path, "anchored_k2", cfg)
    assert cra.main([str(run), str(other)]) == 2
    out = capsys.readouterr().out
    assert "REFUSED" in out and "missing frames" in out
