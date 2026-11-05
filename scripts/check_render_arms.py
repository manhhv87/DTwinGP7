#!/usr/bin/env python
"""Check that the synthetic arms of one comparison were rendered the same way.

Each argument is one synthetic batch: its run directory (data/synth/<arm>, which holds
specs/, render/ and dataset/), or its dataset/ or render/ directory. The render manifest
is read from dataset/render_manifest.json (copied there by --make-labels), else from
render/render_manifest.json. The arms must agree on every key of
hard_profile.MATCH_KEYS: Blender and BlenderProc versions, the devices used, the render
settings, the hash of the scene profile, the digest of the specs' generator profile
(sampler config: materials, light, exposure, camera) and the digest of the generator
code. Every batch must also hold a rendered frame for each of its scene specs, and,
with --expect-kappa K, have been sampled at kappa K.

Packaging compares render manifests only within one package and against the paired
package of an E4 update. The four E3 recipes and the six E5 arms are packaged one by one,
so this is where they are compared:

    python scripts/check_render_arms.py data/synth/anchored_k1 data/synth/anchored_k2 \\
        data/synth/anchored_k4 data/synth/blind
    python scripts/check_render_arms.py --expect-kappa 2 data/synth/anchored_k2 data/synth/e5_*

Prints one line per arm and the verdict; exits 2 when the arms differ or a batch is
incomplete, 0 when they match.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.synthgen.hard_profile import MANIFEST_NAME, check_arms_match  # noqa: E402


def read_arm(path: Path) -> dict:
    """Render manifest, frame count and scene count of one batch."""
    path = Path(path)
    run = path.parent if path.name in ("dataset", "render") else path
    for candidate in (run / "dataset" / MANIFEST_NAME, run / "render" / MANIFEST_NAME):
        if candidate.is_file():
            break
    else:
        raise ValueError(f"{path}: no {MANIFEST_NAME} in {run / 'dataset'} or {run / 'render'}; "
                         "render the batch and run --make-labels first")
    manifest = json.loads(candidate.read_text(encoding="utf-8"))
    specs = run / "specs" / "manifest.json"
    if not specs.is_file():
        raise ValueError(f"{path}: no {specs}; without the scene list the batch cannot be "
                         "checked for missing frames")
    scenes = json.loads(specs.read_text(encoding="utf-8"))
    return {"arm": run.name, "path": str(candidate), "manifest": manifest,
            "frames": len(manifest.get("frames", [])), "specs": len(scenes.get("files", [])),
            "kappa": scenes.get("kappa")}


def check(arms: list[dict], expect_kappa: float | None = None) -> None:
    """ValueError naming the first difference between the arms, an incomplete batch, or
    (with expect_kappa) an arm sampled at another kappa."""
    names = [a["arm"] for a in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"two batches share one name: {names}")
    check_arms_match({a["arm"]: a["manifest"] for a in arms})
    if expect_kappa is not None:
        other = [f"{a['arm']} (kappa {a['kappa']})" for a in arms if a["kappa"] != expect_kappa]
        if other:
            raise ValueError(f"arms not sampled at kappa {expect_kappa}: " + "; ".join(other))
    short = [f"{a['arm']} ({a['frames']} frames, {a['specs']} specs)" for a in arms
             if a["frames"] < a["specs"]]
    if short:
        raise ValueError("incomplete batches: " + "; ".join(short))


def describe(arm: dict) -> str:
    m = arm["manifest"]
    return (f"{arm['arm']}: {arm['frames']}/{arm['specs']} frames, kappa {arm['kappa']}, {m.get('device')}, "
            f"Blender {m.get('blender_version')}, BlenderProc {m.get('blenderproc_version')}, "
            f"samples {m.get('render', {}).get('samples')}, profile {str(m.get('profile_sha256'))[:12]}, "
            f"generator {str(m.get('generator_sha256'))[:12]}, code {str(m.get('code_sha256'))[:12]} "
            f"({m.get('code_commit')})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batches", nargs="+", help="Run, dataset or render directories (two or more)")
    parser.add_argument("--expect-kappa", type=float, default=None,
                        help="Every arm must have been sampled at this kappa (E5: the kappa "
                             "chosen in E3, for the full arm and every ablation)")
    args = parser.parse_args(argv)
    if len(args.batches) < 2:
        parser.error("give at least two batches to compare")
    try:
        arms = [read_arm(Path(p)) for p in args.batches]
    except (ValueError, OSError) as exc:
        print(f"REFUSED: {exc}")
        return 2
    for arm in arms:
        print(describe(arm))
    try:
        check(arms, args.expect_kappa)
    except ValueError as exc:
        print(f"MISMATCH: {exc}")
        return 2
    print(f"MATCH: {len(arms)} arms rendered alike")
    return 0


if __name__ == "__main__":
    sys.exit(main())
