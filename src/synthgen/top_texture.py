"""
top_texture.py
──────────────
Measured top-face textures of the dielectric products (carton, plastic, wood) for the
anchored recipe: the stored file, its decoding, the manifest that lists them.

A texture is the pattern of one product's top face, cut from a TRAIN-split image of
the real dataset (scripts/make_top_textures.py: one clean instance, the top face
located by fitting the calibrated camera's projection of the caliper box to its
label, resampled onto the face in the mesh's own axes) and divided by its own mean
in each LINEAR channel. So a texture carries only the pattern (tape, print, grain,
rim and screws) and none of the camera state or light of the image it came from:
the renderer multiplies it with the class's fitted albedo (objects.materials, with
the sampler's anchored draws), which stays the mean colour of the face.

Storage: an 8-bit sRGB-encoded PNG of t / scale, t being the mean-one linear pattern
and scale the texture's own power of two at or above its largest value (manifest and
spec "scale"), so no value is clipped: a bright screw on the dark plastic face is 20 to
40 times the face's mean in a channel, a carton's seam under 2. Row 0 of the PNG lies at
the mesh's +y edge, column 0 at its -x edge, so the face seen from above is not
mirrored. decode() undoes the encoding exactly; the renderer builds the Blender image
from decode() itself rather than from Blender's own colour management, so the pattern
it renders is the one the script wrote.

The manifest (models/textures/manifest.json) lists per mesh the texture files with
their SHA-256; load_manifest checks every file against it. The sampler puts the
chosen file and its SHA-256 into the spec, and the generator profile of a batch
holds the whole list, so a changed texture changes the batch's generator digest.

Pure numpy + stdlib (PIL only in read_png): render_blenderproc.py loads this file by
path inside Blender, whose Python has no pydantic.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .sensor_model import linear_to_srgb, srgb_to_linear
except ImportError:  # loaded as a top-level module by render_blenderproc.py
    from sensor_model import linear_to_srgb, srgb_to_linear  # type: ignore[no-redef]

MANIFEST_VERSION = 1
ENCODING = "8-bit sRGB-encoded t / scale, t = linear pattern of mean 1 per channel"


def scale_for(t: Any) -> float:
    """The storage scale of a pattern: the smallest power of two (at least 2) at or
    above its largest value."""
    peak = float(np.max(np.asarray(t, dtype=float)))
    return float(2.0 ** max(1, int(np.ceil(np.log2(max(peak, 1e-9))))))


def encode(t: Any, scale: float) -> np.ndarray:
    """Mean-one linear pattern (H, W, 3) to the stored 8-bit codes."""
    t = np.asarray(t, dtype=float)
    if t.max() > scale:
        raise ValueError(f"pattern peaks at {t.max():.3g}, above its scale {scale:g}")
    return np.round(255.0 * linear_to_srgb(t / scale)).astype(np.uint8)


def decode(codes: Any, scale: float) -> np.ndarray:
    """Stored 8-bit codes (H, W, 3) to the linear pattern t (float32)."""
    c = np.asarray(codes, dtype=float) / 255.0
    return (srgb_to_linear(c) * float(scale)).astype(np.float32)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_png(path: str | Path) -> np.ndarray:
    """The stored codes of a texture file, (H, W, 3) uint8, row 0 at the top."""
    from PIL import Image

    with Image.open(path) as im:
        if im.mode != "RGB":
            raise ValueError(f"{path}: texture must be an 8-bit RGB PNG, got mode {im.mode}")
        return np.asarray(im, dtype=np.uint8).copy()


def load_manifest(repo_root: str | Path, manifest: str | Path) -> dict[str, list[dict[str, Any]]]:
    """{mesh: [{"file", "sha256", "scale"}, ...]} of a texture manifest, every file checked.

    Paths are relative to the repo root, as the meshes are. A file whose SHA-256
    differs from the manifest, or a missing one, raises: a texture changed after the
    manifest was written would otherwise enter a batch unrecorded.
    """
    root = Path(repo_root)
    data = json.loads((root / manifest).read_text(encoding="utf-8"))
    if data.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(f"{manifest}: manifest_version {data.get('manifest_version')!r}, "
                         f"expected {MANIFEST_VERSION}")
    if data.get("encoding") != ENCODING:
        raise ValueError(f"{manifest}: encoding {data.get('encoding')!r} is not {ENCODING!r}")
    out: dict[str, list[dict[str, Any]]] = {}
    for mesh, entries in sorted(data["textures"].items()):
        if not entries:
            raise ValueError(f"{manifest}: no texture listed for {mesh}")
        listed = []
        for e in entries:
            path = root / e["file"]
            if not path.is_file():
                raise ValueError(f"{manifest}: texture file {e['file']} is missing")
            sha = file_sha256(path)
            if sha != e["sha256"]:
                raise ValueError(f"{manifest}: {e['file']} has SHA-256 {sha[:12]}, the manifest "
                                 f"{e['sha256'][:12]}: regenerate the manifest with the textures")
            scale = float(e["scale"])
            if scale < 1.0:
                raise ValueError(f"{manifest}: {e['file']} has scale {scale}, below 1")
            listed.append({"file": str(e["file"]), "sha256": sha, "scale": scale})
        out[str(mesh)] = listed
    return out


def pick(entries: list[dict[str, Any]], u: float) -> dict[str, Any]:
    """The texture a uniform draw u in [0, 1) selects."""
    return dict(entries[min(int(u * len(entries)), len(entries) - 1)])
