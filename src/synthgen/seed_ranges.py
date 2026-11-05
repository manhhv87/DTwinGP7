"""
seed_ranges.py
──────────────
Which random streams a batch of scene specs used, so that a new batch can be
checked against the batches it will be trained with (paper §3.7, E4: generation
seed/index ranges do not overlap the initial dataset or earlier updates).

A stream is named by the two 32-bit words numpy seeds it from. SeedSequence pads
its entropy with zero words, so default_rng(k) is the stream of SeedSequence([k, 0]):

  spec_version 1  scene `index` of seed `seed` drew from default_rng(seed + index),
                  words (seed + index, 0): batches with different seeds share every
                  stream whose seed + index they have in common.
  spec_version 2  default_rng(SeedSequence([seed, index])), words (seed, index):
                  batches with different seeds share no stream, batches with one
                  seed share the indices they have in common. Index 0 of seed k is
                  the version 1 stream k. The E5 pose arm also draws from
                  SeedSequence([seed, index, 1]), which no two-word stream equals.
  spec_version 3  as version 2, words (seed, index), and each scene's session exposure
                  gain from SeedSequence([seed, index, 2]): a third word other than 0
                  never equals a two-word stream, so the overlap rule is version 2's.
  spec_version 4  as version 3, and the anchored products' top-face textures from
                  SeedSequence([seed, index, 3]); the overlap rule is version 2's.

A sampler RNG change needs a SPEC_VERSION bump and its rule here; a spec version
without one is reported as unchecked. A seed or index of 2**32 or more is refused,
since numpy splits it into two words: SeedSequence([2**32 + 5, 0]) is
SeedSequence([5, 1]).

Streams are kept as blocks ((lo, hi), (lo, hi)) of the first and second word,
half-open. Pure stdlib — fully unit-testable.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

STREAM_RULE = {1: "default_rng(seed + index)",                  # spec_version → rule
               2: "default_rng(SeedSequence([seed, index]))",
               3: "default_rng(SeedSequence([seed, index])); exposure SeedSequence([seed, index, 2])",
               4: "default_rng(SeedSequence([seed, index])); exposure SeedSequence([seed, index, 2]); "
                  "top textures SeedSequence([seed, index, 3])"}
WORD = 2 ** 32
_SCENE = re.compile(r"scene_(\d+)\.json$")

Ranges = list[tuple[int, int]]   # sorted, disjoint, half-open [lo, hi)
Blocks = list[tuple[tuple[int, int], tuple[int, int]]]   # (first word, second word) ranges


def _ranges(values: Iterable[int]) -> Ranges:
    out: Ranges = []
    for value in sorted(set(values)):
        if out and out[-1][1] == value:
            out[-1] = (out[-1][0], value + 1)
        else:
            out.append((value, value + 1))
    return out


def _words(values: list[int]) -> list[int]:
    for value in values:
        if not 0 <= value < WORD:
            raise ValueError(f"{value} is not one 32-bit seed word: numpy would split it, "
                             "and its stream could be another scene's")
    return values


def scene_streams(spec_version: int, seed: int, indices: Iterable[int]) -> Blocks:
    """Streams of the scenes `indices` drawn with `seed` under a spec version's rule."""
    if spec_version not in STREAM_RULE:
        raise ValueError(f"stream rule of spec_version {spec_version!r} is not known")
    seed, indices = int(seed), [int(index) for index in indices]
    if spec_version == 1:
        return [((lo, hi), (0, 1)) for lo, hi in _ranges(_words([seed + i for i in indices]))]
    _words([seed])
    return [((seed, seed + 1), (lo, hi)) for lo, hi in _ranges(_words(indices))]


def planned_streams(seed: int, n: int, start_index: int = 0, *, spec_version: int) -> Blocks:
    """Streams of n specs the given spec version writes from start_index with the seed."""
    if n < 1 or start_index < 0:
        raise ValueError("n must be positive and start_index non-negative")
    return scene_streams(spec_version, seed, range(start_index, start_index + n))


def manifest_streams(path: str | Path) -> dict[str, Any]:
    """specs/manifest.json (SceneSampler.write_specs) → the streams its scenes used.

    Returns {"path", "seed", "spec_version", "mode", "kappa", "n", "streams",
    "unchecked"}: streams is a list of [[lo, hi], [lo, hi]] blocks, or None with
    the reason in "unchecked" when the spec version's stream rule is not known
    here or a seed word does not fit 32 bits.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    indices = []
    for name in data.get("files", []):
        match = _SCENE.search(str(name))
        if not match:
            raise ValueError(f"{path}: unexpected scene file name {name!r}")
        indices.append(int(match.group(1)))
    version = data.get("spec_version")
    info: dict[str, Any] = {
        "path": str(path), "seed": data.get("seed"), "spec_version": version,
        "mode": data.get("mode"), "kappa": data.get("kappa"), "n": len(indices),
        "streams": None, "unchecked": None,
    }
    try:
        blocks = scene_streams(version, data.get("seed"), indices)
    except ValueError as exc:
        info["unchecked"] = str(exc)
        return info
    info["streams"] = [[list(first), list(second)] for first, second in blocks]
    return info


def _shared(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def shared_streams(a: Blocks, b: Blocks) -> int:
    """Number of streams two block lists have in common."""
    return sum(_shared(a1, b1) * _shared(a2, b2) for a1, a2 in a for b1, b2 in b)


def overlaps(named: dict[str, Blocks]) -> list[tuple[str, str, int]]:
    """Every pair of named batches that shares streams → (name, name, count)."""
    names = list(named)
    found = []
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            count = shared_streams(named[first], named[second])
            if count:
                found.append((first, second, count))
    return found
