"""
seed_ranges.py
──────────────
Which random streams a batch of scene specs used, so that a new batch can be
checked against the batches it will be trained with (paper §3.7, E4: generation
seed/index ranges do not overlap the initial dataset or earlier updates).

SceneSampler (spec_version 1) draws scene `index` of a run with seed `seed` from
numpy.random.default_rng(seed + index). Two scenes with the same seed + index
share their camera, light, background and placement draws even when the seeds
differ, so batches are compared on seed + index, not on the seed value alone.
A spec version with another stream rule is reported as unchecked.

Pure stdlib — fully unit-testable.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

STREAM_RULE = {1: "seed + index"}   # spec_version → stream identity
_SCENE = re.compile(r"scene_(\d+)\.json$")

Ranges = list[tuple[int, int]]   # sorted, disjoint, half-open [lo, hi)


def _ranges(values: list[int]) -> Ranges:
    out: Ranges = []
    for value in sorted(set(values)):
        if out and out[-1][1] == value:
            out[-1] = (out[-1][0], value + 1)
        else:
            out.append((value, value + 1))
    return out


def planned_streams(seed: int, n: int, start_index: int = 0) -> Ranges:
    """Streams of n specs written from start_index with the given seed."""
    if n < 1 or start_index < 0:
        raise ValueError("n must be positive and start_index non-negative")
    return [(seed + start_index, seed + start_index + n)]


def manifest_streams(path: str | Path) -> dict[str, Any]:
    """specs/manifest.json (SceneSampler.write_specs) → the streams its scenes used.

    Returns {"path", "seed", "spec_version", "mode", "kappa", "n", "streams",
    "unchecked"}: streams is a list of [lo, hi) pairs, or None with the reason in
    "unchecked" when the spec version's stream rule is not known here.
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
    if version not in STREAM_RULE:
        info["unchecked"] = f"stream rule of spec_version {version!r} is not known"
        return info
    seed = int(data["seed"])
    info["streams"] = [list(r) for r in _ranges([seed + index for index in indices])]
    return info


def shared_streams(a: Ranges, b: Ranges) -> int:
    """Number of streams two range lists have in common."""
    return sum(max(0, min(hi1, hi2) - max(lo1, lo2)) for lo1, hi1 in a for lo2, hi2 in b)


def overlaps(named: dict[str, Ranges]) -> list[tuple[str, str, int]]:
    """Every pair of named batches that shares streams → (name, name, count)."""
    names = list(named)
    found = []
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            count = shared_streams(named[first], named[second])
            if count:
                found.append((first, second, count))
    return found
