"""
budget_alloc.py
───────────────
C3 step 2: turn a failure report into a rendering-budget allocation.

The budget N_k of loop iteration k is split across failure modes
proportionally to their observed frequency, with a floor for rare modes,
using the largest-remainder method so the parts sum EXACTLY to N_k
(Algorithm 1, line "reallocate the rendering budget").

Each allocation entry also carries the sampler `condition` dict that
SceneSampler.sample() understands — the concrete bridge that maps a physical
failure mode back into renderer parameters.

Pure stdlib — fully unit-testable.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def condition_for(cluster: dict[str, Any]) -> dict[str, Any]:
    """Failure cluster → SceneSampler condition constraint.

    class 'unknown' (e.g. detection_timeout rows) → no class constraint;
    missing region (no coordinates) → no region constraint;
    lighting 'unlabelled' → no lighting constraint.
    """
    cls = cluster.get("class_name")
    lighting = cluster.get("lighting")
    return {
        "class_name": None if cls in (None, "unknown") else cls,
        "region": cluster.get("region_bounds"),
        "lighting": None if lighting in (None, "unlabelled") else lighting,
    }


def allocate(
    clusters: list[dict[str, Any]],
    n_total: int,
    floor_frac: float = 0.05,
) -> list[dict[str, Any]]:
    """Split n_total images across failure clusters.

    Proportional to cluster counts, with every cluster guaranteed at least
    ceil(floor_frac * n_total) images (rare modes must not starve — they are
    often the interesting ones), totals fixed up by largest remainder.

    Args:
        clusters: bin_failures()["clusters"] (needs "count" per cluster).
        n_total: Total image budget N_k for this loop iteration.
        floor_frac: Per-cluster minimum share of n_total.

    Returns:
        List of {"key", "n_images", "condition", "count", "frac"} entries,
        sum(n_images) == n_total. Empty list if there are no clusters.
    """
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")
    if not clusters:
        return []

    floor = max(0, int(-(-floor_frac * n_total // 1)))  # ceil
    if floor * len(clusters) > n_total:
        # Too many modes for the floor to be honourable → plain proportional.
        logger.warning(
            "floor %d x %d clusters exceeds budget %d — dropping the floor",
            floor, len(clusters), n_total,
        )
        floor = 0

    total_count = sum(int(c["count"]) for c in clusters)
    raw = [
        floor + int(c["count"]) / total_count * (n_total - floor * len(clusters))
        for c in clusters
    ]
    base = [int(r) for r in raw]
    remainder = n_total - sum(base)
    # Largest remainder: hand out the leftover images one by one.
    order = sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:remainder]:
        base[i] += 1

    out = []
    for c, n in zip(clusters, base):
        out.append({
            "key": c["key"],
            "n_images": n,
            "condition": condition_for(c),
            "count": int(c["count"]),
            "frac": c.get("frac", 0.0),
        })
    assert sum(e["n_images"] for e in out) == n_total
    return out
