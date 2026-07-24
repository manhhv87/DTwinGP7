"""
stats.py
────────
Statistics for paired grasp-success comparisons (paper §3.7).

Grasp outcomes are BINARY and PAIRED (same pre-drawn pose list run under two
training configurations) → the correct test is McNemar's EXACT test on the
discordant pairs, NOT a t-test. Also provides Holm's step-down correction for
multiple comparisons against a common baseline, and paired-bootstrap CIs for
the success-rate difference.

numpy + scipy only; no pandas dependency → importable everywhere.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def paired_contingency(
    success_a: Sequence[int | bool],
    success_b: Sequence[int | bool],
) -> tuple[int, int, int, int]:
    """Paired outcomes → (n00, n01, n10, n11).

    n01 = A failed, B succeeded; n10 = A succeeded, B failed (the discordant
    counts McNemar tests). Inputs must be aligned pose-by-pose.
    """
    a = np.asarray(success_a, dtype=bool)
    b = np.asarray(success_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"Paired arrays must match: {a.shape} vs {b.shape}")
    n11 = int(np.sum(a & b))
    n00 = int(np.sum(~a & ~b))
    n01 = int(np.sum(~a & b))
    n10 = int(np.sum(a & ~b))
    return n00, n01, n10, n11


def mcnemar_exact(n01: int, n10: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts.

    Under H0 the n01+n10 discordant pairs split Binomial(n, 0.5); the exact
    two-sided p is 2*P(X <= min(n01, n10)), clipped to 1. With zero discordant
    pairs the test is undefined → p = 1.0 (no evidence of a difference).
    """
    from scipy.stats import binom  # lazy import

    if n01 < 0 or n10 < 0:
        raise ValueError("discordant counts must be non-negative")
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return float(min(1.0, 2.0 * binom.cdf(k, n, 0.5)))


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error control).

    Returns adjusted p-values in the ORIGINAL order. Monotonicity is enforced
    (each adjusted p >= the previous in the sorted sequence), values clipped to 1.
    """
    p = np.asarray(p_values, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted.tolist()


def bootstrap_diff_ci(
    success_a: Sequence[int | bool],
    success_b: Sequence[int | bool],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for the success-rate difference (B − A).

    Resamples PAIRS (preserving the pairing) → percentile interval.

    Returns:
        (delta, ci_low, ci_high) where delta = mean(B) − mean(A) in
        proportion units (multiply by 100 for percentage points).
    """
    a = np.asarray(success_a, dtype=float)
    b = np.asarray(success_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("success_a / success_b must be 1-D and aligned")
    n = len(a)
    if n == 0:
        raise ValueError("empty inputs")

    delta = float(b.mean() - a.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return delta, float(lo), float(hi)


def mcnemar_summary(
    success_a: Sequence[int | bool],
    success_b: Sequence[int | bool],
    label_a: str = "A",
    label_b: str = "B",
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    """One-call paired comparison: contingency + exact p + bootstrap CI.

    Returns a dict ready for logging / JSON: rates, (n00, n01, n10, n11),
    p_mcnemar_exact, delta_pp with 95% CI in percentage points.
    """
    n00, n01, n10, n11 = paired_contingency(success_a, success_b)
    p = mcnemar_exact(n01, n10)
    delta, lo, hi = bootstrap_diff_ci(success_a, success_b, n_boot=n_boot, seed=seed)
    a = np.asarray(success_a, dtype=float)
    b = np.asarray(success_b, dtype=float)
    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_pairs": int(len(a)),
        "rate_a": float(a.mean()),
        "rate_b": float(b.mean()),
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "p_mcnemar_exact": p,
        "delta_pp": delta * 100.0,
        "ci95_pp": [lo * 100.0, hi * 100.0],
    }
