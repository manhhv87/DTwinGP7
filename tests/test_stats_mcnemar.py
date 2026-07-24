"""
test_stats_mcnemar.py
─────────────────────
Unit tests for the paired-comparison statistics (paper §3.7):
exact McNemar, Holm correction, paired-bootstrap CI.

Run:
    pytest tests/test_stats_mcnemar.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from src.utils.stats import (
    bootstrap_diff_ci,
    holm,
    mcnemar_exact,
    mcnemar_summary,
    paired_contingency,
)


class TestPairedContingency:
    def test_counts(self):
        a = [1, 1, 0, 0, 1, 0]
        b = [1, 0, 1, 0, 1, 1]
        n00, n01, n10, n11 = paired_contingency(a, b)
        assert (n00, n01, n10, n11) == (1, 2, 1, 2)
        assert n00 + n01 + n10 + n11 == len(a)

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError, match="must match"):
            paired_contingency([1, 0], [1])


class TestMcNemarExact:
    def test_known_value(self):
        # n01=1, n10=9 → p = 2 * P(X<=1 | Bin(10, .5)) = 2 * 11/1024
        assert mcnemar_exact(1, 9) == pytest.approx(2 * 11 / 1024)

    def test_symmetric(self):
        assert mcnemar_exact(2, 7) == mcnemar_exact(7, 2)

    def test_no_discordant_pairs(self):
        assert mcnemar_exact(0, 0) == 1.0

    def test_balanced_discordance_is_one(self):
        assert mcnemar_exact(3, 3) == 1.0

    def test_strong_effect_small_p(self):
        assert mcnemar_exact(0, 20) < 1e-4

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            mcnemar_exact(-1, 2)


class TestHolm:
    def test_known_adjustment(self):
        adj = holm([0.01, 0.04, 0.03])
        assert adj == pytest.approx([0.03, 0.06, 0.06])

    def test_monotone_and_clipped(self):
        adj = holm([0.5, 0.9, 0.7])
        assert all(0.0 <= p <= 1.0 for p in adj)
        assert adj[1] >= adj[2] >= adj[0]

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError):
            holm([0.1, 1.2])


class TestBootstrapCI:
    def test_contains_true_delta(self):
        rng = np.random.default_rng(0)
        a = rng.random(400) < 0.60
        b = rng.random(400) < 0.75
        delta, lo, hi = bootstrap_diff_ci(a, b, n_boot=2000, seed=1)
        assert lo < delta < hi
        assert lo > 0.0                          # 15 pp effect, n=400 → CI excludes 0
        assert delta == pytest.approx(b.mean() - a.mean())

    def test_deterministic_per_seed(self):
        a = [1, 0, 1, 1, 0] * 20
        b = [1, 1, 1, 1, 0] * 20
        r1 = bootstrap_diff_ci(a, b, n_boot=500, seed=7)
        r2 = bootstrap_diff_ci(a, b, n_boot=500, seed=7)
        assert r1 == r2

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            bootstrap_diff_ci([], [])


class TestMcNemarSummary:
    def test_summary_consistent(self):
        a = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0] * 10
        b = [1, 1, 1, 1, 0, 1, 1, 0, 1, 1] * 10
        s = mcnemar_summary(a, b, "baseline", "anchored", n_boot=500, seed=0)
        assert s["label_a"] == "baseline" and s["label_b"] == "anchored"
        assert s["n_pairs"] == 100
        assert s["n00"] + s["n01"] + s["n10"] + s["n11"] == 100
        assert s["p_mcnemar_exact"] == pytest.approx(
            mcnemar_exact(s["n01"], s["n10"]))
        assert s["delta_pp"] == pytest.approx((np.mean(b) - np.mean(a)) * 100)
        assert s["ci95_pp"][0] <= s["delta_pp"] <= s["ci95_pp"][1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
