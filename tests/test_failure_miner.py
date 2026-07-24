"""
test_failure_miner.py
─────────────────────
Unit tests for failure-mode mining + budget allocation (C3, Algorithm 1).

Run:
    pytest tests/test_failure_miner.py -v
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from src.synthgen import (
    allocate,
    bin_failures,
    condition_for,
    kmeans_sensitivity,
    load_report,
    load_trials,
    save_report,
)

REGION = (400.0, 1000.0, -200.0, 200.0)


def _row(success, cls="bottle", reason="", lighting="bright", x=None, y=None):
    r = {"trial_id": "1", "success": str(success), "class_name": cls,
         "failure_reason": reason, "lighting": lighting}
    if x is not None:
        r["det_x_mm"] = str(x)
        r["det_y_mm"] = str(y)
    return r


class TestBinFailures:
    def test_counts_and_fracs(self):
        rows = (
            [_row(0, "bottle", "grasp_failed", "dim", 450, -150)] * 3
            + [_row(0, "cup", "detection_miss", "")]        # no coords, no lighting
            + [_row(1, "bottle")] * 6                        # successes ignored
        )
        report = bin_failures(rows, region=REGION, grid=(3, 3))
        assert report["n_trials"] == 10
        assert report["n_failures"] == 4
        assert len(report["clusters"]) == 2

        top = report["clusters"][0]                          # sorted by count desc
        assert top["count"] == 3
        assert top["class_name"] == "bottle"
        assert top["reason"] == "grasp_failed"
        assert top["lighting"] == "dim"
        assert top["region_cell"] == [0, 0]                  # x=450,y=-150 → cell (0,0)
        x0, x1, y0, y1 = top["region_bounds"]
        assert (x0, x1) == (400.0, 600.0) and (y0, y1) == (-200.0, -200.0 + 400 / 3)
        assert top["frac"] == pytest.approx(0.75)

        other = report["clusters"][1]
        assert other["region_cell"] is None                  # detection_miss: no coords
        assert other["lighting"] == "unlabelled"

    def test_no_failures(self):
        report = bin_failures([_row(1)] * 5, region=REGION)
        assert report["n_failures"] == 0 and report["clusters"] == []


class TestKmeans:
    def test_two_clear_clusters(self):
        rng = np.random.default_rng(0)
        rows = []
        for cx, cy in ((500, -100), (900, 150)):
            for _ in range(10):
                rows.append(_row(0, "bolt", "grasp_failed", "bright",
                                 cx + rng.normal(0, 5), cy + rng.normal(0, 5)))
        km = kmeans_sensitivity(rows, k=2, seed=0)
        assert km is not None and km["n_points"] == 20
        assert sorted(km["counts"]) == [10, 10]
        centers = np.asarray(km["centers_mm"])
        dists = [min(np.linalg.norm(centers - np.array([cx, cy]), axis=1).min(), 1e9)
                 for cx, cy in ((500, -100), (900, 150))]
        assert max(dists) < 50.0

    def test_too_few_points_returns_none(self):
        assert kmeans_sensitivity([_row(0, x=500, y=0)], k=4) is None


class TestAllocate:
    def _clusters(self, counts):
        total = sum(counts)
        return [{"key": f"m{i}", "count": c, "frac": c / total,
                 "class_name": "bottle", "region_bounds": None,
                 "lighting": "dim", "reason": "grasp_failed"}
                for i, c in enumerate(counts)]

    def test_sums_exactly_and_proportional(self):
        alloc = allocate(self._clusters([70, 20, 10]), n_total=1000, floor_frac=0.05)
        n = [e["n_images"] for e in alloc]
        assert sum(n) == 1000
        assert n[0] > n[1] > n[2]
        assert min(n) >= 50                                   # floor honoured

    def test_floor_dropped_when_infeasible(self):
        alloc = allocate(self._clusters([1] * 30), n_total=100, floor_frac=0.05)
        assert sum(e["n_images"] for e in alloc) == 100

    def test_invalid_budget_raises(self):
        with pytest.raises(ValueError, match="positive"):
            allocate(self._clusters([1]), n_total=0)

    def test_empty_clusters(self):
        assert allocate([], n_total=100) == []


class TestConditionFor:
    def test_mapping(self):
        cond = condition_for({"class_name": "cup", "region_bounds": [1, 2, 3, 4],
                              "lighting": "dim"})
        assert cond == {"class_name": "cup", "region": [1, 2, 3, 4],
                        "lighting": "dim"}

    def test_unknowns_become_none(self):
        cond = condition_for({"class_name": "unknown", "region_bounds": None,
                              "lighting": "unlabelled"})
        assert cond == {"class_name": None, "region": None, "lighting": None}


class TestIO:
    def test_csv_roundtrip_and_report_io(self, tmp_path):
        csv_path = tmp_path / "trials.csv"
        rows = [_row(0, "tray", "unreachable", "bright", 800, 100), _row(1)]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            writer.writeheader()
            writer.writerows(rows)

        loaded = load_trials([csv_path])
        assert len(loaded) == 2 and loaded[0]["_source"] == "trials.csv"

        report = bin_failures(loaded, region=REGION)
        report["allocation"] = allocate(report["clusters"], 100)
        path = save_report(report, tmp_path / "failure_modes.json")
        again = load_report(path)
        assert again["n_failures"] == 1
        assert sum(e["n_images"] for e in again["allocation"]) == 100

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_trials([tmp_path / "nope.csv"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
