"""
test_failure_miner.py
─────────────────────
Unit tests for failure-mode mining + budget allocation (C3, Algorithm 1).

Run:
    pytest tests/test_failure_miner.py -v
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

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
from src.synthgen.failure_miner import (
    file_sha256,
    pose_number,
    stage_trials,
    trial_outcome,
)
from src.synthgen.seed_ranges import manifest_streams, overlaps, planned_streams, shared_streams

REGION = (400.0, 1000.0, -200.0, 200.0)


def _row(success, cls="bottle", reason="", lighting="bright", x=None, y=None,
         human_ok="", stack_on=""):
    """A trial already joined to its staged pose (stage_trials output)."""
    r = {"trial_id": "1", "success": str(success), "human_ok": str(human_ok),
         "staged_class": cls, "failure_reason": reason, "lighting": lighting,
         "stack_on": stack_on}
    if x is not None:
        r["staged_x_mm"] = str(x)
        r["staged_y_mm"] = str(y)
    return r


class TestBinFailures:
    def test_counts_and_fracs(self):
        rows = (
            [_row(0, "bottle", "grasp_failed", "dim", 450, -150)] * 3
            + [_row(0, "cup", "detection_miss", "")]        # not staged, no lighting
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
        assert other["region_cell"] is None                  # no staged position
        assert other["lighting"] == "unlabelled"

    def test_missed_detection_bins_by_staged_class_and_card_not_detector(self):
        # A miss logs no detected class or position; a wrong detection logs the
        # wrong ones. Both must bin by what was staged on the card.
        miss = dict(_row(0, "inox_box", "detection_miss", "dim", 575, 20),
                    class_name="", det_x_mm="", det_y_mm="")
        wrong = dict(_row(0, "inox_box", "grasp_failed", "dim", 575, 20),
                     class_name="metal_box", det_x_mm="900", det_y_mm="-190")
        report = bin_failures([miss, wrong], region=(510.0, 640.0, -175.0, 215.0))
        assert [c["key"] for c in report["clusters"]] == [
            "inox_box|r11|dim|detection_miss", "inox_box|r11|dim|grasp_failed"]
        assert all(condition_for(c)["class_name"] == "inox_box" for c in report["clusters"])

    def test_unstaged_rows_form_explicit_unknown_bins(self):
        row = {"success": "0", "failure_reason": "detection_miss", "lighting": "dim",
               "class_name": "carton", "det_x_mm": "500", "det_y_mm": "0"}
        cluster = bin_failures([row], region=REGION)["clusters"][0]
        assert cluster["key"] == "unknown|na|dim|detection_miss"

    def test_stack_on_is_context_not_part_of_the_key(self):
        rows = [_row(0, "inox_box", "detection_miss", "dim", 575, 20, stack_on="carton"),
                _row(0, "inox_box", "detection_miss", "dim", 575, 20)]
        report = bin_failures(rows, region=(510.0, 640.0, -175.0, 215.0))
        assert len(report["clusters"]) == 1
        assert report["clusters"][0]["stack_on"] == {"carton": 1, "none": 1}
        entry = allocate(report["clusters"], 100)[0]
        assert entry["stack_on"] == {"carton": 1, "none": 1}
        assert "stack_on" not in entry["condition"]

    def test_no_failures(self):
        report = bin_failures([_row(1)] * 5, region=REGION)
        assert report["n_failures"] == 0 and report["clusters"] == []


class TestOutcome:
    def test_operator_verdict_overrides_machine_flag(self):
        dropped = _row(1, "carton", "", "dim", 575, 20, human_ok=0)
        assert trial_outcome(dropped) == (True, "human_ok")
        cluster = bin_failures([dropped], region=REGION)["clusters"][0]
        assert cluster["reason"] == "operator_verdict"
        assert trial_outcome(_row(0, human_ok=1)) == (False, "human_ok")

    def test_refused_trial_without_verdict_falls_back_to_success(self):
        assert trial_outcome(_row(0, reason="detection_miss")) == (True, "success")
        assert trial_outcome(_row(1)) == (False, "success")
        report = bin_failures([_row(0), _row(1, human_ok=1)], region=REGION)
        assert report["outcome_sources"] == {"human_ok": 1, "success": 1}

    def test_unreadable_or_missing_outcome_is_refused(self):
        with pytest.raises(ValueError, match="Unreadable"):
            trial_outcome(_row("maybe"))
        with pytest.raises(ValueError, match="neither"):
            trial_outcome({"trial_id": "3", "success": "", "human_ok": ""})

    def test_pose_number(self):
        assert pose_number("P0221") == 221
        with pytest.raises(ValueError):
            pose_number("221")


POSE_FIELDS = ["pose_id", "card_id", "x_mm", "y_mm", "yaw_deg", "class_hint", "condition", "stack_on"]


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _pose_list(path: Path, x_mm: float = 575.0) -> Path:
    rows = [{"pose_id": f"P{n:04d}", "card_id": "11", "x_mm": x_mm, "y_mm": 20.0,
             "yaw_deg": 0.0, "class_hint": "inox_box" if n % 2 else "carton",
             "condition": "novel_background_dim", "stack_on": "metal_box" if n == 223 else ""}
            for n in range(1, 301)]
    return _write_csv(path, rows, POSE_FIELDS)


def _session(directory: Path, poses: list[int], *, run_id: str = "abc123",
             pose_list: str | None = "config/pose_lists/hard_test.csv",
             success: str = "0", stack_on: dict[int, str] | None = None) -> Path:
    """A trial CSV (+ its run metadata) as 03_run_experiment.py writes them."""
    rows = [{"trial_id": str(i + 1), "success": success, "human_ok": "",
             "class_name": "", "failure_reason": "detection_miss" if success == "0" else "",
             "lighting": "dim", "det_x_mm": "", "det_y_mm": "", "pose_id": f"P{n:04d}",
             "condition": "novel_background_dim",
             "stack_on": (stack_on or {}).get(n, "metal_box" if n == 223 else ""),
             "run_id": run_id}
            for i, n in enumerate(poses)]
    path = _write_csv(directory / "experiment_real_20261201_100000.csv", rows, list(rows[0]))
    if pose_list is not None:
        metadata = {"run_id": run_id, "provenance": {"runtime": {
            "pose_list": pose_list, "pose_slice": "220:300"}}}
        (directory / f"{path.stem}.{run_id}.metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8")
    return path


class TestStageTrials:
    def test_join_uses_the_metadata_pose_list_and_records_hashes(self, tmp_path):
        poses_path = _pose_list(tmp_path / "root/config/pose_lists/hard_test.csv")
        csv_path = _session(tmp_path / "session", [221, 222, 223])
        rows, inputs = stage_trials([csv_path], pose_root=tmp_path / "root")
        assert [r["staged_class"] for r in rows] == ["inox_box", "carton", "inox_box"]
        assert rows[2]["stack_on"] == "metal_box" and rows[0]["staged_x_mm"] == 575.0
        assert inputs[0]["sha256"] == file_sha256(csv_path)
        assert inputs[0]["pose_list"]["sha256"] == file_sha256(poses_path)
        assert inputs[0]["metadata"]["pose_slice"] == "220:300"
        assert inputs[0]["pose_ids"] == ["P0221", "P0223"]
        report = bin_failures(rows, region=(510.0, 640.0, -175.0, 215.0))
        assert {c["class_name"] for c in report["clusters"]} == {"inox_box", "carton"}

    def test_evaluation_and_drift_poses_are_refused_unless_overridden(self, tmp_path):
        _pose_list(tmp_path / "root/config/pose_lists/hard_test.csv")
        csv_path = _session(tmp_path / "session", [200, 221])
        with pytest.raises(ValueError, match="outside poses 221-300"):
            stage_trials([csv_path], pose_root=tmp_path / "root")
        rows, _ = stage_trials([csv_path], pose_root=tmp_path / "root", poses=(1, 300))
        assert len(rows) == 2

    def test_pose_list_override_must_have_the_named_file(self, tmp_path):
        _pose_list(tmp_path / "elsewhere/hard_test.csv")
        other = _pose_list(tmp_path / "elsewhere/hard_v1.csv", x_mm=317.0)
        csv_path = _session(tmp_path / "session", [221])
        with pytest.raises(ValueError, match="names the pose list"):
            stage_trials([csv_path], pose_root=tmp_path, pose_list=other)
        rows, _ = stage_trials([csv_path], pose_root=tmp_path,
                               pose_list=tmp_path / "elsewhere/hard_test.csv")
        assert rows[0]["staged_class"] == "inox_box"

    def test_logged_support_must_match_the_pose_list(self, tmp_path):
        _pose_list(tmp_path / "root/config/pose_lists/hard_test.csv")
        csv_path = _session(tmp_path / "session", [221], stack_on={221: "carton"})
        with pytest.raises(ValueError, match="wrong pose list"):
            stage_trials([csv_path], pose_root=tmp_path / "root")

    def test_csv_without_metadata_needs_an_explicit_pose_list(self, tmp_path):
        poses_path = _pose_list(tmp_path / "hard_test.csv")
        csv_path = _session(tmp_path / "session", [221], pose_list=None)
        with pytest.raises(ValueError, match="no run metadata"):
            stage_trials([csv_path], pose_root=tmp_path)
        rows, inputs = stage_trials([csv_path], pose_root=tmp_path, pose_list=poses_path)
        assert rows[0]["staged_class"] == "inox_box" and inputs[0]["metadata"] is None

    def test_unknown_pose_id_is_refused(self, tmp_path):
        _pose_list(tmp_path / "root/config/pose_lists/hard_test.csv")
        csv_path = _session(tmp_path / "session", [301])
        with pytest.raises(ValueError, match="not in hard_test.csv"):
            stage_trials([csv_path], pose_root=tmp_path / "root", poses=(1, 400))


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


class TestSeedRanges:
    def test_streams_are_seed_plus_index(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"spec_version": 1, "seed": 0, "mode": "anchored", "kappa": 2.0,
                                        "files": [f"scene_{i:06d}.json" for i in range(3000)]}))
        e3 = manifest_streams(manifest)
        assert e3["streams"] == [[0, 3000]] and e3["kappa"] == 2.0
        # Seed 2000 reuses E3 scenes 2000..2999; the guide's 20000 does not.
        assert shared_streams(planned_streams(2000, 1000), e3["streams"]) == 1000
        assert shared_streams(planned_streams(20000, 1000), e3["streams"]) == 0
        assert overlaps({"e3": e3["streams"], "guided": planned_streams(1000, 1000),
                         "control": planned_streams(20000, 1000)}) == [("e3", "guided", 1000)]

    def test_unknown_spec_version_is_reported_unchecked(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"spec_version": 99, "seed": 0, "files": ["scene_000000.json"]}))
        info = manifest_streams(manifest)
        assert info["streams"] is None and "spec_version 99" in info["unchecked"]


def _load_cli():
    path = Path(__file__).resolve().parents[1] / "scripts/21_mine_failures.py"
    spec = importlib.util.spec_from_file_location("mine_failures_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCli:
    def _args(self, tmp_path, csv_path, *extra):
        return ["--runs", str(csv_path), "--pose-root", str(tmp_path / "root"),
                "--used-specs", str(tmp_path / "synth/*/specs/manifest.json"),
                "--out", str(tmp_path / "out/failure_modes.json"), *extra]

    def test_report_is_written_when_no_trial_failed(self, tmp_path):
        _pose_list(tmp_path / "root/config/pose_lists/hard_test.csv")
        csv_path = _session(tmp_path / "session", [221, 222], success="1")
        assert _load_cli().main(self._args(tmp_path, csv_path)) == 0
        report = load_report(tmp_path / "out/failure_modes.json")
        assert report["n_failures"] == 0 and report["allocation"] == []
        assert report["update"].startswith("none")
        assert report["inputs"][0]["sha256"] == file_sha256(csv_path)
        assert report["poses"] == [221, 300] and report["pose_range_override"] is False

    def test_evaluation_trials_are_refused_without_a_report(self, tmp_path):
        _pose_list(tmp_path / "root/config/pose_lists/hard_test.csv")
        csv_path = _session(tmp_path / "session", [1, 2])
        assert _load_cli().main(self._args(tmp_path, csv_path)) == 2
        assert not (tmp_path / "out").exists()
        assert _load_cli().main(self._args(tmp_path, csv_path, "--poses", "1", "200")) == 0
        assert load_report(tmp_path / "out/failure_modes.json")["pose_range_override"] is True

    def test_planned_seed_ranges_are_checked_against_written_batches(self, tmp_path):
        _pose_list(tmp_path / "root/config/pose_lists/hard_test.csv")
        csv_path = _session(tmp_path / "session", [221, 223])
        specs = tmp_path / "synth/anchored_k2/specs"
        specs.mkdir(parents=True)
        (specs / "manifest.json").write_text(json.dumps(
            {"spec_version": 1, "seed": 0, "files": [f"scene_{i:06d}.json" for i in range(3000)]}))
        cli = _load_cli()
        for seeds in (["--guided-seed", "1000", "--control-seed", "20000"],
                      ["--guided-seed", "10000", "--control-seed", "10500"]):
            assert cli.main(self._args(tmp_path, csv_path, *seeds)) == 2
        assert not (tmp_path / "out").exists()
        assert cli.main(self._args(tmp_path, csv_path, "--guided-seed", "10000",
                                   "--control-seed", "20000")) == 0
        report = load_report(tmp_path / "out/failure_modes.json")
        plan = report["generation_plan"]
        assert plan["planned_streams"] == {"guided": [[10000, 11000]], "control": [[20000, 21000]]}
        assert plan["checked_against"][0]["streams"] == [[0, 3000]]
        assert sum(e["n_images"] for e in report["allocation"]) == 1000
        assert {e["condition"]["class_name"] for e in report["allocation"]} == {"inox_box"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
