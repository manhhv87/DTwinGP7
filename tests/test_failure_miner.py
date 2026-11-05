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
import itertools
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
from src.synthgen.scene_sampler import SPEC_VERSION
from src.synthgen.seed_ranges import (
    STREAM_RULE,
    manifest_streams,
    overlaps,
    planned_streams,
    scene_streams,
    shared_streams,
)

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


def _specs(path: Path, spec_version: int, seed: int, indices) -> Path:
    """specs/manifest.json of a written batch, as SceneSampler.write_specs lays it out."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"spec_version": spec_version, "seed": seed, "mode": "anchored",
                                "kappa": 2.0, "files": [f"scene_{i:06d}.json" for i in indices]}))
    return path


class TestSeedRanges:
    def test_version_1_streams_are_seed_plus_index(self, tmp_path):
        e3 = manifest_streams(_specs(tmp_path / "manifest.json", 1, 0, range(3000)))
        assert e3["streams"] == [[[0, 3000], [0, 1]]] and e3["kappa"] == 2.0
        # Seed 2000 reused E3 scenes 2000..2999; 20000 did not.
        assert shared_streams(planned_streams(2000, 1000, spec_version=1), e3["streams"]) == 1000
        assert shared_streams(planned_streams(20000, 1000, spec_version=1), e3["streams"]) == 0
        assert overlaps({"e3": e3["streams"], "guided": planned_streams(1000, 1000, spec_version=1),
                         "control": planned_streams(20000, 1000, spec_version=1)}) == [("e3", "guided", 1000)]

    def test_version_2_streams_are_seed_and_index(self, tmp_path):
        e3 = manifest_streams(_specs(tmp_path / "manifest.json", 2, 0, range(3000)))
        assert e3["streams"] == [[[0, 1], [0, 3000]]]
        # Another seed shares nothing whatever its indices; one seed shares the
        # indices its batches have in common.
        assert shared_streams(planned_streams(2000, 1000, spec_version=2), e3["streams"]) == 0
        assert shared_streams(planned_streams(1, 3000, spec_version=2), e3["streams"]) == 0
        assert shared_streams(planned_streams(0, 1000, 2500, spec_version=2), e3["streams"]) == 500
        assert overlaps({"e3": e3["streams"], "guided": planned_streams(10000, 1000, spec_version=2),
                         "control": planned_streams(10000, 1000, spec_version=2)}) \
            == [("guided", "control", 1000)]

    def test_index_0_of_a_version_2_seed_is_a_version_1_stream(self, tmp_path):
        old = manifest_streams(_specs(tmp_path / "manifest.json", 1, 0, range(3000)))
        assert shared_streams(planned_streams(1000, 1000, spec_version=2), old["streams"]) == 1
        assert shared_streams(planned_streams(1000, 999, 1, spec_version=2), old["streams"]) == 0
        assert shared_streams(planned_streams(10000, 1000, spec_version=2), old["streams"]) == 0

    def test_rules_share_a_stream_exactly_when_numpy_seeds_alike(self):
        seeded = {1: lambda seed, i: np.random.SeedSequence(seed + i),
                  2: lambda seed, i: np.random.SeedSequence([seed, i])}
        scenes = [(1, 0, 1000), (1, 1, 999), (1, 5, 0), (1, 0, 0), (2, 1000, 0), (2, 0, 1000),
                  (2, 1, 999), (2, 5, 0), (2, 4, 1), (2, 0, 5), (2, 0, 0)]
        alike = 0
        for a, b in itertools.combinations(scenes, 2):
            same = np.array_equal(seeded[a[0]](*a[1:]).generate_state(4),
                                  seeded[b[0]](*b[1:]).generate_state(4))
            shared = shared_streams(scene_streams(a[0], a[1], [a[2]]),
                                    scene_streams(b[0], b[1], [b[2]]))
            assert shared == int(same), (a, b)
            alike += same
        assert alike == 5   # stream 1000 three ways (3 pairs); 5 and 0 across versions

    def test_version_3_adds_an_exposure_stream_no_scene_shares(self, tmp_path):
        # Version 3 keeps version 2's scene stream and adds SeedSequence([seed, i, 2])
        # for the exposure gain: the overlap rule is version 2's, so a version 2 and a
        # version 3 batch of one seed share the indices they have in common.
        v3 = manifest_streams(_specs(tmp_path / "manifest.json", 3, 0, range(3000)))
        assert v3["streams"] == [[[0, 1], [0, 3000]]] and 3 in STREAM_RULE
        assert shared_streams(planned_streams(0, 1000, spec_version=2), v3["streams"]) == 1000
        assert shared_streams(planned_streams(1, 1000, spec_version=3), v3["streams"]) == 0
        # The third word keeps the exposure stream apart from every two-word stream
        # (SeedSequence pads with zero words) and from the pose arm's word 1.
        states = {key: np.random.SeedSequence(list(key)).generate_state(4).tobytes()
                  for key in ((0, 5), (0, 5, 0), (0, 5, 1), (0, 5, 2), (5, 2), (2, 5))}
        assert states[(0, 5)] == states[(0, 5, 0)]
        assert len({states[k] for k in ((0, 5), (0, 5, 1), (0, 5, 2), (5, 2), (2, 5))}) == 5

    def test_unknown_spec_version_or_split_seed_is_reported_unchecked(self, tmp_path):
        info = manifest_streams(_specs(tmp_path / "a.json", 99, 0, [0]))
        assert info["streams"] is None and "spec_version 99" in info["unchecked"]
        info = manifest_streams(_specs(tmp_path / "b.json", 2, 2 ** 32 + 5, [0]))
        assert info["streams"] is None and "32-bit" in info["unchecked"]
        with pytest.raises(ValueError, match="32-bit"):
            planned_streams(2 ** 32, 10, spec_version=2)

    def test_the_sampler_spec_version_has_a_rule(self):
        assert SPEC_VERSION in STREAM_RULE


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
        _specs(tmp_path / "synth/anchored_k2/specs/manifest.json", SPEC_VERSION, 0, range(3000))
        _specs(tmp_path / "synth/loop_k1/specs/manifest.json", SPEC_VERSION, 10000, range(1000))
        cli = _load_cli()
        # Update 1's guided seed again, then one seed for both arms.
        for seeds in (["--guided-seed", "10000", "--control-seed", "40000"],
                      ["--guided-seed", "30000", "--control-seed", "30000"]):
            assert cli.main(self._args(tmp_path, csv_path, *seeds)) == 2
        assert not (tmp_path / "out").exists()
        assert cli.main(self._args(tmp_path, csv_path, "--guided-seed", "30000",
                                   "--control-seed", "40000")) == 0
        report = load_report(tmp_path / "out/failure_modes.json")
        plan = report["generation_plan"]
        assert plan["spec_version"] == SPEC_VERSION and plan["rule"] == STREAM_RULE[SPEC_VERSION]
        assert plan["planned_streams"] == {"guided": [[[30000, 30001], [0, 1000]]],
                                           "control": [[[40000, 40001], [0, 1000]]]}
        assert [c["streams"] for c in plan["checked_against"]] == [
            [[[0, 1], [0, 3000]]], [[[10000, 10001], [0, 1000]]]]
        assert sum(e["n_images"] for e in report["allocation"]) == 1000
        assert {e["condition"]["class_name"] for e in report["allocation"]} == {"inox_box"}


def _load_generator():
    path = Path(__file__).resolve().parents[1] / "scripts/20_generate_synth.py"
    spec = importlib.util.spec_from_file_location("generate_synth_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestGeneratorUsesTheCheckedPlan:
    """20_generate_synth.py writes, for each E4 arm, the streams the miner checked."""

    def _report(self, tmp_path, *seeds) -> Path:
        _pose_list(tmp_path / "root/config/pose_lists/hard_test.csv")
        csv_path = _session(tmp_path / "session", [221, 223])
        report = tmp_path / "failure_modes.json"
        assert _load_cli().main(["--runs", str(csv_path), "--pose-root", str(tmp_path / "root"),
                                 "--used-specs", str(tmp_path / "synth/*/specs/manifest.json"),
                                 "--n-budget", "4", "--out", str(report), *seeds]) == 0
        return report

    def _generate(self, tmp_path, flag, report, name, *extra):
        calib = tmp_path / "T_BC.npy"
        np.save(calib, np.array([[1., 0., 0., 419.5], [0., -1., 0., 0.],
                                 [0., 0., -1., 1200.], [0., 0., 0., 1.]]))
        out = tmp_path / "synth" / name
        code = _load_generator().main([
            flag, str(report), "--out", str(out),
            "--config", str(Path(__file__).resolve().parents[1] / "config/synthgen.yaml"),
            "--calib", str(calib), "--sigma", str(tmp_path / "absent.json"),
            "--allow-sigma-fallback", "--kappa", "1", *extra])
        manifest = out / "specs/manifest.json"
        return code, manifest if manifest.exists() else None

    def test_each_arm_writes_the_streams_the_miner_checked(self, tmp_path):
        report = self._report(tmp_path, "--guided-seed", "10000", "--control-seed", "20000")
        plan = load_report(report)["generation_plan"]
        code, loop = self._generate(tmp_path, "--from-failures", report, "loop_k1")
        assert code == 0
        code, control = self._generate(tmp_path, "--control-for", report, "control_k1",
                                       "--seed", "20000", "--n", "4")
        assert code == 0
        for arm, manifest in (("guided", loop), ("control", control)):
            assert manifest_streams(manifest)["streams"] == plan["planned_streams"][arm]
        # The control renders the session's lighting, as every guided bin does, so
        # both arms draw the same light and exposure distributions.
        written = json.loads(control.read_text())
        assert written["mode"] == "anchored"
        assert written["condition"] == {"class_name": None, "region": None, "lighting": "dim"}
        guided = json.loads(loop.read_text())
        assert guided["condition"]["class_name"] == "inox_box"
        assert {b["condition"]["lighting"] for b in guided.get("batches", [guided])} == {"dim"}

    @pytest.mark.parametrize("flag", ["--from-failures", "--control-for"])
    def test_a_lighting_label_the_profile_does_not_render_is_refused(self, tmp_path, flag):
        report = self._report(tmp_path, "--guided-seed", "10000", "--control-seed", "20000")
        data = load_report(report)
        data["allocation"][0]["condition"]["lighting"] = "bright"
        report.write_text(json.dumps(data))
        assert self._generate(tmp_path, flag, report, "refused") == (1, None)

    @pytest.mark.parametrize("flag, extra", [
        ("--from-failures", ["--seed", "10001"]),
        ("--from-failures", ["--n", "5"]),
        ("--control-for", ["--seed", "10000"]),
        ("--control-for", ["--n", "1000"]),
    ])
    def test_a_seed_or_count_the_miner_did_not_check_is_refused(self, tmp_path, flag, extra):
        report = self._report(tmp_path, "--guided-seed", "10000", "--control-seed", "20000")
        assert self._generate(tmp_path, flag, report, "refused", *extra) == (1, None)

    def test_a_report_without_the_arms_checked_seed_is_refused(self, tmp_path):
        report = self._report(tmp_path, "--guided-seed", "10000")
        assert self._generate(tmp_path, "--control-for", report, "control_k1") == (1, None)
        unchecked = self._report(tmp_path)
        assert load_report(unchecked)["generation_plan"] is None
        assert self._generate(tmp_path, "--from-failures", unchecked, "loop_k1") == (1, None)

    def test_a_plan_checked_under_another_spec_version_is_refused(self, tmp_path):
        report = self._report(tmp_path, "--guided-seed", "10000", "--control-seed", "20000")
        data = load_report(report)
        data["generation_plan"]["spec_version"] = 1
        report.write_text(json.dumps(data))
        for flag in ("--from-failures", "--control-for"):
            assert self._generate(tmp_path, flag, report, "old") == (1, None)

    def test_the_control_arm_is_one_unablated_anchored_batch(self):
        cli = _load_generator()
        for extra in (["--from-failures", "f.json"], ["--mode", "blind"], ["--ablation", "camera"]):
            with pytest.raises(SystemExit) as exc:
                cli.parse_args(["--out", "unused", "--control-for", "f.json", *extra])
            assert exc.value.code == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
