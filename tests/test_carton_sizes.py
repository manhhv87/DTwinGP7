"""The carton size of every carton slot, fixed per pose in a committed sidecar.

In session 1 of E2 the pose list named the class only and the arms of seven poses
got different cartons. scripts/28_make_carton_sizes.py writes <list>_carton.csv with
a fixed seed; the committed files must be what it writes, cover every carton target
and support, and split the two sizes evenly.

Run:
    pytest tests/test_carton_sizes.py -v
"""
from __future__ import annotations

import csv
import importlib.util
from collections import Counter
from pathlib import Path

import pytest

from src.orchestrator.placement_check import (
    carton_sizes_path, load_carton_sizes, missing_carton_sizes,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LISTS = ["config/pose_lists/std_v2.csv", "config/pose_lists/hard_v2.csv"]


def _script():
    path = PROJECT_ROOT / "scripts" / "28_make_carton_sizes.py"
    spec = importlib.util.spec_from_file_location("carton_sizes_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(pose_list):
    with (PROJECT_ROOT / pose_list).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("pose_list", LISTS)
def test_the_committed_sidecar_is_what_the_script_writes(pose_list, tmp_path):
    mod = _script()
    rows = _rows(pose_list)
    out = tmp_path / "x_carton.csv"
    mod.write_sidecar(rows, mod.assign_sizes(rows), out)
    committed = carton_sizes_path(PROJECT_ROOT / pose_list)
    assert out.read_text(encoding="utf-8") == committed.read_text(encoding="utf-8")


@pytest.mark.parametrize("pose_list", LISTS)
def test_every_carton_slot_has_a_size(pose_list):
    rows = _rows(pose_list)
    sizes = load_carton_sizes(carton_sizes_path(PROJECT_ROOT / pose_list))
    assert missing_carton_sizes(rows, sizes) == []
    carton_rows = {r["pose_id"] for r in rows
                   if r["class_hint"] == "carton" or r["stack_on"] == "carton"}
    assert set(sizes) == carton_rows


@pytest.mark.parametrize("pose_list", LISTS)
def test_the_sizes_are_balanced_per_sitting_and_role(pose_list):
    rows = _rows(pose_list)
    sizes = load_carton_sizes(carton_sizes_path(PROJECT_ROOT / pose_list))
    total = Counter()
    for block in range(0, len(rows), 100):
        for role in ("alone", "on_support", "support"):
            c = Counter()
            for r in rows[block:block + 100]:
                target, support = sizes.get(r["pose_id"], (None, None))
                if role == "alone" and r["class_hint"] == "carton" and not r["stack_on"]:
                    c[target] += 1
                elif role == "on_support" and r["class_hint"] == "carton" and r["stack_on"]:
                    c[target] += 1
                elif role == "support" and r["stack_on"] == "carton":
                    c[support] += 1
            assert abs(c[80] - c[120]) <= 1, (block, role, c)
            total.update(c)
    assert abs(total[80] - total[120]) <= 1 and set(total) == {80, 120}


def test_the_seed_fixes_the_assignment():
    mod = _script()
    rows = _rows(LISTS[0])
    assert mod.assign_sizes(rows, 7) == mod.assign_sizes(rows, 7)
    assert mod.assign_sizes(rows, 7) != mod.assign_sizes(rows, 8)
