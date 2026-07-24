"""
test_dataset_check.py
─────────────────────
Unit tests for the dataset verifier/remapper — the guard against the
Roboflow alphabetical-class-order trap (bolt,bottle,cup,tray vs the
canonical tray,bottle,cup,bolt).

Run:
    pytest tests/test_dataset_check.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="tiny PNGs are written via cv2")

from src.utils.dataset_check import (  # noqa: E402
    EXPECTED_NAMES,
    load_names,
    parse_label_file,
    remap_labels,
    verify_dataset,
)

# Roboflow-style alphabetical order of the canonical set — the trap this module catches.
# canonical = [carton(0), plastic_box(1), wood_box(2), metal_box(3)];
# sorted() → [carton, metal_box, plastic_box, wood_box].
ALPHA_NAMES = ["carton", "metal_box", "plastic_box", "wood_box"]


def _write_png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((8, 8, 3), np.uint8))


def _make_dataset(root, names, label_lines):
    """Minimal ultralytics layout: 2 train images (1 labelled, 1 negative),
    1 val image; data.yaml in Roboflow dict style."""
    _write_png(root / "images" / "train" / "img_a.png")
    _write_png(root / "images" / "train" / "img_neg.png")
    _write_png(root / "images" / "val" / "img_b.png")
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train" / "img_a.txt").write_text(
        "\n".join(label_lines) + "\n", encoding="utf-8")
    (root / "labels" / "train" / "img_neg.txt").write_text("", encoding="utf-8")
    (root / "labels" / "val" / "img_b.txt").write_text(
        "1 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)),
        encoding="utf-8")


class TestLoadNames:
    def test_dict_and_list_forms(self, tmp_path):
        (tmp_path / "a.yaml").write_text("names:\n  0: tray\n  1: bolt\n",
                                         encoding="utf-8")
        (tmp_path / "b.yaml").write_text("names: [tray, bolt]\n", encoding="utf-8")
        assert load_names(tmp_path / "a.yaml") == ["tray", "bolt"]
        assert load_names(tmp_path / "b.yaml") == ["tray", "bolt"]

    def test_missing_names_raises(self, tmp_path):
        (tmp_path / "c.yaml").write_text("train: images/train\n", encoding="utf-8")
        with pytest.raises(ValueError, match="names"):
            load_names(tmp_path / "c.yaml")


class TestParseLabelFile:
    def test_valid_and_invalid_lines(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text(
            "0 0.1 0.1 0.2 0.1 0.2 0.2\n"       # valid triangle
            "9 0.1 0.1 0.2 0.1 0.2 0.2\n"       # id out of range
            "1 0.1 0.1 5.0 0.1 0.2 0.2\n"       # coord not normalized
            "2 0.1 0.1 0.2 0.1\n",              # too few points
            encoding="utf-8")
        ids, problems = parse_label_file(f, n_classes=4)
        assert ids == [0]
        assert len(problems) == 3

    def test_empty_file_is_negative(self, tmp_path):
        f = tmp_path / "neg.txt"
        f.write_text("", encoding="utf-8")
        assert parse_label_file(f, n_classes=4) == ([], [])


class TestVerify:
    def test_alphabetical_order_flagged(self, tmp_path):
        _make_dataset(tmp_path, ALPHA_NAMES,
                      ["0 0.1 0.1 0.2 0.1 0.2 0.2",      # 'carton' in alpha order
                       "3 0.5 0.5 0.6 0.5 0.6 0.6"])     # 'wood_box' in alpha order
        report = verify_dataset(tmp_path)
        assert report["names_match"] is False
        assert report["ok"] is False
        assert any("canonical order" in e for e in report["errors"])
        # Counts are keyed by NAME → correct even under the wrong order.
        assert report["class_counts"]["carton"] == 1
        assert report["class_counts"]["wood_box"] == 1
        assert report["splits"]["train"]["negatives"] == 1

    def test_leakage_detected(self, tmp_path):
        _make_dataset(tmp_path, EXPECTED_NAMES, ["0 0.1 0.1 0.2 0.1 0.2 0.2"])
        _write_png(tmp_path / "images" / "val" / "img_a.png")   # same stem as train
        report = verify_dataset(tmp_path)
        assert any("LEAKAGE" in e for e in report["errors"])

    def test_clean_dataset_ok(self, tmp_path):
        _make_dataset(tmp_path, EXPECTED_NAMES,
                      ["0 0.1 0.1 0.2 0.1 0.2 0.2",
                       "3 0.5 0.5 0.6 0.5 0.6 0.6"])
        report = verify_dataset(tmp_path)
        assert report["ok"] is True and report["names_match"] is True


class TestRemap:
    def test_remap_fixes_ids_and_yaml(self, tmp_path):
        # In alpha order: id0=carton id1=metal_box id2=plastic_box id3=wood_box.
        _make_dataset(tmp_path, ALPHA_NAMES,
                      ["0 0.1 0.1 0.2 0.1 0.2 0.2",      # carton → canonical 0
                       "3 0.5 0.5 0.6 0.5 0.6 0.6"])     # wood_box → canonical 2
        result = remap_labels(tmp_path)
        assert result["noop"] is False
        assert result["mapping"] == {"carton": 0, "metal_box": 3,
                                     "plastic_box": 1, "wood_box": 2}
        assert (tmp_path / "labels_orig").exists()        # backup created

        report = verify_dataset(tmp_path)                 # reads new dataset.yaml
        assert report["names_match"] is True and report["ok"] is True
        assert report["class_counts"]["wood_box"] == 1    # still 1 wood_box, new id
        first_ids = [line.split()[0] for line in
                     (tmp_path / "labels" / "train" / "img_a.txt")
                     .read_text(encoding="utf-8").strip().splitlines()]
        assert first_ids == ["0", "2"]

    def test_remap_noop_when_canonical(self, tmp_path):
        _make_dataset(tmp_path, EXPECTED_NAMES, ["0 0.1 0.1 0.2 0.1 0.2 0.2"])
        result = remap_labels(tmp_path)
        assert result["noop"] is True
        assert verify_dataset(tmp_path)["ok"] is True

    def test_different_name_sets_raise(self, tmp_path):
        _make_dataset(tmp_path, ["carton", "plastic_box", "wood_box", "widget"],
                      ["0 0.1 0.1 0.2 0.1 0.2 0.2"])
        with pytest.raises(ValueError, match="name sets differ"):
            remap_labels(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
