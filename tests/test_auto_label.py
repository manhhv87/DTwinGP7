"""
test_auto_label.py
──────────────────
Unit tests for instance-map → YOLOv8-seg label conversion.

Run:
    pytest tests/test_auto_label.py -v
"""
from __future__ import annotations

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="auto_label needs opencv-python")

from src.synthgen import (  # noqa: E402
    convert_render_output,
    instance_map_to_yolo_lines,
    mask_to_polygon,
)


def _rect_mask(h=720, w=1280, box=(100, 200, 400, 500)):
    m = np.zeros((h, w), np.uint8)
    x1, y1, x2, y2 = box
    m[y1:y2, x1:x2] = 1
    return m


class TestMaskToPolygon:
    def test_rectangle_roundtrip_iou(self):
        mask = _rect_mask()
        poly = mask_to_polygon(mask)
        assert poly is not None and len(poly) >= 6
        assert all(0.0 <= v <= 1.0 for v in poly)

        # Denormalize + refill → must reproduce the original mask.
        h, w = mask.shape
        pts = np.asarray(poly, dtype=float).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        refill = np.zeros_like(mask)
        cv2.fillPoly(refill, [pts.round().astype(np.int32)], 1)
        inter = np.logical_and(mask, refill).sum()
        union = np.logical_or(mask, refill).sum()
        assert inter / union > 0.95

    def test_tiny_mask_dropped(self):
        m = np.zeros((100, 100), np.uint8)
        m[50:53, 50:53] = 1                      # 9 px < MIN_INSTANCE_PIXELS
        assert mask_to_polygon(m) is None

    def test_empty_mask_dropped(self):
        assert mask_to_polygon(np.zeros((50, 50), np.uint8)) is None

    @staticmethod
    def _refill(poly, shape):
        h, w = shape
        pts = np.asarray(poly, dtype=float).reshape(-1, 2) * [w, h]
        refill = np.zeros(shape, np.uint8)
        cv2.fillPoly(refill, [pts.round().astype(np.int32)], 1)
        return refill

    @staticmethod
    def _bbox(m):
        ys, xs = np.nonzero(m)
        return xs.min(), ys.min(), xs.max(), ys.max()

    def test_box_under_a_turned_box_keeps_every_part(self):
        # The visible part of a box under a stacked box, turned 40 deg against it:
        # the top box cuts the lower one into pieces. All of them belong to the label.
        h, w = 720, 1280
        lower = np.zeros((h, w), np.uint8)
        cv2.rectangle(lower, (400, 250), (800, 450), 1, -1)
        top = np.zeros((h, w), np.uint8)
        rect = ((600.0, 350.0), (420.0, 120.0), 40.0)
        cv2.fillPoly(top, [cv2.boxPoints(rect).round().astype(np.int32)], 1)
        visible = lower & (1 - top)
        n_parts = cv2.connectedComponents(visible, connectivity=8)[0] - 1
        assert n_parts >= 2                      # the case the rule is for

        poly = mask_to_polygon(visible)
        refill = self._refill(poly, (h, w))
        lost = (visible & (1 - refill)).sum() / visible.sum()
        assert lost <= 1e-3
        assert self._bbox(refill) == self._bbox(visible)

    def test_parts_far_apart_are_one_polygon(self):
        m = np.zeros((200, 300), np.uint8)
        m[20:60, 20:60] = 1
        m[120:180, 200:280] = 1
        m[150:155, 30:35] = 1                    # small third part, 25 px
        poly = mask_to_polygon(m)
        refill = self._refill(poly, m.shape)
        assert (m & (1 - refill)).sum() == 0     # nothing lost
        # extra pixels: only the one-pixel bridges between the parts
        assert ((refill > 0) & (m == 0)).sum() < 0.05 * m.sum()

    def test_order_of_parts_does_not_matter(self):
        from src.synthgen.auto_label import _join_contours
        a = np.zeros((100, 200), np.uint8)
        a[10:40, 10:40] = 1
        a[50:90, 120:190] = 1
        a[60:70, 20:30] = 1
        cs, _ = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(cs) == 3
        assert np.array_equal(_join_contours(list(cs)), _join_contours(list(cs)[::-1]))


class TestInstanceMapToYoloLines:
    def test_labels_classes_and_skips(self):
        inst = np.zeros((720, 1280), np.uint16)
        inst[100:300, 100:400] = 3               # workpiece, class 1
        inst[400:600, 600:900] = 7               # workpiece, class 2
        inst[10:13, 10:13] = 9                   # tiny → dropped
        inst[650:700, 1000:1100] = 11            # distractor: NOT in mapping

        lines = instance_map_to_yolo_lines(inst, {3: 1, 7: 2, 9: 0})
        classes = sorted(int(line.split()[0]) for line in lines)
        assert classes == [1, 2]
        for line in lines:
            coords = [float(v) for v in line.split()[1:]]
            assert len(coords) >= 6 and len(coords) % 2 == 0
            assert all(0.0 <= v <= 1.0 for v in coords)


class TestConvertRenderOutput:
    def _write_frame(self, render_dir, idx, with_meta=True):
        rgb = np.random.default_rng(idx).integers(
            0, 255, size=(64, 64, 3), dtype=np.uint8)
        inst = np.zeros((64, 64), np.uint16)
        inst[8:40, 8:40] = 5
        stem = f"frame_{idx:06d}"
        cv2.imwrite(str(render_dir / f"{stem}_rgb.png"), rgb)
        cv2.imwrite(str(render_dir / f"{stem}_inst.png"), inst)
        if with_meta:
            (render_dir / f"{stem}_meta.json").write_text(
                json.dumps({"instances": {"5": 2}, "scene": f"scene_{idx:06d}.json"}),
                encoding="utf-8")

    def test_dataset_layout(self, tmp_path):
        render = tmp_path / "render"
        render.mkdir()
        for i in range(5):
            self._write_frame(render, i)

        out = tmp_path / "dataset"
        stats = convert_render_output(render, out, val_frac=0.2, seed=0)
        assert stats["frames"] == 5 and stats["skipped"] == 0
        assert stats["train"] == 4 and stats["val"] == 1
        assert stats["multipart"] == 0
        assert (out / "dataset.yaml").exists()

        label_files = list((out / "labels" / "train").glob("*.txt")) + \
            list((out / "labels" / "val").glob("*.txt"))
        assert len(label_files) == 5
        for lf in label_files:
            first = lf.read_text(encoding="utf-8").strip().split("\n")[0]
            assert first.startswith("2 ")        # class id from meta

    def test_multipart_instances_counted(self, tmp_path):
        render = tmp_path / "render"
        render.mkdir()
        self._write_frame(render, 0)
        inst = cv2.imread(str(render / "frame_000000_inst.png"), cv2.IMREAD_UNCHANGED)
        inst[20:24, 8:40] = 0                    # cut instance 5 in two
        cv2.imwrite(str(render / "frame_000000_inst.png"), inst)
        stats = convert_render_output(render, tmp_path / "ds", val_frac=0.0, seed=0)
        assert stats["multipart"] == 1
        line = (tmp_path / "ds" / "labels" / "train" / "frame_000000.txt").read_text().split("\n")[0]
        ys = [float(v) for v in line.split()[2::2]]
        assert min(ys) < 12 / 64 and max(ys) > 36 / 64   # both parts in the one polygon

    def test_missing_meta_skipped(self, tmp_path):
        render = tmp_path / "render"
        render.mkdir()
        self._write_frame(render, 0)
        self._write_frame(render, 1, with_meta=False)
        stats = convert_render_output(render, tmp_path / "ds", val_frac=0.0, seed=0)
        assert stats["frames"] == 1 and stats["skipped"] == 1

    def test_empty_render_dir_raises(self, tmp_path):
        (tmp_path / "render").mkdir()
        with pytest.raises(FileNotFoundError, match="No .*rgb"):
            convert_render_output(tmp_path / "render", tmp_path / "ds")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
