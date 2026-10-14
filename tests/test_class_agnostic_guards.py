"""Motion guards must read perception and the class table, never the pose list."""
import numpy as np
import pytest

from src.orchestrator.orchestrator import Orchestrator

SIZES = {
    "carton": [[180, 120, 120], [180, 100, 80]],
    "plastic_box": [[180, 140, 30]],
    "inox_box": [[180, 85, 40]],
}


def _orch(**cfg):
    o = Orchestrator.__new__(Orchestrator)
    o.config = {"class_sizes_mm": SIZES,
                "class_heights_mm": {"carton": 80.0, "plastic_box": 30.0,
                                     "inox_box": 40.0},
                **cfg}
    return o


class TestClassGeometry:
    def test_min_height_is_the_lowest_variant(self):
        assert _orch()._class_min_height_mm("carton") == 80.0
        assert _orch()._class_min_height_mm("inox_box") == 40.0

    def test_unknown_class_has_no_height(self):
        assert _orch()._class_min_height_mm("teapot") is None

    def test_measured_rectangle_matches_a_variant(self):
        o = _orch()
        assert o._size_matches_class("carton", 182.0, 118.0, 40.0)
        assert o._size_matches_class("carton", 178.0, 98.0, 40.0)   # small variant

    def test_rectangle_far_from_every_variant_is_rejected(self):
        assert not _orch()._size_matches_class("inox_box", 400.0, 400.0, 40.0)

    def test_unknown_class_cannot_be_contradicted(self):
        assert _orch()._size_matches_class("teapot", 1.0, 1.0, 5.0)


class TestCarriedHeight:
    """The height carried away comes from the detected class, not the pose list."""

    def _ready(self, total_mm, class_name, stack_on):
        o = _orch()
        o.current_stack_on = stack_on
        o._world_to_robot_base = lambda T: T
        o._table_z_base_at_xy = lambda x, y: 0.0
        return o._detected_target_height_mm(
            {"pose_base": (500.0, 0.0, total_mm), "class_name": class_name})

    def test_part_on_the_table(self):
        h, support, top, table = self._ready(120.0, "carton", "")
        assert h == 120.0 and support == 0.0

    def test_part_on_a_support_leaves_the_remainder(self):
        h, support, _, _ = self._ready(150.0, "carton", "")
        assert h == 120.0 and support == pytest.approx(30.0)

    def test_result_does_not_depend_on_the_pose_list(self):
        a = self._ready(190.0, "inox_box", "")
        b = self._ready(190.0, "inox_box", "wood_box")
        assert a == b
        assert a[0] == 40.0 and a[1] == pytest.approx(150.0)

    def test_nearest_variant_wins_for_a_two_size_class(self):
        assert self._ready(80.0, "carton", "")[0] == 80.0
        assert self._ready(118.0, "carton", "")[0] == 120.0

    def test_detected_class_without_a_height_is_refused(self):
        with pytest.raises(RuntimeError, match="No configured height"):
            self._ready(100.0, "teapot", "")
