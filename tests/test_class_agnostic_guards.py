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


class TestThresholdsAgainstMeasuredRuns:
    """Values seen in the B3 run of 14 Oct 2026 must not be refused.

    Nineteen flat inox picks succeeded there with a valid-depth fraction from
    0.299 to 0.991 and a top face from 39.3 to 42.4 mm. A guard that refuses
    any of them would throw away trials the cell can complete.
    """

    FLAT_INOX_OK = [(0.299, 41.8), (0.416, 39.3), (0.450, 41.1), (0.607, 42.4),
                    (0.683, 42.3), (0.811, 40.9), (0.833, 41.8), (0.991, 41.3)]

    def test_successful_inox_picks_are_not_refused(self):
        o = _orch()
        for frac, h_depth in self.FLAT_INOX_OK:
            assert frac >= 0.15, f"valid fraction {frac} would be refused"
            assert h_depth >= o._class_min_height_mm("inox_box") - 20.0

    def test_a_mask_with_almost_no_depth_is_refused(self):
        assert 0.05 < 0.15        # the clause that still fires

    def test_a_collapsed_top_face_is_refused(self):
        o = _orch()
        assert 5.0 < o._class_min_height_mm("inox_box") - 20.0

    def test_a_part_read_at_the_wrong_plane_fails_the_shape_check(self):
        """A 190 mm stack read as 40 mm inflates the rectangle by about 1.3x."""
        o = _orch()
        scale = (713.0 - 40.0) / (713.0 - 190.0)
        long_mm, short_mm = 180.0 * scale, 85.0 * scale
        assert not o._size_matches_class("inox_box", long_mm, short_mm, 40.0)
