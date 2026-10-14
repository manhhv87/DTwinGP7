"""Motion guards must read perception and the class table, never the called pose.

Every test here calls the orchestrator's own method. An earlier version of this
file restated the conditions inside the tests, which is how it missed that the
commanded-height rule was also firing in rgbd and refusing a sound pick.
"""
import pathlib

import pytest
import yaml

from src.orchestrator.orchestrator import Orchestrator

ROOT = pathlib.Path(__file__).resolve().parents[1]

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


def _obj(class_name="inox_box", depth_used="rgbd", frac=0.9, h_depth=41.0,
         rect=None):
    o = {"class_name": class_name, "depth_used": depth_used,
         "depth_valid_frac": frac, "h_depth_mm": h_depth}
    if rect is not None:
        o["yaw_rect_long_mm"], o["yaw_rect_short_mm"] = rect
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

    def test_lookup_ignores_case(self):
        """A capital letter in a label must not switch a guard off."""
        o = _orch()
        assert o._class_min_height_mm("Inox_Box") == 40.0
        assert not o._size_matches_class("INOX_BOX", 400.0, 400.0, 40.0)

    def test_a_class_with_a_height_but_no_size(self):
        """Missing footprint means nothing to compare, not a refusal."""
        o = _orch(class_heights_mm={"metal_box": 85.0})
        assert o._class_min_height_mm("metal_box") == 85.0
        assert o._size_matches_class("metal_box", 1.0, 1.0, 5.0)


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


class TestDepthPlausibility:
    """Clause (1): the depth behind the command must be believable."""

    def test_a_mask_with_almost_no_depth_is_refused(self):
        reason = _orch()._depth_implausible(_obj(frac=0.05))
        assert reason is not None and "valid fraction" in reason

    def test_a_collapsed_top_face_is_refused(self):
        reason = _orch()._depth_implausible(_obj(h_depth=5.0))
        assert reason is not None and "below" in reason

    def test_a_sound_reading_passes(self):
        assert _orch()._depth_implausible(_obj()) is None

    def test_a_stacked_part_reads_higher_and_passes(self):
        assert _orch()._depth_implausible(_obj(h_depth=190.0)) is None

    def test_only_a_command_built_from_depth_is_judged(self):
        o = _orch()
        assert o._depth_implausible(
            _obj(depth_used="plane", frac=0.0, h_depth=0.0)) is None


class TestCommandedAgainstMeasuredHeight:
    """Clause (2): the plane estimate must not drive the tool into a support."""

    def test_plane_estimate_on_a_stacked_part_is_refused(self):
        # inox on a wood box: plane commands 40 mm, depth sees 190 mm.
        o = _orch()
        assert o._depth_contradicts_command(
            _obj(depth_used="plane", h_depth=190.0), 40.0) is not None

    def test_a_part_on_the_table_is_left_alone(self):
        o = _orch()
        assert o._depth_contradicts_command(
            _obj(depth_used="plane", h_depth=41.8), 40.0) is None

    def test_depth_that_cannot_be_trusted_does_not_refuse(self):
        o = _orch()
        assert o._depth_contradicts_command(
            _obj(depth_used="plane", frac=0.05, h_depth=190.0), 40.0) is None

    def test_a_fraction_below_the_veto_bar_does_not_refuse(self):
        """Depth must be as trustworthy as fusion demands before it may veto."""
        o = _orch()
        assert o._depth_contradicts_command(
            _obj(depth_used="plane", frac=0.2, h_depth=190.0), 40.0) is None

    def test_disagreement_inside_the_tolerance_is_left_alone(self):
        o = _orch()
        assert o._depth_contradicts_command(
            _obj(depth_used="plane", h_depth=135.0), 120.0) is None

    def test_rgbd_never_vetoes_its_own_measurement(self):
        """B3 of 14 Oct 2026, trial 9: one unstacked carton, two estimators.

        The commanded height came from the median depth over the whole mask,
        h_depth_mm from the top band of an eroded mask. The pick succeeded.
        """
        o = _orch()
        assert o._depth_contradicts_command(
            _obj(class_name="carton", frac=0.99, h_depth=173.3), 124.3) is None


class TestShape:
    """Clause (3): the measured top rectangle must be the detected class."""

    def test_a_part_read_at_the_wrong_plane_is_refused(self):
        """A 190 mm stack read as 40 mm inflates the rectangle by about 1.3x."""
        scale = (713.0 - 40.0) / (713.0 - 190.0)
        rect = (180.0 * scale, 85.0 * scale)
        assert _orch()._shape_contradicts_class(_obj(rect=rect)) is not None

    def test_a_matching_rectangle_passes(self):
        assert _orch()._shape_contradicts_class(_obj(rect=(182.0, 86.0))) is None

    def test_no_measurement_nothing_to_contradict(self):
        assert _orch()._shape_contradicts_class(_obj()) is None


# ── Measured runs ────────────────────────────────────────────────────────────
# table_top_z_mm recorded in the metadata of every run below.
TABLE_TOP_MM = 593.9006737673905

# Every trial of 12 and 14 Oct 2026 that carries depth fields, copied from
# results/e1/experiment_real_20261014_B3_FINAL50.csv and the two files in
# results/B2_20261012: (run, trial, class, depth_used, valid fraction,
# h_depth_mm, det_z_mm, success). Trial 45 of 14 Oct was rebuilt from the
# console log and trial 5 of 12 Oct was a detection miss; neither has depth.
# None of these files has the rectangle columns, so clause (3) cannot be
# checked against them.
MEASURED = [
    ("B3 14/10", 1, "carton", "rgbd", 0.943, 155.8, 747.8, True),
    ("B3 14/10", 2, "plastic_box", "rgbd", 1.0, 28.1, 620.0, True),
    ("B3 14/10", 3, "wood_box", "rgbd", 0.938, 149.8, 742.0, True),
    ("B3 14/10", 4, "metal_box", "rgbd", 1.0, 84.6, 676.5, True),
    ("B3 14/10", 5, "inox_box", "rgbd", 0.833, 41.8, 633.1, True),
    ("B3 14/10", 6, "inox_box", "rgbd", 0.976, 72.2, 663.1, True),
    ("B3 14/10", 7, "inox_box", "rgbd", 0.928, 41.8, 633.1, True),
    ("B3 14/10", 8, "inox_box", "rgbd", 0.556, 39.5, 628.8, False),
    ("B3 14/10", 9, "carton", "rgbd", 0.99, 173.3, 718.2, True),
    ("B3 14/10", 10, "plastic_box", "rgbd", 1.0, 29.0, 620.5, True),
    ("B3 14/10", 11, "wood_box", "rgbd", 0.984, 236.1, 827.8, True),
    ("B3 14/10", 12, "metal_box", "rgbd", 0.996, 84.2, 676.6, True),
    ("B3 14/10", 13, "inox_box", "rgbd", 0.811, 40.9, 631.6, True),
    ("B3 14/10", 14, "inox_box", "rgbd", 0.892, 41.2, 632.1, True),
    ("B3 14/10", 15, "inox_box", "rgbd", 0.96, 41.5, 631.5, False),
    ("B3 14/10", 16, "inox_box", "rgbd", 0.475, 127.1, 718.1, False),
    ("B3 14/10", 17, "carton", "rgbd", 0.994, 125.4, 717.5, True),
    ("B3 14/10", 18, "plastic_box", "rgbd", 1.0, 29.7, 621.0, True),
    ("B3 14/10", 19, "wood_box", "rgbd", 0.997, 152.8, 744.8, True),
    ("B3 14/10", 20, "metal_box", "rgbd", 0.997, 84.7, 677.0, True),
    ("B3 14/10", 21, "inox_box", "rgbd", 0.453, 167.0, 756.7, True),
    ("B3 14/10", 22, "inox_box", "rgbd", 0.97, 42.3, 633.0, True),
    ("B3 14/10", 23, "inox_box", "rgbd", 0.74, 41.5, 632.7, False),
    ("B3 14/10", 24, "inox_box", "rgbd", 0.299, 41.8, 631.2, True),
    ("B3 14/10", 25, "carton", "rgbd", 0.985, 123.9, 715.7, True),
    ("B3 14/10", 26, "plastic_box", "rgbd", 0.985, 181.3, 773.9, True),
    ("B3 14/10", 27, "wood_box", "rgbd", 0.998, 152.6, 744.7, True),
    ("B3 14/10", 28, "metal_box", "rgbd", 0.999, 84.7, 676.6, True),
    ("B3 14/10", 29, "inox_box", "rgbd", 0.87, 40.9, 632.5, True),
    ("B3 14/10", 30, "inox_box", "rgbd", 0.97, 40.7, 632.5, True),
    ("B3 14/10", 31, "inox_box", "rgbd", 0.59, 40.0, 625.7, False),
    ("B3 14/10", 32, "inox_box", "rgbd", 0.991, 41.3, 632.1, True),
    ("B3 14/10", 33, "carton", "rgbd", 0.928, 82.2, 674.6, True),
    ("B3 14/10", 34, "plastic_box", "rgbd", 1.0, 29.2, 621.5, True),
    ("B3 14/10", 35, "wood_box", "rgbd", 0.999, 153.1, 745.1, True),
    ("B3 14/10", 36, "metal_box", "rgbd", 0.974, 128.1, 719.9, True),
    ("B3 14/10", 37, "inox_box", "rgbd", 0.416, 39.3, 628.8, True),
    ("B3 14/10", 38, "inox_box", "rgbd", 0.896, 41.8, 633.5, True),
    ("B3 14/10", 39, "inox_box", "rgbd", 0.607, 42.4, 633.8, True),
    ("B3 14/10", 40, "inox_box", "rgbd", 0.683, 42.3, 633.9, True),
    ("B3 14/10", 41, "carton", "rgbd", 0.983, 81.8, 673.8, True),
    ("B3 14/10", 42, "plastic_box", "rgbd", 0.995, 181.4, 773.5, True),
    ("B3 14/10", 43, "wood_box", "rgbd", 0.927, 153.2, 744.9, True),
    ("B3 14/10", 44, "metal_box", "rgbd", 0.995, 85.7, 677.1, True),
    ("B3 14/10", 46, "inox_box", "rgbd", 0.815, 41.4, 629.9, False),
    ("B3 14/10", 47, "inox_box", "rgbd", 0.215, 39.8, 630.2, False),
    ("B3 14/10", 48, "inox_box", "rgbd", 0.45, 41.1, 632.9, True),
    ("B3 14/10", 49, "carton", "rgbd", 0.969, 126.4, 717.8, True),
    ("B3 14/10", 50, "plastic_box", "rgbd", 1.0, 29.5, 621.1, True),
    ("B2 12/10", 1, "carton", "rgbd", 0.984, 126.0, 717.8, True),
    ("B2 12/10", 2, "plastic_box", "rgbd", 0.995, 29.2, 621.0, True),
    ("B2 12/10", 3, "metal_box", "rgbd", 1.0, 84.5, 676.6, True),
    ("B2 12/10", 4, "wood_box", "rgbd", 1.0, 149.8, 741.8, True),
    ("B2 12/10", 6, "carton", "rgbd", 0.981, 126.1, 717.8, True),
    ("B2 12/10 rerun", 1, "inox_box", "rgbd", 0.704, 39.5, 630.0, True),
]


def _orch_from_config_file(**override):
    """The orchestrator as 03_run_experiment configures it in real mode."""
    raw = yaml.safe_load((ROOT / "config/experiment.yaml").read_text(
        encoding="utf-8"))
    real = raw.pop("real", None) or {}
    raw.update(real)
    o = Orchestrator.__new__(Orchestrator)
    o.config = {**raw, **override}
    return o


class TestMeasuredRuns:
    """The configured thresholds against every trial the cell has measured."""

    def test_the_data_is_complete(self):
        assert len(MEASURED) == 55
        assert sum(ok for *_, ok in MEASURED) == 48

    def test_no_successful_trial_is_refused(self):
        o = _orch_from_config_file()
        refused = []
        for run, trial, cls, used, frac, h_depth, det_z, ok in MEASURED:
            obj = {"class_name": cls, "depth_used": used,
                   "depth_valid_frac": frac, "h_depth_mm": h_depth}
            reason = (o._depth_implausible(obj)
                      or o._depth_contradicts_command(obj, det_z - TABLE_TOP_MM))
            if ok and reason:
                refused.append((run, trial, reason))
        assert refused == []

    def test_the_floor_lifts_no_measured_reading(self):
        o = _orch_from_config_file()
        lifted = [(run, trial) for run, trial, cls, *_, det_z, _ok in MEASURED
                  if det_z < o._pick_floor_z(cls, TABLE_TOP_MM)]
        assert lifted == []

    def test_a_floor_at_the_nominal_height_would_lift_most_readings(self):
        """The figures quoted in _pick_floor_z: 36 of 55, 21 of 24 inox, 8.2 mm."""
        o = _orch_from_config_file(height_tolerance_mm=0.0)
        lifts = [(cls, o._pick_floor_z(cls, TABLE_TOP_MM) - det_z)
                 for _r, _t, cls, *_, det_z, _ok in MEASURED]
        lifted = [(cls, d) for cls, d in lifts if d > 0.0]
        assert len(lifted) == 36
        assert sum(cls == "inox_box" for cls, _ in lifted) == 21
        assert sum(cls == "inox_box" for cls, _ in lifts) == 24
        assert max(d for _, d in lifted) == pytest.approx(8.2, abs=0.05)
