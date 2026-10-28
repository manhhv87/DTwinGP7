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
        assert o._long_side_matches_class("carton", 182.0, 40.0)
        assert o._long_side_matches_class("carton", 178.0, 40.0)

    def test_rectangle_far_from_every_variant_is_rejected(self):
        assert not _orch()._long_side_matches_class("inox_box", 400.0, 40.0)

    def test_unknown_class_cannot_be_contradicted(self):
        assert _orch()._long_side_matches_class("teapot", 1.0, 5.0)

    def test_lookup_ignores_case(self):
        """A capital letter in a label must not switch a guard off."""
        o = _orch()
        assert o._class_min_height_mm("Inox_Box") == 40.0
        assert not o._long_side_matches_class("INOX_BOX", 400.0, 40.0)

    def test_a_class_with_a_height_but_no_size(self):
        """Missing footprint means nothing to compare, not a refusal."""
        o = _orch(class_heights_mm={"metal_box": 85.0})
        assert o._class_min_height_mm("metal_box") == 85.0
        assert o._long_side_matches_class("metal_box", 1.0, 5.0)


class TestCarriedHeight:
    """The height carried away comes from the detected class, not the pose list."""

    def _ready(self, total_mm, class_name, stack_on, rect=None):
        o = _orch()
        o.current_stack_on = stack_on
        o._world_to_robot_base = lambda T: T
        o._table_z_base_at_xy = lambda x, y: 0.0
        obj = {"pose_base": (500.0, 0.0, total_mm), "class_name": class_name}
        if rect is not None:
            obj["yaw_rect_long_mm"], obj["yaw_rect_short_mm"] = rect
        return o._detected_target_height_mm(obj)

    def test_a_carton_on_a_support_is_sized_by_its_top_face(self):
        """B3 of 15 Oct 2026, trial 1: an 80 mm carton on the plastic box.

        By height alone it read as a 120 mm carton standing on nothing and
        would have been released about 40 mm above the table.
        """
        h, support, _, _ = self._ready(112.8, "carton", "", rect=(196.8, 105.1))
        assert h == 80.0 and support == pytest.approx(32.8)

    def test_a_flat_carton_is_sized_by_its_top_face(self):
        assert self._ready(80.9, "carton", "", rect=(182.6, 102.7))[0] == 80.0
        assert self._ready(124.2, "carton", "", rect=(187.1, 122.6))[0] == 120.0

    def test_a_one_size_class_ignores_the_rectangle(self):
        h, support, _, _ = self._ready(182.5, "plastic_box", "", rect=(183.8, 167.4))
        assert h == 30.0 and support == pytest.approx(152.5)

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

    def test_depth_that_forms_no_top_face_is_refused(self):
        reason = _orch()._depth_implausible(_obj(h_depth=None))
        assert reason is not None and "top face" in reason

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

    def test_a_stack_seen_with_little_depth_is_still_refused(self):
        """15 Oct 2026, trial 47: inox on the wood box read at a fraction of 0.267."""
        o = _orch()
        assert o._depth_contradicts_command(
            _obj(depth_used="plane", frac=0.267, h_depth=194.8), 40.0) is not None

    def test_a_dead_mask_does_not_veto(self):
        o = _orch()
        assert o._depth_contradicts_command(
            _obj(depth_used="plane", frac=0.10, h_depth=190.0), 40.0) is None

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

    def test_an_over_wide_mask_is_not_refused(self):
        """The short side does not move where a finger comes down."""
        assert _orch()._shape_contradicts_class(_obj(rect=(190.3, 125.6))) is None

    def test_the_inox_refusals_of_15_october(self):
        """Seven inox picks were refused on the full rectangle that day.

        Judged on the long side only, the three over-long masks stay refused
        and the four that were only too wide go through.
        """
        cases = {6: (197.4, 177.7), 16: (193.0, 130.2), 23: (227.2, 130.1),
                 30: (190.3, 125.6), 31: (189.3, 155.9), 47: (250.0, 115.8),
                 48: (241.3, 104.1)}
        o = _orch()
        refused = {t for t, rect in cases.items()
                   if o._shape_contradicts_class(_obj(rect=rect)) is not None}
        assert refused == {23, 47, 48}

    def test_inox_read_at_its_support_is_refused(self):
        """28 Oct 2026, P0118: inox on the plastic box, depth read the box.

        rgbd and plane both ended in collision stops (alarm 4315).
        """
        o = _orch()
        for rect, top in (((203.3, 140.9), 29.7), ((203.6, 139.1), 28.4),
                          ((205.0, 142.0), 30.4), ((204.4, 144.1), 29.7)):
            why = o._shape_contradicts_class(_obj(rect=rect, h_depth=top))
            assert why is not None and "rests on" in why

    def test_wide_mask_with_a_right_top_passes(self):
        """A tall carton 120 read 152 mm wide with its top right (P0129)."""
        o = _orch()
        assert o._shape_contradicts_class(
            _obj(class_name="carton", rect=(184.8, 153.0), h_depth=124.8)) is None

    def test_low_inox_top_with_a_narrow_mask_passes(self):
        """A single mirror inox read 21.6 mm high, mask the part's width (P0110)."""
        assert _orch()._shape_contradicts_class(
            _obj(rect=(188.0, 91.8), h_depth=21.6)) is None

    def test_support_check_needs_depth_and_a_footprint(self):
        o = _orch()
        assert not o._reads_its_support(_obj(rect=(203.0, 141.0), h_depth=None), 40.0)
        assert not o._reads_its_support(
            _obj(class_name="teapot", rect=(203.0, 141.0), h_depth=10.0), 40.0)
        assert not o._reads_its_support({"class_name": "inox_box",
                                         "h_depth_mm": 20.0}, 40.0)


class TestCommandedPickZ:
    """Every height rule for the pick, judged on the commanded point."""

    @staticmethod
    def _z(class_name, top_z, **cfg):
        o = _orch(table_safety_margin_mm=35.0, grasp_depth_offset_mm=0.0,
                  height_tolerance_mm=20.0, **cfg)
        return o._commanded_pick_z(
            {"pose_base": (500.0, 0.0, top_z), "class_name": class_name}, 0.0)

    def test_a_sound_reading_of_a_thick_part_is_left_alone(self):
        assert self._z("inox_box", 38.0) == pytest.approx(38.0)

    def test_a_reading_like_trial_45_is_held_at_the_clearance(self):
        """Trial 45 of 14 Oct 2026 was commanded about 25 mm up and collided."""
        assert self._z("inox_box", 25.0) == pytest.approx(35.0)

    def test_a_thin_part_on_the_table_is_lifted_to_the_clearance(self):
        assert self._z("plastic_box", 27.0) == pytest.approx(35.0)

    def test_a_thin_part_on_a_support_keeps_the_clearance_above_it(self):
        # plastic on a 150 mm support: top at 180, support top at 150.
        assert self._z("plastic_box", 180.0) == pytest.approx(185.0)

    def test_a_thick_part_on_a_support_is_left_alone(self):
        assert self._z("carton", 230.0) == pytest.approx(230.0)

    def test_a_collapsed_median_on_a_stack_follows_the_top_band(self):
        """rgbd: a median dragged onto the support must not send the tool there.

        Inox on the wood box: top band at 190 mm, median collapsed to 17 mm
        (15 Oct 2026, trial 24 read 17.3 against 40.9 on a flat inox).
        """
        o = _orch(table_safety_margin_mm=35.0, grasp_depth_offset_mm=0.0,
                  height_tolerance_mm=20.0)
        z = o._commanded_pick_z({"pose_base": (500.0, 0.0, 17.0),
                                 "class_name": "inox_box", "depth_used": "rgbd",
                                 "depth_valid_frac": 0.8, "h_depth_mm": 190.0}, 0.0)
        assert z == pytest.approx(185.0)

    def test_a_plane_command_is_never_lifted_by_depth(self):
        """Plane mode stays the plane estimate; the refusal clause judges it."""
        o = _orch(table_safety_margin_mm=35.0, grasp_depth_offset_mm=0.0,
                  height_tolerance_mm=20.0)
        z = o._commanded_pick_z({"pose_base": (500.0, 0.0, 40.0),
                                 "class_name": "inox_box", "depth_used": "plane",
                                 "depth_valid_frac": 0.8, "h_depth_mm": 190.0}, 0.0)
        assert z == pytest.approx(40.0)

    def test_a_bias_cannot_go_below_the_clearance(self):
        z = self._z("inox_box", 38.0, class_pick_z_bias_mm={"inox_box": -10.0})
        assert z == pytest.approx(35.0)


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


    def test_commanded_heights_keep_the_clearance(self):
        """The configured clearance on every measured reading.

        Every commanded point stays the margin above the table and above
        the support. Only thin parts and low readings move: all eight plastic
        readings, four inox readings by at most 3.2 mm, and one carton by
        4.0 mm (14 Oct, trial 9, whose top band read 49 mm above its median),
        so the other 20 inox readings reach the robot exactly as measured.
        """
        o = _orch_from_config_file()
        margin = float(o.config["table_safety_margin_mm"])
        lifted = []
        for run, trial, cls, used, frac, h_depth, det_z, _ok in MEASURED:
            z = o._commanded_pick_z(
                {"pose_base": (0.0, 0.0, det_z), "class_name": cls,
                 "depth_used": used, "depth_valid_frac": frac,
                 "h_depth_mm": h_depth}, TABLE_TOP_MM)
            support = max(TABLE_TOP_MM, det_z - o._class_min_height_mm(cls))
            assert z - support >= margin - 1e-9, (run, trial)
            if z > det_z + 1e-9:
                lifted.append((cls, z - det_z))
        assert margin == 35.0
        assert len(lifted) == 13
        assert sum(cls == "plastic_box" for cls, _ in lifted) == 8
        inox = [d for cls, d in lifted if cls == "inox_box"]
        assert len(inox) == 4
        assert max(inox) == pytest.approx(3.2, abs=0.05)
        carton = [d for cls, d in lifted if cls == "carton"]
        assert carton == [pytest.approx(4.0, abs=0.05)]


class _Joints:
    """A robot whose joint readings are scripted, and which records moves."""

    def __init__(self, *readings):
        self.readings = list(readings)
        self.moves = []

    def Joints(self):
        return self.readings.pop(0)

    def MoveJ(self, target):
        self.moves.append(target)


class TestAtHome:
    """A fixed-job cycle starts and ends at the HOME seen before the first."""

    HOME = [-86.5, -17.4, -21.3, 0.0, -86.1, 91.8]

    def _o(self, *readings, real=True):
        o = _orch(is_real_mode=real, home_tolerance_deg=0.5)
        o.robot = _Joints(*readings)
        return o

    def test_the_configured_home_is_used_from_the_first_cycle(self):
        """A block that starts away from HOME is stopped before it moves.

        The third run of 14 Oct 2026 started 43.6 deg off at J6.
        """
        away = [-72.954, -14.759, -24.536, -0.009, -80.225, 135.388]
        o = self._o(away)
        o.config["taught_home_joints_deg"] = self.HOME
        reason = o._check_at_home("before")
        assert reason is not None and "before" in reason

    def test_the_first_cycle_learns_home(self):
        o = self._o(self.HOME, self.HOME)
        assert o._check_at_home("before") is None
        assert o._check_at_home("after") is None

    def test_a_cycle_stopped_half_way_is_caught(self):
        """15 Oct 2026, trial 25: stopped by alarm 4746 above the carton."""
        stopped = [1.89, 7.54, 1.27, 0.0, -83.72, 80.54]
        o = self._o(self.HOME, stopped)
        o._check_at_home("before")
        reason = o._check_at_home("after")
        assert reason is not None and "from HOME after" in reason

    def test_a_cycle_does_not_start_away_from_home(self):
        o = self._o(self.HOME, [x + 2.0 for x in self.HOME])
        o._check_at_home("before")
        assert "before" in o._check_at_home("before")

    def test_simulation_is_not_judged(self):
        o = self._o(real=False)
        assert o._check_at_home("before") is None

    def test_the_real_cell_does_not_move_home_on_its_own(self):
        o = self._o()
        o._return_home()
        assert o.robot.moves == []


class TestFitPlateau:
    """22 Oct 2026, kiemtra-2210 first run, trial 3 (P0216): an inox mask spilling
    past the box's end let the known-size rectangle slide 49 to 50 mm along its long
    side; genuine detections slid at most 7 mm."""

    @staticmethod
    def _det(slide, source="known_size_fit", long_mm=184.0):
        return {"class_name": "inox_box", "yaw_rect_long_mm": long_mm,
                "yaw_source": source, "fit_plateau_mm": slide}

    def test_a_sliding_fit_is_refused(self):
        reason = _orch()._shape_contradicts_class(self._det(49.0))
        assert reason is not None and "slides" in reason

    def test_a_pinned_fit_passes(self):
        assert _orch()._shape_contradicts_class(self._det(7.0)) is None

    def test_the_limit(self):
        o = _orch()
        assert o._shape_contradicts_class(self._det(10.0)) is None
        assert o._shape_contradicts_class(self._det(10.5)) is not None

    def test_only_a_fit_in_use_is_judged(self):
        assert _orch()._shape_contradicts_class(
            self._det(49.0, source="metric_base_minAreaRect")) is None

    def test_no_slide_recorded_nothing_to_judge(self):
        assert _orch()._shape_contradicts_class(self._det(None)) is None

    def test_the_long_side_still_refuses_first(self):
        reason = _orch()._shape_contradicts_class(self._det(3.0, long_mm=231.5))
        assert reason is not None and "long side" in reason


class TestMaskSpill:
    """A detection whose mask is largely table, at table height, is dropped."""

    @staticmethod
    def _det(share, h_median=40.0, h_depth=40.8, frac=0.96):
        return {"class_name": "inox_box", "table_share": share,
                "h_median_mm": h_median, "h_depth_mm": h_depth,
                "depth_valid_frac": frac}

    def test_a_spill_onto_the_table_is_caught(self):
        """19 Oct 2026: a mask over the inox's end and the table beside it."""
        reason = _orch()._mask_off_part(self._det(0.6, h_median=0.4))
        assert reason is not None and reason.startswith("on the table")

    def test_a_flat_detection_on_the_cards_is_caught(self):
        """20 Oct 2026: a group of paper position cards labelled inox_box."""
        reason = _orch()._mask_off_part(
            self._det(0.97, h_median=0.3, h_depth=0.5, frac=0.99))
        assert reason is not None and reason.startswith("on the table")

    def test_a_mirror_inox_with_its_median_on_the_table_is_kept(self):
        """What the height test threw away (103 of 842 single inox on the C4
        set): a mirror top's depth falls onto or below the table, little of it
        at table height."""
        assert _orch()._mask_off_part(
            self._det(0.12, h_median=4.0, h_depth=24.0, frac=0.55)) is None

    def test_the_limit(self):
        o = _orch()
        assert o._mask_off_part(self._det(0.299)) is None
        assert o._mask_off_part(self._det(0.30)) is not None

    def test_a_stacked_part_passes(self):
        assert _orch()._mask_off_part(
            self._det(0.02, h_median=186.3, h_depth=194.9)) is None

    def test_no_share_nothing_to_judge(self):
        assert _orch()._mask_off_part(self._det(None)) is None

