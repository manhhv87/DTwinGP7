"""
placement_check.py
──────────────────
Before any motion, compare the scene perception sees with the pose the trial called.

The operator stages every trial by hand from the screen prompt: a part of the called
class on a numbered card, at a called yaw, and for a stacked pose on a called support.
In session 1 of E2 (e2chuan-1, 23 Oct 2026, 300 campaign trials) 18 trials on 16
poses were staged otherwise than called and nothing on the screen said so: 14 of the
60 stacked trials had the support missing or wrong, two had the wrong class, two sat
310 and 109 mm from their card; besides, the carton size differed between the arms
of seven poses because the pose list did not name one. Each of them breaks the
pairing of the arms on the same scene, which is what the paired test relies on.

What is compared, and why each quantity is the same in every depth mode:

  class        no detection of the called class with depth under its mask survived
               the class lock. The detector, the table-share drop and the depth
               under the mask are the same in every mode (see "Which detection is
               judged" below).
  position     the centre of the part's top face against the called card: where
               the ray through the grasp pixel meets the plane of the measured top
               (h_depth_mm, else yaw_top_height_mm, the plane PoseExtractor fitted
               the known-size rectangle on): the fit centre when depth is trusted.
               Neither the pixel nor the height depends on the depth mode. The
               commanded point does: plane stands every part on the table, so on a
               stack its point slides along the ray (up to 53 mm in session 1),
               and the rgbd point lies at the median depth (up to 11 mm off).
  stack        the top face measured from depth (h_depth_mm, computed alike in every
               mode) against the called part plus the called support.
  carton_size  the same height, when it fits the other carton size instead of the
               called one, for a carton target or a carton support.
  yaw          the measured yaw of the part against the called yaw, folded with the
               period perception folds it with: 90 deg for a footprint square to the
               yaw (the wood box), and for a rectangle read near square when the
               known-size fit did not give the yaw; 180 deg otherwise.

None of these reads the depth mode, the commanded height or a refusal, so the same
scene raises the same flags in every arm of a campaign whose arms differ only in the
depth mode (E2). When the arms differ in the detector (E3 to E5), the detections
themselves differ between arms, and so would the flags: such campaigns run without
the check (run_blinded_campaign.py adds --no-placement-check when the arms differ in
anything but --depth-mode).

Which detection is judged. The first detection of the class lock whose mask returned
depth on at least MIN_DEPTH_PIXELS pixels (judged_detection). PoseExtractor in rgbd
mode drops a detection with fewer, since postprocess.masked_depth then gives no
depth, while plane and fusion keep it on the table plane. The class lock ranks the
detections that have depth by the height of their median-depth point, which every
mode measures alike. Passing over the detections without depth therefore judges the
same detection in every mode, and in rgbd it is always the first one, the one the
cycle acts on. In session 1 the fewest valid depth pixels under a logged mask were
3140 (plane P0086, 0.206 of 15245), far from this case.

The check decides nothing about the trial. It returns flags; the runner shows them
and the operator either fixes the placement, after which the orchestrator captures
again, or declares the placement as called, after which the trial proceeds with the
detection that was checked. A perception failure therefore stays a failure.

Thresholds. Session 1 was replayed through check_placement offline, from the logged
detections of its 300 campaign and 40 control trials (316 of them detected and
without a verified deviation):

  POSITION_TOL_MM 40   those 316 came at most 33.0 mm from their card (P0079, an
                       inox mask spilling over the cards), 11.6 mm at the median;
                       the two trials placed off their card read 109 and 310 mm; the
                       cards are 50 mm apart in x and 60 mm in y. Any value from 35
                       to 50 mm flags the same trials.
  YAW_TOL_DEG 20       single parts of the campaign read 8.5 deg off the called yaw
                       at the 95th percentile and 16.9 deg at most; the four turned
                       placements confirmed on the frames read 30 to 68 deg off.
                       Flagged as well: seven stacked trials at 21 to 35 deg, among
                       them P0072 in rgbd, whose frame shows the inox turned on its
                       support.
  HEIGHT_TOL_ABOVE_MM  correctly staged trials read at most 3.7 mm above the expected
  10                   top. The nearest wrong stagings of session 1 read 12.3 mm
                       above (an 80 mm carton on the inox box instead of the plastic
                       box) and 13.0 and 13.7 mm above (a 120 mm carton alone for an
                       80 mm one on the plastic box).
  HEIGHT_TOL_BELOW_MM  for a matte top (carton, plastic, wood and metal box): the 168
  8                    correctly staged readings of these classes lay from 1.6 mm
                       below to 2.9 mm above the expected top. Of the stagings the
                       pose lists allow, the nearest wrong ones below are, from the
                       flat heights, the plastic box in place of the inox box under
                       the metal box (12.0 mm lower) and an 80 mm carton on the
                       plastic box for a 120 mm carton alone (13.7 mm lower).
  HEIGHT_TOL_BELOW_    for the mirror top of the inox box: correctly staged inox
  MIRROR_MM 15         trials read down to 9.9 mm below (single parts), and one more,
                       an inox on the plastic box, 40.1 mm below. A missing support
                       takes away 29.5 mm at least (the plastic box).
                       Not seen with these bands: the 80 mm carton and the metal box
                       swapped as supports (2.5 mm apart).
  MAX_RECAPTURES 3     the operator fixes a placement and the scene is captured again
                       at most this often; then the trial proceeds over the flags.

With these values the replay flags all 22 trials with a verified deviation and 11
others: the seven stacked yaw cases above, one control carton at 34 deg, two inox
trials in which the class lock found no usable inox (class), and that inox on the
plastic box read 40 mm low (stack). The values were chosen on that same session; how
they do on staging errors they were not tuned on is for the frame audit of later
sessions to show.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

POSITION_TOL_MM = 40.0
YAW_TOL_DEG = 20.0
# The measured top may lie this far above or below the called staging's top. A
# mirror top reads low (depth from its edges and reflections); a matte one does not.
HEIGHT_TOL_ABOVE_MM = 10.0
HEIGHT_TOL_BELOW_MM = 8.0
HEIGHT_TOL_BELOW_MIRROR_MM = 15.0
MIRROR_TOP_CLASSES = frozenset({"inox_box"})
MAX_RECAPTURES = 3
# postprocess.masked_depth gives no depth for a mask with fewer valid depth pixels
# than this, and PoseExtractor in rgbd mode then drops the detection.
MIN_DEPTH_PIXELS = 5

# Top face of each part standing flat on the table, measured from depth (median of
# the top band) on 14 and 15 Oct 2026; session 1 of E2 read the same within 0.6 mm.
FLAT_TOP_MM: dict[str, float] = {
    "plastic_box": 29.5,
    "wood_box": 153.1,
    "metal_box": 84.7,
    "inox_box": 41.5,
}
CARTON = "carton"
CARTON_TOP_MM: dict[int, float] = {80: 82.2, 120: 125.4}

# Pose-list yaw = measured yaw + 90 deg. The measured yaw is the long side's angle
# from base +x, while the 0 deg mark of the cards points along base +y (B2, 12 Oct
# 2026: six trials 85.8 to 89.1 deg off when compared directly, 0.9 to 4.2 after).
POSE_YAW_OFFSET_DEG = 90.0
# A footprint whose long side is under this many times its short side is square to
# the yaw: the rule perception folds its yaw with (depth_modes.NEAR_SQUARE_RATIO; a
# test keeps the two equal).
NEAR_SQUARE_RATIO = 1.15
# PoseExtractor's yaw_source when the known-size rectangle gave the yaw.
KNOWN_SIZE_FIT = "known_size_fit"

FLAG_CODES = ("class", "position", "stack", "carton_size", "yaw")


@dataclass(frozen=True)
class CalledPose:
    """The staging a trial called for, from its pose-list row and carton sidecar."""

    pose_id: str = ""
    card_id: str = ""
    x_mm: float | None = None
    y_mm: float | None = None
    yaw_deg: float | None = None
    class_name: str = ""
    stack_on: str = ""
    carton_mm: int | None = None          # called size of a carton target
    stack_on_carton_mm: int | None = None  # called size of a carton support


@dataclass(frozen=True)
class Flag:
    """One way the scene differs from the called pose, with the line shown for it."""

    code: str
    text: str


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def class_label(class_name: str, carton_mm: int | None) -> str:
    """'carton(120)' for a carton of known size, the class name otherwise."""
    if class_name == CARTON and carton_mm:
        return f"{CARTON}({int(carton_mm)})"
    return class_name


def called_pose_from_row(row: Mapping[str, Any],
                         carton_sizes: Mapping[str, tuple[int | None, int | None]]
                         | None = None) -> CalledPose:
    """CalledPose from a pose-list row and the sizes read by load_carton_sizes."""
    pose_id = str(row.get("pose_id", "") or "")
    target_mm, support_mm = (carton_sizes or {}).get(pose_id, (None, None))
    cls = str(row.get("class_hint", "") or "").strip().lower()
    stack_on = str(row.get("stack_on", "") or "").strip().lower()
    return CalledPose(
        pose_id=pose_id,
        card_id=str(row.get("card_id", "") or ""),
        x_mm=_num(row.get("x_mm")),
        y_mm=_num(row.get("y_mm")),
        yaw_deg=_num(row.get("yaw_deg")),
        class_name=cls,
        stack_on=stack_on,
        carton_mm=target_mm if cls == CARTON else None,
        stack_on_carton_mm=support_mm if stack_on == CARTON else None,
    )


# ── carton sizes per pose ──────────────────────────────────────────────────────

def carton_sizes_path(pose_list: str | Path) -> Path:
    """The sidecar of a pose list: std_v2.csv -> std_v2_carton.csv, same folder."""
    p = Path(pose_list)
    return p.with_name(f"{p.stem}_carton.csv")


def load_carton_sizes(path: str | Path,
                      allowed: Iterable[int] = tuple(CARTON_TOP_MM)
                      ) -> dict[str, tuple[int | None, int | None]]:
    """{pose_id: (carton_mm, stack_on_carton_mm)} from a sidecar written by
    scripts/28_make_carton_sizes.py. A blank cell is None.

    Raises:
        ValueError: a size outside ``allowed``, or a pose listed twice.
    """
    allowed = {int(a) for a in allowed}
    out: dict[str, tuple[int | None, int | None]] = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = str(row.get("pose_id", "") or "").strip()
            if not pid:
                continue
            if pid in out:
                raise ValueError(f"{path}: pose {pid} listed twice")
            sizes = []
            for key in ("carton_mm", "stack_on_carton_mm"):
                cell = str(row.get(key, "") or "").strip()
                if not cell:
                    sizes.append(None)
                    continue
                size = int(float(cell))
                if size not in allowed:
                    raise ValueError(f"{path}: pose {pid} {key}={cell}, expected one "
                                     f"of {sorted(allowed)}")
                sizes.append(size)
            out[pid] = (sizes[0], sizes[1])
    return out


def missing_carton_sizes(rows: Sequence[Mapping[str, Any]],
                         sizes: Mapping[str, tuple[int | None, int | None]]) -> list[str]:
    """Pose ids of rows with a carton target or support that has no size."""
    missing = []
    for row in rows:
        pid = str(row.get("pose_id", "") or "")
        target_mm, support_mm = sizes.get(pid, (None, None))
        if (str(row.get("class_hint", "")).strip().lower() == CARTON and not target_mm) or (
                str(row.get("stack_on", "")).strip().lower() == CARTON and not support_mm):
            missing.append(pid)
    return missing


# ── which detection is judged ─────────────────────────────────────────────────

def has_depth(obj: Mapping[str, Any]) -> bool:
    """False when the detection's mask returned depth on fewer than MIN_DEPTH_PIXELS
    pixels (depth_valid_frac times mask_area), the detections PoseExtractor drops in
    rgbd mode; True when either number is missing."""
    frac = _num(obj.get("depth_valid_frac"))
    area = _num(obj.get("mask_area"))
    if frac is None or area is None:
        return True
    return frac * area > MIN_DEPTH_PIXELS - 0.5


def judged_detection(objects: Sequence[Mapping[str, Any]] | None
                     ) -> Mapping[str, Any] | None:
    """The detection the check judges: the first of the class lock's list (topmost
    first) that has depth under its mask, or None. See the module docstring."""
    return next((o for o in objects or () if has_depth(o)), None)


# ── geometry ──────────────────────────────────────────────────────────────────

def top_centre_xy(obj: Mapping[str, Any], T_BC: np.ndarray | None,
                  table_plane: Mapping[str, Any] | None) -> tuple[float, float] | None:
    """Base (x, y) of the centre of the part's top face, or None without the data.

    pose_camera lies on the viewing ray through the grasp pixel (the known-size fit
    centre projected into the image, or the mask centroid), whatever depth mode set
    its distance. The ray is cut by the table plane lifted by the measured top face,
    h_depth_mm, when it lies in the 5 to 350 mm PoseExtractor accepts, and else by
    yaw_top_height_mm, the plane PoseExtractor fitted the outline and the known-size
    rectangle on. Where depth is trusted the two heights are equal and the point is
    the known-size fit centre. On a stacked part whose mask returned depth on less
    than 0.3 of its pixels the fit plane is the class height on the table, a whole
    support below the top: P0047 in session 1, inox on the wood box, came out 44.0
    and 42.7 mm from its card that way, 11.3 and 15.6 mm on the measured top.
    Neither the pixel nor the heights depend on the depth mode.
    """
    pc = obj.get("pose_camera")
    h = _num(obj.get("h_depth_mm"))
    if h is None or not 5.0 <= h <= 350.0:
        h = _num(obj.get("yaw_top_height_mm"))
    if pc is None or h is None or T_BC is None or not table_plane:
        return None
    try:
        a, b, c = (float(table_plane[k]) for k in ("a", "b", "c"))
        d_cam = np.asarray(pc, dtype=float)[:3]
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if not np.all(np.isfinite(d_cam)) or not np.any(d_cam):
        return None
    T = np.asarray(T_BC, dtype=float)
    origin = T[:3, 3]
    direction = T[:3, :3] @ d_cam
    n_z = 1.0 / math.sqrt(a * a + b * b + 1.0)
    c_top = c + h / n_z
    denom = direction[2] - a * direction[0] - b * direction[1]
    if abs(denom) < 1e-9:
        return None
    s = (a * origin[0] + b * origin[1] + c_top - origin[2]) / denom
    if s <= 0.0:
        return None
    p = origin + s * direction
    return float(p[0]), float(p[1])


def yaw_period_deg(class_name: str, class_sizes: Mapping[str, Any] | None) -> float:
    """90 for a class whose footprint is square to the yaw (the wood box), else 180."""
    raw = {str(k).strip().lower(): v for k, v in (class_sizes or {}).items()}.get(
        str(class_name).strip().lower())
    if not raw:
        return 180.0
    first = raw[0] if isinstance(raw[0], (list, tuple)) else raw
    try:
        long_mm, short_mm = sorted((float(first[0]), float(first[1])), reverse=True)
    except (TypeError, ValueError, IndexError):
        return 180.0
    return 90.0 if short_mm > 0 and long_mm / short_mm < NEAR_SQUARE_RATIO else 180.0


def detection_yaw_period_deg(obj: Mapping[str, Any], class_name: str,
                             class_sizes: Mapping[str, Any] | None) -> float:
    """The period the detection's yaw_base was folded with.

    The known-size fit folds by the class footprint (yaw_period_deg). Without it the
    yaw is that of the outline's minimum-area rectangle, which PoseExtractor folds
    modulo 90 deg when the rectangle reads near square (yaw_near_square), whatever
    the class.
    """
    if obj.get("yaw_near_square") and obj.get("yaw_source") != KNOWN_SIZE_FIT:
        return 90.0
    return yaw_period_deg(class_name, class_sizes)


def yaw_error_deg(measured_yaw_deg: float, called_yaw_deg: float,
                  period_deg: float = 180.0) -> float:
    """Signed difference between the measured and the called yaw, in pose-list terms,
    wrapped into [-period/2, period/2)."""
    d = (measured_yaw_deg + POSE_YAW_OFFSET_DEG) - called_yaw_deg
    half = period_deg / 2.0
    return (d + half) % period_deg - half


def _tops(class_name: str, carton_mm: int | None) -> list[float] | None:
    """Candidate flat top heights of a part (both carton sizes when none is called)."""
    if class_name == CARTON:
        if carton_mm:
            top = CARTON_TOP_MM.get(int(carton_mm))
            return None if top is None else [top]
        return sorted(CARTON_TOP_MM.values())
    top = FLAT_TOP_MM.get(class_name)
    return None if top is None else [top]


def _other_carton(size: int | None) -> int | None:
    if not size:
        return None
    others = [s for s in CARTON_TOP_MM if s != int(size)]
    return others[0] if len(others) == 1 else None


def _staging(cls: str, cls_mm: int | None, support: str, support_mm: int | None) -> str:
    part = class_label(cls, cls_mm)
    return f"{part} on {class_label(support, support_mm)}" if support else f"{part} alone"


def lower_tolerance_mm(class_name: str, matte_mm: float = HEIGHT_TOL_BELOW_MM,
                       mirror_mm: float = HEIGHT_TOL_BELOW_MIRROR_MM) -> float:
    """The lower tolerance for a top face of this class: wider for a mirror top."""
    return mirror_mm if class_name in MIRROR_TOP_CLASSES else matte_mm


def _height_flag(called: CalledPose, h: float, above: float,
                 below: float) -> Flag | None:
    part = _tops(called.class_name, called.carton_mm)
    support = _tops(called.stack_on, called.stack_on_carton_mm) if called.stack_on else [0.0]
    if part is None or support is None:
        return None

    def fits(expected_top: float) -> bool:
        return -below <= h - expected_top <= above

    expected = [p + s for p in part for s in support]
    if any(fits(e) for e in expected):
        return None
    nearest = min(expected, key=lambda e: abs(h - e))
    staging = _staging(called.class_name, called.carton_mm,
                       called.stack_on, called.stack_on_carton_mm)
    # The other carton size, for the target, the support or both.
    alt_part = _other_carton(called.carton_mm)
    alt_support = _other_carton(called.stack_on_carton_mm)
    swaps = []
    if alt_part:
        swaps.append((alt_part, called.stack_on_carton_mm))
    if alt_support:
        swaps.append((called.carton_mm, alt_support))
    if alt_part and alt_support:
        swaps.append((alt_part, alt_support))
    for part_mm, support_mm in swaps:
        p = _tops(called.class_name, part_mm)
        s = _tops(called.stack_on, support_mm) if called.stack_on else [0.0]
        if p and s and fits(p[0] + s[0]):
            looks = _staging(called.class_name, part_mm, called.stack_on, support_mm)
            return Flag("carton_size",
                        f"CARTON SIZE: top at {h:.0f} mm, {nearest:.0f} mm expected for "
                        f"{staging}; it reads as {looks}")
    if called.stack_on:
        advice = "check the support"
    elif h > nearest:
        advice = "check that nothing is under the part"
    else:
        # Lower than a single part stands: a part of another size, or a mirror top
        # that returned too little depth; nothing goes under a part to cause it.
        advice = "check the part"
    return Flag("stack", f"STACK: top at {h:.0f} mm, {nearest:.0f} mm expected for "
                         f"{staging}; {advice}")


def check_placement(called: CalledPose, obj: Mapping[str, Any] | None, *,
                    frames_seen: bool = True,
                    T_BC: np.ndarray | None = None,
                    table_plane: Mapping[str, Any] | None = None,
                    class_sizes: Mapping[str, Any] | None = None,
                    position_tol_mm: float = POSITION_TOL_MM,
                    yaw_tol_deg: float = YAW_TOL_DEG,
                    height_tol_above_mm: float = HEIGHT_TOL_ABOVE_MM,
                    height_tol_below_mm: float = HEIGHT_TOL_BELOW_MM,
                    height_tol_below_mirror_mm: float = HEIGHT_TOL_BELOW_MIRROR_MM,
                    ) -> list[Flag]:
    """Flags for the judged detection against the call.

    Args:
        called: the trial's called staging.
        obj: the judged detection (judged_detection of the class lock's list), or
            None when there is none.
        frames_seen: False when no perception message arrived at all; a camera
            fault says nothing about the placement, so nothing is flagged.
        T_BC, table_plane: camera-to-base matrix and table plane, for the centre.
        class_sizes: class_sizes_mm of the config, for the yaw period.
        height_tol_below_mm, height_tol_below_mirror_mm: the lower tolerance for a
            matte top and for a mirror top (MIRROR_TOP_CLASSES).

    A check whose inputs are missing (no top height on a mirror top, no table
    plane, no called yaw) is skipped without a flag.
    """
    flags: list[Flag] = []
    label = class_label(called.class_name, called.carton_mm)
    if obj is None:
        if frames_seen and called.class_name:
            where = f"the part on card {called.card_id or '?'}"
            if called.stack_on:
                where += (f" and the {class_label(called.stack_on, called.stack_on_carton_mm)}"
                          " under it")
            flags.append(Flag("class", f"CLASS: no {label} found; check {where}"))
        return flags

    centre = top_centre_xy(obj, T_BC, table_plane)
    if centre is not None and called.x_mm is not None and called.y_mm is not None:
        dist = math.hypot(centre[0] - called.x_mm, centre[1] - called.y_mm)
        if dist > position_tol_mm:
            flags.append(Flag("position", f"POSITION: part centre {dist:.0f} mm from "
                                          f"card {called.card_id or '?'}; check the card"))

    h = _num(obj.get("h_depth_mm"))
    if h is not None:
        below = lower_tolerance_mm(called.class_name, height_tol_below_mm,
                                   height_tol_below_mirror_mm)
        flag = _height_flag(called, h, height_tol_above_mm, below)
        if flag is not None:
            flags.append(flag)

    yaw = _num(obj.get("yaw_base"))
    if yaw is not None and called.yaw_deg is not None:
        err = yaw_error_deg(math.degrees(yaw), called.yaw_deg,
                            detection_yaw_period_deg(obj, called.class_name, class_sizes))
        if abs(err) > yaw_tol_deg:
            flags.append(Flag("yaw", f"YAW: part turned {abs(err):.0f} deg from the "
                                     f"called {called.yaw_deg:g} deg; check the yaw"))
    return flags
