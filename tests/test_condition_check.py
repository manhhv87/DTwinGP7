"""The image-condition check of B6 step 4 (src/perception/condition_check.py)."""

import importlib.util
import pathlib

import cv2
import numpy as np
import pytest

from src.perception.condition_check import judge_hard, load_criteria, measure

ROOT = pathlib.Path(__file__).resolve().parents[1]
CRITERIA = load_criteria(ROOT / "config" / "condition_check.yaml")

GREEN_DIM = (23, 31, 11)       # BGR of the sheet in the second sample image
BLACK = (10, 9, 8)             # the near-black sheet of 29 Oct 2026
BARE_TABLE = (62, 52, 50)      # the uncovered lower part that day
GREEN_BRIGHT = (70, 130, 40)   # the green sheet under room light
UPPER_2NOV = (15, 21, 14)      # the sheet above the black strip on 2 Nov 2026
LOWER_2NOV = (20, 23, 21)      # and below it
CARDS_DIM = (60, 58, 52)
CARDS_2NOV = (92, 89, 96)
CARDS_ROOM = (250, 250, 250)


def frame(upper, lower, cards=CARDS_DIM):
    img = np.full((720, 1280, 3), 40, np.uint8)
    x0, x1, y0, y1 = CRITERIA["upper_area"]
    img[y0:y1, x0:x1] = upper
    x0, x1, y0, y1 = CRITERIA["lower_area"]
    img[y0:y1, x0:x1] = lower
    x0, x1, y0, y1 = CRITERIA["cards"]
    img[y0:y1, x0:x1] = cards
    return img


def judge(img):
    return judge_hard(measure(img, CRITERIA), CRITERIA)


def test_green_sheet_over_the_whole_table_in_dim_light_passes():
    assert judge(frame(GREEN_DIM, GREEN_DIM)) == []


def test_the_setup_of_2_november_passes():
    assert judge(frame(UPPER_2NOV, LOWER_2NOV, CARDS_2NOV)) == []


def test_the_upper_part_is_held_to_the_green_of_the_reference():
    """The grey-green of the lower part on 2 Nov would not do where the parts stand."""
    problems = judge(frame(LOWER_2NOV, LOWER_2NOV, CARDS_2NOV))
    assert len(problems) == 1 and problems[0].startswith("TABLE, UPPER PART: not green")


def test_the_setup_of_29_october_fails_on_both_parts():
    text = " | ".join(judge(frame(BLACK, BARE_TABLE, (57, 48, 47))))
    assert "TABLE, UPPER PART: not green enough" in text
    assert "TABLE, UPPER PART: too dark" in text
    assert "TABLE, LOWER PART: 100% of it is as bright as the white cards" in text
    assert "TABLE, LOWER PART: not green enough" in text
    assert "TABLE, LOWER PART: too bright" in text


def test_a_bare_patch_is_found_though_the_median_stays_green():
    img = frame(GREEN_DIM, GREEN_DIM)
    x0, x1, y0, y1 = CRITERIA["lower_area"]
    img[y0:y1, x0:x0 + (x1 - x0) // 4] = BARE_TABLE
    problems = judge(img)
    assert len(problems) == 1 and "as bright as the white cards" in problems[0]


def test_green_sheet_in_room_light_is_too_bright():
    problems = judge(frame(GREEN_BRIGHT, GREEN_BRIGHT, CARDS_ROOM))
    assert len(problems) == 2 and all("too bright" in p for p in problems)


def test_the_card_block_is_left_out():
    m = measure(frame(GREEN_DIM, GREEN_DIM), CRITERIA)
    assert m["upper_area"]["bare_share"] == 0.0
    assert m["upper_area"]["green_share"] == pytest.approx(31 / 65, abs=1e-3)
    assert m["card_brightness"] > m["upper_area"]["brightness"]


def test_the_measures_hold_under_pixel_noise():
    """2 Nov 2026: a pixel count near its threshold changed from run to run."""
    clean = frame(UPPER_2NOV, LOWER_2NOV, CARDS_2NOV).astype(float)
    shares = []
    for seed in range(5):
        noise = np.random.default_rng(seed).normal(0.0, 10.0, clean.shape)
        noisy = np.clip(clean + noise, 0, 255).astype(np.uint8)
        m = measure(noisy, CRITERIA)
        shares.append((m["upper_area"]["green_share"], m["lower_area"]["green_share"]))
    assert np.ptp(np.array(shares), axis=0).max() < 0.01


def _cli():
    spec = importlib.util.spec_from_file_location(
        "check_condition", ROOT / "scripts" / "29_check_condition.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("upper, lower, condition, code", [
    (GREEN_DIM, GREEN_DIM, "hard", 0),
    (BLACK, BARE_TABLE, "hard", 1),
    (BLACK, BARE_TABLE, "std", 0),
])
def test_cli_verdict_and_files(tmp_path, upper, lower, condition, code):
    img = tmp_path / "frame.png"
    cv2.imwrite(str(img), frame(upper, lower))
    out = tmp_path / "results"
    assert _cli().main(["--session", "t", "--condition", condition, "--image", str(img),
                        "--out-dir", str(out)]) == code
    text = (out / "do_sang_t.txt").read_text(encoding="utf-8")
    assert (out / "do_sang_t.png").exists() and (out / "do_sang_t_kiem.png").exists()
    if condition == "hard":
        assert ("PASS" if code == 0 else "FAIL") in text
