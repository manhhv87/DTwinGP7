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
CARD_WHITE = (60, 58, 52)


def frame(upper, lower, cards=True):
    img = np.full((720, 1280, 3), 40, np.uint8)
    x0, x1, y0, y1 = CRITERIA["upper_area"]
    img[y0:y1, x0:x1] = upper
    x0, x1, y0, y1 = CRITERIA["lower_area"]
    img[y0:y1, x0:x1] = lower
    if cards:
        x0, x1, y0, y1 = CRITERIA["cards"]
        img[y0:y1, x0:x1] = CARD_WHITE
    return img


def test_green_sheet_over_the_whole_table_in_dim_light_passes():
    assert judge_hard(measure(frame(GREEN_DIM, GREEN_DIM), CRITERIA), CRITERIA) == []


def test_the_setup_of_29_october_fails_on_both_parts():
    problems = judge_hard(measure(frame(BLACK, BARE_TABLE), CRITERIA), CRITERIA)
    text = " | ".join(problems)
    assert "TABLE, UPPER PART: not covered by the green sheet" in text
    assert "TABLE, LOWER PART: not covered by the green sheet" in text
    assert "TABLE, LOWER PART: too bright" in text
    assert "TABLE, UPPER PART: too dark" in text


def test_bare_lower_part_alone_fails():
    problems = judge_hard(measure(frame(GREEN_DIM, BARE_TABLE), CRITERIA), CRITERIA)
    assert problems and all(p.startswith("TABLE, LOWER PART") for p in problems)


def test_green_sheet_in_room_light_is_too_bright():
    problems = judge_hard(measure(frame(GREEN_BRIGHT, GREEN_BRIGHT), CRITERIA), CRITERIA)
    assert problems and all("too bright" in p for p in problems)


def test_the_card_block_is_left_out():
    with_cards = measure(frame(GREEN_DIM, GREEN_DIM, cards=True), CRITERIA)
    assert with_cards["upper_area"]["green_cover"] > 0.95


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
