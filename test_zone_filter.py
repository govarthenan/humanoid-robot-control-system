import numpy as np
import pytest

from zone_filter import apply_gray_overlay, compute_zone_bounds, crop_to_zone, draw_zone_lines

FRAME_W = 960
FRAME_H = 720


@pytest.fixture
def bright_frame() -> np.ndarray:
    return np.full((FRAME_H, FRAME_W, 3), 200, dtype=np.uint8)


# ---------- compute_zone_bounds ----------


def test_compute_zone_bounds_standard_width():
    x1, x2 = compute_zone_bounds(960)
    assert x1 == 320
    assert x2 == 640


def test_compute_zone_bounds_small_width():
    x1, x2 = compute_zone_bounds(300)
    assert x1 == 100
    assert x2 == 200


def test_compute_zone_bounds_middle_slice_is_one_third():
    for width in (90, 960, 1280):
        x1, x2 = compute_zone_bounds(width)
        assert x2 - x1 == width // 3


def test_compute_zone_bounds_x1_less_than_x2():
    x1, x2 = compute_zone_bounds(960)
    assert x1 < x2


# ---------- crop_to_zone ----------


def test_crop_to_zone_shape(bright_frame):
    x1, x2 = compute_zone_bounds(FRAME_W)
    cropped = crop_to_zone(bright_frame, x1, x2)
    assert cropped.shape == (FRAME_H, x2 - x1, 3)


def test_crop_to_zone_is_copy(bright_frame):
    x1, x2 = compute_zone_bounds(FRAME_W)
    cropped = crop_to_zone(bright_frame, x1, x2)
    cropped[0, 0, 0] = 99
    assert bright_frame[0, x1, 0] == 200  # original unchanged


def test_crop_to_zone_correct_pixels():
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    x1, x2 = compute_zone_bounds(FRAME_W)
    frame[:, x1:x2, :] = 128
    cropped = crop_to_zone(frame, x1, x2)
    assert np.all(cropped == 128)


# ---------- apply_gray_overlay ----------


def test_apply_gray_overlay_darkens_left_side(bright_frame):
    x1, x2 = compute_zone_bounds(FRAME_W)
    apply_gray_overlay(bright_frame, x1, x2)
    assert int(bright_frame[360, 0, 0]) < 200


def test_apply_gray_overlay_darkens_right_side(bright_frame):
    x1, x2 = compute_zone_bounds(FRAME_W)
    apply_gray_overlay(bright_frame, x1, x2)
    assert int(bright_frame[360, FRAME_W - 1, 0]) < 200


def test_apply_gray_overlay_preserves_center(bright_frame):
    x1, x2 = compute_zone_bounds(FRAME_W)
    apply_gray_overlay(bright_frame, x1, x2)
    # Center column must be untouched
    assert int(bright_frame[360, FRAME_W // 2, 0]) == 200


def test_apply_gray_overlay_modifies_in_place(bright_frame):
    x1, x2 = compute_zone_bounds(FRAME_W)
    original_id = id(bright_frame)
    apply_gray_overlay(bright_frame, x1, x2)
    assert id(bright_frame) == original_id


# ---------- draw_zone_lines ----------


def test_draw_zone_lines_draws_on_left_boundary(bright_frame):
    x1, x2 = compute_zone_bounds(FRAME_W)
    draw_zone_lines(bright_frame, x1, x2)
    # Line is white (255,255,255) drawn on a 200-gray background
    center_row = FRAME_H // 2
    assert np.array_equal(bright_frame[center_row, x1], [255, 255, 255])


def test_draw_zone_lines_draws_on_right_boundary(bright_frame):
    x1, x2 = compute_zone_bounds(FRAME_W)
    draw_zone_lines(bright_frame, x1, x2)
    center_row = FRAME_H // 2
    assert np.array_equal(bright_frame[center_row, x2], [255, 255, 255])
