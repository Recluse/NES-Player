"""Finding the holes in the floor, which are the one lethal thing that is not a sprite."""

import numpy as np

from nes_player.perception.terrain import (
    MIN_GAP_PX,
    background_colour,
    floor_gaps,
    gap_ahead,
)

SKY = (92, 148, 252)
BRICK = (200, 76, 12)


def _level(holes: list[tuple[int, int]] = (), bg=SKY) -> np.ndarray:
    f = np.zeros((224, 240, 3), np.uint8)
    f[:, :] = bg
    f[196:222, :] = BRICK
    for x0, x1 in holes:
        f[196:222, x0:x1] = bg
    return f


def test_solid_ground_has_no_holes():
    assert not floor_gaps(_level()).any()


def test_a_hole_is_found_where_it_is():
    g = floor_gaps(_level([(120, 160)]))
    assert g.sum() == 40
    assert g[120] and g[159] and not g[119] and not g[160]


def test_a_tile_seam_is_not_a_hole():
    """A few background-coloured columns are a seam; jumping at those means
    jumping constantly."""
    assert not floor_gaps(_level([(100, 100 + MIN_GAP_PX - 1)])).any()


def test_a_dark_level_works_the_same_way():
    """Underground the background is black and the floor is blue. The question
    is "is there anything here", not "is this the colour of a floor"."""
    g = floor_gaps(_level([(60, 90)], bg=(0, 0, 0)))
    assert g.sum() == 30


def test_a_blank_screen_is_not_one_enormous_hole():
    """A death fades to black, and black matches the background by definition."""
    assert not floor_gaps(np.zeros((224, 240, 3), np.uint8)).any()


def test_distance_is_to_the_near_edge_and_only_ahead():
    f = _level([(120, 160)])
    assert gap_ahead(f, 80, 1) == 40
    assert gap_ahead(f, 200, 1) is None       # behind us going right
    assert gap_ahead(f, 200, -1) == 41        # ahead of us going left


def test_a_hole_out_of_reach_is_not_reported():
    f = _level([(200, 230)])
    assert gap_ahead(f, 20, 1, reach=64) is None


def test_background_is_read_from_the_empty_part_of_the_screen():
    assert tuple(background_colour(_level())) == SKY
