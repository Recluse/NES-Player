"""The sprite table reader: parsing, masks, and the replay guard."""

import numpy as np
import pytest

from nes_player.perception.sprites import (
    HIDDEN_Y,
    OAM_PAGE,
    OAM_SPRITES,
    ReplayMismatch,
    sprite_boxes,
    sprite_mask,
)


def _ram_with(sprites: list[tuple[int, int]]) -> np.ndarray:
    """A RAM page holding these (x, y) sprites; the rest parked off-screen."""
    ram = np.zeros(2048, np.uint8)
    ram[OAM_PAGE:OAM_PAGE + 4 * OAM_SPRITES:4] = 0xFF        # every y hidden
    for i, (x, y) in enumerate(sprites):
        ram[OAM_PAGE + 4 * i] = y - 1        # the table stores y minus one
        ram[OAM_PAGE + 4 * i + 3] = x
    return ram


def test_hidden_sprites_are_not_reported():
    assert len(sprite_boxes(_ram_with([]))) == 0


def test_positions_round_trip():
    boxes = sprite_boxes(_ram_with([(40, 100), (200, 33)]))
    assert sorted(map(tuple, boxes)) == [(40, 100), (200, 33)]


def test_y_at_the_hidden_threshold_is_dropped():
    ram = _ram_with([(10, 20)])
    ram[OAM_PAGE] = HIDDEN_Y
    assert len(sprite_boxes(ram)) == 0


def test_mask_marks_the_cell_the_sprite_is_in():
    boxes = np.zeros((OAM_SPRITES, 2), np.uint8)
    boxes[0] = (0, 0)          # padding, must be ignored
    boxes[1] = (120, 112)
    m = sprite_mask(boxes, (224, 240), (10, 11))
    assert m.sum() >= 1
    assert m[112 * 10 // 224, 120 * 11 // 240] == 1


def test_padding_alone_gives_an_empty_mask():
    m = sprite_mask(np.zeros((OAM_SPRITES, 2), np.uint8), (224, 240), (10, 11))
    assert m.sum() == 0


def test_lead_shifts_the_mask_sideways():
    boxes = np.zeros((OAM_SPRITES, 2), np.uint8)
    boxes[0] = (40, 112)
    lead = np.zeros(OAM_SPRITES, np.int32)
    lead[0] = 120
    a = sprite_mask(boxes, (224, 240), (10, 11))
    b = sprite_mask(boxes, (224, 240), (10, 11), lead)
    assert a.sum() == b.sum() == 1
    assert not np.array_equal(a, b)


def test_a_sprite_past_the_right_edge_stays_on_screen():
    boxes = np.zeros((OAM_SPRITES, 2), np.uint8)
    boxes[0] = (250, 112)
    assert sprite_mask(boxes, (224, 240), (10, 11)).sum() == 1


def test_mismatch_says_where_it_diverged():
    with pytest.raises(ReplayMismatch, match="frame 42"):
        raise ReplayMismatch(42, 3.5)
