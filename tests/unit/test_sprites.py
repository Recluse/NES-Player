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


def _ram_with(sprites: list[tuple]) -> np.ndarray:
    """A RAM page holding these (x, y) or (x, y, tile) sprites."""
    ram = np.zeros(2048, np.uint8)
    ram[OAM_PAGE:OAM_PAGE + 4 * OAM_SPRITES:4] = 0xFF        # every y hidden
    for i, sp in enumerate(sprites):
        x, y = sp[0], sp[1]
        ram[OAM_PAGE + 4 * i] = y - 1        # the table stores y minus one
        ram[OAM_PAGE + 4 * i + 1] = sp[2] if len(sp) > 2 else 0
        ram[OAM_PAGE + 4 * i + 3] = x
    return ram


def _boxes(pairs: list[tuple[int, int]]) -> np.ndarray:
    """A padded table of (x, y, tile) rows, as the cache stores them."""
    out = np.zeros((len(pairs), 3), np.uint8)
    for i, (x, y) in enumerate(pairs):
        out[i] = (x, y, 0)
    return out


def test_hidden_sprites_are_not_reported():
    assert len(sprite_boxes(_ram_with([]))) == 0


def test_positions_round_trip():
    boxes = sprite_boxes(_ram_with([(40, 100), (200, 33)]))
    assert sorted((int(x), int(y)) for x, y, _ in boxes) == [(40, 100), (200, 33)]


def test_y_at_the_hidden_threshold_is_dropped():
    ram = _ram_with([(10, 20)])
    ram[OAM_PAGE] = HIDDEN_Y
    assert len(sprite_boxes(ram)) == 0


def test_mask_marks_the_cell_the_sprite_is_in():
    boxes = np.zeros((OAM_SPRITES, 3), np.uint8)
    boxes[0] = (0, 0, 0)       # padding, must be ignored
    boxes[1] = (120, 112, 0)
    m = sprite_mask(boxes, (224, 240), (10, 11))
    assert m.sum() >= 1
    assert m[112 * 10 // 224, 120 * 11 // 240] == 1


def test_padding_alone_gives_an_empty_mask():
    m = sprite_mask(np.zeros((OAM_SPRITES, 3), np.uint8), (224, 240), (10, 11))
    assert m.sum() == 0


def test_lead_shifts_the_mask_sideways():
    boxes = np.zeros((OAM_SPRITES, 3), np.uint8)
    boxes[0] = (40, 112, 0)
    lead = np.zeros(OAM_SPRITES, np.int32)
    lead[0] = 120
    a = sprite_mask(boxes, (224, 240), (10, 11))
    b = sprite_mask(boxes, (224, 240), (10, 11), lead)
    assert a.sum() == b.sum() == 1
    assert not np.array_equal(a, b)


def test_a_sprite_past_the_right_edge_stays_on_screen():
    boxes = np.zeros((OAM_SPRITES, 3), np.uint8)
    boxes[0] = (250, 112, 0)
    assert sprite_mask(boxes, (224, 240), (10, 11)).sum() == 1


def test_mismatch_says_where_it_diverged():
    with pytest.raises(ReplayMismatch, match="frame 42"):
        raise ReplayMismatch(42, 3.5)


def test_the_tile_index_survives_into_the_table():
    """It used to be dropped on the first line, leaving identity to be guessed
    from a pixel crop that is mostly background."""
    boxes = sprite_boxes(_ram_with([(40, 100, 0x70)]))
    assert (int(boxes[0][0]), int(boxes[0][1]), int(boxes[0][2])) == (40, 100, 0x70)


def test_an_object_carries_the_tiles_it_is_drawn_from():
    """Four adjacent sprites are one object, and its identity is their tiles."""
    from nes_player.perception.sprites import SpriteTracker

    ram = _ram_with([(120, 100, 0x70), (128, 100, 0x71),
                     (120, 108, 0x72), (128, 108, 0x73)])
    frame = np.zeros((224, 240, 3), np.uint8)
    t = SpriteTracker()
    # The tracker only reports a slot once it has survived a few frames.
    for _ in range(6):
        slots = t.update(frame, frozenset(), ram)
    assert slots, "the four sprites should form one object"
    assert slots[0].tiles == frozenset({0x70, 0x71, 0x72, 0x73})


def test_two_objects_apart_do_not_share_tiles():
    from nes_player.perception.sprites import SpriteTracker

    ram = _ram_with([(40, 100, 0x70), (41, 100, 0x71),
                     (200, 100, 0x9E), (201, 100, 0x9F)])
    frame = np.zeros((224, 240, 3), np.uint8)
    t = SpriteTracker()
    for _ in range(6):
        slots = t.update(frame, frozenset(), ram)
    sets = sorted((sorted(s.tiles) for s in slots), key=lambda v: v[0])
    assert sets == [[0x70, 0x71], [0x9E, 0x9F]]


def test_rising_while_the_jump_button_is_held_counts_as_control():
    """Sideways evidence dries up when the player is not going anywhere, and
    this agent stalls constantly. Nothing else on screen rises because A was
    pressed."""
    from nes_player.perception.sprites import SpriteTracker

    frame = np.zeros((224, 240, 3), np.uint8)
    ram = _ram_with([(120, 150, 0x32), (128, 150, 0x33)])
    t = SpriteTracker()
    for _ in range(6):
        t.update(frame, frozenset(), ram)
    before = t._slots[0].ctrl_score
    t._slots[0].vy = -2.0                    # rising
    t.update(frame, frozenset({"A"}), ram)
    assert t._slots[0].ctrl_score > before


def test_falling_while_jumping_is_not_evidence():
    """Everything falls; only going up is a response to the button."""
    from nes_player.perception.sprites import SpriteTracker

    frame = np.zeros((224, 240, 3), np.uint8)
    ram = _ram_with([(120, 150, 0x32), (128, 150, 0x33)])
    t = SpriteTracker()
    for _ in range(6):
        t.update(frame, frozenset(), ram)
    before = t._slots[0].ctrl_score
    t._slots[0].vy = +2.0                    # falling
    t.update(frame, frozenset({"A"}), ram)
    assert t._slots[0].ctrl_score <= before
