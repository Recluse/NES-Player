"""No policy may press two opposite directions at once.

TAS movies record LEFT+RIGHT because the emulator accepts the bits, and cloning
them put that combination into the action vocabulary of 24 checkpoints. The
archive is allowed to keep it; a controller is not.
"""

import numpy as np

from nes_player.emulator.controller import BUTTONS, resolve_conflicts
from nes_player.policy.bc import ActionVocab, mask_to_pressed, normalise_mask

LEFT_RIGHT = (1 << BUTTONS.index("LEFT")) | (1 << BUTTONS.index("RIGHT"))
UP_DOWN = (1 << BUTTONS.index("UP")) | (1 << BUTTONS.index("DOWN"))
B_BIT = 1 << BUTTONS.index("B")


def test_opposites_cancel_rather_than_pick_a_side():
    assert resolve_conflicts({"LEFT", "RIGHT"}) == frozenset()
    assert resolve_conflicts({"UP", "DOWN"}) == frozenset()


def test_other_buttons_survive():
    assert resolve_conflicts({"LEFT", "RIGHT", "B"}) == frozenset({"B"})


def test_a_single_direction_is_untouched():
    assert resolve_conflicts({"LEFT", "B"}) == frozenset({"LEFT", "B"})


def test_both_axes_at_once():
    assert resolve_conflicts({"LEFT", "RIGHT", "UP", "DOWN", "A"}) == frozenset({"A"})


def test_a_checkpoint_carrying_the_bad_mask_cannot_emit_it():
    """192 and 193 are in 24 of our vocabularies; loading them stays possible."""
    assert mask_to_pressed(LEFT_RIGHT) == frozenset()
    assert mask_to_pressed(LEFT_RIGHT | B_BIT) == frozenset({"B"})


def test_vocab_built_from_tas_actions_has_no_impossible_entries():
    actions = np.array([0, LEFT_RIGHT, LEFT_RIGHT | B_BIT, UP_DOWN, B_BIT], np.int64)
    vocab = ActionVocab.from_actions(actions)
    for m in vocab.masks:
        assert normalise_mask(m) == m
    # every label still encodes, to the possible action nearest what was recorded
    idx = vocab.encode(actions)
    assert len(idx) == len(actions)
    assert vocab.masks[idx[1]] == 0            # LEFT+RIGHT -> nothing
    assert vocab.masks[idx[2]] == B_BIT        # LEFT+RIGHT+B -> B


def test_normalise_is_idempotent():
    for m in range(256):
        assert normalise_mask(normalise_mask(m)) == normalise_mask(m)
