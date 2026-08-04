"""Whichever slot is "me", it is not the one parked in the status bar.

Super Mario Bros. keeps sprite 0 in the status bar to time its scroll split.
It never moves, and the tracker gives it a control probability of exactly
1.00 — the same as Mario — so `max(slots, key=ctrl_prob)` picks between them
by tie-break. Measured on a live run it picks the status bar in 4% of frames,
and in those frames every relative position in the state vector is measured
from a fixed point in the interface and the floor scan starts from the wrong
column.

No NES game puts the player inside the status bar, so the rule is not specific
to this one.
"""

from dataclasses import dataclass

import pytest

from nes_player.perception.motion import HERO_MIN_CY, pick_hero


@dataclass
class Slot:
    cx: float
    cy: float
    ctrl_prob: float = 1.0


def test_the_status_bar_sprite_is_never_the_hero():
    """The exact tie seen live: both at 1.00, one of them at cy=32.5."""
    hud = Slot(91.5, 32.5, 1.00)
    mario = Slot(119.5, 156.5, 1.00)
    assert pick_hero([hud, mario]) is mario
    assert pick_hero([mario, hud]) is mario


def test_the_highest_jump_still_counts_as_the_hero():
    """A tall jump lifts Mario well up the screen; that is not the HUD."""
    high = Slot(119.5, HERO_MIN_CY + 1, 0.9)
    assert pick_hero([high]) is high


def test_an_unconvincing_slot_is_not_the_hero():
    assert pick_hero([Slot(100, 150, 0.2)]) is None


def test_nothing_on_screen_means_no_hero():
    assert pick_hero([]) is None


def test_the_most_controlled_slot_wins_when_none_are_in_the_hud():
    a, b = Slot(100, 150, 0.7), Slot(130, 150, 0.95)
    assert pick_hero([a, b]) is b


def test_a_hud_slot_does_not_shadow_a_real_one_below_it():
    """Only the HUD candidate is dropped, not the whole frame."""
    assert pick_hero([Slot(91.5, 10, 1.0), Slot(120, 190, 0.6)]).cy == 190


@pytest.mark.parametrize("cy", [0, 20, HERO_MIN_CY - 1])
def test_anything_in_the_top_band_is_interface(cy):
    assert pick_hero([Slot(90, cy, 1.0)]) is None
