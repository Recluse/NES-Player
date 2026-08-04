"""The archive: what counts as the same place, and which place to go back to.

The emulator half of Go-Explore needs a ROM and cannot run here. The
bookkeeping half is where the mistakes live — a cell key that folds in
something it should not, an entry that keeps the first visit instead of the
best one, a selector that never leaves the opening screen — and all of that is
plain data.
"""

import numpy as np

from nes_player.policy.go_explore import (
    CELL_X,
    Archive,
    Entry,
    cell_of,
    mask_of,
    xpos_of,
)


def dbg(x: int, level: int = 0, lives: int = 2, score: int = 0) -> dict:
    return {"xscrollHi": x // 256, "xscrollLo": x % 256,
            "levelHi": level // 4, "levelLo": level % 4,
            "lives": lives, "score": score}


def entry(x: int, level: int = 0, lives: int = 2, score: int = 0,
          actions=()) -> Entry:
    d = dbg(x, level, lives, score)
    return Entry(cell_of(d), b"", list(actions), x, score, lives)


def test_x_survives_the_two_byte_split():
    assert xpos_of(dbg(700)) == 700
    assert xpos_of(dbg(3265)) == 3265


def test_the_same_stretch_of_level_is_one_cell():
    assert cell_of(dbg(640)) == cell_of(dbg(640 + CELL_X - 1))
    assert cell_of(dbg(640)) != cell_of(dbg(640 + CELL_X))


def test_the_same_place_in_a_later_level_is_a_different_cell():
    assert cell_of(dbg(100, level=0)) != cell_of(dbg(100, level=1))


def test_lives_do_not_split_a_cell():
    """Otherwise the archive fills with copies of the opening, one per life."""
    assert cell_of(dbg(300, lives=2)) == cell_of(dbg(300, lives=0))


def test_a_new_cell_is_taken():
    a = Archive()
    assert a.consider(entry(100))
    assert len(a.entries) == 1


def test_arriving_in_better_shape_replaces_the_entry():
    a = Archive()
    a.consider(entry(650, lives=0))
    assert a.consider(entry(650, lives=2))
    assert a.entries[cell_of(dbg(650))].lives == 2


def test_arriving_worse_does_not():
    a = Archive()
    a.consider(entry(650, lives=2))
    assert not a.consider(entry(650, lives=0))
    assert a.entries[cell_of(dbg(650))].lives == 2


def test_replacing_keeps_how_often_the_cell_was_used():
    """Losing the count would make a well-worn cell look untouched and the
    selector would keep going back to it."""
    a = Archive()
    a.consider(entry(650, lives=0))
    a.entries[cell_of(dbg(650))].chosen = 7
    a.consider(entry(650, lives=2))
    assert a.entries[cell_of(dbg(650))].chosen == 7


def test_the_frontier_is_the_furthest_entry():
    a = Archive()
    for x in (100, 900, 400):
        a.consider(entry(x))
    assert a.frontier == 900


def test_the_best_entry_prefers_a_later_level_over_a_bigger_x():
    a = Archive()
    a.consider(entry(3200, level=0))
    a.consider(entry(150, level=1))
    assert a.best().cell[0] == 1


def test_the_selector_favours_places_it_has_used_less():
    a = Archive()
    for x in (100, 200):
        a.consider(entry(x))
    a.entries[cell_of(dbg(100))].chosen = 50
    rng = np.random.default_rng(0)
    picks = [a.pick(rng).x for _ in range(200)]
    assert picks.count(200) > picks.count(100) * 3


def test_the_selector_can_still_reach_every_cell():
    """A weight of zero anywhere turns a dead end into a permanent stop."""
    a = Archive()
    for x in (100, 2000):
        a.consider(entry(x))
    rng = np.random.default_rng(0)
    assert {p.x for p in (a.pick(rng) for _ in range(300))} == {100, 2000}


def test_masks_match_the_controller_bit_order():
    from nes_player.emulator.controller import BUTTONS

    assert mask_of(frozenset({"A"})) == 1 << BUTTONS.index("A")
    assert mask_of(frozenset()) == 0
    assert mask_of(frozenset({"RIGHT", "B"})) == (
        1 << BUTTONS.index("RIGHT")) | (1 << BUTTONS.index("B"))


def test_the_frontier_counts_each_level_once():
    """Plain max-x cannot express progress past the end of a level, because the
    next one starts back at zero."""
    a = Archive()
    a.consider(entry(3100, level=0))
    a.consider(entry(200, level=1))
    assert a.frontier == 3300


def test_a_freshly_reached_level_is_not_starved():
    """x restarts at zero in a new level, so judging "furthest" globally rates
    its first cells below every cell of the level just finished."""
    a = Archive()
    for x in (500, 1500, 2500, 3100):
        a.consider(entry(x, level=0))
    a.consider(entry(120, level=1))
    rng = np.random.default_rng(0)
    picks = [a.pick(rng).cell[0] for _ in range(300)]
    assert picks.count(1) > picks.count(0)


def test_a_death_is_always_written():
    """The scarce label: nothing can be called dangerous without one."""
    from nes_player.policy.go_explore import keep_segment

    assert keep_segment("died", {"died": 500, "alive": 0}, 0.5)


def test_survivals_are_sampled_down():
    from nes_player.policy.go_explore import keep_segment

    assert keep_segment("alive", {"died": 0, "alive": 0}, 0.5)
    assert not keep_segment("alive", {"died": 1, "alive": 9}, 0.5)


def test_a_ratio_of_zero_keeps_only_deaths():
    from nes_player.policy.go_explore import keep_segment

    assert not keep_segment("alive", {"died": 0, "alive": 0}, 0.0)
    assert keep_segment("died", {"died": 0, "alive": 0}, 0.0)
