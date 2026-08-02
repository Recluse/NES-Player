"""A contact is only to blame for what happened soon after it, in this episode.

Both halves used to be wrong: a death was credited to every pending contact
regardless of age, and pending contacts survived a reset with episode-local
frame numbers, so the first death of a new episode was blamed on an object
from the old one.
"""

import numpy as np

from nes_player.perception.memory import EFFECT_WINDOW, ObjectMemory
from nes_player.perception.motion import Slot


def _frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (224, 240, 3), dtype=np.uint8)


def _slots(touching: bool) -> list[Slot]:
    hero = Slot(0, (100, 100, 16, 16), 108.0, 108.0)
    hero.ctrl_score = 20.0        # ctrl_prob ~ 1
    other = Slot(1, (110, 100, 16, 16), 118.0 if touching else 220.0, 108.0)
    return [hero, other]


def _contact(mem: ObjectMemory, frame: np.ndarray, at: int) -> int:
    """Make the hero touch the other object; return that object's cluster id."""
    mem.update(frame, _slots(True), at, score=0, died=False)
    return mem._slot_cluster[1]


def test_a_death_long_after_the_contact_is_not_blamed_on_it():
    mem, f = ObjectMemory(), _frame()
    cid = _contact(mem, f, 0)
    before = mem.clusters[cid].deaths
    mem.update(f, _slots(False), EFFECT_WINDOW + 10, score=0, died=True)
    assert mem.clusters[cid].deaths == before


def test_a_death_inside_the_window_is_blamed_on_it():
    mem, f = ObjectMemory(), _frame()
    cid = _contact(mem, f, 0)
    before = mem.clusters[cid].deaths
    mem.update(f, _slots(False), EFFECT_WINDOW - 1, score=0, died=True)
    assert mem.clusters[cid].deaths == before + 1


def test_one_contact_is_blamed_for_one_death_only():
    mem, f = ObjectMemory(), _frame()
    cid = _contact(mem, f, 0)
    before = mem.clusters[cid].deaths
    for at in (5, 10, 15):
        mem.update(f, _slots(False), at, score=0, died=True)
    assert mem.clusters[cid].deaths == before + 1


def test_pending_contacts_do_not_cross_an_episode_boundary():
    mem, f = ObjectMemory(), _frame()
    cid = _contact(mem, f, 3599)
    before = mem.clusters[cid].deaths
    mem.begin_episode()
    mem.update(f, _slots(False), 1, score=0, died=True)
    assert mem.clusters[cid].deaths == before


def test_begin_episode_keeps_what_was_learned():
    mem, f = ObjectMemory(), _frame()
    _contact(mem, f, 0)
    n = len(mem.clusters)
    mem.begin_episode()
    assert len(mem.clusters) == n
