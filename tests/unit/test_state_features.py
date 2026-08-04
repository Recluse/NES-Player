"""The teacher's state vector, and in particular the floor part of it.

A pit is the only lethal thing on screen that is not a sprite, so it is the
only one the teacher could not see. These check it now does, and that the
answer is still well-defined on the frames where there is no hero to speak of.
"""

from dataclasses import dataclass

import numpy as np

from nes_player.policy.state_teacher import (
    GAP_REACH,
    STATE_DIM,
    GameProgress,
    features,
)

SKY = (92, 148, 252)
BRICK = (200, 76, 12)


@dataclass
class Slot:
    cx: float
    cy: float
    vx: float = 0.0
    vy: float = 0.0
    ctrl_prob: float = 0.9


def _level(holes=()) -> np.ndarray:
    f = np.zeros((224, 240, 3), np.uint8)
    f[:, :] = SKY
    f[196:222, :] = BRICK
    for x0, x1 in holes:
        f[196:222, x0:x1] = SKY
    return f


def test_solid_ground_reads_as_no_hole():
    v = features([Slot(100, 180)], 0.0, _level())
    assert len(v) == STATE_DIM
    assert v[-2] == 1.0 and v[-1] == 0.0


def test_a_hole_ahead_shows_up_as_its_distance():
    v = features([Slot(100, 180)], 0.0, _level([(148, 190)]))
    assert v[-2] == 48 / GAP_REACH
    assert v[-1] == 0.0


def test_standing_over_nothing_is_its_own_number():
    v = features([Slot(160, 180)], 0.0, _level([(148, 190)]))
    assert v[-1] == 1.0


def test_a_hole_behind_is_not_reported():
    """Going right, past the edge: survived, so nothing to say about it."""
    v = features([Slot(200, 180, vx=2.0)], 0.0, _level([(100, 140)]))
    assert v[-2] == 1.0


def test_facing_follows_the_hero():
    f = _level([(100, 140)])
    assert features([Slot(200, 180, vx=-2.0)], 0.0, f)[-2] < 1.0


def test_no_frame_and_no_hero_still_mean_no_hole():
    """Zero would read as "a hole right here", which is the opposite."""
    assert features([Slot(100, 180)], 0.0, None)[-2] == 1.0
    assert features([], 0.0, _level([(100, 140)]))[-2] == 1.0


def test_game_over_is_the_counter_going_negative():
    from nes_player.perception.feedback import game_over

    assert not game_over({"lives": 2})
    assert not game_over({"lives": 0})       # last life, still the agent playing
    assert game_over({"lives": -1})          # wrapped past zero: the demo now
    assert not game_over({})                 # a game that does not report lives
    assert not game_over(None)


def _walk(p: GameProgress, xs, level=0):
    for x in xs:
        p.update({"xscrollHi": x // 256, "xscrollLo": x % 256,
                  "levelHi": level // 4, "levelLo": level % 4}, None)


def test_walking_the_same_ground_twice_is_not_twice_the_progress():
    """Dying at 700 and walking back to 700 got as far as walking to 700 once."""
    once, twice = GameProgress(), GameProgress()
    _walk(once, range(0, 701, 20))
    _walk(twice, range(0, 701, 20))
    _walk(twice, range(0, 701, 20))          # a death puts x back to zero
    assert once.reached == twice.reached == 700
    assert twice.total > once.total          # the old measure counted it twice


def test_a_finished_level_is_banked_before_x_starts_over():
    p = GameProgress()
    _walk(p, range(0, 3201, 100), level=0)
    _walk(p, range(0, 401, 100), level=1)
    assert p.levels == 1
    assert p.reached == 3200 + 400


def test_the_rest_of_the_vector_is_untouched_by_the_floor():
    hero = Slot(120, 100)
    a = features([hero], 0.0, None)
    b = features([hero], 0.0, _level([(148, 190)]))
    assert np.array_equal(a[:-2], b[:-2])
