"""Beat-em-up instincts: face the enemy, strike, back off when surrounded."""

from nes_player.perception.motion import Slot
from nes_player.policy.instinct import InstinctPolicy


def _slot(sid: int, cx: float, cy: float, small: bool = False) -> Slot:
    return Slot(slot_id=sid, bbox=(int(cx) - 8, int(cy) - 8, 16, 16), cx=cx, cy=cy,
                small=small)


def _policy() -> InstinctPolicy:
    p = InstinctPolicy()
    p._plan = []
    return p


HERO = _slot(0, 100.0, 120.0)


def test_hits_enemy_in_range_facing_it():
    p = _policy()
    enemy = _slot(1, 120.0, 122.0)   # to the right, at contact range
    action = p._engage_step([HERO, enemy], HERO)
    assert action == frozenset({"B", "RIGHT"})

    p = _policy()
    enemy_left = _slot(2, 80.0, 118.0)
    assert p._engage_step([HERO, enemy_left], HERO) == frozenset({"B", "LEFT"})


def test_closes_distance_before_hitting():
    p = _policy()
    far = _slot(1, 138.0, 120.0)   # in our lane, but out of reach
    assert p._engage_step([HERO, far], HERO) == frozenset({"RIGHT"})


def test_backs_off_when_surrounded():
    p = _policy()
    action = p._engage_step([HERO, _slot(1, 70.0, 120.0), _slot(2, 130.0, 120.0)], HERO)
    # An unambiguous case: the left enemy is clearly nearer than the right one
    assert action in (frozenset({"RIGHT"}), frozenset({"LEFT"}))

    p = _policy()
    action = p._engage_step([HERO, _slot(1, 90.0, 120.0), _slot(2, 145.0, 120.0)], HERO)
    assert action == frozenset({"RIGHT"}), "retreat away from the nearer one"


def test_aligns_by_depth_before_hitting():
    """The fight is two-dimensional: an enemy in another lane has to be reached
    vertically first, or the strikes pass through empty air."""
    p = _policy()
    below = _slot(1, 112.0, 155.0)   # same column, greater depth
    action = p._engage_step([HERO, below], HERO)
    assert "DOWN" in action and "B" not in action, action

    p = _policy()
    above_far = _slot(2, 135.0, 90.0)   # above and to the side: approach diagonally
    action = p._engage_step([HERO, above_far], HERO)
    assert action == frozenset({"UP", "RIGHT"}), action


def test_ignores_distant_lanes_and_bullets():
    p = _policy()
    far_lane = _slot(1, 115.0, 20.0)   # too far in depth: not our fight
    assert p._engage_step([HERO, far_lane], HERO) is None

    p = _policy()
    bullet = _slot(2, 115.0, 120.0, small=True)   # small blob: a bullet, not an enemy
    assert p._engage_step([HERO, bullet], HERO) is None
