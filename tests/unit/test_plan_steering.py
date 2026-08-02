"""A running plan can change its direction; it cannot change its jump.

A jump is issued as a plan — hold A for N frames, then run — and nothing could
touch it while it played, so the agent could not steer in the air even when it
had every reason to. Height still belongs to the plan: releasing A early is a
different decision from steering, and conflating them would shorten every jump
that passes near anything.
"""

import numpy as np

from nes_player.perception.motion import Slot
from nes_player.policy.instinct import (
    JUMP_RUN,
    RUN,
    STEER_LANE,
    InstinctPolicy,
)


def _policy() -> InstinctPolicy:
    p = InstinctPolicy()
    p.knowledge.jump_height = {"28": 30.0}
    p.reset()
    return p


def _hero() -> Slot:
    s = Slot(0, (100, 100, 16, 16), 108.0, 108.0)
    s.ctrl_score = 20.0
    return s


def _thing(cx: float, cy: float = 108.0, vx: float = 0.0) -> Slot:
    s = Slot(1, (int(cx) - 8, int(cy) - 8, 16, 16), cx, cy, vx=vx)
    return s


def test_a_clear_path_leaves_the_plan_alone():
    p, hero = _policy(), _hero()
    assert p._steer(JUMP_RUN, hero, [hero, _thing(300.0)]) == JUMP_RUN


def test_something_close_ahead_stops_the_push_into_it():
    p, hero = _policy(), _hero()
    out = p._steer(JUMP_RUN, hero, [hero, _thing(120.0)])
    assert "RIGHT" not in out


def test_the_jump_itself_survives_the_steer():
    """A is what makes the jump tall. Steering must not shorten it."""
    p, hero = _policy(), _hero()
    out = p._steer(JUMP_RUN, hero, [hero, _thing(120.0)])
    assert "A" in out and "B" in out


def test_something_far_off_vertically_is_not_in_the_way():
    p, hero = _policy(), _hero()
    another_floor = _thing(120.0, cy=108.0 + STEER_LANE + 10)
    assert p._steer(JUMP_RUN, hero, [hero, another_floor]) == JUMP_RUN


def test_something_below_and_ahead_counts_because_that_is_a_jump():
    """Mid-jump the hero is above what it is about to land on."""
    p = _policy()
    airborne = _hero()
    airborne.cy = 60.0
    ground_thing = _thing(120.0, cy=115.0)
    assert "RIGHT" not in p._steer(JUMP_RUN, airborne, [airborne, ground_thing])


def test_something_behind_is_not_in_the_way():
    p, hero = _policy(), _hero()
    assert p._steer(JUMP_RUN, hero, [hero, _thing(80.0)]) == JUMP_RUN


def test_an_approaching_object_counts_before_it_is_close():
    p, hero = _policy(), _hero()
    coming = _thing(140.0, vx=-3.0)          # towards us, still 32 px away
    assert "RIGHT" not in p._steer(JUMP_RUN, hero, [hero, coming])


def test_a_plan_with_no_direction_is_untouched():
    p, hero = _policy(), _hero()
    only_a = frozenset({"A"})
    assert p._steer(only_a, hero, [hero, _thing(120.0)]) == only_a


def test_without_the_hero_nothing_is_steered():
    p = _policy()
    assert p._steer(RUN, None, []) == RUN


def test_the_ablation_switch_restores_the_old_behaviour():
    p, hero = _policy(), _hero()
    p.steer_running_plans = False
    assert p._steer(JUMP_RUN, hero, [hero, _thing(120.0)]) == JUMP_RUN


def test_small_blobs_are_ignored():
    """Bullets and sparkles pass through the tracker as `small`; a jump should
    not flinch at every one of them."""
    p, hero = _policy(), _hero()
    spark = _thing(120.0)
    spark.small = True
    assert p._steer(JUMP_RUN, hero, [hero, spark]) == JUMP_RUN


def test_steering_survives_a_full_step(monkeypatch):
    """End to end through step(), not just the helper.

    A jump is in the air, so the fight rules stand aside and the plan is
    steered instead of cancelled.
    """
    p = _policy()
    frame = np.zeros((224, 240, 3), np.uint8)
    hero = _hero()
    hero.cy = 70.0
    ahead = _thing(120.0, cy=112.0)
    monkeypatch.setattr(p.tracker, "update", lambda *a, **k: [hero, ahead])
    monkeypatch.setattr(p.memory, "update", lambda *a, **k: {})
    p._plan = [(JUMP_RUN, 10)]
    pressed, _, _ = p.step(frame, 0, False)
    assert "A" in pressed and "RIGHT" not in pressed


def test_a_fight_still_takes_priority_over_a_ground_manoeuvre(monkeypatch):
    """The reason engage sits above the plan queue at all: a retreat must not
    keep running for 135 frames while an enemy stands next to us."""
    from nes_player.policy.instinct import LEFT

    p = _policy()
    frame = np.zeros((224, 240, 3), np.uint8)
    hero, ahead = _hero(), _thing(120.0)
    monkeypatch.setattr(p.tracker, "update", lambda *a, **k: [hero, ahead])
    monkeypatch.setattr(p.memory, "update", lambda *a, **k: {})
    p._plan = [(LEFT, 100)]           # a retreat, not a jump
    p.step(frame, 0, False)
    assert not p._plan, "engaging must discard a ground manoeuvre"


def test_a_jump_in_the_air_is_not_cancelled_by_the_fight_rules(monkeypatch):
    p = _policy()
    frame = np.zeros((224, 240, 3), np.uint8)
    hero, ahead = _hero(), _thing(120.0)
    monkeypatch.setattr(p.tracker, "update", lambda *a, **k: [hero, ahead])
    monkeypatch.setattr(p.memory, "update", lambda *a, **k: {})
    p._plan = [(JUMP_RUN, 10)]
    pressed, _, _ = p.step(frame, 0, False)
    assert p._plan, "a committed jump must keep its remaining frames"
    assert "A" in pressed
