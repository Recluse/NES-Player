"""The evaluator's guarantees, without an emulator.

What has to hold, from the audit's acceptance criteria:
  - decisions land on a fixed grid of frame indices, not on a clock;
  - realtime changes the pace and nothing else;
  - an override gets the same ticks as the policy, so two arms are comparable;
  - the policy observes every frame even when it decides on few of them.
"""

import numpy as np

from nes_player.evaluation.evaluator import evaluate


class FakeObs:
    def __init__(self, i: int):
        self.frame_rgb = np.full((8, 8, 3), i % 256, np.uint8)
        self.audio_pcm = np.zeros(4, np.int16)
        self.debug = {"i": i}


class FakeEnv:
    def __init__(self):
        self.i = 0
        self.actions: list[frozenset[str]] = []

    def reset(self, seed=0):
        self.i = 0
        return FakeObs(0)

    def step_buttons(self, per_player):
        self.actions.append(frozenset(per_player[0]))
        self.i += 1
        return FakeObs(self.i)


class CountingPolicy:
    """Decides RIGHT, then LEFT, alternating; counts what it was given."""

    def __init__(self):
        self.observed = 0
        self.decided = 0
        self.seen: list[int] = []

    def reset(self):
        self.observed = self.decided = 0
        self.seen.clear()

    def observe(self, frame_rgb, audio_pcm=None):
        self.observed += 1
        self.seen.append(int(frame_rgb[0, 0, 0]))

    def decide(self, temperature=1.0):
        self.decided += 1
        return frozenset({"RIGHT" if self.decided % 2 else "LEFT"}), []


def test_every_frame_is_observed_even_when_few_are_decided():
    env, pol = FakeEnv(), CountingPolicy()
    evaluate(env, pol, frames=40, action_repeat=4)
    assert pol.observed == 40, "the frame stack must advance once per frame"
    assert pol.decided == 10


def test_decisions_land_on_a_fixed_grid():
    env, pol = FakeEnv(), CountingPolicy()
    r = evaluate(env, pol, frames=40, action_repeat=4)
    assert r.decision_frames == list(range(0, 40, 4))


def test_the_action_is_held_between_decisions():
    env, pol = FakeEnv(), CountingPolicy()
    r = evaluate(env, pol, frames=12, action_repeat=4)
    assert r.actions[:4] == [frozenset({"RIGHT"})] * 4
    assert r.actions[4:8] == [frozenset({"LEFT"})] * 4


def test_realtime_changes_the_pace_and_nothing_else():
    a = evaluate(FakeEnv(), CountingPolicy(), frames=12, action_repeat=4)
    b = evaluate(FakeEnv(), CountingPolicy(), frames=12, action_repeat=4,
                 realtime=True)
    assert a.trace() == b.trace()
    assert a.actions == b.actions


def test_an_override_gets_exactly_the_policy_ticks():
    seen: list[int] = []

    def override(i, obs, pressed):
        seen.append(i)
        return (frozenset({"A"}), "planner") if i >= 8 else None

    env, pol = FakeEnv(), CountingPolicy()
    r = evaluate(env, pol, frames=16, action_repeat=4, override=override)
    assert seen == r.decision_frames == [0, 4, 8, 12]
    assert r.sources == ["policy", "policy", "planner", "planner"]


def test_both_arms_have_the_same_opportunities():
    """The comparison the audit says was confounded: same ticks, both arms."""
    plain = evaluate(FakeEnv(), CountingPolicy(), frames=24, action_repeat=4)
    with_planner = evaluate(FakeEnv(), CountingPolicy(), frames=24, action_repeat=4,
                            override=lambda i, o, p: (frozenset({"B"}), "planner"))
    assert plain.decision_frames == with_planner.decision_frames


def test_start_is_stripped_from_what_reaches_the_controller():
    class Presser(CountingPolicy):
        def decide(self, temperature=1.0):
            return frozenset({"START", "RIGHT"}), []

    env = FakeEnv()
    r = evaluate(env, Presser(), frames=4, action_repeat=4)
    assert r.actions[0] == frozenset({"RIGHT"})
    assert r.decisions[0] == frozenset({"START", "RIGHT"}), "the record keeps the truth"


def test_idle_start_offsets_without_counting_as_frames():
    env = FakeEnv()
    r = evaluate(env, CountingPolicy(), frames=10, action_repeat=2, idle_start=7)
    assert r.frames == 10
    assert len(env.actions) == 17      # 7 idle plus 10 played
