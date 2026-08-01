"""Tests for the per-frame helpers shared by `play` and `explore`.

These used to be inline copies in two 300-line command functions, where none of
it could be tested. The jump shaper in particular had a real bug — a hold that
restarted itself so A was never released — which is what the first test pins.
"""

import numpy as np

from nes_player.cli.play import JumpShaper
from nes_player.cli.runtime import PauseWatchdog, PingLog, SoundLog, Thoughts, action_entropy
from nes_player.perception.title import TitleWatch


def test_jump_hold_starts_on_edge_and_always_releases():
    jump = JumpShaper(hold=4)
    a = frozenset({"A"})
    # The model holds A forever; the shaper must still let go.
    out = [jump.apply(a) for _ in range(20)]
    held = ["A" in p for p in out]
    assert held[:5] == [True] * 5, "edge frame plus `hold` frames"
    assert not any(held[5:13]), "a release window must follow every hold"
    assert not all(held), "a stuck A would mean the jump never ends"


def test_jump_ignores_a_while_held_then_rearms():
    jump = JumpShaper(hold=2)
    a, none = frozenset({"A"}), frozenset()
    seq = [a, a, a] + [none] * 10 + [a]
    held = ["A" in jump.apply(p) for p in seq]
    assert held[:2] == [True, True]
    assert held[-1] is True, "after a full release the next press must jump again"


def test_jump_passes_other_buttons_through():
    jump = JumpShaper(hold=2)
    out = jump.apply(frozenset({"RIGHT", "B"}))
    assert out == frozenset({"RIGHT", "B"})


def test_pause_watchdog_fires_only_on_identical_frames():
    watch = PauseWatchdog(frames=3)
    still = np.zeros((8, 8, 3), dtype=np.uint8)
    moving = np.ones((8, 8, 3), dtype=np.uint8)
    assert watch.push(still) is False           # first frame, nothing to compare
    assert watch.push(still) is False
    assert watch.push(still) is False
    assert watch.push(still) is True, "three identical frames means paused"
    watch.push(moving)
    assert watch.push(still) is False, "movement must reset the counter"


def test_sound_log_keeps_recent_events_with_ages():
    log = SoundLog(keep=3)
    for cid in range(5):
        log.add(cid)
        log.tick()
    assert [e[0] for e in log.events] == [4, 3, 2], "newest first, oldest dropped"
    assert [e[1] for e in log.events] == [1, 2, 3], "age counts frames since the sound"
    assert log.within(2) == [4]


def test_ping_log_expires_and_caps():
    pings = PingLog(ttl=2, keep=2)
    for _ in range(4):
        pings.add(0.5, 0.5)
    for _ in range(3):
        pings.tick()
    assert pings.pings == [], "pings must fade out"


def test_thoughts_stay_bounded():
    log = Thoughts(limit=10, keep=4)
    for i in range(100):
        log.add(str(i))
    assert len(log.lines) <= 10
    assert log.lines[-1] == "99", "the newest line always survives"


def test_action_entropy_range():
    assert action_entropy([("A", 0.5), ("B", 0.5)]) == 1.0        # maximum doubt
    assert action_entropy([("A", 1.0), ("B", 0.0)]) == 0.0        # certainty


def test_title_watch_refuses_a_moving_screen():
    """The Double Dragon bug: a savestate start must not memorise gameplay."""
    watch = TitleWatch()
    rng = np.random.default_rng(0)
    for _ in range(5):
        moving = rng.integers(0, 255, (224, 240, 3), dtype=np.uint8)
        watch.capture(moving)
    assert watch.sig is None, "a changing screen is not a title screen"

    still = rng.integers(0, 255, (224, 240, 3), dtype=np.uint8)
    watch.capture(still)
    assert watch.capture(still) is True
    assert watch.at_title(still) is True
