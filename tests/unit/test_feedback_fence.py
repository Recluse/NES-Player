"""In strict mode nothing from emulator memory may reach a button.

The audit's acceptance criterion, as a test: the same frames with the telemetry
scrambled must produce the same actions. Measured before this existed, on the
real game, 51% of actions changed — so the claim "pixels and sound only" was
false, and every dataset recorded by that policy carries RAM-conditioned
actions.
"""

import numpy as np
import pytest

from nes_player.perception.feedback import (
    Feedback,
    NoFeedback,
    PrivilegedFeedback,
    make_feedback,
)
from nes_player.policy.instinct import InstinctPolicy


def _frames(n: int, seed: int = 0):
    """Frames with something moving, so the tracker has work to do."""
    rng = np.random.default_rng(seed)
    bg = rng.integers(0, 90, (224, 240, 3), dtype=np.uint8)
    out = []
    for i in range(n):
        f = bg.copy()
        x = 40 + (i * 2) % 150            # the hero
        f[120:140, x:x + 16] = 220
        f[120:140, 200:216] = 180         # something else
        out.append(f)
    return out


def _trace(frames, mode: str, debug_maker) -> list[frozenset[str]]:
    policy = InstinctPolicy()
    policy.knowledge.jump_height = {"28": 30.0}   # skip calibration
    policy.reset()
    feedback = make_feedback(mode)
    out = []
    for i, f in enumerate(frames):
        fb = feedback.update(f, debug_maker(i))
        pressed, _, _ = policy.step(f, fb.score, fb.died)
        out.append(pressed)
    return out


def _sane(i):
    return {"score": i, "lives": 2}


def _nonsense(i):
    rng = np.random.default_rng(i)
    return {"score": int(rng.integers(0, 5000)), "lives": int(rng.integers(-1, 3))}


def test_strict_mode_ignores_the_telemetry_entirely():
    frames = _frames(400)
    assert _trace(frames, "strict", _sane) == _trace(frames, "strict", _nonsense)


def test_strict_mode_reports_nothing():
    fb = NoFeedback().update(np.zeros((224, 240, 3), np.uint8), {"score": 999, "lives": 0})
    assert fb == Feedback(0, False)


def test_privileged_mode_does_see_it():
    """The other half of the claim: the cheat channel is real when asked for."""
    frames = _frames(400)
    a = _trace(frames, "privileged", _sane)
    b = _trace(frames, "privileged", _nonsense)
    assert a != b, "privileged mode ignored the telemetry — the fence has no gate"


def test_privileged_feedback_tracks_score_and_deaths():
    f = np.zeros((224, 240, 3), np.uint8)
    src = PrivilegedFeedback()
    for _ in range(250):     # past the frame at which the baseline is taken
        src.update(f, {"score": 100, "lives": 2})
    assert src.update(f, {"score": 140, "lives": 2}).score == 40
    assert src.update(f, {"score": 140, "lives": 1}).died is True
    assert src.update(f, {"score": 140, "lives": 1}).died is False


def test_uninitialised_lives_are_not_a_death():
    """Before the game starts, the counter reads garbage."""
    f = np.zeros((224, 240, 3), np.uint8)
    src = PrivilegedFeedback()
    src.update(f, {"score": 0, "lives": 2})
    assert src.update(f, {"score": 0, "lives": 255}).died is False


def test_the_default_is_strict():
    assert isinstance(make_feedback(), NoFeedback)
    assert make_feedback().privileged is False
    assert make_feedback("privileged").privileged is True


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown feedback mode"):
        make_feedback("whatever")
