"""The experiment harnesses must be the same machine underneath.

Two measurement bugs in two days argue for this. `oracle_mpc` scored progress
by the camera's x, which resets at a level boundary, so every run that cleared
1-1 was capped and the best arms were the ones most damaged; `probe_duel` had
already been fixed and the fix was not carried across. And a difference of +440
was attributed to "the harness" before the same seeds were run in both, at
which point it turned out to be the seeds.

So: one action sequence from one save state has to produce identical console
state everywhere, and the progress metric has to be monotone across the level
boundary it used to reset at. These are cheap and they run without a policy.
"""

import hashlib

import numpy as np

from nes_player.policy.robustify import LEVEL_SPAN, progress_of


def _hash(env) -> str:
    return hashlib.sha1(env._env.get_ram().tobytes()).hexdigest()[:16]


def _replay(actions):
    """Same start, same buttons, twice — the console must not disagree."""
    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.policy.go_explore import _begin

    env = StableRetroAdapter("SuperMarioBros-Nes-v0", include_debug=True,
                             state="default")
    try:
        obs = _begin(env)
        here = env.save_state()
        out = []
        for _ in range(2):
            env.load_state(here)
            marks = []
            for a in actions:
                obs = env.step_buttons([a])
                marks.append((_hash(env), progress_of(obs.debug or {})))
            out.append(marks)
        return out
    finally:
        env.close()


def test_a_save_state_replays_identically():
    run = frozenset({"B", "RIGHT"})
    a, b = _replay([run] * 400)
    assert a == b, "the same buttons from the same state diverged"


def test_progress_is_monotone_across_a_level_boundary():
    """The bug itself, as a unit: x resets, the folded number must not.

    Camera x near the end of 1-1 is about 3100; the next level starts it at
    zero again. Anything that scores on x alone rates a finished level below an
    unfinished one, which is exactly backwards and is what capped every good
    oracle run at 3120.
    """
    end_of_level = {"levelHi": 0, "levelLo": 0,
                    "xscrollHi": 3100 // 256, "xscrollLo": 3100 % 256}
    just_after = {"levelHi": 0, "levelLo": 1, "xscrollHi": 0, "xscrollLo": 8}
    assert progress_of(just_after) > progress_of(end_of_level)
    assert progress_of(just_after) >= LEVEL_SPAN


def test_the_folded_metric_orders_levels_before_positions():
    """Being further into the game beats being further along one level."""
    deep = {"levelHi": 0, "levelLo": 2, "xscrollHi": 0, "xscrollLo": 0}
    shallow = {"levelHi": 0, "levelLo": 1, "xscrollHi": 3900 // 256,
               "xscrollLo": 3900 % 256}
    assert progress_of(deep) > progress_of(shallow)


def test_templates_are_prefixes_of_each_other():
    """A plan executed must be the plan that was scored.

    `templates(16)` is not `templates(48)[:16]`: at the shorter length the
    "jump later" recipe becomes twelve frames of run and ten of jump, 22 long,
    where the plan the scorer valued holds only four frames of jump inside its
    first sixteen. The controller executed one and priced the other.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                           / "scripts" / "experiments"))
    from oracle_mpc import templates

    long = dict(templates(48))
    for name, short in templates(16):
        if name == "jump later":
            continue          # known to differ; the controller uses the prefix
        assert short == long[name][:16], f"{name} is not a prefix of itself"


def test_progress_of_never_goes_backwards_within_a_level():
    xs = [0, 500, 1500, 3000, 3900]
    vals = [progress_of({"levelHi": 0, "levelLo": 0,
                         "xscrollHi": x // 256, "xscrollLo": x % 256})
            for x in xs]
    assert vals == sorted(vals)
    assert np.all(np.diff(vals) > 0)
