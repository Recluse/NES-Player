"""The same moment, played four ways.

Everything the ego model had been trained on was a single history: the hero was
in some state, one button was held, and something happened. From that it is not
possible to separate what the button did from what the state was going to do
anyway — and measured, the model had not separated them. From a standstill it
ranked doing nothing above running, and the one contrast it got right was
running against walking left, which is the only pair the data makes obvious.

Weighting the frames where the button changes did not fix it, because there are
not many: about 7.8k changes in 165k frames, a mean hold of twenty frames, and
only 2369 windows that both start at a change and have a visible hero.

The emulator can do what a real robot cannot. Save the state, hold one action
for a while, restore, hold another, restore. Four branches from one moment,
identical in every respect except the button. Whatever the branches disagree
about is caused by the button, and nothing else is left to explain it.

Positions come from the sprite tracker and the phase-correlation scroll — the
same estimate the offline extractor uses, so branch data and episode data are
measured the same way and can be trained on together.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from nes_player.emulator.controller import BUTTONS
from nes_player.perception.motion import pick_hero
from nes_player.policy.planner import JUMP, LEFT, NOOP, RUN
from nes_player.world_model.ego import CROP, MAX_SCROLL, MAX_STEP, SEQ, _crop

# The actions the planner chooses between. Branching over anything else would
# be measuring a question nobody asks.
ACTIONS = [NOOP, LEFT, RUN, JUMP]
BRANCH = SEQ + 8      # long enough for a few training windows per branch
PROBE_EVERY = 20      # frames of ordinary play between branch points


def _mask(pressed: frozenset) -> int:
    return sum(1 << k for k, b in enumerate(BUTTONS) if b in pressed)


def _roll(env, tracker, hero_id: int, pressed: frozenset, steps: int):
    """Hold one action for `steps` frames from wherever the emulator is.

    The tracker is handed in already warmed up — a fresh one has no slots and
    no confidence, so it cannot name the hero for the first several frames,
    which is most of a branch.

    Who the hero is was settled before the branch, and the branch is not
    allowed to reopen the question: it follows that slot by id. The confidence
    score would say otherwise, because it is built from the hero moving the way
    the buttons say — so a branch that holds LEFT while Mario still drifts
    right reads as evidence that this is not the hero at all. First attempt
    lost him in 53% of branch frames that way.
    """
    from nes_player.perception.sprites import sprite_boxes

    crops = np.zeros((steps, CROP, CROP, 3), np.uint8)
    pos = np.zeros((steps, 2), np.float32)
    valid = np.zeros(steps, bool)
    world = 0.0
    last = (120.0, 112.0)
    prev: float | None = None
    gap = 1
    for i in range(steps):
        obs = env.step_buttons([pressed])
        frame_rgb = obs.frame_rgb
        slots = tracker.update(frame_rgb, pressed,
                               boxes=sprite_boxes(env._env.get_ram()))
        world -= float(np.clip(tracker.scroll_dx, -MAX_SCROLL, MAX_SCROLL))
        hero = next((s for s in slots if s.slot_id == hero_id), None)
        moved = (abs(world + hero.cx - prev) / gap
                 if hero is not None and prev is not None else 0.0)
        if hero is not None and hero.missed == 0 and moved <= MAX_STEP:
            last = (hero.cx, hero.cy)
            prev = world + hero.cx
            valid[i] = True
            gap = 1
        else:
            gap += 1
        crops[i] = _crop(frame_rgb, *last)
        pos[i] = (world + last[0], last[1])
    return crops, pos, valid


def collect(out_path: str | Path, checkpoint: str = "runs/hero_pre_1",
            game: str = "SuperMarioBros-Nes-v0", state: str | None = "default",
            frames: int = 6000, seed: int = 0,
            branch: int = BRANCH, probe_every: int = PROBE_EVERY) -> dict:
    """Play, and at regular moments branch the world four ways."""
    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.perception.feedback import game_over
    from nes_player.policy.go_explore import _begin
    from nes_player.policy.state_teacher import StatePolicy

    np.random.seed(seed)
    env = StableRetroAdapter(game, include_debug=True, state=state)
    obs = _begin(env)
    policy = StatePolicy(checkpoint)
    crops, pos, valid, masks = [], [], [], []
    try:
        for i in range(frames):
            # act() advances the tracker to this frame, so the copy the
            # branches start from already knows where the hero is and how
            # confident it is about him.
            pressed, _ = policy.act(obs.frame_rgb, env._env.get_ram(), 1.0)
            # Only a slot seen in this very frame will do. The tracker keeps a
            # confident hero alive for 300 missed frames so that he survives
            # being hidden behind scenery, and picking one of those ghosts
            # meant the branch followed an id that never matched again: 44% of
            # branch points came back with no hero in any of their 24 frames,
            # identically across all four actions.
            hero = pick_hero([s for s in policy.tracker._slots
                              if s.missed == 0 and s.age >= 3])
            if i % probe_every == 0 and i > 60 and hero is not None:
                here = env.save_state()
                for held in ACTIONS:
                    env.load_state(here)
                    c, p, v = _roll(env, copy.deepcopy(policy.tracker),
                                    hero.slot_id, held, branch)
                    crops.append(c)
                    pos.append(p)
                    valid.append(v)
                    masks.append(np.full(branch, _mask(held), np.int64))
                env.load_state(here)
            obs = env.step_buttons([pressed - {"START", "SELECT"}])
            # Out of lives, SMB plays its own attract-mode demo, and every
            # branch taken from there is the same footage four times over with
            # four different action labels — supervision that the buttons do
            # nothing. Unguarded, that was 488 of 604 branch points.
            if game_over(obs.debug or {}):
                obs = _begin(env)
                policy.reset()
    finally:
        env.close()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "crops": np.stack(crops),
        "pos": np.stack(pos),
        "valid": np.stack(valid),
        "masks": np.stack(masks),
    }
    np.savez_compressed(out, **arrays)
    n = len(crops)
    return {
        "path": str(out),
        "branches": n,
        "points": n // len(ACTIONS),
        "frames_per_branch": branch,
        "hero_visible": float(arrays["valid"].mean()),
    }


def load_packs(path: str | Path, vocab) -> list[dict]:
    """Branches as ego-model packs, one per branch."""
    # Each subscript of an NpzFile decompresses that whole array again, so the
    # four are pulled out once. Indexing them inside the comprehension instead
    # decompressed a gigabyte of crops per branch and the trainer was killed.
    z = np.load(Path(path))
    crops, pos, valid, masks = z["crops"], z["pos"], z["valid"], z["masks"]
    # A quad whose four branches came back byte-identical is a moment where the
    # buttons did nothing — a cutscene, a death animation, the attract-mode
    # demo. Kept, it teaches exactly the opposite of what a branch is for: the
    # loss is smallest at the action-averaged prediction. The collector's
    # game_over guard stops most of them at the source; this catches the rest,
    # and files recorded before the guard existed.
    k = len(ACTIONS)
    live = [any(not np.array_equal(crops[q * k], crops[q * k + j])
                for j in range(1, k))
            for q in range(len(crops) // k)]
    return [
        {"crops": crops[i], "pos": pos[i], "valid": valid[i],
         "labels": vocab.encode(masks[i])}
        for i in range(len(crops)) if live[i // k]
    ]
