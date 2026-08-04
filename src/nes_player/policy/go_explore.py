"""Reach further by remembering where you got to, and going back there.

Every attempt to make this agent play further has failed in the same way, and
the measurements say why. On one life it reaches x 650-720 of a 3266-pixel
level and dies; the rest of Super Mario Bros. 1-1 is territory it has never
been in. Reweighting the frames it already has cannot teach what is not in
them, and softmax sampling from a cloned policy is far too weak an explorer to
cross four hundred pixels of unseen ground by luck.

Go-Explore (Ecoffet et al., Nature 2021) attacks exactly that: keep an archive
of states reached, *return* to a promising one by restoring it rather than by
replaying it, and explore onward from there. The frontier advances by saving
and reloading, so the hard part of the level stops being unreachable.

Two things make it cheap here. The core is deterministic and the adapter
already exposes `save_state`/`load_state`, so returning costs one memcpy
instead of a replay. And progress is a *counted* fact — how many cells the
archive holds, how far the furthest one is — rather than a mean over noisy
rollouts, which matters in a project where one pretrain varies by 282 points
from its seed alone.

On the observation fence (spec §3): restoring emulator states and reading the
level position out of RAM are *search-time* tools, used to find a trajectory.
They are not available to the agent that plays. Phase two clones the found
trajectory into an ordinary policy, and that policy sees pixels like every
other. This is the same fence the privileged teacher sits behind, and it is
worth saying out loud because a save-state is easy to mistake for cheating
later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nes_player.perception.feedback import game_over
from nes_player.perception.sprites import SPRITES_VERSION, _pad, sprite_boxes

# Pixels of level per cell. Small enough that "further" is a frequent event,
# wide enough that the archive does not fill with the same corridor.
CELL_X = 64
# Long enough to cross the end of a level. The first search plateaued at a
# frontier of 3130 for fourteen hundred iterations and it was not stuck: it had
# reached the flagpole on iteration 100 and finished 1-1. Flag, descent, walk
# into the castle and the WORLD 1-2 screen take longer than 300 frames, so
# every segment ended inside the celebration and the next level was never seen.
EXPLORE_FRAMES = 900
# Actions are held for a stretch rather than resampled every frame. A NES jump
# is a held button, and an explorer that redraws sixty times a second never
# holds anything long enough to leave the ground — the same reason the teacher
# could not clear the first pipe until the jump was shaped.
HOLD_MIN, HOLD_MAX = 4, 16

# Button sets worth trying, weighted. Standing still is nearly useless and
# going left almost always is, but neither is zero: some rooms need a step back
# before a run-up.
ACTIONS: tuple[tuple[frozenset[str], float], ...] = (
    (frozenset({"RIGHT", "B"}), 4.0),
    (frozenset({"RIGHT", "B", "A"}), 4.0),
    (frozenset({"RIGHT"}), 2.0),
    (frozenset({"RIGHT", "A"}), 2.0),
    (frozenset({"A"}), 1.0),
    (frozenset({"LEFT"}), 0.5),
    (frozenset(), 0.5),
)


def level_of(debug: dict) -> int:
    return int(debug.get("levelHi", 0)) * 4 + int(debug.get("levelLo", 0))


def xpos_of(debug: dict) -> int:
    return int(debug.get("xscrollHi", 0)) * 256 + int(debug.get("xscrollLo", 0))


def cell_of(debug: dict) -> tuple[int, int, int]:
    """Where in the game this is, coarsely.

    Lives are deliberately not part of it: the same place reached with fewer
    lives is the same place, and folding lives in would fill the archive with
    copies of the opening.

    The clock is, in coarse steps. `xscroll` is the *camera*, and the camera
    stops before the end of a level while the hero keeps going, so every state
    on the last stretch collapses into one cell and selection loses its
    gradient exactly where the interesting part is. Time still moves there, and
    on the flag it moves fast, which separates approaching the flagpole from
    having reached it.
    """
    return (level_of(debug), xpos_of(debug) // CELL_X,
            int(debug.get("time", 0)) // 25)


@dataclass
class Entry:
    """One remembered place, and how to get back to it."""

    cell: tuple[int, int, int]
    state: bytes
    actions: list[int]        # button masks from the very start, for phase two
    x: int
    score: int
    lives: int
    chosen: int = 0           # times explored from; the selector prefers few


@dataclass
class Archive:
    """The frontier, keyed by cell.

    An entry is replaced when a later visit arrives in better shape — further
    into the cell, or as well placed but with more lives in hand. Without that
    the archive keeps whichever visit happened to be first, which is usually
    the one that arrived dying.
    """

    entries: dict[tuple[int, int, int], Entry] = field(default_factory=dict)
    found: int = 0            # cells seen for the first time, ever

    def consider(self, e: Entry) -> bool:
        """Offer an entry; returns whether the archive took it."""
        old = self.entries.get(e.cell)
        if old is None:
            self.entries[e.cell] = e
            self.found += 1
            return True
        if (e.lives, e.x, e.score) > (old.lives, old.x, old.score):
            e.chosen = old.chosen
            self.entries[e.cell] = e
            return True
        return False

    def _reach(self) -> dict[int, int]:
        """Furthest x seen in each level."""
        out: dict[int, int] = {}
        for e in self.entries.values():
            out[e.cell[0]] = max(out.get(e.cell[0], 0), e.x)
        return out

    def pick(self, rng: np.random.Generator) -> Entry:
        """A promising place to go back to.

        Weighted towards cells explored from least and towards the frontier.
        Purely taking the furthest cell gets stuck the moment the frontier is a
        dead end; purely uniform wastes its time in the opening, which is
        already solved.

        "Furthest" has to be judged inside a level. x restarts at zero in a new
        one, so a global comparison rates the first frames of a freshly reached
        level below every cell of the level just finished — and the search
        starves the very frontier it just opened. Measured: after reaching 1-2
        the archive gained five cells in two hundred iterations.
        """
        es = list(self.entries.values())
        reach = self._reach()
        top = max(reach)
        w = np.array([(1.0 / (1 + e.chosen))
                      * (0.2 + e.x / (reach[e.cell[0]] or 1))
                      * (4.0 if e.cell[0] == top else 1.0) for e in es])
        return es[int(rng.choice(len(es), p=w / w.sum()))]

    @property
    def frontier(self) -> int:
        """Total ground covered, counting each level's best once.

        Plain max-x cannot describe getting further once a level is finished:
        1-2 starts back at zero, so the number would sit at 1-1's ending for
        the rest of the run.
        """
        return sum(self._reach().values())

    def best(self) -> Entry | None:
        """The entry that got furthest, which is the one phase two clones."""
        return max(self.entries.values(),
                   key=lambda e: (e.cell[0], e.x), default=None)


def mask_of(pressed: frozenset[str]) -> int:
    from nes_player.emulator.controller import BUTTONS

    return sum(1 << i for i, b in enumerate(BUTTONS) if b in pressed)


def _roll(rng: np.random.Generator) -> tuple[frozenset[str], int]:
    """One button set, and how many frames to hold it."""
    sets = [a for a, _ in ACTIONS]
    w = np.array([p for _, p in ACTIONS])
    a = sets[int(rng.choice(len(sets), p=w / w.sum()))]
    return a, int(rng.integers(HOLD_MIN, HOLD_MAX + 1))


def _begin(env):
    """Play, actually started — not the title screen and not paused.

    START has to be pulsed. Held down it starts the game and then, because the
    same button is pause during play, immediately stops it again: the level
    loads, lives and time appear, and the clock sits at 400 forever while every
    button does nothing. The clock is therefore also the test — the game is
    running exactly when the timer is going down.
    """
    obs = env.reset(seed=0)
    prev = None
    for i in range(1500):
        pulse = i < 400 and i % 60 in (0, 1)
        obs = env.step_buttons([frozenset({"START"}) if pulse else frozenset()])
        now = (obs.debug or {}).get("time")
        if prev is not None and now is not None and now < prev:
            return obs
        prev = now
    raise RuntimeError("never got past the title screen")


def search(game: str = "SuperMarioBros-Nes-v0", iterations: int = 400,
           seed: int = 0, state: str | None = "default",
           out_dir: str | Path | None = None,
           explore_frames: int = EXPLORE_FRAMES,
           log_every: int = 25) -> dict:
    """Phase one: push the frontier as far into the game as it will go.

    Each iteration returns to a remembered place, explores from it for a while,
    and files away everywhere new it got to. Dying ends an iteration — nothing
    after a death is worth remembering, and the archive is a record of places
    reached alive.
    """
    from nes_player.emulator.stable_retro import StableRetroAdapter

    rng = np.random.default_rng(seed)
    env = StableRetroAdapter(game, include_debug=True, state=state)
    obs = _begin(env)

    archive = Archive()
    d = obs.debug or {}
    archive.consider(Entry(cell_of(d), env.save_state(), [], xpos_of(d),
                           int(d.get("score", 0)), int(d.get("lives", 0))))
    history: list[dict] = []

    for it in range(iterations):
        start = archive.pick(rng)
        start.chosen += 1
        env.load_state(start.state)
        actions = list(start.actions)
        # One idle frame to read the restored position back out of the core.
        obs = env.step_buttons([frozenset()])
        actions.append(0)
        lives0 = int((obs.debug or {}).get("lives", 0))

        held, left = _roll(rng)
        for _ in range(explore_frames):
            if left == 0:
                held, left = _roll(rng)
            left -= 1
            obs = env.step_buttons([held])
            actions.append(mask_of(held))
            d = obs.debug or {}
            if game_over(d) or int(d.get("lives", lives0)) < lives0:
                break
            archive.consider(Entry(cell_of(d), env.save_state(), list(actions),
                                   xpos_of(d), int(d.get("score", 0)),
                                   int(d.get("lives", 0))))

        if it % log_every == 0 or it == iterations - 1:
            b = archive.best()
            rec = {"iteration": it, "cells": len(archive.entries),
                   "frontier": archive.frontier,
                   "level": b.cell[0] if b else 0}
            history.append(rec)
            print(json.dumps(rec), flush=True)
    env.close()

    best = archive.best()
    result = {"cells": len(archive.entries), "frontier": archive.frontier,
              "iterations": iterations, "history": history,
              "best_x": best.x if best else 0,
              "best_len": len(best.actions) if best else 0}
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if best is not None:
            np.save(out / "best_actions.npy",
                    np.asarray(best.actions, np.uint8))
        (out / "search.json").write_text(json.dumps(result, indent=2))
    return result


def keep_segment(kind: str, written: dict[str, int], alive_ratio: float) -> bool:
    """Whether to write this segment out.

    A death is always kept. It is the scarce label — the object memory can only
    call something dangerous if it saw a contact followed by a death — and the
    explorer produces far more survivals than deaths once the archive is deep,
    so left alone the set would end up almost entirely one kind.
    """
    if kind == "died":
        return True
    return written["alive"] < alive_ratio * (1 + sum(written.values()))


def collect(game: str = "SuperMarioBros-Nes-v0", iterations: int = 600,
            record: str | Path = "datasets/goex_smb", seed: int = 0,
            state: str | None = "default", explore_frames: int = 600,
            max_episodes: int = 240, keep_alive_ratio: float = 0.5,
            log_every: int = 50) -> dict:
    """The search, but keeping the experience instead of only the winner.

    A memorised button sequence is worth nothing on a game it was not found on.
    What survives the move to another game is what can be *learned* from the
    experience: which objects hurt, which pay, and what a button does to the
    hero. Both need frames labelled with what happened next, and both are
    already implemented here — `perception/memory.py` and `world_model/ego.py`
    — starved of anything to chew on.

    They were starved because every dataset this project owns is the first
    fifth of 1-1: eighty-seven episodes that nearly all die before x=720 of
    3266. That is why the clone never learned to jump at enemies — measured,
    our own play jumps at them 1.2 points above its own baseline, which is
    nothing. The search goes where the datasets do not.

    So each explore segment is written as an ordinary episode, outcome in the
    metadata. Deaths are kept preferentially: a segment that ends in a death is
    the labelled negative the object memory needs and the rarer of the two.
    Ordinary episodes rather than a new format, because everything downstream
    — sprite tables, state vectors, cloning — already reads them.
    """
    from nes_player.data.writer import EpisodeWriter
    from nes_player.emulator.stable_retro import StableRetroAdapter

    rng = np.random.default_rng(seed)
    out_root = Path(record)
    env = StableRetroAdapter(game, include_debug=True, state=state)
    obs = _begin(env)

    archive = Archive()
    d = obs.debug or {}
    archive.consider(Entry(cell_of(d), env.save_state(), [], xpos_of(d),
                           int(d.get("score", 0)), int(d.get("lives", 0))))
    written = {"died": 0, "alive": 0}

    for it in range(iterations):
        if sum(written.values()) >= max_episodes:
            break
        start = archive.pick(rng)
        start.chosen += 1
        env.load_state(start.state)
        obs = env.step_buttons([frozenset()])
        d0 = obs.debug or {}
        lives0 = int(d0.get("lives", 0))
        score0 = int(d0.get("score", 0))

        frames: list = []
        acts: list[frozenset[str]] = []
        boxes: list[np.ndarray] = []
        died = False
        held, left = _roll(rng)
        for _ in range(explore_frames):
            if left == 0:
                held, left = _roll(rng)
            left -= 1
            obs = env.step_buttons([held])
            frames.append(obs)
            acts.append(held)
            boxes.append(_pad(sprite_boxes(env._env.get_ram())))
            d = obs.debug or {}
            if game_over(d) or int(d.get("lives", lives0)) < lives0:
                died = True
                break
            archive.consider(Entry(cell_of(d), env.save_state(),
                                   [], xpos_of(d), int(d.get("score", 0)),
                                   int(d.get("lives", 0))))

        kind = "died" if died else "alive"
        if keep_segment(kind, written, keep_alive_ratio) and len(frames) > 30:
            ep = out_root / f"{game}_seg{sum(written.values()):04d}"
            w = EpisodeWriter(out_dir=ep, metadata={
                "game": game, "source": "go-explore",
                "sample_rate": env.sample_rate,
                "outcome": kind,
                "level": level_of(d0),
                "x_start": xpos_of(d0), "x_end": xpos_of(obs.debug or {}),
                "score_gain": int((obs.debug or {}).get("score", 0)) - score0,
            })
            for o, a in zip(frames, acts, strict=True):
                w.append(o, (a,))
            w.close()
            # The sprite table, written now rather than recovered later.
            # `episode_sprites` rebuilds it by replaying an episode from a
            # fixed start and checking the frames match — and these segments
            # begin from a restored save state, so that replay can never
            # reproduce them. Saved under the name the cache looks for, they
            # are ordinary episodes to everything downstream; without it they
            # would be the only episodes in the project with no object
            # positions, which is most of what they were collected for.
            np.save(ep / f"sprites.v{SPRITES_VERSION}.npy",
                    np.stack(boxes).astype(np.uint8))
            written[kind] += 1

        if it % log_every == 0:
            print(json.dumps({"iteration": it, **written,
                              "cells": len(archive.entries),
                              "frontier": archive.frontier}), flush=True)
    env.close()
    result = {**written, "cells": len(archive.entries),
              "frontier": archive.frontier}
    print(json.dumps({"done": result}), flush=True)
    return result
