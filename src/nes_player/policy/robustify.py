"""Learn the found route backwards, from the end towards the start.

Cloning the search directly was tried and measured worse: Go-Explore presses
buttons at random, so a segment that survived is a lucky one rather than a
skilled one, and imitating it teaches randomness that happened to work. Data
has to be good, not merely from the right place.

What the paper actually prescribes for its second phase is a curriculum. Put
the agent a few frames from the end of a found trajectory, where almost any
behaviour succeeds, and let it play to the finish itself. When it succeeds
often enough, move its starting point further back. The agent is always facing
a problem just slightly harder than the one it has solved, and — the part that
matters here — everything it learns from is **its own** successful play. The
trajectory supplies the starting positions, never the actions.

This also sidesteps the thing that killed self-imitation six times. There the
policy cloned its own average behaviour and drifted towards it. Here a rollout
only enters the training set if it reached the goal, and the goal moves further
away only when the policy can already handle where it is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nes_player.policy.go_explore import _begin, level_of, xpos_of

# How far back from the end to begin. Close enough that the untrained policy
# succeeds sometimes by accident, or the curriculum never gets its first rung.
START_BACK = 150
STEP = 120            # frames the start moves back on a pass
ATTEMPTS = 8          # rollouts per rung
PASS_RATE = 0.5       # share that must reach the goal before the start moves
# Budget for a rung, as a multiple of the frames the trajectory itself needed.
# A policy that is merely slower than the search should still be allowed to
# finish; one that is stuck should not hold the loop open.
BUDGET = 2.0


@dataclass
class Rung:
    """One difficulty setting: where the agent starts, and how it did there."""

    index: int                       # frame of the trajectory it starts from
    attempts: int = 0
    wins: int = 0

    @property
    def rate(self) -> float:
        return self.wins / self.attempts if self.attempts else 0.0

    @property
    def passed(self) -> bool:
        return self.attempts >= ATTEMPTS and self.rate >= PASS_RATE


@dataclass
class Curriculum:
    """Bookkeeping for the moving start.

    Kept apart from the emulator so the rule — when to move back, when to give
    up on a rung — can be tested without a ROM. The rules that decide what gets
    trained on are where this project's mistakes have lived.
    """

    total: int                       # length of the trajectory in frames
    index: int                       # current starting frame
    step: int = STEP
    history: list[dict] = field(default_factory=list)
    stalled: int = 0

    def record(self, rung: Rung) -> bool:
        """File a finished rung; returns whether the start moved back."""
        self.history.append({"index": rung.index, "attempts": rung.attempts,
                             "wins": rung.wins, "rate": round(rung.rate, 2)})
        if rung.passed:
            self.index = max(0, self.index - self.step)
            self.stalled = 0
            return True
        self.stalled += 1
        return False

    @property
    def done(self) -> bool:
        return self.index <= 0

    @property
    def progress(self) -> float:
        """How much of the trajectory the agent can now finish by itself."""
        return 1.0 - self.index / self.total if self.total else 1.0


def snapshots(game: str, actions: np.ndarray, state: str | None = "default",
              every: int = 30) -> tuple[dict[int, bytes], int]:
    """Emulator states along the trajectory, every `every` frames.

    Replaying once and snapshotting is cheaper and less error-prone than
    storing states during the search: the run is deterministic, so this
    reproduces it exactly, and it keeps the search's memory small.
    """
    from nes_player.emulator.controller import BUTTONS
    from nes_player.emulator.stable_retro import StableRetroAdapter

    env = StableRetroAdapter(game, include_debug=True, state=state)
    obs = _begin(env)
    out: dict[int, bytes] = {0: env.save_state()}
    goal = progress_of(obs.debug or {})
    for i, m in enumerate(actions, 1):
        pressed = frozenset(b for k, b in enumerate(BUTTONS) if int(m) >> k & 1)
        obs = env.step_buttons([pressed])
        goal = max(goal, progress_of(obs.debug or {}))
        if i % every == 0:
            out[i] = env.save_state()
    env.close()
    return out, goal


LEVEL_SPAN = 4000     # wider than any single level, so levels never overlap


def progress_of(debug: dict) -> int:
    """One number for "how far into the game", across level boundaries.

    Comparing (level, x) directly does not work: x restarts at zero in a new
    level, so a run that has just finished one scores below where it was a
    moment earlier. Folding the level in at a stride wider than any level makes
    the number monotone in the thing we care about.
    """
    return level_of(debug) * LEVEL_SPAN + xpos_of(debug)


def reached_goal(best: int, goal: int, margin: int = 48) -> bool:
    """Did this run get as far as the trajectory did?

    `best` is the furthest point reached during the attempt, not the current
    one. The level counter flips a moment before the camera resets, so the
    furthest point is often a frame that has already passed — and a test on the
    current position calls a finished level a failure one frame later.
    """
    return best >= goal - margin


def _attempt(env, policy, snap: bytes, budget: int, goal: int,
             temperature: float, jump_hold: int):
    """One try from a restored point. Returns (won, states, labels)."""
    from nes_player.cli.play import JumpShaper
    from nes_player.perception.feedback import game_over
    from nes_player.policy.state_teacher import stack

    env.load_state(snap)
    policy.reset()
    jump = JumpShaper(jump_hold) if jump_hold else None
    obs = env.step_buttons([frozenset()])
    lives0 = int((obs.debug or {}).get("lives", 0))
    states, labels = [], []
    best = progress_of(obs.debug or {})
    for _ in range(budget):
        d = obs.debug or {}
        best = max(best, progress_of(d))
        if game_over(d) or int(d.get("lives", lives0)) < lives0:
            return False, states, labels        # died: nothing to learn from
        if reached_goal(best, goal):
            return True, states, labels
        pressed, _ = policy.act(obs.frame_rgb, env._env.get_ram(), temperature)
        states.append(stack(policy._history))
        labels.append(policy.last_index)
        if jump is not None:
            pressed = jump.apply(pressed)
        obs = env.step_buttons([pressed - {"START", "SELECT"}])
    return reached_goal(best, goal), states, labels


def robustify(checkpoint: str | Path, trajectory: str | Path,
              out_dir: str | Path, game: str = "SuperMarioBros-Nes-v0",
              state: str | None = "default", passes: int = 60,
              temperature: float = 1.0, jump_hold: int = 32,
              epochs: int = 3, lr: float = 3e-4, give_up: int = 4,
              demo_dir: str | Path | None = None, demo_frac: float = 0.5,
              ) -> dict:
    """Walk the starting point back along a found trajectory.

    Training data is only ever the agent's own runs that reached the goal.
    Losing runs are dropped rather than used as negatives: cross-entropy has no
    way to say "not this", and the last six attempts at self-imitation all
    failed by pulling the policy towards its own average.
    """
    import shutil

    import torch
    from torch import nn

    from nes_player.policy.bc import ActionVocab, device
    from nes_player.policy.state_teacher import (
        Episode,
        StateNet,
        StatePolicy,
        episode_dirs,
        episode_states,
        stacked_dataset,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(checkpoint)
    for f in ("model.pt", "meta.json"):
        shutil.copy(src / f, out / f)
    meta = json.loads((out / "meta.json").read_text())

    demo_x = demo_y = None
    if demo_dir:
        vocab = ActionVocab(meta["vocab_masks"])
        eps = [Episode(d) for d in episode_dirs(demo_dir)]
        demo_x = np.concatenate([stacked_dataset(episode_states(e)) for e in eps])
        demo_y = np.concatenate([vocab.encode(e.actions[:, 0]) for e in eps])

    actions = np.load(trajectory)
    snaps, goal = snapshots(game, actions, state)
    keys = sorted(snaps)
    print(json.dumps({"trajectory": len(actions), "snapshots": len(keys),
                      "goal": goal}), flush=True)

    from nes_player.emulator.stable_retro import StableRetroAdapter

    env = StableRetroAdapter(game, include_debug=True, state=state)
    # The gym wrapper refuses to step before a reset, and restoring a state is
    # not one as far as it is concerned.
    env.reset(seed=0)
    dev = device()
    net = StateNet(len(meta["vocab_masks"]),
                   width=meta.get("width", 256)).to(dev)
    net.load_state_dict(torch.load(out / "model.pt", map_location=dev))
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    cur = Curriculum(total=len(actions), index=max(0, len(actions) - START_BACK))
    for p in range(passes):
        if cur.done or cur.stalled >= give_up:
            break
        # The nearest snapshot at or before the wanted index; states exist only
        # every `every` frames.
        idx = max(k for k in keys if k <= cur.index)
        budget = int((len(actions) - idx) * BUDGET) + 120
        policy = StatePolicy(out)
        rung = Rung(index=idx)
        xs, ys = [], []
        for _ in range(ATTEMPTS):
            won, st, lb = _attempt(env, policy, snaps[idx], budget, goal,
                                   temperature, jump_hold)
            rung.attempts += 1
            rung.wins += int(won)
            if won and st:
                xs.append(np.stack(st))
                ys.append(np.asarray(lb, np.int64))
        moved = cur.record(rung)
        print(json.dumps({"pass": p, "start": idx,
                          "of": len(actions), "wins": rung.wins,
                          "attempts": rung.attempts, "moved": moved,
                          "learned": round(cur.progress, 3)}), flush=True)

        if not xs:
            continue
        x = np.concatenate(xs).astype(np.float32)
        y = np.concatenate(ys)
        if demo_x is not None and demo_frac > 0:
            n = int(len(x) * demo_frac)
            pick = np.random.default_rng(p).choice(len(demo_x), size=n,
                                                   replace=False)
            x = np.concatenate([x, demo_x[pick]])
            y = np.concatenate([y, demo_y[pick]])
        xt, yt = torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)
        net.train()
        for _ in range(epochs):
            perm = torch.randperm(len(xt), device=dev)
            for i in range(0, len(perm), 512):
                b = perm[i:i + 512]
                loss = nn.functional.cross_entropy(net(xt[b]), yt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()},
                   out / "model.pt")
    env.close()

    result = {"learned": round(cur.progress, 3), "start": cur.index,
              "total": cur.total, "stalled": cur.stalled,
              "history": cur.history}
    meta["source"] = "backward-curriculum"
    meta["curriculum"] = result
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"done": {k: result[k] for k in
                               ("learned", "start", "total", "stalled")}}),
          flush=True)
    return result
