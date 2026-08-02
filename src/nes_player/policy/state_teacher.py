"""A teacher that plays from object positions instead of pixels.

Why this exists, in order of what was measured:

- Behavioural cloning on the instinct policy's play was taken to a near-perfect
  clone (validation 0.981) and played no better than a barely-trained one
  (0.567). Cloning is finished as a source of improvement: its ceiling is the
  policy that produced the data, and that policy does not finish a level.
- Giving the instinct policy exact object positions instead of inferred ones
  raised its score by half (t=+2.42) but also its deaths (t=+3.81) and lowered
  its progress. Perception was a real bottleneck for fighting and not the
  bottleneck for getting anywhere.

So the teacher's advantage cannot be that it sees more; the agent above saw
everything and still stood still. The advantage has to be that it is trained on
*results* rather than on imitation — and that is where a compact state pays:
self-imitation needs many rollouts and a small network over thirty-odd numbers
trains in seconds where the pixel network takes an hour.

The teacher is allowed to read the machine. The student it will eventually
teach is not (spec §3) — this file is never imported by the pixel policy.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nes_player import provenance
from nes_player.data.reader import Episode
from nes_player.perception.sprites import SpriteTracker, episode_sprites

# How many objects besides the hero the state describes, nearest first. Six
# covers a Double Dragon crowd; beyond that they are off screen or irrelevant.
N_OTHERS = 6
PER_OBJECT = 5                       # dx, dy, vx, vy, present
STATE_DIM = PER_OBJECT * (1 + N_OTHERS) + 1     # + camera scroll
# Same geometric spacing as the pixel model's frame stack, for the same reason:
# a window that reaches back without growing the input.
STATE_OFFSETS = (8, 4, 2, 1)
INPUT_DIM = STATE_DIM * len(STATE_OFFSETS)
CTRL_MIN = 0.55                      # confidence needed to call a slot "me"
SCREEN_W, SCREEN_H = 240.0, 224.0
VEL_SCALE = 8.0


def features(slots: list, scroll_dx: float) -> np.ndarray:
    """One frame of state: the hero, then the nearest objects, all relative.

    Positions are relative to the hero on purpose. Absolute screen coordinates
    would make the teacher memorise places, and the places change every screen;
    "an enemy is twenty pixels to my right" is the same fact everywhere.
    """
    out = np.zeros(STATE_DIM, np.float32)
    hero = max(slots, key=lambda s: s.ctrl_prob, default=None)
    if hero is None or hero.ctrl_prob < CTRL_MIN:
        out[-1] = scroll_dx / VEL_SCALE
        return out
    out[:PER_OBJECT] = (hero.cx / SCREEN_W, hero.cy / SCREEN_H,
                        hero.vx / VEL_SCALE, hero.vy / VEL_SCALE, 1.0)
    others = sorted((s for s in slots if s is not hero),
                    key=lambda s: abs(s.cx - hero.cx) + abs(s.cy - hero.cy))
    for k, s in enumerate(others[:N_OTHERS]):
        o = PER_OBJECT * (1 + k)
        out[o:o + PER_OBJECT] = ((s.cx - hero.cx) / SCREEN_W,
                                 (s.cy - hero.cy) / SCREEN_H,
                                 s.vx / VEL_SCALE, s.vy / VEL_SCALE, 1.0)
    out[-1] = scroll_dx / VEL_SCALE
    return out


def stack(history: list[np.ndarray]) -> np.ndarray:
    """The last states at geometric offsets, oldest first."""
    n = len(history)
    return np.concatenate([history[max(0, n - o)] for o in STATE_OFFSETS])


class StateNet(nn.Module):
    """Thirty-odd numbers in, one action out. Deliberately small: the point of
    a compact state is that the network can be cheap enough to retrain inside a
    self-imitation loop, not that it can be as expressive as the pixel model."""

    def __init__(self, n_actions: int, width: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class StatePolicy:
    """Plays from the sprite table. Same act() shape as BCPolicy, so the
    experiment scripts can drive either without knowing which is which."""

    checkpoint: str | Path

    def __post_init__(self) -> None:
        from nes_player.policy.bc import ActionVocab, device

        path = Path(self.checkpoint)
        self.meta = json.loads((path / "meta.json").read_text())
        self.vocab = ActionVocab(self.meta["vocab_masks"])
        self.dev = device()
        self.net = StateNet(len(self.vocab.masks)).to(self.dev)
        self.net.load_state_dict(torch.load(path / "model.pt", map_location=self.dev))
        self.net.eval()
        self.reset()

    def reset(self) -> None:
        self.tracker = SpriteTracker()
        self._history: list[np.ndarray] = [np.zeros(STATE_DIM, np.float32)]
        self._pressed: frozenset[str] = frozenset()

    def act(self, frame_rgb: np.ndarray, ram: np.ndarray,
            temperature: float = 1.0) -> tuple[frozenset[str], np.ndarray]:
        slots = self.tracker.update(frame_rgb, self._pressed, ram)
        self._history.append(features(slots, self.tracker.scroll_dx))
        x = torch.from_numpy(stack(self._history)).unsqueeze(0).to(self.dev)
        with torch.no_grad():
            logits = self.net(x)[0]
        from nes_player.policy.bc import mask_to_pressed

        probs = torch.softmax(logits / max(temperature, 1e-6), dim=0).cpu().numpy()
        idx = (int(probs.argmax()) if temperature <= 0
               else int(np.random.choice(len(probs), p=probs / probs.sum())))
        self._pressed = mask_to_pressed(self.vocab.masks[idx])
        self.last_index = idx   # what to train on; decoding back would round-trip
        ranked = sorted(zip(self.vocab.names, probs.tolist(), strict=True),
                        key=lambda t: -t[1])
        return self._pressed, ranked


def episode_states(ep: Episode) -> np.ndarray:
    """Per-frame state vectors for a recorded episode, (N, STATE_DIM); cached.

    Built from the cached sprite table and the recorded frames, so the episode
    is not emulated again — the table was already checked against the frames
    when it was made.
    """
    cache = ep.path / "states.v1.npy"
    if cache.exists():
        m = np.load(cache)
        if m.shape[1] == STATE_DIM:
            return m

    from nes_player.emulator.controller import BUTTONS

    boxes = episode_sprites(ep)
    frames, actions = ep.frames, ep.actions
    tracker = SpriteTracker()
    out = np.zeros((len(boxes), STATE_DIM), np.float32)
    pressed: frozenset[str] = frozenset()
    for i in range(len(boxes)):
        slots = tracker.update(np.asarray(frames[i]), pressed, boxes=boxes[i])
        out[i] = features(slots, tracker.scroll_dx)
        mask = int(actions[i, 0])
        pressed = frozenset(b for k, b in enumerate(BUTTONS) if mask & (1 << k))
    np.save(cache, out)
    return out


def stacked_dataset(states: np.ndarray) -> np.ndarray:
    """(N, STATE_DIM) states -> (N, INPUT_DIM) stacks, clamped at the start."""
    n = len(states)
    idx = np.stack([np.clip(np.arange(n) - (o - 1), 0, None) for o in STATE_OFFSETS])
    return states[idx].transpose(1, 0, 2).reshape(n, INPUT_DIM)


def pretrain(episode_dir: str | Path, out_dir: str | Path, epochs: int = 40,
             batch_size: int = 512, lr: float = 1e-3, val_frac: float = 0.1,
             seed: int = 0) -> dict:
    """Clone the recorded play from state, as a starting point for improvement.

    This is not expected to be *good* — cloning tops out at the policy that
    made the data, which was measured today and is not enough to finish a
    level. It exists so the self-imitation loop starts from something that
    fights and walks rather than from noise, which in a beat-em-up produces
    rollouts that are all equally worthless and nothing to select between.
    """
    from nes_player.policy.bc import ActionVocab, device

    torch.manual_seed(seed)
    np.random.seed(seed)
    ep_dirs: list[Path] = []
    for part in str(episode_dir).split(","):
        root = Path(part)
        if (root / "metadata.json").exists():
            ep_dirs.append(root)
        else:
            ep_dirs.extend(sorted(d for d in root.iterdir()
                                  if (d / "metadata.json").exists()))
    eps = [Episode(d) for d in ep_dirs]
    print(f"episodes: {len(eps)}", flush=True)

    vocab = ActionVocab.from_actions(
        np.concatenate([e.actions[:, 0] for e in eps]))
    xs, ys = [], []
    for k, e in enumerate(eps, 1):
        xs.append(stacked_dataset(episode_states(e)))
        ys.append(vocab.encode(e.actions[:, 0]))
        print(f"  states {k}/{len(eps)}", flush=True)

    n_val_eps = max(1, int(len(eps) * val_frac))
    xtr = torch.from_numpy(np.concatenate(xs[:-n_val_eps]))
    ytr = torch.from_numpy(np.concatenate(ys[:-n_val_eps]))
    xva = torch.from_numpy(np.concatenate(xs[-n_val_eps:]))
    yva = torch.from_numpy(np.concatenate(ys[-n_val_eps:]))

    dev = device()
    net = StateNet(len(vocab)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xva_d, yva_d = xva.to(dev), yva.to(dev)
    history, best_acc, best_state = [], -1.0, None
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(len(xtr))
        total, correct, loss_sum = 0, 0, 0.0
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            x, y = xtr[b].to(dev), ytr[b].to(dev)
            logits = net(x)
            loss = nn.functional.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(y)
            correct += int((logits.argmax(1) == y).sum())
            total += len(y)
        net.eval()
        with torch.no_grad():
            val_acc = float((net(xva_d).argmax(1) == yva_d).float().mean())
        rec = {"epoch": epoch, "train_loss": loss_sum / total,
               "train_acc": correct / total, "val_acc": val_acc}
        history.append(rec)
        if val_acc > best_acc:
            best_acc, best_state = val_acc, {k: v.detach().cpu().clone()
                                             for k, v in net.state_dict().items()}
        print(rec, flush=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out / "model.pt")
    meta = {"vocab_masks": vocab.masks, "vocab_names": vocab.names,
            "state_dim": STATE_DIM, "input_dim": INPUT_DIM,
            "state_offsets": list(STATE_OFFSETS), "n_others": N_OTHERS,
            "episode": str(episode_dir), "history": history,
            "val_acc": best_acc, "source": "state-pretrain"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    provenance.write(out, config={"epochs": epochs, "lr": lr, "seed": seed,
                                  "val_frac": val_frac, "batch_size": batch_size,
                                  "kind": "state-pretrain"},
                     episodes=ep_dirs, game=eps[0].metadata.get("game"))
    print(f"val_acc={best_acc:.3f}")
    return meta


# ---------- improvement by result, not by imitation ----------

DEFAULT_GAME = "DoubleDragon-Nes-v0"
DEATH_COST = 300.0     # a life is worth this much progress
SCORE_WEIGHT = 1.0
LEVEL_BONUS = 1000.0   # finishing a level beats any amount of walking
IDLE_STEP, IDLE_MAX = 37, 1800   # start offset per seed, capped at 30 seconds


class GameProgress:
    """How far into the level, from the game itself where it says so.

    Camera scroll from pixels is the fallback, and on Double Dragon it was a
    bad one: the camera holds still while enemies are alive, so every rollout
    "progressed" about the same and there was nothing for selection to climb.
    Super Mario Bros. publishes its own x position and level number, which vary
    between policies by hundreds rather than by ten. Where the game tells us,
    listen to the game.
    """

    def __init__(self) -> None:
        self.total = 0.0
        self.levels = 0
        self._x: int | None = None
        self._level: int | None = None
        self.from_ram = False
        self._visual = None

    def update(self, debug: dict, frame_rgb) -> None:
        if "xscrollLo" not in debug:
            if self._visual is None:
                from nes_player.policy.improve import VisualProgress

                self._visual = VisualProgress()
            self._visual.update(frame_rgb)
            self.total = self._visual.total
            return
        self.from_ram = True
        x = int(debug.get("xscrollHi", 0)) * 256 + int(debug.get("xscrollLo", 0))
        if self._x is not None:
            dx = x - self._x
            if dx < -128:      # the low byte wrapped, not a jump backwards
                dx += 256
            if 0 < dx < 128:   # a level restart moves it back; that is not progress
                self.total += dx
        self._x = x
        level = int(debug.get("levelHi", 0)) * 4 + int(debug.get("levelLo", 0))
        if self._level is not None and level > self._level:
            self.levels += 1
        self._level = level


def _rollout(arg) -> dict:
    """One episode played by the checkpoint; returns its states, actions, reward.

    Run in a separate process: rollouts are the whole cost of this loop and
    they are independent, exactly like the attention masks.
    """
    checkpoint, seed, frames, temperature, game, state = arg
    from nes_player.emulator.stable_retro import StableRetroAdapter

    env = StableRetroAdapter(game, include_debug=True, state=state)
    policy = StatePolicy(checkpoint)
    progress = GameProgress()
    obs = env.reset(seed=seed)
    # Each rollout starts at a different moment, or every one of them measures
    # the same memorised opening. Bounded on purpose: the plain `seed * 37`
    # used elsewhere is fine for seeds 0-9 and absurd for the held-out set —
    # seed 901 idled for 33000 frames, long enough for the game to give up and
    # return to its demo, and every evaluation came back exactly zero.
    for _ in range(IDLE_STEP * seed % IDLE_MAX):
        obs = env.step_buttons([frozenset()])

    states: list[np.ndarray] = []
    labels: list[int] = []
    score0, lives0, lives_now, deaths = None, None, None, 0
    for i in range(frames):
        d = obs.debug or {}
        if score0 is None and i > 200:
            score0 = d.get("score", 0)
        lv = d.get("lives")
        if lv is not None and lv <= 90:
            if lives0 is None:
                lives0 = lv
            if lives_now is not None and lv < lives_now:
                deaths += 1
            lives_now = lv
        pressed, _ = policy.act(obs.frame_rgb, env._env.get_ram(), temperature)
        states.append(stack(policy._history))
        labels.append(policy.last_index)
        obs = env.step_buttons([pressed - {"START", "SELECT"}])
        progress.update(obs.debug or {}, obs.frame_rgb)
    env.close()

    score = max(0, (obs.debug or {}).get("score", 0) - (score0 or 0))
    reward = (progress.total + SCORE_WEIGHT * score
              + LEVEL_BONUS * progress.levels - DEATH_COST * deaths)
    return {"seed": seed, "reward": float(reward), "progress": round(progress.total, 1),
            "levels": progress.levels, "score": int(score), "deaths": deaths,
            "states": np.stack(states).astype(np.float32),
            "labels": np.asarray(labels, np.int64)}


def _evaluate(checkpoint: Path, seeds: list[int], frames: int, workers: int,
              game: str, state: str | None) -> dict:
    """The same seeds every round, at low temperature: the actual progress curve."""
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as pool:
        rs = list(pool.map(_rollout, [(str(checkpoint), s, frames, 0.35, game, state)
                                      for s in seeds]))
    return {"eval_reward": round(float(np.mean([x["reward"] for x in rs])), 1),
            "eval_progress": round(float(np.mean([x["progress"] for x in rs])), 1),
            "eval_levels": round(float(np.mean([x["levels"] for x in rs])), 2),
            "eval_score": round(float(np.mean([x["score"] for x in rs])), 1),
            "eval_deaths": round(float(np.mean([x["deaths"] for x in rs])), 2)}


def self_improve(checkpoint: str | Path, out_dir: str | Path, rounds: int = 6,
                 rollouts: int = 12, keep: int = 4, frames: int = 2400,
                 temperature: float = 1.0, epochs: int = 8, lr: float = 3e-4,
                 demo_dir: str | Path | None = None, demo_frac: float = 0.5,
                 workers: int | None = None,
                 eval_seeds: tuple[int, ...] = (901, 902, 903, 904, 905, 906),
                 game: str = DEFAULT_GAME, state: str | None = "default",
                 ) -> list[dict]:
    """Play, keep the rollouts that did best, retrain on those. Repeat.

    Two guards against the collapse this project already hit once, where
    retraining on the top two of eight round after round produced a policy so
    deterministic that six evaluation runs came back identical:

    - a wider slice is kept (a third, not a quarter);
    - the original demonstrations are mixed back in every round, so the policy
      is pulled towards *better than the demos* rather than away from any
      behaviour that a lucky rollout happened not to use.
    """
    import os
    import shutil
    from concurrent.futures import ProcessPoolExecutor

    from nes_player.policy.bc import device

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(checkpoint)
    for f in ("model.pt", "meta.json"):
        shutil.copy(src / f, out / f)
    meta = json.loads((out / "meta.json").read_text())

    demo_x = demo_y = None
    if demo_dir:
        from nes_player.policy.bc import ActionVocab

        eps = [Episode(d) for d in sorted(Path(demo_dir).iterdir())
               if (d / "metadata.json").exists()]
        vocab = ActionVocab(meta["vocab_masks"])
        demo_x = np.concatenate([stacked_dataset(episode_states(e)) for e in eps])
        demo_y = np.concatenate([vocab.encode(e.actions[:, 0]) for e in eps])

    dev = device()
    net = StateNet(len(meta["vocab_masks"])).to(dev)
    net.load_state_dict(torch.load(out / "model.pt", map_location=dev))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n_workers = workers or max(1, min(rollouts, (os.cpu_count() or 4) - 2))
    eval_seeds = list(eval_seeds)
    log: list[dict] = [{"round": -1, "rollout_mean": None, "kept": [],
                        **_evaluate(out, eval_seeds, frames, n_workers,
                                    game, state)}]
    print(json.dumps(log[0]), flush=True)   # where it started, before any round

    for r in range(rounds):
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_rollout, [
                (str(out), r * rollouts + k, frames, temperature, game, state)
                for k in range(rollouts)]))
        results.sort(key=lambda x: -x["reward"])
        best = results[:keep]
        # Rollout seeds differ every round on purpose — training on one fixed
        # opening teaches that opening. But that makes the round's own mean
        # useless as a progress curve, because it compares different starting
        # points. So improvement is read off a fixed held-out set instead, the
        # same seeds every round, never trained on.
        rec = {"round": r,
               "rollout_mean": round(float(np.mean([x["reward"] for x in results])), 1),
               "kept": [round(x["reward"], 1) for x in best],
               **_evaluate(out, eval_seeds, frames, n_workers, game, state)}
        log.append(rec)
        print(json.dumps(rec), flush=True)

        x = np.concatenate([b["states"] for b in best])
        y = np.concatenate([b["labels"] for b in best])
        if demo_x is not None and demo_frac > 0:
            n = int(len(x) * demo_frac)
            pick = np.random.default_rng(r).choice(len(demo_x), size=n, replace=False)
            x = np.concatenate([x, demo_x[pick]])
            y = np.concatenate([y, demo_y[pick]])
        xt = torch.from_numpy(x).to(dev)
        yt = torch.from_numpy(y).to(dev)
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

    meta["source"] = "state-self-improve"
    meta["improve_log"] = log
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    provenance.write(out, config={"kind": "state-self-improve", "rounds": rounds,
                                  "rollouts": rollouts, "keep": keep,
                                  "frames": frames, "temperature": temperature,
                                  "epochs": epochs, "lr": lr,
                                  "demo_dir": str(demo_dir), "from": str(checkpoint)},
                     game=game)
    return log
