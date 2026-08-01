"""Self-imitation: play with sampling, keep the best rollouts, retrain on them.

A cheap stand-in for reinforcement learning. The reward is used by the training
loop only and never reaches the policy's observations:

- from RAM: level progress plus score change minus deaths;
- from PIXELS (`--visual`): accumulated forward camera scroll. This works on
  games for which no memory map exists and opens no telemetry at all. Measured
  against true progress it correlates at 0.87.

One caution, seen on Battletoads & Double Dragon: retraining on the top two of
eight rollouts round after round collapsed the policy into a deterministic one,
and six evaluation runs returned identical results. If that happens, widen the
top slice or mix the original demonstrations back in.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nes_player.emulator.controller import BUTTONS
from nes_player.emulator.stable_retro import StableRetroAdapter
from nes_player.perception.motion import HUD_H
from nes_player.policy.bc import (
    FRAME_STACK,
    INPUT_HW,
    BCPolicy,
    EpisodeBCDataset,
)

AUTO_START_AT = 60
MAX_SCROLL = 16.0   # anything larger is a scene change, not camera movement


class VisualProgress:
    """Accumulated forward camera scroll — progress measured without RAM."""

    def __init__(self) -> None:
        self._prev = None
        self.total = 0.0

    def update(self, frame_rgb: np.ndarray) -> float:
        import cv2

        play = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)[HUD_H:].astype(np.float32)
        if self._prev is not None:
            (dx, _), _ = cv2.phaseCorrelate(self._prev, play)
            if abs(dx) < MAX_SCROLL:
                self.total += -dx   # camera moving right gives a negative dx
        self._prev = play
        return self.total


class _AVRolloutDataset(EpisodeBCDataset):
    """Frames and mel windows from a rollout, for audio-visual checkpoints."""

    def __init__(self, frames: np.ndarray, labels: np.ndarray, mels: np.ndarray):
        super().__init__(frames, labels)
        self.mels = mels

    def __getitem__(self, idx: int):
        x, y = super().__getitem__(idx)
        return x, torch.from_numpy(self.mels[idx + FRAME_STACK]).unsqueeze(0), y


@dataclass
class Rollout:
    small_frames: np.ndarray   # (N, H, W, 3) uint8, the policy input
    action_idx: np.ndarray  # (N,) int64
    reward: float
    progress: float
    score: int
    deaths: int
    mels: np.ndarray | None = None   # (N, MEL_N, MEL_FRAMES) for AV models


def _progress(d: dict) -> int:
    return d["levelHi"] * 100000 + d["levelLo"] * 10000 + d["xscrollHi"] * 256 + d["xscrollLo"]


def run_rollout(env: StableRetroAdapter, policy: BCPolicy, frames: int,
                temperature: float, repeat: int, visual: bool = False,
                start_pulses: int = 1, idle_start: int = 0) -> Rollout:
    import cv2

    obs = env.reset(seed=0)
    # Every rollout used to begin from exactly the same frame, so on a game
    # started from a savestate the only difference between them was sampling
    # noise. The policy then improves on one situation and nothing else: the
    # round statistics climb while play at a different point in the level does
    # not. `idle_start` waits a different number of frames before handing over.
    for _ in range(idle_start):
        obs = env.step_buttons([frozenset()])
    policy.reset()
    is_av = policy.modality == "av"
    smalls, actions, mels = [], [], []
    best_progress, score0, deaths, lives_prev = 0, None, 0, None
    vis = VisualProgress() if visual else None
    pressed, idx = frozenset(), 0
    pulses = {AUTO_START_AT + k * 180 + j for k in range(start_pulses) for j in range(4)}
    for i in range(frames):
        if is_av:
            policy.push_audio(obs.audio_pcm)   # the AV model has to hear here too
        if i % repeat == 0:
            pressed, _ranked = policy.act(obs.frame_rgb, temperature=temperature)
            mask = sum(1 << k for k, b in enumerate(BUTTONS) if b in pressed)
            idx = policy.vocab.masks.index(mask)
        if i in pulses:
            pressed = frozenset({"START"})
        else:
            smalls.append(cv2.resize(obs.frame_rgb, (INPUT_HW[1], INPUT_HW[0]),
                                     interpolation=cv2.INTER_AREA))
            actions.append(idx)
            if is_av:
                mels.append(policy._current_mel(cache_ok=True)[0, 0].cpu().numpy())
        obs = env.step_buttons([pressed])
        if vis is not None:
            best_progress = vis.update(obs.frame_rgb)
            continue
        d = obs.debug
        best_progress = max(best_progress, _progress(d))
        # Before the game starts the score addresses hold garbage, so the
        # baseline is taken after START
        if score0 is None and i > AUTO_START_AT + 40:
            score0 = d["score"]
        if lives_prev is not None and d["lives"] < lives_prev:
            deaths += 1
        lives_prev = d["lives"]
    if visual:
        reward = best_progress   # no deaths or score without RAM, by design
        score = 0
    else:
        score = max(0, (obs.debug["score"] or 0) - (score0 or 0))
        reward = best_progress + score * 2 - deaths * 300
    return Rollout(np.asarray(smalls), np.asarray(actions, dtype=np.int64),
                   float(reward), float(best_progress), score, deaths,
                   np.asarray(mels, dtype=np.float32) if is_av else None)


def self_imitation(
    checkpoint: str | Path,
    game: str = "SuperMarioBros-Nes-v0",
    rounds: int = 5,
    rollouts_per_round: int = 8,
    keep_frac: float = 0.33,
    frames: int = 1500,
    temperature: float = 1.1,
    repeat: int = 4,
    lr: float = 1e-4,
    ft_epochs: int = 2,
    visual: bool = False,
    integrations: str | Path | None = None,
    start_pulses: int = 1,
    state: str | None = None,
    idle_step: int = 37,
) -> list[dict]:
    policy = BCPolicy(checkpoint)
    # `state` matters more than it looks: some games (Double Dragon) cannot get
    # past their title screen from power-on by any button sequence, so without
    # it every rollout is a recording of the title, every reward is zero, and
    # the fine-tuning happily runs on nothing. That is how a whole dataset was
    # once collected before anyone looked at it.
    env = StableRetroAdapter(game, integration_dir=integrations,
                             include_debug=not visual, state=state)
    dev = policy.dev
    opt = torch.optim.AdamW(policy.model.parameters(), lr=lr)
    log = []
    try:
        for r in range(rounds):
            rollouts = [run_rollout(env, policy, frames, temperature, repeat,
                                    visual=visual, start_pulses=start_pulses,
                                    idle_start=k * idle_step)
                        for k in range(rollouts_per_round)]
            rollouts.sort(key=lambda x: -x.reward)
            keep = rollouts[: max(1, int(len(rollouts) * keep_frac))]
            rec = {
                "round": r,
                "reward_kind": "visual_scroll" if visual else "ram",
                "mean_progress": round(float(np.mean([x.progress for x in rollouts])), 1),
                "max_progress": round(max(x.progress for x in rollouts), 1),
                "mean_score": float(np.mean([x.score for x in rollouts])),
                "deaths": sum(x.deaths for x in rollouts),
                "kept_rewards": [x.reward for x in keep],
            }
            log.append(rec)
            print(json.dumps(rec), flush=True)   # rounds take minutes; show them as they land
            # Every rollout identical and going nowhere means the game never
            # started — a title screen that needs `state`, or an intro that
            # swallowed the START pulses. Fine-tuning on that trains the model
            # to reproduce a still picture, and nothing about it looks wrong
            # from the outside.
            spread = max(x.reward for x in rollouts) - min(x.reward for x in rollouts)
            if rec["max_progress"] <= 1.0 and spread <= 1e-6:
                raise RuntimeError(
                    f"round {r}: no rollout made any progress and all rewards are "
                    f"identical — the game is probably not running. Check --state "
                    f"and --start-pulses before trusting anything trained here.")

            xs = np.concatenate([x.small_frames for x in keep])
            ys = np.concatenate([x.action_idx for x in keep])
            if policy.modality == "av":
                ds = _AVRolloutDataset(xs, ys, np.concatenate([x.mels for x in keep]))
            else:
                ds = EpisodeBCDataset(xs, ys)
            dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
            policy.model.train()
            for _ in range(ft_epochs):
                for batch in dl:
                    *inputs, y = (t.to(dev) for t in batch)
                    loss = nn.functional.cross_entropy(policy.model(*inputs), y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            policy.model.eval()
    finally:
        env.close()
    run = Path(checkpoint)
    torch.save(policy.model.state_dict(), run / "model.pt")
    (run / "self_imitation_log.json").write_text(json.dumps(log, indent=2))
    return log


