#!/usr/bin/env python3
"""DT transfer gameplay: train on 3 games, play the 4th.

E3 showed DT predicts actions on an unseen game at 99% accuracy (offline).
This tests whether that transfers to actual gameplay.

For each game G:
1. Train DT on the other 3 games.
2. Play G with the transferred DT.
3. Compare with BC and in-domain DT on G.

    uv run python scripts/transfer_play.py --game Galaga-Nes-v0 --frames 5000
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/experiments")

from nes_player.policy.decision_transformer import DTConfig, count_params  # noqa: E402

GAMES = ["Battletoads-Nes-v0", "ChipnDale-Nes-v0", "Galaga-Nes-v0", "ViceProjectDoom-Nes-v0"]


def bitmask_to_index(mask: int) -> int:
    bits = [i for i in range(8) if mask & (1 << i)]
    if len(bits) == 1:
        return bits[0]
    return 8


class StateProj(torch.nn.Module):
    def __init__(self, d_model=384):
        super().__init__()
        self.proj = torch.nn.Linear(6, d_model)

    def forward(self, x):
        return self.proj(x)


def load_game_windows(game: str, window=32, stride=16):
    ep = Path(f"datasets/agent_play/{game}_agent_long")
    if not ep.exists():
        return None

    actions = np.load(ep / "actions.npy")
    rtg = np.load(ep / "return_to_go.npy")

    import zarr
    z = zarr.open(str(ep / "episode.zarr"), mode="r")
    frames = z["frames"][:]

    # Expand rtg to frame length if needed
    if len(rtg) < len(frames):
        factor = len(frames) // len(rtg)
        rtg = np.repeat(rtg, factor)[:len(frames)]

    T = len(frames)
    returns, states, actions_out = [], [], []
    for i in range(0, T - window, stride):
        chunk = frames[i : i + window]
        rtg_chunk = rtg[i : i + window]
        act_chunk = actions[i : i + window]
        if len(chunk) < window or len(rtg_chunk) < window or len(act_chunk) < window:
            continue
        feats = []
        for f in chunk:
            band = f[8:24, 195:240]
            sc = int(((band[..., 0] > 180) & (band[..., 1] > 180)
                      & (band[..., 2] > 180)).sum())
            bright = float(f.mean())
            red = float((f[..., 0] > 150).mean())
            green = float((f[..., 1] > 150).mean())
            blue = float((f[..., 2] > 150).mean())
            dark = float((f.mean(axis=-1) < 40).mean())
            feats.append([sc, bright, red, green, blue, dark])
        states.append(feats)
        returns.append(rtg_chunk)
        acts = [bitmask_to_index(a) for a in act_chunk]
        actions_out.append(acts)
    return (
        torch.tensor(np.array(returns), dtype=torch.float32),
        torch.tensor(np.array(states), dtype=torch.float32),
        torch.tensor(np.array(actions_out), dtype=torch.long),
    )


def train_dt_on_games(games: list[str], epochs=5) -> torch.nn.Module:
    """Train DT on combined windows from the given games."""
    all_returns, all_states, all_actions = [], [], []
    for g in games:
        data = load_game_windows(g)
        if data is None:
            print(f"  {g}: MISSING")
            continue
        all_returns.append(data[0])
        all_states.append(data[1])
        all_actions.append(data[2])
        print(f"  {g}: {data[0].shape[0]} windows")

    returns = torch.cat(all_returns)
    states = torch.cat(all_states)
    actions = torch.cat(all_actions)
    print(f"  total: {returns.shape[0]} windows")

    cfg = DTConfig(n_actions=9, d_model=384, n_layers=6, n_heads=6)
    model = cfg.build(state_encoder=StateProj())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    n = len(returns)
    bs = 32
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss, total_correct, total = 0, 0, 0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            r, s, a = returns[idx], states[idx], actions[idx]
            logits = model(r, s, a)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 9), a.reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            preds = logits.argmax(dim=-1)
            total_loss += float(loss.detach()) * len(idx)
            total_correct += int((preds == a).sum())
            total += len(idx) * a.shape[1]
        acc = total_correct / total
        print(f"  epoch {epoch}: loss={total_loss/n:.4f} acc={acc:.4f}")

    return model


def score_of(frame_rgb: np.ndarray, game: str) -> int:
    if game.startswith("Galaga") or game.startswith("PacMan"):
        band = frame_rgb[62:78, 215:240]
        return int(((band[..., 0] > 180) & (band[..., 1] > 180)
                    & (band[..., 2] > 180)).sum())
    if game.startswith("ChipnDale"):
        band = frame_rgb[16:28, 200:240]
        return int(((band[..., 0] > 180) & (band[..., 1] > 180)
                    & (band[..., 2] > 180)).sum())
    return 0


def play_episode(policy, game: str, integrations: str | None,
                 n_frames: int, seed: int = 0, temperature: float = 0.5) -> dict:
    from nes_player.emulator.stable_retro import StableRetroAdapter
    from oracle_mpc import begin_any

    env = StableRetroAdapter(game, include_debug=True, state=None,
                             integration_dir=integrations)
    obs = begin_any(env, game)

    for _ in range(seed * 7 % 30):
        obs = env.step_buttons([frozenset()])

    policy.reset()
    best_score = 0
    deaths = 0
    lives = (obs.debug or {}).get("lives")

    for i in range(n_frames):
        pressed, _ = policy.act(obs.frame_rgb, temperature)
        pressed = pressed - {"START", "SELECT"}
        obs = env.step_buttons([pressed])

        score = score_of(obs.frame_rgb, game)
        best_score = max(best_score, score)

        now = (obs.debug or {}).get("lives")
        if lives is not None and now is not None and now < lives:
            deaths += 1
        lives = now

    return {"best_score": best_score, "deaths": deaths, "frames": n_frames}


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--game", default="Galaga-Nes-v0", help="game to play (held out)")
    p.add_argument("--frames", type=int, default=5000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--bc-checkpoint", default="runs/bc_galaga_shoot")
    p.add_argument("--integrations", default="integrations/")
    args = p.parse_args()

    test_game = args.game
    train_games = [g for g in GAMES if g != test_game]
    print(f"=== DT Transfer Gameplay: train on {train_games}, play {test_game} ===")

    # Step 1: Train DT on 3 games
    print("\n--- Train DT on 3 games ---")
    t0 = time.time()
    transfer_dt = train_dt_on_games(train_games, epochs=args.epochs)
    train_time = time.time() - t0
    print(f"trained in {train_time:.0f}s, params={count_params(transfer_dt):,}")

    # Save transferred model
    ckpt_path = Path(f"runs/dt_transfer_{test_game}.pt")
    torch.save(transfer_dt.state_dict(), ckpt_path)
    print(f"saved to {ckpt_path}")

    # Step 2: Play with transferred DT
    from nes_player.policy.dt_policy import DTPolicy

    results = {}

    print("\n--- Play with transferred DT ---")
    dt_transfer_policy = DTPolicy(ckpt_path, target_rtg=50.0)
    dt_scores = []
    for seed in range(args.seeds):
        r = play_episode(dt_transfer_policy, test_game, args.integrations,
                         args.frames, seed)
        dt_scores.append(r["best_score"])
        print(f"  seed {seed}: score={r['best_score']}, deaths={r['deaths']}")
    results["dt_transfer"] = dt_scores

    # Step 3: Play with BC (in-domain baseline)
    print("\n--- Play with BC (in-domain) ---")
    from nes_player.policy.bc import BCPolicy
    bc = BCPolicy(args.bc_checkpoint)
    bc_scores = []
    for seed in range(args.seeds):
        r = play_episode(bc, test_game, args.integrations, args.frames, seed)
        bc_scores.append(r["best_score"])
        print(f"  seed {seed}: score={r['best_score']}, deaths={r['deaths']}")
    results["bc_indomain"] = bc_scores

    # Summary
    print("\n=== Summary ===")
    dt_mean = np.mean(dt_scores)
    bc_mean = np.mean(bc_scores)
    print(f"DT (transfer, 3 games): {dt_mean:.0f}")
    print(f"BC (in-domain):         {bc_mean:.0f}")
    print(f"DT transfer - BC:       {dt_mean - bc_mean:+.0f}")

    out = {
        "test_game": test_game,
        "train_games": train_games,
        "frames_per_episode": args.frames,
        "seeds": args.seeds,
        "dt_transfer_scores": dt_scores,
        "bc_indomain_scores": bc_scores,
        "dt_transfer_mean": float(dt_mean),
        "bc_indomain_mean": float(bc_mean),
        "diff": float(dt_mean - bc_mean),
        "train_time_s": round(train_time, 1),
    }
    out_path = Path(f"runs/transfer_play_{test_game}.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n-> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
