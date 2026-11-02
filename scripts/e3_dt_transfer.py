#!/usr/bin/env python3
"""E3: DT on all 4 games — does it transfer between games?

Train one DT on all agent-play episodes, test action prediction per game.
Leave-one-out: train on 3, test on 4th.

    uv run python scripts/e3_dt_transfer.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from nes_player.policy.decision_transformer import DTConfig, count_params  # noqa: E402

GAMES = ["Battletoads-Nes-v0", "ChipnDale-Nes-v0", "Galaga-Nes-v0", "ViceProjectDoom-Nes-v0"]


def bitmask_to_index(mask: int) -> int:
    bits = [i for i in range(8) if mask & (1 << i)]
    if len(bits) == 1:
        return bits[0]
    return 8


def load_game_data(game: str, window=32, stride=16):
    ep = Path(f"datasets/agent_play/{game}_agent_long")
    if not ep.exists():
        return None

    actions = np.load(ep / "actions.npy")
    rtg = np.load(ep / "return_to_go.npy")

    import zarr
    z = zarr.open(str(ep / "episode.zarr"), mode="r")
    frames = z["frames"][:]

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


class StateProj(torch.nn.Module):
    def __init__(self, d_model=384):
        super().__init__()
        self.proj = torch.nn.Linear(6, d_model)

    def forward(self, x):
        return self.proj(x)


def train_dt(data, epochs=5):
    """Train DT on combined data."""
    returns, states, actions = data
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


def evaluate(model, data):
    """Evaluate action prediction accuracy."""
    returns, states, actions = data
    model.eval()
    with torch.no_grad():
        logits = model(returns, states, actions)
        preds = logits.argmax(dim=-1)
        acc = (preds == actions).float().mean().item()
    model.train()
    return acc


def main() -> int:
    print("=== E3: DT Transfer across 4 games ===")

    # Load all game data
    game_data = {}
    for g in GAMES:
        data = load_game_data(g)
        if data is None:
            print(f"  {g}: MISSING")
            continue
        game_data[g] = data
        print(f"  {g}: {data[0].shape[0]} windows")

    if len(game_data) < 2:
        print("Need at least 2 games")
        return 1

    # Leave-one-out cross-validation
    results = {}
    for test_game in game_data:
        print(f"\n--- Test on {test_game} ---")

        # Train on all games except test_game
        train_returns, train_states, train_actions = [], [], []
        for g, data in game_data.items():
            if g == test_game:
                continue
            train_returns.append(data[0])
            train_states.append(data[1])
            train_actions.append(data[2])

        train_data = (
            torch.cat(train_returns),
            torch.cat(train_states),
            torch.cat(train_actions),
        )
        print(f"  train: {train_data[0].shape[0]} windows from {len(game_data)-1} games")

        t0 = time.time()
        model = train_dt(train_data, epochs=5)
        train_time = time.time() - t0

        # Evaluate on test game
        test_acc = evaluate(model, game_data[test_game])

        # Baseline: majority action on test game
        test_actions = game_data[test_game][2]
        flat = test_actions.reshape(-1).numpy()
        majority = Counter(flat.tolist()).most_common(1)[0]
        baseline_acc = majority[1] / len(flat)

        results[test_game] = {
            "test_acc": round(test_acc, 4),
            "baseline_acc": round(baseline_acc, 4),
            "train_windows": int(train_data[0].shape[0]),
            "train_time_s": round(train_time, 1),
        }
        print(f"  test acc={test_acc:.4f}, baseline={baseline_acc:.4f}")
        print(f"  DT - baseline: {test_acc - baseline_acc:+.4f}")

    # Summary
    print("\n=== Summary ===")
    dt_accs = [r["test_acc"] for r in results.values()]
    base_accs = [r["baseline_acc"] for r in results.values()]
    print(f"DT mean acc:      {np.mean(dt_accs):.4f}")
    print(f"Baseline mean:    {np.mean(base_accs):.4f}")
    print(f"DT - baseline:    {np.mean(dt_accs) - np.mean(base_accs):+.4f}")

    # Save results
    out = {
        "experiment": "E3: DT transfer across 4 games",
        "leave_one_out": results,
        "mean_dt_acc": float(np.mean(dt_accs)),
        "mean_baseline_acc": float(np.mean(base_accs)),
    }
    out_path = Path("runs/e3_dt_transfer.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n-> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
