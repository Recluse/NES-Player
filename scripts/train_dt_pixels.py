#!/usr/bin/env python3
"""Train DT with pixel state encoder (rich features vs 6 hand-crafted).

Compares:
- DT with 6-dim hand-crafted features (baseline)
- DT with pixel CNN encoder (rich features)

Question: does richer state representation make RTG conditioning work?

    uv run python scripts/train_dt_pixels.py --episode datasets/agent_play/Galaga-Nes-v0_agent_long --game Galaga-Nes-v0
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
from nes_player.policy.pixel_encoder import PixelStateEncoder  # noqa: E402


def bitmask_to_index(mask: int) -> int:
    bits = [i for i in range(8) if mask & (1 << i)]
    if len(bits) == 1:
        return bits[0]
    return 8


def load_episode_data(ep_dir: Path):
    ep_dir = Path(ep_dir)
    actions = np.load(ep_dir / "actions.npy")
    rtg = np.load(ep_dir / "return_to_go.npy")

    import zarr
    z = zarr.open(str(ep_dir / "episode.zarr"), mode="r")
    frames = z["frames"][:]

    return frames, actions, rtg


def make_windows(frames, actions, rtg, window=32, stride=8):
    """Create (return, state, action) windows with raw frames."""
    T = len(frames)
    returns, states, actions_out = [], [], []
    for i in range(0, T - window, stride):
        chunk = frames[i : i + window]
        rtg_chunk = rtg[i : i + window]
        act_chunk = actions[i : i + window]
        if len(chunk) < window or len(rtg_chunk) < window or len(act_chunk) < window:
            continue
        # Downsample frames to 112x120 to save memory
        small_frames = []
        for f in chunk:
            small = f[::2, ::2]  # (112, 120, 3)
            small_frames.append(small)
        states.append(np.array(small_frames, dtype=np.float32) / 255.0)
        returns.append(rtg_chunk)
        acts = [bitmask_to_index(a) for a in act_chunk]
        actions_out.append(acts)
    return (
        torch.tensor(np.array(returns), dtype=torch.float32),
        torch.tensor(np.array(states), dtype=torch.float32),
        torch.tensor(np.array(actions_out), dtype=torch.long),
    )


def train_dt_pixel(returns, states, actions, epochs=10, lr=1e-4):
    """Train DT with pixel encoder."""
    cfg = DTConfig(n_actions=9, d_model=384, n_layers=4, n_heads=6)
    encoder = PixelStateEncoder(d_model=384)
    model = cfg.build(state_encoder=encoder)
    print(f"DT params: {count_params(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = len(returns)
    bs = 8  # smaller batch for pixel data
    dt_log = []
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
        dt_log.append({"epoch": epoch, "loss": round(total_loss/n, 4), "acc": round(acc, 4)})
        print(f"  epoch {epoch}: loss={total_loss/n:.4f} acc={acc:.4f}")

    return model, dt_log


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--episode", default="datasets/agent_play/Galaga-Nes-v0_agent_long")
    p.add_argument("--game", default="Galaga-Nes-v0")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--stride", type=int, default=16)
    args = p.parse_args()

    print(f"=== DT + Pixel Encoder: {args.game} ===")

    # Load episode
    t0 = time.time()
    frames, actions, rtg = load_episode_data(Path(args.episode))
    if len(rtg) < len(frames):
        factor = len(frames) // len(rtg)
        rtg = np.repeat(rtg, factor)[:len(frames)]
    print(f"episode: {len(frames)} frames, rtg range [{rtg.min()}, {rtg.max()}]")

    # Create windows with raw frames
    print("\n--- Create windows ---")
    returns, states, actions_w = make_windows(
        frames, actions, rtg, window=args.window, stride=args.stride
    )
    print(f"windows: {returns.shape[0]}, state shape: {states.shape}")

    # Train DT with pixel encoder
    print("\n--- Train DT with pixel encoder ---")
    model, dt_log = train_dt_pixel(returns, states, actions_w, epochs=args.epochs)

    # Test RTG conditioning
    print("\n--- RTG conditioning test ---")
    model.eval()
    n_test = min(200, len(returns))
    test_returns = returns[-n_test:]
    test_states = states[-n_test:]
    test_actions = actions_w[-n_test:]

    rtg_targets = [0, 20, 40, 60, 79]
    for target in rtg_targets:
        test_r = test_returns.clone()
        test_r[:] = float(target)
        with torch.no_grad():
            logits = model(test_r, test_states, test_actions)
            preds = logits[:, -1, :].argmax(dim=-1)
        acc = (preds == test_actions[:, -1]).float().mean().item()
        dist = Counter(preds.numpy().tolist())
        print(f"  rtg={target:3d}: acc={acc:.4f}, actions={dict(dist)}")

    # Check if action distributions change with rtg
    dists = []
    for target in rtg_targets:
        test_r = test_returns.clone()
        test_r[:] = float(target)
        with torch.no_grad():
            logits = model(test_r, test_states, test_actions)
            preds = logits[:, -1, :].argmax(dim=-1)
        dists.append(Counter(preds.numpy().tolist()))

    all_actions = set()
    for d in dists:
        all_actions.update(d.keys())

    print("\n--- Distribution shifts ---")
    for i, t1 in enumerate(rtg_targets):
        for j, t2 in enumerate(rtg_targets[i+1:], i+1):
            d1, d2 = dists[i], dists[j]
            tvd = 0.5 * sum(
                abs(d1.get(a, 0) / n_test - d2.get(a, 0) / n_test) for a in all_actions
            )
            print(f"  rtg {t1} vs {t2}: TVD={tvd:.4f}")

    # Save results
    result = {
        "game": args.game,
        "n_windows": int(returns.shape[0]),
        "state_shape": list(states.shape),
        "dt_params": count_params(model),
        "rtg_range": [float(rtg.min()), float(rtg.max())],
        "dt_log": dt_log,
    }
    out = Path(f"runs/dt_pixels_{args.game}.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"\n-> {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
