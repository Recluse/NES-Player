#!/usr/bin/env python3
"""E2: Return-to-go as a dial — does DT change behavior with target?

Test: same state, different rtg targets → different actions?
If yes, the dial works.

    uv run python scripts/e2_rtg_dial.py --episode datasets/agent_play/Galaga-Nes-v0_agent_long --game Galaga-Nes-v0
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
from nes_player.policy.decision_transformer import DTConfig  # noqa: E402


def bitmask_to_index(mask: int) -> int:
    bits = [i for i in range(8) if mask & (1 << i)]
    if len(bits) == 1:
        return bits[0]
    return 8


def load_episode(ep_dir: Path):
    """Load frames, actions, return-to-go from episode directory."""
    ep_dir = Path(ep_dir)
    actions = np.load(ep_dir / "actions.npy")
    rtg = np.load(ep_dir / "return_to_go.npy")

    import zarr
    z = zarr.open(str(ep_dir / "episode.zarr"), mode="r")
    frames = z["frames"][:]

    return frames, actions, rtg


def make_windows(ep, actions, rtg, window=32, stride=8):
    """Create (return, state, action) windows for DT training."""
    T = len(ep)
    returns, states, actions_out = [], [], []
    for i in range(0, T - window, stride):
        chunk = ep[i : i + window]
        if len(chunk) < window:
            continue
        rtg_chunk = rtg[i : i + window]
        act_chunk = actions[i : i + window]
        if len(rtg_chunk) < window or len(act_chunk) < window:
            continue
        # Use score-band brightness as state feature
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


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--episode", default="datasets/agent_play/Galaga-Nes-v0_agent_long")
    p.add_argument("--game", default="Galaga-Nes-v0")
    p.add_argument("--ckpt", default="runs/dt_galaga.pt")
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--n-samples", type=int, default=100, help="Number of test windows")
    args = p.parse_args()

    print(f"=== E2: RTG Dial on {args.game} ===")

    # Load model
    from nes_player.policy.decision_transformer import count_params
    class StateProj(torch.nn.Module):
        def __init__(self, d_model=384):
            super().__init__()
            self.proj = torch.nn.Linear(6, d_model)
        def forward(self, x):
            return self.proj(x)

    cfg = DTConfig(n_actions=9, d_model=384, n_layers=6, n_heads=6)
    model = cfg.build(state_encoder=StateProj())
    ckpt_path = Path(args.ckpt)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"No checkpoint at {ckpt_path}, using untrained model")
    model.eval()
    print(f"DT params: {count_params(model):,}")

    # Load episode
    t0 = time.time()
    frames, actions, rtg = load_episode(Path(args.episode))
    print(f"episode: {len(frames)} frames, rtg range [{rtg.min()}, {rtg.max()}]")

    # Create test windows
    returns, states, actions_w = make_windows(frames, actions, rtg, window=args.window)
    n = min(args.n_samples, len(returns))
    print(f"test windows: {n}")

    # Test: same states, different rtg targets
    rtg_targets = [0, 20, 40, 60, 79]  # from min to max
    results = {}

    for target in rtg_targets:
        # Set all rtg to the target (constant return-to-go)
        test_returns = returns[:n].clone()
        test_returns[:] = float(target)

        with torch.no_grad():
            logits = model(test_returns, states[:n], actions_w[:n])
            preds = logits[:, -1, :].argmax(dim=-1)  # predict last action

        # Compare with ground truth
        gt = actions_w[:n, -1]
        acc = (preds == gt).float().mean().item()
        action_dist = Counter(preds.numpy().tolist())

        results[target] = {
            "accuracy": acc,
            "action_distribution": dict(action_dist),
        }
        print(f"  rtg={target:3d}: acc={acc:.4f}, actions={dict(action_dist)}")

    # Key question: do action distributions change with rtg?
    dists = [results[t]["action_distribution"] for t in rtg_targets]
    all_actions = set()
    for d in dists:
        all_actions.update(d.keys())

    # Compute distribution differences
    print("\n=== Distribution shifts ===")
    for i, t1 in enumerate(rtg_targets):
        for t2 in rtg_targets[i+1:]:
            d1, d2 = dists[i], dists[rtg_targets.index(t2)]
            # Total variation distance
            tvd = 0.5 * sum(
                abs(d1.get(a, 0) / n - d2.get(a, 0) / n) for a in all_actions
            )
            print(f"  rtg {t1} vs {t2}: TVD={tvd:.4f}")

    # Does rtg=79 (high) produce more "aggressive" actions than rtg=0 (low)?
    # In Galaga, aggressive = fire (button B = index 1)
    high_dist = results[79]["action_distribution"]
    low_dist = results[0]["action_distribution"]
    fire_high = high_dist.get(1, 0) / n
    fire_low = low_dist.get(1, 0) / n
    print(f"\nFire rate (action=1): rtg=0 → {fire_low:.3f}, rtg=79 → {fire_high:.3f}")

    # Save results
    out = {
        "game": args.game,
        "n_windows": n,
        "rtg_range": [int(rtg.min()), int(rtg.max())],
        "results": {str(t): results[t] for t in rtg_targets},
        "fire_rate_low": fire_low,
        "fire_rate_high": fire_high,
    }
    out_path = Path(f"runs/e2_rtg_dial_{args.game}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n-> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
