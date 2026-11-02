#!/usr/bin/env python3
"""E4: DT + planner — does return-conditioned DT beat BC?

Offline comparison: on test windows, have DT predict actions with high rtg
target, compare with BC predictions. Metric: which policy's predictions
better match the actions that led to high returns.

    uv run python scripts/e4_dt_planner.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from nes_player.policy.decision_transformer import DTConfig  # noqa: E402

GAMES = ["Galaga-Nes-v0", "ChipnDale-Nes-v0"]


def bitmask_to_index(mask: int) -> int:
    bits = [i for i in range(8) if mask & (1 << i)]
    if len(bits) == 1:
        return bits[0]
    return 8


def load_game_data(game: str, window=32, stride=8):
    ep = Path(f"datasets/agent_play/{game}_agent_long")
    if not ep.exists():
        return None

    actions = np.load(ep / "actions.npy")
    rtg = np.load(ep / "return_to_go.npy")

    import zarr
    z = zarr.open(str(ep / "episode.zarr"), mode="r")
    frames = z["frames"][:]

    T = len(frames)
    returns, states, actions_out, rtg_levels = [], [], [], []
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
        # Classify window by its max rtg (high vs low return)
        rtg_levels.append(float(rtg_chunk.max()))

    return (
        torch.tensor(np.array(returns), dtype=torch.float32),
        torch.tensor(np.array(states), dtype=torch.float32),
        torch.tensor(np.array(actions_out), dtype=torch.long),
        np.array(rtg_levels),
    )


class StateProj(torch.nn.Module):
    def __init__(self, d_model=384):
        super().__init__()
        self.proj = torch.nn.Linear(6, d_model)

    def forward(self, x):
        return self.proj(x)


def main() -> int:
    print("=== E4: DT + planner — return-conditioned vs BC ===")

    # Load DT
    cfg = DTConfig(n_actions=9, d_model=384, n_layers=6, n_heads=6)
    dt_model = cfg.build(state_encoder=StateProj())
    ckpt = Path("runs/dt_galaga.pt")
    if ckpt.exists():
        dt_model.load_state_dict(
            torch.load(ckpt, map_location="cpu", weights_only=True)
        )
        print(f"Loaded DT from {ckpt}")
    dt_model.eval()

    results = {}
    for game in GAMES:
        print(f"\n--- {game} ---")
        data = load_game_data(game)
        if data is None:
            continue

        returns, states, actions, rtg_levels = data
        n = len(returns)
        print(f"windows: {n}, rtg range [{rtg_levels.min():.0f}, {rtg_levels.max():.0f}]")

        # Split into high-return and low-return windows (upper/lower half)
        q75 = np.percentile(rtg_levels, 75)
        q25 = np.percentile(rtg_levels, 25)
        high_mask = rtg_levels >= q75
        low_mask = rtg_levels <= q25
        print(f"high-return windows: {high_mask.sum()}, low: {low_mask.sum()}")

        # Test 1: DT with high rtg target vs low rtg target
        # Which better predicts actions in high-return windows?
        rtg_targets = [0, 20, 40, 60, 79]
        high_return_accs = {}
        low_return_accs = {}

        for target in rtg_targets:
            test_returns = returns.clone()
            test_returns[:] = float(target)

            with torch.no_grad():
                logits = dt_model(test_returns, states, actions)
                preds = logits[:, -1, :].argmax(dim=-1)

            # Accuracy on high-return windows
            high_preds = preds[high_mask]
            high_gt = actions[high_mask, -1]
            high_acc = (high_preds == high_gt).float().mean().item()
            high_return_accs[target] = high_acc

            # Accuracy on low-return windows
            low_preds = preds[low_mask]
            low_gt = actions[low_mask, -1]
            low_acc = (low_preds == low_gt).float().mean().item()
            low_return_accs[target] = low_acc

            print(f"  rtg={target:3d}: high-return acc={high_acc:.4f}, low-return acc={low_acc:.4f}")

        # Find best rtg target for high-return windows
        best_target = max(high_return_accs, key=high_return_accs.get)
        best_high_acc = high_return_accs[best_target]

        # Baseline: majority action on high-return windows
        high_gt_flat = actions[high_mask, -1].numpy()
        majority = Counter(high_gt_flat.tolist()).most_common(1)[0]
        majority_acc = majority[1] / len(high_gt_flat)

        print(f"\nBest rtg target for high-return: {best_target} (acc={best_high_acc:.4f})")
        print(f"Majority baseline on high-return: {majority_acc:.4f}")
        print(f"DT - baseline: {best_high_acc - majority_acc:+.4f}")

        results[game] = {
            "n_windows": int(n),
            "high_return_windows": int(high_mask.sum()),
            "low_return_windows": int(low_mask.sum()),
            "high_return_accs": {str(k): round(v, 4) for k, v in high_return_accs.items()},
            "low_return_accs": {str(k): round(v, 4) for k, v in low_return_accs.items()},
            "best_target": best_target,
            "best_high_acc": round(best_high_acc, 4),
            "majority_acc": round(majority_acc, 4),
            "dt_minus_baseline": round(best_high_acc - majority_acc, 4),
        }

    # Summary
    print("\n=== Summary ===")
    for game, r in results.items():
        print(f"{game}: DT {r['best_high_acc']:.4f} vs baseline {r['majority_acc']:.4f} "
              f"({r['dt_minus_baseline']:+.4f})")

    out = {
        "experiment": "E4: DT + planner — return-conditioned vs BC",
        "results": results,
    }
    out_path = Path("runs/e4_dt_planner.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n-> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
