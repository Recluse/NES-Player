#!/usr/bin/env python3
"""QDT: Train DT with relabelled return-to-go.

Steps:
1. Train value function V(s) on agent-play data.
2. Relabel RTG using V(s) (Algorithm 1 from QDT paper).
3. Train DT on relabelled data.
4. Compare: DT (original rtg) vs QDT (relabelled rtg).

    uv run python scripts/train_qdt.py --episode datasets/agent_play/Galaga-Nes-v0_agent_long --game Galaga-Nes-v0
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from nes_player.policy.decision_transformer import DTConfig, count_params  # noqa: E402
from nes_player.policy.qdt import (  # noqa: E402
    consistency_relabel,
    relabel_rtg,
    train_value_function,
)


def bitmask_to_index(mask: int) -> int:
    bits = [i for i in range(8) if mask & (1 << i)]
    if len(bits) == 1:
        return bits[0]
    return 8


def load_episode_data(ep_dir: Path):
    """Load frames, actions, return-to-go from episode directory."""
    ep_dir = Path(ep_dir)
    actions = np.load(ep_dir / "actions.npy")
    rtg = np.load(ep_dir / "return_to_go.npy")

    import zarr
    z = zarr.open(str(ep_dir / "episode.zarr"), mode="r")
    frames = z["frames"][:]

    return frames, actions, rtg


def compute_rewards(rtg: np.ndarray) -> np.ndarray:
    """Derive immediate rewards from return-to-go: r_t = R_t - R_{t+1}."""
    rewards = np.zeros_like(rtg)
    rewards[:-1] = rtg[:-1] - rtg[1:]
    rewards[-1] = rtg[-1]  # last step: R_T = r_T
    return rewards


def make_windows(frames, actions, rtg, window=32, stride=8):
    """Create (return, state, action) windows for DT training."""
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


def train_dt(returns, states, actions, epochs=10):
    """Train DT on given data."""
    cfg = DTConfig(n_actions=9, d_model=384, n_layers=6, n_heads=6)
    model = cfg.build(state_encoder=StateProj())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    n = len(returns)
    bs = 32
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
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--vf-epochs", type=int, default=50)
    p.add_argument("--window", type=int, default=32)
    args = p.parse_args()

    print(f"=== QDT: {args.game} ===")

    # Load episode
    t0 = time.time()
    frames, actions, rtg = load_episode_data(Path(args.episode))
    # Expand rtg to match frame length (rtg may be subsampled)
    if len(rtg) < len(frames):
        factor = len(frames) // len(rtg)
        rtg = np.repeat(rtg, factor)[:len(frames)]
    print(f"episode: {len(frames)} frames, rtg range [{rtg.min()}, {rtg.max()}]")

    # Derive rewards from rtg
    rewards = compute_rewards(rtg)
    print(f"rewards range [{rewards.min():.2f}, {rewards.max():.2f}]")

    # Step 1: Train value function
    print("\n--- Step 1: Train value function ---")
    # Use frame features as states
    T = len(frames)
    states_raw = np.zeros((T, 6), dtype=np.float32)
    for i, f in enumerate(frames):
        band = f[8:24, 195:240]
        sc = int(((band[..., 0] > 180) & (band[..., 1] > 180)
                  & (band[..., 2] > 180)).sum())
        bright = float(f.mean())
        red = float((f[..., 0] > 150).mean())
        green = float((f[..., 1] > 150).mean())
        blue = float((f[..., 2] > 150).mean())
        dark = float((f.mean(axis=-1) < 40).mean())
        states_raw[i] = [sc, bright, red, green, blue, dark]

    # Reshape for value function training: (N, T, state_dim) where N=1 (one episode)
    # Actually train on all timesteps
    vf = train_value_function(
        states_raw[None, :, :],  # (1, T, 6)
        rtg[None, :],  # (1, T)
        epochs=args.vf_epochs,
    )

    # Step 2: Relabel RTG (Algorithm 1)
    print("\n--- Step 2: Relabel RTG ---")
    relabeled_rtg = relabel_rtg(
        states_raw[None, :, :],  # (1, T, 6)
        actions[None, :],  # (1, T)
        rewards[None, :],  # (1, T)
        vf,
    )[0]  # (T,)
    print(f"relabelled rtg range [{relabeled_rtg.min():.2f}, {relabeled_rtg.max():.2f}]")
    print(f"original rtg range [{rtg.min():.2f}, {rtg.max():.2f}]")

    # Step 3: Consistency relabel (Algorithm 2)
    print("\n--- Step 3: Consistency relabel ---")
    consistent_rtg = consistency_relabel(
        rewards[None, :],  # (1, T)
        relabeled_rtg[None, :],  # (1, T)
        context_len=args.window,
    )[0]  # (T,)
    print(f"consistent rtg range [{consistent_rtg.min():.2f}, {consistent_rtg.max():.2f}]")

    # Step 4: Train DT on original data (baseline)
    print("\n--- Step 4: Train DT (original rtg) ---")
    returns_orig, states_w, actions_w = make_windows(
        frames, actions, rtg, window=args.window
    )
    dt_orig, dt_log_orig = train_dt(returns_orig, states_w, actions_w, epochs=args.epochs)

    # Step 5: Train QDT on relabelled data
    print("\n--- Step 5: Train QDT (relabelled rtg) ---")
    returns_qdt, _, _ = make_windows(
        frames, actions, consistent_rtg, window=args.window
    )
    dt_qdt, dt_log_qdt = train_dt(returns_qdt, states_w, actions_w, epochs=args.epochs)

    # Compare: action prediction accuracy on test windows
    print("\n=== Comparison ===")
    dt_orig.eval()
    dt_qdt.eval()
    with torch.no_grad():
        # Test on last 20% of windows
        n_test = int(len(returns_orig) * 0.2)
        test_orig = returns_orig[-n_test:]
        test_qdt = returns_qdt[-n_test:]
        test_states = states_w[-n_test:]
        test_actions = actions_w[-n_test:]

        logits_orig = dt_orig(test_orig, test_states, test_actions)
        preds_orig = logits_orig[:, -1, :].argmax(dim=-1)
        acc_orig = (preds_orig == test_actions[:, -1]).float().mean().item()

        logits_qdt = dt_qdt(test_qdt, test_states, test_actions)
        preds_qdt = logits_qdt[:, -1, :].argmax(dim=-1)
        acc_qdt = (preds_qdt == test_actions[:, -1]).float().mean().item()

    print(f"DT (original):  acc={acc_orig:.4f}")
    print(f"QDT (relabel):  acc={acc_qdt:.4f}")
    print(f"QDT - DT:       {acc_qdt - acc_orig:+.4f}")

    # Check if rtg conditioning works now
    print("\n--- RTG conditioning test ---")
    rtg_targets = [0, 20, 40, 60, 79]
    for target in rtg_targets:
        test_returns = test_qdt.clone()
        test_returns[:] = float(target)
        with torch.no_grad():
            logits = dt_qdt(test_returns, test_states, test_actions)
            preds = logits[:, -1, :].argmax(dim=-1)
        acc = (preds == test_actions[:, -1]).float().mean().item()
        from collections import Counter
        dist = Counter(preds.numpy().tolist())
        print(f"  rtg={target:3d}: acc={acc:.4f}, actions={dict(dist)}")

    # Save results
    result = {
        "game": args.game,
        "n_frames": int(len(frames)),
        "rtg_original": [float(rtg.min()), float(rtg.max())],
        "rtg_relabelled": [float(relabeled_rtg.min()), float(relabeled_rtg.max())],
        "rtg_consistent": [float(consistent_rtg.min()), float(consistent_rtg.max())],
        "dt_original_acc": round(acc_orig, 4),
        "qdt_acc": round(acc_qdt, 4),
        "qdt_minus_dt": round(acc_qdt - acc_orig, 4),
        "dt_log_original": dt_log_orig,
        "dt_log_qdt": dt_log_qdt,
    }
    out = Path(f"runs/qdt_{args.game}.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"\n-> {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
