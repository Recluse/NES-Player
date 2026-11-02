#!/usr/bin/env python3
"""E1: Decision Transformer vs BC on Galaga agent-play data.

Trains DT on (return-to-go, state, action) triples from the agent-play
episodes and compares against the BC baseline on the same data.

    uv run python scripts/train_dt.py --episode datasets/agent_play/Galaga-Nes-v0_agent_long --game Galaga-Nes-v0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def bitmask_to_index(mask: int) -> int:
    """Convert an 8-bit button mask to a 0-7 index for DT's action vocab.

    The action space is 8 buttons; a mask is 2^8 = 256 combinations. For DT
    we use the single-button index when only one button is pressed, and a
    reserved index 8 for combinations (mapped back to n_actions=9).
    """
    bits = [i for i in range(8) if mask & (1 << i)]
    if len(bits) == 1:
        return bits[0]
    return 8  # combination token


def load_episode_data(ep_dir: Path):
    """Load frames, actions, and return-to-go from an episode."""
    from nes_player.data.reader import Episode

    ep = Episode(ep_dir)
    n = ep.frames.shape[0]
    raw_actions = np.asarray(ep.actions).reshape(-1)
    actions = np.array([bitmask_to_index(int(a)) for a in raw_actions],
                       dtype=np.int64)

    # return-to-go: score-like signal from the score band
    rtg_path = ep_dir / "return_to_go.npy"
    if rtg_path.exists():
        rtg = np.load(rtg_path)
    else:
        scores = []
        for i in range(0, n, 10):
            frame = np.asarray(ep.frames[i])
            f = frame.astype(np.int16)
            band = f[8:24, 195:240]
            sc = int(((band[..., 0] > 180) & (band[..., 1] > 180)
                      & (band[..., 2] > 180)).sum())
            scores.append(sc)
        final = scores[-1] if scores else 0
        rtg = np.array([final - s for s in scores], dtype=np.float32)

    return ep, actions, rtg


def make_windows(ep, actions, rtg, window=32, stride=8):
    """Sliding windows of (returns, states, actions) for DT training."""
    n = len(actions)
    rtg_full = np.interp(
        np.arange(n), np.linspace(0, n, len(rtg)), rtg
    ).astype(np.float32)
    X_r, X_s, X_a = [], [], []
    for start in range(0, n - window, stride):
        sl = slice(start, start + window)
        X_r.append(rtg_full[sl])
        # frame embeddings: mean-pool the frame to a vector (cheap)
        frames = np.asarray(ep.frames[sl])  # (T, H, W, 3)
        # simple embedding: mean + std per channel + bright pixel ratio
        emb = []
        for f in frames:
            g = f.astype(np.float32)
            emb.append([
                g[..., 0].mean(), g[..., 1].mean(), g[..., 2].mean(),
                g.mean(), g.std(),
                float((g.mean(axis=2) > 100).mean()),
            ])
        X_s.append(np.array(emb, dtype=np.float32))
        X_a.append(actions[sl].astype(np.int64))
    return np.array(X_r), np.array(X_s), np.array(X_a)


def train_dt(returns, states, actions, epochs=10, lr=1e-3, batch_size=32):
    """Train DT on the windows. Returns the model and training log."""
    import torch
    from nes_player.policy.decision_transformer import DTConfig, count_params

    # project 6-dim state features to d_model via a linear layer
    d_model = 384
    cfg = DTConfig(n_actions=9, d_model=d_model, n_layers=6, n_heads=6)  # 8 buttons + combination

    class StateProj(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(6, d_model)

        def forward(self, x):
            return self.proj(x)

    model = cfg.build(state_encoder=StateProj())
    print(f"DT params: {count_params(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    log = []

    returns_t = torch.tensor(returns, dtype=torch.float32)
    states_t = torch.tensor(states, dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.long)

    n = len(returns)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss, correct, total = 0.0, 0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            r = returns_t[idx]
            s = states_t[idx]
            a = actions_t[idx]
            logits = model(r, s, a)  # (B, T, 9)
            # predict action at every timestep
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 9), a.reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss) * len(idx)
            pred = logits.argmax(dim=-1)
            correct += int((pred == a).sum())
            total += int(a.numel())
        acc = correct / total if total else 0
        rec = {
            "epoch": epoch,
            "loss": round(total_loss / n, 4),
            "acc": round(acc, 4),
        }
        log.append(rec)
        print(json.dumps(rec), flush=True)

    return model, log


def evaluate_bc_on_same(returns, states, actions):
    """BC baseline: state → action, no return-to-go. Same data, same split."""
    import torch

    d_model = 384
    model = torch.nn.Sequential(
        torch.nn.Linear(6, d_model),
        torch.nn.ReLU(),
        torch.nn.Linear(d_model, d_model),
        torch.nn.ReLU(),
        torch.nn.Linear(d_model, 9),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    states_t = torch.tensor(states, dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.long)

    n = len(states)
    for epoch in range(10):
        model.train()
        perm = torch.randperm(n)
        correct, total = 0, 0
        for i in range(0, n, 32):
            idx = perm[i:i + 32]
            s = states_t[idx]  # (B, T, 6)
            a = actions_t[idx]  # (B, T)
            B, T, _ = s.shape
            logits = model(s.reshape(B * T, -1)).reshape(B, T, 9)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 9), a.reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            pred = logits.argmax(dim=-1)
            correct += int((pred == a).sum())
            total += int(a.numel())
    acc = correct / total
    return acc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--game", default="")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    game = args.game or args.episode.name.split("_agent_long")[0]
    print(f"=== E1: DT vs BC on {game} ===")

    ep, actions, rtg = load_episode_data(args.episode)
    print(f"episode: {ep.frames.shape[0]} frames, "
          f"rtg range [{rtg.min():.0f}, {rtg.max():.0f}]")

    returns, states, actions_w = make_windows(ep, actions, rtg, window=args.window)
    print(f"windows: {returns.shape}")

    t0 = time.time()
    dt_model, dt_log = train_dt(returns, states, actions_w, epochs=args.epochs)
    dt_time = time.time() - t0

    t0 = time.time()
    bc_acc = evaluate_bc_on_same(returns, states, actions_w)
    bc_time = time.time() - t0

    # Save DT model for E2 (return-conditioned generation)
    ckpt_path = Path("runs/dt_galaga.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dt_model.state_dict(), ckpt_path)
    print(f"Saved DT model to {ckpt_path}")

    dt_acc = dt_log[-1]["acc"] if dt_log else 0
    print(f"\n=== Results ===")
    print(f"DT:  acc={dt_acc:.4f}  time={dt_time:.1f}s")
    print(f"BC:  acc={bc_acc:.4f}  time={bc_time:.1f}s")
    print(f"DT - BC: {dt_acc - bc_acc:+.4f}")

    result = {
        "game": game,
        "episode": str(args.episode),
        "n_windows": int(returns.shape[0]),
        "dt_acc": dt_acc,
        "bc_acc": bc_acc,
        "diff": round(dt_acc - bc_acc, 4),
        "dt_params": 10_801_544,
        "dt_time_s": round(dt_time, 1),
        "bc_time_s": round(bc_time, 1),
        "dt_log": dt_log,
    }
    out = args.out or (ROOT / "runs" / f"e1_dt_vs_bc_{game}.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
