#!/usr/bin/env python3
"""Compare DT with 6-dim features vs DINOv2 features.

Bottleneck: state features (6 hand-crafted numbers is too few).
Solution: DINOv2 frozen pretrained encoder (384-dim visual features).

    uv run python scripts/train_dt_dinov2.py --game Galaga-Nes-v0 \
        --episode datasets/agent_play/Galaga-Nes-v0_agent_long
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from nes_player.policy.decision_transformer import DTConfig, count_params  # noqa: E402


def bitmask_to_index(mask: int) -> int:
    bits = [i for i in range(8) if mask & (1 << i)]
    if len(bits) == 1:
        return bits[0]
    return 8


def frame_features_6dim(frame_rgb: np.ndarray) -> list[float]:
    """Original 6 hand-crafted features."""
    band = frame_rgb[8:24, 195:240]
    sc = int(((band[..., 0] > 180) & (band[..., 1] > 180)
              & (band[..., 2] > 180)).sum())
    bright = float(frame_rgb.mean())
    red = float((frame_rgb[..., 0] > 150).mean())
    green = float((frame_rgb[..., 1] > 150).mean())
    blue = float((frame_rgb[..., 2] > 150).mean())
    dark = float((frame_rgb.mean(axis=-1) < 40).mean())
    return [sc, bright, red, green, blue, dark]


class StateProj6(torch.nn.Module):
    """6-dim hand-crafted features -> d_model."""
    def __init__(self, d_model=384):
        super().__init__()
        self.proj = torch.nn.Linear(6, d_model)
    def forward(self, x):
        return self.proj(x)


class StateProjDINO(torch.nn.Module):
    """384-dim DINOv2 features -> d_model."""
    def __init__(self, d_model=384):
        super().__init__()
        self.proj = torch.nn.Linear(384, d_model)
    def forward(self, x):
        return self.proj(x)


def load_episode(ep_dir: Path):
    ep_dir = Path(ep_dir)
    actions = np.load(ep_dir / "actions.npy")
    rtg = np.load(ep_dir / "return_to_go.npy")
    import zarr
    z = zarr.open(str(ep_dir / "episode.zarr"), mode="r")
    frames = z["frames"][:]
    return frames, actions, rtg


def make_windows_6dim(frames, actions, rtg, window=32, stride=8):
    T = len(frames)
    returns, states, actions_out = [], [], []
    for i in range(0, T - window, stride):
        chunk = frames[i : i + window]
        rtg_chunk = rtg[i : i + window]
        act_chunk = actions[i : i + window]
        if len(chunk) < window or len(rtg_chunk) < window or len(act_chunk) < window:
            continue
        feats = [frame_features_6dim(f) for f in chunk]
        states.append(feats)
        returns.append(rtg_chunk)
        acts = [bitmask_to_index(a) for a in act_chunk]
        actions_out.append(acts)
    return (
        torch.tensor(np.array(returns), dtype=torch.float32),
        torch.tensor(np.array(states), dtype=torch.float32),
        torch.tensor(np.array(actions_out), dtype=torch.long),
    )


def make_windows_dino(dino_feats, actions, rtg, window=32, stride=8):
    """Use precomputed DINOv2 features."""
    T = len(dino_feats)
    returns, states, actions_out = [], [], []
    for i in range(0, T - window, stride):
        rtg_chunk = rtg[i : i + window]
        act_chunk = actions[i : i + window]
        feat_chunk = dino_feats[i : i + window]
        if len(feat_chunk) < window or len(rtg_chunk) < window or len(act_chunk) < window:
            continue
        states.append(feat_chunk)
        returns.append(rtg_chunk)
        acts = [bitmask_to_index(a) for a in act_chunk]
        actions_out.append(acts)
    return (
        torch.tensor(np.array(returns), dtype=torch.float32),
        torch.tensor(np.array(states), dtype=torch.float32),
        torch.tensor(np.array(actions_out), dtype=torch.long),
    )


def train_dt(returns, states, actions, encoder, epochs=10, lr=1e-4):
    cfg = DTConfig(n_actions=9, d_model=384, n_layers=4, n_heads=6)
    model = cfg.build(state_encoder=encoder)
    print(f"  DT params: {count_params(model):,}")
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    n = len(returns)
    bs = 32
    log = []
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
        log.append({"epoch": epoch, "loss": round(total_loss / n, 4), "acc": round(acc, 4)})
        print(f"  epoch {epoch}: loss={total_loss/n:.4f} acc={acc:.4f}")
    return model, log


def evaluate(model, returns, states, actions, n_test=200):
    model.eval()
    with torch.no_grad():
        r = returns[-n_test:]
        s = states[-n_test:]
        a = actions[-n_test:]
        logits = model(r, s, a)
        preds = logits[:, -1, :].argmax(dim=-1)
        acc = (preds == a[:, -1]).float().mean().item()
    model.train()
    return acc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="Galaga-Nes-v0")
    p.add_argument("--episode", default="datasets/agent_play/Galaga-Nes-v0_agent_long")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--stride", type=int, default=8)
    args = p.parse_args()

    print(f"=== DT: 6-dim vs DINOv2 features on {args.game} ===")

    # Load episode
    t0 = time.time()
    frames, actions, rtg = load_episode(Path(args.episode))
    if len(rtg) < len(frames):
        factor = len(frames) // len(rtg)
        rtg = np.repeat(rtg, factor)[:len(frames)]
    print(f"episode: {len(frames)} frames, rtg [{rtg.min()}, {rtg.max()}]")
    print(f"load time: {time.time()-t0:.1f}s")

    # === Experiment A: 6-dim features ===
    print("\n--- A: 6-dim hand-crafted features ---")
    t0 = time.time()
    ret6, st6, act6 = make_windows_6dim(
        frames, actions, rtg, window=args.window, stride=args.stride
    )
    print(f"  windows: {ret6.shape[0]}, state dim: {st6.shape[-1]}")

    encoder_6 = StateProj6(d_model=384)
    model_6, log_6 = train_dt(ret6, st6, act6, encoder_6, epochs=args.epochs)
    acc_6 = evaluate(model_6, ret6, st6, act6)
    time_6 = time.time() - t0
    print(f"  test acc: {acc_6:.4f}  ({time_6:.0f}s)")

    # === Experiment B: DINOv2 features ===
    print("\n--- B: DINOv2 frozen features (384-dim) ---")
    t0 = time.time()
    print("  Extracting DINOv2 features...")
    from nes_player.policy.dinov2_encoder import DINOv2Features
    dino = DINOv2Features()
    dino_feats = dino.extract(frames, batch_size=16)
    print(f"  extracted: {dino_feats.shape}  ({time.time()-t0:.0f}s)")

    # Save features for reuse
    np.save(f"runs/dino_feats_{args.game}.npy", dino_feats)

    ret_d, st_d, act_d = make_windows_dino(
        dino_feats, actions, rtg, window=args.window, stride=args.stride
    )
    print(f"  windows: {ret_d.shape[0]}, state dim: {st_d.shape[-1]}")

    encoder_d = StateProjDINO(d_model=384)
    model_d, log_d = train_dt(ret_d, st_d, act_d, encoder_d, epochs=args.epochs)
    acc_d = evaluate(model_d, ret_d, st_d, act_d)
    time_d = time.time() - t0
    print(f"  test acc: {acc_d:.4f}  ({time_d:.0f}s)")

    # === Comparison ===
    print("\n=== Comparison ===")
    print(f"6-dim features:    acc={acc_6:.4f}  ({time_6:.0f}s)")
    print(f"DINOv2 features:   acc={acc_d:.4f}  ({time_d:.0f}s)")
    print(f"DINOv2 - 6-dim:    {acc_d - acc_6:+.4f}")

    # Save results
    result = {
        "game": args.game,
        "n_windows_6": int(ret6.shape[0]),
        "n_windows_dino": int(ret_d.shape[0]),
        "acc_6dim": round(acc_6, 4),
        "acc_dinov2": round(acc_d, 4),
        "diff": round(acc_d - acc_6, 4),
        "time_6dim_s": round(time_6, 1),
        "time_dinov2_s": round(time_d, 1),
        "log_6dim": log_6,
        "log_dinov2": log_d,
    }
    out = Path(f"runs/dt_dinov2_{args.game}.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
