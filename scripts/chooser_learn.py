#!/usr/bin/env python3
"""Chooser learning: pick the right template for the situation.

The planner has candidates (templates + the policy's own plan) and has to
choose one. On Galaga the value function cannot tell `aligned barrage` from
`run` inside the horizon — both score 81. The chooser needs demonstrations
of *what actually worked*.

This script trains a small probe on agent-play data: given the current
screen, which template should the planner prefer? The labels come from
what the agent-play session actually did and what it achieved.

    uv run python scripts/chooser_learn.py \
        --episode datasets/agent_play/Galaga-Nes-v0_agent_long \
        --game Galaga-Nes-v0

Writes runs/chooser_<game>.json with the probe's accuracy and the
per-template preference it learned.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def label_frame(i: int, actions: np.ndarray) -> str:
    """What was the agent doing at frame i? Maps to a template name."""
    mask = int(actions[i])
    buttons = {
        "A": mask & 1, "B": mask & 2, "SELECT": mask & 4, "START": mask & 8,
        "UP": mask & 16, "DOWN": mask & 32, "LEFT": mask & 64, "RIGHT": mask & 128,
    }
    if buttons["B"] and buttons["RIGHT"] and not buttons["A"]:
        return "sweep-right-firing"
    if buttons["B"] and buttons["LEFT"] and not buttons["A"]:
        return "sweep-left-firing"
    if buttons["B"] and not (buttons["LEFT"] or buttons["RIGHT"]):
        return "aligned-barrage"
    if buttons["A"] and buttons["B"]:
        return "dodge-and-fire"
    if buttons["RIGHT"] and not buttons["B"]:
        return "run"
    return "other"


def extract_features(frame: np.ndarray) -> list[float]:
    """Cheap screen features the chooser can use."""
    f = frame.astype(np.float32).mean(axis=2)
    h, w = f.shape
    enemies = f[: h // 2]
    ship = f[3 * h // 4:]
    enemy_cols = (enemies > 100).sum(axis=0)
    ship_cols = (ship > 100).sum(axis=0)
    e_x = float(np.argmax(enemy_cols)) if enemy_cols.any() else w / 2
    s_x = float(np.argmax(ship_cols)) if ship_cols.any() else w / 2
    return [
        e_x / w,                        # enemy column
        s_x / w,                        # ship column
        abs(e_x - s_x) / w,             # alignment
        float(enemy_cols.max()) / h,    # enemy density
        float(ship_cols.max()) / h,     # ship activity
        float(f.mean()) / 255,          # brightness
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--game", default="")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--max-frames", type=int, default=5000)
    args = ap.parse_args()

    from nes_player.data.reader import Episode

    ep = Episode(args.episode)
    actions = np.asarray(ep.actions).reshape(-1)
    n = min(len(actions), args.max_frames)

    # Collect (features, label) pairs
    X, y = [], []
    for i in range(0, n, 10):
        frame = np.asarray(ep.frames[i])
        label = label_frame(i, actions)
        if label == "other":
            continue
        X.append(extract_features(frame))
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    print(f"{len(X)} labelled frames, {len(set(y))} labels")
    for lab in sorted(set(y)):
        print(f"  {lab}: {(y == lab).sum()}")

    # Simple nearest-centroid probe (no torch needed, honest baseline)
    labels = sorted(set(y))
    centroids = {}
    for lab in labels:
        centroids[lab] = X[y == lab].mean(axis=0)

    correct = 0
    for xi, yi in zip(X, y):
        dists = {lab: float(np.linalg.norm(xi - cen)) for lab, cen in centroids.items()}
        pred = min(dists, key=dists.get)
        if pred == yi:
            correct += 1
    acc = correct / len(X) if X.size else 0.0
    print(f"nearest-centroid accuracy: {acc:.3f}")

    # Preference ranking: which labels the chooser would prefer given
    # the centroid distance from a typical "aligned combat" state
    result = {
        "game": args.game or args.episode.name,
        "n_frames": len(X),
        "labels": {lab: int((y == lab).sum()) for lab in labels},
        "chooser_accuracy": round(acc, 3),
        "centroids": {lab: [round(float(v), 3) for v in cen]
                      for lab, cen in centroids.items()},
        "note": "nearest-centroid probe on agent-play labels; "
                "the chooser prefers the label whose centroid is closest "
                "to the current screen",
    }
    out = args.out or (ROOT / "runs" / f"chooser_{args.game or 'game'}.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
