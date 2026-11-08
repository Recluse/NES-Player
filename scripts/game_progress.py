#!/usr/bin/env python3
"""Measure game progress as levels/rooms completed, not just pixels.

For Double Dragon II: count screen transitions (room changes) as progress.
For other games: adapt the metric to their structure.

    uv run python scripts/game_progress.py --episode datasets/dagger_dd2_all/xxx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def measure_dd2_progress(frames: np.ndarray) -> dict:
    """DD II progress: screen transitions = room changes.

    A beat-em-up scrolls horizontally. When the camera reaches a door,
    there's a screen transition (large frame diff, then new scene).
    Each transition = one room step toward the mission end.
    """
    T = len(frames)
    diffs = []
    for i in range(1, T):
        diff = np.abs(frames[i].astype(float) - frames[i-1].astype(float)).mean()
        diffs.append(diff)

    diffs = np.array(diffs)
    # Screen transitions: very large frame changes
    transitions = np.where(diffs > 50)[0]
    # Avoid counting rapid flicker as transitions
    filtered = []
    last = -100
    for t in transitions:
        if t - last > 30:  # at least 30 frames apart
            filtered.append(int(t))
            last = t

    # Position proxy: cumulative scroll
    pos = np.zeros(T)
    for i in range(1, T):
        pos[i] = pos[i-1] + max(0, diffs[i-1] - 2)

    return {
        "frames": int(T),
        "screen_transitions": len(filtered),
        "transition_frames": filtered[:20],
        "max_position": float(pos.max()),
        "final_position": float(pos[-1]),
        "rooms_estimated": len(filtered) + 1,
    }


def measure_generic_progress(frames: np.ndarray) -> dict:
    """Generic progress: frame brightness changes as movement proxy."""
    T = len(frames)
    diffs = []
    for i in range(1, T):
        diff = np.abs(frames[i].astype(float) - frames[i-1].astype(float)).mean()
        diffs.append(diff)
    diffs = np.array(diffs)
    transitions = np.where(diffs > 50)[0]
    filtered = []
    last = -100
    for t in transitions:
        if t - last > 30:
            filtered.append(int(t))
            last = t
    return {
        "frames": int(T),
        "screen_transitions": len(filtered),
        "max_frame_diff": float(diffs.max()) if len(diffs) > 0 else 0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episode", required=True)
    p.add_argument("--game", default="DoubleDragonII-Nes-v0")
    args = p.parse_args()

    ep = Path(args.episode)
    import zarr
    z = zarr.open(str(ep / "episode.zarr"), mode="r")
    frames = z["frames"][:]

    if "DoubleDragon" in args.game:
        result = measure_dd2_progress(frames)
    else:
        result = measure_generic_progress(frames)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
