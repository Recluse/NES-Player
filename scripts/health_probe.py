#!/usr/bin/env python3
"""Where are Battletoads' hearts? Half a heart per hit, so says the owner.

A hit removes half a heart (Recluse, 2026-10-08). That is a manual-grade
fact about the game's damage granularity, and it tells us what to look for
in RAM: a small counter (2..12) stepping by 1 per hit, not a 72-point wall.

This script records candidate bytes through a stretch of recorded play and
prints the ones that look like health. It does not promote them — a heart
byte is `scan_confirmed` only when a controlled hit moves it and a controlled
non-hit does not.

    uv run python scripts/health_probe.py \
        --episode datasets/explore_battletoads/Battletoads-Nes-v0_ep003 \
        --state runs/knowledge/bt_ep003_f700.state

Writes runs/knowledge/health_<game>.json (candidates, not facts).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nes_player.data.reader import Episode  # noqa: E402
from nes_player.emulator.stable_retro import StableRetroAdapter  # noqa: E402
from nes_player.perception.sprites import buttons_from_mask  # noqa: E402

MIN_V, MAX_V = 2, 24          # hearts / half-hearts live here
MAX_STEP = 3                  # one hit is 1 half-heart; allow a little more


def red_pixels(frame_rgb: np.ndarray, band: int = 48) -> int:
    hud = frame_rgb[:band].astype(np.int16)
    r, g, b = hud[..., 0], hud[..., 1], hud[..., 2]
    return int(((r > 120) & (r > g + 40) & (r > b + 40)).sum())


def probe(game: str, integrations: str | None, episode: Path | None,
          state: Path | None, frames: int) -> dict:
    acts = np.asarray(Episode(episode).actions).reshape(-1) if episode else None
    em = StableRetroAdapter(game, integration_dir=integrations)
    em.reset()
    if state:
        em.load_state(Path(state).read_bytes())

    start = np.array(em._env.get_ram(), dtype=np.int16)
    snaps = [(0, start.copy(), 0)]
    skip = 0
    if episode and state:
        # caller is expected to pass --start-frame matching the state
        pass
    for i in range(frames):
        if acts is not None:
            mask = int(acts[(i + skip) % len(acts)])
            obs = em.step_buttons([buttons_from_mask(mask)])
        else:
            obs = em.step_buttons([frozenset({"B"}) if i % 2 == 0 else frozenset()])
        if i % 10 == 0:
            snaps.append((i + 1, np.array(em._env.get_ram(), dtype=np.int16),
                          red_pixels(obs.frame_rgb)))
    em.close()

    series = np.stack([s[1] for s in snaps])
    reds = [s[2] for s in snaps]
    cands = []
    for a in range(series.shape[1]):
        seq = series[:, a]
        v0, v1 = int(seq[0]), int(seq[-1])
        if not (MIN_V <= v0 <= MAX_V):
            continue
        diffs = np.diff(seq)
        if diffs.min() < -MAX_STEP or diffs.max() > MAX_STEP:
            continue
        drops = int((diffs < 0).sum())
        rises = int((diffs > 0).sum())
        if drops >= 1 and rises <= 2 and v0 != v1:
            cands.append({
                "addr": a,
                "hex": hex(a),
                "start": v0,
                "end": v1,
                "drops": drops,
                "rises": rises,
                "seq": [int(x) for x in seq[:20]],
            })

    return {
        "game": game,
        "episode": str(episode) if episode else None,
        "state": str(state) if state else None,
        "frames": frames,
        "owner_fact": "one hit removes half a heart (Recluse, 2026-10-08)",
        "candidates": cands,
        "red_pixel_trace": reds,
        "status": "proposed",
        "note": (
            "candidates are small counters that stepped down during this "
            "stretch; promotion needs a controlled hit vs no-hit pair"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="Battletoads-Nes-v0")
    ap.add_argument("--integrations", default="integrations")
    ap.add_argument("--episode", type=Path)
    ap.add_argument("--state", type=Path)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    result = probe(args.game, args.integrations, args.episode, args.state,
                   args.frames)
    out = args.out or (ROOT / "runs/knowledge" / f"health_{args.game}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "game": result["game"],
        "n_candidates": len(result["candidates"]),
        "candidates": result["candidates"][:12],
        "red_min": min(result["red_pixel_trace"]) if result["red_pixel_trace"] else None,
        "red_max": max(result["red_pixel_trace"]) if result["red_pixel_trace"] else None,
        "note": result["note"],
    }, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
