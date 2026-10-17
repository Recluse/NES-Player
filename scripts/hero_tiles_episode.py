#!/usr/bin/env python3
"""Hero tiles from a recorded episode, when a live scan cannot reach the fight.

Battletoads' intro is ~5000 frames of title and story; `hero_tiles.py` boots
with `begin_any` and shoots at whatever is on screen, which for that game is
not a fight. The explore episodes already contain real play with button
bitmasks. Every sprite in the table carries a tile index, and the player's
tiles are the ones whose screen x follows LEFT/RIGHT — the same rule as the
live scan, applied offline.

    uv run python scripts/hero_tiles_episode.py \
        --episode datasets/explore_battletoads/Battletoads-Nes-v0_ep001

Writes runs/knowledge/hero_<game>.json in the same shape as hero_tiles.py.
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
from nes_player.emulator.controller import BUTTONS  # noqa: E402
from nes_player.perception.sprites import episode_sprites  # noqa: E402

MOVE = ("LEFT", "RIGHT")
HIDDEN_Y = 200      # sprites below this are parked off-screen
MIN_SAMPLES = 30
MIN_DX = 2.0        # pixels of mean dx per frame while the button is held
MIN_SCORE = 3.0     # held-minus-idle; lower than this is noise / camera drift


def hero_tiles_from_episode(ep: Episode) -> dict:
    sp = np.asarray(episode_sprites(ep))
    acts = np.asarray(ep.actions).reshape(-1)
    bit = {name: 1 << i for i, name in enumerate(BUTTONS)}

    # score[tile][button] = dx_held - dx_idle
    scores: dict[int, dict[str, float]] = {}
    detail = []
    tiles = np.unique(sp[:, :, 1])
    for tile in tiles:
        xs, bits = [], []
        for t in range(0, len(sp), 3):
            rows = sp[t][sp[t][:, 1] == tile]
            rows = rows[rows[:, 0] < HIDDEN_Y]
            if len(rows) == 0:
                continue
            xs.append(float(rows[:, 2].mean()))
            bits.append(int(acts[t]))
        if len(xs) < MIN_SAMPLES:
            continue
        xs_arr = np.asarray(xs)
        bits_arr = np.asarray(bits)
        dx = np.diff(xs_arr)
        for name in MOVE:
            held = (bits_arr & bit[name]) > 0
            h = held[1:]
            if h.sum() < 8 or (~h).sum() < 8:
                continue
            m_held = float(np.mean(dx[h]))
            m_idle = float(np.mean(dx[~h]))
            if abs(m_held) > 64:   # screen wrap
                continue
            score = m_held - m_idle
            if score > MIN_DX:
                scores.setdefault(int(tile), {})[name] = score
                detail.append({
                    "tile": int(tile), "button": name,
                    "dx_held": round(m_held, 2), "dx_idle": round(m_idle, 2),
                    "score": round(score, 2), "n_held": int(h.sum()),
                })

    # The player answers the pad in *both* directions. A tile that only
    # drifts right when RIGHT is held is scenery scrolling past.
    hero = sorted(
        t for t, by_btn in scores.items()
        if set(by_btn) >= set(MOVE)
        and min(by_btn.values()) >= MIN_SCORE
    )
    # If nothing answers both ways (a one-way vehicle stage), fall back to
    # the strongest single-direction responders, capped at eight tiles.
    if not hero:
        ranked = sorted(
            ((max(by_btn.values()), t) for t, by_btn in scores.items()
             if max(by_btn.values()) >= MIN_SCORE),
            reverse=True,
        )
        hero = sorted(t for _, t in ranked[:8])

    return {
        "game": ep.metadata.get("game") if isinstance(ep.metadata, dict) else None,
        "episode": str(ep.path),
        "tiles": hero,
        "votes": {str(t): {b: round(s, 2) for b, s in by_btn.items()}
                  for t, by_btn in scores.items() if t in set(hero)},
        "detail": detail,
        "probes": len(detail),
        "method": (
            "episode_sprites + dx vs LEFT/RIGHT; hero answers both "
            "directions with score >= "
            f"{MIN_SCORE} (hero_tiles.py rule, offline)"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    ep = Episode(args.episode)
    result = hero_tiles_from_episode(ep)
    # game id from the directory name when metadata has none
    if not result.get("game"):
        result["game"] = args.episode.name.split("_ep")[0]

    out = args.out or (ROOT / "runs/knowledge" / f"hero_{result['game']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("game", "tiles", "probes", "method")},
                     indent=2))
    print(f"-> {out}")
    return 0 if result["tiles"] else 1


if __name__ == "__main__":
    sys.exit(main())
