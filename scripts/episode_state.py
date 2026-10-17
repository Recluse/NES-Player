#!/usr/bin/env python3
"""Save an emulator state at a given frame of a recorded episode.

`object_tables.py` and `hero_tiles.py` need a savestate with a fight on
screen. Explore episodes already contain fights; this replays one up to a
frame and writes the state so those scans can fire at it.

    uv run python scripts/episode_state.py \
        --episode datasets/explore_battletoads/Battletoads-Nes-v0_ep003 \
        --frame 680 --out runs/knowledge/battletoads_ep003_f680.state

Pick a frame where the player is fighting: --auto-busy scores windows by
button activity and takes the densest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nes_player.data.reader import Episode  # noqa: E402
from nes_player.perception.sprites import (  # noqa: E402
    _open_matching_env,
    buttons_from_mask,
)


def busiest_frame(acts: np.ndarray, window: int = 60) -> int:
    """Frame at the start of the densest button window."""
    rates = []
    for i in range(0, max(1, len(acts) - window), window // 2):
        chunk = acts[i:i + window]
        rates.append((float(np.mean(chunk > 0)), i))
    rates.sort(reverse=True)
    return rates[0][1] + window // 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--frame", type=int, default=-1)
    ap.add_argument("--auto-busy", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    ep = Episode(args.episode)
    acts = np.asarray(ep.actions).reshape(-1)
    frame = args.frame
    if frame < 0 or args.auto_busy:
        frame = busiest_frame(acts)
    if frame >= len(acts):
        print(f"frame {frame} beyond episode ({len(acts)})", file=sys.stderr)
        return 2

    env, started_from = _open_matching_env(ep, int(acts[0]))
    for i in range(frame):
        env.step_buttons([buttons_from_mask(int(acts[i]))])
    state = env.save_state()
    env.close()

    out = args.out or (ROOT / "runs/knowledge"
                       / f"{args.episode.name}_f{frame}.state")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(state)
    print(f"game={ep.metadata.get('game')} frame={frame} "
          f"from={started_from} -> {out} ({len(state)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
