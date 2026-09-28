"""Turn a search's button recording into training episodes, keeping only what
moved the game on.

A Go-Explore path is mostly noise: random button runs, of which a few happened
to lead somewhere. Cloned whole, the noise is what gets learned — the Mario
Go-Explore paths taught a clone to press nothing useful. So replay the path,
score every frame by progress (position, room, damage to a boss), and keep
only the stretches that are followed by a new best within --window frames.
Each kept stretch becomes its own episode, so frame stacks never straddle a
cut.

    uv run python scripts/experiments/search_to_episodes.py \
        runs/vice/geboss3/boss.json --pos runs/vice/pos_rooms.json \
        --boss 644 --out datasets/vice_search --name boss2-2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def useful_spans(progress: np.ndarray, window: int, min_len: int):
    """Frames followed by a new best within `window`, as (start, end) spans."""
    best = np.maximum.accumulate(progress)
    new = np.zeros(len(progress), bool)
    new[1:] = best[1:] > best[:-1]
    keep = np.zeros(len(progress), bool)
    for t in np.flatnonzero(new):
        keep[max(0, t - window):t + 1] = True
    spans, start = [], None
    for i, k in enumerate(keep):
        if k and start is None:
            start = i
        if not k and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(keep)))
    return [(a, b) for a, b in spans if b - a >= min_len]


def main() -> int:
    import oracle_mpc as m

    from nes_player.data.writer import EpisodeWriter
    from nes_player.emulator.stable_retro import StableRetroAdapter

    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--game", default="ViceProjectDoom-Nes-v0")
    ap.add_argument("--pos", required=True)
    ap.add_argument("--boss", type=int, default=0)
    ap.add_argument("--boss-px", type=int, default=100,
                    help="progress per unit of boss health taken")
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--min-len", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    rec = json.loads(Path(args.recording).read_text())
    m.SCAN_POS[args.game] = json.loads(Path(args.pos).read_text())[
        "position_bytes_consistent"]
    env = StableRetroAdapter(args.game, include_debug=True, state="default",
                             integration_dir=None)
    env.reset(seed=0)
    env.load_state(Path(rec["load_state"]).read_bytes())
    boss0 = int(env._env.get_ram()[args.boss]) if args.boss else 0
    obs, acts, prog = [], [], []
    for b in rec["actions"]:
        pressed = frozenset(b)
        o = env.step_buttons([pressed])
        ram = env._env.get_ram()
        p = m.game_pos(env, args.game)
        if args.boss:
            p += args.boss_px * (boss0 - int(ram[args.boss]))
        obs.append(o)
        acts.append(pressed)
        prog.append(p)
    spans = useful_spans(np.array(prog), args.window, args.min_len)
    kept = 0
    for i, (a, b) in enumerate(spans):
        ep = Path(args.out) / f"{args.game}_{args.name}_{i:02d}"
        w = EpisodeWriter(out_dir=ep, metadata={
            "game": args.game, "source": "go-explore",
            "sample_rate": env.sample_rate, "recording": args.recording,
            "span": [a, b],
            "note": "a search's path, cut to the stretches that led to a new "
                    "best within the window; the searcher saw RAM and "
                    "rewound, the learner does not"})
        for o, pressed in zip(obs[a:b], acts[a:b], strict=True):
            w.append(o, (pressed,))
        w.close()
        kept += b - a
    print(f"{len(acts)} frames, kept {kept} in {len(spans)} episodes "
          f"({kept / max(1, len(acts)):.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
