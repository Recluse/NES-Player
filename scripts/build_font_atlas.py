"""Build the NES font atlas: 8×8 glyph to character.

Digits the agent learns by itself from counter dynamics. Letters have no such
dynamics — there is nothing to derive a "P" from — so they are supplied as a
prior, the same way a person arrives at a game already knowing how to read.

The bootstrap is straightforward: take several screens whose text is known and
align the non-empty cells of a line onto the characters of that text. Spaces
are empty cells and are skipped.

Usage: uv run python scripts/build_font_atlas.py [--check]
"""

import argparse
import json
from pathlib import Path

from nes_player.emulator.stable_retro import StableRetroAdapter
from nes_player.perception.text import frame_cells

ROOT = Path(__file__).parents[1]
OUT = ROOT / "assets" / "nes_font.json"

# A source is either a live game boot or a frame from a recorded episode: the
# screen carrying the text is sometimes only reachable after START pulses.
SOURCES = [
    {"game": "SuperMarioBros-Nes-v0", "frame": 120, "rows": {
        1: "MARIO WORLD TIME",
        17: "1 PLAYER GAME",
        19: "2 PLAYER GAME",
    }},
    {"episode": "datasets/explore_battletoads/Battletoads-Nes-v0_ep001", "frame": 1200,
     "rows": {
         17: "PRESS START TO PLAY",
         22: "COPYRIGHT 1991 RARE LTD.",
         24: "LICENSED TO TRADEWEST",
         25: "BY RARE COIN-IT,INC.",
     }},
]


def grab(src: dict):
    if "episode" in src:
        from nes_player.data.reader import Episode

        return Episode(ROOT / src["episode"]).frames[src["frame"]]
    em = StableRetroAdapter(src["game"], integration_dir=src.get("integrations"))
    obs = em.reset()
    for _ in range(src["frame"]):
        obs = em.step_buttons([frozenset()])
    em.close()
    return obs.frame_rgb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only show the alignment")
    args = ap.parse_args()

    atlas: dict[str, list[str]] = {}
    for src in SOURCES:
        frame = grab(src)
        name = src.get("game") or Path(src["episode"]).name
        cells = frame_cells(frame)
        for row, text in src["rows"].items():
            got = [(c, cells[(row, c)]) for (r, c) in sorted(cells) if r == row]
            want = [ch for ch in text if ch != " "]
            status = "OK" if len(got) == len(want) else "MISMATCH"
            print(f"{name} row {row}: {len(got)} cells, {len(want)} characters — {status}"
                  f"  «{text}»")
            if len(got) != len(want):
                continue
            for (_, sig), ch in zip(got, want, strict=True):
                variants = atlas.setdefault(ch, [])
                if str(sig) not in variants:
                    variants.append(str(sig))
    if args.check:
        return
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(atlas, indent=1, sort_keys=True, ensure_ascii=False))
    print(f"characters in the atlas: {len(atlas)} -> {OUT}")


if __name__ == "__main__":
    main()
