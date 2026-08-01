"""What share of TAS movies replays on our core without desynchronising.

This is the decisive number for learning from emulator recordings: the buttons
in them are exact, but our core lags on different frames than FCEUX does and
some movies drift.

Desync is detected without a reference. A TAS is dense play with no idling, so
if the core drifts the hero dies and the screen sits still on a game over. We
measure the longest run of near-motionless frames in the second half.

Usage: uv run python scripts/experiments/tas_sync_rate.py [--frames 12000]
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from nes_player.tas.fm2 import parse_fm2
from nes_player.tas.replay import replay_frames

ROOT = Path(__file__).parents[2]
CACHE = ROOT / "rnd" / "tas" / "sample"
STATIC_RUN = 240   # four still seconds in a row is almost certainly a death


def find_rom(movie, romset: Path) -> Path | None:
    want = movie.rom_md5
    if want is None:
        return None
    for p in romset.glob("*.nes"):
        data = p.read_bytes()
        body = data[16:] if data[:4] == b"NES\x1a" else data
        if hashlib.md5(body).digest() == want:
            return p
    return None


def make_integration(rom: Path, tmp: Path) -> str:
    game_id = "TasProbe-Nes-v0"
    dest = tmp / game_id
    dest.mkdir(parents=True, exist_ok=True)
    data = rom.read_bytes()
    body = data[16:] if data[:4] == b"NES\x1a" else data
    shutil.copy(rom, dest / "rom.nes")
    (dest / "rom.sha").write_text(hashlib.sha1(body).hexdigest() + "\n")
    (dest / "data.json").write_text('{"info": {}}\n')
    (dest / "metadata.json").write_text("{}\n")
    return game_id


ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=12000)
ap.add_argument("--romset", default=os.environ.get("NES_ROMSET"),
                help="directory of ROM files; defaults to $NES_ROMSET")
a = ap.parse_args()
if not a.romset:
    raise SystemExit("pass --romset, or set NES_ROMSET")
romset = Path(a.romset)

ok = bad = skipped = 0
for path in sorted(CACHE.glob("*.fm2")):
    movie = parse_fm2(path)
    rom = find_rom(movie, romset)
    if rom is None:
        skipped += 1
        continue
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        game_id = make_integration(rom, tmp)
        prev, longest, streak, n = None, 0, 0, 0
        try:
            for obs, _pressed in replay_frames(game_id, path, integration_dir=tmp,
                                               max_frames=min(a.frames, len(movie.inputs))):
                g = cv2.cvtColor(obs.frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.int16)
                if prev is not None and float(np.abs(g - prev).mean()) < 0.5:
                    streak += 1
                    longest = max(longest, streak)
                else:
                    streak = 0
                prev = g
                n += 1
        except Exception as e:  # noqa: BLE001 — the failure rate is what we measure
            print(json.dumps({"movie": path.name, "error": str(e)[:70]}), flush=True)
            bad += 1
            continue
    synced = longest < STATIC_RUN
    ok += synced
    bad += not synced
    print(json.dumps({"movie": path.name, "rom": rom.name, "frames": n,
                      "longest_static": longest, "looks_synced": synced},
                     ensure_ascii=False), flush=True)

print(json.dumps({"synced": ok, "desynced_or_failed": bad, "no_rom": skipped}))
