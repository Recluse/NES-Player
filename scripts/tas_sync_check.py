"""Find the frame where our replay diverges from an FCEUX reference trace.

The reference is captured by scripts/fceux_ram_trace.lua: 2048 bytes of RAM
plus one lag byte per frame. Here the same movie is replayed through our
adapter and RAM is compared frame by frame, printing the first divergence and
the reference's lag statistics.

Kept for a future FCEUX build. The one available crashed around frame 300 of
every movie, which is why the verified-prefix approach exists instead.

Usage:
  uv run python scripts/tas_sync_check.py <game-id> <movie.fm2> <trace.bin> \
      [--integrations DIR] [--boot-offset 2]
"""

import argparse

import numpy as np

from nes_player.emulator.stable_retro import StableRetroAdapter
from nes_player.tas.fm2 import parse_fm2

RAM = 2048
REC = RAM + 1

ap = argparse.ArgumentParser()
ap.add_argument("game")
ap.add_argument("movie")
ap.add_argument("trace")
ap.add_argument("--integrations", default=None)
ap.add_argument("--boot-offset", type=int, default=2)
args = ap.parse_args()

raw = np.fromfile(args.trace, dtype=np.uint8)
n_ref = len(raw) // REC
ref = raw[: n_ref * REC].reshape(n_ref, REC)
ref_ram, ref_lag = ref[:, :RAM], ref[:, RAM]
print(f"reference: {n_ref} frames, lag frames: {int(ref_lag.sum())}")

movie = parse_fm2(args.movie)
em = StableRetroAdapter(args.game, integration_dir=args.integrations,
                        players=max(1, movie.players))
em.reset()
for _ in range(args.boot_offset):
    em.step_buttons([frozenset()] * max(1, movie.players))

first_bad = None
for i, ports in enumerate(movie.inputs[:n_ref]):
    em.step_buttons(list(ports))
    ram = np.asarray(em._env.get_ram(), dtype=np.uint8)
    if not np.array_equal(ram, ref_ram[i]):
        first_bad = i
        diff = int((ram != ref_ram[i]).sum())
        print(f"DESYNC at frame {i} ({i / 60.1:.1f}s): {diff}/2048 bytes differ")
        break
em.close()
if first_bad is None:
    print(f"IN SYNC for all {min(n_ref, len(movie.inputs))} compared frames")
