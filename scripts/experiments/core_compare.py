"""Compare emulation cores by how long they hold third-party TAS movies.

The movies were recorded in FCEUX and BizHawk while we play on fceumm, which is
where the desynchronisation comes from. Cores plug in (see emulator/cores.py),
so the question is settled by measurement.

One core per process — libretro is loaded into the address space — so the script
re-runs itself in a subprocess for each core.

"Still playing" is judged by the share of near-motionless frames: a run stuck on
the title screen approaches 100%, a live game sits in the tens of percent. This
is a rough proxy and it misfires on games with small sprites over a static
background; read the numbers with that in mind.

Usage: uv run python scripts/experiments/core_compare.py [--frames 6000]
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parents[2]
CACHE = ROOT / "rnd" / "tas" / "sample"
CORES = ["fceumm", "nestopia", "quicknes"]   # mesen segfaults the process


def temp_integration(rom: Path, tmp: Path) -> str:
    game_id = "CoreProbe-Nes-v0"
    d = tmp / game_id
    d.mkdir(parents=True, exist_ok=True)
    data = rom.read_bytes()
    body = data[16:] if data[:4] == b"NES\x1a" else data
    shutil.copy(rom, d / "rom.nes")
    (d / "rom.sha").write_text(hashlib.sha1(body).hexdigest() + "\n")
    (d / "data.json").write_text('{"info": {}}\n')
    (d / "metadata.json").write_text("{}\n")
    return game_id


def run_core(core: str, frames: int) -> None:
    """Run every movie on one core; called inside the subprocess."""
    from nes_player.tas.fm2 import parse_fm2
    from nes_player.tas.replay import replay_frames

    index = json.loads((ROOT / "rnd" / "rom_index.json").read_text())
    for path in sorted(CACHE.glob("*.fm2")):
        movie = parse_fm2(path)
        md5 = movie.rom_md5
        entry = index.get(md5.hex()) if md5 else None
        if entry is None or "::" in entry:
            continue
        rom = Path(entry)
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            game_id = temp_integration(rom, tmp)
            prev, static, n = None, 0, 0
            try:
                for obs, _ in replay_frames(game_id, path, integration_dir=tmp,
                                            max_frames=frames, core=core):
                    g = cv2.cvtColor(obs.frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.int16)
                    if prev is not None and float(np.abs(g - prev).mean()) < 0.3:
                        static += 1
                    prev, n = g, n + 1
            except Exception as e:  # noqa: BLE001 — a core may fail to load a ROM
                print(json.dumps({"core": core, "movie": path.stem,
                                  "error": str(e)[:50]}), flush=True)
                continue
            print(json.dumps({"core": core, "movie": path.stem, "rom": rom.name[:30],
                              "static": round(static / max(n, 1), 2), "frames": n},
                             ensure_ascii=False), flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--frames", type=int, default=6000)
ap.add_argument("--core", default=None, help="internal: run a single core")
a = ap.parse_args()

if a.core:
    run_core(a.core, a.frames)
    raise SystemExit(0)

rows: dict[str, dict[str, float]] = {}
for core in CORES:
    out = subprocess.run([sys.executable, __file__, "--core", core,
                          "--frames", str(a.frames)],
                         capture_output=True, text=True, check=False).stdout
    for line in out.splitlines():
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        rows.setdefault(r["movie"], {})[core] = r.get("static", -1.0)
    print(f"- {core} done", flush=True)

print("\nshare of motionless frames (lower means the game is running):")
print("movie      " + "".join(f"{c:>11}" for c in CORES))
wins = dict.fromkeys(CORES, 0)
for movie, per_core in sorted(rows.items()):
    line = f"{movie:<11}"
    best = min((v for v in per_core.values() if v >= 0), default=None)
    for c in CORES:
        v = per_core.get(c, -1.0)
        mark = "*" if best is not None and v == best else " "
        line += f"{v:>10.0%}{mark}" if v >= 0 else f"{'error':>11}"
        if best is not None and v == best:
            wins[c] += 1
    print(line)
print("\nwins per core:", json.dumps(wins))
