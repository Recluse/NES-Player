"""How many TASVideos movies are actually usable as data.

Emulator recordings already contain exact button presses, so no inverse
dynamics is needed, and replay runs thousands of times faster than video. The
question is the yield: is it .fm2 (we do not parse BizHawk's bk2), is there a
ROM with the matching MD5, and does the movie stay in sync on our core.

Desync is detected without a reference: under a TAS the screen changes
constantly, so if the core drifts the hero dies and the screen sits still on a
game over. This proxy is imperfect in both directions — see docs/experiments.md
before quoting a yield figure from it.

Usage: uv run python scripts/experiments/tas_yield.py [--sample 12]
"""

import argparse
import io
import json
import os
import subprocess
import zipfile
from pathlib import Path

from nes_player.tas.fm2 import parse_fm2

ROOT = Path(__file__).parents[2]
CACHE = ROOT / "rnd" / "tas" / "sample"
API = "https://tasvideos.org/api/v1/publications?pageSize=100&currentPage={}"


def api_page(page: int) -> list:
    out = subprocess.run(["curl", "-s", "--max-time", "60", "-H", "accept: application/json",
                          API.format(page)], capture_output=True, text=True, check=False).stdout
    return json.loads(out) if out.strip().startswith("[") else []


def download(pub_id: int) -> Path | None:
    """A publication's movie: a zip with an .fm2 inside."""
    dest = CACHE / f"{pub_id}.fm2"
    if dest.exists():
        return dest
    raw = subprocess.run(
        ["curl", "-sL", "--max-time", "120", f"https://tasvideos.org/{pub_id}M?handler=Download"],
        capture_output=True, check=False).stdout
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile:
        return None
    names = [n for n in zf.namelist() if n.lower().endswith(".fm2")]
    if not names:
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(zf.read(names[0]))
    return dest


def rom_for(movie) -> Path | None:
    """Find a ROM with the MD5 the movie asks for."""
    import hashlib

    want = movie.rom_md5
    if want is None:
        return None
    for p in ROMSET.glob("*.nes"):
        data = p.read_bytes()
        body = data[16:] if data[:4] == b"NES\x1a" else data
        if hashlib.md5(body).digest() == want:
            return p
    return None


ap = argparse.ArgumentParser()
ap.add_argument("--sample", type=int, default=12)
ap.add_argument("--romset", default=os.environ.get("NES_ROMSET"),
                help="directory of ROM files; defaults to $NES_ROMSET")
a = ap.parse_args()
if not a.romset:
    raise SystemExit("pass --romset, or set NES_ROMSET")
ROMSET = Path(a.romset)

pubs = []
for page in range(1, 12):
    # .fm2 only: .fcm is the old FCEU format and needs converting, .bk2 is BizHawk
    pubs += [p for p in api_page(page)
             if p.get("systemCode") == "NES"
             and (p.get("movieFileName") or "").lower().endswith(".fm2")]
    if len(pubs) >= a.sample:
        break

stats = {"tried": 0, "fm2": 0, "rom_found": 0}
for pub in pubs[: a.sample]:
    if stats["tried"] >= a.sample:
        break
    stats["tried"] += 1
    path = download(pub["id"])
    if path is None:
        continue
    stats["fm2"] += 1
    try:
        movie = parse_fm2(path)
    except Exception as e:  # noqa: BLE001 — the share of broken ones is the point
        print(json.dumps({"id": pub["id"], "parse_error": str(e)[:60]}), flush=True)
        continue
    rom = rom_for(movie)
    if rom:
        stats["rom_found"] += 1
    print(json.dumps({"id": pub["id"], "game": pub.get("title", "")[:48],
                      "frames": len(movie.inputs), "players": movie.players,
                      "rom": rom.name if rom else None}, ensure_ascii=False), flush=True)

print(json.dumps({"summary": stats}))
print("(the in-sync share is measured separately, by replaying the pairs found)")
