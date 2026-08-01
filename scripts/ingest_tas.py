"""Bulk ingestion of TAS movies into episodes: download, replay, truncate.

Emulator recordings already contain exact buttons, so no inverse dynamics is
needed. The problem is that our core lags on different frames than FCEUX does
and some movies drift. The answer is to keep the VERIFIED PREFIX: replay, find
where it breaks, keep the beginning.

The break detector comes from the game itself and is pleasingly honest: **in a
TAS the player does not lose lives**. We read the HUD counters off the screen —
digits learned without labels — and cut as soon as a life-like counter goes
down. Two fallbacks cover the rest: the screen freezing for a long time, or
returning to the starting frame.

Usage:
  uv run python scripts/ingest_tas.py --limit 20 --out datasets/tas_pack
"""

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

from nes_player.data.writer import EpisodeWriter
from nes_player.perception.text import HudReader
from nes_player.tas.fm2 import parse_fm2
from nes_player.tas.replay import replay_frames

ROOT = Path(__file__).parents[1]
CACHE = ROOT / "rnd" / "tas" / "pack"
STATIC_LIMIT = 900   # 15 seconds of total stillness means we are stuck
WARMUP = 600         # frames allowed for menus and intros, where counters jump


def api(page: int) -> list:
    out = subprocess.run(
        ["curl", "-s", "--max-time", "60", "-H", "accept: application/json",
         f"https://tasvideos.org/api/v1/publications?pageSize=100&currentPage={page}"],
        capture_output=True, text=True, check=False).stdout
    return json.loads(out) if out.strip().startswith("[") else []


def download(pub_id: int) -> Path | None:
    dest = CACHE / f"{pub_id}.fm2"
    if dest.exists():
        return dest
    raw = subprocess.run(["curl", "-sL", "--max-time", "120",
                          f"https://tasvideos.org/{pub_id}M?handler=Download"],
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


def resolve_rom(entry: str, tmp: Path) -> Path | None:
    """Path to a ROM from the index; "archive::name" is extracted on demand."""
    if "::" not in entry:
        return Path(entry)
    archive, inner = entry.split("::", 1)
    dest = tmp / "rom_src"
    dest.mkdir(parents=True, exist_ok=True)
    cmd = (["7zz", "x", "-y", f"-o{dest}", archive, inner]
           if archive.endswith(".7z")
           else ["unzip", "-o", "-q", archive, inner, "-d", str(dest)])
    if subprocess.run(cmd, capture_output=True, check=False).returncode != 0:
        return None
    got = dest / inner
    return got if got.exists() else next(iter(dest.rglob("*.nes")), None)


def temp_integration(rom: Path, tmp: Path) -> str:
    game_id = "TasPack-Nes-v0"
    d = tmp / game_id
    d.mkdir(parents=True, exist_ok=True)
    data = rom.read_bytes()
    body = data[16:] if data[:4] == b"NES\x1a" else data
    shutil.copy(rom, d / "rom.nes")
    (d / "rom.sha").write_text(hashlib.sha1(body).hexdigest() + "\n")
    (d / "data.json").write_text('{"info": {}}\n')
    (d / "metadata.json").write_text("{}\n")
    return game_id


def life_like(reader: HudReader, series: list[list[int]]) -> int | None:
    """Index of a counter that looks like lives or health: short and decreasing."""
    if not series:
        return None
    arr = np.array(series, dtype=float)
    for gi in range(arr.shape[1]):
        v = arr[:, gi]
        v = v[v >= 0]
        if len(v) < 20 or len(reader.groups[gi]) > 2:
            continue
        if (np.diff(v) < 0).any() and v.max() <= 99:
            return gi
    return None


ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=20)
ap.add_argument("--out", default="datasets/tas_pack")
ap.add_argument("--max-frames", type=int, default=18000, help="cap per movie")
a = ap.parse_args()

sys_path = ROOT / "rnd" / "rom_index.json"
index = json.loads(sys_path.read_text()) if sys_path.exists() else {}
print(f"ROMs in the index: {len(index)}")

pubs = []
for page in range(1, 20):
    pubs += [p for p in api(page)
             if p.get("systemCode") == "NES"
             and (p.get("movieFileName") or "").lower().endswith(".fm2")]
    if len(pubs) >= a.limit:
        break

out_root = Path(a.out)
kept = skipped = 0
for pub in pubs[: a.limit]:
    path = download(pub["id"])
    if path is None:
        continue
    movie = parse_fm2(path)
    md5 = movie.rom_md5
    rom = index.get(md5.hex()) if md5 else None
    if rom is None:
        skipped += 1
        print(json.dumps({"id": pub["id"], "skip": "no ROM"}), flush=True)
        continue

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rom_path = resolve_rom(rom, tmp)
        if rom_path is None:
            skipped += 1
            print(json.dumps({"id": pub["id"], "skip": "extraction failed"},
                             ensure_ascii=False), flush=True)
            continue
        game_id = temp_integration(rom_path, tmp)
        ep_dir = out_root / f"tas_{pub['id']}"
        writer = None
        reader, hud_buf, series = HudReader(), [], []
        prev, static, cut, n = None, 0, None, 0
        try:
            for obs, pressed in replay_frames(game_id, path, integration_dir=tmp,
                                              max_frames=a.max_frames):
                g = cv2.cvtColor(obs.frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.int16)
                static = static + 1 if prev is not None and float(
                    np.abs(g - prev).mean()) < 0.3 else 0
                prev = g
                if static >= STATIC_LIMIT:
                    cut = n - static
                    break
                if n > WARMUP:
                    if not reader.groups and len(hud_buf) < 240 and n % 4 == 0:
                        hud_buf.append(obs.frame_rgb.copy())
                    elif not reader.groups and len(hud_buf) >= 240:
                        reader.fit(hud_buf)
                    elif reader.groups and n % 30 == 0:
                        series.append(reader.read(obs.frame_rgb))
                        gi = life_like(reader, series)
                        if gi is not None:
                            cut = n
                            break
                if writer is None:
                    writer = EpisodeWriter(out_dir=ep_dir, metadata={
                        "game": rom_path.name, "movie": path.name,
                        "source": "tasvideos.org", "publication": pub["id"],
                        "sample_rate": obs.sample_rate})
                writer.append(obs, pressed)
                n += 1
        except Exception as e:  # noqa: BLE001 — the failure rate is what we measure
            print(json.dumps({"id": pub["id"], "error": str(e)[:70]}, ensure_ascii=False),
                  flush=True)
        finally:
            if writer is not None:
                writer.close()
        kept += 1
        print(json.dumps({"id": pub["id"], "game": rom_path.name[:40],
                          "frames_kept": n, "cut_at": cut,
                          "reason": "life lost" if cut and cut == n else
                                    ("stuck" if cut else "played to the end")},
                         ensure_ascii=False), flush=True)

print(json.dumps({"episodes": kept, "no_rom": skipped}))
