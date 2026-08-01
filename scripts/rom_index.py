"""Index a ROM library by checksum: MD5 of the body, without the iNES header.

An .fm2 movie identifies its ROM by exactly that MD5. Rescanning the library
for every movie is not viable — across tens of thousands of files that is
minutes per movie — so the index is built once and cached.

Usage:
  uv run python scripts/rom_index.py build <rom-directory> [more directories...]
  uv run python scripts/rom_index.py find <md5-hex>
"""

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
INDEX = ROOT / "rnd" / "rom_index.json"


def rom_md5(path: Path) -> str:
    data = path.read_bytes()
    body = data[16:] if data[:4] == b"NES\x1a" else data   # movies hash without the header
    return hashlib.md5(body).hexdigest()


def index_archive(archive: Path, index: dict[str, str]) -> int:
    """Look inside .7z and .zip archives, where large libraries keep their ROMs.

    The index value is "archive::name inside", extracted later one at a time.
    """
    import subprocess
    import tempfile

    added = 0
    with tempfile.TemporaryDirectory() as td:
        cmd = (["7zz", "x", "-y", f"-o{td}", str(archive)] if archive.suffix == ".7z"
               else ["unzip", "-o", "-q", str(archive), "-d", td])
        if subprocess.run(cmd, capture_output=True, check=False).returncode != 0:
            return 0
        for p in Path(td).rglob("*"):
            if p.suffix.lower() not in (".nes", ".fds", ".unf", ".unif"):
                continue
            key = rom_md5(p)
            if key not in index:
                index[key] = f"{archive}::{p.relative_to(td)}"
                added += 1
    return added


def build(dirs: list[str]) -> dict[str, str]:
    index: dict[str, str] = {}
    if INDEX.exists():
        index = json.loads(INDEX.read_text())
    seen = 0
    for d in dirs:
        for p in sorted(Path(d).rglob("*")):
            suf = p.suffix.lower()
            if suf in (".7z", ".zip"):
                seen += 1
                index_archive(p, index)
            elif suf in (".nes", ".fds", ".unf", ".unif"):
                seen += 1
                try:
                    index.setdefault(rom_md5(p), str(p))
                except OSError as e:
                    print(f"skipped {p.name}: {e}")
            else:
                continue
            if seen % 200 == 0:
                print(f"scanned {seen}, unique ROMs {len(index)}", flush=True)
                INDEX.parent.mkdir(parents=True, exist_ok=True)
                INDEX.write_text(json.dumps(index, indent=0, ensure_ascii=False))
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(index, indent=0, ensure_ascii=False))
    print(f"files scanned: {seen}, unique ROMs: {len(index)} -> {INDEX}")
    return index


def load() -> dict[str, str]:
    return json.loads(INDEX.read_text()) if INDEX.exists() else {}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("build", "find"):
        print(__doc__)
        raise SystemExit(1)
    if sys.argv[1] == "build":
        roots = sys.argv[2:] or [os.environ.get("NES_ROMSET", "")]
        if not roots[0]:
            raise SystemExit("pass a ROM directory, or set NES_ROMSET")
        build(roots)
    else:
        print(load().get(sys.argv[2], "not found"))
