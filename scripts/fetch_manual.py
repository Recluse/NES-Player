#!/usr/bin/env python3
"""Fetch a NES manual PDF into manuals/ (gitignored) and extract page text.

The scans themselves never enter the repository (docs/manuals.md). This
script is the one place that pulls them; derived facts go to assets/priors/.

Usage:
  uv run python scripts/fetch_manual.py --list
  uv run python scripts/fetch_manual.py Contra --out manuals/
  uv run python scripts/fetch_manual.py --pages manuals/Contra-US.pdf \
      --dump manuals/pages/Contra-US/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUALS = ROOT / "manuals"

# Named items from docs/manuals.md. The first two rows are the ones used so
# far; the rest are alternate sources when one is down.
ITEMS = {
    "ContraJ": {
        "url": "https://archive.org/download/rx2MRPnes1250pxPDF/Contra%20%28Re-Translation%29.pdf",
        "file": "Contra-ReTranslation.pdf",
        "note": "Japanese Famicom manual matching ContraJ-Nes-v0",
    },
    "ContraUS": {
        "url": "https://archive.org/download/contra-usa_202412/contra-usa.pdf",
        "file": "Contra-US.pdf",
        "note": "US NES manual, 600 dpi with OCR (archive.org/details/contra-usa_202412)",
    },
    "rx2-index": {
        "url": "https://archive.org/download/rx2MRPnes1250pxPDF/",
        "file": "",
        "note": "Collection index: 3316 NES manuals, most with a text layer",
    },
}

ALT = [
    "https://www.nesfiles.com/",
    "https://www.digitpress.com/library/manuals/nes/index.html",
    "https://www.gamesdatabase.org/",
    "https://www.gamingalexandria.com/",
]


def extract_pages(pdf: Path, out_dir: Path) -> list[Path]:
    """Dump one .txt per page with pdftotext -layout; OCR when there is no
    text layer (scanned manuals from gamesdatabase are images)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("page-*.txt"):
        old.unlink()
    rc = subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), "-"],
        capture_output=True,
    )
    if rc.returncode != 0:
        print(f"pdftotext failed on {pdf}: {rc.stderr.decode()[:400]}",
              file=sys.stderr)
        return []
    text = rc.stdout.decode("utf-8", errors="replace")
    pages = [p for p in text.split("\f")]
    # a scanned PDF yields form-feeds but no words
    if sum(len(p.split()) for p in pages) < 20:
        print(f"no text layer in {pdf.name}; running OCR")
        return _ocr_pages(pdf, out_dir)
    written: list[Path] = []
    for i, page in enumerate(pages, start=1):
        if not page.strip():
            continue
        path = out_dir / f"page-{i:03d}.txt"
        path.write_text(page)
        written.append(path)
    return written


def _ocr_pages(pdf: Path, out_dir: Path) -> list[Path]:
    """pdftoppm -> tesseract per page."""
    written: list[Path] = []
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        subprocess.run(
            ["pdftoppm", "-png", "-r", "200", str(pdf), str(tdp / "p")],
            check=True,
        )
        images = sorted(tdp.glob("p*.png"))
        for i, img in enumerate(images, start=1):
            rc = subprocess.run(
                ["tesseract", str(img), "stdout", "--psm", "6", "-l", "eng"],
                capture_output=True,
            )
            page = rc.stdout.decode("utf-8", errors="replace")
            if not page.strip():
                continue
            path = out_dir / f"page-{i:03d}.txt"
            path.write_text(page)
            written.append(path)
    return written


def fetch(name: str, out_dir: Path) -> Path | None:
    if name not in ITEMS:
        print(f"unknown item {name!r}; try --list", file=sys.stderr)
        return None
    item = ITEMS[name]
    if not item["file"]:
        print(item["note"])
        print("open the index and pick a file name")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / item["file"]
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"already present: {dest} ({dest.stat().st_size} bytes)")
        return dest
    url = item["url"]
    print(f"fetching {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
    except Exception as exc:  # network is allowed to fail; report, do not stack
        print(f"fetch failed: {exc}", file=sys.stderr)
        print(f"alternates: {', '.join(ALT)}", file=sys.stderr)
        return None
    dest.write_bytes(data)
    print(f"wrote {dest} ({len(data)} bytes) — gitignored, never commit")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", nargs="?", help="item name, see --list")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", type=Path, default=MANUALS)
    ap.add_argument("--pages", type=Path,
                    help="PDF to dump into page text files")
    ap.add_argument("--dump", type=Path,
                    help="directory for --pages output")
    args = ap.parse_args()

    if args.list:
        for k, v in ITEMS.items():
            print(f"{k:12} {v['file'] or '(index)':28} {v['note']}")
        return 0

    if args.pages:
        if not args.dump:
            args.dump = args.out / "pages" / args.pages.stem
        written = extract_pages(args.pages, args.dump)
        print(f"{len(written)} pages -> {args.dump}")
        return 0 if written else 1

    if not args.name:
        ap.error("give a name, or use --list / --pages")
    dest = fetch(args.name, args.out)
    if dest is None:
        return 1
    dump = args.out / "pages" / dest.stem
    written = extract_pages(dest, dump)
    print(f"{len(written)} pages -> {dump}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
