"""Download trained checkpoints into runs/.

The weights are not in the repository — a single behavioural-cloning checkpoint
is 16 MB and there are seven of them, which is not what git is for. They are
published as release assets instead, described by assets/models.json with a
sha256 for each.

    uv run python scripts/fetch_models.py                 # everything
    uv run python scripts/fetch_models.py bc_smb_attn3    # one model
    uv run python scripts/fetch_models.py --list

You do not need any of this to use the project. The instinct policy plays
without training, and a model for a new game is about half an hour away:

    uv run nes-player explore --game <id> --record datasets/x --loop
    uv run nes-player train-bc --episode datasets/x --out runs/x --audio --attn 1.0
"""

import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets" / "models.json"
RUNS = ROOT / "runs"

# Override with NES_MODELS_URL when mirroring the assets somewhere else.
DEFAULT_BASE = "https://github.com/Recluse/NES-Player/releases/download/models-v1"


def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, spec: dict, base: str) -> None:
    dest = RUNS / spec.get("dest", name)
    marker = dest / spec.get("marker", "meta.json")
    if marker.exists():
        print(f"{name}: already present, skipping")
        return
    # data assets (scans, savestates, campaign logs) live in their own
    # dated release; the entry names it
    url = f"{spec.get('base', base)}/{spec['file']}"
    print(f"{name}: downloading {spec['bytes'] / 1e6:.1f} MB from {url}")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / spec["file"]
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 — URL comes from the manifest
        got = _sha256(tmp)
        if got != spec["sha256"]:
            raise SystemExit(
                f"{name}: checksum mismatch\n  expected {spec['sha256']}\n  got      {got}")
        RUNS.mkdir(exist_ok=True)
        # filter='data' refuses absolute paths and symlinks pointing outside.
        with tarfile.open(tmp) as tar:
            tar.extractall(RUNS, filter="data")
    print(f"{name}: unpacked into {dest.relative_to(ROOT)}")


def main() -> int:
    doc = manifest()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="model names; default is all of them")
    ap.add_argument("--list", action="store_true", help="show what is available")
    ap.add_argument("--base-url", default=os.environ.get("NES_MODELS_URL", DEFAULT_BASE))
    args = ap.parse_args()

    if args.list:
        for name, spec in doc["models"].items():
            acc = (f"  val {spec['val_acc']:.3f} (majority {spec['majority_baseline']:.3f})"
                   if "val_acc" in spec else "")
            print(f"{name:24} {spec['bytes'] / 1e6:6.1f} MB  {spec['title']}{acc}")
        return 0

    unknown = [m for m in args.models if m not in doc["models"]]
    if unknown:
        raise SystemExit(f"unknown model(s): {', '.join(unknown)}; try --list")
    for name in args.models or doc["models"]:
        fetch(name, doc["models"][name], args.base_url.rstrip("/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
