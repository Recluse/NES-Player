"""Build the attention masks for a dataset, several episodes at a time.

Training does this by itself on first use, one episode after another. That is
fine for one episode and painful for a directory of them: the tracker runs over
every frame, which is about two minutes per episode, so a 36-episode dataset
spends over an hour of wall clock on one core while the rest of the machine
idles. Episodes are independent, so they can simply go in parallel.

    uv run python scripts/precompute_attn.py datasets/explore_dd4

Already-cached episodes are skipped, so this is safe to re-run and safe to
interrupt. Training afterwards finds every mask in place and starts immediately.
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def build(path_str: str) -> tuple[str, int, float]:
    from nes_player.data.reader import Episode
    from nes_player.policy.bc import episode_attn_masks

    t0 = time.monotonic()
    masks = episode_attn_masks(Episode(Path(path_str)))
    return path_str, int(masks.shape[0]), time.monotonic() - t0


def main() -> int:
    from nes_player.perception.motion import TRACKER_VERSION

    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="directory of episodes, or a single episode")
    ap.add_argument("--workers", type=int, default=None,
                    help="default: cores minus two, leaving the machine usable")
    args = ap.parse_args()

    root = Path(args.dataset)
    episodes = ([root] if (root / "metadata.json").exists()
                else sorted(d for d in root.iterdir() if (d / "metadata.json").exists()))
    todo = [d for d in episodes
            if not (d / f"attn_mask.v{TRACKER_VERSION}.npy").exists()]
    print(f"{len(episodes)} episodes, {len(episodes) - len(todo)} already cached, "
          f"{len(todo)} to build (tracker v{TRACKER_VERSION})")
    if not todo:
        return 0

    import os

    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (name, frames, secs) in enumerate(
                pool.map(build, [str(d) for d in todo]), 1):
            print(f"[{i}/{len(todo)}] {Path(name).name}  {frames} frames  {secs:.0f}s",
                  flush=True)
    print(f"done in {(time.monotonic() - t0) / 60:.1f} min on {workers} workers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
