"""Look at a --arch linear policy: its weights are the picture.

The linear policy holds one weight per pixel per action, so the map from
screen to button is not hidden behind a representation — it can be printed.
This writes one panel per action per frame in the stack: red where a bright
pixel pushes the action's score up, blue where it pushes it down.

    uv run python scripts/experiments/linear_map.py runs/bc_px_linear_0 --out map.png
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from nes_player.policy.bc import INPUT_HW


def action_maps(run: Path) -> tuple[np.ndarray, list[str]]:
    """(n_actions, n_frames, H, W) signed weights, summed over colour."""
    meta = json.loads((run / "meta.json").read_text())
    if meta.get("arch") != "linear":
        raise SystemExit(f"{run} is --arch {meta.get('arch', 'conv')}; this reads linear ones")
    w = torch.load(run / "model.pt", map_location="cpu")["fc.weight"].numpy()
    n_frames = len(meta["frame_offsets"])
    w = w.reshape(len(w), n_frames, 3, *INPUT_HW)
    # A softmax ignores whatever every action shares, so the part of the weight
    # that is common to all actions cannot move a single decision. Subtracting
    # it is not cosmetic: it is the difference between showing the policy and
    # showing the mean brightness of Contra.
    w = w - w.mean(0, keepdims=True)
    return w.sum(2), meta["vocab_names"]


def panel(m: np.ndarray, scale: float) -> np.ndarray:
    """One signed map to a BGR tile: red positive, blue negative, white zero."""
    v = np.clip(m / scale, -1, 1)
    pos, neg = np.clip(v, 0, 1), np.clip(-v, 0, 1)
    white = 1.0 - np.maximum(pos, neg)
    return (np.stack([white + neg, white, white + pos], axis=-1).clip(0, 1) * 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default="linear_map.png")
    ap.add_argument("--zoom", type=int, default=2)
    ap.add_argument("--shared-scale", action="store_true",
                    help="one colour scale for every action instead of one per "
                         "action: shows which actions carry the strongest "
                         "weights, at the cost of washing the weaker ones out")
    args = ap.parse_args()

    maps, names = action_maps(Path(args.run))
    if args.shared_scale:   # panels comparable across actions, faint rows stay faint
        scales = np.full(len(maps), float(np.abs(maps).max()))
    else:                   # each action to its own peak, so every row is readable
        scales = np.abs(maps).reshape(len(maps), -1).max(1)
    scale = float(scales.max())
    pad, label_w = 4, 130
    tiles = [[panel(maps[a, f], float(scales[a]) + 1e-12) for f in range(maps.shape[1])]
             for a in range(len(maps))]
    th, tw = tiles[0][0].shape[:2]
    rows = []
    for a, row in enumerate(tiles):
        strip = np.full((th, label_w + len(row) * (tw + pad), 3), 255, np.uint8)
        cv2.putText(strip, names[a][:16], (4, th // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 1, cv2.LINE_AA)
        for f, t in enumerate(row):
            x = label_w + f * (tw + pad)
            strip[:, x:x + tw] = t
        rows.append(strip)
        rows.append(np.full((pad, strip.shape[1], 3), 255, np.uint8))
    sheet = np.vstack(rows)
    if args.zoom != 1:
        sheet = cv2.resize(sheet, None, fx=args.zoom, fy=args.zoom,
                           interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(args.out, sheet)
    print(f"{len(maps)} actions x {maps.shape[1]} frames, "
          f"peak weight {scale:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
