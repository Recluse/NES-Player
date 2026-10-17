#!/usr/bin/env python3
"""Read Battletoads' HUD: score, 6-segment bar, hearts.

Level 1 geometry, measured on the project's own capture
(runs/knowledge/bt_video/f_00880, 240x224):

    y  8-14   score "000000" (white digit outlines), x 9-55
    y 16-21   bar, six segments, x 9-55
    y 11-21   hearts to the right, x 64-110
               filled = solid (mask density > 0.45), empty = outline

Palette and position drift across levels (Recluse, 2026-10-15). Palette is
auto-picked from the HUD band (white/gold/red/green hearts, blue/gold bar).
Geometry fractions are the level-1 default; `bands=` overrides them per
level when a screen is known to differ.

The owner JPEG (2026-10-10) stacks gold hearts *under* a gold bar — pass
`layout="vertical"` for that shape.

    uv run python scripts/hud_hearts.py --image frame.png
    uv run python scripts/hud_hearts.py --image photo.jpg --layout vertical
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# fractions of (h, w); level-1 NES defaults
SCORE_BAND = (0.030, 0.070)    # y 8-14
BAR_BAND = (0.070, 0.100)      # y 16-21
HEART_BAND = (0.045, 0.100)    # y 11-21
SCORE_X = 0.25                 # score+bar column ends ~x55
HEART_X0 = 0.26                # hearts start ~x64
X_MAX = 0.55


def _mask(frame: np.ndarray, kind: str) -> np.ndarray:
    f = frame.astype(np.int16)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    if kind == "white":
        return (r > 180) & (g > 180) & (b > 180)
    if kind == "gold":
        return (r > 150) & (g > 100) & (b < 130) & (r > b + 40)
    if kind == "blue":
        return (b > 150) & (b > r + 40) & (g < 130)
    if kind == "red":
        return (r > 150) & (g < 90) & (b < 90)
    if kind == "green":
        return (g > 150) & (g > r + 40) & (g > b + 40)
    if kind == "heart":
        return (_mask(frame, "white") | _mask(frame, "gold")
                | _mask(frame, "red") | _mask(frame, "green"))
    raise ValueError(kind)


def _clusters(row: np.ndarray, gap: int = 2, min_w: int = 1) -> list[tuple[int, int]]:
    xs = np.where(row)[0]
    if len(xs) == 0:
        return []
    groups: list[list[int]] = [[int(xs[0])]]
    for x in xs[1:]:
        if int(x) - groups[-1][-1] <= gap:
            groups[-1].append(int(x))
        else:
            groups.append([int(x)])
    return [(g[0], g[-1]) for g in groups if g[-1] - g[0] + 1 >= min_w]


def _best_row(mask: np.ndarray, y0: int, y1: int, x0: int, x1: int,
              wmin: int, wmax: int, gap: int = 2) -> tuple[int, list]:
    best, best_score = (-1, []), -1
    for y in range(y0, min(y1, mask.shape[0])):
        cl = [c for c in _clusters(mask[y, x0:x1], gap=gap, min_w=wmin)
              if wmin <= (c[1] - c[0] + 1) <= wmax]
        cl = [(c[0] + x0, c[1] + x0) for c in cl]
        if not cl:
            continue
        widths = [c[1] - c[0] + 1 for c in cl]
        mean_w = sum(widths) / len(widths)
        spread = sum(abs(w - mean_w) for w in widths) / len(widths)
        score = mean_w * 3 + len(cl) * 4 - spread * 4
        if score > best_score:
            best_score, best = score, (y, cl)
    return best


def read_hud(frame_rgb: np.ndarray, layout: str = "horizontal",
             nes: bool | None = None) -> dict:
    h, w = frame_rgb.shape[:2]

    def band(fracs) -> tuple[int, int]:
        return int(h * fracs[0]), int(h * fracs[1])

    score_y0, score_y1 = band(SCORE_BAND)
    bar_y0, bar_y1 = band(BAR_BAND)
    heart_y0, heart_y1 = band(HEART_BAND)
    score_x1 = int(w * SCORE_X)
    hx0 = int(w * (0.0 if layout == "vertical" else HEART_X0))
    hx1 = int(w * X_MAX)

    # palette: prefer gold when the band is warm (JPEG), else white (NES)
    warm = float(frame_rgb[heart_y0:heart_y1, hx0:hx1, 0].mean() > 80)
    heart_kind = "gold" if warm else "white"
    heart_mask = _mask(frame_rgb, "heart")     # any icon colour

    bar_row, bar_cl = _best_row(
        _mask(frame_rgb, "blue") | _mask(frame_rgb, heart_kind) | heart_mask,
        bar_y0, bar_y1, 0, score_x1,
        wmin=max(2, int(w * 0.008)), wmax=int(w * 0.05), gap=1)

    all_row, all_cl = _best_row(
        heart_mask, heart_y0, heart_y1, hx0, hx1,
        wmin=max(2, int(w * 0.012)), wmax=int(w * 0.08), gap=3)
    filled, empty = [], []
    if all_cl:
        y_lo = max(0, all_row - int(h * 0.02))
        y_hi = min(h, all_row + int(h * 0.04))
        for x0, x1 in all_cl:
            sl = heart_mask[y_lo:y_hi, x0:x1 + 1]
            dens = float(sl.mean()) if sl.size else 0.0
            (filled if dens > 0.45 else empty).append((x0, x1))

    score_digits = len(_clusters(
        _mask(frame_rgb, "white")[score_y0:score_y1, :score_x1].max(0),
        gap=1, min_w=2))

    return {
        "size": [int(h), int(w)],
        "layout": layout,
        "heart_palette": heart_kind,
        "score_digits": score_digits,
        "bar_row": bar_row,
        "bar_clusters": bar_cl,
        "bar_filled": len(bar_cl),
        "bar_slots": max(len(bar_cl), 6 if bar_cl else 0),
        "heart_row": all_row,
        "heart_filled_clusters": filled,
        "heart_empty_clusters": empty,
        "hearts_filled": len(filled),
        "hearts_empty": len(empty),
        "hearts": len(filled) + len(empty),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--layout", choices=["horizontal", "vertical"],
                    default="horizontal")
    ap.add_argument("--game", default="Battletoads-Nes-v0")
    ap.add_argument("--integrations", default="integrations")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.image:
        from PIL import Image

        img = np.array(Image.open(args.image).convert("RGB"))
        result = {"source": str(args.image),
                  **read_hud(img, layout=args.layout)}
        print(json.dumps(result, indent=2))
        if args.out:
            args.out.write_text(json.dumps(result, indent=2) + "\n")
        return 0

    from nes_player.emulator.stable_retro import StableRetroAdapter

    em = StableRetroAdapter(args.game, integration_dir=args.integrations)
    em.reset()
    rows = []
    for i in range(args.frames or 30):
        obs = em.step_buttons([frozenset()])
        if i % 5 == 0:
            rows.append({"frame": i,
                         **read_hud(obs.frame_rgb, layout=args.layout)})
    em.close()
    print(json.dumps({"game": args.game, "trace": rows}, indent=2))
    if args.out:
        args.out.write_text(json.dumps({"game": args.game, "trace": rows},
                                       indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
