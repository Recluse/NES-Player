"""A7: the section counter, from the transitions the picture shows.

The last hand-written input a stage needed was its progress counter —
Contra's base has rooms, and the byte that counts them (100) was found by
an agent's scripted play and a RAM diff. The same thing can be read off
any run that actually changes section: the picture says when (the scene
hash jumps and, in Contra, the screen fades to black on the way), and the
byte that steps exactly then, and holds still in between, is the counter.

Two traces are required, played differently. A single transition admits
impostors — a free-running timer that happened to tick then — and the
room-counter search met one (byte 280, period ~768 frames). A byte that
steps at the transition in both traces and nowhere else in either has
earned the name.

    uv run python scripts/experiments/oracle_mpc.py ... --trace runs/knowledge/trace_base
    uv run python scripts/experiments/section_scan.py runs/knowledge/trace_base_s0.npz \
        runs/knowledge/trace_base_s4.npz --game ContraJ-Nes-v0

Writes runs/knowledge/sections_<game>.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

FADE = 40.0        # mean frame brightness below which the screen is "black"


def transitions(scene: np.ndarray, lum: np.ndarray, gap: int = 90) -> list:
    """Frames at which the section changes, from the picture alone.

    Fades to black are the marker when a game has them: in two base traces
    the fade began one frame after the room counter stepped, and nowhere
    else. The scene hash is only a fallback for games that cut without a
    fade — a coarse hash of a busy room changes seventy times in 1800
    frames, so a hash change counts only if the picture never returns to
    anything it looked like in the previous 300 frames.
    """
    dark = lum < FADE
    marks = [t for t in range(1, len(lum)) if dark[t] and not dark[t - 1]]
    if not marks:
        for t in np.nonzero(np.diff(scene) != 0)[0] + 1:
            before = set(scene[max(0, t - 300):t].tolist())
            after = scene[t:t + gap]
            if len(after) >= gap and not (set(after.tolist()) & before):
                marks.append(int(t))
    out = []
    for t in sorted(marks):
        if not out or t - out[-1] > gap:
            out.append(t)
    return out


def counters(ram: np.ndarray, marks: list, slack: int = 90) -> dict:
    """Bytes constant inside every segment and different across them.

    `slack` frames around each transition are excluded from the constancy
    test — the fade and the load shuffle a lot of memory — but the value
    must have changed once the dust settles.
    """
    bounds = [0, *marks, len(ram)]
    segs = []
    for a, b in zip(bounds, bounds[1:], strict=False):
        lo, hi = a + slack, b - slack
        if hi - lo < 60:
            return {}
        segs.append(ram[lo:hi])
    out = {}
    for addr in range(ram.shape[1]):
        vals = []
        ok = True
        for seg in segs:
            col = seg[:, addr]
            if (col != col[0]).any():
                ok = False
                break
            vals.append(int(col[0]))
        if ok and len(set(vals)) == len(vals):
            out[addr] = vals
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--game", required=True)
    args = ap.parse_args()

    per_trace = []
    for path in args.traces:
        z = np.load(path)
        marks = transitions(z["scene"], z["lum"])
        found = counters(z["ram"], marks)
        print(f"{Path(path).name}: {len(z['lum'])} frames, transitions at "
              f"{marks}, {len(found)} candidate byte(s)")
        per_trace.append((marks, found))

    with_transitions = [f for m, f in per_trace if m]
    if len(with_transitions) < 2:
        print("need two traces that each change section; got",
              len(with_transitions))
        return 1
    common = set.intersection(*(set(f) for f in with_transitions))
    # the values must step the same way in every trace: +1 per section
    steady = sorted(a for a in common if all(
        np.diff(f[a]).tolist() == [1] * (len(f[a]) - 1) for f in with_transitions))
    print("bytes that step +1 at every transition in every trace:", steady)
    for a in steady:
        print(f"  0x{a:03x} ({a}):", [f[a] for f in with_transitions])
    out = Path("runs/knowledge") / f"sections_{args.game}.json"
    out.write_text(json.dumps({"game": args.game, "counters": steady,
                               "traces": args.traces,
                               "transitions": [m for m, _ in per_trace]},
                              indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
