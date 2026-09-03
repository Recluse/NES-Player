"""A0: what in this game answers to the buttons? Ask the console, causally.

The button probe (probe_buttons.py) was circular: it found "forward" through
the game's position counter — the very per-game variable a universal agent
should discover. This scan uses nothing per-game. From one save-state it
runs a synchronous no-op branch and, beside it, one branch per chord and
mode, and asks only: what differs from doing nothing? Animation, timers and
enemies march identically in every branch, so the difference is the causal
effect of the button. Three readouts per branch, all relative to no-op:

    screen   pixels that differ at the end, and when they first differ
    ram      bytes that differ at the end; for LEFT/RIGHT, the bytes that
             move in opposite directions are position candidates
    audio    whether the sound differs (a shot is heard before it is seen)

From these, without any RAM map: whether the game is controllable at all
here (a title screen answers to nothing), where the controlled thing is on
screen, which bytes are position, which chord fires and whether it must be
tapped, and which moves cost a life.

    uv run python scripts/experiments/controllability.py SuperMarioBros-Nes-v0
    uv run python scripts/experiments/controllability.py ContraJ-Nes-v0 --state none
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle_mpc import begin_any  # noqa: E402

CHORDS = [frozenset({b}) for b in ("A", "B", "UP", "DOWN", "LEFT", "RIGHT")] + [
    frozenset({"RIGHT", "A"}), frozenset({"RIGHT", "B"}),
    frozenset({"UP", "B"}), frozenset({"DOWN", "B"}),
]
T = 32
MODES = {
    "press": lambda k: k == 0,          # one-frame edge, then release
    "hold": lambda k: True,
    "tap": lambda k: k % 4 < 2,
}


def branch(env, state, chord, mode):
    env.load_state(state)
    frames, rams, pcm, died = [], [], [], False
    lives0 = None
    for k in range(T):
        on = MODES[mode](k)
        obs = env.step_buttons([chord if on else frozenset()])
        frames.append(obs.frame_rgb.astype(np.int16))
        rams.append(env._env.get_ram().astype(np.int16).copy())
        pcm.append(obs.audio_pcm.astype(np.int32) if obs.audio_pcm is not None
                   else np.zeros(1, np.int32))
        lv = (obs.debug or {}).get("lives")
        if lives0 is None:
            lives0 = lv
        elif lv is not None and lives0 is not None and lv < lives0:
            died = True
    return np.stack(frames), np.stack(rams), pcm, died


def compare(b, noop):
    fr, ram, pcm, died = b
    fr0, ram0, pcm0, _ = noop
    # A one-pixel camera shift makes every brick edge "differ"; a button
    # that nudges residual velocity would then own the whole screen. So
    # the branch is compared at the best horizontal alignment within
    # +/-8 px, and only what still differs is the button's own effect.
    best = None
    for sh in range(-8, 9):
        d = (np.abs(np.roll(fr, sh, axis=2) - fr0).sum(-1) > 30)
        if best is None or d[-1].sum() < best[-1].sum():
            best = d
    dpix = best                                        # (T, H, W)
    npix = dpix.sum((1, 2))
    onset = int(np.argmax(npix > 20)) if (npix > 20).any() else -1
    dram = (ram[-1] != ram0[-1])
    # the core hands over 533 or 534 samples a frame; compare the overlap
    daud = float(np.mean([np.abs(a[:min(len(a), len(a0))]
                                 - a0[:min(len(a), len(a0))]).mean()
                          for a, a0 in zip(pcm, pcm0)]))
    return {"onset": onset, "npix_end": int(npix[-1]), "mask": dpix[-1],
            "dram": dram, "ram_delta": (ram[-1] - ram0[-1]),
            "audio": round(daud, 1), "died": died}


def main() -> int:
    from nes_player.emulator.stable_retro import StableRetroAdapter

    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--state", default="default")
    ap.add_argument("--at", type=int, nargs="+", default=[60, 300, 600])
    args = ap.parse_args()
    state = None if args.state in ("none", "") else args.state
    root = Path(__file__).resolve().parents[2] / "integrations"
    integ = str(root) if (root / args.game).exists() else None
    env = StableRetroAdapter(args.game, include_debug=True, state=state,
                             integration_dir=integ)

    # a title-screen state first: the scan must call it uncontrollable
    obs = env.reset(seed=0)
    for _ in range(90):
        obs = env.step_buttons([frozenset()])
    states = [("title", env.save_state())]
    obs = begin_any(env, args.game)
    t = 0
    for target in sorted(args.at):
        while t < target:
            obs = env.step_buttons([frozenset({"RIGHT"})])
            t += 1
        # let momentum die before the snapshot: a hero still sliding from
        # the walk keeps sliding differently under every chord, and the
        # camera then votes the whole screen into the body
        for _ in range(90):
            obs = env.step_buttons([frozenset()])
        states.append((f"play+{target}", env.save_state()))

    report = {}
    for label, st in states:
        noop = branch(env, st, frozenset(), "hold")
        rows = {}
        for chord in CHORDS:
            for mode in MODES:
                rows[("+".join(sorted(chord)), mode)] = compare(
                    branch(env, st, chord, mode), noop)
        controllable = max(r["npix_end"] for r in rows.values()) > 20
        # controlled region: pixels that differ under at least two chords
        # that do NOT scroll the camera — LEFT/RIGHT move the whole
        # background and would vote the entire screen into the body
        votes = sum(r["mask"].astype(int) for (n, _), r in rows.items()
                    if "LEFT" not in n and "RIGHT" not in n)
        region = votes >= 2
        ys, xs = np.where(region)
        centre = ((int(xs.mean()), int(ys.mean())) if len(xs) else None)
        # position candidates: bytes pushed one way by RIGHT and the
        # other by LEFT, both held
        dr = rows[("RIGHT", "hold")]["ram_delta"]
        dl = rows[("LEFT", "hold")]["ram_delta"]
        # a position byte answers to the directions and to nothing else:
        # object-slot coordinates also move under LEFT/RIGHT (the camera
        # carries them) but they jump under B too — bullets are objects
        quiet = np.ones_like(dr, bool)
        for other in ("B", "UP", "DOWN"):
            quiet &= rows[(other, "hold")]["ram_delta"] == 0
        # ...and it moves like a position: at most ~2 px a frame, so a
        # byte that jumps by 200 in 32 frames is a wrapping counter
        sane = (np.abs(dr) <= 64) & (np.abs(dl) <= 64)
        pos = [int(i) for i in np.where((dr > 0) & (dl < 0) & quiet & sane)[0]]
        pos_r = [int(i) for i in np.where((dr > 0) & (dl == 0) & quiet & sane)[0]]
        fire = {}
        for name, mode in rows:
            if "B" in name and "RIGHT" not in name and "LEFT" not in name:
                r = rows[(name, mode)]
                extra = int((r["mask"] & ~region).sum())
                fire[f"{name}/{mode}"] = {"pixels_off_body": extra,
                                          "audio": r["audio"]}
        deaths = [f"{n}/{m}" for (n, m), r in rows.items() if r["died"]]
        # push: how far each chord drives the position bytes found above,
        # so "forward" and its fastest variant can be chosen without the
        # game's own counter — the circularity the button probe had
        pb = np.array(pos + pos_r, int)
        push = {f"{n}/{m}": int(r["ram_delta"][pb].sum()) if len(pb) else 0
                for (n, m), r in rows.items()}
        report[label] = {
            "controllable": bool(controllable),
            "controlled_centre": centre,
            "controlled_pixels": int(region.sum()),
            "position_bytes_bidirectional": pos[:12],
            "position_bytes_right_only": pos_r[:12],
            "fire": fire, "lethal": deaths, "push": push,
            "onsets": {f"{n}/{m}": r["onset"] for (n, m), r in rows.items()
                       if r["onset"] >= 0},
        }
        print(f"== {label}: controllable={controllable} centre={centre} "
              f"body_px={int(region.sum())}")
        print("   position bytes <-/->:", pos[:12], " right-only:", pos_r[:12])
        best = sorted(fire.items(), key=lambda kv: -kv[1]["pixels_off_body"])[:4]
        print("   fire candidates (px off body):",
              [(k, v["pixels_off_body"]) for k, v in best])
        top_push = sorted(push.items(), key=lambda kv: -kv[1])[:4]
        print("   push (position bytes moved):", top_push)
        if deaths:
            print("   lethal within 32 frames:", deaths)
    env.close()
    play = [r for r in report.values() if r["controllable"]]
    if play:
        # majority vote across controllable states: a hero pinned against
        # a pipe answers to RIGHT only, and one strict intersection would
        # lose the true byte to that one state
        from collections import Counter

        votes = Counter()
        for r in play:
            votes.update(set(r["position_bytes_bidirectional"]
                             + r["position_bytes_right_only"]))
        need = (len(play) + 1) // 2
        common = sorted(b for b, c in votes.items() if c >= need)
        report["position_bytes_consistent"] = common
        print(f"position bytes in >= {need} of {len(play)} states:", common)
    out = Path("runs/knowledge") / f"control_{args.game}.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
