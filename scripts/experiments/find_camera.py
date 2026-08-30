"""Find a game's camera address by watching RAM move.

The stable-retro registry ships hundreds of NES ROMs whose RAM maps list
only lives and score — nothing a planner could value progress by. But a
side-scroller's camera is easy to catch red-handed: boot the game, stand
still, then advance, and keep the bytes that are flat while idle and
monotone while moving. The low byte wraps every 256 px; whichever byte
increments exactly at those wraps is the high byte. Super C's 0x6B/0x6C
pair was found this way in minutes and verified through two wraps.

    uv run python scripts/experiments/find_camera.py NinjaGaiden-Nes-v0
    uv run python scripts/experiments/find_camera.py KungFu-Nes-v0 --left
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

ADVANCE_R = frozenset({"B", "RIGHT"})
ADVANCE_L = frozenset({"B", "LEFT"})
JUMP_R = frozenset({"A", "B", "RIGHT"})
JUMP_L = frozenset({"A", "B", "LEFT"})


def boot(env):
    obs = env.reset(seed=0)
    for i in range(2000):
        d = obs.debug or {}
        if i > 120 and int(d.get("lives", 0) or 0) > 0:
            break
        pulse = i % 60 in (0, 1)
        obs = env.step_buttons([frozenset({"START"}) if pulse
                                else frozenset()])
    for _ in range(600):
        obs = env.step_buttons([frozenset()])
    return obs


def main() -> int:
    from nes_player.emulator.stable_retro import StableRetroAdapter

    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--left", action="store_true",
                    help="the level scrolls right-to-left; look for bytes "
                         "that decrease under advance instead")
    ap.add_argument("--frames", type=int, default=1800,
                    help="length of the long run used for wrap detection")
    args = ap.parse_args()

    adv, jmp = ((ADVANCE_L, JUMP_L) if args.left else (ADVANCE_R, JUMP_R))
    sgn = -1 if args.left else 1

    env = StableRetroAdapter(args.game, include_debug=True, state=None)
    obs = boot(env)
    idle = []
    for _ in range(200):
        obs = env.step_buttons([frozenset()])
        idle.append(env._env.get_ram().copy())
    snaps = []
    for k in range(args.frames):
        btn = jmp if (k // 30) % 4 == 3 else adv
        obs = env.step_buttons([btn])
        snaps.append(env._env.get_ram().copy())
    env.close()

    I = np.stack(idle).astype(int)
    S = np.stack(snaps).astype(int) * sgn
    flat = (np.diff(I * sgn, axis=0) == 0).all(0)
    cands = []
    for a in range(S.shape[1]):
        if not flat[a]:
            continue
        col = S[:, a]
        dd = np.diff(col)
        big_down = dd < -128        # a wrap, not a decrease
        if ((dd >= 0) | big_down).all() and (dd > 0).sum() >= 5 \
                and col[-1] + 256 * big_down.sum() - col[0] >= 20:
            cands.append((a, int(big_down.sum())))
    print("moving, idle-flat bytes (addr, wraps):", cands)

    los = [a for a, w in cands if w > 0]
    for lo, wraps in [(a, w) for a, w in cands if w > 0]:
        col = S[:, lo]
        wrap_at = np.where(np.diff(col) < -128)[0]
        his = None
        for a, _ in cands + [(x, 0) for x in range(S.shape[1])]:
            if a == lo or not flat[a]:
                continue
            hi = S[:, a]
            if all(hi[w + 1] - hi[w] == 1 for w in wrap_at) \
                    and (np.diff(hi) >= 0).all():
                his = a
                break
        if his is not None:
            val = S[:, his] * 256 + S[:, lo]
            print(f"PAIR: hi={his} lo={lo} "
                  f"(sign {sgn:+d}), monotone={bool((np.diff(val) >= -2).all())}, "
                  f"range {int(val[0])} -> {int(val[-1])}, wraps {wraps}")
    if not los:
        print("no wrapping byte seen — run longer (--frames) or the level "
              "is shorter than 256 px of camera")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
