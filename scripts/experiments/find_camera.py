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
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

ADVANCE_R = frozenset({"B", "RIGHT"})
ADVANCE_L = frozenset({"B", "LEFT"})
JUMP_R = frozenset({"A", "B", "RIGHT"})
ADVANCE_U = frozenset({"B", "UP"})
JUMP_U = frozenset({"A", "B", "UP"})
JUMP_L = frozenset({"A", "B", "LEFT"})


def boot(env):
    """One boot for every game: oracle_mpc's, which asks the pad."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from oracle_mpc import begin_any

    return begin_any(env, "")


def main() -> int:
    from nes_player.emulator.stable_retro import StableRetroAdapter

    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--up", action="store_true",
                    help="the level scrolls vertically and the way on is up "
                         "(Ikari Warriors, Kid Icarus): advance with UP and "
                         "measure the picture's vertical shift")
    ap.add_argument("--left", action="store_true",
                    help="the level scrolls right-to-left; look for bytes "
                         "that decrease under advance instead")
    ap.add_argument("--frames", type=int, default=1800,
                    help="length of the long run used for wrap detection")
    args = ap.parse_args()

    adv, jmp = ((ADVANCE_U, JUMP_U) if args.up else
                (ADVANCE_L, JUMP_L) if args.left else (ADVANCE_R, JUMP_R))
    # A vertical camera may count either way; the agreement test below is
    # sign-blind, so only the monotone filter needs to know, and it is
    # tried both ways for --up.
    sgn = -1 if args.left else 1

    integ_root = Path(__file__).resolve().parents[2] / "integrations"
    integ = str(integ_root) if (integ_root / args.game).exists() else None
    env = StableRetroAdapter(args.game, include_debug=True, state=None,
                             integration_dir=integ)
    obs = boot(env)
    idle = []
    for _ in range(200):
        obs = env.step_buttons([frozenset()])
        idle.append(env._env.get_ram().copy())
    snaps, pics = [], []
    for k in range(args.frames):
        btn = jmp if (k // 30) % 4 == 3 else adv
        obs = env.step_buttons([btn])
        snaps.append(env._env.get_ram().copy())
        pics.append(obs.frame_rgb[16:224:2].mean(-1) if args.up
                    else obs.frame_rgb[64:200:2].mean(-1))
    env.close()

    def shift(a, b, R=12):
        # the offset that best explains picture b from picture a, along
        # the axis the level scrolls
        if args.up:
            return min(range(-R, R + 1), key=lambda d: np.abs(
                a[R + d:a.shape[0] - R + d, :] - b[R:b.shape[0] - R, :]).mean())
        return min(range(-R, R + 1), key=lambda d: np.abs(
            a[:, R + d:a.shape[1] - R + d] - b[:, R:b.shape[1] - R]).mean())
    # How many frames apart to compare depends on how fast the level
    # scrolls: Contra's jungle moves a pixel or two per frame and 4 is
    # plenty, Ikari Warriors climbs at 0.16 px/frame and at 4 the measured
    # shift is zero in a third of the samples and the correlation drowns
    # in quantisation. Take the first lag that actually sees motion.
    for LAG in (4, 8, 16, 32, 64):
        dx = np.array([shift(pics[t - LAG], pics[t])
                       for t in range(LAG, len(pics))], dtype=float)
        if (dx != 0).mean() >= 0.5 and abs(dx.mean()) >= 1.0:
            break
    print(f"comparing frames {LAG} apart: shift {dx.mean():+.2f} px, "
          f"non-zero in {(dx != 0).mean():.0%} of samples")

    I = np.stack(idle).astype(int)
    S_raw = np.stack(snaps).astype(int)

    def candidates(sg):
        S_ = S_raw * sg
        flat_ = (np.diff(I * sg, axis=0) == 0).all(0)
        out = []
        for a in range(S_.shape[1]):
            if not flat_[a]:
                continue
            col = S_[:, a]
            dd = np.diff(col)
            big_down = dd < -128        # a wrap, not a decrease
            if ((dd >= 0) | big_down).all() and (dd > 0).sum() >= 5 \
                    and col[-1] + 256 * big_down.sum() - col[0] >= 20:
                out.append((a, int(big_down.sum()), sg))
        return out

    # a vertical counter may run either way as the hero climbs; try both
    # and let the agreement with the picture decide
    cands3 = candidates(sgn) + (candidates(-sgn) if args.up else [])
    best_sign = {}
    for a, w, sg in cands3:
        best_sign.setdefault(a, sg)
    sgn = max(set(sg for _, _, sg in cands3), key=lambda g: sum(
        1 for _, _, x in cands3 if x == g)) if cands3 else sgn
    S = S_raw * sgn
    flat = (np.diff(I * sgn, axis=0) == 0).all(0)
    cands = [(a, w) for a, w, sg in cands3 if sg == sgn]
    print("moving, idle-flat bytes (addr, wraps):", cands)

    def agree(a):
        d = (S[LAG:, a] - S[:-LAG, a]).astype(float)
        d[d < -128] += 256
        return abs(float(np.corrcoef(d, dx)[0, 1])) if d.std() > 0 else 0.0
    cands.sort(key=lambda c: -agree(c[0]))
    print("ranked by agreement with screen scroll:",
          [(a, w, round(agree(a), 2)) for a, w in cands[:6]])

    los = [a for a, w in cands if w > 0]
    found = None
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
            found = found or {"hi": int(his), "lo": int(lo), "sign": sgn}
    if not found and cands:
        # No pair. Two shapes qualify, and the wrapping one must not be
        # assumed: Ikari Warriors' best-agreeing byte counts screens
        # climbed (0 -> 228, never wrapping), while the wrapping
        # candidates there are animation counters that turn over eighty
        # times. Take whichever agrees with the picture most, and say
        # which kind it is.
        best_a = max(cands, key=lambda c: agree(c[0]))[0]
        wraps = dict((a, w) for a, w in cands)[best_a]
        found = {"hi": None, "lo": int(best_a), "sign": sgn,
                 "wraps": int(wraps)}
        kind = "8-bit scroll, unwrap it" if wraps else "a counter, use as is"
        print(f"SINGLE BYTE: {best_a} (agreement {agree(best_a):.2f}, "
              f"{wraps} wraps) — {kind}")
    if found:
        out = Path("runs/knowledge") / f"camera_{args.game}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(found))
        print("wrote", out)
    if not los:
        print("no wrapping byte seen — run longer (--frames) or the level "
              "is shorter than 256 px of camera")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
