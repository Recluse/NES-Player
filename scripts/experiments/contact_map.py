"""Build the contact map of a level: solidity proven by Mario's body.

The battery's privileged geometry input needs exact ground truth, and the
console's tile map could not be pinned to an address that survives two levels.
The contact reference decodes it at 99% anyway, so the reference is promoted
to the artefact: every tile Mario's box ever covered is empty, every tile
under his feet while standing is solid, and everything else says unknown
instead of pretending.

One file per level, world-tile coordinates, with evidence counts rather than
booleans — a downstream user can set its own threshold and see the support.

    uv run python scripts/experiments/contact_map.py runs/bc_smb_new \
        --state Level3-1 --seeds 8
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

MARIO_Y, AIRBORNE = 0xCE, 0x1D
BOX_W, BOX_H = 16, 16


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--game", default="SuperMarioBros-Nes-v0")
    ap.add_argument("--state", default="default")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--out", default=None,
                    help="default runs/knowledge/contact_<state>.npz")
    args = ap.parse_args()

    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.policy.bc import BCPolicy
    from nes_player.policy.go_explore import _begin

    solid: dict = {}
    empty: dict = {}
    env = StableRetroAdapter(args.game, include_debug=True, state=args.state)
    policy = BCPolicy(args.checkpoint)
    played = 0
    for seed in range(args.seed0, args.seed0 + args.seeds):
        np.random.seed(seed)
        policy.reset()
        obs = _begin(env)
        pressed: frozenset = frozenset()
        for i in range(args.frames):
            if i % args.repeat == 0:
                pressed, _ = policy.act(obs.frame_rgb, args.temperature)
                pressed = pressed - {"START", "SELECT"}
            obs = env.step_buttons([pressed])
            played += 1
            ram = env._env.get_ram()
            sy = int(ram[MARIO_Y])
            if not (0 < sy < 208):
                continue
            wx = int(ram[0x6D]) * 256 + int(ram[0x86])
            for tx in range(wx // 16, (wx + BOX_W - 1) // 16 + 1):
                for ty in range(sy // 16, (sy + BOX_H - 1) // 16 + 1):
                    empty[(tx, ty)] = empty.get((tx, ty), 0) + 1
            if int(ram[AIRBORNE]) == 0:
                ty = (sy + BOX_H) // 16
                for tx in range(wx // 16, (wx + BOX_W - 1) // 16 + 1):
                    solid[(tx, ty)] = solid.get((tx, ty), 0) + 1
    env.close()

    keys = sorted(set(solid) | set(empty))
    arr = np.array([(tx, ty, solid.get((tx, ty), 0), empty.get((tx, ty), 0))
                    for tx, ty in keys], np.int32)
    both = sum(1 for k in keys if k in solid and k in empty)
    out = Path(args.out or
               f"runs/knowledge/contact_{args.state}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, evidence=arr, state=args.state)
    print(json.dumps({
        "path": str(out), "state": args.state, "frames": played,
        "tiles": len(keys),
        "solid_only": sum(1 for k in keys
                          if k in solid and k not in empty),
        "empty_only": sum(1 for k in keys
                          if k in empty and k not in solid),
        "contradictory": both,
        "max_tx": int(arr[:, 0].max()) if len(arr) else 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
