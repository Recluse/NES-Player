#!/usr/bin/env python3
"""Play Galaga with DT policy vs BC policy — actual gameplay comparison.

DT has never played before (all experiments were offline). This runs
both policies as reactive agents and compares scores.

    uv run python scripts/play_dt.py --game Galaga-Nes-v0 --frames 5000
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")


def score_of(frame_rgb: np.ndarray, game: str) -> int:
    """Extract score from frame based on game."""
    if game.startswith("Galaga"):
        band = frame_rgb[62:78, 215:240]
        return int(((band[..., 0] > 180) & (band[..., 1] > 180)
                    & (band[..., 2] > 180)).sum())
    return 0


def play_episode(policy, game: str, integrations: str | None,
                 n_frames: int, seed: int = 0, temperature: float = 0.5) -> dict:
    """Run one episode with the given policy."""
    from nes_player.emulator.stable_retro import StableRetroAdapter

    # Use begin_any for non-SMB games
    import sys
    sys.path.insert(0, 'scripts/experiments')
    from oracle_mpc import begin_any

    env = StableRetroAdapter(game, include_debug=True, state=None,
                             integration_dir=integrations)
    obs = begin_any(env, game)

    # Idle steps for seed variation
    for _ in range(seed * 7 % 30):
        obs = env.step_buttons([frozenset()])

    policy.reset()
    best_score = 0
    deaths = 0
    lives = (obs.debug or {}).get("lives")
    t0 = time.time()

    for i in range(n_frames):
        pressed, _ = policy.act(obs.frame_rgb, temperature)
        pressed = pressed - {"START", "SELECT"}
        obs = env.step_buttons([pressed])

        score = score_of(obs.frame_rgb, game)
        best_score = max(best_score, score)

        now = (obs.debug or {}).get("lives")
        if lives is not None and now is not None and now < lives:
            deaths += 1
        lives = now

    elapsed = time.time() - t0
    return {
        "best_score": best_score,
        "deaths": deaths,
        "frames": n_frames,
        "elapsed_s": round(elapsed, 1),
    }


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--game", default="Galaga-Nes-v0")
    p.add_argument("--frames", type=int, default=5000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--bc-checkpoint", default="runs/bc_galaga_shoot")
    p.add_argument("--dt-checkpoint", default="runs/dt_galaga.pt")
    p.add_argument("--integrations", default=None)
    p.add_argument("--temperature", type=float, default=0.5)
    args = p.parse_args()

    print(f"=== Play {args.game}: DT vs BC ===")

    # Load policies
    from nes_player.policy.bc import BCPolicy
    from nes_player.policy.dt_policy import DTPolicy

    bc = BCPolicy(args.bc_checkpoint)
    dt = DTPolicy(args.dt_checkpoint, target_rtg=50.0)
    print(f"BC: {args.bc_checkpoint}")
    print(f"DT: {args.dt_checkpoint}")

    # Run episodes
    results = {"bc": [], "dt": []}
    for seed in range(args.seeds):
        print(f"\n--- Seed {seed} ---")

        print("  BC playing...")
        bc_result = play_episode(bc, args.game, args.integrations,
                                 args.frames, seed, args.temperature)
        results["bc"].append(bc_result)
        print(f"    score={bc_result['best_score']}, deaths={bc_result['deaths']}")

        print("  DT playing...")
        dt_result = play_episode(dt, args.game, args.integrations,
                                 args.frames, seed, args.temperature)
        results["dt"].append(dt_result)
        print(f"    score={dt_result['best_score']}, deaths={dt_result['deaths']}")

    # Summary
    bc_scores = [r["best_score"] for r in results["bc"]]
    dt_scores = [r["best_score"] for r in results["dt"]]
    bc_deaths = [r["deaths"] for r in results["bc"]]
    dt_deaths = [r["deaths"] for r in results["dt"]]

    print("\n=== Summary ===")
    print(f"BC: mean score={np.mean(bc_scores):.0f}, deaths={np.mean(bc_deaths):.1f}")
    print(f"DT: mean score={np.mean(dt_scores):.0f}, deaths={np.mean(dt_deaths):.1f}")
    print(f"DT - BC: score {np.mean(dt_scores) - np.mean(bc_scores):+.0f}")

    # Save results
    out = {
        "game": args.game,
        "frames_per_episode": args.frames,
        "seeds": args.seeds,
        "bc_checkpoint": args.bc_checkpoint,
        "dt_checkpoint": args.dt_checkpoint,
        "bc_scores": bc_scores,
        "dt_scores": dt_scores,
        "bc_deaths": bc_deaths,
        "dt_deaths": dt_deaths,
        "bc_mean_score": float(np.mean(bc_scores)),
        "dt_mean_score": float(np.mean(dt_scores)),
        "diff": float(np.mean(dt_scores) - np.mean(bc_scores)),
    }
    out_path = Path(f"runs/play_dt_{args.game}.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n-> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
