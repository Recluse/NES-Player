"""The strict held-out transfer protocol.

Few-shot on 3+1 episodes of games that appeared in neither the base nor any
hyperparameter tuning: scratch against transfer from each base, seeds 1-3.
Results are JSON lines. Transfer wins on two of the three; on Balloon Fight it
hurts, and that result is kept rather than dropped.

Usage:
  uv run python scripts/experiments/heldout_transfer.py [game ...]
Datasets are expected in datasets/explore_<game>/, collected with explore --record.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from nes_player.policy.bc import train_bc

ROOT = Path(__file__).parents[2]
OUT = ROOT / "runs" / "exp_heldout"
GAMES = sys.argv[1:] or ["gradius", "balloonfight", "battletoads"]
BASES = [("scratch", None),
         ("transfer_av", "runs/bc_base41_av"),
         ("transfer_attn", "runs/bc_base41_attn1")]
SEEDS = (1, 2, 3)

for g in GAMES:
    for seed in SEEDS:
        for name, init in BASES:
            meta = train_bc(ROOT / f"datasets/explore_{g}", OUT / f"{g}_{name}_{seed}",
                            epochs=3, use_audio=True, seed=seed,
                            max_episodes=4, init_from=init)
            print(json.dumps({
                "game": g, "variant": name, "seed": seed,
                "curve": [round(h["val_acc"], 3) for h in meta["history"]],
                "majority": round(meta["val_majority_baseline"], 3)}), flush=True)
