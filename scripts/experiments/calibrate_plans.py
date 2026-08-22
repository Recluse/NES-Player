"""Is the model blind, or just miscalibrated between its own branches?

On the decision battery the learned model picks the plan the console would pick
27% of the time against a chance rate of 17%, and in play it takes the policy's
own plan 5% of the time where the oracle takes it 72%, calling for a jump
instead in a quarter of all ticks. Blindness is one explanation. Another,
much cheaper one, is that the six action-conditioned rollouts are not on a
common scale: a constant offset on `jump now` would produce exactly this
behaviour while leaving the model's ordering *within* each branch intact.

So fit one scale and one offset per template on half the battery and score the
other half. If a straight line per branch recovers a real part of the gap, the
signal is already there and the branches simply do not agree with each other.

    uv run python scripts/experiments/calibrate_plans.py runs/ego_world_v6
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

DEFAULT_BATTERY = "runs/knowledge/decisions_SuperMarioBros-Nes-v0.npz"


def predictions(model: str, battery) -> np.ndarray:
    from decision_battery import _Hero, _buttons
    from oracle_mpc import learned_dx

    from nes_player.world_model.ego import GhostPredictor

    ghost = GhostPredictor(model)
    frames, heroes, plans = battery["frames"], battery["heroes"], battery["plans"]
    return np.array([
        [learned_dx(ghost, frames[i], _Hero(*heroes[i]),
                    [frozenset(_buttons(m)) for m in plans[i, k]])
         for k in range(plans.shape[1])]
        for i in range(len(frames))], np.float32)


def report(pred: np.ndarray, truth: np.ndarray, idx: np.ndarray) -> dict:
    pick = pred[idx].argmax(1)
    best = truth[idx].argmax(1)
    regret = truth[idx].max(1) - truth[idx][np.arange(len(idx)), pick]
    pair_ok = pair_n = 0
    for i in idx:
        for a in range(pred.shape[1]):
            for b in range(a + 1, pred.shape[1]):
                if truth[i, a] == truth[i, b]:
                    continue
                pair_n += 1
                pair_ok += (pred[i, a] > pred[i, b]) == (truth[i, a] > truth[i, b])
    return {"top1": round(float((pick == best).mean()), 3),
            "pairwise": round(pair_ok / max(pair_n, 1), 3),
            "regret_mean_px": round(float(regret.mean()), 1),
            "regret_p90_px": round(float(np.percentile(regret, 90)), 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--battery", default=DEFAULT_BATTERY)
    args = ap.parse_args()

    z = np.load(args.battery)
    truth = np.where(z["died"], -1e9, z["values"]).astype(np.float64)
    names = [str(n) for n in z["names"]]
    pred = predictions(args.model, z).astype(np.float64)

    n = len(pred)
    # Blocked, not interleaved. Points are kept in the order they were played,
    # so neighbours are seconds apart in the same stretch of level and an
    # even/odd split would fit and test on nearly the same moments.
    dev, test = np.arange(n // 2), np.arange(n // 2, n)
    fitted = pred.copy()
    coeffs = {}
    for k, name in enumerate(names):
        ok = truth[dev, k] > -1e8
        if ok.sum() < 4:
            continue
        # One line per branch: the least-squares scale and offset that map this
        # template's predicted displacement onto the console's own number.
        a, b = np.polyfit(pred[dev, k][ok], truth[dev, k][ok], 1)
        fitted[:, k] = a * pred[:, k] + b
        coeffs[name] = {"scale": round(float(a), 3), "offset": round(float(b), 1),
                        "mean_pred": round(float(pred[:, k].mean()), 1),
                        "mean_truth": round(float(truth[:, k][truth[:, k] > -1e8].mean()), 1)}

    print(json.dumps({
        "model": args.model, "points": n, "held_out": len(test),
        "raw": report(pred, truth, test),
        "calibrated": report(fitted, truth, test),
        "per_template": coeffs,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
