"""How the probe bends the truth, per plan.

Gaussian noise the size of the probe's error costs a perfect planner nothing:
1.2 px of scatter left the oracle at 2401 against the policy's 1126, and even
4 px did no harm. So the probe's accuracy is sufficient in magnitude and its
failure must be in the shape of the error — a bias that does not average out
over hundreds of decisions the way noise does.

This fits that shape: for each of the six plans, the line that carries the
console's own return to the probe's score. Feeding those lines to the oracle
then asks the decisive question — does a perfect planner, distorted exactly the
way the probe distorts, stop working?

    uv run python scripts/experiments/fit_distortion.py runs/plan_probe_strip.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

DEFAULT_DATA = "runs/knowledge/plan_returns_SuperMarioBros-Nes-v0.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", nargs="?", default="runs/plan_probe_strip.pt")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default="runs/knowledge/probe_distortion.json")
    args = ap.parse_args()

    from plan_probe import DEAD, features
    from probe_duel import ProbePlanner

    z = np.load(args.data)
    p = ProbePlanner(args.probe)
    imgs, vecs = features(z, p.variant)
    x_img = torch.from_numpy(imgs).permute(0, 3, 1, 2)
    x_vec = (torch.from_numpy(vecs) - p.mean) / p.std
    with torch.no_grad():
        pred = np.concatenate([
            p.model(x_img[i:i + 256], x_vec[i:i + 256]).numpy()
            for i in range(0, len(x_img), 256)])
    truth = np.where(z["died"], DEAD, z["values"]).astype(np.float64)
    names = [str(n) for n in z["names"]]

    out = {"names": names, "lines": {}}
    for k, name in enumerate(names):
        ok = truth[:, k] > -1e8
        a, b = np.polyfit(truth[ok, k], pred[ok, k], 1)
        resid = pred[ok, k] - (a * truth[ok, k] + b)
        out["lines"][name] = {"a": float(a), "b": float(b),
                              "resid_std": float(resid.std())}
    # The scores are compared to each other, so only differences matter; a
    # common offset is not a distortion. Report them relative to the policy's
    # slot, which is the one the probe undervalues.
    base = out["lines"][names[0]]["b"]
    for name in names:
        out["lines"][name]["b_rel"] = out["lines"][name]["b"] - base
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
