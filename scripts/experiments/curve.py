"""Is the crop a bad representation, or an under-fed one?

Every offline result in this project was measured on a few thousand labelled
decisions, because a label cost 9360 emulator frames. Two candidates and four
draws cost 848, which buys tens of thousands. That turns the question into one
a curve can answer: hold the test set fixed, train on more and more of the
rest, and see whether the score climbs or sits.

Two splits, because they ask different things:

* **by run** — new playthroughs of a level the model has already seen. Every
  run walks the same 1-1, so 74.6% of held-out points have a training point at
  exactly the same world x. This measures recognising a known place.
* **by world x, purged** — whole regions of the level held out, with a buffer
  around them removed from training so the visual context and the continuation
  of a training point cannot reach into the test region. This measures
  transfer to ground the model has not seen.

    uv run python scripts/experiments/curve.py runs/knowledge/cheap_all.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

BUFFER_PX = 320      # visual context plus what a 96-frame continuation covers


def label(z, tie_as_half: bool):
    """P(jump beats defer), and the tie mask, with death terminal."""
    from analyse_draws import DEATH_MARGIN

    names = [str(n) for n in z["names"]]
    bc_i, jump_i = names.index("bc"), names.index("jump now")
    r = z["returns"].astype(np.float64)
    if r.ndim == 4:
        r = r[:, 0]
    d = z["died"]
    if d.ndim == 4:
        d = d[:, 0]
    for p in range(len(r)):
        live = r[p][~d[p]]
        floor = (live.min() if live.size else 0.0) - DEATH_MARGIN
        r[p][d[p]] = floor
    tie = (r[:, jump_i] == r[:, bc_i]).all(1)
    delta = r[:, jump_i] - r[:, bc_i]                    # (P, N)
    p_jump = (delta.mean(1) > 0).astype(np.float32)
    # A decision with no consequence is not evidence against jumping. Left at
    # zero it was 45% of the training set, all of it teaching refusal.
    if tie_as_half:
        p_jump[tie] = 0.5
    return p_jump, tie, delta.mean(1)


def splits(z, kind: str, fold: int, folds: int):
    runs = z["run"]
    if kind == "run":
        uniq = np.unique(runs)
        q = max(1, len(uniq) // folds)
        te_runs = set(uniq[fold * q:(fold + 1) * q].tolist())
        te = np.array([i for i, r in enumerate(runs) if r in te_runs])
        tr = np.array([i for i, r in enumerate(runs) if r not in te_runs])
        return tr, te
    x = z["ram"][:, 0x6D].astype(int) * 256 + z["ram"][:, 0x86].astype(int)
    # Quantile edges, not equal width: runs die early, so the level's start
    # holds most of the points and equal-width blocks put 70% of the data in
    # one test fold.
    edges = np.quantile(x, np.linspace(0, 1, folds + 1))
    a, b = edges[fold], edges[fold + 1] + (1 if fold == folds - 1 else 0)
    te = np.where((x >= a) & (x < b))[0]
    # Purge a buffer on both sides: a training point whose context or whose
    # continuation reaches into the test region would put the test region's
    # geometry into the training labels.
    keep = (x < a - BUFFER_PX) | (x >= b + BUFFER_PX)
    return np.where(keep)[0], te


def train_eval(z, tr, te, y, args):
    from plan_probe import Probe

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    imgs = z["imgs"].astype(np.float32) / 255.0
    vecs = z["vecs"].astype(np.float32)
    mean, std = vecs[tr].mean(0), vecs[tr].std(0) + 1e-6
    x_img = torch.from_numpy(imgs).permute(0, 3, 1, 2)
    x_vec = torch.from_numpy((vecs - mean) / std)
    yt = torch.from_numpy(y)

    model = Probe(imgs.shape[3], vecs.shape[1], 2, use_img=True)
    with torch.no_grad():
        model(x_img[:1], x_vec[:1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    for _ in range(args.epochs):
        model.train()
        order = rng.permutation(tr)
        for k in range(0, len(order), args.batch):
            idx = torch.from_numpy(order[k:k + args.batch])
            loss = bce(model(x_img[idx], x_vec[idx])[:, 1], yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pr = torch.sigmoid(model(x_img[te], x_vec[te])[:, 1]).numpy()
    return pr


def auc(pr, pos) -> float:
    a, b = pr[pos], pr[~pos]
    if not len(a) or not len(b):
        return float("nan")
    gt = (a[:, None] > b[None, :]).astype(float)
    gt += 0.5 * (a[:, None] == b[None, :])
    return float(gt.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("matrix")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[2500, 5000, 10000, 20000, 40000])
    ap.add_argument("--split", choices=("run", "space"), default="space")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-ties", action="store_true",
                    help="label a tie 0 instead of 0.5, the old behaviour")
    args = ap.parse_args()

    z = np.load(args.matrix)
    y, tie, mean_delta = label(z, tie_as_half=not args.keep_ties)
    tr, te = splits(z, args.split, args.fold, args.folds)
    pos = mean_delta[te] > 0
    rng = np.random.default_rng(args.seed)
    print(json.dumps({"points": len(y), "ties": round(float(tie.mean()), 3),
                      "split": args.split, "fold": args.fold,
                      "train_pool": len(tr), "test": len(te),
                      "jump is right in": round(float(pos.mean()), 3)}))
    seen = set()
    for n in args.sizes:
        # Clamp rather than skip. Twice now a requested size sat just above the
        # pool and the point vanished from the curve without saying so.
        m = min(n, len(tr))
        if m in seen:
            continue
        seen.add(m)
        if m != n:
            print(f"  (asked {n}, pool has {len(tr)})")
        sub = rng.permutation(tr)[:m]
        n = m
        pr = train_eval(z, sub, te, y, args)
        print(f"  train {n:6d}   test AUC {auc(pr, pos):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
