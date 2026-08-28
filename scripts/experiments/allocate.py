"""What is the prior worth as a compute allocator, before any online run?

The battery said no static input predicts the choice past ~20% of the
oracle's gain. The surviving role for a learner is allocating the planner's
rollouts. That has different success metrics — recall of the truly best
candidate in the kept set, pruning regret, and the compute-regret frontier
against a compute-matched uniform baseline — and all of them are computable
offline from the stored per-draw returns.

Procedures simulated per decision point, all costed in rollouts (a full
search is 6 candidates x 4 draws = 24):

    full            all six, all four draws                cost 24
    uniform-2       all six, first two draws               cost 12
    prior-3         bc + prior's top-2 others, four draws  cost 12
    prior-soft      one draw for all six, then three more
                    for bc and the prior's top-2           cost 15

Selection uses only the draws the procedure paid for; regret is measured
against the full four-draw mean. The draws overlap between selection and
evaluation, so regrets are slightly optimistic for every procedure alike —
stated rather than hidden.

    uv run python scripts/experiments/allocate.py --seeds 0 1 2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from battery import LEVELS, VARIANTS, load_level  # noqa: E402

SUBSTANTIAL = 16.0        # px by which a pruned candidate must win to count


def train_prior(variant, holdout, data, seed, epochs, batch, lr):
    """The battery's trainer, returning held-out predictions."""
    from plan_probe import Probe

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    use_img, keys = VARIANTS[variant]

    def pack(levels):
        imgs = np.concatenate([data[L][0] for L in levels])
        vecs = np.concatenate(
            [np.concatenate([data[L][1][k] for k in keys], 1)
             for L in levels])
        q = np.concatenate([data[L][2] for L in levels])
        return imgs, vecs, q

    tr_lv = [L for L in LEVELS if L != holdout]
    imgs_tr, vec_tr, q_tr = pack(tr_lv)
    imgs_te, vec_te, _ = pack([holdout])
    mean, std = vec_tr.mean(0), vec_tr.std(0) + 1e-6
    a_tr = q_tr - q_tr[:, [0]]
    qm, qs = a_tr.mean(), a_tr.std() + 1e-6

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tensors(imgs, vecs):
        xi = torch.from_numpy(imgs.astype(np.float32) / 255.0) \
            .permute(0, 3, 1, 2)
        if not use_img:
            xi = torch.zeros(len(vecs), 1, 1, 1)
        return xi.to(dev), torch.from_numpy((vecs - mean) / std).to(dev)

    xi_tr, xv_tr = tensors(imgs_tr, vec_tr)
    xi_te, xv_te = tensors(imgs_te, vec_te)
    yt = torch.from_numpy((a_tr - qm) / qs).to(dev)
    model = Probe(xi_tr.shape[1], xv_tr.shape[1], 6, use_img=bool(use_img))
    with torch.no_grad():
        model(xi_tr[:1].cpu(), xv_tr[:1].cpu())
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(yt))
        for k in range(0, len(order), batch):
            idx = torch.from_numpy(order[k:k + batch]).to(dev)
            loss = torch.nn.functional.huber_loss(
                model(xi_tr[idx], xv_tr[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(xi_te, xv_te).cpu().numpy()


def evaluate(pred, draws):
    """All procedures on one held-out level. draws: (P, 6, 4) penalised."""
    q = draws.mean(2)                            # the reference value
    best = q.max(1)
    n = len(q)

    def regret(pick):
        return float((best - q[np.arange(n), pick]).mean())

    out = {}
    out["full"] = {"cost": 24, "regret": regret(q.argmax(1))}
    u2 = draws[:, :, :2].mean(2)
    out["uniform-2"] = {"cost": 12, "regret": regret(u2.argmax(1))}

    # bc plus the prior's top-2 others, full draws on the kept three.
    order = np.argsort(-pred[:, 1:], 1) + 1
    kept = np.concatenate([np.zeros((n, 1), int), order[:, :2]], 1)
    kq = np.take_along_axis(q, kept, 1)
    pick3 = np.take_along_axis(kept, kq.argmax(1)[:, None], 1)[:, 0]
    out["prior-3"] = {"cost": 12, "regret": regret(pick3)}

    # soft: one draw for everyone, three more for bc and the prior's top-2.
    one = draws[:, :, 0]
    soft = one.copy()
    rows = np.arange(n)[:, None]
    soft[rows, kept] = np.take_along_axis(draws, kept[:, :, None], 1).mean(2)
    out["prior-soft"] = {"cost": 15, "regret": regret(soft.argmax(1))}

    truebest = q.argmax(1)
    in3 = (kept == truebest[:, None]).any(1)
    out["recall"] = {
        "top3_recall": float(in3.mean()),
        "pruned_wins_substantially": float(
            ((~in3) & (best - np.take_along_axis(q, kept, 1).max(1)
                       > SUBSTANTIAL)).mean()),
        "prune_regret": float((best - np.take_along_axis(q, kept, 1)
                               .max(1)).mean()),
    }
    return out


def main() -> int:
    from analyse_draws import DEATH_MARGIN

    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v5")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    data = {L: load_level(L) for L in LEVELS}
    raw = {}
    for L in LEVELS:
        z = np.load(f"runs/knowledge/battery_{L}.npz")
        r = z["returns"].astype(np.float64)[:, 0]
        d = z["died"][:, 0]
        for p in range(len(r)):
            live = r[p][~d[p]]
            floor = (live.min() if live.size else 0.0) - DEATH_MARGIN
            r[p][d[p]] = floor
        raw[L] = r

    for seed in args.seeds:
        rows = {}
        for L in LEVELS:
            pred = train_prior(args.variant, L, data, seed,
                               args.epochs, args.batch, args.lr)
            rows[L] = evaluate(pred, raw[L])
        macro = {}
        for proc in ("full", "uniform-2", "prior-3", "prior-soft"):
            macro[proc] = {
                "cost": rows[LEVELS[0]][proc]["cost"],
                "regret": round(float(np.mean(
                    [rows[L][proc]["regret"] for L in LEVELS])), 2),
            }
        macro["recall"] = {k: round(float(np.mean(
            [rows[L]["recall"][k] for L in LEVELS])), 3)
            for k in rows[LEVELS[0]]["recall"]}
        print(json.dumps({"seed": seed, "variant": args.variant,
                          "macro": macro,
                          "per_level": {L: rows[L] for L in LEVELS}}),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
