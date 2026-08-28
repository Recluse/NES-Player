"""One head, seven inputs: is the choice predictable, and from what?

Everything measured so far says the six-way choice is what matters, per-point
labels are decent (four draws rank an independent sixteen at 0.839), and the
failure is transfer. This battery holds the model and protocol fixed and
varies only the information: each variant trains the same small trunk to
regress E[return] of all six candidates (Huber), and is scored leave-one-
level-out — train on five levels, test on the sixth, macro-average.

Inputs, cumulative where marked:
    v0  airborne + velocity                      (the one-bit baseline, fair)
    v1  pixels (48x48x6 strip)                   (what we had)
    v2  geometry profile from the contact map    (privileged, exact, honest
                                                  about unknown)
    v3  v2 + phase/history (airborne, 16 actions)
    v4  v3 + dynamic objects (nearest sprites, relative, from OAM)
    v5  v4 + BC action probabilities
    v6  v5 + pixels too                          (everything)

Metrics per held-out level, then macro: mean oracle regret in px, regret
weighted by |gap|, top-1 agreement with the 4-draw oracle, and the fraction
of the oracle's per-point gain captured. AUC is deliberately secondary.

    uv run python scripts/experiments/battery.py --variant v2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

LEVELS = ["default", "Level2-1", "Level3-1", "Level4-1", "Level5-1",
          "Level6-1"]
AIRBORNE = 0x1D


def load_level(level: str):
    from analyse_draws import DEATH_MARGIN
    from geometry import load_map, profile

    z = np.load(f"runs/knowledge/battery_{level}.npz")
    r = z["returns"].astype(np.float64)[:, 0]           # (P, C, N)
    d = z["died"][:, 0]
    for p in range(len(r)):
        live = r[p][~d[p]]
        floor = (live.min() if live.size else 0.0) - DEATH_MARGIN
        r[p][d[p]] = floor
    q = r.mean(2).astype(np.float32)                    # (P, 6) target

    ram = z["ram"]
    heroes = z["heroes"].astype(np.float32)
    airborne = (ram[:, AIRBORNE] != 0).astype(np.float32)

    solid, empty = load_map(f"runs/knowledge/contact_{level}.npz")
    wx = ram[:, 0x6D].astype(int) * 256 + ram[:, 0x86].astype(int)
    sy = ram[:, 0xCE].astype(float)
    geo = np.stack([profile(solid, empty, float(wx[i]), float(sy[i]),
                            facing=1 if heroes[i, 2] >= 0 else -1)
                    for i in range(len(wx))]).reshape(len(wx), -1)

    # Nearest sprites from shadow OAM, hero-relative: (dx, dy) of the four
    # closest visible ones. Privileged, like the rest of the exact inputs.
    oam = ram[:, 0x200:0x200 + 4 * 64].reshape(len(ram), 64, 4)
    objs = np.zeros((len(ram), 8), np.float32)
    for i in range(len(ram)):
        ys, xs = oam[i, :, 0].astype(float), oam[i, :, 3].astype(float)
        vis = ys < 0xEF
        dx, dy = xs[vis] - 120.0, ys[vis] - sy[i]
        near = np.argsort(np.abs(dx) + np.abs(dy))[:4]
        for j, k in enumerate(near):
            objs[i, 2 * j:2 * j + 2] = (dx[k] / 8.0, dy[k] / 8.0)

    feats = {
        "base": np.concatenate([airborne[:, None], heroes[:, 2:4]], 1),
        "geo": geo.astype(np.float32),
        "hist": z["act_hist"].astype(np.float32) / 255.0,
        "objs": objs,
        "bc": z["bc_probs"].astype(np.float32),
    }
    return z["imgs"], feats, q


VARIANTS = {
    "v0": ([], ["base"]),
    "v1": (["imgs"], ["base"]),
    "v2": ([], ["base", "geo"]),
    "v3": ([], ["base", "geo", "hist"]),
    "v4": ([], ["base", "geo", "hist", "objs"]),
    "v5": ([], ["base", "geo", "hist", "objs", "bc"]),
    "v6": (["imgs"], ["base", "geo", "hist", "objs", "bc"]),
}


def run_variant(variant, holdout, data, args):
    from plan_probe import Probe

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
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
    imgs_te, vec_te, q_te = pack([holdout])
    mean, std = vec_tr.mean(0), vec_tr.std(0) + 1e-6
    # BC-centred advantages, not absolute returns: the absolute level is
    # place information by another name, and argmax is shift-invariant.
    a_tr = q_tr - q_tr[:, [0]]
    qm, qs = a_tr.mean(), a_tr.std() + 1e-6

    def tensors(imgs, vecs):
        xi = torch.from_numpy(imgs.astype(np.float32) / 255.0) \
            .permute(0, 3, 1, 2)
        if not use_img:
            xi = torch.zeros(len(vecs), 1, 1, 1)
        return xi, torch.from_numpy((vecs - mean) / std)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xi_tr, xv_tr = tensors(imgs_tr, vec_tr)
    xi_te, xv_te = tensors(imgs_te, vec_te)
    xi_tr, xv_tr = xi_tr.to(dev), xv_tr.to(dev)
    xi_te, xv_te = xi_te.to(dev), xv_te.to(dev)
    yt = torch.from_numpy((a_tr - qm) / qs).to(dev)

    model = Probe(xi_tr.shape[1], xv_tr.shape[1], 6, use_img=bool(use_img))
    with torch.no_grad():
        model(xi_tr[:1].cpu(), xv_tr[:1].cpu())   # materialise LazyLinear
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for _ in range(args.epochs):
        model.train()
        order = rng.permutation(len(yt))
        for k in range(0, len(order), args.batch):
            idx = torch.from_numpy(order[k:k + args.batch]).to(dev)
            loss = torch.nn.functional.huber_loss(
                model(xi_tr[idx], xv_tr[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(xi_te, xv_te).cpu().numpy() * qs + qm

    pick = pred.argmax(1)
    best = q_te.max(1)
    took = q_te[np.arange(len(q_te)), pick]
    regret = best - took
    gap = best - np.sort(q_te, 1)[:, -2]        # top-2 gap = the stake
    bc_gain = best - q_te[:, 0]
    captured = ((took - q_te[:, 0]).sum() / bc_gain.sum()
                if bc_gain.sum() > 0 else float("nan"))
    return {
        "regret_px": round(float(regret.mean()), 1),
        "regret_weighted": round(float((regret * np.abs(gap)).sum()
                                       / (np.abs(gap).sum() + 1e-9)), 1),
        "top1": round(float((pick == q_te.argmax(1)).mean()), 3),
        "captured": round(float(captured), 3),
        "n": len(q_te),
    }


def main() -> int:
    global LEVELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--levels", nargs="+", default=LEVELS)
    args = ap.parse_args()
    LEVELS = args.levels
    data = {L: load_level(L) for L in LEVELS}
    for L in LEVELS:
        print(f"# {L}: {len(data[L][2])} points", file=sys.stderr)

    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    for v in variants:
        rows = [run_variant(v, L, data, args) for L in LEVELS]
        macro = {k: round(float(np.mean([r[k] for r in rows])), 3)
                 for k in ("regret_px", "regret_weighted", "top1", "captured")}
        print(json.dumps({"variant": v, "macro": macro, "per_level": rows}),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
