"""One question instead of six: should the policy be overridden with a jump?

A six-way chooser needs 84% accuracy before imitation pays, and the best input
reaches 58%. A gate over {bc, jump now} has a quarter of the ceiling — +1110
against +4590 — but keeps 62% of it at a 20% error rate where the six-way arm
keeps 36%. At the accuracy actually available, the smaller question is the one
worth asking.

The target is the difference, not the pair:

    delta = Q(jump now) - Q(bc)

and the label is P(delta > 0) by bootstrap over the sixteen saved draws, so a
state where the two are level is not taught as a decision. Each point is
weighted by |E delta|, because being wrong where it does not matter should not
drive the gradient. At inference the gate jumps only when confident; anything
else defers to the policy, which the corruption grid says is the cheaper
mistake — a spurious jump costs 421 px and a missed one 250.

    uv run python scripts/experiments/gate.py \
        runs/knowledge/draw_matrix_dagger.npz --save runs/gate.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

BOOT = 200
# From the two-dimensional corruption grid: what a perfect gate is worth, and
# what it keeps at a 20% error rate in each direction, so an offline error rate
# can be turned into an expected game value instead of a shrug.
GATE_CEILING, KEEP_FP20, KEEP_FN20 = 1110.0, 0.62, 0.78


def main() -> int:
    from analyse_draws import penalised
    from plan_probe import Probe, features

    ap = argparse.ArgumentParser()
    ap.add_argument("matrix")
    ap.add_argument("--variant", default="strip",
                    choices=("crop", "strip", "oam", "privileged"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-weight", action="store_true",
                    help="drop the |delta| weighting, the ablation that says "
                         "whether it is what helps")
    ap.add_argument("--no-speed", action="store_true",
                    help="zero the hero velocity in the input — the shortcut "
                         "ablation")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--folds", type=int, default=1,
                    help="rotate the held-out quarter instead of always "
                         "taking the last one")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--knn", type=int, default=0,
                    help="also report a training-free k-NN readout on the same "
                         "features and split, and a random projection of the "
                         "same width as a control")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    z = np.load(args.matrix)
    names = [str(n) for n in z["names"]]
    bc_i, jump_i = names.index("bc"), names.index("jump now")
    r = penalised(z)
    delta = r[:, jump_i, :] - r[:, bc_i, :]          # (P, N)

    boot = np.stack([delta[:, rng.integers(0, delta.shape[1],
                                           delta.shape[1])].mean(1)
                     for _ in range(BOOT)])
    p_jump = (boot > 0).mean(0)
    mean_delta = delta.mean(1)
    weight = (np.ones(len(p_jump)) if args.no_weight
              else np.abs(mean_delta) / (np.abs(mean_delta).mean() + 1e-9))

    imgs, vecs = features(z, args.variant)
    if args.no_speed:
        vecs = vecs.copy()
        vecs[:, :2] = 0.0
    runs = z["run"]
    uniq = np.unique(runs)
    # A single fixed split is a choice, and on this data it is worth up to
    # 0.15 of train-test gap. Rotating the quarter that is held out reports
    # the spread instead of the luckiest slice of it.
    q = max(1, len(uniq) // 4)
    lo = args.fold * q
    test_runs = set(uniq[lo:lo + q].tolist()) if args.folds > 1 \
        else set(uniq[-q:].tolist())
    te = np.array([i for i, r_ in enumerate(runs) if r_ in test_runs])
    tr = np.array([i for i, r_ in enumerate(runs) if r_ not in test_runs])
    if not len(te) or not len(tr):
        raise SystemExit(f"fold {args.fold} leaves nothing on one side")
    # Does the tracker actually have the hero here? The console knows. This is
    # a stratification variable and never an input: mixing the two populations
    # is how an input ablation can look like a representation result.
    ram_cy = z["ram"][:, 0xCE].astype(np.float64) + 16.0
    hero_ok = np.abs(z["heroes"][:, 1].astype(np.float64) - ram_cy) < 8.0

    vec_mean, vec_std = vecs[tr].mean(0), vecs[tr].std(0) + 1e-6
    x_img = torch.from_numpy(imgs).permute(0, 3, 1, 2)
    x_vec = torch.from_numpy((vecs - vec_mean) / vec_std)
    y = torch.from_numpy(p_jump.astype(np.float32))
    w = torch.from_numpy(weight.astype(np.float32))

    model = Probe(imgs.shape[3], vecs.shape[1], 2,
                  use_img=args.variant != "privileged")
    with torch.no_grad():
        model(x_img[:1], x_vec[:1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    for _ in range(args.epochs):
        model.train()
        order = rng.permutation(tr)
        for k in range(0, len(order), args.batch):
            idx = torch.from_numpy(order[k:k + args.batch])
            logit = model(x_img[idx], x_vec[idx])[:, 1]
            loss = (bce(logit, y[idx], reduction="none") * w[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(x_img[te], x_vec[te])[:, 1]).numpy()
        prob_tr = torch.sigmoid(model(x_img[tr], x_vec[tr])[:, 1]).numpy()
    truth_jump = mean_delta[te] > 0
    # One number for "is there any signal at all", independent of threshold.
    # Plain AUC counts every decision alike, which flatters a model trained
    # without the |delta| weighting and punishes one trained with it for
    # ignoring points that do not matter. The weighted version scores each
    # pair by how much is at stake, and is the fair comparison between them.
    def _auc(pos_w=None):
        pos, neg = prob[truth_jump], prob[~truth_jump]
        if not len(pos) or not len(neg):
            return float("nan")
        wp = (np.ones(len(pos)) if pos_w is None else pos_w[truth_jump])
        wn = (np.ones(len(neg)) if pos_w is None else pos_w[~truth_jump])
        gt = (pos[:, None] > neg[None, :]).astype(float)
        gt += 0.5 * (pos[:, None] == neg[None, :])
        return float((gt * wp[:, None] * wn[None, :]).sum()
                     / (wp.sum() * wn.sum()))

    stake = np.abs(mean_delta[te])
    auc, auc_w = _auc(), _auc(stake)

    def _sub(m):
        pos, neg = prob[m & truth_jump], prob[m & ~truth_jump]
        if not len(pos) or not len(neg):
            return float("nan"), 0
        gt = (pos[:, None] > neg[None, :]).astype(float)
        gt += 0.5 * (pos[:, None] == neg[None, :])
        return float(gt.mean()), int(m.sum())

    ok = hero_ok[te]
    auc_ok, n_ok = _sub(ok)
    auc_lost, n_lost = _sub(~ok)

    # Train AUC separates "not enough data" from "not enough information". If
    # the network cannot even fit the states it was shown, no amount of them
    # will help and the input does not determine the label.
    def _auc_on(pr, tj):
        pos, neg = pr[tj], pr[~tj]
        if not len(pos) or not len(neg):
            return float("nan")
        gt = (pos[:, None] > neg[None, :]).astype(float)
        gt += 0.5 * (pos[:, None] == neg[None, :])
        return float(gt.mean())

    auc_train = _auc_on(prob_tr, mean_delta[tr] > 0)
    out = {"variant": args.variant, "weighted": not args.no_weight,
           "auc": round(float(auc), 3),
           "auc_weighted_by_stake": round(float(auc_w), 3),
           "auc_train": round(float(auc_train), 3),
           "fold": args.fold,
           "auc_hero_ok": round(float(auc_ok), 3), "n_hero_ok": n_ok,
           "auc_hero_lost": round(float(auc_lost), 3), "n_hero_lost": n_lost,
           "speed": not args.no_speed, "train": len(tr), "test": len(te),
           "jump is right in": round(float(truth_jump.mean()), 3),
           "thresholds": []}
    for t in args.thresholds:
        take = prob > t
        fp = float((take & ~truth_jump).sum() / max((~truth_jump).sum(), 1))
        fn = float((~take & truth_jump).sum() / max(truth_jump.sum(), 1))
        # Linear in each rate, from the measured 20% points. Crude, and stated
        # as crude: it is a way to compare offline settings on the axis that
        # matters, not a prediction of a run.
        keep = max(0.0, 1 - fp / 0.2 * (1 - KEEP_FP20)
                   - fn / 0.2 * (1 - KEEP_FN20))
        out["thresholds"].append({
            "t": t, "jumps": round(float(take.mean()), 3),
            "fp": round(fp, 3), "fn": round(fn, 3),
            "expected_px": round(keep * GATE_CEILING, 0)})
    if args.knn:
        # Training-free: does the representation put states with the same
        # answer near each other? A random projection of the same width is the
        # control that says whether any structure found is the features' or
        # just the dimension's.
        parts = [vecs]
        if args.variant != "privileged":
            parts.append(imgs.reshape(len(imgs), -1).astype(np.float32) / 255.0)
        flat = np.concatenate(parts, axis=1)
        proj = rng.normal(size=(flat.shape[1], vecs.shape[1])).astype(np.float32)
        for label, f in (("features", flat), ("random projection", flat @ proj)):
            f = (f - f[tr].mean(0)) / (f[tr].std(0) + 1e-6)
            a, b = f[te], f[tr]
            # |a-b|^2 by matmul; the broadcast form is 803 x 3597 x 13826 and
            # takes the process out with the OOM killer.
            d = ((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :]
                 - 2.0 * (a @ b.T))
            near = np.argsort(d, axis=1)[:, :args.knn]
            sc = p_jump[tr][near].mean(1)
            pos, neg = sc[truth_jump], sc[~truth_jump]
            gt = (pos[:, None] > neg[None, :]).astype(float)
            gt += 0.5 * (pos[:, None] == neg[None, :])
            out[f"knn_{label.replace(' ', '_')}"] = round(float(gt.mean()), 3)
    print(json.dumps(out, indent=2))

    if args.save:
        best = max(out["thresholds"], key=lambda d: d["expected_px"])
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "variant": args.variant,
                    "vec_mean": vec_mean, "vec_std": vec_std,
                    "in_ch": int(imgs.shape[3]), "n_vec": int(vecs.shape[1]),
                    "names": ["bc", "jump now"], "gate_threshold": best["t"],
                    "no_speed": args.no_speed}, args.save)
        print(f"saved with threshold {best['t']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
