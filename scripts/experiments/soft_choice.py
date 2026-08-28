"""Learn which candidate, not how far — and not from a single winner.

Every attempt so far regressed the value and took an argmax. The controller
only ever uses the argmax, so learn that instead. But a hard six-way label
would move the noise from the regression into the class: the teacher's winner
is itself the argmax of a few noisy means, and where candidates tie — which
is most of the time — the winner is a coin.

So the target is a distribution. From the saved draws, bootstrap says how
often each candidate would have been named best:

    p_i = P(i = argmax_j Q_j)

Near-equal candidates share the mass, a candidate that wins every resample
gets all of it, and the loss is cross-entropy against that. Points where the
teacher is confident weigh more, because a point whose label is a coin should
not be taught as though it were a fact.

Scored on held-out runs against the honest sixteen-draw mean: how often the
student's pick is the teacher's pick, how often it lands anywhere in the
top-set, and — the number that matters — the regret it leaves on the table.

    uv run python scripts/experiments/soft_choice.py \
        runs/knowledge/draw_matrix_crn.npz --save runs/soft_choice.pt
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


def soft_target(r: np.ndarray, rng, boot: int = BOOT) -> np.ndarray:
    """P(i is best), by resampling the draws that were actually taken."""
    p, c, n = r.shape
    wins = np.zeros((p, c))
    for _ in range(boot):
        idx = rng.integers(0, n, n)
        wins[np.arange(p), r[:, :, idx].mean(2).argmax(1)] += 1
    return wins / boot


def main() -> int:
    from analyse_draws import penalised, top_set
    from plan_probe import Probe, features

    ap = argparse.ArgumentParser()
    ap.add_argument("matrix")
    ap.add_argument("--variant", default="strip",
                    choices=("crop", "strip", "oam", "privileged"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hard", action="store_true",
                    help="one-hot on the teacher's winner instead of the "
                         "bootstrap distribution — the ablation that says "
                         "whether softness is what helps")
    ap.add_argument("--no-weight", action="store_true",
                    help="treat a coin-flip point as worth as much as a "
                         "decided one")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    z = np.load(args.matrix)
    names = [str(n) for n in z["names"]]
    r = penalised(z)
    truth = r.mean(2)
    err = r.std(2, ddof=1) / np.sqrt(r.shape[2])
    sets = top_set(truth, err)

    p_i = soft_target(r, rng)
    if args.hard:
        p_i = np.eye(len(names))[truth.argmax(1)]
    # A point whose label is a coin teaches a coin. Confidence is one minus the
    # normalised entropy, floored so nothing is discarded outright.
    ent = -(p_i * np.log(p_i + 1e-9)).sum(1) / np.log(len(names))
    weight = np.ones(len(p_i)) if args.no_weight else (1.0 - ent).clip(0.1, 1.0)

    imgs, vecs = features(z, args.variant)
    runs = z["run"]
    uniq = np.unique(runs)
    test_runs = set(uniq[-max(1, len(uniq) // 4):].tolist())
    te = np.array([i for i, q in enumerate(runs) if q in test_runs])
    tr = np.array([i for i, q in enumerate(runs) if q not in test_runs])
    if not len(tr) or not len(te):
        raise SystemExit("the split by run left nothing on one side")

    vec_mean, vec_std = vecs[tr].mean(0), vecs[tr].std(0) + 1e-6
    x_img = torch.from_numpy(imgs).permute(0, 3, 1, 2)
    x_vec = torch.from_numpy((vecs - vec_mean) / vec_std)
    y = torch.from_numpy(p_i.astype(np.float32))
    w = torch.from_numpy(weight.astype(np.float32))

    model = Probe(imgs.shape[3], vecs.shape[1], len(names),
                  use_img=args.variant != "privileged")
    with torch.no_grad():
        model(x_img[:1], x_vec[:1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for _ in range(args.epochs):
        model.train()
        order = rng.permutation(tr)
        for k in range(0, len(order), args.batch):
            idx = torch.from_numpy(order[k:k + args.batch])
            logp = torch.log_softmax(model(x_img[idx], x_vec[idx]), dim=1)
            loss = (-(y[idx] * logp).sum(1) * w[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pick = model(x_img[te], x_vec[te]).argmax(1).numpy()
    teacher = truth[te].argmax(1)
    regret = (truth[te].max(1) - truth[te][np.arange(len(te)), pick]).mean()
    in_set = np.mean([pick[i] in sets[j] for i, j in enumerate(te)])
    per_point = truth[te].max(1) - truth[te][np.arange(len(te)), pick]
    base = {n: (truth[te].max(1) - truth[te][:, k]).mean()
            for k, n in enumerate(names)}
    best_k = int(np.argmin([base[n] for n in names]))
    gap = (truth[te].max(1) - truth[te][:, best_k]) - per_point
    boot = np.array([rng.choice(gap, len(gap)).mean() for _ in range(10000)])
    out = {"variant": args.variant, "hard": args.hard, "train": len(tr),
           "test": len(te), "test_runs": len(test_runs),
           "agree_with_teacher": round(float((pick == teacher).mean()), 3),
           "in_top_set": round(float(in_set), 3),
           "regret_px": round(float(regret), 1),
           "teacher_entropy": round(float(ent.mean()), 3),
           "always_best_constant": min(base, key=base.get),
           "constant_regret_px": round(float(min(base.values())), 1),
           # Reported, not a gate: the pre-registered stop rule asks only
           # whether the student's regret is worse than the constant's.
           "beats_constant_by_px": round(float(gap.mean()), 2),
           "beats_constant_ci": [round(float(np.percentile(boot, 2.5)), 2),
                                 round(float(np.percentile(boot, 97.5)), 2)]}
    print(json.dumps(out, indent=2))

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "variant": args.variant,
                    "vec_mean": vec_mean, "vec_std": vec_std,
                    "in_ch": int(imgs.shape[3]), "n_vec": int(vecs.shape[1]),
                    "names": names}, args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
