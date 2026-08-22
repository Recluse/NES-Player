"""Can the observation say which of six plans wins, without a world model?

The ego model predicts a trajectory and the planner sums it. That chain has
three suspects in it — what the observation contains, what the loss rewards,
and what a 48-step recurrent rollout does to an error — and a duel cannot tell
them apart. This removes two of the three: one forward pass from one
observation straight to the six real returns the console measured. No
trajectory, no recurrence.

Then vary only the input:

    crop        the 48x48 square the ego model sees, plus velocity
    strip       + a wide band of the level ahead of the hero
    oam         + every object's position relative to him, and whether he is
                on the ground
    privileged  the console's own numbers, as an upper bound

The reading is clean. If `crop` already ranks the plans, the observation was
never the problem and the rollout or the loss was. If only `strip` does, the
crop is blind and the fix is what it looks at. If only `privileged` does, the
information is reachable but not from these pixels. If nothing does, it is the
data, the capacity or the objective.

Trained listwise rather than by regression: the returns are mostly ties with a
few expensive exceptions, and squared error on the value happily optimises a
median that is already right. Cross-entropy against softmax(G / tau), each
point weighted by how much the choice is worth, puts the gradient where the
decision is.

    uv run python scripts/experiments/plan_probe.py --inputs crop strip oam privileged
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

DEFAULT_DATA = "runs/knowledge/plan_returns_SuperMarioBros-Nes-v0.npz"
CROP = 48
STRIP_W, STRIP_H = 128, 48     # band ahead of the hero, downscaled by 4
STRIP_SCALE = 4
N_OBJ = 8                      # nearest objects handed to the `oam` variant
TAU = 20.0                     # px; softness of the target ranking
DEAD = -1e9


def _crop(frame, cx, cy, w=CROP, h=CROP, dx=0):
    fh, fw = frame.shape[:2]
    x0 = int(np.clip(cx - w // 2 + dx, 0, fw - w))
    y0 = int(np.clip(cy - h // 2, 0, fh - h))
    return frame[y0:y0 + h, x0:x0 + w]


def features(z, variant: str):
    """(images, vectors) for every point, for one input variant."""
    import cv2

    from nes_player.perception.sprites import sprite_boxes

    frames, heroes = z["frames"], z["heroes"]
    n = len(frames)
    imgs, vecs = [], []
    for i in range(n):
        cx, cy, vx, vy = heroes[i]
        parts = [_crop(frames[i], cx, cy)]
        vec = [vx, vy]
        if variant in ("strip", "oam"):
            # Where he is going, not where he is. The crop stops 24 px ahead of
            # the hero and a pit begins further out than that.
            band = _crop(frames[i], cx + STRIP_W // 2, cy, STRIP_W, STRIP_H)
            band = cv2.resize(band, (STRIP_W // STRIP_SCALE, STRIP_H // STRIP_SCALE),
                              interpolation=cv2.INTER_AREA)
            flat = np.zeros((CROP, CROP, 3), np.uint8)
            flat[:band.shape[0], :band.shape[1]] = band
            parts.append(flat)
        if variant == "oam":
            boxes = sprite_boxes(z["ram"][i])
            rel = []
            for x, y, tile in boxes[:64]:
                if x == 0 and y == 0:
                    continue
                rel.append((float(x) - cx, float(y) - cy, float(tile)))
            rel.sort(key=lambda r: abs(r[0]) + abs(r[1]))
            for k in range(N_OBJ):
                vec += list(rel[k]) if k < len(rel) else [0.0, 0.0, 0.0]
            vec.append(float(cy))          # height stands in for "on the ground"
        if variant == "privileged":
            ram = z["ram"][i]
            # The console's own account of the situation: where Mario is in the
            # level, his speeds, his state, and where every enemy is.
            vec = [float(ram[0x6D]), float(ram[0x86]), float(ram[0x03AD]),
                   float(np.int8(ram[0x0057])), float(np.int8(ram[0x009F])),
                   float(ram[0x001D]), float(ram[0x000E]), float(ram[0x0754]),
                   float(ram[0x070C]), float(ram[0x03B8])]
            for k in range(5):
                vec += [float(ram[0x000F + k]), float(ram[0x0087 + k]),
                        float(ram[0x006E + k]), float(ram[0x00CF + k])]
        imgs.append(np.concatenate(parts, axis=2) if len(parts) > 1 else parts[0])
        vecs.append(vec)
    return (np.stack(imgs).astype(np.float32) / 255.0,
            np.array(vecs, np.float32))


class Probe(nn.Module):
    """Six plan values, and — when asked — what else the outcome contains.

    A tail return is not one smooth quantity. "+80 px", "died", and "crossed
    into the next level" are different kinds of event, and asking one head to
    average over them is asking it to predict a number that never occurs.
    Separate heads let the encoder keep them apart; the controller still ranks
    on the value alone.
    """

    def __init__(self, in_ch: int, n_vec: int, n_plans: int, use_img: bool,
                 aux: bool = False):
        super().__init__()
        self.use_img = use_img
        self.aux = aux
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Flatten(), nn.LazyLinear(96), nn.ReLU(),
        ) if use_img else None
        self.trunk = nn.Sequential(
            nn.LazyLinear(128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.head = nn.Linear(128, n_plans)
        self.dead = nn.Linear(128, n_plans) if aux else None
        self.cross = nn.Linear(128, n_plans) if aux else None

    def features(self, img, vec):
        parts = [vec]
        if self.enc is not None:
            parts.insert(0, self.enc(img))
        return self.trunk(torch.cat(parts, dim=1))

    def forward(self, img, vec):
        return self.head(self.features(img, vec))

    def all_heads(self, img, vec):
        h = self.features(img, vec)
        return self.head(h), self.dead(h), self.cross(h)


def evaluate(pred: np.ndarray, truth: np.ndarray) -> dict:
    pick = pred.argmax(1)
    best = truth.argmax(1)
    regret = truth.max(1) - truth[np.arange(len(truth)), pick]
    pair_ok = pair_n = 0
    for i in range(len(pred)):
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


def run_variant(z, variant: str, args) -> dict:
    truth = np.where(z["died"], DEAD, z["values"]).astype(np.float32)
    # Death is terminal, but -1e9 cannot go into a softmax; the worst live
    # return in the point, minus a level's worth of progress, says the same
    # thing on a scale the loss can hold.
    soft = truth.copy()
    for i in range(len(soft)):
        live = soft[i][soft[i] > -1e8]
        floor = (live.min() if len(live) else 0.0) - 200.0
        soft[i][soft[i] < -1e8] = floor

    imgs, vecs = features(z, variant)
    runs = z["run"]
    uniq = np.unique(runs)
    # Split by playthrough. Neighbouring points are seconds apart in the same
    # stretch of level, so a split on rows trains and tests on one moment.
    n_test = max(1, len(uniq) // 4)
    test_runs = set(uniq[-n_test:].tolist())
    te = np.array([i for i, r in enumerate(runs) if r in test_runs])
    tr = np.array([i for i, r in enumerate(runs) if r not in test_runs])
    if not len(tr):
        # One playthrough in the file means the by-run split leaves nothing to
        # train on, and the loss comes out NaN rather than saying so.
        raise SystemExit(f"{len(uniq)} playthrough(s) in {args.data}: the "
                         "split by run leaves no training set")

    torch.manual_seed(args.seed)
    use_img = variant != "privileged"
    aux = args.aux and "crossed" in z
    model = Probe(imgs.shape[3], vecs.shape[1], truth.shape[1], use_img, aux)
    x_img = torch.from_numpy(imgs).permute(0, 3, 1, 2)
    x_vec = torch.from_numpy(vecs)
    vec_mean, vec_std = x_vec[tr].mean(0), x_vec[tr].std(0) + 1e-6
    x_vec = (x_vec - vec_mean) / vec_std
    if use_img:
        model.enc(x_img[:1])
    model.features(x_img[:1], x_vec[:1])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    y_dead = torch.from_numpy(z["died"].astype(np.float32))
    y_cross = (torch.from_numpy(z["crossed"].astype(np.float32))
               if "crossed" in z else torch.zeros_like(y_dead))
    # Ranked on the advantage over the policy's own slot rather than on six
    # absolute numbers: the state's own value is common to all six and cancels,
    # leaving only what the choice is worth.
    if args.advantage:
        soft = soft - soft[:, :1]
    target = torch.softmax(torch.from_numpy(soft) / TAU, dim=1)
    # A point where every plan pays the same teaches nothing about choosing.
    spread = torch.from_numpy(soft.max(1) - soft.min(1)).clamp(0, 300)
    weight = (spread / spread[tr].mean()).clamp(0, 8)

    rng = np.random.default_rng(args.seed)
    for _epoch in range(args.epochs):
        order = rng.permutation(tr)
        model.train()
        losses = []
        for b in range(0, len(order), args.batch):
            idx = order[b:b + args.batch]
            if aux:
                logits, dead, cross = model.all_heads(x_img[idx], x_vec[idx])
            else:
                logits, dead, cross = model(x_img[idx], x_vec[idx]), None, None
            logp = torch.log_softmax(logits, dim=1)
            loss = -(target[idx] * logp).sum(1)
            loss = (loss * weight[idx]).mean()
            if aux:
                # Not for a veto — that was measured worthless — but so the
                # trunk is allowed to represent "this one dies" as its own
                # fact instead of smearing it into the value.
                bce = nn.functional.binary_cross_entropy_with_logits
                loss = loss + 0.2 * bce(dead, y_dead[idx]) \
                            + 0.2 * bce(cross, y_cross[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
    model.eval()
    with torch.no_grad():
        pred = model(x_img[te], x_vec[te]).numpy()
    if args.save and variant == args.save_variant:
        # Everything needed to run it live: the weights, which input it was
        # built for, and the normalisation the vector half was trained under.
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "variant": variant,
                    "vec_mean": vec_mean, "vec_std": vec_std,
                    "in_ch": int(imgs.shape[3]), "n_vec": int(vecs.shape[1]),
                    "names": [str(n) for n in z["names"]]}, args.save)
    out = {"variant": variant, "train": len(tr), "test": len(te),
           "test_runs": len(test_runs), "loss": round(float(np.mean(losses)), 4)}
    out.update(evaluate(pred, truth[te]))
    return out


def test_index(z) -> np.ndarray:
    runs = z["run"]
    uniq = np.unique(runs)
    test_runs = set(uniq[-max(1, len(uniq) // 4):].tolist())
    return np.array([i for i, r in enumerate(runs) if r in test_runs])


def score_ego(z, model: str) -> dict:
    from decision_battery import _buttons, _Hero
    from oracle_mpc import learned_dx

    from nes_player.world_model.ego import GhostPredictor

    ghost = GhostPredictor(model)
    te = test_index(z)
    frames, heroes, plans = z["frames"], z["heroes"], z["plans"]
    pred = np.array([
        [learned_dx(ghost, frames[i], _Hero(*heroes[i]),
                    [frozenset(_buttons(m)) for m in plans[i, k]])
         for k in range(plans.shape[1])]
        for i in te], np.float32)
    truth = np.where(z["died"], DEAD, z["values"]).astype(np.float32)[te]
    return {"variant": f"ego {Path(model).name}", "test": len(te),
            **evaluate(pred, truth)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--inputs", nargs="+",
                    default=["crop", "strip", "oam", "privileged"])
    ap.add_argument("--hero", choices=("tracked", "ram"), default="tracked",
                    help="where the hero comes from; 'ram' is the privileged "
                         "identity ablation")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aux", action="store_true",
                    help="separate heads for dying in the tail and crossing a "
                         "level, so the value head is not asked to average "
                         "over kinds of outcome")
    ap.add_argument("--advantage", action="store_true",
                    help="rank on the advantage over the policy's slot")
    ap.add_argument("--save", default=None,
                    help="write the trained probe here, to drive a planner")
    ap.add_argument("--save-variant", default="strip",
                    help="which input variant --save writes")
    ap.add_argument("--ego", default=None,
                    help="also score this recurrent ego model on the "
                         "same held-out points")
    args = ap.parse_args()

    z = np.load(args.data)
    if args.hero == "ram":
        # The identity ablation: same points, same targets, but the hero is the
        # one the console knows rather than the one the tracker guessed. If a
        # perfect hero does not help, tracking is not what limits the probe.
        from nes_player.perception.sprites import RamHero
        z = dict(z)
        z["heroes"] = np.array([[h.cx, h.cy, h.vx, h.vy]
                                for h in map(RamHero, z["ram"])], np.float32)
    # On the held-out runs, the same points the probes are scored on. Measured
    # over everything they read as easier than they are, and the table then
    # compares two different sets.
    truth = np.where(z["died"], DEAD, z["values"])[test_index(z)]
    rows = []
    # What always picking one template would score, so "better than nothing" is
    # a number and not a feeling.
    names = [str(n) for n in z["names"]]
    for k, name in enumerate(names):
        pred = np.zeros_like(truth, np.float32)
        pred[:, k] = 1.0
        rows.append({"variant": f"always {name}", **evaluate(pred, truth)})
    for variant in args.inputs:
        rows.append(run_variant(z, variant, args))
        print(json.dumps(rows[-1]), flush=True)
    if args.ego:
        # The recurrent world model, on exactly these held-out points. Without
        # this the probe's numbers float free: they would be measured on a
        # different set from every number the ego model has ever produced.
        rows.append(score_ego(z, args.ego))
        print(json.dumps(rows[-1]), flush=True)
    print()
    print(f"{'variant':16} {'top1':>6} {'pairwise':>9} {'regret':>8} {'p90':>7}")
    for r in rows:
        print(f"{r['variant']:16} {r['top1']:6.3f} {r['pairwise']:9.3f} "
              f"{r['regret_mean_px']:8.1f} {r['regret_p90_px']:7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
