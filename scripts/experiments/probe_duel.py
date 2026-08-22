"""Let the plan-value probe drive, and see whether the game agrees.

The probe ranks the six candidate plans from one observation, with mean regret
2.8 px against the recurrent world model's 6.1 and 7.4 for always doing what
the policy says. This is the same question asked of the game.

It needs no emulator branching, unlike every arm before it. The policy's plan
is a *slot* the probe was trained to value — "how far does following the policy
get me" — not a sequence that has to be simulated first. When the probe picks
that slot, the policy simply keeps the wheel for the commitment window; when it
picks a template, the template is executed. So this arm is a controller that
could actually be shipped, not a measurement device.

    uv run python scripts/experiments/probe_duel.py runs/bc_smb_new \
        --probe runs/plan_probe_strip.pt --runs 8 --seed0 200
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

IDLE_STEP, IDLE_MAX = 37, 1800


class ProbePlanner:
    """One probe, or several — `path` may be a comma-separated list.

    With several, `rank` returns their mean and `agreement` says what fraction
    of them named the same best candidate. A scorer that is wrong 87% of the
    way to the ceiling is still driving every decision; the members disagreeing
    is the cheapest available signal for when it should not be.
    """

    def __init__(self, path: str):
        from plan_probe import Probe

        paths = [p for p in path.split(",") if p]
        self.members = ([ProbePlanner(p) for p in paths]
                        if len(paths) > 1 else [])
        if self.members:
            self.variant = self.members[0].variant
            self.names = self.members[0].names
            self.agreement = 1.0
            return
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.variant = ck["variant"]
        self.names = ck["names"]
        self.mean, self.std = ck["vec_mean"], ck["vec_std"]
        self.model = Probe(ck["in_ch"], ck["n_vec"], len(ck["names"]),
                           use_img=self.variant != "privileged",
                           # A probe trained with --aux carries two more heads.
                           # The controller never reads them, but the weights
                           # have to have somewhere to land.
                           aux="dead.weight" in ck["state_dict"])
        # LazyLinear needs one pass before the weights exist to load into.
        with torch.no_grad():
            img = torch.zeros(1, ck["in_ch"], 48, 48)
            self.model(img, torch.zeros(1, ck["n_vec"]))
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()

    def rank(self, frame_rgb, hero, ram=None) -> np.ndarray:
        from plan_probe import features

        if self.members:
            each = [m.rank(frame_rgb, hero, ram) for m in self.members]
            mean = np.mean(each, axis=0)
            top = int(mean.argmax())
            self.agreement = float(np.mean([int(r.argmax()) == top
                                            for r in each]))
            return mean
        z = {"frames": frame_rgb[None], "ram": None if ram is None else ram[None],
             "heroes": np.array([[hero.cx, hero.cy, hero.vx, hero.vy]], np.float32)}
        imgs, vecs = features(z, self.variant)
        x_img = torch.from_numpy(imgs).permute(0, 3, 1, 2)
        x_vec = (torch.from_numpy(vecs) - self.mean) / self.std
        with torch.no_grad():
            return self.model(x_img, x_vec)[0].numpy()


AUDIT_HORIZON = 48    # what the probe was trained to value


def _branch_deaths(env, policy, obs, temperature, repeat, commit,
                   causal: bool):
    """Which of the six plans kills him, and how far each one gets.

    Two readings of "fatal", because the controller does not execute what it
    scores. `branch_death_48` holds the plan for the whole horizon, which is
    what the plan value means. `causal_death` executes only the commitment and
    then hands back to the policy, which is what the *decision* actually
    causes — and the two are far apart: 12.5 fatal labels a run against 5 real
    deaths.
    """
    from oracle_mpc import mario_x, templates

    here = env.save_state()
    x0, l0 = mario_x(env), (obs.debug or {}).get("lives")
    stack = list(policy._stack)

    seq, o = [], obs
    for k in range(AUDIT_HORIZON):
        if k % repeat == 0:
            p, _ = policy.act(o.frame_rgb, temperature)
            p = p - {"START", "SELECT"}
        seq.append(p)
        o = env.step_buttons([p])
    policy._stack = list(stack)

    vals, dead, when = [], [], []
    for _, plan in [("bc", seq), *templates(AUDIT_HORIZON)]:
        env.load_state(here)
        policy._stack = list(stack)
        died_at = None
        ob, last = obs, plan[0]
        for k in range(AUDIT_HORIZON):
            # After the commitment the plan is over; the controller would have
            # replanned, and the honest stand-in for "whatever it does next" is
            # the policy it falls back on.
            press = plan[k]
            if causal and k >= commit:
                if k % repeat == 0:
                    p, _ = policy.act(ob.frame_rgb, temperature)
                    press = p - {"START", "SELECT"}
                else:
                    press = last
            last = press
            ob = env.step_buttons([press])
            now = (ob.debug or {}).get("lives")
            if died_at is None and l0 is not None and now is not None and now < l0:
                died_at = k
                if not causal:
                    break
        vals.append(-1e9 if died_at is not None else float(mario_x(env) - x0))
        dead.append(died_at is not None)
        when.append(died_at)
    env.load_state(here)
    policy._stack = list(stack)
    return vals, dead, when


def _audit_point(env, policy, obs, pick, scores, temperature, repeat, commit,
                 causal, at, bag=None, hero=None, ram=None, seed=0):
    """What the choice was worth here, and everything about it worth knowing."""
    from oracle_mpc import mario_x

    vals, dead, when = _branch_deaths(env, policy, obs, temperature, repeat,
                                      commit, causal)
    if bag is not None and hero is not None:
        # The same rows decision_battery writes, but gathered where the probe
        # itself drives. Its errors are rare and concentrated in the states
        # just before trouble, and those are exactly the states the policy
        # never visits — so no amount of the original data contains them.
        from decision_battery import HORIZON, _masks
        from oracle_mpc import templates

        from nes_player.emulator.controller import BUTTONS

        seq, o = [], obs
        stack = list(policy._stack)
        here = env.save_state()
        for k in range(HORIZON):
            if k % repeat == 0:
                p, _ = policy.act(o.frame_rgb, temperature)
                p = p - {"START", "SELECT"}
            seq.append(p)
            o = env.step_buttons([p])
        policy._stack = list(stack)
        env.load_state(here)
        bag["frames"].append(obs.frame_rgb.copy())
        bag["heroes"].append((hero.cx, hero.cy, hero.vx, hero.vy))
        bag["plans"].append(np.stack(
            [_masks(seq, BUTTONS)]
            + [_masks(pl, BUTTONS) for _, pl in templates(HORIZON)]))
        bag["values"].append([0.0 if v < -1e8 else v for v in vals])
        bag["died"].append(list(dead))
        bag["run"].append(seed)
        bag["ram"].append(ram.copy())
    best = max(vals)
    took = vals[pick]
    return {"pick": int(pick), "at": at,
            "regret": float(max(best - took, 0.0)) if took > -1e8 else None,
            "best_is_bc": int(vals.index(best) == 0),
            "died": int(dead[pick]), "died_at": when[pick],
            "bc_safe": int(not dead[0]),
            "n_dead": int(sum(dead)),
            "unavoidable": int(all(dead)),
            "margin_over_bc": float(scores[pick] - scores[0]),
            "x": mario_x(env)}


def run(checkpoint: str, probe_path: str | None, game: str, state: str | None,
        frames: int, seed: int, temperature: float, repeat: int,
        commit: int, audit: list | None = None,
        veto: str = "", causal: bool = False,
        bag: dict | None = None) -> dict:
    from oracle_mpc import templates

    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.perception.motion import pick_hero
    from nes_player.perception.sprites import SpriteTracker, sprite_boxes
    from nes_player.policy.bc import BCPolicy
    from nes_player.policy.go_explore import _begin
    from nes_player.policy.robustify import progress_of

    np.random.seed(seed)
    env = StableRetroAdapter(game, include_debug=True, state=state)
    policy = BCPolicy(checkpoint)
    obs = _begin(env)
    for _ in range(IDLE_STEP * seed % IDLE_MAX):
        obs = env.step_buttons([frozenset()])

    probe = ProbePlanner(probe_path) if probe_path else None
    tracker = SpriteTracker() if probe else None
    # The prefix of the plan that was scored, not a plan rebuilt at the
    # commitment length. `templates(16)` is not `templates(48)[:16]`: at 16 the
    # "jump later" recipe becomes twelve frames of running and ten of jumping,
    # 22 frames long, where the plan the probe valued has only four frames of
    # jump inside its first sixteen. That template is the one the probe picks
    # most often, so the controller was executing something the score never
    # described.
    plans = {n: pl[:commit] for n, pl in templates(AUDIT_HORIZON)} if probe else {}
    chosen: Counter = Counter()
    best_x, deaths = 0, 0
    lives = (obs.debug or {}).get("lives")
    held: list = []
    pressed: frozenset = frozenset()
    defer = 0

    for i in range(frames):
        if probe is not None:
            ram = env._env.get_ram()
            hero = pick_hero(tracker.update(obs.frame_rgb, pressed, boxes=sprite_boxes(ram)))
            if not held and defer <= 0 and hero is not None:
                scores = probe.rank(obs.frame_rgb, hero, ram)
                if veto:
                    # A perfect safety head, borrowed from the console. The
                    # probe still scores; the emulator only says which plans
                    # are off the table. Same trick as the oracle planner:
                    # substitute the ideal component before building it.
                    _, dead, _ = _branch_deaths(env, policy, obs, temperature,
                                                repeat, commit, causal)
                    safe = [k for k in range(len(scores)) if not dead[k]]
                    if not safe:
                        pick = 0 if veto == "bc" else int(scores.argmax())
                    elif dead[int(scores.argmax())]:
                        pick = 0 if (veto == "bc" and not dead[0]) else max(
                            safe, key=lambda k: scores[k])
                    else:
                        pick = int(scores.argmax())
                else:
                    pick = int(scores.argmax())
                name = probe.names[pick]
                chosen[name] += 1
                if audit is not None:
                    # What that choice was really worth, here, on the
                    # trajectory the probe itself is producing. Offline it is
                    # measured on states the policy visits; once the probe
                    # drives, those are not the same states.
                    audit.append(_audit_point(env, policy, obs, pick, scores,
                                              temperature, repeat, commit,
                                              causal, i, bag, hero, ram, seed))
                if name == "bc":
                    defer = commit        # the policy keeps the wheel
                else:
                    held = list(plans[name])
        if held:
            pressed = held.pop(0)
        else:
            defer = max(0, defer - 1)
            if i % repeat == 0:
                pressed, _ = policy.act(obs.frame_rgb, temperature)
                pressed = pressed - {"START", "SELECT"}
        obs = env.step_buttons([pressed])

        d = obs.debug or {}
        now = d.get("lives")
        if lives is not None and now is not None and now < lives:
            deaths += 1
        lives = now
        # Across level boundaries, not within one. The camera's x restarts at
        # zero in a new level, so a longer window measured on xscroll alone
        # scores a run *lower* the moment it finishes 1-1.
        best_x = max(best_x, progress_of(d))
    env.close()
    return {"seed": seed, "best_x": best_x, "deaths": deaths,
            "chosen": dict(chosen)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--probe", default="runs/plan_probe_strip.pt")
    ap.add_argument("--game", default="SuperMarioBros-Nes-v0")
    ap.add_argument("--state", default="default")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=200)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--commit", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--veto", choices=("", "next", "bc"), default="",
                    help="oracle safety head: mask the plans the console says "
                         "are fatal, then take the probe's best survivor "
                         "('next') or fall back to the policy ('bc')")
    ap.add_argument("--causal", action="store_true",
                    help="call a plan fatal by its consequence — execute the "
                         "commitment, hand back to the policy, then look — "
                         "rather than by holding it for the whole horizon")
    ap.add_argument("--collect", default=None,
                    help="write the audited decisions as a training set, in "
                         "decision_battery's format (implies --audit)")
    ap.add_argument("--audit", action="store_true",
                    help="oracle-evaluate every decision the probe makes, on "
                         "the trajectory the probe itself produced")
    args = ap.parse_args()

    if args.collect:
        args.audit = True
    bag = {k: [] for k in ("frames", "heroes", "plans", "values", "died",
                           "run", "ram")} if args.collect else None
    label = "bc+probe" + (f"+veto:{args.veto}" if args.veto else "")
    arms = {"bc": None, label: args.probe}
    out: dict[str, list[dict]] = {}
    audit: list = []
    for name, probe in arms.items():
        rows = []
        for seed in range(args.seed0, args.seed0 + args.runs):
            rows.append(run(args.checkpoint, probe, args.game, args.state,
                            args.frames, seed, args.temperature, args.repeat,
                            args.commit,
                            audit if (probe and args.audit) else None,
                            args.veto if probe else "", args.causal,
                            bag if probe else None))
            print(json.dumps({"arm": name, **rows[-1]}), flush=True)
        out[name] = rows

    if bag and bag["frames"]:
        from decision_battery import HORIZON

        p = Path(args.collect)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p, frames=np.stack(bag["frames"]),
            heroes=np.array(bag["heroes"], np.float32),
            plans=np.stack(bag["plans"]),
            values=np.array(bag["values"], np.float32),
            died=np.array(bag["died"], bool), run=np.array(bag["run"], np.int32),
            ram=np.stack(bag["ram"]),
            names=np.array(["bc", "run", "jump now", "jump later", "wait",
                            "back off"]))
        print(f"wrote {len(bag['frames'])} decisions to {p} "
              f"(horizon {HORIZON})", flush=True)

    a = np.array([r["best_x"] for r in out[label]], float)
    b = np.array([r["best_x"] for r in out["bc"]], float)
    d = a - b
    sd = d.std(ddof=1)
    t = d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else 0.0
    print()
    for name, rows in out.items():
        v = np.array([r["best_x"] for r in rows], float)
        print(f"{name:10} mean {v.mean():7.1f}  median {np.median(v):7.1f}  "
              f"min {v.min():6.0f}  deaths {sum(r['deaths'] for r in rows):3d}")
    print(f"diff {d.mean():+8.1f}  t={t:+.2f}  "
          f"{int((d > 0).sum())}W/{int((d < 0).sum())}L")
    if audit:
        reg = np.array([a["regret"] for a in audit if a["regret"] is not None])
        died = sum(a["died"] for a in audit)
        bc_best = sum(a["best_is_bc"] for a in audit)
        took_bc = sum(a["pick"] == 0 for a in audit)
        print()
        print(f"decisions audited: {len(audit)}   on the probe's own states")
        print(f"  regret  mean {reg.mean():6.2f} px   median "
              f"{np.median(reg):6.2f}   p90 {np.percentile(reg, 90):6.1f}")
        for th in (16, 32, 64):
            print(f"  P(regret > {th:2d} px) {(reg > th).mean():5.1%}")
        worst = np.sort(reg)[-max(1, len(reg) // 20):]
        print(f"  CVaR of the worst 5%: {worst.mean():.1f} px")
        print(f"  the policy's plan was best in {bc_best / len(audit):.0%} of "
              f"them; the probe took it {took_bc / len(audit):.0%}")
        print(f"  chose a plan that dies: {died}/{len(audit)}")
        fatal = [a for a in audit if a["died"]]
        if fatal:
            avoid = [a for a in fatal if not a["unavoidable"]]
            # Consecutive fatal decisions looking at the same approaching death
            # are one event, not several; counting them separately would let
            # one future corpse supply a handful of correlated positives.
            events, last = 0, -999
            for a in sorted(fatal, key=lambda r: r["at"]):
                if a["at"] - last > 48:
                    events += 1
                last = a["at"]
            print(f"  of those: {len(avoid)} avoidable, "
                  f"{len(fatal) - len(avoid)} with every plan fatal")
            print(f"  they represent about {events} distinct deaths")
            print(f"  the policy's slot was safe in "
                  f"{sum(a['bc_safe'] for a in avoid)}/{len(avoid)} avoidable ones")
            names = ["bc", "run", "jump now", "jump later", "wait", "back off"]
            print("  template chosen: " + ", ".join(
                f"{names[k]} {v}" for k, v in Counter(
                    a["pick"] for a in fatal).most_common()))
            m = np.array([a["margin_over_bc"] for a in fatal])
            print(f"  margin over the policy's score: median {np.median(m):+.3f}")
            w = [a["died_at"] for a in fatal if a["died_at"] is not None]
            if w:
                print(f"  frames to death: median {int(np.median(w))}, "
                      f"<=16 in {sum(x <= 16 for x in w)}, "
                      f"<=32 in {sum(x <= 32 for x in w)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
