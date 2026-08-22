"""A fixed set of moments where the choice of action actually matters.

Judging a world model by a duel costs twenty minutes and answers with a number
whose noise is larger than most of the effects being tested — the policy's own
range across seeds is 563 to 1550. Judging it by prediction error is quick and
measures the wrong thing: accuracy on (dx, dy) improved by a factor of four
while play did not move at all.

So build the benchmark out of the question the planner actually asks. Play,
and at moments where the candidate plans genuinely diverge — a pit, an enemy,
the foot of a pipe — record the frame, the hero, and what the console says
each plan is really worth. Once built, no emulator is needed: any model can be
scored offline in seconds on

* top-1 — how often it picks the plan the console would have picked
* pairwise — how often it orders two plans the way the console does
* regret — how many pixels of real progress its choice gives up
* deaths chosen — how often it picks a plan that actually dies

A point is only kept when the outcomes spread wide enough for the choice to
matter, or when one of the plans dies. Everywhere else any model looks good and
nothing is learned.

    uv run python scripts/experiments/decision_battery.py build --points 200
    uv run python scripts/experiments/decision_battery.py score runs/ego_world_v6
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

DEFAULT_PATH = "runs/knowledge/decisions_SuperMarioBros-Nes-v0.npz"
HORIZON = 48
SPREAD_MIN = 24.0     # px between best and worst plan; below this, nothing is at stake
PROBE_EVERY = 24


def _masks(plan, buttons) -> np.ndarray:
    return np.array([sum(1 << k for k, b in enumerate(buttons) if b in p)
                     for p in plan], np.int64)


def build(args) -> int:
    from oracle_mpc import templates

    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.perception.sprites import SpriteTracker
    from nes_player.policy.bc import BCPolicy
    from nes_player.policy.go_explore import _begin

    env = StableRetroAdapter(args.game, include_debug=True, state=args.state)
    policy = BCPolicy(args.checkpoint)
    driver = None
    if args.driver:
        from probe_duel import ProbePlanner
        driver = ProbePlanner(args.driver)
    cands = templates(HORIZON)
    names = ["bc", *[n for n, _ in cands]]

    frames, heroes, plans, values, dies, runs, rams = [], [], [], [], [], [], []
    crosses: list = []
    seed = args.seed - 1
    i = 0
    while len(frames) < args.points and i < args.max_frames:
        # A fresh playthrough per run id. The probe's train/test split has to
        # be by run: consecutive points are seconds apart in the same stretch
        # of level, so splitting on neighbouring frames would train and test on
        # very nearly the same moments.
        seed += 1
        np.random.seed(seed)
        policy.reset()
        tracker = SpriteTracker()
        obs = _begin(env)
        pressed: frozenset = frozenset()
        for _ in range(37 * seed % 1800):
            obs = env.step_buttons([frozenset()])
        i = _play_one(args, env, policy, tracker, obs, pressed, cands, seed, i,
                      frames, heroes, plans, values, dies, runs, rams, crosses,
                      driver)
    env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, frames=np.stack(frames), heroes=np.array(heroes, np.float32),
        plans=np.stack(plans), values=np.array(values, np.float32),
        died=np.array(dies, bool), names=np.array(names),
        run=np.array(runs, np.int32), ram=np.stack(rams),
        crossed=np.array(crosses, bool))
    print(json.dumps({
        "path": str(out), "points": len(frames), "frames_played": i,
        "runs": len(set(runs)), "candidates": names, "horizon": HORIZON,
        "points_with_a_death": int(np.array(dies).any(1).sum()),
        "kept_everything": bool(args.keep_all),
    }, indent=2))
    return 0


def _play_one(args, env, policy, tracker, obs, pressed, cands, seed, i,
              frames, heroes, plans, values, dies, runs, rams, crosses,
              driver=None) -> int:
    """One playthrough, appending the points worth keeping. Returns the frame count."""
    from oracle_mpc import mario_x

    from nes_player.emulator.controller import BUTTONS
    from nes_player.perception.feedback import game_over
    from nes_player.perception.motion import pick_hero
    from nes_player.perception.sprites import sprite_boxes
    from nes_player.policy.robustify import level_of, progress_of

    end = i + args.frames_per_run
    held: list = []
    while len(frames) < args.points and i < min(end, args.max_frames):
        i += 1
        hero = pick_hero(tracker.update(obs.frame_rgb, pressed,
                                        boxes=sprite_boxes(env._env.get_ram())))
        if i % PROBE_EVERY == 0 and hero is not None:
            here = env.save_state()
            x0, l0 = mario_x(env), (obs.debug or {}).get("lives")
            stack = list(policy._stack)

            seq, o = [], obs
            for k in range(HORIZON):
                if k % args.repeat == 0:
                    p, _ = policy.act(o.frame_rgb, args.temperature)
                    p = p - {"START", "SELECT"}
                seq.append(p)
                o = env.step_buttons([p])
            policy._stack = list(stack)

            vals, died_flags, plan_masks, crossed = [], [], [], []
            p0 = progress_of(obs.debug or {})
            stack0 = list(policy._stack)
            for _, plan in [("bc", seq), *cands]:
                env.load_state(here)
                policy._stack = list(stack0)
                died = False
                span = args.commit if args.tail else HORIZON
                ob = obs
                for k in range(span):
                    ob = env.step_buttons([plan[k]])
                    now = (ob.debug or {}).get("lives")
                    if l0 is not None and now is not None and now < l0:
                        died = True
                        break
                if args.tail and not died:
                    # What the *decision* causes, not what an unexecuted
                    # template would have scored: the committed prefix, then
                    # the continuation. One draw of a sampling policy is a
                    # future rather than a value — scoring the same plan twice
                    # at 0.9 differed by as much as scoring two different ones
                    # — and averaging several is what makes the label a
                    # function of the state. An oracle scored this way reaches
                    # a median of 7117 against 3121 on a single draw.
                    mid = env.save_state()
                    reached, crossings, dead_n = [], 0, 0
                    for _ in range(args.tail_draws):
                        env.load_state(mid)
                        policy._stack = list(stack0)
                        press: frozenset = frozenset()
                        ob2, gone = ob, False
                        for k in range(args.tail):
                            if k % args.repeat == 0:
                                pr, _ = policy.act(ob2.frame_rgb, args.tail_temp)
                                press = pr - {"START", "SELECT"}
                            ob2 = env.step_buttons([press])
                            now = (ob2.debug or {}).get("lives")
                            if l0 is not None and now is not None and now < l0:
                                gone = True
                                break
                        if gone:
                            dead_n += 1
                            continue
                        d2 = ob2.debug or {}
                        reached.append(float(progress_of(d2) - p0))
                        crossings += level_of(d2) > level_of(obs.debug or {})
                    died = dead_n * 2 > args.tail_draws
                    val = 0.0 if died else float(np.mean(reached))
                    cross = bool(reached) and crossings * 2 > len(reached)
                else:
                    d1 = ob.debug or {}
                    val = (float(progress_of(d1) - p0) if args.tail
                           else float(mario_x(env) - x0))
                    cross = level_of(d1) > level_of(obs.debug or {})
                vals.append(val)
                crossed.append(cross)
                died_flags.append(died)
                plan_masks.append(_masks(plan, BUTTONS))
            policy._stack = list(stack0)
            env.load_state(here)

            alive = [v for v, d in zip(vals, died_flags, strict=True) if not d]
            # A death only counts as a decision if it was avoidable. The first
            # battery admitted any point with a death in it, and at all 28 of
            # them every one of the six plans died — Mario was already doomed
            # when the branch was taken, so the point says nothing about
            # choosing, and it is a fact about the 48-frame horizon rather than
            # about any model.
            avoidable_death = any(died_flags) and not all(died_flags)
            worth_asking = args.keep_all or avoidable_death or (
                len(alive) > 1 and max(alive) - min(alive) >= SPREAD_MIN)
            if worth_asking:
                frames.append(obs.frame_rgb.copy())
                heroes.append((hero.cx, hero.cy, hero.vx, hero.vy))
                plans.append(np.stack(plan_masks))
                values.append(vals)
                dies.append(died_flags)
                crosses.append(crossed)
                runs.append(seed)
                # Console memory alongside the pixels. Nothing the student
                # sees, but it lets an ablation ask what a probe could do
                # with a perfect description of the state — the upper
                # bound that separates 'cannot be seen' from 'not learned'.
                rams.append(env._env.get_ram().copy())

        if driver is not None and not held and hero is not None:
            # DAgger: the states worth labelling next are the ones this probe
            # takes the game into, and no amount of the policy's own play
            # contains them. Same commitment and same candidates as the
            # deployed arm, so the trajectory is the one being measured.
            here = env.save_state()
            stack = list(policy._stack)
            seq, o = [], obs
            for k in range(HORIZON):
                if k % args.repeat == 0:
                    p, _ = policy.act(o.frame_rgb, args.temperature)
                    p = p - {"START", "SELECT"}
                seq.append(p)
                o = env.step_buttons([p])
            policy._stack = list(stack)
            env.load_state(here)
            ranks = driver.rank(obs.frame_rgb, hero, env._env.get_ram())
            pick = int(np.argmax(ranks))
            held = list((seq if pick == 0 else cands[pick - 1][1])[:args.commit])

        if held:
            pressed = held.pop(0)
        elif i % args.repeat == 0:
            pressed, _ = policy.act(obs.frame_rgb, args.temperature)
            pressed = pressed - {"START", "SELECT"}
        obs = env.step_buttons([pressed])
        if game_over(obs.debug or {}):
            return i        # the run is over; the caller starts a fresh one
    return i


class _Hero:
    def __init__(self, cx, cy, vx, vy):
        self.cx, self.cy, self.vx, self.vy = cx, cy, vx, vy


def score(args) -> int:
    from oracle_mpc import learned_dx

    from nes_player.world_model.ego import GhostPredictor

    z = np.load(args.battery)
    frames, heroes = z["frames"], z["heroes"]
    plans, values, died = z["plans"], z["values"], z["died"]
    names = [str(n) for n in z["names"]]
    ghost = GhostPredictor(args.model)
    # A plan that dies is worth less than any plan that does not, however far
    # it got before dying: death is terminal, not a discount.
    truth = np.where(died, -1e9, values)

    top1 = pair_ok = pair_n = 0
    regrets, chose_death = [], 0
    for i in range(len(frames)):
        hero = _Hero(*heroes[i])
        pred = np.array([
            learned_dx(ghost, frames[i], hero,
                       [frozenset(_buttons(m)) for m in plans[i, k]])
            for k in range(plans.shape[1])])
        pick = int(pred.argmax())
        best = int(truth[i].argmax())
        top1 += pick == best
        chose_death += bool(died[i, pick])
        regrets.append(float(max(truth[i].max() - truth[i][pick], 0.0)
                             if truth[i][pick] > -1e8 else np.nan))
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                if truth[i, a] == truth[i, b]:
                    continue
                pair_n += 1
                pair_ok += (pred[a] > pred[b]) == (truth[i, a] > truth[i, b])

    n = len(frames)
    finite = [r for r in regrets if not np.isnan(r)]
    print(json.dumps({
        "model": args.model, "points": n,
        "top1": round(top1 / n, 3), "chance": round(1 / len(names), 3),
        "pairwise": round(pair_ok / max(pair_n, 1), 3),
        "regret_px_median": round(float(np.median(finite)), 1) if finite else None,
        "regret_px_mean": round(float(np.mean(finite)), 1) if finite else None,
        "picked_a_plan_that_dies": f"{chose_death}/{n}",
        "deaths_available": f"{int(died.any(1).sum())}/{n}",
    }, indent=2))
    return 0


def _buttons(mask: int):
    from nes_player.emulator.controller import BUTTONS

    return {b for k, b in enumerate(BUTTONS) if mask >> k & 1}


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)

    b = sub.add_parser("build", help="play, and keep the moments that matter")
    b.add_argument("--checkpoint", default="runs/bc_smb_new")
    b.add_argument("--game", default="SuperMarioBros-Nes-v0")
    b.add_argument("--state", default="default")
    b.add_argument("--points", type=int, default=200)
    b.add_argument("--max-frames", type=int, default=60000)
    b.add_argument("--frames-per-run", type=int, default=3000)
    b.add_argument("--commit", type=int, default=16,
                   help="frames of the plan actually executed before the "
                        "continuation takes over")
    b.add_argument("--tail-temp", type=float, default=0.0,
                   help="temperature of the continuation. Zero on purpose: at "
                        "0.9 the same decision scored twice differed by as "
                        "much as different decisions did, and the label was "
                        "half dice")
    b.add_argument("--tail", type=int, default=0,
                   help="frames of fixed continuation after the commitment; "
                        "the target becomes the consequence of the decision "
                        "rather than an unexecuted template's own progress")
    b.add_argument("--keep-all", action="store_true",
                   help="keep every probed moment, not only the ones "
                        "where the plans diverge — for training a probe "
                        "rather than for challenging one")
    b.add_argument("--tail-draws", type=int, default=1,
                   help="how many futures to average the tail over, "
                        "so the label is an expectation and not one "
                        "realised continuation")
    b.add_argument("--driver", default=None,
                   help="let this probe drive the playthrough instead of the "
                        "policy, so the points come from the states it visits "
                        "rather than the ones it was trained on")
    b.add_argument("--repeat", type=int, default=4)
    b.add_argument("--temperature", type=float, default=0.9)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default=DEFAULT_PATH)
    b.set_defaults(func=build)

    s = sub.add_parser("score", help="rank a model against the console's answer")
    s.add_argument("model")
    s.add_argument("--battery", default=DEFAULT_PATH)
    s.set_defaults(func=score)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
