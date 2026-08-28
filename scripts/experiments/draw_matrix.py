"""Sixteen futures per candidate, saved once, so the rest is arithmetic.

Every question about the tail target reduces to the same measurement: at a
decision point, what does each candidate's return look like as a *random
variable*? One draw of it is what the planner used to score with, and that
turned out to be as noisy as the differences it was meant to rank.

So draw sixteen and keep them all. From the saved matrix, offline and free:

* nested N = 1, 2, 4, 8, 16 built from the same rollouts, so "is four enough"
  is answered without another emulator hour and without confounding N with a
  fresh sample;
* self-agreement of two independent panels, agreement of their top-*sets*,
  the variance of paired differences (which is what a ranking actually uses —
  correlation can come out positive while every difference is wrong);
* the regret of one panel's label measured against the other panel;
* a soft teacher p_i = P(i = argmax), by bootstrap over the draws, and its
  entropy, and the gap between first and second;
* mean against median, lower quartile and CVaR, so risk-sensitive selection
  is tested before anything is trained on it.

Common random numbers are the cheap variance reduction: one seeded stream per
draw, shared by all six candidates, independent between draws. The marginal
law of each continuation is unchanged, but the future policy's luck is largely
differenced out when candidates are compared. `--independent` turns it off, so
the two can be measured against each other.

    uv run python scripts/experiments/draw_matrix.py runs/bc_smb_new \
        --points 600 --draws 16
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

HORIZON, COMMIT, TAIL = 48, 16, 96
PROBE_EVERY = 20
IDLE_STEP, IDLE_MAX = 37, 1800


def _continue(env, policy, obs, l0, repeat, temperature):
    """Play the continuation; return (died, the observation it ended on)."""
    pressed: frozenset = frozenset()
    for k in range(TAIL):
        if k % repeat == 0:
            pr, _ = policy.act(obs.frame_rgb, temperature)
            pressed = pr - {"START", "SELECT"}
        obs = env.step_buttons([pressed])
        now = (obs.debug or {}).get("lives")
        if l0 is not None and now is not None and now < l0:
            return True, obs
    return False, obs


def _point(env, policy, obs, cands, args, rng):
    """Every candidate's committed prefix, then `draws` futures of each.

    With `--repeats R`, the whole matrix is built R times from the *same* save
    state, with independent draw streams. Two matrices on one state measure
    the Monte Carlo variance of the label alone; the same measurement across
    different states that share a representation then says how much variance
    the representation is hiding. Subtracting one argmax disagreement rate
    from another does not decompose anything — variances do.
    """
    from nes_player.policy.robustify import progress_of

    here = env.save_state()
    stack = list(policy._stack)
    state = np.random.get_state()
    l0 = (obs.debug or {}).get("lives")
    p0 = progress_of(obs.debug or {})

    seq, o = [], obs
    for k in range(HORIZON):
        if k % args.repeat == 0:
            p, _ = policy.act(o.frame_rgb, args.temperature)
            p = p - {"START", "SELECT"}
        seq.append(p)
        o = env.step_buttons([p])
    policy._stack = list(stack)
    env.load_state(here)

    # One stream per draw, shared across candidates. Drawn from `rng` so the
    # streams themselves do not depend on how many candidates there are.
    reps = max(1, args.repeats)
    seeds = rng.integers(0, 2**31 - 1, size=(reps, args.draws))

    rets = np.zeros((reps, len(cands) + 1, args.draws), np.float32)
    dead = np.zeros((reps, len(cands) + 1, args.draws), bool)
    det_ret = np.zeros(len(cands) + 1, np.float32)
    det_dead = np.zeros(len(cands) + 1, bool)

    for c, (_, plan) in enumerate([("bc", seq), *cands]):
        env.load_state(here)
        policy._stack = list(stack)
        ob, died = obs, False
        for k in range(COMMIT):
            ob = env.step_buttons([plan[k]])
            now = (ob.debug or {}).get("lives")
            if l0 is not None and now is not None and now < l0:
                died = True
                break
        if died:
            dead[:, c, :] = True
            det_dead[c] = True
            continue
        mid = env.save_state()
        mid_stack = list(policy._stack)
        for r_ in range(reps):
            for d in range(args.draws):
                env.load_state(mid)
                policy._stack = list(mid_stack)
                if not args.independent:
                    np.random.seed(int(seeds[r_, d]))
                gone, end = _continue(env, policy, ob, l0, args.repeat,
                                      args.temperature)
                dead[r_, c, d] = gone
                rets[r_, c, d] = progress_of(end.debug or {}) - p0
        # The greedy continuation, on the same state. This is the target that
        # produced a probe worth +449, and the question of whether its bias is
        # accidentally useful is answered by scoring its choice against the
        # honest sixteen.
        env.load_state(mid)
        policy._stack = list(mid_stack)
        gone, end = _continue(env, policy, ob, l0, args.repeat, 0.0)
        det_dead[c] = gone
        det_ret[c] = progress_of(end.debug or {}) - p0

    env.load_state(here)
    policy._stack = list(stack)
    np.random.set_state(state)
    return rets, dead, det_ret, det_dead


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--game", default="SuperMarioBros-Nes-v0")
    ap.add_argument("--state", default="default")
    ap.add_argument("--points", type=int, default=600)
    ap.add_argument("--draws", type=int, default=16)
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=9000,
                    help="first playthrough seed; kept away from the seeds any "
                         "arm is scored on")
    ap.add_argument("--frames-per-run", type=int, default=3000)
    ap.add_argument("--max-frames", type=int, default=400000)
    ap.add_argument("--driver", default=None,
                    help="let this probe drive, so the points are the states "
                         "it visits; its own pick is recorded beside the "
                         "teacher's, which separates covariate shift from "
                         "misclassification")
    ap.add_argument("--stratify-cap", type=int, default=0,
                    help="accept at most this many points per (x//16, "
                         "airborne, sign vx) cell, so a pipe cannot mint ten "
                         "near-identical examples")
    ap.add_argument("--save-states", action="store_true",
                    help="store the 13 kB emulator save state of every point, "
                         "so any point can be relabelled or replayed later")
    ap.add_argument("--candidates", default="",
                    help="restrict to these templates, comma separated; bc is "
                         "always kept. Two candidates and four draws cost 848 "
                         "emulator frames a point against 9360, which is what "
                         "makes a hundred thousand labels affordable")
    ap.add_argument("--store", choices=("frames", "features"),
                    default="frames",
                    help="'features' saves the 48x48x6 input the model reads "
                         "instead of the 224x240x3 frame — 14 kB a point "
                         "against 161 kB, which is what makes a hundred "
                         "thousand points fit on disk")
    ap.add_argument("--repeats", type=int, default=1,
                    help="build the whole matrix this many times from the same "
                         "save state, with independent draw streams — the "
                         "Monte Carlo half of the variance decomposition")
    ap.add_argument("--independent", action="store_true",
                    help="sample each candidate's futures independently, "
                         "instead of sharing one stream per draw")
    ap.add_argument("--out", default="runs/knowledge/draw_matrix.npz")
    args = ap.parse_args()

    from oracle_mpc import templates

    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.perception.feedback import game_over
    from nes_player.perception.motion import pick_hero
    from nes_player.perception.sprites import SpriteTracker, sprite_boxes
    from nes_player.policy.bc import BCPolicy
    from nes_player.policy.go_explore import _begin

    env = StableRetroAdapter(args.game, include_debug=True, state=args.state)
    policy = BCPolicy(args.checkpoint)
    driver = None
    if args.driver:
        from probe_duel import ProbePlanner
        driver = ProbePlanner(args.driver)
    cands = templates(HORIZON)
    if args.candidates:
        want = [c.strip() for c in args.candidates.split(",")]
        cands = [(n, pl) for n, pl in cands if n in want]
        missing = [w for w in want if w != "bc"
                   and w not in [n for n, _ in cands]]
        if missing:
            raise SystemExit(f"no such template: {missing}")
    names = ["bc", *[n for n, _ in cands]]

    frames, heroes, rams, runs, picks = [], [], [], [], []
    rets, dead, det_r, det_d = [], [], [], []
    saves, logits_l, hist_l, frame_ix = [], [], [], []
    cell_count: dict = {}
    hist: list = []
    seed = args.seed - 1
    played = 0
    while len(frames) < args.points and played < args.max_frames:
        seed += 1
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        policy.reset()
        tracker = SpriteTracker()
        obs = _begin(env)
        pressed: frozenset = frozenset()
        for _ in range(IDLE_STEP * seed % IDLE_MAX):
            obs = env.step_buttons([frozenset()])
        end = played + args.frames_per_run
        held: list = []
        while len(frames) < args.points and played < min(end, args.max_frames):
            played += 1
            hero = pick_hero(tracker.update(
                obs.frame_rgb, pressed, boxes=sprite_boxes(env._env.get_ram())))
            if driver is not None and not held and hero is not None:
                mine = int(np.argmax(driver.rank(obs.frame_rgb, hero,
                                                 env._env.get_ram())))
            take = played % PROBE_EVERY == 0 and hero is not None
            if take and args.stratify_cap:
                ram_now = env._env.get_ram()
                wx = int(ram_now[0x6D]) * 256 + int(ram_now[0x86])
                cell = (args.state, wx // 16, int(ram_now[0x1D]) != 0,
                        int(np.sign(hero.vx)))
                if cell_count.get(cell, 0) >= args.stratify_cap:
                    take = False
                else:
                    cell_count[cell] = cell_count.get(cell, 0) + 1
            if take:
                r, d, dr, dd = _point(env, policy, obs, cands, args, rng)
                frames.append(obs.frame_rgb.copy())
                heroes.append((hero.cx, hero.cy, hero.vx, hero.vy))
                rams.append(env._env.get_ram().copy())
                runs.append(seed)
                picks.append(mine if driver is not None else -1)
                saves.append(np.frombuffer(env.save_state(), np.uint8)
                             if args.save_states else np.zeros(0, np.uint8))
                _, ranked = policy.act(obs.frame_rgb, args.temperature)
                policy._stack.pop()          # act() advanced the stack; undo
                logits_l.append(np.array([pr for _, pr in ranked], np.float32))
                hist_l.append(np.array(hist[-16:] + [0] * (16 - len(hist)),
                                       np.uint8))
                frame_ix.append(played)
                rets.append(r)
                dead.append(d)
                det_r.append(dr)
                det_d.append(dd)
            if driver is not None and not held and hero is not None:
                # The student drives with the same commitment the planner uses,
                # so the states are the ones it actually produces.
                seq, o = [], obs
                here = env.save_state()
                stack = list(policy._stack)
                for k in range(HORIZON):
                    if k % args.repeat == 0:
                        p, _ = policy.act(o.frame_rgb, args.temperature)
                        p = p - {"START", "SELECT"}
                    seq.append(p)
                    o = env.step_buttons([p])
                policy._stack = list(stack)
                env.load_state(here)
                held = list((seq if mine == 0
                             else cands[mine - 1][1])[:COMMIT])
            if held:
                pressed = held.pop(0)
            elif played % args.repeat == 0:
                pressed, _ = policy.act(obs.frame_rgb, args.temperature)
                pressed = pressed - {"START", "SELECT"}
            from nes_player.emulator.controller import BUTTONS
            hist.append(sum(1 << k for k, b in enumerate(BUTTONS)
                            if b in pressed) & 0xFF)
            del hist[:-16]
            obs = env.step_buttons([pressed])
            if game_over(obs.debug or {}):
                break
    env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stack = np.stack(frames)
    extra = {}
    if args.store == "features":
        from plan_probe import features as _feat
        imgs, vecs = _feat({"frames": stack, "ram": np.stack(rams),
                            "heroes": np.array(heroes, np.float32)}, "strip")
        # uint8: the crops are byte images to begin with, and float32
        # would be 55 kB a point, which is 5.5 GB at a hundred thousand.
        extra = {"imgs": imgs.clip(0, 255).astype(np.uint8), "vecs": vecs}
        stack = np.zeros((len(frames), 1, 1, 3), np.uint8)
    more = {}
    if saves and len(saves[0]):
        more["saves"] = np.stack(saves)
    if logits_l:
        more["bc_probs"] = np.stack(logits_l)
        more["act_hist"] = np.stack(hist_l)
        more["frame_ix"] = np.array(frame_ix, np.int32)
    np.savez_compressed(
        out, **extra, **more, frames=stack, heroes=np.array(heroes, np.float32),
        ram=np.stack(rams), run=np.array(runs, np.int32),
        student=np.array(picks, np.int32),
        names=np.array(names), returns=np.stack(rets), died=np.stack(dead),
        det_returns=np.stack(det_r), det_died=np.stack(det_d),
        crn=not args.independent)
    print(json.dumps({
        "path": str(out), "points": len(frames), "draws": args.draws,
        "frames_played": played, "runs": len(set(runs)),
        "common_random_numbers": not args.independent,
        "repeats": max(1, args.repeats),
        "driver": args.driver or "",
        "points_with_a_death": int(np.stack(dead).any((1, 2)).sum()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
