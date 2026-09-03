"""The planner with a perfect world model: the emulator itself.

Every duel so far compared the reactive policy against MPC on a learned model,
and improving that model — action advantage 1.03 to 2.39, rollout error 3.68 to
0.94 — moved the game metric from 553.9 to 638.5 against the policy's 1058.2.
That says further accuracy on (dx, dy) does not help. It does not say the model
is irrelevant: a model can predict its chosen quantities perfectly and still
lack what control needs. In front of a pit, `run` and `jump` have entirely
plausible kinematics for thirty frames, and the outcome is decided by the edge
of the platform, which is not in the crop.

So replace the model with save-state branches through the real console and
leave everything else alone. Three readings, all decisive:

* oracle also loses to the policy -> the scorer, the horizon or the five
  templates are what is broken, and the model was never the question.
* oracle wins and the learned model loses -> the model really is insufficient,
  and this is the number it has to reach.
* both choose well here and still lose online -> replanning, the estimate of
  the current state, or the commitment window.

The policy's own next moves are among the candidates, rolled out as a sequence
rather than as its first press, so the comparison is between whole plans.

    uv run python scripts/experiments/oracle_mpc.py runs/bc_smb_new \
        --horizons 48 96 144 --runs 4
"""

import argparse
import os
import json
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

IDLE_STEP, IDLE_MAX = 37, 1800
WEAPON_MAIN = 400   # px the planner credits for holding spread; CLI-settable
SCAN_POS: dict = {}  # game -> position bytes from controllability.py, opt-in
_UNWRAP: dict = {}  # game -> [last raw, turns] for an 8-bit scroll byte
# Contra object tables (found 2026-09-03 from a RAM dump under point-blank
# fire): type per slot at 0x530, HP per slot at 0x580, 16 slots. The wall
# is four objects — types 17 (HP 32), 16 (16), 16 (16), 4 (8) — in
# whatever slots they land; earlier code summed three *type* bytes and one
# HP byte at fixed addresses, which is why its "66" was not damage.
WALL_TYPES = frozenset({4, 16, 17})
WALL_HP0 = 72
WALL_CLIP = True  # --wall-unclipped: price every typed HP point, no baseline


def wall_hp(ram) -> int:
    return sum(int(ram[0x580 + i]) for i in range(16)
               if int(ram[0x530 + i]) in WALL_TYPES)


def _unwrap(game: str, raw: int) -> int:
    """Turn an 8-bit scroll byte into a running position.

    Symmetric: a drop past 128 is a wrap forward, a rise past 128 a wrap
    back, so restoring an earlier savestate across a wrap unwinds it.
    ponytail: breaks if one jump moves the true scroll > 128 px (a
    respawn pulled back a screen); anchor from x0 per decision if seen.
    """
    st = _UNWRAP.setdefault(game, [raw, 0])
    d = raw - st[0]
    if d < -128:
        st[1] += 1
    elif d > 128:
        st[1] -= 1
    st[0] = raw
    return st[1] * 256 + raw


def anchor_pos(env, game: str, pos0: int) -> None:
    """Pin the unwrap counter so this process agrees with the caller's x0."""
    sp = SCAN_POS.get(game)
    if isinstance(sp, dict) and sp.get("hi") is None:
        raw = int(env._env.get_ram()[sp["lo"]])
        _UNWRAP[game] = [raw, (sp["sign"] * pos0 - raw) // 256]
DEATH = -1e9        # a plan that dies is not compared on distance


def templates(h: int):
    """The planner's five behaviours, at whatever horizon is being tested.

    Jump lengths stay physical rather than scaled: holding A beyond about ten
    frames raises the arc, it does not lengthen the plan.
    """
    from nes_player.policy.planner import JUMP, LEFT, NOOP, RUN

    return [
        ("run", [RUN] * h),
        ("jump now", [JUMP] * 10 + [RUN] * (h - 10)),
        ("jump later", [RUN] * 12 + [JUMP] * 10 + [RUN] * (h - 22)),
        ("wait", [NOOP] * h),
        ("back off", [LEFT] * 12 + [RUN] * (h - 12)),
    ]


FIRE_DIAG = frozenset({"UP", "RIGHT", "B"})


def game_templates(h: int, game: str):
    """The shared five, plus what a game's own weapons demand.

    Contra's wall boss is killed by firing up-right — a button chord no
    Mario behaviour ever needed. The first game-specific template of the
    project, recorded as such: the six-for-everything claim now reads
    six-plus-what-the-weapon-needs.
    """
    cands = templates(h)
    if game.startswith("Contra") or game.startswith("SuperC"):
        # The wall fight needs three things Mario never did: the diagonal
        # itself, going prone under the bullet stream, and firing the
        # diagonal from a jump. Survival is already priced by the death
        # floor; these give the search moves that survive.
        # B must be TAPPED: held down it fires exactly one bullet, which
        # made the first version of these templates decorative — the
        # point-blank diagonal script that strips the wall taps 2-on
        # 2-off, and so do these now.
        def taps(active, rest, n):
            return [active if k % 4 < 2 else rest for k in range(n)]

        diag_rest = frozenset({"UP", "RIGHT"})
        prone = frozenset({"DOWN", "B"})
        prone_rest = frozenset({"DOWN"})
        jump_diag = frozenset({"A", "B", "UP", "RIGHT"})
        cands += [
            ("fire up-right", taps(FIRE_DIAG, diag_rest, h)),
            ("prone fire", taps(prone, prone_rest, h)),
            ("jump fire", [jump_diag] * 10
             + taps(FIRE_DIAG, diag_rest, h - 10)),
        ]
    return cands


def auto_templates(h: int, game: str):
    """Templates assembled from the button probe instead of from folklore.

    Reads runs/knowledge/buttons_<game>.json (probe_buttons.py) and derives:
        forward   the chord with the most position gained, held
        fire      among chords that do not move, the one spawning the most
                  new sprites, in whichever mode (tap/hold) spawns more —
                  this is where "B must be tapped" is read off the console
        jump      A + forward, by NES convention (the probe measures no
                  vertical yet, so this one is assumed, and said so)
    and builds the same family the hand set has: forward, jump now, jump
    later, wait, back off, plus the fire chords where the game has a gun.
    """
    import json

    path = Path("runs/knowledge") / f"buttons_{game}.json"
    rows = json.loads(path.read_text())
    chord = lambda r: frozenset(r["chord"].split("+"))  # noqa: E731
    moving = [r for r in rows if r["mode"] == "hold" and "RIGHT" in r["chord"]
              and not r["died"]]
    fwd = chord(max(moving, key=lambda r: r["dpos"]))
    still = [r for r in rows if abs(r["dpos"]) < 5 and "RIGHT" not in r["chord"]
             and "LEFT" not in r["chord"] and r["sprites"] > 0.5]
    fire = max(still, key=lambda r: r["sprites"]) if still else None
    jump = fwd | {"A"}
    left = frozenset({"LEFT"})
    cands = [
        ("forward", [fwd] * h),
        ("jump now", [jump] * 10 + [fwd] * (h - 10)),
        ("jump later", [fwd] * 12 + [jump] * 10 + [fwd] * (h - 22)),
        ("wait", [frozenset()] * h),
        ("back off", [left] * 12 + [fwd] * (h - 12)),
    ]
    if fire is not None:
        btn = chord(fire) - {"UP", "DOWN"}      # the trigger itself
        tap = fire["mode"] == "tap"

        def pattern(active, rest, n):
            if not tap:
                return [active] * n
            return [active if k % 4 < 2 else rest for k in range(n)]

        up, down = frozenset({"UP", "RIGHT"}), frozenset({"DOWN"})
        cands += [
            ("fire up-forward", pattern(btn | up, up, h)),
            ("prone fire", pattern(btn | down, down, h)),
            ("jump fire", [jump | btn] * 10 + pattern(btn | up, up, h - 10)),
        ]
    return cands


def scan_templates(h: int, game: str):
    """Templates from the causal scan alone (A1 on top of A0).

    Reads runs/knowledge/control_<game>.json and uses nothing per-game:
        forward   the RIGHT-chord that pushes the scan's own position bytes
                  furthest, averaged over controllable states; ties go to
                  the chord that also fires (walk-and-shoot)
        fire      the B-chord/mode with the most projectile pixels off the
                  body — the mode carries the tap/hold semantics
        jump      A + forward, still by convention (no vertical readout yet)
    """
    import json

    rep = json.loads((Path("runs/knowledge") / f"control_{game}.json")
                     .read_text())
    play = [r for k, r in rep.items()
            if isinstance(r, dict) and r.get("controllable")]
    push, fire_px = {}, {}
    for r in play:
        for k, v in r["push"].items():
            push[k] = push.get(k, 0) + v
        for k, v in r["fire"].items():
            fire_px[k] = fire_px.get(k, 0) + v["pixels_off_body"]
    fire_key = max(fire_px, key=fire_px.get) if fire_px else None
    has_gun = fire_key is not None and fire_px[fire_key] >= 5 * max(1, len(play))
    fwd_keys = [k for k in push if "RIGHT" in k and k.endswith("/hold")]
    top = max(push[k] for k in fwd_keys)
    tied = [k for k in fwd_keys if push[k] >= 0.9 * top]
    fk = next((k for k in tied if "B" in k), tied[0]) if has_gun else \
        max(tied, key=lambda k: push[k])
    fwd = frozenset(fk.split("/")[0].split("+"))
    jump = fwd | {"A"}
    cands = [
        ("forward", [fwd] * h),
        ("jump now", [jump] * 10 + [fwd] * (h - 10)),
        ("jump later", [fwd] * 12 + [jump] * 10 + [fwd] * (h - 22)),
        ("wait", [frozenset()] * h),
        ("back off", [frozenset({"LEFT"})] * 12 + [fwd] * (h - 12)),
    ]
    if has_gun:
        chord, mode = fire_key.split("/")
        btn = frozenset(chord.split("+")) - {"UP", "DOWN"}
        tap = mode == "tap"

        def pattern(active, rest, n):
            return ([active if k % 4 < 2 else rest for k in range(n)]
                    if tap else [active] * n)

        up, down = frozenset({"UP", "RIGHT"}), frozenset({"DOWN"})
        cands += [
            ("fire up-forward", pattern(btn | up, up, h)),
            ("prone fire", pattern(btn | down, down, h)),
            ("jump fire", [jump | btn] * 10 + pattern(btn | up, up, h - 10)),
        ]
    return cands


def mario_x(env) -> int:
    ram = env._env.get_ram()
    return int(ram[0x6D]) * 256 + int(ram[0x86])


def game_pos(env, game: str) -> int:
    """Forward position for the value, from the game's own counters.

    Mario publishes a 16-bit world x. Contra-style integrations publish a
    big-endian 16-bit camera scroll and a level counter; the level folds in
    at the same stride progress_of uses, so the value stays monotone across
    a level boundary inside a branch.
    """
    ram = env._env.get_ram()
    if SCAN_POS.get(game):
        # no RAM map, no hand rule: the camera pair find_camera.py caught
        # if it exists, else A0's LEFT/RIGHT-responsive bytes summed
        sp = SCAN_POS[game]
        if isinstance(sp, dict):
            if sp.get("hi") is None:
                return sp["sign"] * _unwrap(game, int(ram[sp["lo"]]))
            return sp["sign"] * (int(ram[sp["hi"]]) * 256 + int(ram[sp["lo"]]))
        return int(sum(int(ram[b]) for b in sp))
    if game.startswith("SuperMario"):
        return mario_x(env)
    if game.startswith("SuperC"):
        # found empirically: lo wraps at 0x6B, hi ticks at 0x6C, monotone
        # through two wraps under a scripted run to x=704
        return int(ram[108]) * 256 + int(ram[107])
    return int(ram[48]) * 4000 + int(ram[100]) * 256 + int(ram[101])


def game_value(env, game: str) -> int:
    """What the planner maximises: position, plus what position cannot see.

    Contra's wall boss stops the camera, and killing it takes ~2250 frames
    of sustained fire — far beyond any rollout window. Damage is the only
    progress left; it lives in four object-slot bytes that are flat
    without fire and fall monotonically under hits (sum 66 -> 0). The term
    switches on only where the camera has hit the wall, and at that moment
    hp is still full, so the value stays continuous.

    This is the PLANNER'S objective, not the run metric: mixing the bonus
    into best_x once produced a run that read as a level clear while the
    wall still stood. The metric stays pure position.
    """
    pos = game_pos(env, game)
    if game.startswith("Contra"):
        ram = env._env.get_ram()
        xs, lvl = pos % 4000, pos // 4000
        PX_PER_HIT = 40
        if lvl > 0:
            pos += PX_PER_HIT * WALL_HP0
        elif xs >= 3070:
            # The cliff's turrets share type 16 with the cannons and can
            # still be in the tables at arrival (typed sum 120-135, not
            # 72), so the clipped term is silent until they are gone.
            # Unclipped, the baseline is irrelevant: only differences
            # within a decision are ever used.
            pos += (PX_PER_HIT * max(0, WALL_HP0 - wall_hp(ram)) if WALL_CLIP
                    else PX_PER_HIT * (WALL_HP0 - wall_hp(ram)))
        # The owner's capsule idea: a weapon upgrade is worth diverting
        # for. 0xAA verified behaviourally by poke-and-look — 0 is the
        # rifle, 3 fans out as spread, others are mid-tier. Spread is
        # what kills the wall; death resets the byte to zero, so the
        # planner also prices keeping the gun alive.
        if WEAPON_MAIN:
            # low nibble is the gun (3 = spread), high bits are flags
            # (0x10 showed up on every real pickup as rapid/red variant)
            tier = int(ram[170]) & 0x0F
            pos += (WEAPON_MAIN if tier == 3
                    else WEAPON_MAIN // 2 if ram[170] else 0)
    return pos


def game_progress(d: dict, progress_of) -> int:
    """The run metric: SMB's folded progress, or the integration's own."""
    if "xscroll" in d:
        return int(d.get("level", 0)) * 4000 + int(d["xscroll"])
    return progress_of(d)


def begin_any(env, game: str):
    """Past the title screen, for games without SMB's countdown clock.

    SMB's _begin detects the running game by its timer ticking down; games
    without a `time` variable get the blunt version — pulse START until the
    lives counter reads positive, then hand over.
    """
    from nes_player.policy.go_explore import _begin

    if game.startswith("SuperMario"):
        return _begin(env)
    obs = env.reset(seed=0)
    for i in range(2000):
        d = obs.debug or {}
        if i > 120 and int(d.get("lives", 0) or 0) > 0:
            break
        pulse = i % 60 in (0, 1)
        obs = env.step_buttons([frozenset({"START"}) if pulse
                                else frozenset()])
    # No more START once the game has accepted it — the same button is
    # pause during play. The level intro (AREA screen, spawn animation)
    # runs itself out in well under six hundred frames.
    for _ in range(600):
        obs = env.step_buttons([frozenset()])
    return obs


def learned_dx(ghost, frame_rgb, hero, plan) -> float:
    """The same question asked of the ego model instead of the console.

    Everything around it is held identical — same candidates, same commitment,
    same "furthest along the level wins" — so the difference between this arm
    and the oracle is the model and nothing else.
    """
    import torch

    from nes_player.emulator.controller import BUTTONS
    from nes_player.world_model.ego import _crop

    m = ghost.model
    with torch.no_grad():
        feat = m.enc(torch.from_numpy(_crop(frame_rgb, hero.cx, hero.cy))
                     .float().div_(255).permute(2, 0, 1).unsqueeze(0))
        h = torch.zeros(1, 128)
        v = torch.tensor([[hero.vx, hero.vy]], dtype=torch.float32)
        x = 0.0
        for pressed in plan:
            mask = sum(1 << k for k, b in enumerate(BUTTONS) if b in pressed)
            try:
                aid = ghost.vocab.masks.index(mask)
            except ValueError:
                aid = 0
            h, pred = m.forward_step(h, feat, v, torch.tensor([aid]))
            x += float(pred[0, 0])
            v = pred
    return x


def _continue(env, policy, obs, l0, tail: int, repeat: int,
              temperature: float):
    """Keep playing under a fixed continuation, and say whether it ends badly.

    The continuation has to be one policy and always the same one, or the value
    is a mixture of whatever controller happened to be running — a target that
    moves as the thing being trained changes. This uses the reactive policy,
    which is the cheap choice; the expensive and more meaningful one is
    oracle-5x48 continuing itself, at six branches per replan.
    """
    died, used = False, 0
    pressed: frozenset = frozenset()
    for k in range(tail):
        if k % repeat == 0:
            pressed, _ = policy.act(obs.frame_rgb, temperature)
            pressed = pressed - {"START", "SELECT"}
        obs = env.step_buttons([pressed])
        used += 1
        now = (obs.debug or {}).get("lives")
        if l0 is not None and now is not None and now < l0:
            died = True
            break
    return died, used


_W: dict = {}


def _worker_init(game: str, checkpoint: str, integ: str | None,
                 scan_pos=None, wall_clip: bool = True,
                 weapon_main: int = 400):
    """One emulator and one policy per worker process, loaded once.

    A spawned worker re-imports this module, so anything main set after
    parsing flags is gone here: SCAN_POS arrives explicitly, or the worker
    silently scores a scan-positioned game with the Contra formula (it
    did: every candidate identical, the planner blind, best_x fiction).
    """
    import torch

    global WALL_CLIP, WEAPON_MAIN
    if scan_pos is not None:
        SCAN_POS[game] = scan_pos
    WALL_CLIP, WEAPON_MAIN = wall_clip, weapon_main

    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.policy.bc import BCPolicy

    torch.set_num_threads(1)
    env = StableRetroAdapter(game, include_debug=True, state=None,
                             integration_dir=integ)
    env.reset(seed=0)
    _W["env"], _W["policy"], _W["game"] = env, BCPolicy(checkpoint), game


def _eval_job(job: dict) -> dict:
    """Score one candidate: prefix, then `draws` continuations from `mid`.

    Identical to the serial loop, including the CRN seeding of every draw,
    so a parallel run reproduces a serial one byte for byte — the test
    that guards this path.
    """
    env, policy, game = _W["env"], _W["policy"], _W["game"]
    env.load_state(job["here"])
    l0, x0 = job["l0"], job["x0"]
    anchor_pos(env, game, x0)
    died, used, o = False, 0, None
    for k in range(job["span"]):
        o = env.step_buttons([job["plan"][k]])
        used += 1
        now = (o.debug or {}).get("lives")
        if l0 is not None and now is not None and now < l0:
            died = True
            break
    if died or not job["tail"]:
        val = DEATH if died else game_value(env, game) - x0
        return {"name": job["name"], "died": died, "outs": None,
                "val": val, "frames": used}
    mid = env.save_state()
    outs = []
    for di in range(job["draws"]):
        env.load_state(mid)
        policy._stack = list(job["stack"])
        if job["crn"]:
            np.random.seed(zlib.crc32(
                f"{job['seed']}:{int(x0)}:{job['name']}:{di}".encode())
                & 0xFFFFFFFF)
        gone, extra = _continue(env, policy, o, l0, job["tail"],
                                job["repeat"], job["tail_temp"])
        used += extra
        outs.append((gone, game_value(env, game) - x0))
        if os.environ.get("POS_DEBUG"):
            sp = SCAN_POS.get(game) or {}
            print(f"[w] {job['name']} di={di} x0={x0} gone={gone} "
                  f"raw={int(env._env.get_ram()[sp.get('lo', 0)])} "
                  f"unwrap={_UNWRAP.get(game)} gain={outs[-1][1]}",
                  flush=True)
    return {"name": job["name"], "died": False, "outs": outs,
            "val": None, "frames": used}


def value_of(outs, draws: int, death_price: float) -> float:
    """The candidate's value from its draws — one place for the rule."""
    if death_price:
        return float(np.mean([x - death_price * g for g, x in outs]))
    dead_n = sum(g for g, _ in outs)
    if dead_n * 2 > draws:
        # die usefully: the floor keeps every live plan above every dead
        # one; among the dead, damage dealt before dying decides
        return DEATH + float(np.mean([x for _, x in outs]))
    return float(np.mean([x for g, x in outs if not g]))


def run(checkpoint: str, game: str, state: str | None, frames: int, seed: int,
        temperature: float, repeat: int, horizon: int | None,
        commit: int, ghost_path: str | None = None,
        margin: float = 0.0, noise: float = 0.0,
        frozen: bool = False, heavy: float = 0.0,
        probe_path: str | None = None, bc_live: bool = False,
        tail: int = 0, tail_from: int = 0, ram_hero: bool = False,
        defer_below: float = 0.0, tail_temp: float = 0.9,
        draws: int = 1, video: str | None = None,
        fixed: str = "", rescue: bool = False,
        corrupt: float = 0.0, knn_memory: str = "", gate: bool = False,
        gate_fp: float = 0.0, gate_fn: float = 0.0,
        adaptive: float | None = None, adaptive_g2: float = 0.0,
        crn: bool = False, two_step: bool = False,
        death_price: float = 0.0, escapes: bool = False,
        save_final: str = "", load_state: str = "",
        save_at: int = 0, workers: int = 1, auto_tpl: str = "",
        rollback: int = 0) -> dict:
    from nes_player.emulator.controller import BUTTONS
    from nes_player.emulator.stable_retro import StableRetroAdapter
    from nes_player.perception.motion import pick_hero
    from nes_player.perception.sprites import (RamHero, SpriteTracker,
                                               sprite_boxes)
    from nes_player.policy.bc import BCPolicy
    from nes_player.policy.go_explore import _begin
    from nes_player.policy.robustify import progress_of
    from nes_player.world_model.ego import GhostPredictor

    trig = None
    if adaptive is not None:
        from adaptive import trigger as trig
        from analyse_draws import DEATH_MARGIN

    def keyed_draw(x0, name, di, fn):
        """Common random numbers for one continuation draw.

        Without this every draw consumes the shared stream, so two arms
        diverge in pure noise the moment one of them spends a different
        number of draws — even at decisions where both pick the same
        candidate. Keyed by (run seed, world x of the decision, candidate,
        draw index), identical states get identical continuations: arms
        stay frame-identical until a genuine policy difference, and an
        escalated decision sees exactly the draws the 4-draw oracle would.
        The swap also keeps the on-trajectory stream untouched by draws.
        """
        if not crn:
            return fn()
        st = np.random.get_state()
        np.random.seed(zlib.crc32(
            f"{seed}:{int(x0)}:{name}:{di}".encode()) & 0xFFFFFFFF)
        try:
            return fn()
        finally:
            np.random.set_state(st)

    np.random.seed(seed)
    # Games with a repo integration (ROM, RAM map) load from it; everything
    # else — including SMB — comes from stable-retro's own registry.
    integ_root = Path(__file__).resolve().parents[2] / "integrations"
    integ = str(integ_root) if (integ_root / game).exists() else None
    env = StableRetroAdapter(game, include_debug=True, state=state,
                             integration_dir=integ)
    pool = None
    if workers > 1:
        import multiprocessing as mp

        pool = mp.get_context("spawn").Pool(
            workers, initializer=_worker_init,
            initargs=(game, checkpoint, integ, SCAN_POS.get(game),
                      WALL_CLIP, WEAPON_MAIN))
    policy = BCPolicy(checkpoint)
    obs = begin_any(env, game)
    # the boot/title screens may have spun an 8-bit scroll byte: the run's
    # position starts at its raw value, or arms would carry 256-px offsets
    # from each other's title screens into best_x
    _UNWRAP.pop(game, None)
    if load_state:
        # A lab tool: start where a previous run ended (e.g. at a boss
        # wall), with a couple of lives granted so the probe is about the
        # boss and not about arriving there on the last life.
        env.load_state(Path(load_state).read_bytes())
        try:
            env._env.data.set_value("lives", 2)
        except Exception:
            pass
        obs = env.step_buttons([frozenset()])
    for _ in range(IDLE_STEP * seed % IDLE_MAX):
        obs = env.step_buttons([frozenset()])

    ghost = GhostPredictor(ghost_path) if ghost_path else None
    knn = None
    if knn_memory:
        from analyse_draws import DEATH_MARGIN
        zk = np.load(knn_memory)
        rk = zk["returns"].astype(np.float64)[:, 0]
        dk = zk["died"][:, 0]
        for pi in range(len(rk)):
            live = rk[pi][~dk[pi]]
            floor = (live.min() if live.size else 0.0) - DEATH_MARGIN
            rk[pi][dk[pi]] = floor
        keys = (zk["ram"][:, 0x6D].astype(int) * 256
                + zk["ram"][:, 0x86].astype(int))
        knn = (keys, rk.mean(2).astype(np.float32))
    probe = None
    if probe_path:
        from probe_duel import ProbePlanner

        probe = ProbePlanner(probe_path)
    # The dashboard draws the tracked objects, so a recording gets a tracker
    # even when the arm itself has no use for one.
    tracker = (SpriteTracker() if ((ghost or probe) and not ram_hero) or video
               else None)
    def make_cands(h, compose=False):
        base = (scan_templates(h, game) if auto_tpl == "scan"
                else auto_templates(h, game) if auto_tpl
                else game_templates(h, game))
        if not compose:
            return base
        # every ordered pair at half the horizon: the rescue re-plan buys
        # the composition depth a single template cannot express
        halves = make_cands(h // 2)
        return base + [(f"{n1}+{n2}", p1 + p2)
                       for n1, p1 in halves for n2, p2 in halves]
    cands = make_cands(horizon) if horizon else []
    if two_step and horizon:
        # Two-step search: every ordered pair of the five behaviours, each at
        # half the horizon. The one thing a single template cannot express is
        # a composition — run up, then jump — and the big pits want exactly
        # that. The policy's own plan stays in as the 26th candidate.
        halves = templates(horizon // 2)
        cands = [(f"{n1}+{n2}", p1 + p2)
                 for n1, p1 in halves for n2, p2 in halves]
    elif escapes and horizon:
        # The cheap corollary of the two-step result: its safety came from
        # compositions the six templates cannot express, and three pairs
        # carried most of its choices. Add just those to the plain set, so
        # the escape hatch exists at a fraction of the 26-candidate cost.
        t = dict(templates(horizon // 2))
        pairs = ((("run", "jump now"), ("run", "jump later"),
                  ("jump later", "run"), ("jump now", "run"),
                  ("jump now", "jump now"))
                 if game.startswith("Contra") else
                 (("jump later", "jump now"), ("back off", "jump now"),
                  ("jump later", "jump later")))
        # Contra's cliff (2026-09-03): two-step passes it 8/8 where the
        # plain set passes 1/8, and its choices there are spread over the
        # run+jump family — a finer jump-timing grid, not one pair.
        cands += [(f"{a}+{b}", t[a] + t[b]) for a, b in pairs]
    best_x, deaths, lives = 0, 0, (obs.debug or {}).get("lives")
    chosen: Counter = Counter()
    held: list = []
    last_choice, last_score = "bc", 0.0
    last_ranked: list = []
    thoughts: list[str] = []
    entropy_hist: list[float] = []
    slots: list = []
    verdicts: dict = {}
    cam = None
    ghost_path = ghost_view = None
    view = memory = ears = sounds = locator = pings = None
    if video:
        # Everything the dashboard can draw, drawn. The panels are not
        # decoration: the attention map and the object memory are what make the
        # difference between a video of Mario and a video of a controller
        # deciding.
        from nes_player.cli.runtime import PingLog, SoundLog
        from nes_player.evaluation.viewer import Viewer
        from nes_player.perception.audio_events import AudioEventDetector
        from nes_player.perception.memory import ObjectMemory

        view = Viewer(video_out=video, fps=60.1, title="oracle MPC")
        memory, sounds, pings = ObjectMemory(), SoundLog(), PingLog()
        # The learned world model's guess at where the hero is going, drawn
        # beside the console's answer. It scores nothing here — this arm does
        # not use it — but the video is the one place the two can be compared.
        if Path("runs/ego_smb4").exists():
            ghost_view = ((ghost or GhostPredictor("runs/ego_smb4"))
                          if game.startswith("SuperMario") else ghost)
        ears = AudioEventDetector(env.sample_rate)
        if Path("runs/av_smb").exists():
            from nes_player.perception.av_align import SoundLocator

            locator = SoundLocator("runs/av_smb")
    defer = 0
    pressed: frozenset = frozenset()
    branch_frames = 0
    escalated = decisions = 0
    credits_used, stuck, deep_in_level = 0, 0, False
    wmax, hp_arrival, hp_min = 0, None, None
    saved_at_done = False
    # B1': failure-triggered rollback. A ring of the last `rollback`
    # decision states; when every candidate is doomed the console is
    # rewound one decision at a time, the horizon grows by the depth so
    # the re-plan still reaches the death it is trying to avoid, and the
    # first surviving plan is committed whole. Discarded frames count
    # against the budget: `i` keeps running while the game goes back.
    hist: list = []
    rb = {"attempts": 0, "rescued": 0, "failed": 0, "depths": [],
          "discarded": 0}
    rb_h = rb_depth = 0
    rb_block = None
    danger_until = 0  # game time until which the rich candidate set stays on
    gt = 0  # game time: executed frames minus rewound ones

    for i in range(frames):
        if rollback and not rb_h and gt % 16 == 0 and not held:
            # snapshot on the game's clock, not the decision clock: a
            # rescued plan is held whole, and a ring of decision states
            # would then reach back a hundred frames in one step
            hist.append((env.save_state(), list(policy._stack),
                         dict(_UNWRAP), obs, best_x, deaths, lives, gt))
            del hist[:-(rollback + 1)]
        hero = None
        if tracker is not None:
            slots = tracker.update(obs.frame_rgb, pressed,
                                   boxes=sprite_boxes(env._env.get_ram()))
            hero = pick_hero(slots)
            if memory is not None:
                d0 = obs.debug or {}
                verdicts = memory.update(obs.frame_rgb, slots, i,
                                         int(d0.get("score", 0) or 0), False)
        elif ram_hero:
            hero = RamHero(env._env.get_ram())
        needs_hero = ghost is not None or probe is not None
        if (horizon and not held and defer <= 0
                and (not needs_hero or hero is not None)):
            here = env.save_state()
            h = rb_h or horizon
            if rollback and gt >= danger_until:
                rb_depth = 0  # the danger that opened the window is behind us
            x0, l0 = game_value(env, game), (obs.debug or {}).get("lives")
            if os.environ.get("POS_DEBUG"):
                sp = SCAN_POS.get(game) or {}
                print(f"[m] i={i} x0={x0} l0={l0} "
                      f"raw={int(env._env.get_ram()[sp.get('lo', 0)])} "
                      f"unwrap={_UNWRAP.get(game)}", flush=True)
            stack = list(policy._stack)

            # The policy's own plan, played out in the branch so that what is
            # compared is a sequence and not a first press.
            seq, o = [], obs
            for k in range(h):
                if k % repeat == 0:
                    p, ranked = policy.act(o.frame_rgb, temperature)
                    if k == 0:
                        last_ranked = ranked
                    p = p - {"START", "SELECT"}
                seq.append(p)
                o = env.step_buttons([p])
            policy._stack = list(stack)
            rich = bool(rb_h) or gt < danger_until
            options = [("bc", seq),
                       *(make_cands(h, compose=True) if rich else cands)]
            env.load_state(here)

            scored = []
            pend = []
            if knn is not None:
                # Episodic memory instead of a network: the oracle's own past
                # decisions on this level, keyed by world x. Legitimate for
                # replaying a level already studied, and keyed by RAM, so this
                # arm prices the ceiling of place-keyed memory, not a
                # deliverable. Far from any memory, the policy keeps the wheel.
                mx = mario_x(env)
                d = np.abs(knn[0] - mx)
                near = np.argsort(d)[:5]
                near = near[d[near] <= 32]
                if len(near):
                    val = knn[1][near].mean(0)
                    scored = [(float(val[k]), name, plan)
                              for k, (name, plan) in enumerate(options)]
                else:
                    scored = [(1.0 if name == "bc" else 0.0, name, plan)
                              for name, plan in options]
            elif fixed and not rescue:
                # The control nobody ran: how much of a learned scorer's gain
                # is just a standing habit? "always jump now" costs 16.7 px of
                # regret against the policy's 28.1 without looking at anything.
                scored = [(1.0 if name == fixed else 0.0, name, plan)
                          for name, plan in options]
            elif probe is not None:
                # The learned scorer inside the oracle's own harness, so the
                # only thing differing between the arms is who assigns the
                # numbers. How the policy's candidate is built, the
                # commitment, which prefix is executed — all shared, which
                # they were not while the two lived in separate scripts.
                ranks = probe.rank(obs.frame_rgb, hero, env._env.get_ram())
                if defer_below and probe.agreement < defer_below:
                    # The members named different best candidates, so this is a
                    # decision the ensemble has no opinion about. Handing it
                    # back costs whatever the probe would have gained here and
                    # saves whatever it would have lost.
                    ranks = np.array([1.0] + [0.0] * (len(options) - 1))
                scored = [(float(ranks[k]), name, plan)
                          for k, (name, plan) in enumerate(options)]
            # A fixed arm must not also be scored: the oracle's numbers were
            # being appended after the one-hot ones and winning whenever any
            # plan gained more than a pixel, so the "habit" control was an
            # oracle wearing a habit's name.
            skip = probe is not None or (fixed and not rescue)
            parallel = (pool is not None and not skip and knn is None
                        and ghost is None and adaptive is None)
            if parallel:
                span = h if rb_h else (horizon if not tail
                                       else min(tail_from, horizon))
                jobs = [{"name": name, "plan": plan, "here": here,
                         "span": span, "l0": l0, "x0": x0, "tail": tail,
                         "draws": draws, "stack": stack, "crn": crn,
                         "seed": seed, "repeat": repeat,
                         "tail_temp": tail_temp}
                        for name, plan in options]
                plans = dict(options)
                for r in pool.map(_eval_job, jobs):
                    branch_frames += r["frames"]
                    val = (r["val"] if r["outs"] is None
                           else value_of(r["outs"], draws, death_price))
                    scored.append((val, r["name"], plans[r["name"]]))
            for name, plan in (() if skip or knn is not None or parallel
                               else options):
                if ghost is not None:
                    scored.append((learned_dx(ghost, obs.frame_rgb, hero, plan),
                                   name, plan))
                    continue
                env.load_state(here)
                died = False
                # `tail` turns the plan's own return into a value: play the
                # part that will actually be executed, then keep playing under
                # a fixed continuation and see where that ends up. Without it
                # the score is the progress of an open-loop template, which
                # says nothing about whether the place it lands in is a corner.
                #
                # Two semantics, because the controller does not execute what
                # it scores. `tail_from` = commit measures the consequence of
                # the *decision*: the committed prefix, then whatever comes
                # next. `tail_from` = horizon measures the plan held to the end
                # and then continued, which is the classic terminal value.
                span = h if rb_h else (horizon if not tail
                                       else min(tail_from, horizon))
                for k in range(span):
                    o = env.step_buttons([plan[k]])
                    now = (o.debug or {}).get("lives")
                    if l0 is not None and now is not None and now < l0:
                        died = True
                        break
                branch_frames += span
                if tail and not died:
                    # One draw of a sampling policy is one future, not the
                    # value of the decision, and at temperature 0.9 two draws
                    # of the same plan differ by as much as two different
                    # plans do. Averaging several keeps the continuation the
                    # one that actually follows while making the number mean
                    # something. `draws` = 1 is the single-future version.
                    mid = env.save_state()
                    outs = []
                    for di in range(draws):
                        env.load_state(mid)
                        policy._stack = list(stack)
                        gone, extra = keyed_draw(
                            x0, name, di,
                            lambda o=o, l0=l0: _continue(env, policy, o, l0,
                                                         tail, repeat,
                                                         tail_temp))
                        branch_frames += extra
                        outs.append((gone, game_value(env, game) - x0))
                    # The continuation consumed the policy's frame
                    # stack; the next candidate must start from the
                    # same history this one did.
                    policy._stack = list(stack)
                    if adaptive is not None:
                        # Scored below, once every candidate has its draws
                        # in and the trigger has spoken.
                        pend.append((name, plan, mid, o, outs))
                        continue
                    if death_price:
                        # Death as a price, not a veto. The majority rule
                        # makes a death in the minority of draws free —
                        # the dead draw simply leaves the mean — which is
                        # exactly the slack the two-step arm exposed. Here
                        # every draw counts: a dead one contributes the x
                        # it reached minus the price, so the value is
                        # E[progress] - price * P(death).
                        val = float(np.mean(
                            [x - death_price * g for g, x in outs]))
                    else:
                        dead_n = sum(g for g, _ in outs)
                        died = dead_n * 2 > draws
                        if died:
                            # Die usefully: the floor keeps every live
                            # plan above every dead one, but among the
                            # dead the damage dealt before dying decides.
                            # Without this a doomed state is an all-DEATH
                            # tie, argmax falls to the policy's plan, and
                            # at Contra's wall the inert policy stands in
                            # the bullet stream doing nothing — the loop
                            # that capped every honest run at 28 hits.
                            val = DEATH + float(np.mean(
                                [x for _, x in outs]))
                        else:
                            val = float(np.mean(
                                [x for g, x in outs if not g]))
                else:
                    val = DEATH if died else game_value(env, game) - x0
                scored.append((val, name, plan))
            if pend:
                # The adaptive budget: every candidate has `draws` paired
                # continuations; only where the frozen trigger — winner
                # instability first, then expected stopping regret under a
                # per-point sigma — says the early ranking is unreliable are
                # `draws` more paid for, for all candidates alike, so nothing
                # is ever pruned. Deaths enter the trigger through the same
                # penalised floor the offline calibration used.
                decisions += 1
                live = [x for *_, os_ in pend for g, x in os_ if not g]
                floor = (min(live) if live else 0.0) - DEATH_MARGIN
                mat = np.array([[floor if g else x for g, x in os_]
                                for *_, os_ in pend])
                if len(pend) > 1 and \
                        float(trig(mat[None], adaptive_g2)[0]) > adaptive:
                    escalated += 1
                    for _name, _plan, mid, o2, outs in pend:
                        for di in range(draws):
                            env.load_state(mid)
                            policy._stack = list(stack)
                            gone, extra = keyed_draw(
                                x0, _name, draws + di,
                                lambda o2=o2, l0=l0: _continue(
                                    env, policy, o2, l0, tail, repeat,
                                    tail_temp))
                            branch_frames += extra
                            outs.append((gone, 0.0 if gone
                                         else game_value(env, game) - x0))
                        policy._stack = list(stack)
                for name, plan, _mid, _o2, outs in pend:
                    dead_n = sum(g for g, _ in outs)
                    died = dead_n * 2 > len(outs)
                    val = DEATH if died else float(np.mean(
                        [x for g, x in outs if not g]))
                    scored.append((val, name, plan))
                pend = []
            env.load_state(here)
            # The policy keeps the wheel unless a plan is clearly better. With
            # a perfect model this changes nothing — the oracle already picks
            # the policy's own plan about 72% of the time — but the learned
            # model picks it 5% of the time and overrides into a jump instead,
            # so the margin is what stops a weak model from driving.
            if noise:
                # The oracle's numbers, blurred to the accuracy the learned
                # probe actually has. Fresh noise every time answers whether
                # the *size* of the error matters; it does not, up to 4 px.
                #
                # `frozen` instead makes the error a function of where Mario is
                # standing, so returning to the same place reproduces the same
                # mistake. That is what a learned model's error is really like:
                # deterministic, not resampled. A controller that loops back to
                # a place it misjudges will misjudge it again, every time,
                # while noise gets a new draw and averages away.
                if heavy:
                    # Mostly right, occasionally badly wrong — the shape the
                    # probe's error actually has. Its median regret is zero and
                    # its worst 5% costs 24 px, which a gaussian of the same
                    # standard deviation does not reproduce: a bell curve has
                    # no tail worth speaking of at 1.2 px, and the oracle
                    # shrugged both it and a frozen version off.
                    hit = np.random.random(len(scored)) < heavy
                    scored = [(v + (float(np.random.normal(0.0, noise)) if h else 0.0),
                               n, pl)
                              for (v, n, pl), h in zip(scored, hit, strict=True)]
                elif frozen:
                    rng = np.random.default_rng(
                        (int(mario_x(env)) // 8) * 977 + len(scored))
                    scored = [(v + float(rng.normal(0.0, noise)), n, pl)
                              for v, n, pl in scored]
                else:
                    scored = [(v + float(np.random.normal(0.0, noise)), n, pl)
                              for v, n, pl in scored]
            if fixed and rescue:
                # The habit does the work; the console is consulted only about
                # whether it is about to be fatal. If choosing well really is
                # "know when not to jump", this recovers most of what choosing
                # is worth, and the part it recovers is a binary label.
                keep = next(t for t in scored if t[1] == fixed)
                if keep[0] > DEATH / 2:
                    scored = [(1.0 if n == fixed else 0.0, n, pl)
                              for _, n, pl in scored]
            if gate:
                # A binary gate's two errors do not cost the same, so they get
                # separate rates: `fp` jumps when the console says defer, `fn`
                # defers when it says jump. One number cannot show that.
                only = [t for t in scored if t[1] in ("bc", "jump now")]
                if len(only) == 2:
                    pick = max(only, key=lambda t: t[0])
                    other = min(only, key=lambda t: t[0])
                    rate = gate_fp if pick[1] == "bc" else gate_fn
                    keep = other if np.random.random() < rate else pick
                    scored = [(1.0 if t is keep else 0.0, t[1], t[2])
                              for t in scored]
            if corrupt and np.random.random() < corrupt:
                # A perfect scorer that is wrong this often. Before asking a
                # student to imitate an argmax, it is worth knowing how good
                # an imitator has to be before imitating helps at all.
                wrong = [t for t in scored if t[1] != max(scored,
                                                          key=lambda q: q[0])[1]]
                if wrong:
                    scored = [(1.0, *wrong[np.random.randint(len(wrong))][1:])] \
                        + [(0.0, n, pl) for _, n, pl in scored]
            best = max(scored, key=lambda t: t[0])
            bc_score = next(s for s, n, _ in scored if n == "bc")
            score, name, plan = (best if best[0] > bc_score + margin
                                 else next(t for t in scored if t[1] == "bc"))
            chosen[name] += 1
            if rollback and best[0] < DEATH / 2 \
                    and (rb_block is None or lives != rb_block):
                if len(hist) > rb_depth + 1:
                    rb_depth += 1
                    if rb_depth == 1:
                        rb["attempts"] += 1
                    # the death is within one lookahead of here; stay rich
                    # until the game clock is past it
                    danger_until = max(danger_until, gt + horizon + tail)
                    here_, stack_, unw, obs, best_x, deaths, lives, gt0 = \
                        hist[-(rb_depth + 1)]
                    env.load_state(here_)
                    policy._stack = list(stack_)
                    _UNWRAP.clear()
                    _UNWRAP.update(unw)
                    rb["discarded"] += gt - gt0
                    # the re-plan must still reach the death it is avoiding
                    rb_h = horizon + (gt - gt0)
                    gt = gt0
                    held, defer = [], 0
                    continue
                # nothing survives from as far back as we keep: play the
                # least bad plan and do not try again on this life
                rb["failed"] += 1
                rb_block, rb_h, rb_depth, danger_until = lives, 0, 0, 0
            elif rb_h:
                # a survivor from the rolled-back state: committed like any
                # other plan, with the rich set kept on through the window
                rb["rescued"] += 1
                rb["depths"].append(rb_depth)
                rb_h = 0
            # Choosing the policy can mean two things, and they are not worth
            # the same. Replaying the sequence that was scored commits to it
            # for the whole window. Handing the wheel back lets the policy
            # re-decide every `repeat` frames, so what gets played is not what
            # was valued — and the same probe drew in a harness that did that
            # and won by 348 in one that did not.
            if name == "bc" and bc_live:
                held, defer = [], commit
            else:
                held = list(plan[:commit])
            last_choice, last_score = name, score
            if view is not None:
                from nes_player.cli.runtime import action_entropy

                thoughts = [f"{n:11s} {v:+8.0f}" + ("   <-- taken" if n == name
                                                    else "")
                            for v, n, _ in sorted(scored, key=lambda t: -t[0])]
                if last_ranked:
                    entropy_hist.append(action_entropy(last_ranked))
                    del entropy_hist[:-500]

        if held:
            pressed = held.pop(0)
        else:
            defer = max(0, defer - 1)
            if i % repeat == 0:
                pressed, _ = policy.act(obs.frame_rgb, temperature)
                pressed = pressed - {"START", "SELECT"}
        obs = env.step_buttons([pressed])
        gt += 1
        if view is not None:
            for ev in ears.push(obs.audio_pcm, i):
                sounds.add(ev.cluster_id, ears.clusters[ev.cluster_id].heard)
                if locator is not None:
                    sm = locator.sound_map(obs.frame_rgb)
                    py_, px_ = np.unravel_index(int(sm.argmax()), sm.shape)
                    pings.add((px_ + 0.5) / sm.shape[1],
                              (py_ + 0.5) / sm.shape[0])
            sounds.tick()
            pings.tick()
            # Grad-CAM costs a backward pass, so not on every frame; it does
            # not advance the policy's stack, which is why it can be called at
            # all while a committed plan is being played.
            if i % 4 == 0:
                cam = policy.compute_cam(obs.frame_rgb)
                if ghost_view is not None and hero is not None:
                    mask = sum(1 << k for k, b in enumerate(BUTTONS)
                               if b in pressed)
                    ghost_path = ghost_view.predict(
                        obs.frame_rgb, hero.cx, hero.cy, (hero.vx, hero.vy),
                        mask, steps=12)
            # Only the frames that were played. The branches rewind the
            # console, and recording them would show a controller flickering
            # through futures it never took.
            view.show(obs, (pressed,), info={
                "arm": f"oracle h={horizon} tail={tail} draws={draws}",
                "plan": last_choice, "value": f"{last_score:.0f}",
                "progress": str(progress_of(obs.debug or {})),
                "deaths": str(deaths),
                "branch frames": f"{branch_frames:,}"},
                thoughts=thoughts, action_probs=last_ranked, slots=slots,
                entropy_hist=entropy_hist, verdicts=verdicts, heatmap=cam,
                features=policy.last_features,
                gallery=[(c.proto, c.verdict, c.cluster_id, c.seen)
                         for c in sorted(memory.clusters,
                                         key=lambda c: -c.seen)[:8]],
                audio_events=[(e[0], e[1], ears.clusters[e[0]].verdict)
                              for e in sounds.events],
                ghost=ghost_path, sound_pings=pings.pings)

        d = obs.debug or {}
        now = d.get("lives")
        if lives is not None and now is not None and now < lives:
            deaths += 1
        lives = now
        # Folded across levels. While no arm ever finished 1-1 the camera's x
        # was enough; now 26 of 32 runs do, and their counter resets to zero on
        # the next level, so the metric was quietly capping every good run at
        # about 3120 and hiding whatever happened afterwards.
        best_x = max(best_x, progress_of(d)
                     if game.startswith("SuperMario")
                     else game_pos(env, game))
        # Contra-family: after the last life the game sits on the continue
        # screen with the camera back at zero. A human presses START; so
        # does the runner, and the credit is counted. The level restarts
        # from its beginning — continues buy attempts, not position.
        if save_at and save_final and not saved_at_done \
                and (game_pos(env, game) if save_at >= 4000
                     else game_pos(env, game) % 4000) >= save_at:
            Path(save_final).write_bytes(env.save_state())
            saved_at_done = True
        if game.startswith("Contra"):
            ram_ = env._env.get_ram()
            wmax = max(wmax, int(ram_[170]))
            if game_pos(env, game) % 4000 >= 3070:
                # typed HP of the wall's objects; a kill is still only the
                # level counter, which game_pos folds in at 4000
                hp_ = wall_hp(ram_)
                if hp_arrival is None:
                    hp_arrival, hp_min = hp_, hp_
                hp_min = min(hp_min, hp_)
        if game.startswith("Contra") or game.startswith("SuperC"):
            xs_now = game_pos(env, game) % 4000
            if xs_now > 1000:
                deep_in_level = True
            # Game over resets the camera to 0 *and* lives to 0. The base
            # stages never scroll horizontally, so the camera alone once
            # read a freshly cleared level as a game over, pressed START
            # into a live game and left it paused (2026-09-03, seen in the
            # first level-2 recording).
            game_over = (deep_in_level and xs_now == 0
                         and int(lives or 0) <= 0)
            stuck = stuck + 1 if game_over else 0
            if stuck > 120:
                for k2 in range(600):
                    pulse = k2 % 60 in (0, 1)
                    obs = env.step_buttons(
                        [frozenset({"START"}) if pulse else frozenset()])
                    if int((obs.debug or {}).get("lives", 0) or 0) > 0:
                        break
                for _ in range(600):
                    obs = env.step_buttons([frozenset()])
                lives = (obs.debug or {}).get("lives")
                credits_used += 1
                deep_in_level = False
                stuck = 0
                held = []
                defer = 0
    if view is not None:
        view.close()
    if save_final and not save_at:
        Path(save_final).write_bytes(env.save_state())
    if pool is not None:
        pool.close()
        pool.join()
    env.close()
    return {"seed": seed, "best_x": best_x, "deaths": deaths,
            "branch_frames": branch_frames, "chosen": dict(chosen),
            **({"credits": credits_used} if credits_used else {}),
            **({"wmax": wmax, "wall_hp_arrival": hp_arrival,
                "wall_hp_min": hp_min}
               if game.startswith("Contra") else {}),
            **({"escalated": escalated, "decisions": decisions}
               if adaptive is not None else {}),
            **({"rollback": rb} if rollback else {})}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--game", default="SuperMarioBros-Nes-v0")
    ap.add_argument("--state", default="default")
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0,
                    help="first seed; a margin tuned on one set of seeds "
                         "has to be confirmed on another")
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--commit", type=int, default=16)
    ap.add_argument("--wall-unclipped", action="store_true",
                    help="Contra: damage term without the max(0, 72 - HP) "
                         "clip, so turrets sharing the cannons' type do not "
                         "mute it at arrival")
    ap.add_argument("--rollback", type=int, default=0,
                    help="B1': when every candidate is doomed, rewind up to "
                         "this many decisions (16 frames each), re-plan with "
                         "a horizon grown by the depth, commit the first "
                         "surviving plan whole")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--horizons", type=int, nargs="+", default=[48, 96, 144])
    ap.add_argument("--knn-memory", default="",
                    help="steer by the oracle's stored decisions on this "
                         "level, keyed by world x — the episodic-memory arm")
    ap.add_argument("--gate", action="store_true",
                    help="restrict the arm to {bc, jump now}; with no\n                         corruption this is the gate's ceiling")
    ap.add_argument("--gate-fp", type=float, default=0.0,
                    help="binary gate over {bc, jump now}: jump this\n                         often when the console says defer")
    ap.add_argument("--gate-fn", type=float, default=0.0,
                    help="...and defer this often when it says jump")
    ap.add_argument("--corrupt", type=float, default=0.0,
                    help="take a wrong candidate this often, to price\n                         how accurate an imitator has to be")
    ap.add_argument("--rescue", action="store_true",
                    help="with --fixed: keep the habit unless the console\n                         says it is fatal, then take the best "
                         "survivor")
    ap.add_argument("--fixed", default="",
                    help="always take this template, whatever the state — \n                         the standing-habit control for a learned scorer")
    ap.add_argument("--video", default=None,
                    help="record the frames actually executed — not the\n                         branches — through the dashboard, to this file")
    ap.add_argument("--draws", type=int, default=1,
                    help="how many futures to average the tail over, "
                         "so the value is an expectation rather than "
                         "one realised continuation")
    ap.add_argument("--load-state", default="",
                    help="start from this saved emulator state instead of "
                         "the game's own beginning — a lab tool for boss "
                         "work; grants two lives")
    ap.add_argument("--auto-templates", nargs="?", const="probe", default="",
                    help="assemble the candidate set from a measurement "
                         "instead of the hand-written one: 'probe' (the "
                         "button probe, circular) or 'scan' (the causal "
                         "controllability scan, no RAM map)")
    ap.add_argument("--pos-from-scan", action="store_true",
                    help="take the progress position from the scan's own "
                         "consistent position bytes instead of a hand rule")
    ap.add_argument("--workers", type=int, default=1,
                    help="emulator processes scoring candidates in "
                         "parallel; a CRN run reproduces the serial one "
                         "byte for byte")
    ap.add_argument("--save-at", type=int, default=0,
                    help="with --save-final: save the state the first time "
                         "pure in-level position crosses this x, instead of "
                         "at the end of the run")
    ap.add_argument("--save-final", default="",
                    help="write the emulator state at the last frame to this "
                         "file, so a later probe can start where the run "
                         "ended — e.g. at a boss wall")
    ap.add_argument("--weapon-px", type=int, default=400,
                    help="px the Contra value credits for holding spread "
                         "(half for mid-tier); 0 turns the term off")
    ap.add_argument("--escapes", action="store_true",
                    help="add the three rescue compositions the two-step "
                         "search actually used to the plain template set; "
                         "use with --horizons 96")
    ap.add_argument("--death-price", type=float, default=0.0,
                    help="px subtracted from a draw that dies, instead of "
                         "the majority-death veto; prices P(death) into the "
                         "value where the veto makes minority deaths free")
    ap.add_argument("--two-step", action="store_true",
                    help="search ordered pairs of the five behaviours at half "
                         "the horizon each (25 pairs + the policy's own "
                         "plan), so compositions like run-up-then-jump "
                         "become expressible; use with --horizons 96")
    ap.add_argument("--crn", action="store_true",
                    help="common random numbers for the continuation draws, "
                         "keyed by (seed, world x, candidate, draw index): "
                         "arms sharing a seed stay frame-identical until a "
                         "genuine policy difference, so their paired "
                         "difference is causal and not draw noise")
    ap.add_argument("--adaptive", type=float, default=None,
                    help="escalate from --draws to twice that many where the "
                         "frozen trigger — winner instability first, then "
                         "expected stopping regret under a per-point sigma — "
                         "exceeds this tau; 26.709 is the tau frozen on the "
                         "five non-1-1 battery levels at target fraction 0.5")
    ap.add_argument("--adaptive-g2", type=float, default=1255.02,
                    help="global single-draw pairwise variance the per-point "
                         "sigma is shrunk toward, from the same five levels")
    ap.add_argument("--tail-temp", type=float, default=0.9,
                    help="temperature of the continuation the value is\n                         measured under; at 0 the score reproduces")
    ap.add_argument("--defer", type=float, default=0.0,
                    help="hand the decision back to the policy when fewer\n                         than this fraction of an ensemble's members agree\n                         on the best candidate")
    ap.add_argument("--ram-hero", action="store_true",
                    help="take the hero from console memory instead of the "
                         "sprite tracker — the identity ablation, and only "
                         "meaningful with a probe trained the same way")
    ap.add_argument("--no-oracle", action="store_true",
                    help="skip the save-state arms and run only bc against "
                         "--probe or --ghost, when the ceiling is already "
                         "measured on these seeds and costs an hour to repeat")
    ap.add_argument("--tail", type=int, nargs="+", default=[0],
                    help="frames of fixed continuation after the plan, so the "
                         "score is a value and not an open-loop template's "
                         "own progress")
    ap.add_argument("--tail-from", type=int, default=0,
                    help="how much of the plan to play before continuing: the "
                         "commitment (the consequence of the decision) or the "
                         "whole horizon (a classic terminal value)")
    ap.add_argument("--bc-live", action="store_true",
                    help="when the policy's option wins, hand it the wheel to "
                         "re-decide, instead of replaying the sequence that "
                         "was scored")
    ap.add_argument("--probe", default=None,
                    help="score the same candidates with a plan-value probe, "
                         "inside this harness, so only the scorer differs")
    ap.add_argument("--heavy", type=float, default=0.0,
                    help="fraction of scores that get the error at all; the "
                         "rest are exact, which is the shape the probe's own "
                         "error has")
    ap.add_argument("--frozen", action="store_true",
                    help="make the injected error a function of position, so "
                         "the same spot is always misjudged the same way")
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0],
                    help="px of gaussian error added to the oracle's own plan "
                         "values, to price how much accuracy the margin needs")
    ap.add_argument("--margin", type=float, nargs="+", default=[0.0],
                    help="px a plan must beat the policy's own by to override it")
    ap.add_argument("--ghost", default=None,
                    help="also run an arm scoring the same candidates with this "
                         "learned ego model, to price the model against the "
                         "objective")
    args = ap.parse_args()
    global WEAPON_MAIN, WALL_CLIP
    WEAPON_MAIN = args.weapon_px
    WALL_CLIP = not args.wall_unclipped
    if args.pos_from_scan:
        cam = Path("runs/knowledge") / f"camera_{args.game}.json"
        if cam.exists():
            SCAN_POS[args.game] = json.loads(cam.read_text())
        else:
            rep = json.loads((Path("runs/knowledge") / f"control_{args.game}.json")
                             .read_text())
            SCAN_POS[args.game] = rep["position_bytes_consistent"]
        print("pos-from-scan:", SCAN_POS[args.game])
    if args.state in ("none", ""):
        # power-on boot, for integrations that ship no default savestate
        args.state = None

    plans = [(None, None, 0.0, 0.0, None, 0)] + ([] if args.no_oracle else [
        (h, None, 0.0, nz, None, tl)
        for h in args.horizons for nz in args.noise for tl in args.tail])
    if args.probe:
        plans += [(h, None, 0.0, 0.0, args.probe, 0) for h in args.horizons]
    if args.knn_memory:
        plans += [(h, None, 0.0, 0.0, None, 0) for h in args.horizons]
    if args.ghost:
        plans += [(h, args.ghost, m, 0.0, None, 0)
                  for h in args.horizons for m in args.margin]
    arms: dict[str, list[dict]] = {}
    for horizon, ghost, margin, noise, use_probe, tail in plans:
        name = ("bc" if horizon is None
                else f"{'learned' if ghost else 'oracle'} h={horizon}"
                     + (f" m={margin:g}" if ghost else "")
                     + (f" noise={noise:g}" if noise else "")
                     + (" frozen" if noise and args.frozen else "")
                     + (f" heavy={args.heavy:g}" if noise and args.heavy else ""))
        if tail:
            name += f" tail={tail}@{args.tail_from or horizon}"
        if args.adaptive is not None and horizon:
            name += f" adaptive={args.adaptive:g}"
        if args.crn and horizon:
            name += " crn"
        if args.auto_templates and horizon:
            name += f" auto={args.auto_templates}"
        if args.rollback and horizon:
            name += f" rb={args.rollback}"
        if args.wall_unclipped and horizon:
            name += " wall-unclipped"
        if args.pos_from_scan and horizon:
            name += " scanpos"
        if args.two_step and horizon:
            name += " two-step"
        if args.death_price and horizon:
            name += f" dp={args.death_price:g}"
        if args.escapes and horizon:
            name += " escapes"
        if use_probe:
            name = f"probe h={horizon}"
        if args.gate and horizon:
            name += f" gate fp={args.gate_fp:g} fn={args.gate_fn:g}"
        if args.knn_memory and horizon:
            name = "knn memory"
        if args.corrupt and horizon:
            name += f" corrupt={args.corrupt:g}"
        if args.fixed and horizon:
            # The scoring is short-circuited, so whatever this arm was called,
            # what it does is take the same template every time.
            name = (f"rescue {args.fixed}" if args.rescue
                    else f"always {args.fixed}")
        rows = []
        for seed in range(args.seed0, args.seed0 + args.runs):
            row = run(args.checkpoint, args.game, args.state, args.frames, seed,
                      args.temperature, args.repeat, horizon, args.commit, ghost,
                      margin, noise, args.frozen, args.heavy,
                      use_probe, args.bc_live, tail,
                      args.tail_from or (horizon or 0), args.ram_hero,
                      args.defer, args.tail_temp, args.draws, args.video,
                      args.fixed, args.rescue, args.corrupt, args.knn_memory,
                      args.gate,
                      args.gate_fp, args.gate_fn,
                      None if horizon is None else args.adaptive,
                      args.adaptive_g2, args.crn, args.two_step,
                      args.death_price, args.escapes, args.save_final,
                      args.load_state, args.save_at, args.workers,
                      args.auto_templates,
                      rollback=args.rollback if horizon else 0)
            rows.append(row)
            print(json.dumps({"arm": name, **row}), flush=True)
        arms[name] = rows

    base = np.array([r["best_x"] for r in arms["bc"]], float)
    print()
    for name, rows in arms.items():
        a = np.array([r["best_x"] for r in rows], float)
        d = a - base
        sd = d.std(ddof=1) if len(d) > 1 else 0.0
        t = d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else 0.0
        print(f"{name:14} best_x mean {a.mean():7.1f}  median {np.median(a):7.1f}  "
              f"min {a.min():6.0f}  deaths {sum(r['deaths'] for r in rows):3d}  "
              f"vs bc {d.mean():+8.1f}  t={t:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
