"""A5: find the game's object tables — what has hit points, and what it is.

Contra's wall cost a day of RAM archaeology and one retraction: three of
the four bytes I had called "wall HP" were object *type* bytes, and the
sum I reported as damage was a constant of one savestate. The tables were
there all along — type per slot at 0x530, hit points per slot at 0x580,
sixteen slots — and nothing about finding them was specific to Contra.

This scan asks the console instead. From one save-state it runs two
synchronous branches, identical but for the trigger: one doing nothing,
one firing the chord A0 said fires (tapped, if A0 said tapped). Bytes
that fall in the firing branch and not in the idle one are hit points of
whatever is being shot. Their addresses are then folded into arrays: a
byte that counts down in small steps is hit points; the fields wiped when
that object dies name the parallel table its type lives in.

    uv run python scripts/experiments/object_tables.py ContraJ-Nes-v0 \
        --load-state runs/oracle_adaptive/contra_wall.state
    uv run python scripts/experiments/object_tables.py RushnAttack-Nes-v0

Writes runs/knowledge/objects_<game>.json. The output is a hypothesis with
its evidence, not a fact: a byte that falls under fire may be an ammo
counter, and the honest test of a damage signal is still whether killing
the thing advances the game.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle_mpc import begin_any  # noqa: E402

FIRE_FALLBACK = frozenset({"B"})


def fire_chords(game: str) -> list:
    """Every way A0 saw this game shoot, plus the aims it did not try.

    One chord is not enough: Contra's prone fire goes along the ground and
    reaches the lower cannon only, so a scan that shoots once finds one
    slot and calls the table a single object. Firing every way and taking
    the union of what falls is what a person does with a new boss.
    """
    out, seen = [], set()

    def add(chord, mode):
        key = (tuple(sorted(chord)), mode)
        if key not in seen:
            seen.add(key)
            out.append((frozenset(chord), mode))

    p = Path("runs/knowledge") / f"control_{game}.json"
    if p.exists():
        rep = json.loads(p.read_text())
        scored = []
        for r in rep.values():
            if isinstance(r, dict) and r.get("controllable"):
                for key, v in r["fire"].items():
                    scored.append((v["pixels_off_body"], key))
        for px, key in sorted(scored, reverse=True):
            if px <= 0:
                continue
            chord, mode = key.split("/")
            add(chord.split("+"), mode)
    if not out:
        add(FIRE_FALLBACK, "tap")
    # The aims a probe of single chords never reaches: a gun that fires
    # forward also fires up and diagonally. The base must be the trigger
    # alone — A0's best chord here was prone fire, and keeping its DOWN
    # while adding UP left the soldier prone, shooting along the ground,
    # never at the wall's upper cannon.
    base = {b for b in out[0][0] if b not in ("UP", "DOWN", "LEFT", "RIGHT")}
    base |= {"B"}
    mode = out[0][1]
    for extra in (("UP",), ("UP", "RIGHT"), ("RIGHT",), ()):
        add(base | set(extra), mode)
    return out


def branch(env, state, chord, mode: str, frames: int):
    """Run `frames` from `state`, pressing `chord` per `mode`.

    Returns the RAM per frame, cut at the frame a life is lost: after a
    death the game clears half its memory, and every one of those bytes
    "falls under fire" in exactly one step. That is how a chord containing
    RIGHT — which walks the soldier into the wall — produced 259 hit-point
    candidates.
    """
    env.load_state(state)
    ram, l0 = [], None
    for k in range(frames):
        if chord is None:
            press = frozenset()
        elif mode == "tap":
            press = chord if k % 4 < 2 else frozenset()
        else:
            press = chord
        o = env.step_buttons([press])
        now = (o.debug or {}).get("lives")
        if l0 is None:
            l0 = now
        if l0 is not None and now is not None and now < l0:
            break
        ram.append(env._env.get_ram().copy())
    return np.stack(ram).astype(int) if ram else np.zeros((1, 2048), int)


MAX_STEP = 16          # a hit takes a few points, not half the byte


def falls(idle: np.ndarray, fire: np.ndarray, min_drop: int) -> list:
    """Bytes that go down under fire, in repeated small steps, not while idle.

    Repeated and small is the whole discriminator: damage arrives one hit
    at a time. A byte that drops 193 to zero in a single frame is memory
    being cleared, not a thing being shot.
    """
    # each branch is judged over its own window: standing still under a
    # boss's fire is short-lived, and truncating the firing branch to the
    # idle one's length hid the very steps this scan looks for
    out = []
    for a in range(fire.shape[1]):
        f, i = fire[:, a], idle[:, a]
        d = np.diff(f)
        if not (d <= 0).all():
            continue
        drop = int(f[0] - f[-1])
        if drop < min_drop or f[0] == 0 or f[0] > 250:
            continue
        if (d < 0).sum() < 2 or int(-d.min()) > MAX_STEP:
            continue
        # Rates, not totals: the branches have different lengths (standing
        # still under fire is short-lived), so a clock ticking down at the
        # same speed in both looked like damage in the longer one. Rush'n
        # Attack answered with seventeen copies of one such counter.
        fire_rate = drop / max(1, len(f) - 1)
        idle_rate = (int(i[0] - i[-1])) / max(1, len(i) - 1)
        if fire_rate <= 2 * idle_rate:
            continue
        out.append({"addr": a, "from": int(f[0]), "to": int(f[-1]),
                    "steps": int((d < 0).sum()),
                    "sizes": sorted({int(-x) for x in d[d < 0]}),
                    "idle_drop": int(i[0] - i[-1]),
                    "rate": round(fire_rate, 4),
                    "idle_rate": round(idle_rate, 4)})
    return out


def drop_mirrors(cands: list, fire: np.ndarray, limit: int = 4) -> tuple:
    """Discard values that fall in lockstep at many addresses.

    Damage is not synchronised: two objects shot in the same second do not
    lose their last point on the same frame. A value that falls
    identically at a dozen addresses is one counter mirrored through a
    table — Rush'n Attack answered the first version of this scan with
    seventeen copies of a scroll counter, each an immaculate 215 -> 105 in
    110 steps of one.
    """
    groups: dict = {}
    for c in cands:
        key = tuple(fire[:, c["addr"]].tolist())
        groups.setdefault(key, []).append(c)
    keep, mirrored = [], []
    for members in groups.values():
        (mirrored if len(members) >= limit else keep).extend(members)
    return keep, mirrored


def classify(cands: list) -> tuple:
    """Split what falls into hit points and fields cleared on death.

    This is the distinction the wall retraction turned on. A hit-point
    byte is counted down: many steps, each a hit's worth. An object's type
    and flags do not decrease — they are wiped in one or two lumps when
    the thing dies, which reads as a fall if you only look at the ends.
    """
    hp = [c for c in cands if c["steps"] >= 3 and max(c["sizes"]) <= 4]
    cleared = [c for c in cands if c not in hp and c["to"] == 0]
    return hp, cleared


def tables_around(fire: np.ndarray, hp_addr: int, slots: int,
                  cleared: list) -> dict:
    """Given one hit-point byte, name its table and the type table beside it.

    NES object tables are arrays of one field, aligned and indexed by slot,
    so the hit points of slot k sit at H+k and its type at T+k. H is the
    aligned block holding the byte we watched count down. T is any other
    aligned block whose entries hold still while the hit points fall and
    which is non-zero in exactly the slots H is — the same objects, seen
    through a different field. Busy slots alone are not enough: a block of
    text bytes sitting in RAM matched all sixteen. The table must also
    hold the fields that were wiped when the objects died.
    """
    hp_base = hp_addr - hp_addr % slots
    early = fire[: max(2, len(fire) // 4)]
    busy = fire[0, hp_base:hp_base + slots] != 0
    best = None
    for base in range(0, fire.shape[1] - slots, slots):
        if base == hp_base:
            continue
        vals = fire[0, base:base + slots]
        if int((vals != 0).sum()) < 2:
            continue
        # every slot that holds a type must hold hit points; the converse
        # fails on real data — an emptied slot keeps its last hit-point
        # value until something is put there again
        occupied = vals != 0
        if not (busy | ~occupied).all():
            continue
        agree = int((occupied == busy).sum())
        wiped = sum(1 for c in cleared if base <= c["addr"] < base + slots)
        if cleared:
            # the fields that died with their objects are the evidence; a
            # slot's type is not constant (slots are reused), so requiring
            # constancy rejected the real table and kept a block of text
            if wiped < 2:
                continue
            score = (wiped, agree)
        else:
            cols = early[:, base:base + slots]
            if not (cols == cols[0]).all():
                continue
            score = (0, agree)
        if best is None or score > best["score"]:
            best = {"type_base": int(base), "hp_base": int(hp_base),
                    "slots": slots, "agree": agree, "wiped": wiped,
                    "score": score,
                    "types": [int(x) for x in vals],
                    "hit_points": [int(x) for x in
                                   fire[0, hp_base:hp_base + slots]]}
    if best:
        best.pop("score")
    return best or {"hp_base": int(hp_base), "slots": slots,
                    "hit_points": [int(x) for x in
                                   fire[0, hp_base:hp_base + slots]]}


def main() -> int:
    from nes_player.emulator.stable_retro import StableRetroAdapter

    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--load-state", default="",
                    help="start from this savestate instead of the game's "
                         "own beginning — a boss is not at the boot screen")
    ap.add_argument("--approach", type=int, default=0,
                    help="frames of the forward chord before the probe, to "
                         "walk into range of whatever is ahead")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--min-drop", type=int, default=4)
    ap.add_argument("--slots", type=int, default=16)
    args = ap.parse_args()

    integ_root = Path(__file__).resolve().parents[2] / "integrations"
    integ = str(integ_root) if (integ_root / args.game).exists() else None
    env = StableRetroAdapter(args.game, include_debug=True, state=None,
                             integration_dir=integ)
    if args.load_state:
        env.reset(seed=0)
        env.load_state(Path(args.load_state).read_bytes())
        env._env.data.set_value("lives", 2)
    else:
        begin_any(env, args.game)
    if args.approach:
        for _ in range(args.approach):
            env.step_buttons([frozenset({"B", "RIGHT"})])

    here = env.save_state()
    idle = branch(env, here, None, "tap", args.frames)
    cands, by_addr, fire = [], {}, None
    for chord, mode in fire_chords(args.game):
        f = branch(env, here, chord, mode, args.frames)
        hit = falls(idle, f, args.min_drop)
        name = f"{'+'.join(sorted(chord))}/{mode}"
        print(f"{name}: {len(hit)} byte(s) fall")
        if fire is None or len(hit) > 0 and len(hit) > len(cands):
            fire = f          # the branch that saw the most, for the tables
        for c in hit:
            c["chord"] = name
            if c["addr"] not in by_addr or c["from"] - c["to"] > \
                    by_addr[c["addr"]]["from"] - by_addr[c["addr"]]["to"]:
                by_addr[c["addr"]] = c
        cands = list(by_addr.values())
    cands.sort(key=lambda c: c["addr"])
    cands, mirrored = drop_mirrors(cands, fire)
    if mirrored:
        print(f"discarded {len(mirrored)} lockstep copies of "
              f"{len({tuple(m['sizes']) + (m['from'],) for m in mirrored})} "
              f"value(s): a mirrored counter, not damage")
    hp, cleared = classify(cands)
    print(f"bytes falling under fire only (union): {len(cands)} — "
          f"{len(hp)} counted down, {len(cleared)} cleared on death")
    for c in cands[:12]:
        print(f"  0x{c['addr']:03x}: {c['from']} -> {c['to']} in "
              f"{c['steps']} steps, sizes {c['sizes']}")

    for c in hp:
        print(f"  HP    0x{c['addr']:03x}: {c['from']} -> {c['to']} in "
              f"{c['steps']} steps of {c['sizes']} [{c['chord']}]")
    for c in cleared:
        print(f"  field 0x{c['addr']:03x}: {c['from']} -> 0 in {c['steps']}")
    pair = tables_around(fire, hp[0]["addr"], args.slots, cleared) \
        if hp else {}
    if pair.get("type_base") is not None:
        print(f"tables: types at 0x{pair['type_base']:03x}, hit points at "
              f"0x{pair['hp_base']:03x}, {pair['slots']} slots, "
              f"{pair['agree']}/{pair['slots']} slots agree on who is busy")
        print("  types:", pair["types"])
        print("  hp:   ", pair["hit_points"])
    elif pair:
        print(f"hit-point table at 0x{pair['hp_base']:03x}, no aligned type "
              f"table matched: {pair['hit_points']}")
    report = {"game": args.game, "hp": hp, "cleared_on_death": cleared,
              "tables": pair, "candidates": cands}
    out = Path("runs/knowledge") / f"objects_{args.game}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("wrote", out)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
