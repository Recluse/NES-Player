#!/usr/bin/env python3
"""Does destroying the wall end the zone? Ask the console.

The prior prices the wall's hit points as "worth the rest of the stage".
That is a claim about the world, not about arithmetic, and the grounding
table will not call the wall `scan_confirmed` until something like this
has been observed:

    wall_hp (types 4/16/17 at 0x530/0x580) falls to 0 under fire
    AND the screen-index byte 100 resets (12 -> 0)
    AND the picture fades to black

A stage that ends any other way, or a wall that can be skipped, is a
retraction waiting to happen. The check is cheap and it is the difference
between a priced target and a named one.

    uv run python scripts/verify_wall_zone_end.py --game ContraJ-Nes-v0

Writes runs/knowledge/wall_zone_end_<game>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from oracle_mpc import WALL_HP0, WALL_TYPES, wall_hp  # noqa: E402

from nes_player.emulator.stable_retro import StableRetroAdapter  # noqa: E402

FADE_BELOW = 40.0     # the same brightness the base-room script calls a fade
FIRE = frozenset({"B", "RIGHT", "UP"})   # the chord the objects scan used


def verify(game: str, integrations: str | None, state: str,
           fire_budget: int = 2000) -> dict:
    em = StableRetroAdapter(game, integration_dir=integrations)
    em.reset()
    em.load_state(Path(state).read_bytes())

    def reading(tag: str, obs) -> dict:
        ram = em._env.get_ram()
        return {
            "tag": tag,
            "wall_hp": wall_hp(ram),
            "level": int(ram[48]),
            "screen_index": int(ram[100]),
            "screen_shadow": int(ram[405]),
            "brightness": round(float(obs.frame_rgb.mean()), 2),
        }

    obs = em.step_buttons([frozenset()])
    start = reading("start", obs)
    if start["wall_hp"] <= 0:
        em.close()
        return {"game": game, "state": state, "ok": False,
                "why": "wall already dead in this state", "start": start}

    wall_died_at = None
    death = None
    for i in range(fire_budget):
        pressed = FIRE if i % 4 < 2 else frozenset()
        obs = em.step_buttons([pressed])
        if wall_hp(em._env.get_ram()) == 0:
            wall_died_at = i
            death = reading("wall_dead", obs)
            break
    if death is None:
        em.close()
        return {"game": game, "state": state, "ok": False,
                "why": f"wall still at {start['wall_hp']} HP after "
                       f"{fire_budget} frames of fire",
                "start": start}

    # Wait out the fade and record how dark it goes.
    faded = None
    for i in range(120):
        obs = em.step_buttons([frozenset()])
        row = reading(f"post_{i}", obs)
        if row["brightness"] < FADE_BELOW and faded is None:
            faded = {**row, "frame": i}
    end = reading("end", obs)

    screen_reset = end["screen_index"] != start["screen_index"]
    hit_points = start["wall_hp"]
    ok = bool(
        death["wall_hp"] == 0
        and screen_reset
        and faded is not None
    )
    em.close()
    return {
        "game": game,
        "state": state,
        "ok": ok,
        "wall_types": sorted(WALL_TYPES),
        "wall_hp0": WALL_HP0,
        "hit_points_observed": hit_points,
        "fire_chord": sorted(FIRE),
        "wall_died_at_frame": wall_died_at,
        "start": start,
        "death": death,
        "faded": faded,
        "end": end,
        "checks": {
            "wall_hp_reached_zero": death["wall_hp"] == 0,
            "screen_index_reset": screen_reset,
            "screen_faded": faded is not None,
        },
        "verify_sentence": (
            f"wall_hp {hit_points}->0 under {sorted(FIRE)} in "
            f"{wall_died_at} frames; screen index "
            f"{start['screen_index']}->{end['screen_index']}; "
            f"brightness {start['brightness']}->{faded['brightness'] if faded else 'n/a'}"
            f" (fade below {FADE_BELOW})"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="ContraJ-Nes-v0")
    ap.add_argument("--integrations", default="integrations")
    ap.add_argument("--state",
                    default="runs/oracle_adaptive/contra_wall.state")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    result = verify(args.game, args.integrations, args.state)
    out = args.out or (ROOT / "runs/knowledge" / f"wall_zone_end_{args.game}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in
                      ("game", "ok", "hit_points_observed",
                       "wall_died_at_frame", "checks", "verify_sentence")
                      if k in result},
                     ensure_ascii=False, indent=2))
    print(f"-> {out}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
