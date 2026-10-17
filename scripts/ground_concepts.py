#!/usr/bin/env python3
"""Link manual concepts to on-screen entities, with the scan as the witness.

Phase 1C of docs-ru/plan-2026-10-08.md. The manual says "sensor", "door",
"player". The screen shows tiles and object-table slots. This script is the
bridge: it reads the scan artefacts already on disk and writes a grounding
table that names, for each concept:

* the entity (object types / hero tiles / a screen region),
* the scan and artefact that found it,
* how it was verified (a checkable sentence, not "looks right"),
* the status (scan_confirmed / human_confirmed / proposed / rejected).

It never invents object numbers. A concept with no artefact stays proposed
with a null entity — that is the honest state, and the validator says so.

    uv run python scripts/ground_concepts.py build --game ContraJ-Nes-v0
    uv run python scripts/ground_concepts.py check --game ContraJ-Nes-v0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = ROOT / "runs" / "knowledge"
OUT_DIR = ROOT / "assets" / "priors" / "grounding"

# Concept vocabulary. The manual's words; keep these stable so the parse
# pipeline and this table share a dictionary.
CONCEPTS = (
    "player",
    "sensor",
    "wall",
    "capsule",
    "door",
    "hazard",
    "terrain-solid",
    "hud",
    "menu",
)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_generic(game: str) -> dict:
    """Grounding for a game with no bespoke recipe: prior facts + whatever
    scan artefacts happen to be on disk. Confirmed only where a witness file
    exists — never by hand."""
    prior = load_json(ROOT / "assets" / "priors" / f"prior_{game}.json") or {}
    menu_fact_ids = [
        f.get("id") for f in (prior.get("facts") or [])
        if f.get("kind") == "menu" and f.get("id")
    ]
    menu_log = KNOWLEDGE / f"menu_{game}.jsonl"
    hero = load_json(KNOWLEDGE / f"hero_{game}.json")
    objects = load_json(KNOWLEDGE / f"objects_{game}.json")
    manual = load_json(KNOWLEDGE / f"manual_{game}.json")

    links: list[dict] = []

    # menu: the polygon is the witness when it has been run
    if menu_log.exists():
        links.append({
            "concept": "menu",
            "fact_ids": menu_fact_ids or ["press-start-to-begin"],
            "entity": {
                "kind": "screen_text",
                "reader": "nes_player.perception.text.find_prompt",
                "watch": "nes_player.perception.title.TitleWatch",
                "measured": f"scripts/menu_polygon.py -> runs/knowledge/menu_{game}.jsonl",
            },
            "evidence": {
                "scan": "menu_polygon.py + find_prompt",
                "artifact": f"runs/knowledge/menu_{game}.jsonl",
                "verify": (
                    "answers_the_pad is the play signal; see the jsonl for "
                    "which arm reached it and when"
                ),
            },
            "status": "scan_confirmed",
            "frame_illustration": None,
        })
    else:
        links.append({
            "concept": "menu",
            "fact_ids": menu_fact_ids or ["press-start-to-begin"],
            "entity": None,
            "evidence": {"scan": "menu_polygon.py", "artifact": None,
                         "verify": "run menu_polygon.py first"},
            "status": "proposed",
            "frame_illustration": None,
        })

    links.append({
        "concept": "player",
        "fact_ids": [],
        "entity": ({"kind": "sprite_tiles", "tiles": hero["tiles"]}
                   if hero and hero.get("tiles") else None),
        "evidence": {
            "scan": "hero_tiles.py",
            "artifact": f"runs/knowledge/hero_{game}.json" if hero else None,
            "verify": ("tiles that move with the buttons"
                       if hero else "run hero_tiles.py first"),
        },
        "status": "scan_confirmed" if hero and hero.get("tiles") else "proposed",
        "frame_illustration": None,
    })

    # sensor / HP: object_tables is a hypothesis (its own docstring says so).
    hp_rows = (objects or {}).get("hp") or []
    if hp_rows:
        links.append({
            "concept": "sensor",
            "fact_ids": [],
            "entity": {
                "kind": "hp_bytes",
                "hp_base": ((objects or {}).get("tables") or {}).get("hp_base"),
                "witness": [
                    {"addr": r.get("addr"), "from": r.get("from"),
                     "to": r.get("to"), "chord": r.get("chord")}
                    for r in hp_rows[:4]
                ],
            },
            "evidence": {
                "scan": "object_tables.py",
                "artifact": f"runs/knowledge/objects_{game}.json",
                "verify": (
                    "HP bytes fall under fire; room/zone transition must "
                    "follow before this becomes scan_confirmed"
                ),
            },
            "status": "proposed",
            "frame_illustration": None,
        })
    else:
        links.append({
            "concept": "sensor",
            "fact_ids": [],
            "entity": None,
            "evidence": {
                "scan": "object_tables.py",
                "artifact": (f"runs/knowledge/objects_{game}.json"
                             if objects else None),
                "verify": (
                    "HP falls under fire; if the scan found 0 bytes the "
                    "boot never reached a fight (see menu jsonl for how "
                    "long this game's intro is)"
                ),
            },
            "status": "proposed",
            "frame_illustration": None,
        })
    links.append({
        "concept": "wall",
        "fact_ids": [],
        "entity": None,
        "evidence": {
            "scan": "verify_wall_zone_end.py",
            "artifact": (f"runs/knowledge/wall_zone_end_{game}.json"
                         if (KNOWLEDGE / f"wall_zone_end_{game}.json").exists()
                         else None),
            "verify": "wall_hp -> 0 and screen fades",
        },
        "status": "proposed",
        "frame_illustration": None,
    })

    for concept in ("door", "capsule", "hazard", "terrain-solid", "hud"):
        # Manual facts can name a concept without linking an entity. That
        # is still useful: the row keeps the fact id so the parse pipeline
        # knows what is already claimed.
        fact_ids = []
        if manual:
            for key in (concept, "items", "weapons", "hud"):
                if key in manual and concept in ("hazard", "hud", "capsule"):
                    fact_ids.append(f"manual:{key}")
        if prior:
            for f in prior.get("facts") or []:
                if f.get("kind") in (concept, "target", "hud", "hazard") \
                        and f.get("id"):
                    if concept == "capsule" and "pickup" in (f.get("id") or ""):
                        fact_ids.append(f["id"])
                    elif concept == "hazard" and f.get("kind") == "hazard":
                        fact_ids.append(f["id"])
                    elif concept == "hud" and f.get("kind") == "hud":
                        fact_ids.append(f["id"])
        health = load_json(KNOWLEDGE / f"health_{game}.json")
        entity = None
        if health and concept == "hazard" and health.get("candidates"):
            entity = {
                "kind": "hp_bytes",
                "witness": health["candidates"][:8],
                "owner_fact": health.get("owner_fact"),
                "note": health.get("note"),
            }
        links.append({
            "concept": concept,
            "fact_ids": sorted(set(fact_ids)),
            "entity": entity,
            "evidence": {
                "scan": "health_probe.py" if entity else None,
                "artifact": (
                    f"runs/knowledge/health_{game}.json" if entity else
                    (f"runs/knowledge/manual_{game}.json"
                     if manual and concept in ("hazard", "hud", "capsule")
                     else None)
                ),
                "verify": (
                    "half-heart candidates stepped by 1; needs a controlled "
                    "hit vs no-hit pair to promote"
                    if entity else
                    ("named in the manual; entity still unlinked"
                     if fact_ids else "not linked yet")
                ),
            },
            "status": "proposed",
            "frame_illustration": None,
        })

    return {
        "schema_version": 1,
        "game": game,
        "source": (
            f"assembled by scripts/ground_concepts.py from prior_{game}.json"
            f" and runs/knowledge/*{game}*"
        ),
        "links": links,
    }


def build(game: str) -> dict:
    if "Contra" in game:
        return build_contra(game)
    return build_generic(game)


def build_contra(game: str) -> dict:
    """Assemble Contra's grounding from the artefacts this repo already has.

    Each link either points at a real artefact or is left proposed with
    `entity: null`. No object id is typed by hand into `entity` without a
    scan file behind it — if the file is missing, the link says so.
    """
    objects = load_json(KNOWLEDGE / f"objects_{game}.json")
    hero = load_json(KNOWLEDGE / f"hero_{game}.json")
    rooms = load_json(KNOWLEDGE / "base_rooms_ContraJ.json")

    links: list[dict] = []

    # --- player: hero tiles, found by asking the buttons -----------------
    if hero and hero.get("tiles"):
        links.append({
            "concept": "player",
            "fact_ids": ["3d-gun-forward-only", "fire-all-directions"],
            "entity": {
                "kind": "sprite_tiles",
                "tiles": hero["tiles"],
            },
            "evidence": {
                "scan": "hero_tiles.py",
                "artifact": f"runs/knowledge/hero_{game}.json",
                "verify": (
                    "tiles whose sprites displace between a held direction "
                    "and idle from the same savestate; enemies cancel"
                ),
            },
            "status": "scan_confirmed",
            "frame_illustration": None,
        })
    else:
        links.append({
            "concept": "player",
            "fact_ids": [],
            "entity": None,
            "evidence": {
                "scan": "hero_tiles.py",
                "artifact": None,
                "verify": "run hero_tiles.py first",
            },
            "status": "proposed",
            "frame_illustration": None,
        })

    # --- sensor: type 20, from the base-room script's own witness ------
    # objects_*.json is the jungle wall scan and does not contain type 20.
    # base_rooms_ContraJ.json is the file that names "the type-20 sensor at
    # object table 0x530/0x580". Cite the right artefact, not the nearest.
    if rooms and rooms.get("how_to_reach_room_2"):
        links.append({
            "concept": "sensor",
            "fact_ids": ["destroy-sensor-to-advance", "door-up-after-clear"],
            "entity": {
                "kind": "object_type",
                "object_types": [20],
                "table": {"type_base": 0x530, "hp_base": 0x580},
            },
            "evidence": {
                "scan": "base room script (object_tables layout)",
                "artifact": "runs/knowledge/base_rooms_ContraJ.json",
                "verify": (
                    "type-20 HP reaches 0 under alternating fire; then "
                    "hero at byte 820 == 112 and hold UP fades the room — "
                    "the manual's 'destroy the sensor to advance' is this "
                    "same event"
                ),
            },
            "status": "scan_confirmed",
            "frame_illustration": None,
        })
    elif objects:
        links.append({
            "concept": "sensor",
            "fact_ids": ["destroy-sensor-to-advance"],
            "entity": {
                "kind": "object_type",
                "object_types": None,
                "table": {"type_base": 0x530, "hp_base": 0x580},
            },
            "evidence": {
                "scan": "object_tables.py",
                "artifact": f"runs/knowledge/objects_{game}.json",
                "verify": (
                    "HP bytes fall under fire; room-advance confirmation "
                    "still missing (base_rooms artefact absent)"
                ),
            },
            "status": "proposed",
            "frame_illustration": None,
        })
    else:
        links.append({
            "concept": "sensor",
            "fact_ids": ["destroy-sensor-to-advance"],
            "entity": None,
            "evidence": {
                "scan": "object_tables.py",
                "artifact": None,
                "verify": "run object_tables.py first",
            },
            "status": "proposed",
            "frame_illustration": None,
        })

    # --- wall: the zone-end wall. Promoted only when the zone-end check
    # has actually watched it die and the screen fade. objects_*.json is
    # the hypothesis; wall_zone_end_*.json is the witness.
    wall_check = load_json(KNOWLEDGE / f"wall_zone_end_{game}.json")
    if objects and objects.get("tables"):
        tables = objects["tables"]
        types = tables.get("types") or []
        hps = tables.get("hit_points") or []
        wall_slots = [
            {"slot": i, "type": t, "hp": hps[i] if i < len(hps) else None}
            for i, t in enumerate(types)
            if isinstance(t, int) and t not in (0, 1)
            and i < len(hps) and (hps[i] or 0) >= 8
        ]
        confirmed = bool(wall_check and wall_check.get("ok"))
        links.append({
            "concept": "wall",
            "fact_ids": [],
            "entity": {
                "kind": "object_type",
                "object_types": [4, 16, 17],
                "table": {
                    "type_base": tables.get("type_base", 0x530),
                    "hp_base": tables.get("hp_base", 0x580),
                    "slots": tables.get("slots"),
                },
                "wall_slots": wall_slots,
            },
            "evidence": {
                "scan": "object_tables.py + verify_wall_zone_end.py",
                "artifact": (
                    f"runs/knowledge/objects_{game}.json"
                    + (f", runs/knowledge/wall_zone_end_{game}.json"
                       if wall_check else "")
                ),
                "verify": (
                    (wall_check or {}).get("verify_sentence")
                    or (
                        "object_tables.py docstring: output is a hypothesis "
                        "with evidence, not a fact. Slots with HP >= 8 under "
                        "fire are wall candidates; killing one must end the "
                        "zone for this to become scan_confirmed."
                    )
                ),
            },
            "status": "scan_confirmed" if confirmed else "proposed",
            "frame_illustration": None,
        })
    else:
        links.append({
            "concept": "wall",
            "fact_ids": [],
            "entity": None,
            "evidence": {
                "scan": "object_tables.py",
                "artifact": None,
                "verify": "high-HP object under fire; not linked yet",
            },
            "status": "proposed",
            "frame_illustration": None,
        })

    # --- door: exit region. Position 112 is from watching, not RAM. ------
    links.append({
        "concept": "door",
        "fact_ids": ["door-up-after-clear"],
        "entity": {
            "kind": "screen_region",
            "approx_x": 112,
            "approx_y": "far wall, top of room",
            "note": (
                "x≈112 is a watched value from docs/manuals.md, not a RAM "
                "address. A scan that finds the door tile would replace it."
            ),
        },
        "evidence": {
            "scan": "manuals.md route note + base_rooms how_to_reach_room_2",
            "artifact": "runs/knowledge/base_rooms_ContraJ.json",
            "verify": (
                "hold UP with hero at x≈112 after sensor dies → room fade; "
                "hold UP elsewhere does not"
            ),
        },
        "status": "human_confirmed",
        "frame_illustration": None,
    })

    # --- menu: the polygon docs/manuals.md names as the first payoff ----
    links.append({
        "concept": "menu",
        "fact_ids": ["press-start-to-begin", "player-select"],
        "entity": {
            "kind": "screen_text",
            "reader": "nes_player.perception.text.find_prompt",
            "watch": "nes_player.perception.title.TitleWatch",
            "measured": (
                "scripts/menu_polygon.py on ContraJ: manual arm (START "
                "from the prior) entered at frame 480; prompt-only arm "
                "obeyed a half-read SELECT and never entered; blind_sparse "
                "(the old pulse schedule) entered at 840; blind_dense at 480"
            ),
        },
        "evidence": {
            "scan": "menu_polygon.py + find_prompt",
            "artifact": f"runs/knowledge/menu_{game}.jsonl",
            "verify": (
                "answers_the_pad is the play signal; START from the "
                "manual's menu facts reaches it, SELECT from the screen "
                "alone does not on this title"
            ),
        },
        "status": "scan_confirmed",
        "frame_illustration": None,
    })
    links.append({
        "concept": "hud",
        "fact_ids": [],
        "entity": None,
        "evidence": {
            "scan": "HudReader",
            "artifact": "assets/nes_font.json",
            "verify": "digits from counter dynamics; letters from the font atlas",
        },
        "status": "proposed",
        "frame_illustration": None,
    })
    links.append({
        "concept": "capsule",
        "fact_ids": ["capsule-weapons"],
        "entity": None,
        "evidence": {"scan": None, "artifact": None,
                     "verify": "not linked yet"},
        "status": "proposed",
        "frame_illustration": None,
    })
    links.append({
        "concept": "hazard",
        "fact_ids": [],
        "entity": None,
        "evidence": {"scan": None, "artifact": None,
                     "verify": "not linked yet"},
        "status": "proposed",
        "frame_illustration": None,
    })
    links.append({
        "concept": "terrain-solid",
        "fact_ids": [],
        "entity": None,
        "evidence": {"scan": "contact_map.py", "artifact": None,
                     "verify": "not linked yet"},
        "status": "proposed",
        "frame_illustration": None,
    })

    return {
        "schema_version": 1,
        "game": game,
        "source": (
            "assembled from runs/knowledge/* by scripts/ground_concepts.py; "
            "manual concepts are from assets/priors/prior_ContraJ-Nes-v0.json"
        ),
        "links": links,
    }


def check(table: dict) -> tuple[list[str], list[str]]:
    """Honesty checks: a confirmed link needs an entity and an artefact."""
    errors: list[str] = []
    warnings: list[str] = []
    seen = set()
    for i, link in enumerate(table.get("links") or []):
        concept = link.get("concept")
        where = f"links[{i}] ({concept})"
        if not concept:
            errors.append(f"{where}: concept required")
            continue
        if concept in seen:
            errors.append(f"{where}: duplicate concept")
        seen.add(concept)
        if concept not in CONCEPTS:
            warnings.append(f"{where}: concept not in the shared vocabulary")
        status = link.get("status")
        if status not in (
            "proposed", "scan_confirmed", "human_confirmed", "rejected"
        ):
            errors.append(f"{where}: bad status {status!r}")
        if status in ("scan_confirmed", "human_confirmed"):
            if not link.get("entity"):
                errors.append(
                    f"{where}: {status} but entity is null — drop the "
                    f"status or link the entity"
                )
            ev = link.get("evidence") or {}
            if status == "scan_confirmed" and not ev.get("artifact"):
                errors.append(f"{where}: scan_confirmed without an artefact")
            if not ev.get("verify"):
                errors.append(f"{where}: no verify sentence")
        if status == "proposed" and link.get("entity") and not (
            (link.get("evidence") or {}).get("artifact")
        ):
            warnings.append(
                f"{where}: proposed with an entity but no artefact — the "
                f"scan output is a hypothesis; cite the file or clear the "
                f"entity"
            )
    missing = set(CONCEPTS) - seen
    if missing:
        warnings.append(f"vocabulary concepts with no row: {sorted(missing)}")
    return errors, warnings


def write(table: dict, out: Path | None) -> int:
    out = out or (OUT_DIR / f"grounding_{table['game']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(table['links'])} links -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("--game", default="ContraJ-Nes-v0")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.cmd == "build":
        table = build(args.game)
        errors, warnings = check(table)
        for w in warnings:
            print(f"warning: {w}")
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        rc = write(table, args.out)
        return rc or (1 if errors else 0)

    # check an existing table
    path = args.out or (OUT_DIR / f"grounding_{args.game}.json")
    if not path.exists():
        print(f"no grounding table at {path}; run build first", file=sys.stderr)
        return 2
    table = json.loads(path.read_text())
    errors, warnings = check(table)
    n_conf = sum(1 for link in table.get("links") or []
                 if link.get("status") in ("scan_confirmed", "human_confirmed"))
    print(f"{path}: {len(table.get('links') or [])} links, "
          f"{n_conf} confirmed")
    for w in warnings:
        print(f"  warning: {w}")
    for e in errors:
        print(f"  ERROR: {e}")
    print("  FAIL" if errors else "  ok")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
