#!/usr/bin/env python3
"""Build planner templates from the skill library.

A new game's manual names abilities ("jump", "throw", "fire"). The skill
library supplies the button composition that realises an ability, measured
on other games. What does not transfer is the button mapping — that comes
from the game's own control scan or its manual's control table.

    uv run python scripts/skill_library.py list
    uv run python scripts/skill_library.py build --game ChipnDale-Nes-v0
    uv run python scripts/skill_library.py build --game Galaga-Nes-v0 \
        --abilities fire,strafe

Writes a prior-ready `templates` list. Button names are the NES pad;
rebind them per game if a scan says otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "assets" / "skills" / "skill_library.json"


def load() -> dict:
    return json.loads(LIBRARY.read_text())


def build_templates(abilities: list[str], game: str = "") -> list[dict]:
    """Expand abilities and their compositions into planner templates."""
    lib = load()
    out: list[dict] = []
    for name in abilities:
        skill = lib["abilities"].get(name)
        if not skill:
            continue
        t = {"name": name, **skill["buttons"]}
        if skill.get("citation"):
            t["citation"] = {"page": 0, "quote": skill["citation"]}
        t["found_by"] = f"skill_library.json ability:{name}"
        out.append(t)
    for cname, comp in lib.get("compositions", {}).items():
        steps = comp.get("steps", [])
        if not set(steps) & set(abilities):
            continue
        phases = []
        for step in steps:
            skill = lib["abilities"].get(step)
            if not skill:
                continue
            phases.append(skill["buttons"])
        if not phases:
            continue
        t = {"name": cname, "phases": phases}
        if comp.get("citation"):
            t["citation"] = {"page": 0, "quote": comp["citation"]}
        t["found_by"] = f"skill_library.json composition:{cname}"
        out.append(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    bl = sub.add_parser("build")
    bl.add_argument("--game", default="")
    bl.add_argument("--abilities", default="",
                    help="comma list; default = every ability the library knows")
    args = ap.parse_args()

    lib = load()
    if args.cmd == "list":
        print("abilities:")
        for name, sk in lib["abilities"].items():
            games = ", ".join(sk.get("games", []))
            print(f"  {name:16} {sk['description'][:50]:52} [{games}]")
        print("\ncompositions:")
        for name, comp in lib["compositions"].items():
            if name == "note":
                continue
            steps = " → ".join(comp.get("steps", []))
            print(f"  {name:22} {steps}")
        return 0

    if args.abilities:
        names = [a.strip() for a in args.abilities.split(",") if a.strip()]
    else:
        names = list(lib["abilities"].keys())
    templates = build_templates(names, args.game)
    print(json.dumps({"game": args.game, "templates": templates}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
