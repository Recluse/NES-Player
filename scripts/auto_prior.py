#!/usr/bin/env python3
"""Generate gold set and prior from LLM lexicon output.

Reduces per-game pipeline time by automating the manual steps:
  lexicon JSON → gold set JSON + prior JSON

    uv run python scripts/auto_prior.py --lexicon manuals/parsed/galaga_llm.json \
        --game Galaga-Nes-v0 --out-gold assets/priors/gold/galaga_us.json \
        --out-prior assets/priors/prior_Galaga-Nes-v0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def make_gold(lexicon: dict, game: str, scan_id: str) -> dict:
    """Generate a gold set from LLM lexicon facts."""
    facts = []
    for f in lexicon.get("facts", []):
        facts.append({
            "id": f["id"],
            "kind": f["kind"],
            "page": f["citation"]["page"],
            "statement_key": f.get("statement_key", f["id"]),
            "provenance": "us_manual",
            "quote_hint": f["citation"]["quote"][:80],
        })
    return {
        "game": game,
        "scan_id": scan_id,
        "language": "en",
        "note": f"Auto-generated from LLM lexicon. {len(facts)} facts.",
        "facts": facts,
    }


def make_prior(lexicon: dict, game: str, scan_id: str) -> dict:
    """Generate a prior from LLM lexicon facts."""
    facts = []
    templates = []
    for f in lexicon.get("facts", []):
        facts.append({
            "id": f["id"],
            "kind": f["kind"],
            "statement": f["statement"],
            "citation": f["citation"],
            "status": "extracted",
        })
        # Generate templates from controls/abilities
        if f["kind"] in ("controls", "ability") and "statement_key" in f:
            sk = f["statement_key"]
            if "jump" in sk and "a" in sk:
                templates.append({
                    "name": "jump",
                    "phases": [{"tap": ["A"], "rest": [], "frames": 12}],
                    "citation": f["citation"],
                    "note": f["statement"][:60],
                })
            elif "punch" in sk or "attack" in sk or "fire" in sk:
                templates.append({
                    "name": sk,
                    "phases": [{"tap": ["B"], "rest": [], "frames": 10}],
                    "citation": f["citation"],
                    "note": f["statement"][:60],
                })
            elif "move" in sk and ("right" in sk or "dpad" in sk):
                templates.append({
                    "name": "walk-right",
                    "phases": [{"hold": ["RIGHT"], "rest": 0, "frames": 20}],
                    "citation": f["citation"],
                    "note": f["statement"][:60],
                })

    # Deduplicate templates by name
    seen = set()
    unique_templates = []
    for t in templates:
        if t["name"] not in seen:
            seen.add(t["name"])
            unique_templates.append(t)

    return {
        "schema_version": 2,
        "game": game,
        "source": f"Auto-generated from LLM lexicon ({scan_id}).",
        "manual": {
            "scan_id": scan_id,
            "language": "en",
            "pages_used": sorted(set(f["citation"]["page"] for f in lexicon.get("facts", []))),
            "parse": {
                "method": "llm-lexicon",
                "pipeline_version": "llm_lexicon.py/1",
                "judge": "MiMo as model behind the prompt",
            },
        },
        "notes": [f["statement"][:80] for f in lexicon.get("facts", [])[:5]],
        "facts": facts,
        "templates": unique_templates,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lexicon", required=True, help="LLM lexicon JSON")
    p.add_argument("--game", required=True)
    p.add_argument("--scan-id", default="")
    p.add_argument("--out-gold", required=True)
    p.add_argument("--out-prior", required=True)
    args = p.parse_args()

    lexicon = json.loads(Path(args.lexicon).read_text())
    scan_id = args.scan_id or args.lexicon

    gold = make_gold(lexicon, args.game, scan_id)
    prior = make_prior(lexicon, args.game, scan_id)

    Path(args.out_gold).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_gold).write_text(json.dumps(gold, indent=2, ensure_ascii=False))
    print(f"Gold: {len(gold['facts'])} facts -> {args.out_gold}")

    Path(args.out_prior).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_prior).write_text(json.dumps(prior, indent=2, ensure_ascii=False))
    print(f"Prior: {len(prior['facts'])} facts, {len(prior['templates'])} templates -> {args.out_prior}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
