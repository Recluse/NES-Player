#!/usr/bin/env python3
"""LLM lexicon: manual page text → schema-v2 facts.

Regex lexicon failed on held-out (recall 0.1 / 0.0 against raw OCR 1.0).
The wording differs per publisher; the *concepts* do not. This is the
lexicon layer of the three-layer transfer model:

    1. ontology   — game-agnostic kinds (schema v2, frozen)
    2. lexicon    — this script: page text → facts in those kinds  ← HERE
    3. grounding  — facts → buttons / object types / RAM (scans)

The extractor is a prompt. Any model that can emit JSON matching
assets/priors/schema/prior.schema.json can stand behind it: a local LLM, an
API, or a human. The script validates the JSON and scores it against gold.

    uv run python scripts/llm_lexicon.py --pages manuals/pages/ChipnDale \
        --game ChipnDale-Nes-v0 --out manuals/parsed/chipndale_llm.json
    uv run python scripts/llm_lexicon.py score \
        --parsed manuals/parsed/chipndale_llm.json \
        --gold assets/priors/gold/chipndale_us.json \
        --page-text manuals/parsed/chipndale_text.txt

`extract` prints the prompt and writes an empty skeleton; fill it from a
model call, or use `--from-file` to ingest a model's JSON directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from parse_manual import load_gold, score  # noqa: E402

PROMPT = """You are extracting a game manual's facts into a fixed schema.

Read the page text below and emit a JSON object:
{{
  "facts": [
    {{
      "id": "kebab-case-id",
      "kind": "goal|target|controls|exit|hazard|player|economy|hud|menu|collectible|ability|other",
      "statement": "one sentence, the manual's claim",
      "citation": {{"page": <pdf page number>, "quote": "<exact words from the page>"}},
      "statement_key": "short-semantic-key shared across games"
    }}
  ]
}}

Kinds (game-agnostic ontology):
- controls: what buttons do (fire, jump, move, pause)
- ability:  a verb the player can do (jump, duck, throw, pick-up, hide)
- goal:     what the game wants done (destroy X, reach Y, survive)
- target:   a thing worth destroying or collecting for progress
- exit:     the way on
- hazard:   what kills or ends the run
- player:   who the player is
- economy:  lives, coins, stars, 1-ups
- hud:      what the screen shows (score, hearts, bar)
- menu:     title / select screens
- collectible: an item to pick up (flower, acorn, crate)

Rules:
* Every fact must have a citation.quote copied from the text.
* statement_key is the concept, not the wording: "Press A to jump" and
  "A Button: Jump" both map to statement_key "a-jump".
* Do not invent numbers or objects the text does not mention.
* Emit only JSON, no commentary.

PAGES:
{pages}
"""


def build_prompt(pages_dir: Path) -> str:
    chunks = []
    for p in sorted(pages_dir.glob("page-*.txt")):
        num = int(p.stem.split("-")[-1])
        chunks.append(f"--- PDF page {num} ---\n{p.read_text(errors='replace')}")
    return PROMPT.format(pages="\n\n".join(chunks))


def empty_skeleton(game: str, scan_id: str = "") -> dict:
    return {
        "schema_version": 2,
        "game": game,
        "source": f"llm-lexicon prompt; {scan_id}",
        "manual": {
            "scan_id": scan_id,
            "language": "en",
            "parse": {"method": "llm-lexicon", "pipeline_version": "llm_lexicon.py/1"},
        },
        "facts": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract")
    ex.add_argument("--pages", type=Path, required=True)
    ex.add_argument("--game", required=True)
    ex.add_argument("--scan-id", default="")
    ex.add_argument("--out", type=Path)
    ex.add_argument("--from-file", type=Path,
                    help="JSON the model already produced; wrap it in schema v2")
    ex.add_argument("--print-prompt", action="store_true")

    sc = sub.add_parser("score")
    sc.add_argument("--parsed", type=Path, required=True)
    sc.add_argument("--gold", type=Path, required=True)
    sc.add_argument("--page-text", type=Path)

    args = ap.parse_args()

    if args.cmd == "extract":
        prompt = build_prompt(args.pages)
        if args.print_prompt:
            sys.stdout.write(prompt)
            return 0
        if args.from_file:
            model_out = json.loads(args.from_file.read_text())
            out = empty_skeleton(args.game, args.scan_id)
            out["facts"] = model_out.get("facts", [])
            out["manual"]["parse"]["judge"] = args.from_file.name
        else:
            out = empty_skeleton(args.game, args.scan_id)
            (args.out or Path("/dev/stdout")).write_text(
                json.dumps(out, indent=2) + "\n")
            print("wrote skeleton; fill facts from a model call of the prompt",
                  file=sys.stderr)
            print(f"prompt length {len(prompt)} chars; use --print-prompt",
                  file=sys.stderr)
            return 0
        text = json.dumps(out, ensure_ascii=False, indent=2) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text)
            print(f"{len(out['facts'])} facts -> {args.out}")
        else:
            sys.stdout.write(text)
        return 0

    parsed = json.loads(args.parsed.read_text())
    gold = load_gold(args.gold)
    page_text = (args.page_text.read_text(errors="replace")
                 if args.page_text else None)
    result = score(parsed, gold, page_text=page_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nprecision={result['precision']} recall={result['recall']}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
