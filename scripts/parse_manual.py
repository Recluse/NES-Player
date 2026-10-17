#!/usr/bin/env python3
"""Parse manual pages into prior facts, and score the parse against gold.

Phase 1B of docs-ru/plan-2026-10-08.md. Two stages, kept separate on purpose:

1. extract   — page text -> structured facts (schema v2 concept layer).
               Rule-based over the page text layer today; the same JSON
               shape is what an offline VLM/LLM step must emit later.
2. score     — extracted facts vs a gold set, with the two baselines the
               protocol names: empty schema, and raw page text treated as
               one giant "fact" per page (the "shove OCR at the model"
               baseline, scored as a single hit per matching gold row).

Every scored fact needs a page. A fact without a page is not a fact.

Usage:
  uv run python scripts/parse_manual.py extract --pages manuals/pages/Contra-US \
      --game ContraJ-Nes-v0 --out manuals/parsed/contra_us.json
  uv run python scripts/parse_manual.py score \
      --parsed manuals/parsed/contra_us.json \
      --gold assets/priors/gold/contra_j.json
  uv run python scripts/parse_manual.py score --fixture   # self-check on samples
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

# (kind, compiled pattern, statement template, statement_key)
# statement_key is what scoring matches on; wording of statement is free.
RULES: list[tuple[str, re.Pattern, str, str]] = [
    (
        "controls",
        re.compile(r"(fire|shoot).{0,40}(eight|8).{0,20}direction", re.I),
        "The gun fires in eight directions.",
        "eight-directions",
    ),
    (
        "controls",
        re.compile(r"(左に発砲|左斜め上45°に発砲)", re.I),
        "Firing left (and up-left 45°) is a printed control.",
        "fire-left",
    ),
    (
        "controls",
        re.compile(r"(3D画面では前方にのみ攻撃発砲|3D.{0,20}(only|forward).{0,20}fire)", re.I),
        "On 3D screens the gun points forward only.",
        "3d-forward-only",
    ),
    (
        "controls",
        re.compile(r"(スクロールが縦、横、3D|scroll.{0,30}(vertical|horizontal|3D))", re.I),
        "Scroll is vertical, horizontal or 3D by area.",
        "scroll-modes",
    ),
    (
        "goal",
        re.compile(r"(detection\s+sensor|センサー)", re.I),
        "Destroying the detection sensor is the advance condition.",
        "sensor-advance",
    ),
    (
        "economy",
        re.compile(r"(extra\s+(man|life)|エキストラ)", re.I),
        "A cleared stage grants an extra life.",
        "extra-life",
    ),
    (
        "menu",
        re.compile(r"press\s+(the\s+)?start", re.I),
        "Press Start to begin.",
        "press-start",
    ),
    (
        "controls",
        re.compile(r"(十字キー|control\s+(pad|stick)|controller)", re.I),
        "Control pad / d-pad is the movement input.",
        "dpad",
    ),
    (
        "hazard",
        re.compile(r"(water|underwater|水中)", re.I),
        "Underwater areas have their own rules (no jumping, no firing downward).",
        "underwater",
    ),
    (
        "exit",
        re.compile(r"(door|ドア|出口|way\s+on|exit)", re.I),
        "The way on is through the exit/door.",
        "door",
    ),
    # --- Vice-style manual facts (English, weapons/items/HUD) -----------
    (
        "controls",
        re.compile(r"(SELECT.{0,40}(select|weapon)|select\s+weapon)", re.I),
        "SELECT selects the weapon (laser whip, .44 Magnum, hand grenade).",
        "select-weapon",
    ),
    (
        "controls",
        re.compile(
            r"(B\s*(use|fires|press).{0,30}weapon|use the selected weapon|^\s*FIRE\b)",
            re.I),
        "B uses the selected weapon (fire).",
        "fire-weapon",
    ),
    (
        "controls",
        re.compile(r"(START.{0,20}(pause|skip)|pause\s*/?\s*skip\s+cinema)", re.I),
        "START is pause / skip cinema.",
        "start-pause",
    ),
    (
        "economy",
        re.compile(r"(100\s+coins.{0,30}(extra|man|life)|coins?.{0,20}extra\s+(man|life))", re.I),
        "100 coins grant an extra man.",
        "extra-man",
    ),
    (
        "goal",
        re.compile(r"(grenade.{0,60}(boss|keep|limited)|keep them for the)", re.I),
        "Hand grenades are limited; keep them for the stage-2 boss.",
        "grenades-boss",
    ),
    (
        "target",
        re.compile(r"(bullets?\s*\+10|grenades?\s*\+5|heal-aid|meat\s*\+10)", re.I),
        "Pickups: bullets +10, grenades +5, heal-aid +2, meat +10 lifeline.",
        "heal-item",
    ),
    (
        "hud",
        re.compile(r"(lifeline|P\s*[:=].{0,20}player|E\s*[:=].{0,20}enemy)", re.I),
        "P and E are the player and enemy lifeline bars on the HUD.",
        "lifeline",
    ),
    (
        "controls",
        re.compile(r"(SELECT\s*WEAPON|switch\s+weapon)", re.I),
        "SELECT selects the weapon (laser whip, .44 Magnum, hand grenade).",
        "select-weapon",
    ),
    (
        "economy",
        re.compile(r"(100\s*C[aoi]ins?\s*=?\s*1\s*extra\s*m[ae]n|NO\.\s*OF\s*COINS)", re.I),
        "100 coins grant an extra man.",
        "extra-man",
    ),
    (
        "target",
        re.compile(r"(NO\.\s*OF\s*GRENADES|hand\s*Grenade)", re.I),
        "Hand grenades are limited; keep them for the stage-2 boss.",
        "grenades-boss",
    ),
    (
        "target",
        re.compile(r"(Heal-?[Ff]aid|NO\.\s*OF\s*BULLETS)", re.I),
        "Pickups: bullets, grenades, heal-aid, meat.",
        "heal-item",
    ),
    # --- Contra US manual (gamesdatabase OCR, 2026-10-15) ---------------
    (
        "goal",
        re.compile(r"(detection\s+sensor|wipe-out.{0,40}sensor)", re.I),
        "To advance, wipe out the detection sensors at the end of each zone.",
        "sensor-advance",
    ),
    (
        "economy",
        re.compile(r"(every stage.{0,40}extra life|receive one extra life)", re.I),
        "Every stage successfully overcome grants one extra life.",
        "extra-life",
    ),
    (
        "controls",
        re.compile(r"(shoot\s+either\s+left|left\s+or\s+right)", re.I),
        "B shoots either left or right (and with the pad, eight directions).",
        "fire-left",
    ),
    (
        "controls",
        re.compile(r"(press\s+to\s+begin\s+your\s+mission|press\s+to\s+PAUSE)", re.I),
        "START begins the mission and pauses the action.",
        "start-pause",
    ),
    (
        "controls",
        re.compile(r"(select\s+(either\s+)?1\s*player|1\s*player\s*or\s*2)", re.I),
        "SELECT chooses one- or two-player mode; A jumps.",
        "player-select",
    ),
    (
        "controls",
        re.compile(r"(3-D|3-dimensionally|Defense Zone 2 and Defense Zone 4)", re.I),
        "Defense Zones 2 and 4 are 3D: pad to move, B to fire, A to jump; pad down lies prone.",
        "3d-forward-only",
    ),
    (
        "goal",
        re.compile(r"(eight defense zones|eight lines of defense)", re.I),
        "There are eight defense zones; the last is Red Falcon's lair.",
        "eight-zones",
    ),
    (
        "target",
        re.compile(r"(flying capsules and pill box|power-?up)", re.I),
        "Destroy flying capsules and pill-box sensors to release Falcon symbols for power-ups.",
        "capsule-weapons",
    ),
    # --- Contra JP manual (Rx2.0, jpn OCR, 2026-10-15) ------------------
    (
        "menu",
        re.compile(r"(ゲームを始める前に|1人プレイ|2人同時)"),
        "Before starting, SELECT chooses 1-player or 2-player mode.",
        "player-select",
    ),
    (
        "controls",
        re.compile(r"(STARTボタン|ゲームをスタートさせる)"),
        "START starts the game and pauses during play.",
        "start-pause",
    ),
    (
        "economy",
        re.compile(r"(CONTINUEは3回まで|CONTINUE.*3回)"),
        "CONTINUE is allowed up to 3 times per mission.",
        "continue-three",
    ),
    (
        "menu",
        re.compile(r"(スタートデモをとばしたい時|もう一度ボタンを押して)"),
        "Press START again to skip the start demo.",
        "skip-demo",
    ),
    (
        "controls",
        re.compile(r"(スク?フ?ロールが縦、横|横スク[ロフ]ール、縦スク[ロフ]ール)"),
        "Scroll is horizontal or vertical or 3D depending on the area.",
        "scroll-modes",
    ),
    (
        "controls",
        re.compile(r"(左に発砲|左斜め上|発砲|はっぽう|8方向|八方向)"),
        "Firing in eight directions including left is a printed control.",
        "fire-left",
    ),
    (
        "controls",
        re.compile(r"(8方向|八方向|8 directions)"),
        "The gun fires in eight directions.",
        "eight-directions",
    ),
    (
        "controls",
        re.compile(r"(水の中では|水中では)"),
        "Underwater: d-pad to swim, no jumping with A.",
        "underwater",
    ),
    (
        "controls",
        re.compile(r"(3D画面|前方にのみ|3Dと切り替わり|3Dと切り替)"),
        "On 3D screens the gun points forward only.",
        "3d-forward-only",
    ),
]

# Kinds we accept as concept-layer rows (schema v2).
KINDS = {
    "goal", "target", "controls", "exit", "hazard", "player",
    "economy", "menu", "other",
}


def extract_from_pages(pages_dir: Path, game: str, scan_id: str = "") -> dict:
    """Page .txt files -> a schema v2 prior fragment (facts only)."""
    facts: list[dict] = []
    seen: set[tuple[str, int]] = set()
    page_files = sorted(pages_dir.glob("page-*.txt"))
    if not page_files:
        raise SystemExit(f"no page-*.txt in {pages_dir}")

    for path in page_files:
        m = re.search(r"(\d+)", path.stem)
        page_no = int(m.group(1)) if m else 1
        text = path.read_text(errors="replace")
        if not text.strip():
            continue
        for kind, pattern, statement, key in RULES:
            match = pattern.search(text)
            if not match:
                continue
            dedup = (key, page_no)
            if dedup in seen:
                continue
            seen.add(dedup)
            # Snip a short quote around the match for the citation.
            start = max(0, match.start() - 40)
            end = min(len(text), match.end() + 40)
            quote = " ".join(text[start:end].split())
            facts.append({
                "id": key,
                "kind": kind,
                "statement": statement,
                "citation": {"page": page_no, "quote": quote},
                "status": "extracted",
                "statement_key": key,
            })

    return {
        "schema_version": 2,
        "game": game,
        "source": scan_id or f"pages at {pages_dir}",
        "manual": {
            "scan_id": scan_id or pages_dir.name,
            "language": "ja" if any("センサー" in p.read_text(errors="replace")
                                    or "発砲" in p.read_text(errors="replace")
                                    for p in page_files[:3]) else "en",
            "pages_used": sorted({f["citation"]["page"] for f in facts}),
            "parse": {
                "method": "rule-based-page-text",
                "pipeline_version": "parse_manual.py/1",
            },
        },
        "facts": facts,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def load_gold(path: Path) -> dict:
    return json.loads(path.read_text())


def match_key(extracted: dict) -> str | None:
    return extracted.get("statement_key") or extracted.get("id")


def score(parsed: dict, gold: dict, include_keys: set[str] | None = None,
          page_text: str | None = None) -> dict:
    """Precision / recall on statement_key, restricted to gold rows.

    include_keys: if set, only gold rows whose provenance is in this set are
    scored (e.g. {'ja_manual'} when parsing the Japanese scan). A fact that
    matches a gold row outside that set is `unscored_match`, not a false
    positive — the Japanese parse is not punished for the US manual's rows.
    """
    scored_rows = []
    unscored_keys = set()
    for row in gold["facts"]:
        key = row.get("statement_key")
        if include_keys is not None and row.get("provenance") not in include_keys:
            if key:
                unscored_keys.add(key)
            continue
        scored_rows.append(row)

    gold_by_key = {r["statement_key"]: r for r in scored_rows}
    extracted = parsed.get("facts") or []
    hits, misses, extra, unscored, wrong_page = [], [], [], [], []
    extracted_keys = set()
    for f in extracted:
        key = match_key(f)
        if key is None:
            extra.append(f)
            continue
        if key in unscored_keys:
            unscored.append(key)
            continue
        extracted_keys.add(key)
        if key in gold_by_key:
            gpage = gold_by_key[key].get("page")
            fpage = (f.get("citation") or {}).get("page")
            # gold without a pinned page is not scored on the page number
            if gpage and fpage and int(fpage) != int(gpage):
                wrong_page.append({"id": key, "got": fpage, "gold": gpage})
            else:
                hits.append(key)
        else:
            extra.append(f)

    for key in gold_by_key:
        if key not in extracted_keys:
            misses.append(key)

    empty = {"precision": None, "recall": 0.0, "hits": 0,
             "note": "empty schema extracts nothing"}

    # Raw-text baseline: gold rows whose quote_hint appears anywhere in the
    # page text the parser saw. Unscored without page text — never guessed.
    raw = {"note": "unscored: pass --page-text to evaluate"}
    if page_text is not None:
        raw_hits = []
        for key, row in gold_by_key.items():
            hint = row.get("quote_hint") or ""
            if hint and hint in page_text:
                raw_hits.append(key)
        raw = {
            "hits": raw_hits,
            "n_gold": len(gold_by_key),
            "recall": (len(raw_hits) / len(gold_by_key)) if gold_by_key else None,
            "note": "one bag hit per gold row whose quote_hint appears in the text",
        }

    n_gold = len(gold_by_key)
    n_hits = len(set(hits))
    n_pred = n_hits + len(wrong_page) + len(extra)
    precision = n_hits / n_pred if n_pred else None
    recall = n_hits / n_gold if n_gold else None

    return {
        "n_gold": n_gold,
        "n_extracted": len(extracted),
        "hits": sorted(set(hits)),
        "misses": misses,
        "wrong_page": wrong_page,
        "unscored_match": unscored,
        "extra": [e.get("id") or e.get("statement", "?") for e in extra],
        "precision": precision,
        "recall": recall,
        "baseline_empty": empty,
        "baseline_raw_ocr": raw,
    }


# ---------------------------------------------------------------------------
# Fixtures (self-check without a PDF on disk)
# ---------------------------------------------------------------------------

FIXTURE_PAGES = {
    9: (
        "CONTENTS\n\tController\n\tThe gun fires in eight directions.\n"
        "\tController uses the 十字キー.\n"
    ),
    10: "CONTROLS\n\t左に発砲 左斜め上45°に発砲\n",
    11: (
        "3D画面では前方にのみ攻撃発砲します\n"
        "上 advances, 左 and 右 strafe, 下 goes prone\n"
    ),
    12: (
        "AREAによって画面のスクロールが縦、横、3Dと切り替わります\n"
        "Destroy the detection sensor to advance.\n"
        "Clear a stage for an extra man.\n"
    ),
}

VICE_FIXTURE_PAGES = {
    2: (
        "CONTROLS\n"
        "SELECT selects the weapon: laser whip -> .44 Magnum -> hand grenade.\n"
        "B uses the selected weapon.\n"
        "START is pause / skip cinema.\n"
    ),
    3: (
        "WEAPONS AND ITEMS\n"
        "Laser whip: close range.\n"
        ".44 Magnum: ranged, limited bullets.\n"
        "Hand grenade: long range, limited; keep them for the stage-2 boss.\n"
        "Bullets +10, grenades +5, heal-aid +2 lifeline, meat +10 lifeline.\n"
    ),
    4: (
        "HUD\n"
        "P is the player's lifeline.\n"
        "E is the enemy lifeline.\n"
        "100 coins grant an extra man.\n"
    ),
}


def run_fixture(which: str = "contra") -> int:
    """Parse synthetic pages and score against gold. Must not report a
    respectable number — these are pattern hits on samples, a pipeline
    self-check only."""
    import tempfile

    pages = FIXTURE_PAGES if which == "contra" else VICE_FIXTURE_PAGES
    gold_path = ("assets/priors/gold/contra_j.json" if which == "contra"
                 else "assets/priors/gold/vice_j.json")
    include = {"ja_manual"} if which == "contra" else {"us_manual"}

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for no, text in pages.items():
            (tdp / f"page-{no:03d}.txt").write_text(text)
        parsed = extract_from_pages(tdp, game=f"fixture-{which}",
                                    scan_id="fixture-not-a-scan")
        gold = load_gold(ROOT / gold_path)
        page_text = "\n".join(pages.values())
        result = score(parsed, gold, include_keys=include, page_text=page_text)
        print(json.dumps({
            "fixture": which,
            "warning": "self-check on synthetic pages, not a measurement",
            "n_facts": len(parsed["facts"]),
            "score": result,
        }, ensure_ascii=False, indent=2))
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="pages -> prior fragment")
    ex.add_argument("--pages", type=Path, required=True)
    ex.add_argument("--game", required=True)
    ex.add_argument("--scan-id", default="")
    ex.add_argument("--out", type=Path)

    sc = sub.add_parser("score", help="parsed prior vs gold")
    sc.add_argument("--parsed", type=Path)
    sc.add_argument("--gold", type=Path,
                    default=ROOT / "assets/priors/gold/contra_j.json")
    sc.add_argument("--provenance", default="",
                    help="comma list; score only these gold rows "
                         "(e.g. ja_manual)")
    sc.add_argument("--page-text", type=Path,
                    help="concatenated page text for the raw-OCR baseline")
    sc.add_argument("--fixture", action="store_true",
                    help="self-check on bundled sample pages")
    sc.add_argument("--fixture-game", choices=["contra", "vice"],
                    default="contra")

    args = ap.parse_args()

    if args.cmd == "extract":
        parsed = extract_from_pages(args.pages, args.game, args.scan_id)
        text = json.dumps(parsed, ensure_ascii=False, indent=2) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text)
            print(f"{len(parsed['facts'])} facts -> {args.out}")
        else:
            sys.stdout.write(text)
        return 0

    if args.fixture:
        return run_fixture(args.fixture_game)

    if not args.parsed:
        ap.error("score needs --parsed or --fixture")
    parsed = json.loads(args.parsed.read_text())
    gold = load_gold(args.gold)
    include = set(args.provenance.split(",")) if args.provenance else None
    page_text = args.page_text.read_text(errors="replace") if args.page_text else None
    result = score(parsed, gold, include_keys=include, page_text=page_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nprecision={result['precision']} recall={result['recall']}",
          file=sys.stderr)
    print("baseline empty: recall=0.0", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
