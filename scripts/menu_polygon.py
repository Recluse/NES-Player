#!/usr/bin/env python3
"""Menu polygon: does reading the screen (and the manual) beat pressing blind?

docs/manuals.md names the menus as the first place teaching the agent to read
the manual pays off. This is the measurement, on a real boot from power-on.

Evaluator-side only. `answers_the_pad` (oracle_mpc) saves the console state
and asks whether the picture answers a held direction — a title screen does
not, a level does. That is the `in_game` signal. START pressed once the
picture answers the pad is a pause mistake and is counted.

| arm | what it presses | knowledge |
|---|---|---|
| blind_sparse | START on the old cli/play.py pulse schedule | none |
| blind_dense | START every 20 frames | none |
| prompt | whatever `find_prompt` reads off the screen | the picture |
| manual | prompt, plus the prior's menu facts (press START to begin) | the boxed documentation |

    uv run python scripts/menu_polygon.py --game ContraJ-Nes-v0
    uv run python scripts/menu_polygon.py --game ContraJ-Nes-v0 --arm prompt

Appends to runs/knowledge/menu_<game>.jsonl and prints a table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from oracle_mpc import answers_the_pad  # noqa: E402

from nes_player.emulator.stable_retro import StableRetroAdapter  # noqa: E402
from nes_player.perception.text import find_prompt  # noqa: E402

# The old cli/play.py schedule — what the project used to do blindly.
BLIND_SPARSE = frozenset({60, 61, 240, 241, 420, 421, 600, 601, 780, 781})
CHECK_EVERY = 60
PROMPT_EVERY = 12
ARMS = ("blind_sparse", "blind_dense", "prompt", "manual")


def prior_menu_buttons(prior_path: Path | None) -> set[str]:
    """Buttons the manual's menu facts name."""
    if not prior_path or not prior_path.exists():
        return set()
    prior = json.loads(prior_path.read_text())
    out: set[str] = set()
    for fact in prior.get("facts") or []:
        if fact.get("kind") != "menu":
            continue
        for word in ("START", "SELECT", "A", "B"):
            if word in (fact.get("statement") or "").upper():
                out.add(word)
    for note in prior.get("notes") or []:
        if "PRESS" in note.upper() or "START" in note.upper():
            out.add("START")
    # The schema's controls block may name the start button too.
    for cit in ((prior.get("controls") or {}).get("notes") or []):
        if "START" in cit.upper() or "PRESS" in cit.upper():
            out.add("START")
    return out


def choose(arm: str, i: int, in_game: bool, prompt: str | None,
           menu_buttons: set[str]) -> frozenset[str]:
    if in_game:
        return frozenset()          # START during play is pause
    if arm == "blind_sparse":
        return frozenset({"START"}) if i in BLIND_SPARSE else frozenset()
    if arm == "blind_dense":
        return frozenset({"START"}) if i % 20 < 2 else frozenset()
    if arm == "prompt":
        # Only a live reading, not a stale one: a prompt seen minutes ago
        # is not a prompt now.
        if prompt and i % 24 < 2:
            return frozenset({prompt})
        return frozenset()
    if arm == "manual":
        # The screen wins on the button when it is talking *now*; the
        # manual fills the gaps. Both at the dense cadence — a half-read
        # title must not slow the press rate down to nothing.
        if prompt and i % 20 < 2:
            return frozenset({prompt})
        if menu_buttons and i % 20 < 2:
            for b in ("START", "SELECT", "A", "B"):
                if b in menu_buttons:
                    return frozenset({b})
        return frozenset()
    raise ValueError(arm)


def run_arm(game: str, integrations: str | None, arm: str, frames: int,
            prior_path: Path | None) -> dict:
    em = StableRetroAdapter(game, integration_dir=integrations)
    menu_buttons = prior_menu_buttons(prior_path)
    obs = em.reset()
    t0 = time.time()
    entered_at = None
    pause_mistakes = 0
    presses = 0
    prompt_seen = None

    for i in range(frames):
        frame = obs.frame_rgb
        prompt = None
        if i % PROMPT_EVERY == 0:
            prompt = find_prompt(frame)
            if prompt:
                prompt_seen = prompt

        in_game = False
        if i % CHECK_EVERY == 0 and i > 0:
            in_game = answers_the_pad(em, frames=16)
            if in_game and entered_at is None:
                entered_at = i

        # Once play has begun, every later frame is play: the pad test is
        # periodic and expensive, and menus do not come back mid-stage.
        if entered_at is not None and i > entered_at:
            in_game = True

        pressed = choose(arm, i, in_game, prompt, menu_buttons)
        if pressed:
            presses += 1
            if "START" in pressed and in_game:
                pause_mistakes += 1
        obs = em.step_buttons([pressed])

    em.close()
    return {
        "game": game,
        "arm": arm,
        "frames": frames,
        "entered_at": entered_at,
        "presses": presses,
        "pause_mistakes": pause_mistakes,
        "menu_buttons": sorted(menu_buttons),
        "prompt_seen": prompt_seen,
        "wall_s": round(time.time() - t0, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="ContraJ-Nes-v0")
    ap.add_argument("--integrations", default="integrations")
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--arm", choices=list(ARMS) + ["all"], default="all")
    ap.add_argument("--prior", type=Path,
                    default=ROOT / "assets/priors/prior_ContraJ-Nes-v0.json")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    arms = list(ARMS) if args.arm == "all" else [args.arm]
    rows = [
        run_arm(args.game, args.integrations, arm, args.frames, args.prior)
        for arm in arms
    ]

    out = args.out or (ROOT / "runs/knowledge" / f"menu_{args.game}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    keys = ["arm", "entered_at", "presses", "pause_mistakes", "prompt_seen",
            "wall_s"]
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in keys}
    print(" ".join(k.ljust(widths[k]) for k in keys))
    for r in rows:
        print(" ".join(str(r[k]).ljust(widths[k]) for k in keys))
    print(f"\nappended -> {out}")
    print("entered_at = first frame where answers_the_pad said play; null = never.")
    print("pause_mistakes = START after that frame.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
