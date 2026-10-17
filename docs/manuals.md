# Where the manuals come from

The planner's one remaining human input is the manual prior in
`assets/priors/`: a small file per game saying what to destroy and where the
way on is. Everything numeric underneath it — buttons, camera, object tables,
the player's own sprites, the section counter — is found by a scan. The prior
is not, and until 2026-09-16 it was written from recollection of the manual
rather than from the manual, which is a weaker claim than it looked.

These are the scans, opened and checked rather than cited from memory.

| what | where |
|---|---|
| NES / Famicom / FDS manuals, 3316 PDFs, most with a text layer | https://archive.org/details/rx2MRPnes1250pxPDF |
| Contra, US NES manual, 600 dpi with OCR | https://archive.org/details/contra-usa_202412 |
| The same Rx2.0 project as a Calibre library | https://archive.org/details/Rx2MRPnesCalibre |
| NES manuals, curated, full colour | https://www.nesfiles.com/ |
| NES manuals, "culled from various sources" | https://www.digitpress.com/library/manuals/nes/index.html |
| Manuals across platforms, per-game pages | https://www.gamesdatabase.org/ |
| Scanned documentation, preservation project | https://www.gamingalexandria.com/ |
| NES downloads including documentation | https://www.oldgames.sk/en/downloads?platform=nes |
| SNES manuals | https://sites.google.com/view/snesmanuals |
| Super Famicom manuals | https://sites.google.com/view/superfamicommanuals |
| Arcade service and operator documentation | https://arcadertfm.com/ |
| replacementdocs, manuals by request | https://replacementdocs.com/ |
| nesworld, manuals section | https://www.nesworld.com/manuals.php |

Every row answered when this table was written; `nesworld` sits behind an
interstitial and needs a browser rather than a fetch. The first two rows are
the ones actually used so far — the rest are there because a single collection
is a single point of failure, and because the Super Famicom and arcade rows
cover platforms this repository does not play yet.

The collection holds the games this repository plays, including
`Contra (Re-Translation).pdf` — the **Japanese** Famicom manual, which is the
one that matches the `ContraJ-Nes-v0` ROM. Fetch a single file by name:

```sh
curl -sL -o contra.pdf \
  'https://archive.org/download/rx2MRPnes1250pxPDF/Contra%20%28Re-Translation%29.pdf'
```

**The scans do not go in this repository.** They are Konami's, not ours. What
goes in is the derived fact with a page citation, the same rule the project
already applies to ROMs. A prior entry should be traceable to a page a person
can open, which is the whole point of sourcing it this way: it takes the
author out of the loop rather than adding knowledge to it.

## What the Japanese manual settled straight away

Three things, on the control pages, that a week of experiments had been
circling:

* **Firing left is in the game.** The table on pages 9–10 lists all eight
  directions, including 左に発砲 and 左斜め上45°に発砲. The mirrored templates
  added on 15 September as "a symmetry the candidate set was missing" were
  printed in 1988.
* **On 3D screens the gun points forward only** (page 11): 上 advances, 左 and
  右 strafe, 下 goes prone, and 3D画面では前方にのみ攻撃発砲します. So the base's
  sensor is not reached by aiming left — it is reached by *stepping sideways*
  until the shot lines up. The result of the mirrored templates stands; the
  explanation given for it on 15 September does not.
* **The manual scopes its own facts by scroll mode** (page 12):
  AREAによって画面のスクロールが縦、横、3Dと切り替わります — vertical, horizontal or
  3D by area, with underwater a fourth mode of its own (no jumping, no firing
  downward). The `scope` field in the prior should follow that axis, which is
  the source's, rather than the base-versus-jungle split written by hand.

Also worth having, from the US manual's text: destroying the zone's detection
sensor is the advance condition **in every zone**, not a base speciality; Base
1 is a maze of sensors around an "evil core" at its centre, which is the
type-16 object found in its sixth room; and a cleared stage grants an extra
life, which the death price does not currently know.

## The point of all this: manual-assisted zero-shot play

The target this sets up, stated before it is reached: **the model has never
seen the game, and before it starts it is handed the documentation that
shipped in the box.** Not a tuned prior written by whoever is running the
experiment, not addresses read out of memory — the manual, as a person would
get it.

That is a fair description of how a human meets a new cartridge, and it is a
harder and more honest test than anything here so far. Every result in this
repository was obtained on a game the author had already watched for weeks.

Where it actually stands today, so the goal is not mistaken for a result:

* the mechanism exists — `assets/priors/` is read by the planner and measurably
  worth something: the base's first room opens 17 times in 32 with it;
* as of 16 September its `source` names a real scan rather than recollection,
  and the stage scope came from the manual's own control pages;
* what has **not** happened is the test itself: a game nobody here has tuned
  for, its manual, and a number. Until that runs, this is a direction.

Two practical notes. Recordings for the site are made from the **English**
release, so the screen is readable to more people than the Japanese one is.
And a manual is text and pictures in two languages. The agent now reads both
— the text layer through OCR, the control tables by vision — into a structured
prior with page citations. Menus were the first place it paid off, and the
crate combat in Chip 'n Dale was the second: a sequence of buttons a scan
cannot discover.

## Reading them with a machine

The pipeline that does that reading lives beside this file, and the protocol
it feeds is [zero_shot_protocol.md](zero_shot_protocol.md):

| piece | what |
|---|---|
| `scripts/fetch_manual.py` | pulls a named scan into `manuals/` (gitignored) and dumps one text file per page with `pdftotext -layout` |
| `scripts/parse_manual.py` | page text → schema-v2 facts with page citations; `score` compares against gold with the two protocol baselines |
| `assets/priors/schema/prior.schema.json` | the contract: concept layer (`facts`, every row cited) separate from scan realisations (`targets.realisation`, every number `found_by`) |
| `assets/priors/gold/contra_j.json` | the gold set for Contra's control pages — read off the scan by a person, not invented |
| `scripts/validate_prior.py` | structural check and honesty warnings: fact without a page, realisation without a concept, `target_px` without its arithmetic |
| `scripts/ground_concepts.py` | the grounding table: manual concept → on-screen entity, each link naming the scan artefact that witnessed it |
| `scripts/menu_polygon.py` | the menu measurement: blind / prompt / manual arms, `answers_the_pad` as the play signal |
| `scripts/hero_tiles_episode.py` | hero tiles from a recorded episode when a live scan cannot reach the fight (Battletoads' intro) |
| `scripts/episode_state.py` | savestate at a frame of an episode, so `object_tables.py` has something to shoot at |
| `scripts/verify_wall_zone_end.py` | wall_hp → 0 and the screen fades: the witness that promoting the wall needs |
| `assets/priors/grounding/grounding_*.json` | Contra 5/9 confirmed, Vice and Battletoads 2/9 (menu + player) |

The shipped `prior_ContraJ-Nes-v0.json` is schema v2 with the legacy flat
fields kept, so `oracle_mpc.py` runs unchanged. Of its eight facts, six name a
page a person can open; the door position and the capsule note are still route
notes and are marked `extracted` rather than `human_confirmed` until their
pages are pinned.

Numbers from the parse go in `docs/claims.jsonl` only when measured on real
scans. The fixture self-check (`parse_manual.py score --fixture`) is a
pipeline test on synthetic pages and is not a measurement.

### HUD reading (Battletoads, 2026-10-10)

The owner's screenshot settles the layout: score, then a 6-segment yellow
bar, then hearts, top-left. Two hits leave 4 filled and 2 empty slots still
drawn. `scripts/hud_hearts.py` reads `bar_filled` / `bar_slots` / `hearts`
from that photo (4/6/5, gold in `runs/knowledge/hud_gold_battletoads.json`).

Live polling is **not** yet measured: the explore episodes never reach the
jungle screen with that HUD (intro/story wall), and a single yellow threshold
does not span the JPEG screenshot and the NES palette. Auto-palette by frame
brightness is in place; the missing artefact is a savestate on that screen.
Do not quote live HUD numbers until that state exists.

### Scope by level type (Battletoads, 2026-10-15)

Battletoads changes the *gameplay* per level — about ten kinds across the
game (owner). That is the same axis Contra's manual uses to scope its facts
(`scroll_mode`: vertical / horizontal / 3D / underwater). A prior entry or a
trained policy that works on one level type can be inert on the next; the
`scope` field of the prior schema is where that goes. Level 1 so far:
platformer + falling intro, white hearts, blue bar, hearts and bar drop
independently.
