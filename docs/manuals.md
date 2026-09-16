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
And a manual is text and pictures in two languages, which the agent cannot yet
read — teaching it to is the next piece of work, and the menus are the first
place it pays off.
