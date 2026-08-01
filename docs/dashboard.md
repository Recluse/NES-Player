# The dashboard

A 1280×720 panel, or 1920×1080 with `--hd`. The same in `play` and `explore`,
except that in `explore` the neural widgets only appear with `--observer`.

```
┌────────────────────────┬──────────────────────────────────────┐
│ Game, 4:3              │ Title                                │
│  + Grad-CAM overlay    │ info           │ action probabilities │
│  + object boxes and    │ frame/episode/ │ (bars)               │
│    velocity vectors    │ score/hud read │                      │
│                        ├────────────────┴─────────────────────┤
├──────────────┬─────────┤ uncertainty (full width)             │
│ NES gamepad  │ legend  ├────────────────┬─────────────────────┤
│ (pressed     │ CAM     │ objects        │ conv features (live)│
│  buttons     │ toggles │ thoughts       │ sprite memory       │
│  light up)   │ CAM/    │                │ audio events        │
│              │ BOXES   │      MUTE  RESTART  STOP             │
└──────────────┴─────────┴──────────────────────────────────────┘
```

## What each element shows

- **Grad-CAM** — where the network is looking, normalised by percentile so the
  top ~30% of attention shows. The weak→strong legend sits under the gamepad.
- **Object boxes** — amber is the controlled object (`ctrl_prob` ≥ 0.7), red is
  dangerous, green is rewarding, blue is unknown; dimmed means a ghost, an
  object that has stopped moving and is being held at its last position. Green
  arrows are velocity vectors.
- **action probabilities** — the model's distribution over its action
  vocabulary; in `explore` this is the observer's.
- **uncertainty** — normalised entropy of that distribution over about four
  seconds. Flat and low means the model is committed; a jagged high line means
  it is guessing.
- **objects** — the top slots with their id, velocity and `ctrl_prob`.
- **conv features (live)** — the four most active channels of the last conv
  layer.
- **sprite memory** — cluster prototypes from the object memory, tinted by
  verdict.
- **audio events** — recent sounds; the number and colour identify the cluster,
  brightness fades with age, and a border appears once the meaning is known
  (red for danger, green for reward).
- **thoughts** — a running log of decisions and events.
- **sound rings** on the game image — when a sound fires, the contrastive AV
  model (`--sound-loc`) marks where on screen it probably came from.
- **hud read** in the info column — the numbers the agent has read off the
  screen by itself. Before the digit reader has trained it says
  `learning digits...`.

## Controls in the live window

| Key or button | Action |
|---|---|
| `q`, Esc, window close, STOP | quit cleanly |
| `r`, RESTART | restart the episode |
| `m`, MUTE | speakers on/off — the model always hears the original audio |
| `c`, CAM toggle | heat map on/off; off frees the GPU |
| `b`, BOXES toggle | object boxes and vectors on/off |
| click a pad button | human hint: that button is pressed into the game for 12 frames |

Clicking the drawn gamepad overrides the model. It exists for the cases a policy
cannot reasonably be expected to handle — picking a character in Battletoads,
for instance. The button lights up and `human hint` appears in the thoughts log.
The CAM and BOXES toggles are drawn only in the live window, never in a
recording.

## Streaming

Capture the "NES Player" window and run with `--window --realtime --hd --loop`.
Game audio goes to the system output and is picked up by ordinary audio capture.
