# Command line

Everything runs as `uv run nes-player <command>`.

Flags shared by the commands that can show a picture: `--window` (live window),
`--realtime` (real speed, with sound), `--hd` (1920×1080), `--video-out
file.mp4` (recording; a `.wav` lands next to it, mux with
`ffmpeg -i v.mp4 -i v.mp4.wav -c:v copy -c:a aac out.mp4`), `--loop` (endless
episodes).

## `play` — run a trained model

```bash
uv run nes-player play --game SuperMarioBros-Nes-v0 --checkpoint runs/bc_smb_attn3 \
    --window --realtime --hd --auto-start --loop
```

| Flag | Meaning |
|---|---|
| `--auto-start` | get through the menus, recover after a game over, and never press START during play. Without it the model decides for itself and tends to stall on the title screen |
| `--temperature 0.9` | action sampling temperature |
| `--repeat 4` | how long a planned action is held; the policy itself decides at ~15 Hz |
| `--jump-hold N` | frames to hold A for a jump; the default comes from `runs/knowledge` with a floor of 32 |
| `--cam` / `--no-cam` | Grad-CAM overlay; turning it off frees the GPU |
| `--max-frames` | frames per episode, default 3600, unlimited under `--loop` |
| `--ghost PATH` | ego world model for the ghost trajectory |
| `--planner` | MPC on top of the ego model |
| `--sound-loc PATH` | AV-align checkpoint: draw a ring where a sound came from |
| `--state default` | start from the integration's savestate — needed by games whose title screen cannot be passed from power-on |
| `--core nestopia` | pick a different emulation core; the binary is checked against `cores.lock.json` before it is loaded |
| `--feedback strict` | where "something good or bad happened" comes from. See [Feedback](#feedback) — this is an input, not telemetry |

The frame stack advances once per **emulator frame**, in the game loop, while
the network decides at ~15 Hz in its own thread. Those used to be the same
event, which meant `--memory long` reached 128 *decisions* back rather than 128
frames — several times its documented window, and a different number on a
different machine.

## `explore` — instincts, no training involved

```bash
uv run nes-player explore --game ContraU-Nes-v0 --integrations integrations \
    --observer runs/bc_base41_attn1 --window --realtime --hd --loop
```

Phases: wait for the controls to respond → calibrate running speed and jump
height into `runs/knowledge/<game>.json` → explore, which means chasing
progress, jumping when stuck, poking at unfamiliar objects and dodging what has
proven dangerous.

`--observer CHECKPOINT` puts a trained network in the passenger seat: it fills
the panel with its attention map, action probabilities, uncertainty and conv
features without touching the controls.

`--record DIR` writes every episode as a Zarr dataset. This is where training
data for games without a synchronised TAS comes from — headless it produces
hundreds of episodes in minutes. An episode directory appears only once it is
complete: recording happens in `<name>.partial` and is renamed at the end, so a
crash or a quit leaves nothing that looks loadable but is not.

`--feedback` selects the same channel as in `play`; see below.

## Feedback

The instinct policy learns which objects are dangerous from two facts: the score
went up, and we died. Those become `danger` and `reward` labels, and the labels
choose buttons — so this is an **input**, and where it comes from decides whether
"pixels and sound only" is true.

| Mode | Source | Use |
|---|---|---|
| `strict` (default) | nothing | the agent is told nothing, so the claim holds. Danger and reward labels never form |
| `privileged` | emulator memory | a teacher may see more than its student; also for measurement |
| `pixel` | the on-screen panel, read like a person | the honest source, and not good enough yet |

Measured. Replaying identical frames with the telemetry scrambled changed 51% of
the actions before the fence existed and 0% after it. Turning the channel off
cost nothing measurable in play (score −15.8, t = −0.40 over eight paired
seeds), which says the labels were adding noise rather than knowledge.

`pixel` is left in place and not default because it does not work yet: against
memory over 4000 frames it agreed 12% of the time on Double Dragon, where the
digit reader locks onto the health bar, and 26% on Mario, where the number of
lives is not drawn during play at all.

## `train-bc` — behavioural cloning

```bash
uv run nes-player train-bc --episode datasets/explore_gradius --out runs/bc_gradius \
    --audio --attn 1.0 --epochs 3
```

`--audio` trains the multimodal model (video plus log-mel). `--episode` also
accepts a directory containing many episodes, in which case validation is the
last 10% of episodes taken whole rather than a random split — a random split
across frames of the same episode leaks.

`--attn 1.0` turns on attention supervision: the last conv layer's attention map
is penalised for looking outside the object boxes. Masks are cached next to the
episode, in a filename carrying the version of whatever produced them.

`--attn-source {tracker,oam}` chooses where those boxes come from. `tracker`
infers objects from motion in the pixels — the same thing the agent has to do,
and wrong often: of the cells it calls an object only **31%** contain a real
sprite, and it misses **43%** of the ones that are there. `oam` reads the
console's own sprite table, which is exact and works in every NES game. That is
a cheat, and supervision is allowed to cheat: the target says where to look, and
the network still has to find it in pixels when it plays.

`--memory {short,wide,long,epic}` sets how far back the frame stack reaches, in
frames: 4, 32, 128, 1280. The samples are spaced geometrically, so the window
grows without the network growing.

Training keeps the **best epoch by validation accuracy**, not the last one, and
says so in the log when they differ. Audio normalisation is computed from the
training split alone and stored in the checkpoint, so the same numbers apply
during training, validation and play.

`--init-from CHECKPOINT` reuses a base model's body and audio encoder and
retrains the heads — this is the few-shot transfer path.

The command prints validation accuracy next to the majority-class baseline. Read
them together, always.

## `improve` — self-imitation

```bash
uv run nes-player improve --checkpoint runs/bc_smb_si --rounds 8 --rollouts 8
```

Plays with sampling, keeps the best third of rollouts by reward and retrains on
them. It overwrites the checkpoint's `model.pt`, so copy the checkpoint first if
you want to keep the original.

- by default the reward comes from debug RAM: progress plus score change minus
  deaths;
- `--visual` takes the reward **from pixels only** — accumulated camera scroll —
  so it works on games with no RAM map, and RAM is never opened at all;
- `--start-pulses N` sends a series of STARTs at the beginning of each rollout,
  for games with an intro.

## `train-av` — contrastive sound↔frame

```bash
uv run nes-player train-av --episode datasets/smb_warps_tas,datasets/explore_smb \
    --out runs/av_smb
```

Prints `val_top1`, the accuracy of retrieving the right frame from a sound among
64 candidates, against a chance level of 1/64.

## `train-idm` — inverse dynamics

```bash
uv run nes-player train-idm --episode datasets/explore --out runs/idm_x
```

Guesses which button was pressed between two frames. This is the route to
learning from recordings that have no button track.

## `train-ego` — ego world model

```bash
uv run nes-player train-ego --episode datasets/explore_smb --out runs/ego_x
```

Predicts where the hero ends up for a given button sequence. **This is the model
`play --ghost` and `play --planner` actually load.** It had no CLI entry for a
while, so anyone following the documented commands trained the other one.

## `train-wm` — legacy world model

```bash
uv run nes-player train-wm --episode datasets/smb_warps_tas --out runs/wm_x
```

The action-blind latent model, kept to reproduce the project's recorded negative
result rather than because anything uses it. Prints `action_advantage`: how much
better it predicts when told the action than when not. Above 1 means actions
carry information the model can use.

## `train-slots` — neural slots (experimental)

```bash
uv run nes-player train-slots --episode datasets/explore_smb --out runs/slots_x
```

A slot-attention autoencoder. Version 1 is a documented negative result; see
[experiments.md](experiments.md). Kept for a second attempt.

## `tas-replay` — replay a TAS movie

```bash
uv run nes-player tas-replay --game SuperMarioBros-Nes-v0 \
    --movie 'tas/happylee-supermariobros,warped.fm2' --window
```

Checks the ROM MD5 recorded in the movie header, searches for the boot offset,
and prints frames, fps and the SHA-1 of the last frame.

## `dataset-build` — record an episode from a TAS movie

```bash
uv run nes-player dataset-build --game SuperMarioBros-Nes-v0 \
    --movie 'tas/...fm2' --out datasets/smb_warps_tas
```
