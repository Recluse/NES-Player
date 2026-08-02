# Architecture

## Layers

```
ROM → stable-retro (fceumm, headless) ─ EmulatorAdapter
        │ 240×224 RGB frame, 32040 Hz mono PCM, debug RAM
        │ ↑ CANONICAL form, identical whichever core is loaded — see below
        ▼
Perception (CPU, classical — no neural networks)
  ├── MotionTracker: camera scroll (phase correlation) + moving blobs → slots
  │                  (blobs under 24 px pass only with a world velocity: bullets)
  ├── SpriteTracker: the same slots read from the console's sprite table —
  │                  exact, privileged, for supervision and measurement only
  ├── ObjectMemory: sprite clusters + contact outcomes → danger / reward
  ├── Feedback: strict / privileged / pixel — see the fence below
  ├── AudioEventDetector: log-mel flux onsets → sound clusters and their meaning
  ├── HudReader: on-screen counters, digits learned without labels
  └── SoundLocator (neural, optional): contrastive sound↔frame → where it came from
        ▼
Policies
  ├── BCPolicy: CNN over video, or video + audio; behavioural cloning plus self-imitation
  ├── StatePolicy: the privileged teacher — 36 numbers from the sprite table
  └── InstinctPolicy: control calibration and exploration rules, no training at all
        ▼
Action: an NES button bitmask, opposite directions resolved away
```

## The fence

The agent plays from pixels and sound. Emulator memory exists and is trivially
readable, so the boundary has to be a mechanism rather than an intention.

- **The observation channel** is pixels and audio. Nothing else is passed to a
  policy that plays.
- **The feedback channel** — "the score went up", "we died" — is an input too,
  because it becomes danger/reward labels and those pick buttons. Its source is
  named explicitly (`perception/feedback.py`) and defaults to telling the agent
  nothing.
- **Supervision and teachers may cheat.** Attention targets from the sprite
  table, and the state teacher that reads object positions, are on the training
  side of the fence. The student they produce is not.
- **The check is a test, not a comment.** Replay the same frames with the
  telemetry scrambled; in strict mode the action trace must be identical. It was
  48.9% identical before this existed.

## Modules

```
src/nes_player/
├── emulator/
│   ├── adapter.py        EmulatorAdapter protocol and EmulatorObservation
│   ├── stable_retro.py   headless backend: power-on snapshot, raw buttons, 2 players
│   ├── controller.py     ControllerState, bitmasks, impossible-direction guard
│   └── cores.py          switching the libretro core, per-OS binaries
├── tas/
│   ├── fm2.py            FCEUX .fm2 parser (header, commands, ports)
│   └── replay.py         deterministic replay: ROM MD5 check, boot offset search
├── data/
│   ├── writer.py         episode → Zarr (zstd frames and audio) + npy + metadata
│   └── reader.py         Episode: lazy reads, audio sliced per frame
├── perception/
│   ├── motion.py         MotionTracker/Slot: scroll, difference, tracking, ctrl_prob
│   ├── memory.py         ObjectMemory: 16×16 prototypes, contacts, verdicts
│   ├── audio_events.py   AudioEventDetector: onsets, clusters, death/reward meaning
│   ├── av_align.py       AVAlign/SoundLocator: contrastive sound↔frame (InfoNCE)
│   ├── sprites.py        shadow OAM at $0200 → exact object positions, SpriteTracker
│   ├── feedback.py       where "good/bad happened" comes from; strict by default
│   ├── text.py           HudReader: unsupervised digits, screen prompts
│   ├── title.py          TitleWatch/TitleTracker: title screen from pixels
│   └── slots.py          SlotAE: neural slots (v1 — a negative result)
├── policy/
│   ├── bc.py             BCNet / BCNetAV, ActionVocab, train_bc, BCPolicy, Grad-CAM
│   ├── improve.py        self-imitation: rollouts → retrain on the best ones
│   ├── planner.py        EgoPlanner: MPC over the ego model (templates × 16 steps)
│   ├── idm.py            inverse dynamics: recover the button between two frames
│   ├── state_teacher.py  privileged teacher: 36 numbers in, trained on results
│   └── instinct.py       InstinctPolicy: wait → calibrate → explore, knowledge base
├── world_model/
│   ├── model.py          WorldModel v1/v2 (latent, action-blind — kept as history)
│   └── ego.py            EgoModel v4: crop around the hero → (dx, dy), GhostPredictor
├── provenance.py         run.json: commit, dirty flag, versions, ROM/core/data ids
├── evaluation/
│   ├── evaluator.py      synchronous, frame-indexed play: observe every frame,
│   │                     decide on a fixed grid, hold in between
│   ├── dashboard.py      the 1280×720 panel (×1.5 for HD)
│   └── viewer.py         window, recording, sound, threading — see below
└── cli/
    ├── args.py           the argument parser, nothing else
    ├── play.py           trained policy
    ├── explore.py        instinct policy and the observer network
    ├── train.py          one wrapper per learning module
    ├── data.py           TAS replay, dataset building
    └── runtime.py        per-frame helpers shared by play and explore
```

## Threading in the live window

The constraint: emulation and audio must hold 60.1 frames per second. The macOS
GUI is slow (`imshow` plus `waitKey` costs 10–25 ms) and GPU calls spike to
15 ms. Nothing that spikes may sit in the path that feeds the speakers.

| Thread | Work | Rate |
|---|---|---|
| Game (background) | emulation, perception, panel rendering, PCM push | 60.1 fps, paced by blocking on the audio ring (~60 ms) |
| Brain | `policy.act` and Grad-CAM — every GPU call | ~15 decisions/s |
| Audio pump | ring buffer → blocking PortAudio write | paced by CoreAudio |
| GUI (main) | `imshow` of the newest frame, keyboard and mouse | as fast as it manages, ≈60 |

Two lessons paid for in debugging: a Python callback inside the CoreAudio
realtime thread crackles, because it loses the GIL — only a blocking write from
a thread we own works; and pacing with a fixed `waitKey(16 ms)` adds to the
processing time instead of absorbing it, slowing everything down.

## Canonical frame and audio

The PPU draws 256×240, but the edges disappeared behind the bezel of a CRT, and
**many games leave garbage there** — a dirty left column, fragments of tiles
along the top. Without cropping, the motion tracker picks that garbage up as
moving objects.

Cores also disagree: our fceumm build already crops to 240×224 at 32040 Hz,
while nestopia and mesen hand back the full 256×240 at 48 kHz. The adapter
normalises both (`normalize_frame`, `resample_pcm`), so **a trained model does
not care which core produced its input**.

| | canonical (`viewport="tv"`, the default) | `viewport="raw"` |
|---|---|---|
| frame | 240×224 (8 px cropped from each side) | whatever the core returned |
| audio | 32040 Hz | the core's own rate |

## Emulation cores

Selected with `--core`, or `StableRetroAdapter(core=...)`. See
[cores.md](cores.md) for the comparison and for the trap in how switching has
to be implemented.

## Episode data

```
datasets/<episode>/
├── metadata.json      game, movie, sample rate, frame count
├── episode.zarr/      frames (N,224,240,3) uint8 zstd; audio (S,) int16 zstd
├── actions.npy        (N, players) uint8 — button bitmasks
└── audio_offsets.npy  (N+1,) — frame i owns audio[off[i]:off[i+1]]
```

A full Super Mario Bros. warp run is 17,866 frames and 9.5M samples in 26 MB,
roughly 127× compression.

## Surviving a long stream

- Auto-start presses START at frame 60 and again in a series of pulses, because
  games with an intro swallow the first one; a stray pause is cleared by the
  frozen-screen watchdog.
- `TitleWatch` fingerprints the logo band, so a return to the title screen after
  a game over is detected from pixels alone and answered with START.
- The policy is not allowed to press START or SELECT during play — there it is
  the pause button — and the menus are handled by auto-start instead.
- MUTE silences the speakers only: the policy and the recording still get the
  original PCM.
