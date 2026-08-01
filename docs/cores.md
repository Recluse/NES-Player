# Emulation cores

The core is chosen with `--core`, or `StableRetroAdapter(core=...)`. Frame and
audio are normalised to a canonical form afterwards, so **a model trained on one
core runs on another without retraining**.

| Core | Status | Native audio rate | Notes |
|---|---|---|---|
| **fceumm** | default, bundled | 32040 Hz | every trained model saw this one |
| **nestopia** | works | 48000 Hz | more accurate on lag frames, but won nothing on our movies |
| **quicknes** | works | 48000 Hz | fast; on some TAS movies it holds sync *longer* than the others |
| mesen | crashes the process | — | needs frontend services we do not have |
| FCEUX (not libretro) | unusable | — | the Qt build dies around frame 300 of a movie and writes no WAV |

Binaries are downloaded on demand from the official libretro builds
(`nes_player.emulator.cores.fetch`) for the host platform: `.dylib` on macOS,
`.so` on Linux, `.dll` on Windows.

There is no single winner on TAS synchronisation. Across ten movies fceumm held
sync longest on eight and quicknes on two — but on those two it lasted twice as
long, which is why choosing the core per movie is on the roadmap. Numbers are in
[experiments.md](experiments.md).

## How switching works, and how to break it

This is the part worth reading before touching `cores.py`.

The obvious approach — drop the second core's binary and its json next to the
default one — **silently replaces fceumm for every game and every process**.
Nothing errors. Frame size and audio rate change underneath models that were
trained on something else, and quality degrades with no signal at all. This
happened on 2026-08-01 and was caught only by regression tests holding golden
frame and audio hashes, which had been added that same morning.

Switching the core directory at runtime does not work either: the path is fixed
when `stable_retro` is imported, and importing the `retro` alias puts it back.

What does work: the binary is copied into the core directory, where by itself it
changes nothing because nothing references it, and the "Nes" platform is
re-registered **in the memory of this process only**, through
`RetroEmulator.load_core_info`. Other processes are untouched.

One core per process — libretro is loaded into the address space. To compare
cores, run one process per core; `scripts/experiments/core_compare.py` does
exactly that.
