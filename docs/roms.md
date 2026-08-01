# ROMs

**This project distributes no ROM images and never will.** Nothing here is a
game; these are the checksums the tests and integrations expect, so that you can
tell whether a dump you already own is the right one. Verify with
`shasum <file>.nes`.

Import your own dumps with:

```bash
uv run python -m retro.import /path/to/roms
```

Until that is done the integration tests skip themselves and the unit tests run
normally.

## Games with a built-in stable-retro integration

The SHA-1 must match exactly — a different revision of the same game has a
different memory layout, and the evaluation numbers would quietly become
meaningless.

| Game | Integration id | SHA-1 |
|---|---|---|
| Super Mario Bros. | `SuperMarioBros-Nes` | `facee9c577a5262dbe33ac4930bb0b58c8c037f7` |
| Battle City | `BattleCity-Nes` | `65185a71dbbc4c2be3cdb80182b10f761d75e848` |
| Double Dragon | `DoubleDragon-Nes` | `96356da50ff4cd4e3d399181ac19dafbf298eea8` |
| Ice Climber | `IceClimber-Nes` | `22d57ac6066529d199fcd299159d94820042c7d0` |
| Gradius | `Gradius-Nes` | `9282e87c643557682aee674174821bf3a0fe3876` |
| Battletoads | `Battletoads-Nes` | `d85c9ff489672534fbf61a15f8fa56fff489a34b` |

Game ids in stable-retro carry a `-v0` suffix, for example
`SuperMarioBros-Nes-v0`.

## Custom integrations

Kept in `integrations/`, created by `scripts/make_integration.py`. The ROMs
themselves are not in this repository; only the memory map, the metadata and the
expected checksum are.

- `BattletoadsDoubleDragon-Nes-v0`
- `ContraJ-Nes-v0` — Contra (Japan), historical: the `bc_contra_av` dataset
- `ContraU-Nes-v0` — Contra (USA), the current one. Same memory map; only
  `xscroll` is trustworthy, `lives` and `score` are garbage in both regions,
  which is a good reminder of why the agent reads the screen instead
- `Excitebike-Nes-v0`

## TAS movies

`.fm2` (FCEUX) or `.bk2` (BizHawk), from tasvideos.org, whose content is
CC BY-NC-SA with the authors named in the file headers. Each publication page
lists the ROM hash the movie was made against — check it before importing, or
the replay will desynchronise and you will spend an afternoon wondering why.
