# TAS movies

Movies come from [tasvideos.org](https://tasvideos.org), whose content is
CC BY-NC-SA with the authors named in the file headers. Download one with
`https://tasvideos.org/<id>M?handler=Download`.

**The movie files themselves are not in this repository**, and neither are the
ROMs they were recorded against. Both are yours to supply.

| File | Publication | ROM (verified by the MD5 in the header) |
|---|---|---|
| `lobsterzelda,feos-battletoadsanddoubledragonut-nes-2p.fm2` | [4323M](https://tasvideos.org/4323M) | Battletoads-Double Dragon (USA) |
| `happylee-supermariobros,warped.fm2` | [1715M](https://tasvideos.org/1715M) | Super Mario Bros. (World) |
| `happylee_mars608-smb-warpless.fm2` | [3728M](https://tasvideos.org/3728M) | Super Mario Bros. (World) |
| `mars608,aiqiyou-contraj-1p.fm2` | [4623M](https://tasvideos.org/4623M) | Contra (Japan) |
| `adelikat-gradius.fcm` | [711M](https://tasvideos.org/711M) | old FCM format, needs converting first |

Check the ROM hash on the publication page before importing. A movie replayed
against a different revision desynchronises, and the failure looks like the
agent simply playing badly rather than like an error.

Not every movie survives replay in our core, because fceumm lags on different
frames than FCEUX does. Where a movie drifts, the verified prefix before the
drift is still perfectly good training data — see
[../docs/experiments.md](../docs/experiments.md).
