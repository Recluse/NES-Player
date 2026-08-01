"""Parser for the FCEUX .fm2 text format: https://fceux.com/web/FM2.html"""

import base64
from dataclasses import dataclass
from pathlib import Path

# Character order inside an FM2 port field: RLDUTSBA
_FIELD_ORDER = ("RIGHT", "LEFT", "DOWN", "UP", "START", "SELECT", "B", "A")

CMD_SOFT_RESET = 1
CMD_POWER = 2


@dataclass
class FM2Movie:
    header: dict[str, str]
    commands: list[int]   # per-frame flags: soft reset, power, fds
    inputs: list[tuple[frozenset[str], ...]]   # per-frame buttons, by port

    @property
    def rom_md5(self) -> bytes | None:
        """MD5 of the ROM without its iNES header, from the romChecksum field.

        None when the field is missing or corrupt. Real movies from TASVideos
        sometimes carry base64 without its '=' padding, so we pad it ourselves —
        otherwise the parser dies part way through a bulk download.
        """
        value = self.header.get("romChecksum", "").removeprefix("base64:").strip()
        if not value:
            return None
        try:
            return base64.b64decode(value + "=" * (-len(value) % 4))
        except Exception:  # noqa: BLE001 — a corrupt checksum just counts as absent
            return None

    @property
    def players(self) -> int:
        return sum(1 for k in ("port0", "port1") if self.header.get(k) == "1")


def _parse_port(field: str) -> frozenset[str]:
    # strict=False on purpose: real movies in the wild have short or padded
    # port fields, and a truncated one should read as "those buttons up",
    # not crash the parser.
    return frozenset(
        name for ch, name in zip(field, _FIELD_ORDER, strict=False) if ch not in ". "
    )


def parse_fm2(path: str | Path) -> FM2Movie:
    header: dict[str, str] = {}
    commands: list[int] = []
    inputs: list[tuple[frozenset[str], ...]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith("|"):
                fields = line.split("|")  # ['', cmd, port0, port1, port2, '']
                commands.append(int(fields[1] or 0))
                inputs.append(tuple(_parse_port(p) for p in fields[2:4]))
            elif line and not line.startswith("comment"):
                key, _, value = line.partition(" ")
                header[key] = value
    if header.get("fourscore", "0") != "0":
        raise NotImplementedError("fourscore movies are not supported")
    if int(header.get("version", 3)) != 3:
        raise ValueError(f"unsupported FM2 version: {header.get('version')}")
    return FM2Movie(header=header, commands=commands, inputs=inputs)
