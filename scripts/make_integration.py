"""Create a minimal custom stable-retro integration from a .nes file.

Usage: uv run python scripts/make_integration.py "/path/to/Game (USA).nes" GameId-Nes-v0
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

INTEGRATIONS = Path(__file__).parent.parent / "integrations"


def main() -> None:
    rom_path, game_id = Path(sys.argv[1]), sys.argv[2]
    data = rom_path.read_bytes()
    body = data[16:] if data[:4] == b"NES\x1a" else data   # retro hashes without the header
    dest = INTEGRATIONS / game_id
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(rom_path, dest / "rom.nes")
    (dest / "rom.sha").write_text(hashlib.sha1(body).hexdigest() + "\n")
    (dest / "data.json").write_text(json.dumps({"info": {}}, indent=2) + "\n")
    (dest / "metadata.json").write_text(json.dumps({}, indent=2) + "\n")
    print(f"{game_id}: {dest}")


if __name__ == "__main__":
    main()
