"""verify_rom_matches: MD5 of the ROM without its iNES header.

A movie replayed against the wrong ROM revision does not fail loudly; it plays
for a while and then quietly desynchronises, which reads as the agent playing
badly. Hence the check up front.
"""

import base64
import hashlib

import pytest

from nes_player.tas.fm2 import FM2Movie
from nes_player.tas.replay import verify_rom_matches


def _movie_for(body: bytes) -> FM2Movie:
    md5 = base64.b64encode(hashlib.md5(body).digest()).decode()
    return FM2Movie(header={"romChecksum": f"base64:{md5}"}, commands=[], inputs=[])


def test_match_with_ines_header(tmp_path):
    """The movie's checksum covers the ROM body, not the 16-byte iNES header."""
    body = b"\x01\x02\x03" * 100
    rom = tmp_path / "a.nes"
    rom.write_bytes(b"NES\x1a" + b"\x00" * 12 + body)
    verify_rom_matches(_movie_for(body), rom)   # must not raise


def test_mismatch_raises(tmp_path):
    rom = tmp_path / "b.nes"
    rom.write_bytes(b"NES\x1a" + b"\x00" * 12 + b"other")
    with pytest.raises(ValueError, match="does not match"):
        verify_rom_matches(_movie_for(b"expected"), rom)
