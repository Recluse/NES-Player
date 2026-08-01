from pathlib import Path

import pytest

from nes_player.tas.fm2 import parse_fm2

FM2 = Path(__file__).parent.parent.parent / "tas" / "happylee-supermariobros,warped.fm2"


@pytest.mark.skipif(not FM2.exists(), reason="tas files not downloaded")
def test_parse_smb_warped():
    movie = parse_fm2(FM2)
    assert movie.header["romFilename"].startswith("Super Mario Bros.")
    assert movie.rom_md5.hex() == "8e3630186e35d477231bf8fd50e54cdd"
    assert movie.players >= 1
    assert len(movie.inputs) > 17000  # 04:57 @ 60.1 fps
    assert len(movie.commands) == len(movie.inputs)
    pressed_ever = set().union(*(p for frame in movie.inputs for p in frame))
    assert "RIGHT" in pressed_ever and "A" in pressed_ever


def test_parse_minimal(tmp_path):
    fm2 = tmp_path / "m.fm2"
    fm2.write_text(
        "version 3\nromFilename X\nromChecksum base64:AAAAAAAAAAAAAAAAAAAAAA==\n"
        "port0 1\nport1 0\nfourscore 0\n"
        "|2|R......A|........||\n|0|.L......|........||\n"
    )
    movie = parse_fm2(fm2)
    assert movie.inputs[0][0] == frozenset({"RIGHT", "A"})
    assert movie.inputs[1][0] == frozenset({"LEFT"})
    assert movie.commands[0] == 2
