"""Reading menu prompts: "PRESS START" on screen tells the agent what to press."""

import numpy as np
import pytest

from nes_player.perception.text import (
    TILE,
    _fuzzy_button,
    _is_ask,
    find_prompt,
    font_atlas,
    read_lines,
)

ATLAS = font_atlas()
pytestmark = pytest.mark.skipif(not ATLAS, reason="assets/nes_font.json is missing")


def _glyph_image(sig: int) -> np.ndarray:
    bits = [(sig >> k) & 1 for k in range(TILE * TILE)]
    return (np.array(bits, np.uint8).reshape(TILE, TILE) * 255)


def _render(text: str, row: int = 10, col0: int = 4) -> np.ndarray:
    """Draw the phrase with the very glyphs the atlas holds."""
    by_char: dict[str, int] = {}
    for sig, ch in ATLAS.items():
        by_char.setdefault(ch, sig)
    f = np.zeros((224, 240, 3), np.uint8)
    for k, ch in enumerate(text):
        if ch == " ":
            continue
        sig = by_char.get(ch)
        if sig is None:
            pytest.skip(f"the atlas has no glyph for {ch!r}")
        tile = _glyph_image(sig)
        c = col0 + k
        f[row * TILE:(row + 1) * TILE, c * TILE:(c + 1) * TILE] = tile[..., None]
    return f


def test_reads_press_start():
    frame = _render("PRESS START")
    assert "PRESS START" in read_lines(frame)
    assert find_prompt(frame) == "START"


def test_reads_select_prompt():
    assert find_prompt(_render("PRESS SELECT")) == "SELECT"


def test_no_prompt_on_plain_text():
    assert find_prompt(_render("GAME")) is None


def test_fuzzy_reads_a_half_decoded_title():
    # Contra's title comes back as "P??? SE?E??" — the atlas lacks glyphs,
    # but the known letters line up with PRESS SELECT.
    assert _is_ask("P???") is True
    assert _fuzzy_button("SE?E??") == "SELECT"
    assert _fuzzy_button("ST?RT") == "START"


def test_fuzzy_refuses_contradictions_and_all_unknowns():
    assert _fuzzy_button("HELLO?") is None
    assert _fuzzy_button("?????") is None
    assert _is_ask("???") is False
    assert _is_ask("HELLO") is False
