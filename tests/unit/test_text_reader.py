"""HudReader learns digits with no labels, checked on a synthetic HUD."""

import numpy as np

from nes_player.perception.text import HudReader

RNG = np.random.default_rng(7)


def _glyphs() -> list[np.ndarray]:
    """Ten distinguishable font tiles. Two properties mirror real fonts and are
    what makes label-free recognition possible: the 1 has the least ink, and the
    0 encloses a hole."""
    out = []
    for d in range(10):
        g = np.zeros((8, 8), np.uint8)
        if d == 0:   # a ring, with a hole
            g[1:7, 1:7] = 255
            g[3:5, 3:5] = 0
        elif d == 1:   # a thin stroke
            g[1:7, 4] = 255
        else:
            idx = RNG.choice(64, size=12 + d, replace=False)
            g.ravel()[idx] = 255
            g[3:5, 3:5] = 255   # filled in, so no enclosed hole
        out.append(g)
    return out


def _frame(value: int, glyphs, digits: int = 3) -> np.ndarray:
    f = np.zeros((224, 240, 3), np.uint8)
    text = str(value).zfill(digits)
    for k, ch in enumerate(text):
        col = 25 + k - (digits - 1)
        tile = glyphs[int(ch)]
        f[2 * 8:3 * 8, col * 8:(col + 1) * 8] = tile[..., None]
    return f


def test_learns_digits_and_reads_counter():
    glyphs = _glyphs()
    # A counter from 0 to 240, each value held for 3 frames, like timer ticks
    frames = [_frame(v, glyphs) for v in range(240) for _ in range(3)]
    reader = HudReader().fit(frames)

    assert len(reader.digits) == 10, "all ten digits should be learned"
    assert reader.groups, "digit cells should group into a number"

    for value in (7, 42, 199):
        assert reader.read(_frame(value, glyphs)) == [value]


def test_unknown_glyph_is_unreadable():
    glyphs = _glyphs()
    reader = HudReader().fit([_frame(v, glyphs) for v in range(200) for _ in range(3)])
    alien = _frame(123, glyphs)
    alien[16:24, 25 * 8:26 * 8] = 255   # a filled cell where a digit should be
    assert reader.read(alien) == [-1]
