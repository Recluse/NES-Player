"""Reading on-screen text with no labels and no emulator memory (spec §10.11).

The NES draws text as fixed 8×8 tiles, so no neural network is needed: a glyph
is an exact 64-bit signature. The problem is the other half — which signature
means "7" is unknown, because every game ships its own font.

Digits are learned from the DYNAMICS of counters:

1. Text cells are the ones that change over time but take few distinct values.
2. The digit ring: in the lowest digit of a counter, transitions form a cycle of
   length 10 (0→1→…→9→0). Build a "most frequent successor" graph and find the
   cycle in it.
3. Anchor zero by shape, not by frequency — see `fit` for why the obvious
   frequency approach reads every number one too high.
4. Direction: a timer counts down and a score counts up, so the ring alone
   cannot say which neighbour of zero is 1 and which is 9. Shape settles it:
   "1" is the thinnest glyph in any font.

`read(frame)` then returns one number per group of adjacent cells.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

TILE = 8
MIN_INK = 4       # fewer lit pixels than this and the cell counts as empty
MAX_VALUES = 14   # more distinct glyphs than this and it is artwork, not text


def _binarize_cell(cell: np.ndarray) -> int:
    """An 8×8 patch to a 64-bit signature; lit pixels are ink."""
    g = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
    thr = (int(g.max()) + int(g.min())) // 2
    bits = (g > max(thr, 40)).astype(np.uint64).ravel()
    if bits.sum() < MIN_INK or bits.sum() > TILE * TILE - MIN_INK:
        return 0
    return int(np.dot(bits, 1 << np.arange(64, dtype=np.uint64)))


def frame_cells(frame_rgb: np.ndarray) -> dict[tuple[int, int], int]:
    """Every cell of the frame to its signature; empty cells are skipped."""
    h, w = frame_rgb.shape[:2]
    out = {}
    for r in range(h // TILE):
        for c in range(w // TILE):
            sig = _binarize_cell(frame_rgb[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE])
            if sig:
                out[(r, c)] = sig
    return out


@dataclass
class HudReader:
    """Learns from the frames of one episode, then reads any frame."""

    digits: dict[int, int] = field(default_factory=dict)        # signature -> digit
    cells: list[tuple[int, int]] = field(default_factory=list)  # cells holding digits
    groups: list[list[tuple[int, int]]] = field(default_factory=list)

    # ---------- learning ----------

    def fit(self, frames, anchor_frames: int = 60) -> HudReader:
        seqs: dict[tuple[int, int], list[int]] = defaultdict(list)
        for f in frames:
            for pos, sig in frame_cells(f).items():
                seqs[pos].append(sig)
        n = len(frames)
        # Candidates: almost always non-empty, few distinct values, but changing.
        cand = {pos: s for pos, s in seqs.items()
                if len(s) > n * 0.5 and 1 < len(set(s)) <= MAX_VALUES}
        if not cand:
            return self
        # 1) Find the ring PER CELL and keep the longest. A graph built over
        # all cells at once mixes counters running in opposite directions —
        # a timer counting down against a score counting up — and degenerates.
        cycle: list[int] = []
        for s in cand.values():
            succ: dict[int, Counter] = defaultdict(Counter)
            for a, b in pairwise(s):
                if a != b:
                    succ[a][b] += 1
            c = _find_cycle(succ)
            if len(c) > len(cycle):
                cycle = c
        if len(cycle) < 8:   # no ring formed; the digits were not learned
            return self
        # 2) Anchor zero BY SHAPE. The tempting alternative — "counters read
        # zero at the start of an episode" — only holds if learning begins on
        # the very first frames; start later and every number comes out
        # shifted (Castlevania's 000000 score read as 111111).
        # "1" is the thinnest glyph in any font. Of its two neighbours in the
        # ring, zero is the one with a closed hole inside it.
        one = min(cycle, key=_ink)
        j = cycle.index(one)
        prev_g, next_g = cycle[j - 1], cycle[(j + 1) % len(cycle)]
        if _has_hole(prev_g) != _has_hole(next_g):
            prefer_prev = _has_hole(prev_g)
        else:   # both or neither have a hole: fall back to ink volume
            prefer_prev = _ink(prev_g) >= _ink(next_g)
        if prefer_prev:
            zero, order = prev_g, cycle[j - 1:] + cycle[:j - 1]
        else:   # the ring runs the other way round, so reverse it
            zero = next_g
            rev = cycle[::-1]
            k = rev.index(zero)
            order = rev[k:] + rev[:k]
        self.digits = {g: i for i, g in enumerate(order)}
        assert self.digits[zero] == 0 and self.digits.get(one) == 1
        # 3) Which cells hold digits, grouped into numbers by adjacency.
        # A digit cell is one where nearly every glyph seen turned out to be a
        # digit; the odd unknown is tolerated and makes that number read -1.
        # Searched over ALL cells rather than only the changing ones: a score
        # counter can sit on zero for the whole training run and still needs
        # to be readable afterwards.
        self.cells = sorted(
            p for p, s in seqs.items()
            if len(s) > n * 0.5
            and sum(v in self.digits for v in set(s)) >= 0.8 * len(set(s)))
        self.groups = _group_cells(self.cells)
        return self

    # ---------- inference ----------

    def read(self, frame_rgb: np.ndarray) -> list[int]:
        """One number per group of cells, ordered top to bottom, left to right."""
        if not self.groups:
            return []
        # Only the learned cells are read: scanning all 840 cells of a frame
        # is too expensive to do sixty times a second.
        cells = {p: _binarize_cell(frame_rgb[p[0] * TILE:(p[0] + 1) * TILE,
                                             p[1] * TILE:(p[1] + 1) * TILE])
                 for p in self.cells}
        out = []
        for grp in self.groups:
            value = 0
            for pos in grp:
                d = self.digits.get(cells.get(pos, 0))
                if d is None:   # one unknown glyph makes the whole number unreadable
                    value = -1
                    break
                value = value * 10 + d
            out.append(value)
        return out


def _ink(sig: int) -> int:
    return sig.bit_count()


def _has_hole(sig: int) -> bool:
    """Does the glyph enclose a hole? A zero does; a two does not.

    Topology beats ink volume here: in the Castlevania font the 2 is heavier
    than the 0 — 30 lit pixels against 26 — so picking zero by thickness chose
    the wrong glyph and every reading came out one too high.
    """
    grid = [[(sig >> (r * TILE + c)) & 1 for c in range(TILE)] for r in range(TILE)]
    seen = [[False] * TILE for _ in range(TILE)]
    stack = [(r, c) for r in range(TILE) for c in (0, TILE - 1) if not grid[r][c]]
    stack += [(r, c) for c in range(TILE) for r in (0, TILE - 1) if not grid[r][c]]
    while stack:   # flood the background inwards from the edges
        r, c = stack.pop()
        if seen[r][c]:
            continue
        seen[r][c] = True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < TILE and 0 <= nc < TILE and not grid[nr][nc] and not seen[nr][nc]:
                stack.append((nr, nc))
    return any(not grid[r][c] and not seen[r][c]
               for r in range(TILE) for c in range(TILE))


# ---------- letters: an atlas prior, and reading phrases ----------

_ATLAS_PATH = Path(__file__).parents[3] / "assets" / "nes_font.json"
_FUZZY_BITS = 5   # any looser and it starts confusing C with O, and 0 with D
_atlas_cache: dict | None = None


def font_atlas() -> dict[int, str]:
    """Glyph to character. Digits the agent works out itself; letters are given
    as a prior, because they have no dynamics to derive them from — the same way
    a person arrives at a game already knowing how to read."""
    global _atlas_cache
    if _atlas_cache is None:
        path = _ATLAS_PATH if _ATLAS_PATH.exists() else Path("assets/nes_font.json")
        try:
            raw = json.loads(path.read_text())
        except OSError:
            raw = {}
        _atlas_cache = {int(sig): ch for ch, sigs in raw.items() for sig in sigs}
    return _atlas_cache


def _match_glyph(sig: int, atlas: dict[int, str]) -> str:
    ch = atlas.get(sig)
    if ch is not None:
        return ch
    best, best_d = "?", _FUZZY_BITS + 1
    for known, c in atlas.items():
        d = (known ^ sig).bit_count()
        if d < best_d:
            best, best_d = c, d
    return best


def read_lines(frame_rgb: np.ndarray, gap: int = 2,
               reader: HudReader | None = None) -> list[str]:
    """Lines of on-screen text. Adjacent cells join up, a gap of `gap` or more
    becomes a space, an unknown glyph becomes "?". Letters come from the atlas
    prior and digits from whatever the HudReader learned for this game."""
    atlas = font_atlas()
    if reader is not None and reader.digits:
        atlas = atlas | {sig: str(d) for sig, d in reader.digits.items()}
    if not atlas:
        return []
    rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (r, c), sig in frame_cells(frame_rgb).items():
        rows[r].append((c, sig))
    out = []
    for r in sorted(rows):
        cells = sorted(rows[r])
        line, prev_c = [], None
        for c, sig in cells:
            if prev_c is not None and c - prev_c >= gap:
                line.append(" ")
            line.append(_match_glyph(sig, atlas))
            prev_c = c
        text = "".join(line).strip()
        if sum(ch != "?" for ch in text) >= 3:   # drop rows that are just artwork
            out.append(text)
    return out


# The button the screen is asking for: read the prompt instead of pressing blind.
_BUTTON_WORDS = ("START", "SELECT", "BUTTON")
_ASK_WORDS = ("PRESS", "PUSH", "HIT")


def find_prompt(frame_rgb: np.ndarray, reader: HudReader | None = None) -> str | None:
    """Returns 'START', 'SELECT' or None. Understands "PRESS START" and
    "PUSH START BUTTON".

    Prompts addressed to the SECOND player are ignored. In a two-player game
    "PLAYER 2 PRESS START" sits on screen throughout the level, and obeying it
    pauses the game over and over.
    """
    for line in read_lines(frame_rgb, reader=reader):
        words = line.replace("-", " ").split()
        if not any(w in _ASK_WORDS for w in words):
            continue
        if "2" in words or "2P" in words or "II" in words:
            continue   # that is aimed at player two, not at us
        for w in words:
            if w in ("START", "SELECT"):
                return w
        if "BUTTON" in words:   # "PUSH BUTTON" unnamed: START is the likeliest
            return "START"
    return None


def _find_cycle(succ: dict[int, Counter]) -> list[int]:
    """The longest cycle in the most-frequent-successor graph, up to 10 nodes."""
    nxt = {a: c.most_common(1)[0][0] for a, c in succ.items() if c}
    best: list[int] = []
    for start in nxt:
        seen, node = [], start
        while node in nxt and node not in seen:
            seen.append(node)
            node = nxt[node]
            if len(seen) > 12:
                break
        if node in seen:
            cyc = seen[seen.index(node):]
            if len(cyc) > len(best):
                best = cyc
    return best if len(best) <= 10 else []


def _group_cells(cells: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    groups: list[list[tuple[int, int]]] = []
    for pos in cells:
        if groups and groups[-1][-1][0] == pos[0] and pos[1] - groups[-1][-1][1] == 1:
            groups[-1].append(pos)
        else:
            groups.append([pos])
    return groups
