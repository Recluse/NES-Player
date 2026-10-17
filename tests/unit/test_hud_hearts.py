import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from hud_hearts import read_hud  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
JUNGLE = ROOT / "runs/knowledge/bt_video/f_00880.png"


def test_reads_the_project_jungle_frame():
    """f_00880: level 1, player just spawned.

    6/6 bar (blue), 3 filled + 2 empty hearts (white), horizontal layout.
    This is the reproducible gold — owner JPEG uses a different layout and
    palette and is checked separately when present.
    """
    if not JUNGLE.exists():
        pytest.skip("jungle frame not on this machine")
    from PIL import Image

    img = np.array(Image.open(JUNGLE).convert("RGB"))
    r = read_hud(img)
    assert r["layout"] == "horizontal"
    assert r["bar_filled"] == 6
    assert r["bar_slots"] == 6
    assert r["hearts_filled"] == 3
    assert r["hearts_empty"] == 2
    assert r["hearts"] == 5
    assert r["score_digits"] == 6


def test_empty_frame_reads_nothing():
    f = np.zeros((224, 240, 3), np.uint8)
    r = read_hud(f)
    assert r["hearts"] == 0
    assert r["bar_filled"] == 0


def test_layout_flag_is_reported():
    f = np.zeros((224, 240, 3), np.uint8)
    assert read_hud(f, layout="vertical")["layout"] == "vertical"
    assert read_hud(f, layout="horizontal")["layout"] == "horizontal"
