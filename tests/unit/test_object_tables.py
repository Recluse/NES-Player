"""The scan must not call a mirrored counter damage, nor a wiped field HP.

Both mistakes are real. The wall retraction of 2026-09-03 came from
reading object *type* bytes as hit points, and the first Rush'n Attack
scan reported seventeen hit-point bytes that were one scroll counter
mirrored through a table.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "scripts" / "experiments"))


def test_counted_down_is_hp_and_wiped_is_not():
    import object_tables as ot

    counted = {"addr": 0x582, "from": 30, "to": 0, "steps": 30, "sizes": [1]}
    wiped = {"addr": 0x531, "from": 6, "to": 0, "steps": 2, "sizes": [2, 4]}
    hp, cleared = ot.classify([counted, wiped])
    assert hp == [counted]
    assert cleared == [wiped]


def test_lockstep_copies_are_discarded():
    import object_tables as ot

    fire = np.zeros((100, 2048), dtype=int)
    trace = np.maximum(215 - np.arange(100) // 2, 0)
    mirror = [0x5A3, 0x5C3, 0x5E3, 0x603, 0x623]
    for a in mirror:
        fire[:, a] = trace
    fire[:, 0x582] = np.maximum(30 - np.arange(100) // 3, 0)   # a real one
    cands = [{"addr": a} for a in [*mirror, 0x582]]
    keep, mirrored = ot.drop_mirrors(cands, fire)
    assert [c["addr"] for c in keep] == [0x582]
    assert len(mirrored) == len(mirror)
