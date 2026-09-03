"""An 8-bit scroll byte unwraps into a running position, in both directions.

Rush'n Attack keeps no 16-bit camera: byte 20 wraps every 256 px and no
byte anywhere ticks at the wrap. The planner restores savestates back and
forth across that boundary, so the unwrap has to be symmetric or a branch
that crossed a wrap would leave the main line 256 px off.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "scripts" / "experiments"))


def test_unwrap_is_symmetric_across_restores():
    import oracle_mpc as m

    m._UNWRAP.clear()
    seq = [250, 254, 2, 6,      # forward across the wrap
           250, 254, 2, 10,     # savestate restore back, forward again
           130, 129, 3]         # a 126-px drop is a move, not a wrap
    out = [m._unwrap("g", r) for r in seq]
    assert out == [250, 254, 258, 262, 250, 254, 258, 266, 386, 385, 259]
