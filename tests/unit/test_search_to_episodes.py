import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "experiments"))

from search_to_episodes import useful_spans  # noqa: E402


def test_keeps_only_the_run_up_to_each_new_best():
    # flat noise, a climb at 100-110, flat again, a climb at 300-305
    p = np.zeros(400)
    p[100:111] = np.arange(1, 12)
    p[111:] = 11
    p[300:306] = 11 + np.arange(1, 7)
    p[306:] = 17
    spans = useful_spans(p, window=20, min_len=5)
    assert spans == [(80, 111), (280, 306)]


def test_a_path_that_never_improves_keeps_nothing():
    assert useful_spans(np.full(500, 7.0), window=50, min_len=1) == []


def test_short_spans_are_dropped():
    p = np.zeros(100)
    p[50] = 1
    assert useful_spans(p, window=3, min_len=10) == []
