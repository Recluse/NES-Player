"""Answers known before the test was written.

A wrong confidence interval looks exactly like a right one, which is why this
file checks against arithmetic that can be done by hand rather than against
whatever the code happened to print.
"""

import numpy as np

from nes_player.evaluation.stats import (
    detectable, mcnemar_exact, paired_bootstrap, permutation_p, t_interval,
    variance_split,
)


def test_permutation_is_one_on_symmetric_differences():
    """Differences symmetric about zero cannot favour either arm."""
    d = np.array([-3.0, 3.0, -1.0, 1.0, -2.0, 2.0])
    assert permutation_p(d) > 0.9


def test_permutation_is_small_when_every_pair_agrees():
    """Six identical wins: only the all-plus and all-minus sign patterns reach
    the observed mean, so the exact answer is 2/64 = 0.031. The estimate lands
    either side of that by sampling, so the bound is around it, not above it."""
    d = np.full(6, 5.0)
    p = permutation_p(d)
    assert abs(p - 2 / 64) < 0.01, p


def test_bootstrap_interval_covers_a_known_mean():
    rng = np.random.default_rng(0)
    inside = 0
    for trial in range(200):
        d = rng.normal(10.0, 5.0, 40)
        _, lo, hi, _ = paired_bootstrap(d, n=2000, seed=trial)
        inside += lo <= 10.0 <= hi
    assert inside >= 180, f"coverage {inside}/200, expected about 190"


def test_bootstrap_reports_wins_and_mean():
    d = np.array([1.0, 2.0, 3.0, -4.0])
    mean, lo, hi, wins = paired_bootstrap(d, n=2000)
    assert wins == 3
    assert abs(mean - 0.5) < 1e-9
    assert lo < mean < hi


def test_mcnemar_ignores_agreements():
    """Ties carry no information: padding them must not move the p-value."""
    a = np.array([True, True, False, False])
    b = np.array([False, False, True, True])
    padded_a = np.concatenate([a, np.ones(50, bool)])
    padded_b = np.concatenate([b, np.ones(50, bool)])
    assert mcnemar_exact(a, b) == mcnemar_exact(padded_a, padded_b) == 1.0


def test_mcnemar_matches_the_binomial_by_hand():
    """Five disagreements all one way: 2 * 2^-5 = 0.0625."""
    a = np.array([True] * 5 + [False] * 3)
    b = np.array([False] * 5 + [False] * 3)
    assert abs(mcnemar_exact(a, b) - 0.0625) < 1e-12


def test_t_interval_matches_the_table():
    """Six runs, sd 1: half-width is 2.571 / sqrt(6) = 1.0496."""
    x = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])   # sd exactly sqrt(0.8)
    mean, lo, hi = t_interval(x)
    expect = 2.571 * x.std(ddof=1) / np.sqrt(6)
    assert abs(mean) < 1e-12
    assert abs((hi - lo) / 2 - expect) < 1e-9


def test_detectable_shrinks_with_runs():
    assert detectable(263, 6) > detectable(263, 13) > detectable(263, 20)
    assert abs(detectable(165, 6) - 2.571 * 165 / np.sqrt(6)) < 1e-9


def test_variance_split_finds_pure_evaluation_noise():
    """Runs that differ only through their evaluation sample have no training
    noise, and the split has to say so."""
    rng = np.random.default_rng(1)
    per_run = [rng.normal(200.0, 700.0, 32) for _ in range(6)]
    out = variance_split(per_run)
    assert out["eval_share"] > 0.6
    assert out["train_noise"] < out["eval_noise"]


def test_variance_split_share_stays_a_share():
    """When the evaluation estimate exceeds the total, the training lottery is
    indistinguishable from zero — not negative, and not more than everything."""
    rng = np.random.default_rng(3)
    per_run = [rng.normal(0.0, 700.0, 32) for _ in range(4)]
    for _ in range(20):                       # some draws land above total by chance
        out = variance_split(per_run)
        assert 0.0 <= out["eval_share"] <= 1.0
        assert out["train_noise"] >= 0.0
        per_run = [rng.normal(0.0, 700.0, 32) for _ in range(4)]


def test_variance_split_finds_training_noise():
    """A large per-run offset on top of the same sampling must show up as
    training noise rather than evaluation noise."""
    rng = np.random.default_rng(2)
    offsets = [-600, -300, 0, 300, 600, 900]
    per_run = [rng.normal(o, 700.0, 32) for o in offsets]
    out = variance_split(per_run)
    assert out["train_noise"] > out["eval_noise"]
    assert out["eval_share"] < 0.4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
