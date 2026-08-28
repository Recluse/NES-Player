# Pre-registration: the learned chooser

Written 2026-08-23, before the training set it governs finished collecting and
before any number from it was seen. It exists because the 32 seeds every arm
has been scored on are no longer a test set: dozens of decisions have been
made after looking at them, and an effect the size of the learned probe's
+449 cannot survive that kind of reuse. Large effects — the +4590 of an
averaged-tail oracle — are not seriously threatened. Small ones are.

Everything below is fixed now. If any of it changes, this file changes with a
dated note saying what and why, and the run does not count as confirmatory.

## The claim to be tested

A student trained to imitate *which candidate* an averaged-tail oracle picks
plays further than the reactive policy, and further than the best fixed
template.

Every previous student regressed the plan's value and took an argmax of its
own estimate. This one is given the choice directly.

## Seeds

* **Development**: 0-31. Everything measured so far. Model selection,
  hyper-parameters, sanity checks, anything that involves looking.
* **Training data**: 9100 and up (`draw_matrix_big.npz`), 5000-5041 (the
  DAgger collection), 9000-9003 (the first matrices). Disjoint from both other
  blocks.
* **Confirmatory**: **4000-4031**, thirty-two seeds, never used by any arm in
  this project. They are looked at once, after everything below is frozen.

## The student

* Architecture: `plan_probe.Probe`, the `strip` input variant — the hero crop
  plus the forward band, hero velocity as the vector part. Unchanged from
  every previous probe, so the comparison is about the target and not the net.
* Output: six logits, one per candidate (`bc`, `run`, `jump now`,
  `jump later`, `wait`, `back off`).
* Target: `p_i = P(i = argmax_j Q_j)`, estimated by 200 bootstrap resamples of
  the sixteen saved returns per candidate. Death is terminal: a draw that dies
  scores the worst surviving return in that point minus 200 px.
* Loss: cross-entropy against `p_i`, each point weighted by
  `1 - H(p_i)/log 6`, floored at 0.1. A point whose label is a coin is not
  taught as a fact.
* Optimiser: Adam, lr 1e-3, batch 128, 40 epochs, seed 0.
* Draws: sixteen, with common random numbers — one stream per draw shared
  across candidates. Measured to shrink the standard deviation of a paired
  difference by 18-22% against independent sampling.

## Selecting the one configuration to test

On the development data only — the held-out *runs* of the training matrix, not
the confirmatory seeds — exactly one choice is made: soft target against hard
one-hot (`--hard`), and weighted against unweighted (`--no-weight`). Four
configurations, judged by regret under the sixteen-draw mean on held-out runs.
The winner is the single model that goes to the confirmatory seeds. Ties go to
the soft, weighted version, as the pre-registered default.

Nothing else is tuned. If the winner's regret is worse than the best constant
template's on the same held-out runs, the confirmatory run is not made at all
and the result is reported as a failure at the development stage.

## The confirmatory run

Three arms on seeds 4000-4031, 3000 frames, commit 16, horizon 48, paired:

1. `bc` — the reactive policy.
2. `always jump now` — the best fixed template, with the scoring loop actually
   skipped.
3. the student.

Metric: `progress = level * 4000 + x`, the best reached in the run.

## Success criterion, stated in advance

The claim is confirmed only if **both** hold:

* the student beats `bc` — paired permutation test on the per-seed
  differences, 10000 permutations, two-sided, p < 0.05;
* the student beats `always jump now` by the same test, p < 0.05.

Reported alongside, not as criteria: the median and mean progress, a paired
bootstrap 95% interval on each difference, deaths per arm, and McNemar's exact
test on 1-1 completions.

If only the first holds, the honest statement is "better than the policy, not
shown to be better than a fixed habit". If neither holds, the direction is
reported as failed, and the failure is not re-analysed into a success by
subsetting seeds, changing the metric, or extending the frame budget.

## What would make this invalid

* Looking at 4000-4031 before the model is frozen.
* Running the confirmatory arms more than once and reporting the better run.
* Choosing the comparison template after seeing the results — it is
  `jump now`, chosen now, because it carries 52% of the full candidate set's
  value on the development matrix and had the lowest constant regret there.

  Noted the same day, before the confirmatory run: on the development *game*
  metric the strongest constant is `run` (-268 px against the policy) rather
  than `jump now` (-292 px). The arm stays `jump now` — it was chosen on the
  matrix, which is the development artefact the student is trained from, and
  switching now would be selection on the game numbers. The 24 px between
  them is far inside the seed spread, and both lose heavily to the policy, so
  the binding half of the criterion is beating `bc` either way.
* Changing the metric, the frame budget or the commitment after the fact.
