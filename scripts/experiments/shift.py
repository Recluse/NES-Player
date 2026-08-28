"""Covariate shift, or the model being wrong? They look the same from outside.

The failed student picks `wait` 34% of the time where its teacher, measured on
the teacher's own states, picks it 5%. Two very different things produce that:

* the teacher would also wait on the states the student reaches, and the
  marginal moved because the states did — covariate shift, which DAgger
  addresses;
* the teacher would still choose otherwise there, and the student is simply
  wrong — which DAgger does not fix.

`draw_matrix.py --driver` records both picks on the same states. This reads
them out: the two histograms side by side, the confusion between them, the
oracle regret each error costs, and — the number the uniform corruption curve
could not give — what a perfect scorer would score if it made errors with
exactly this structure.

    uv run python scripts/experiments/shift.py \
        runs/knowledge/draw_matrix_student.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    from analyse_draws import penalised

    ap = argparse.ArgumentParser()
    ap.add_argument("matrix")
    ap.add_argument("--reference", default=None,
                    help="a matrix collected on the policy's own states, for "
                         "the teacher's marginal there")
    args = ap.parse_args()

    z = np.load(args.matrix)
    names = [str(n) for n in z["names"]]
    if "student" not in z or (z["student"] < 0).all():
        raise SystemExit("this matrix has no student picks; collect with "
                         "--driver")
    truth = penalised(z).mean(2)
    teacher = truth.argmax(1)
    student = z["student"]
    n = len(student)
    print(f"{n} decisions on the student's own states, "
          f"{len(set(z['run'].tolist()))} runs")

    print()
    print(f"{'candidate':12} {'teacher here':>13} {'student':>9}", end="")
    if args.reference:
        zr = np.load(args.reference)
        ref = penalised(zr).mean(2).argmax(1)
        print(f" {'teacher on bc states':>21}", end="")
    print()
    for k, name in enumerate(names):
        row = (f"{name:12} {float((teacher == k).mean()):12.1%} "
               f"{float((student == k).mean()):8.1%}")
        if args.reference:
            row += f" {float((ref == k).mean()):20.1%}"
        print(row)

    agree = float((student == teacher).mean())
    print()
    print(f"student agrees with the teacher here: {agree:.1%}")
    print()
    print("confusion, rows = teacher, columns = student")
    print(f"{'':12}" + "".join(f"{s[:9]:>10}" for s in names))
    for k, name in enumerate(names):
        m = teacher == k
        cells = "".join(
            f"{float((student[m] == j).mean()) if m.any() else 0.0:10.0%}"
            for j in range(len(names)))
        print(f"{name:12}{cells}   n={int(m.sum())}")

    regret = truth.max(1) - truth[np.arange(n), student]
    wrong = student != teacher
    print()
    print(f"oracle regret of the student's choice: {regret.mean():.1f} px "
          f"overall, {regret[wrong].mean():.1f} px on the ones it got wrong")
    to_wait = wrong & (student == names.index("wait"))
    print(f"errors that are 'the teacher did not say wait, the student did': "
          f"{float(to_wait.mean()):.1%} of all decisions, "
          f"{float(to_wait.sum() / max(wrong.sum(), 1)):.1%} of the errors, "
          f"costing {regret[to_wait].mean() if to_wait.any() else 0.0:.1f} px "
          f"each")
    print(f"regret if it had followed the teacher exactly: 0.0 px; "
          f"if it always took bc: "
          f"{(truth.max(1) - truth[:, 0]).mean():.1f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
