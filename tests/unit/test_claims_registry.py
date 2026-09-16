"""The claims registry has to stay loadable and complete.

`recheck_claims.py` can only recompute what the registry describes correctly,
and it runs against logs that are not in the repository. This checks the part
that is: that every line parses, carries the fields the recheck needs, and
names paired arms of equal length.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CLAIMS = ROOT / "docs/claims.jsonl"
REQUIRED = {"id", "claim", "doc", "unit", "effect", "lo", "hi", "wins", "of", "a", "b"}


def entries():
    for n, line in enumerate(CLAIMS.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)          # a broken line fails here, with its number
        if "_comment" in row:
            continue
        yield n, row


def test_every_claim_is_complete_and_paired():
    seen = set()
    count = 0
    for n, row in entries():
        count += 1
        missing = REQUIRED - row.keys()
        assert not missing, f"line {n} ({row.get('id')}) is missing {sorted(missing)}"
        assert row["id"] not in seen, f"line {n}: duplicate id {row['id']}"
        seen.add(row["id"])
        assert row["unit"] in ("run", "seed"), f"line {n}: unit {row['unit']!r}"
        assert len(row["a"]) == len(row["b"]), f"line {n}: arms are not paired"
        assert row["lo"] <= row["effect"] <= row["hi"], f"line {n}: effect outside interval"
        assert 0 <= row["wins"] <= row["of"], f"line {n}: wins out of range"
    assert count, "the registry is empty"


def test_the_doc_each_claim_points_at_exists():
    for n, row in entries():
        assert (ROOT / row["doc"]).exists(), f"line {n}: no {row['doc']}"


@pytest.mark.skipif(not (ROOT / "runs/oracle_adaptive").exists(),
                    reason="needs the run logs, which are not in the repository")
def test_recheck_runs_over_the_registry():
    import subprocess
    r = subprocess.run(["uv", "run", "python", "scripts/experiments/recheck_claims.py"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"a published number moved:\n{r.stdout}\n{r.stderr}"


if __name__ == "__main__":
    test_every_claim_is_complete_and_paired()
    test_the_doc_each_claim_points_at_exists()
    print("ok")
