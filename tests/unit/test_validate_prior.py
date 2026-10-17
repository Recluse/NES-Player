import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_prior import validate  # noqa: E402

PRIOR = ROOT / "assets/priors/prior_ContraJ-Nes-v0.json"
SCHEMA = ROOT / "assets/priors/schema/prior.schema.json"


def test_shipped_prior_passes_validator(capsys):
    rc = validate(PRIOR)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "ok" in out


def test_shipped_prior_is_schema_v2_with_cited_facts():
    prior = json.loads(PRIOR.read_text())
    assert prior["schema_version"] == 2
    assert prior["facts"], "concept layer must not be empty"
    for f in prior["facts"]:
        page = (f.get("citation") or {}).get("page")
        assert isinstance(page, int) and page >= 1, f
    # legacy flat fields still present so oracle_mpc.py runs unchanged
    for key in ("templates", "target_types", "target_px", "scope"):
        assert key in prior


def test_schema_file_names_the_split():
    schema = json.loads(SCHEMA.read_text())
    props = schema["properties"]
    assert "facts" in props
    assert "targets" in props
    assert "realisation" in json.dumps(props["targets"])


def test_validator_rejects_fact_without_page(tmp_path):
    bad = {
        "schema_version": 2,
        "game": "X-Nes-v0",
        "source": "a scan",
        "facts": [{
            "id": "oops",
            "kind": "goal",
            "statement": "no page",
            "citation": {"quote": "only a quote"},
        }],
    }
    path = tmp_path / "prior_bad.json"
    path.write_text(json.dumps(bad))
    rc = validate(path)
    assert rc == 1


def test_validator_warns_when_realisation_has_no_concepts(tmp_path):
    mixed = {
        "schema_version": 2,
        "game": "X-Nes-v0",
        "source": "a scan",
        "targets": {"realisation": {"types": [1, 2]}},
    }
    path = tmp_path / "prior_mixed.json"
    path.write_text(json.dumps(mixed))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_prior.py"), str(path)],
        capture_output=True, text=True,
    )
    assert "without a concept" in proc.stdout or "without a concept" in proc.stderr
