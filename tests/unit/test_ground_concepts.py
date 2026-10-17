import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from ground_concepts import build_contra, check  # noqa: E402

TABLE_PATH = (
    ROOT / "assets/priors/grounding/grounding_ContraJ-Nes-v0.json"
)


def test_shipped_contra_grounding_checks_clean():
    table = json.loads(TABLE_PATH.read_text())
    errors, warnings = check(table)
    assert errors == [], errors


def test_sensor_is_scan_confirmed_against_base_rooms():
    table = json.loads(TABLE_PATH.read_text())
    sensor = next(x for x in table["links"] if x["concept"] == "sensor")
    assert sensor["status"] == "scan_confirmed"
    assert sensor["entity"]["object_types"] == [20]
    assert "base_rooms" in sensor["evidence"]["artifact"]
    assert sensor["evidence"]["verify"]


def test_player_tiles_come_from_hero_scan():
    table = json.loads(TABLE_PATH.read_text())
    player = next(x for x in table["links"] if x["concept"] == "player")
    assert player["status"] == "scan_confirmed"
    assert player["entity"]["kind"] == "sprite_tiles"
    assert player["entity"]["tiles"]
    assert "hero_tiles" in player["evidence"]["scan"]


def test_confirmed_link_without_entity_is_an_error():
    table = {
        "schema_version": 1,
        "game": "X",
        "links": [{
            "concept": "sensor",
            "entity": None,
            "evidence": {"artifact": "x.json", "verify": "something"},
            "status": "scan_confirmed",
        }],
    }
    errors, _ = check(table)
    assert any("entity is null" in e for e in errors)


def test_proposed_with_entity_needs_an_artifact():
    table = {
        "schema_version": 1,
        "game": "X",
        "links": [{
            "concept": "sensor",
            "entity": {"kind": "object_type", "object_types": [1]},
            "evidence": {"artifact": None, "verify": "v"},
            "status": "proposed",
        }],
    }
    errors, warnings = check(table)
    assert any("no artefact" in w for w in warnings)


def test_unlinked_concepts_stay_proposed_with_null_entity():
    table = build_contra("ContraJ-Nes-v0")
    caps = next(x for x in table["links"] if x["concept"] == "capsule")
    assert caps["status"] == "proposed"
    assert caps["entity"] is None


def test_wall_is_scan_confirmed_when_zone_end_witness_exists():
    table = json.loads(TABLE_PATH.read_text())
    wall = next(x for x in table["links"] if x["concept"] == "wall")
    # The witness is scripts/verify_wall_zone_end.py watching wall_hp hit 0
    # and the screen fade. Without it the link must stay proposed.
    assert wall["status"] == "scan_confirmed"
    assert "wall_zone_end" in wall["evidence"]["artifact"]
    assert "wall_hp" in wall["evidence"]["verify"]
    assert wall["entity"]["object_types"] == [4, 16, 17]
