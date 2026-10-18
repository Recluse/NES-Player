"""Skill library: composed templates that transfer across games."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from skill_library import build_templates, load  # noqa: E402


def test_library_loads_and_has_abilities():
    lib = load()
    assert "abilities" in lib
    assert "compositions" in lib
    assert "jump" in lib["abilities"]
    assert "fire" in lib["abilities"]
    assert "pick-and-throw-side" in lib["compositions"]


def test_build_templates_expands_ability():
    t = build_templates(["jump"])
    names = [x["name"] for x in t]
    assert "jump" in names
    jump = next(x for x in t if x["name"] == "jump")
    assert jump["tap"] == ["A"]
    assert "found_by" in jump
    # run-jump-throw includes jump as a step, so it comes along
    assert "run-jump-throw" in names


def test_build_templates_expands_composition():
    t = build_templates(["pick-up", "throw-side"])
    names = [x["name"] for x in t]
    assert "pick-up" in names
    assert "throw-side" in names
    assert "pick-and-throw-side" in names
    # the composition has phases
    comp = next(x for x in t if x["name"] == "pick-and-throw-side")
    assert "phases" in comp
    assert len(comp["phases"]) == 2


def test_composition_only_applies_when_ability_present():
    t = build_templates(["fire"])
    names = [x["name"] for x in t]
    assert "fire" in names
    assert "run-and-fire" in names   # fire is a step
    assert "pick-and-throw-side" not in names


def test_unknown_ability_is_skipped():
    t = build_templates(["nonexistent"])
    assert t == []


def test_every_ability_has_a_citation():
    lib = load()
    for name, sk in lib["abilities"].items():
        assert sk.get("citation"), f"{name} missing citation"
