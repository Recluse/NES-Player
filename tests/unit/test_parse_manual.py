import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from parse_manual import extract_from_pages, score  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GOLD = json.loads((ROOT / "assets/priors/gold/contra_j.json").read_text())


def test_extract_finds_control_facts_with_pages(tmp_path):
    (tmp_path / "page-009.txt").write_text(
        "The gun fires in eight directions.\nController uses the 十字キー.\n"
    )
    (tmp_path / "page-011.txt").write_text(
        "3D画面では前方にのみ攻撃発砲します\n"
    )
    parsed = extract_from_pages(tmp_path, game="ContraJ-Nes-v0")
    keys = {f["statement_key"] for f in parsed["facts"]}
    assert "eight-directions" in keys
    assert "3d-forward-only" in keys
    assert "dpad" in keys
    for f in parsed["facts"]:
        assert f["citation"]["page"] >= 1
        assert f["citation"]["quote"]


def test_score_hits_only_scoped_gold_rows():
    parsed = {
        "facts": [
            {"id": "eight-directions", "statement_key": "eight-directions",
             "citation": {"page": 9, "quote": "eight directions"}},
            {"id": "sensor-advance", "statement_key": "sensor-advance",
             "citation": {"page": 12, "quote": "detection sensor"}},
            {"id": "novel-thing", "statement_key": "novel-thing",
             "citation": {"page": 3, "quote": "something else"}},
        ]
    }
    result = score(parsed, GOLD, include_keys={"ja_manual"})
    assert result["hits"] == ["eight-directions"]
    # us_manual row matched but out of scope for a Japanese parse
    assert result["unscored_match"] == ["sensor-advance"]
    assert result["extra"] == ["novel-thing"]
    assert "sensor-advance" not in result["misses"]


def test_score_wrong_page_is_not_a_hit():
    parsed = {
        "facts": [
            {"id": "eight-directions", "statement_key": "eight-directions",
             "citation": {"page": 99, "quote": "eight directions"}},
        ]
    }
    result = score(parsed, GOLD, include_keys={"ja_manual"})
    assert result["hits"] == []
    assert result["wrong_page"] == [
        {"id": "eight-directions", "got": 99, "gold": 9}
    ]
    # extracted but mis-cited: not a hit, not a miss either
    assert "eight-directions" not in result["misses"]
    assert "fire-left" in result["misses"]


def test_raw_ocr_baseline_needs_text_and_never_guesses():
    parsed = {"facts": []}
    without = score(parsed, GOLD, include_keys={"ja_manual"})
    assert "unscored" in without["baseline_raw_ocr"]["note"]
    with_text = score(
        parsed, GOLD, include_keys={"ja_manual"},
        page_text="…左に発砲 … eight directions …",
    )
    # hint strings from gold appear in the text
    assert "fire-left" in with_text["baseline_raw_ocr"]["hits"]
    assert "eight-directions" in with_text["baseline_raw_ocr"]["hits"]


def test_empty_schema_baseline_is_zero_recall():
    parsed = {"facts": []}
    result = score(parsed, GOLD, include_keys={"ja_manual"})
    assert result["baseline_empty"]["recall"] == 0.0
    assert result["recall"] == 0.0


def test_vice_fixture_hits_all_gold_rows():
    """Synthetic Vice pages recover every gold row; still a pipeline check."""
    import json as _json

    from parse_manual import VICE_FIXTURE_PAGES, extract_from_pages
    from parse_manual import score as _score

    gold = _json.loads((ROOT / "assets/priors/gold/vice_j.json").read_text())
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for no, text in VICE_FIXTURE_PAGES.items():
            (tdp / f"page-{no:03d}.txt").write_text(text)
        parsed = extract_from_pages(tdp, game="fixture-vice")
    result = _score(parsed, gold, include_keys={"us_manual"})
    assert result["misses"] == []
    assert result["recall"] == 1.0
    assert result["precision"] is not None and result["precision"] > 0.7


def test_gold_without_page_skips_page_check():
    parsed = {"facts": [{
        "id": "lifeline", "statement_key": "lifeline",
        "citation": {"page": 99, "quote": "P is the player's lifeline"},
    }]}
    gold = {"facts": [{"statement_key": "lifeline", "page": None}]}
    result = score(parsed, gold)
    assert result["hits"] == ["lifeline"]
    assert result["wrong_page"] == []
