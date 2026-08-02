"""A result records where it came from, or says plainly that it does not know."""

import json

from nes_player import provenance


def test_a_record_names_the_code_and_the_environment():
    r = provenance.collect(command=["nes-player", "train-bc"])
    for key in ("schema", "command", "git_commit", "dirty", "python",
                "platform", "uv_lock_sha256"):
        assert key in r, f"{key} missing from the run record"
    assert isinstance(r["dirty"], bool)


def test_the_dirty_flag_is_separate_from_the_commit():
    """A SHA describes the tree only if the tree matches it."""
    r = provenance.collect()
    assert "dirty" in r and r["git_commit"] is not None


def test_episodes_are_identified_without_reading_the_frames(tmp_path):
    ep = tmp_path / "ep001"
    ep.mkdir()
    (ep / "metadata.json").write_text('{"game": "Test-Nes-v0", "frames": 2}')
    (ep / "actions.npy").write_bytes(b"\x93NUMPY fake actions")
    a = provenance.episode_id(ep)
    assert a["id"] == "ep001" and len(a["sha256"]) == 32

    (ep / "actions.npy").write_bytes(b"\x93NUMPY different actions")
    assert provenance.episode_id(ep)["sha256"] != a["sha256"]


def test_write_puts_run_json_next_to_the_checkpoint(tmp_path):
    provenance.write(tmp_path, config={"epochs": 3})
    r = json.loads((tmp_path / "run.json").read_text())
    assert r["config"] == {"epochs": 3}


def test_an_older_checkpoint_is_labelled_not_invented(tmp_path):
    provenance.mark_unknown(tmp_path)
    r = json.loads((tmp_path / "run.json").read_text())
    assert r["provenance"] == "unknown"
    assert "git_commit" not in r, "a guessed commit is worse than none"


def test_marking_never_overwrites_a_real_record(tmp_path):
    provenance.write(tmp_path, config={"epochs": 3})
    assert provenance.mark_unknown(tmp_path) is None
    assert json.loads((tmp_path / "run.json").read_text())["config"] == {"epochs": 3}


def test_an_unresolvable_rom_is_reported_not_hidden():
    r = provenance.collect(game="NoSuchGame-Nes-v0")
    assert r["rom_sha256"] is None
    assert "rom_note" in r
