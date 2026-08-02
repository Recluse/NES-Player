"""A native library is not loaded unless we have said which one we trust.

The core is downloaded from a URL whose last path segment is `latest`, made
executable and loaded into this process. What arrives there changes without
notice, so an unpinned or changed binary must be an error, not a surprise.
"""

import json

import pytest

from nes_player.emulator import cores


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the module at an empty store and manifest."""
    monkeypatch.setattr(cores, "STORE", tmp_path / "cores")
    monkeypatch.setattr(cores, "MANIFEST", tmp_path / "cores.lock.json")
    cores.STORE.mkdir()
    return cores.STORE


def _fake_core(store, name: str, body: bytes) -> None:
    (store / f"{name}_libretro{cores.LIBEXT}").write_bytes(body)


def test_an_unpinned_core_is_refused_and_its_digest_reported(store):
    _fake_core(store, "nestopia", b"pretend this is a dylib")
    with pytest.raises(RuntimeError, match="not in cores.lock.json"):
        cores.fetch("nestopia")


def test_pinning_records_the_digest_and_then_it_loads(store):
    _fake_core(store, "nestopia", b"pretend this is a dylib")
    cores.fetch("nestopia", pin=True)
    entry = json.loads(cores.MANIFEST.read_text())[cores._manifest_key("nestopia")]
    assert entry["sha256"] == cores.sha256(store / f"nestopia_libretro{cores.LIBEXT}")
    assert cores.fetch("nestopia")            # no exception the second time


def test_a_changed_binary_is_refused(store):
    _fake_core(store, "quicknes", b"the build we pinned")
    cores.fetch("quicknes", pin=True)
    _fake_core(store, "quicknes", b"something else entirely")
    with pytest.raises(RuntimeError, match="does not match"):
        cores.fetch("quicknes")


def test_the_manifest_is_per_platform(store):
    """An arm64 dylib and an x86_64 one are different artifacts."""
    _fake_core(store, "mesen", b"arm64 build")
    cores.fetch("mesen", pin=True)
    keys = json.loads(cores.MANIFEST.read_text()).keys()
    assert all(cores._DIR in k for k in keys)


def test_an_unknown_core_is_rejected_before_any_download(store):
    with pytest.raises(ValueError, match="unknown core"):
        cores.fetch("definitely-not-a-core")


def test_the_checked_in_manifest_matches_the_cores_on_disk():
    """The real manifest, against the real binaries, if they are present."""
    for name in cores.KNOWN:
        lib = cores.STORE / f"{name}_libretro{cores.LIBEXT}"
        if not lib.exists():
            continue
        recorded = cores.digest_of(name)
        assert recorded is not None, f"{name} is on disk but not pinned"
        assert recorded == cores.sha256(lib), f"{name} differs from the manifest"
