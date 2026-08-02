"""Choosing the NES emulation core (spec §8.2): fceumm by default.

Why this exists: TAS movies were recorded in FCEUX and BizHawk while we play on
fceumm, which is where the desynchronisation comes from. Cores differ in
accuracy, and the choice has to be settled by measurement rather than argument.

How the switching works — three approaches were tried and only the third
survives:

1. Put the second core and its json next to the default one. This SILENTLY
   replaces fceumm for every game and every process. Nothing errors; frame size
   and audio rate change underneath models trained on something else. Caught
   only by the golden frame and audio hashes in the regression tests.
2. Switch the core directory at runtime. Impossible: the path is fixed when
   `stable_retro` is imported, and importing the `retro` alias puts it back.
3. WORKING: copy the binary into the core directory, where by itself it changes
   nothing because nothing references it, then re-register the "Nes" platform
   IN THIS PROCESS ONLY via `RetroEmulator.load_core_info`.

One core per process — libretro is loaded into the address space. To compare
cores, run one process per core.

Frame and audio are normalised by the adapter, so a model always receives
240×224 at 32040 Hz whichever core produced it.
"""

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

BUILTIN = "fceumm"
KNOWN = ("fceumm", "nestopia", "quicknes", "mesen")   # open libretro builds


def _target() -> tuple[str, str]:
    """(buildbot directory, library extension) for the host platform."""
    arch = {"arm64": "arm64", "aarch64": "arm64"}.get(platform.machine(), "x86_64")
    if sys.platform == "darwin":
        return f"apple/osx/{arch}", ".dylib"
    if sys.platform == "win32":
        return f"windows/{arch}", ".dll"
    return f"linux/{arch}", ".so"


_DIR, LIBEXT = _target()
BUILDBOT = (f"https://buildbot.libretro.com/nightly/{_DIR}/latest/"
            "{name}_libretro" + LIBEXT + ".zip")
STORE = Path(__file__).parents[3] / "rnd" / "cores"   # binaries live outside the repo
# Which builds we have decided to trust. Checked in, so an experiment can say
# exactly which binary produced its numbers, and so a changed upstream artifact
# under the same `latest` URL is an error rather than a surprise.
MANIFEST = Path(__file__).parents[3] / "cores.lock.json"

_NES_SPEC = {
    "lib": None,
    "ext": ["nes"],
    "keybinds": ["Z", None, "TAB", "ENTER", "UP", "DOWN", "LEFT", "RIGHT", "X"],
    "buttons": ["B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"],
    "actions": [
        [[], ["UP"], ["DOWN"]],
        [[], ["LEFT"], ["RIGHT"]],
        [[], ["A"], ["B"], ["A", "B"]],
    ],
}
_active: str | None = None


def available() -> list[str]:
    """Cores that are ready to use right now."""
    got = sorted(p.stem.replace("_libretro", "") for p in STORE.glob(f"*_libretro{LIBEXT}"))
    return [BUILTIN] + [c for c in got if c != BUILTIN]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def _manifest_key(name: str) -> str:
    """One entry per core per platform: the binaries are not interchangeable."""
    return f"{name}/{_DIR}"


def fetch(name: str, pin: bool = False) -> Path:
    """Download a core from the official libretro builds, digest checked.

    The URL says `latest`, so what arrives under it changes without notice.
    Downloading a native library, marking it executable and loading it into the
    process is the most trusting thing this project does, and it used to happen
    with no record of what had been loaded — which is both a supply-chain hole
    and a reproducibility one: an experiment could not say which binary produced
    its numbers.

    So a core must be in the manifest, with its SHA-256, before it will run.
    `pin=True` records a new one — a deliberate act, on a build the operator has
    decided to trust, rather than something that happens quietly on first use.
    """
    if name not in KNOWN:
        raise ValueError(f"unknown core {name!r}; known: {', '.join(KNOWN)}")
    STORE.mkdir(parents=True, exist_ok=True)
    dest = STORE / f"{name}_libretro{LIBEXT}"
    manifest = _manifest()
    key = _manifest_key(name)
    expected = manifest.get(key, {}).get("sha256")

    if dest.exists():
        got = sha256(dest)
        if expected is None:
            if not pin:
                raise RuntimeError(
                    f"{dest} is not in {MANIFEST.name}. Its digest is {got}. "
                    f"Run with pin=True (or `nes-player`'s --pin-core) to record it "
                    f"after checking where it came from.")
            _pin(manifest, key, name, got)
        elif got != expected:
            raise RuntimeError(
                f"{dest} does not match {MANIFEST.name}: expected {expected}, "
                f"got {got}. Delete it to re-download, or investigate.")
        return dest

    url = BUILDBOT.format(name=name)
    with tempfile.TemporaryDirectory() as td:
        zip_path = Path(td) / "core.zip"
        subprocess.run(["curl", "-sL", "--max-time", "180", "-o", str(zip_path), url],
                       check=True)
        with zipfile.ZipFile(zip_path) as zf:
            inner = next(n for n in zf.namelist() if n.endswith(LIBEXT))
            # Written next to the destination and checked before it becomes the
            # real name, so a mismatched download never sits there executable.
            staged = Path(td) / dest.name
            staged.write_bytes(zf.read(inner))
        got = sha256(staged)
        if expected is not None and got != expected:
            raise RuntimeError(
                f"downloaded {name} does not match {MANIFEST.name}: expected "
                f"{expected}, got {got}. Upstream changed under the same URL.")
        if expected is None:
            if not pin:
                raise RuntimeError(
                    f"{name} is not pinned in {MANIFEST.name}. The download from "
                    f"{url} has digest {got}; record it with pin=True to accept it.")
            _pin(manifest, key, name, got, url)
        shutil.move(str(staged), dest)
    dest.chmod(0o755)
    return dest


def _pin(manifest: dict, key: str, name: str, digest: str, url: str | None = None) -> None:
    manifest[key] = {"core": name, "platform": _DIR, "sha256": digest,
                     "url": url or BUILDBOT.format(name=name)}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n")
    print(f"pinned {key} = {digest}", flush=True)


def digest_of(name: str) -> str | None:
    """The recorded digest of a core, for run metadata. None if not pinned."""
    return _manifest().get(_manifest_key(name), {}).get("sha256")


def use(name: str) -> None:
    """Switch the NES core. Must be called BEFORE the emulator first starts."""
    global _active
    if name == _active:
        return
    if _active is not None:
        raise RuntimeError(f"core already switched to {_active!r}: one core per process")
    if name == BUILTIN:
        _active = name
        return

    import stable_retro
    import stable_retro.data as data
    from stable_retro._retro import RetroEmulator

    lib = fetch(name)   # verifies the digest whether it was cached or downloaded
    # The binary has to sit in the core directory, whose path was baked in at
    # import time. On its own it switches nothing: nothing references it until
    # the platform is re-registered, and that happens only in this process.
    installed = Path(stable_retro.core_path()) / lib.name
    if not installed.exists():
        shutil.copy(lib, installed)
    if not RetroEmulator.load_core_info(json.dumps({"Nes": {**_NES_SPEC, "lib": name}})):
        raise RuntimeError(f"core {name!r} failed to register")
    data.EMU_CORES["Nes"] = installed.name
    _active = name


def active() -> str:
    return _active or BUILTIN
