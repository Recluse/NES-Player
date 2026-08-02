"""What produced a result: written next to it, at the moment it was produced.

A checkpoint used to record its hyperparameters and nothing about the world it
was trained in — no commit, no library versions, no identity for the ROM, the
emulator core or the episodes. So a number could be reported but not repeated,
and two numbers from different weeks could not be told apart from two numbers
from different code.

The dirty flag matters as much as the commit. During the audit of this project
the working tree changed while the audit ran, which is exactly the case a bare
SHA describes wrongly.

Nothing here is expensive enough to think about: it runs once per training run.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 1


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def episode_id(path: Path) -> dict[str, Any]:
    """Identify an episode without reading gigabytes of frames.

    ponytail: metadata plus the action stream, not the frames. Frames are the
    bulk and are a deterministic function of the actions and the start state
    (checked — a replay reproduces them exactly), so hashing the small parts
    identifies the episode. Hash the frames too if a dataset ever arrives from
    somewhere that does not hold that property.
    """
    parts = [_sha256_file(path / "metadata.json"), _sha256_file(path / "actions.npy")]
    h = hashlib.sha256("".join(p or "" for p in parts).encode()).hexdigest()
    return {"id": path.name, "sha256": h[:32]}


def _torch_bits() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {}
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return {"torch": torch.__version__, "device": device,
            # PyTorch says outright that results are not guaranteed identical
            # across versions and platforms even with the same seed, so these
            # are part of the result rather than decoration.
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled())}


def collect(command: list[str] | None = None, *, config: dict | None = None,
            episodes: list[Path] | None = None, game: str | None = None,
            core: str | None = None, **extra: Any) -> dict[str, Any]:
    """Everything needed to say where a number came from."""
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "command": command if command is not None else sys.argv,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "uv_lock_sha256": _sha256_file(ROOT / "uv.lock"),
        **_torch_bits(),
    }
    if config is not None:
        out["config"] = config
    if episodes:
        out["episodes"] = [episode_id(p) for p in episodes]
    if game:
        out["game"] = game
        try:
            import stable_retro

            # The full id, version suffix included — that is what the data
            # module indexes by, and trimming it finds nothing.
            rom = Path(stable_retro.data.get_romfile_path(game))
            out["rom_sha256"] = _sha256_file(rom)
        except Exception as e:      # no ROM here, or a game id we cannot resolve
            out["rom_sha256"] = None
            out["rom_note"] = f"{type(e).__name__}: {e}"
    if core:
        from nes_player.emulator import cores

        out["core"] = {"name": core, "sha256": cores.digest_of(core)}
    out.update(extra)
    return out


def write(out_dir: str | Path, **kwargs: Any) -> Path:
    """Write run.json next to a checkpoint or a result."""
    path = Path(out_dir) / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collect(**kwargs), indent=2) + "\n")
    return path


UNKNOWN = {
    "schema": SCHEMA,
    "provenance": "unknown",
    "note": ("This checkpoint predates run.json. Its commit, library versions "
             "and dataset identity were not recorded and cannot be recovered "
             "without guessing, so nothing is claimed here rather than "
             "something plausible being invented."),
}


def mark_unknown(out_dir: str | Path) -> Path | None:
    """Label an older checkpoint honestly. Never overwrites a real record."""
    path = Path(out_dir) / "run.json"
    if path.exists():
        return None
    path.write_text(json.dumps(UNKNOWN, indent=2) + "\n")
    return path
