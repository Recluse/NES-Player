"""A text launcher: pick a game and options from a menu instead of typing flags."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# (display name, integration id, --integrations directory)
GAMES = [
    ("Super Mario Bros.", "SuperMarioBros-Nes-v0", None),
    ("Contra (US)", "ContraU-Nes-v0", "integrations"),
    ("Battletoads & Double Dragon", "BattletoadsDoubleDragon-Nes-v0", "integrations"),
    ("Excitebike", "Excitebike-Nes-v0", "integrations"),
    ("Battle City", "BattleCity-Nes-v0", None),
    ("Double Dragon", "DoubleDragon-Nes-v0", None),   # needs --state default
    ("Ice Climber", "IceClimber-Nes-v0", None),
    ("Gradius", "Gradius-Nes-v0", None),
    ("Battletoads", "Battletoads-Nes-v0", None),
    ("Balloon Fight", "BalloonFight-Nes-v0", None),
]


def ask(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or default


def pick(title: str, options: list[str], default: int = 1) -> int:
    print(f"\n{title}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    while True:
        raw = input(f"choice [{default}]: ").strip()
        if not raw:
            return default - 1
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print("please enter a number from the list")


def checkpoints() -> list[str]:
    out = []
    for d in sorted((ROOT / "runs").glob("*/meta.json")):
        try:
            meta = json.loads(d.read_text())
        except Exception:
            continue
        if "vocab_names" in meta and not d.parent.name.startswith("abl_"):
            out.append(str(d.parent.relative_to(ROOT)))   # a BC checkpoint, not an ablation
    return out


def main() -> None:
    print("=== NES Player ===")
    mode = pick("Mode:", [
        "the model plays (play) — needs a trained checkpoint",
        "instincts explore (explore) — any game, no training needed",
        "instincts with a watching network (explore --observer)",
    ])
    gi = pick("Game:", [g[0] for g in GAMES])
    name, game, integ = GAMES[gi]

    cmd = ["uv", "run", "nes-player"]
    if mode == 0:
        cps = checkpoints() or ["runs/bc_smb_av"]
        ci = pick("Checkpoint:", cps,
                  default=cps.index("runs/bc_smb_av") + 1 if "runs/bc_smb_av" in cps else 1)
        cmd += ["play", "--game", game, "--checkpoint", cps[ci], "--auto-start"]
        cmd += ["--temperature", ask("Sampling temperature", "0.9")]
    else:
        cmd += ["explore", "--game", game]
        if mode == 2:
            cps = checkpoints() or ["runs/bc_smb_av"]
            ci = pick("Observer checkpoint:", cps,
                      default=cps.index("runs/bc_smb_av") + 1 if "runs/bc_smb_av" in cps else 1)
            cmd += ["--observer", cps[ci]]
    if integ:
        cmd += ["--integrations", integ]

    if ask("1920x1080 window? y/n", "y").lower().startswith("y"):
        cmd.append("--hd")
    if ask("Endless episodes? y/n", "y").lower().startswith("y"):
        cmd.append("--loop")
    rec = ask("Record video to a file (blank for none)", "")
    if rec:
        cmd += ["--video-out", rec]
    cmd += ["--window", "--realtime"]

    print("\nCommand:\n  " + " ".join(cmd))
    if ask("Run it? y/n", "y").lower().startswith("y"):
        os.chdir(ROOT)
        sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\ncancelled")
