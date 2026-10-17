"""Play a game yourself from a savestate, and keep what you pressed.

The planner stalls where a human knows the move — Vice's collapsing bridge,
where a hundred and fifty scripted jumps all fell. This is the outside
control for that: a window, the keyboard, and a recording of every frame's
buttons next to the state it started from, so the passage can be replayed,
shown to the agent as a demonstration, or just looked at.

    uv run python scripts/human_play.py ViceProjectDoom-Nes-v0 \
        --load-state runs/vice/room2.state --out runs/vice/human_room2

Keys: arrows = d-pad, Z = B, X = A, Enter = Start, Right Shift = Select,
F5 = save a state here (written next to --out), Esc = stop and save.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

KEYS = {"Up": "UP", "Down": "DOWN", "Left": "LEFT", "Right": "RIGHT",
        "z": "B", "Z": "B", "x": "A", "X": "A", "Return": "START",
        "Shift_R": "SELECT"}


def main() -> int:
    import tkinter as tk

    from PIL import Image, ImageTk

    from nes_player.emulator.stable_retro import StableRetroAdapter

    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--state", default="default")
    ap.add_argument("--load-state", default="")
    ap.add_argument("--out", required=True, help="path prefix for .npz/.state")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--replay", action="store_true",
                    help="replay <out>.json from its start state and check it "
                         "ends where the recording ended")
    args = ap.parse_args()
    if args.replay:
        return replay(args)

    integ = Path(__file__).resolve().parents[1] / "integrations"
    env = StableRetroAdapter(
        args.game, include_debug=True,
        state=None if args.state == "none" else args.state,
        integration_dir=str(integ) if (integ / args.game).exists() else None)
    obs = env.reset(seed=0)
    if args.load_state:
        env.load_state(Path(args.load_state).read_bytes())
        obs = env.step_buttons([frozenset()])
    start = env.save_state()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    held: set[str] = set()
    pressed_log: list[list[str]] = []
    lives: list[int] = []
    saves = 0
    root = tk.Tk()
    root.title(f"{args.game} — Esc to stop")
    h, w = obs.frame_rgb.shape[:2]
    label = tk.Label(root)
    label.pack()

    def down(e):
        nonlocal saves
        if e.keysym == "Escape":
            root.quit()
        elif e.keysym == "F5":
            saves += 1
            Path(f"{out}_{saves}.state").write_bytes(env.save_state())
            print(f"saved {out}_{saves}.state at frame {len(pressed_log)}",
                  flush=True)
        elif e.keysym in KEYS:
            held.add(KEYS[e.keysym])

    def up(e):
        if e.keysym in KEYS:
            held.discard(KEYS[e.keysym])

    root.bind("<KeyPress>", down)
    root.bind("<KeyRelease>", up)
    t0 = time.perf_counter()

    def tick():
        nonlocal obs
        b = frozenset(held)
        obs = env.step_buttons([b])
        pressed_log.append(sorted(b))
        lives.append(int((obs.debug or {}).get("lives", 0) or 0))
        img = Image.fromarray(obs.frame_rgb).resize(
            (w * args.scale, h * args.scale), Image.NEAREST)
        label.img = ImageTk.PhotoImage(img)
        label.configure(image=label.img)
        # hold ~60 fps against the wall clock, not the render time
        due = t0 + len(pressed_log) / 60.0
        root.after(max(1, int((due - time.perf_counter()) * 1000)), tick)

    root.after(1, tick)
    root.mainloop()
    root.destroy()

    Path(f"{out}_start.state").write_bytes(start)
    Path(f"{out}_end.state").write_bytes(env.save_state())
    Path(f"{out}.json").write_text(json.dumps({
        "game": args.game, "load_state": args.load_state,
        "start_state": f"{out}_start.state", "frames": len(pressed_log),
        "buttons": pressed_log}))
    np.save(f"{out}_lives.npy", np.array(lives))
    print(f"{len(pressed_log)} frames -> {out}.json", flush=True)
    return 0


def replay(args) -> int:
    from nes_player.emulator.stable_retro import StableRetroAdapter

    rec = json.loads(Path(f"{args.out}.json").read_text())
    env = StableRetroAdapter(rec["game"], include_debug=True,
                             state=None if args.state == "none" else args.state,
                             integration_dir=None)
    env.reset(seed=0)
    env.load_state(Path(rec["start_state"]).read_bytes())
    for b in rec["buttons"]:
        env.step_buttons([frozenset(b)])
    got = bytes(env._env.get_ram())
    env.load_state(Path(f"{args.out}_end.state").read_bytes())
    same = got == bytes(env._env.get_ram())
    print(f"{len(rec['buttons'])} frames replayed; RAM matches the recording's "
          f"end: {same}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
