"""A graphical launcher: click to choose, and the menu comes back when the game
exits. Drawn with the same cv2 primitives as the dashboard, so it adds no
dependency of its own."""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
W, H = 900, 640
BG, PANEL, FG, DIM = (24, 20, 18), (38, 33, 30), (210, 205, 200), (110, 105, 100)
ACCENT = (60, 200, 255)
WIN = "NES Player Launcher"

GAMES = [
    ("Super Mario Bros.", "SuperMarioBros-Nes-v0", None, None),
    ("Contra (US)", "ContraU-Nes-v0", "integrations", None),
    ("Battle City", "BattleCity-Nes-v0", None, None),
    ("Double Dragon", "DoubleDragon-Nes-v0", None, "default"),   # title is unpassable
    ("Ice Climber", "IceClimber-Nes-v0", None, None),
    ("Gradius", "Gradius-Nes-v0", None, None),
    ("Battletoads", "Battletoads-Nes-v0", None, None),
    ("Balloon Fight", "BalloonFight-Nes-v0", None, None),
    ("BT & Double Dragon", "BattletoadsDoubleDragon-Nes-v0", "integrations", None),
    ("Excitebike", "Excitebike-Nes-v0", "integrations", None),
]
MODES = ["model plays (play)", "instincts explore", "instincts + observer"]

# Checkpoint picked automatically per game; the observer gets the broadest base
PREFERRED = {
    "SuperMarioBros-Nes-v0": ["runs/bc_smb_attn3", "runs/bc_smb_av", "runs/bc_smb_si"],
    "ContraU-Nes-v0": ["runs/bc_contra_attn", "runs/bc_contra_av"],   # trained on (J)
    "ContraJ-Nes-v0": ["runs/bc_contra_attn", "runs/bc_contra_av"],
    "Gradius-Nes-v0": ["runs/bc_gradius_attn"],
    "BalloonFight-Nes-v0": ["runs/bc_balloonfight_attn"],
    "Battletoads-Nes-v0": ["runs/bc_battletoads_attn"],
    "DoubleDragon-Nes-v0": ["runs/bc_dd_attn2", "runs/bc_dd_attn"],
}
OBSERVER_DEFAULT = ["runs/bc_base41_attn1", "runs/bc_base41_av", "runs/bc_base_av"]


def checkpoints() -> list[str]:
    out = []
    for m in sorted((ROOT / "runs").glob("*/meta.json")):
        try:
            meta = json.loads(m.read_text())
        except Exception:
            continue
        name = m.parent.name
        if "vocab_names" in meta and not name.startswith(("abl_", "duel_", "big_", "ice_")):
            out.append(f"runs/{name}")
    return out or ["runs/bc_smb_av"]


class Menu:
    def __init__(self):
        self.mode = 0
        self.game = 0
        self.cps = checkpoints()
        self.cp = self.cps.index("runs/bc_smb_av") if "runs/bc_smb_av" in self.cps else 0
        self.hd = True
        self.loop = True
        self.record = False   # write an mp4 with sound into recordings/
        self.video_out: Path | None = None
        self.launch = False
        self.quit = False
        self._hits: list[tuple[tuple[int, int, int, int], callable]] = []

    def _text(self, img, s, xy, color=FG, scale=0.5, thick=1):
        cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def autopick(self):
        _, game, *_ = GAMES[self.game]
        wants = (OBSERVER_DEFAULT if self.mode == 2
                 else PREFERRED.get(game, OBSERVER_DEFAULT))
        for w in wants:
            if w in self.cps:
                self.cp = self.cps.index(w)
                return

    def _item(self, img, x, y, w, text, active, cb):
        color = ACCENT if active else PANEL
        tcol = (20, 20, 20) if active else FG
        cv2.rectangle(img, (x, y), (x + w, y + 26), color, -1)
        self._text(img, text, (x + 8, y + 18), tcol, 0.45)
        self._hits.append(((x, y, x + w, y + 26), cb))

    def render(self) -> np.ndarray:
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = BG
        self._hits = []
        self._text(img, "NES PLAYER", (24, 40), FG, 0.9, 2)
        self._text(img, "mode", (24, 80), DIM)
        for i, m in enumerate(MODES):
            self._item(img, 24, 92 + i * 32, 260, m, self.mode == i,
                       lambda i=i: (setattr(self, "mode", i), self.autopick()))
        self._text(img, "game", (24, 216), DIM)
        for i, (name, *_) in enumerate(GAMES):
            self._item(img, 24, 228 + i * 32, 260, name, self.game == i,
                       lambda i=i: (setattr(self, "game", i), self.autopick()))
        self._text(img, "checkpoint (play/observer)", (320, 80), DIM)
        for i, cp in enumerate(self.cps[:12]):
            self._item(img, 320, 92 + i * 32, 280, cp, self.cp == i,
                       lambda i=i: setattr(self, "cp", i))
        self._text(img, "options", (640, 80), DIM)
        self._item(img, 640, 92, 220, f"HD 1080: {'ON' if self.hd else 'OFF'}",
                   self.hd, lambda: setattr(self, "hd", not self.hd))
        self._item(img, 640, 124, 220, f"loop: {'ON' if self.loop else 'OFF'}",
                   self.loop, lambda: setattr(self, "loop", not self.loop))
        self._item(img, 640, 156, 220, f"record video: {'ON' if self.record else 'OFF'}",
                   self.record, lambda: setattr(self, "record", not self.record))
        if self.record:
            self._text(img, "-> recordings/<game>_<time>.mp4", (640, 200), DIM, 0.38)
        # buttons
        cv2.rectangle(img, (640, H - 90), (860, H - 50), (70, 140, 190), -1)
        self._text(img, "LAUNCH (enter)", (668, H - 64), (240, 240, 240), 0.55, 2)
        self._hits.append(((640, H - 90, 860, H - 50), lambda: setattr(self, "launch", True)))
        cv2.rectangle(img, (640, H - 44, ), (860, H - 12), (60, 60, 160), -1)
        self._text(img, "QUIT (q)", (700, H - 22), (240, 240, 240), 0.55, 2)
        self._hits.append(((640, H - 44, 860, H - 12), lambda: setattr(self, "quit", True)))
        self._text(img, "click to select; window will return here after the game",
                   (24, H - 20), DIM, 0.42)
        return img

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for (x1, y1, x2, y2), cb in self._hits:
            if x1 <= x <= x2 and y1 <= y <= y2:
                cb()

    def command(self) -> list[str]:
        name, game, integ, state = GAMES[self.game]
        cmd = ["uv", "run", "nes-player"]
        if self.mode == 0:
            cmd += ["play", "--game", game, "--checkpoint", self.cps[self.cp],
                    "--auto-start", "--temperature", "0.9"]
            if game == "SuperMarioBros-Nes-v0":
                cmd.append("--planner")   # with ego v4 the planner goes far further
        else:
            cmd += ["explore", "--game", game]
            if self.mode == 2:
                cmd += ["--observer", self.cps[self.cp]]
        if integ:
            cmd += ["--integrations", integ]
        if state:   # games whose title screen cannot be passed from power-on
            cmd += ["--state", state]
        if self.hd:
            cmd.append("--hd")
        if self.loop:
            cmd.append("--loop")
        self.video_out = None
        if self.record:
            rec = ROOT / "recordings"
            rec.mkdir(exist_ok=True)
            self.video_out = rec / f"{game}_{time.strftime('%H%M%S')}.rec.mp4"
            cmd += ["--video-out", str(self.video_out)]
        cmd += ["--window", "--realtime"]
        return cmd

    def mux_recording(self) -> None:
        """Mux the viewer's video and wav into one h264+aac file."""
        v = self.video_out
        if not v or not v.exists():
            return
        wav = Path(str(v) + ".wav")
        final = v.with_name(v.name.replace(".rec", ""))
        if shutil.which("ffmpeg") and wav.exists():
            # The video is already h264 from the viewer, so just attach the
            # audio; a legacy mp4v file is re-encoded instead.
            ok = subprocess.call(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(v), "-i", str(wav),
                 "-c:v", "copy", "-c:a", "aac", str(final)]) == 0
            if not ok:
                ok = subprocess.call(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", str(v), "-i", str(wav),
                     "-c:v", "h264_videotoolbox", "-b:v", "6M", "-c:a", "aac",
                     str(final)]) == 0
            if ok:
                v.unlink()
                wav.unlink()
                print(f"recorded: {final}", flush=True)
                return
        v.rename(final)   # no ffmpeg or no audio: leave it as it is
        print(f"recorded (no audio mux): {final}", flush=True)


def main() -> None:
    menu = Menu()
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, menu.on_mouse)
    while True:
        cv2.imshow(WIN, menu.render())
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27) or menu.quit:
            break
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (13, 10) or menu.launch:
            menu.launch = False
            cmd = menu.command()
            print("launch:", " ".join(cmd), flush=True)
            cv2.destroyWindow(WIN)
            subprocess.call(cmd, cwd=ROOT)   # wait for the game to exit
            menu.mux_recording()
            cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)   # then bring the menu back
            cv2.setMouseCallback(WIN, menu.on_mouse)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
