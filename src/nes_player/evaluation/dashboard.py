"""The 16:9 dashboard: game top left, an NES pad below it, telemetry on the right.

Everything is drawn with cv2 primitives and no external assets, so the panel has
no dependencies beyond what the emulator already needs. The pad lights up as
buttons are pressed, and clicking it feeds a human hint into the game.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from nes_player.emulator.adapter import EmulatorObservation

W, H = 1280, 720
# Control buttons, bottom right of the panel: (x1, y1, x2, y2)
BTN_MUTE = (744, 670, 880, 696)
BTN_RESTART = (900, 670, 1060, 696)
BTN_STOP = (1080, 670, 1240, 696)
# Display toggles, under the attention legend
BTN_TOGGLE_CAM = (492, 634, 622, 660)
BTN_TOGGLE_BOXES = (492, 668, 622, 694)
BG = (24, 20, 18)
PANEL = (38, 33, 30)
FG = (210, 205, 200)
DIM = (95, 90, 85)
ACCENT = (60, 200, 255)   # amber, in BGR
RED = (60, 60, 230)
GREEN = (90, 200, 90)

GAME_X, GAME_Y = 24, 24
GAME_W, GAME_H = 592, 444   # 4:3, as the NES looked on a TV — its pixels are not square
# The real controller is about 2.25:1, 12.5 by 5.5 cm
PAD_W, PAD_H = 450, 200
PAD_X, PAD_Y = 24, 496
INFO_X = 648


def _text(img, s, xy, color=FG, scale=0.5, thick=1):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _rounded_rect(img, x, y, w, h, r, color, thickness=-1):
    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, thickness)
    cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, thickness)
    for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)):
        cv2.circle(img, (cx, cy), r, color, thickness)


def draw_gamepad(img, pressed: frozenset[str], x=PAD_X, y=PAD_Y, w=PAD_W, h=PAD_H) -> None:
    """The controller: body, D-pad, SELECT and START, B and A."""
    def on(b: str) -> bool:
        return b in pressed

    _rounded_rect(img, x, y, w, h, 14, (60, 56, 52))
    _rounded_rect(img, x + 8, y + 8, w - 16, h - 16, 10, (44, 40, 37))
    cv2.rectangle(img, (x + 14, y + 14), (x + w - 14, y + 40), (78, 74, 70), -1)
    _text(img, "NES PLAYER", (x + w // 2 - 52, y + 32), (55, 55, 190), 0.5, 2)

    # D-pad; the pressed direction turns amber
    cx, cy, arm, thick = x + 92, y + 122, 44, 30
    cv2.rectangle(img, (cx - arm, cy - thick // 2), (cx + arm, cy + thick // 2), (20, 20, 20), -1)
    cv2.rectangle(img, (cx - thick // 2, cy - arm), (cx + thick // 2, cy + arm), (20, 20, 20), -1)
    dirs = {
        "LEFT": (cx - arm + 14, cy),
        "RIGHT": (cx + arm - 14, cy),
        "UP": (cx, cy - arm + 14),
        "DOWN": (cx, cy + arm - 14),
    }
    for name, (px, py) in dirs.items():
        color = ACCENT if on(name) else (55, 55, 55)
        cv2.circle(img, (px, py), 10, color, -1)
    cv2.circle(img, (cx, cy), 8, (40, 40, 40), -1)

    # SELECT / START
    for i, name in enumerate(("SELECT", "START")):
        bx = x + w // 2 - 68 + i * 76
        color = ACCENT if on(name) else (25, 25, 25)
        _rounded_rect(img, bx, cy + 8, 56, 18, 8, color)
        _text(img, name, (bx + 2, cy + 44), DIM, 0.38)

    # B / A
    for i, name in enumerate(("B", "A")):
        bx = x + w - 118 + i * 64
        base = (35, 35, 160) if not on(name) else RED
        cv2.circle(img, (bx, cy + 6), 24, (20, 20, 20), -1)
        cv2.circle(img, (bx, cy + 4), 20, base, -1)
        if on(name):
            cv2.circle(img, (bx, cy + 4), 24, ACCENT, 2)
        _text(img, name, (bx - 7, cy + 52), DIM, 0.55, 2)


def _pad_hits(x=PAD_X, y=PAD_Y, w=PAD_W, h=PAD_H) -> dict[str, tuple[int, int, int, int]]:
    """Clickable regions of the drawn pad, in the same geometry as draw_gamepad."""
    cx, cy, arm = x + 92, y + 122, 44
    hits = {
        "LEFT": (cx - arm, cy - 15, cx - 10, cy + 15),
        "RIGHT": (cx + 10, cy - 15, cx + arm, cy + 15),
        "UP": (cx - 15, cy - arm, cx + 15, cy - 10),
        "DOWN": (cx - 15, cy + 10, cx + 15, cy + arm),
    }
    for i, name in enumerate(("SELECT", "START")):
        bx = x + w // 2 - 68 + i * 76
        hits[name] = (bx, cy + 8, bx + 56, cy + 26)
    for i, name in enumerate(("B", "A")):
        bx = x + w - 118 + i * 64
        hits[name] = (bx - 24, cy - 20, bx + 24, cy + 28)
    return hits


PAD_HITS = _pad_hits()

VERDICT_COLORS = {"danger": (60, 60, 230), "reward": (90, 200, 90), "unknown": (200, 200, 80)}


def _draw_slots(frame: np.ndarray, slots: list, src_shape, verdicts: dict | None = None) -> None:
    """Object boxes and velocity vectors over the scaled-up game image."""
    sy, sx = frame.shape[0] / src_shape[0], frame.shape[1] / src_shape[1]
    for s in slots:
        x, y, w, h = s.bbox
        p1 = (int(x * sx), int(y * sy))
        p2 = (int((x + w) * sx), int((y + h) * sy))
        controlled = s.ctrl_prob > 0.7
        verdict = (verdicts or {}).get(s.slot_id, "unknown")
        color = ACCENT if controlled else VERDICT_COLORS[verdict]
        if s.missed > 4:   # a ghost: stalled object, box held at its last position
            color = tuple(c // 2 for c in color)
        cv2.rectangle(frame, p1, p2, color, 2 if controlled else 1)
        c = (int(s.cx * sx), int(s.cy * sy))
        tip = (int((s.cx + s.vx * 6) * sx), int((s.cy + s.vy * 6) * sy))
        if abs(s.vx) + abs(s.vy) > 0.3:
            cv2.arrowedLine(frame, c, tip, (120, 255, 120), 2, tipLength=0.35)
        label = f"#{s.slot_id}"
        if controlled:
            label += f" ctrl {s.ctrl_prob:.2f}"
        elif verdict != "unknown":
            label += f" {verdict}"
        cv2.putText(frame, label, (p1[0], max(12, p1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


@dataclass
class Dashboard:
    """Assembles one dashboard frame. `thoughts` is the running commentary."""

    title: str = "NES Player"
    fps_smooth: float = field(default=0.0)

    def render(
        self,
        obs: EmulatorObservation,
        pressed: frozenset[str],
        info: dict[str, str] | None = None,
        thoughts: list[str] | None = None,
        action_probs: list[tuple[str, float]] | None = None,
        slots: list | None = None,
        verdicts: dict | None = None,
        heatmap: np.ndarray | None = None,
        entropy_hist: list[float] | None = None,
        toggles: dict | None = None,          # {'cam': bool, 'boxes': bool}
        features: np.ndarray | None = None,   # (8, h, w) conv-channel activations
        gallery: list | None = None,  # [(proto16x16, verdict, cluster_id, seen)]
        audio_events: list | None = None,  # [(cluster_id, frames_ago, heard_total)]
        ghost: list | None = None,        # [(x, y)] predicted hero trajectory
        sound_pings: list | None = None,  # [(x01, y01, age)] predicted sound sources
    ) -> np.ndarray:
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:] = BG

        # The game image
        frame = cv2.cvtColor(obs.frame_rgb, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (GAME_W, GAME_H), interpolation=cv2.INTER_NEAREST)
        if heatmap is not None:   # Grad-CAM: where the network is looking
            hm = cv2.resize(heatmap, (GAME_W, GAME_H), interpolation=cv2.INTER_LINEAR)
            # Normalise by percentile so only the top ~30% of attention shows.
            # On an uneventful frame the raw map is flat and floods the screen.
            lo, hi = np.percentile(hm, 70), np.percentile(hm, 99)
            hm = np.clip((hm - lo) / max(hi - lo, 1e-6), 0, 1)
            colored = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            mask = (hm ** 1.5)[..., None] * 0.5
            frame = (frame * (1 - mask) + colored * mask).astype(np.uint8)
        if slots:
            _draw_slots(frame, slots, obs.frame_rgb.shape, verdicts)
        if ghost:   # the world model's guess at where the hero is heading
            sy, sx = frame.shape[0] / obs.frame_rgb.shape[0], \
                frame.shape[1] / obs.frame_rgb.shape[1]
            for k, (gx_, gy_) in enumerate(ghost):
                alpha = 1.0 - k / (len(ghost) + 2)
                cv2.circle(frame, (int(gx_ * sx), int(gy_ * sy)),
                           max(2, 5 - k // 4), (255, 255, int(180 * alpha)), -1)
        if sound_pings:   # an expanding ring at the predicted source of a sound
            for px, py, age in sound_pings:
                c = (int(px * frame.shape[1]), int(py * frame.shape[0]))
                r = 6 + age
                fade = max(0, 1 - age / 45)
                col = (int(80 + 175 * fade), int(220 * fade), int(80 * fade))
                cv2.circle(frame, c, r, col, 2)
                if age < 20:
                    cv2.putText(frame, "sound", (c[0] - 20, c[1] - r - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        img[GAME_Y : GAME_Y + GAME_H, GAME_X : GAME_X + GAME_W] = frame
        cv2.rectangle(img, (GAME_X - 2, GAME_Y - 2),
                      (GAME_X + GAME_W + 1, GAME_Y + GAME_H + 1), (70, 65, 60), 2)

        draw_gamepad(img, pressed)

        # Attention legend, to the right of the pad
        if heatmap is not None:
            lx, ly = BTN_TOGGLE_CAM[0], PAD_Y + 36
            lw = BTN_TOGGLE_CAM[2] - BTN_TOGGLE_CAM[0]
            _text(img, "model attention", (lx, ly - 12), FG, 0.45)
            grad = np.linspace(0.35, 1, lw, dtype=np.float32)[None, :]
            bar = cv2.applyColorMap((np.repeat(grad, 12, 0) * 255).astype(np.uint8),
                                    cv2.COLORMAP_JET)
            img[ly : ly + 12, lx : lx + lw] = bar
            _text(img, "ignored", (lx, ly + 30), DIM, 0.4)
            _text(img, "in focus", (lx + lw - 54, ly + 30), DIM, 0.4)
            _text(img, "hot = pixels that drive", (lx, ly + 54), DIM, 0.38)
            _text(img, "the current decision", (lx, ly + 70), DIM, 0.38)

        # Display toggles: click, or the c and b keys
        if toggles is not None:
            for (x1, y1, x2, y2), key, label in (
                (BTN_TOGGLE_CAM, "cam", "CAM"),
                (BTN_TOGGLE_BOXES, "boxes", "BOXES"),
            ):
                on = toggles.get(key, True)
                color = (70, 140, 190) if on else (60, 56, 52)
                _rounded_rect(img, x1, y1, x2 - x1, y2 - y1, 6, color)
                state = "ON" if on else "OFF"
                _text(img, f"{label}: {state}", (x1 + 12, y1 + 18),
                      (240, 240, 240) if on else DIM, 0.42, 1)

        # Right-hand panel
        cv2.rectangle(img, (INFO_X, 24), (W - 24, H - 24), PANEL, -1)
        _text(img, self.title, (INFO_X + 20, 56), FG, 0.7, 2)
        cv2.line(img, (INFO_X + 20, 70), (W - 44, 70), (70, 65, 60), 1)

        # Column 1: info. Column 2: action probabilities
        ty = 100
        base_info = {"frame": str(obs.frame_index)}
        for k, v in (base_info | (info or {})).items():
            _text(img, k, (INFO_X + 20, ty), DIM, 0.5)
            _text(img, v, (INFO_X + 130, ty), FG, 0.5)
            ty += 26

        py = 100
        if action_probs:
            cx0 = INFO_X + 330
            _text(img, "action probabilities", (cx0, py - 14), DIM, 0.5)
            for name, p in action_probs[:11]:
                bar_w = int(p * 130)
                cv2.rectangle(img, (cx0 + 95, py - 10), (cx0 + 95 + bar_w, py + 2),
                              ACCENT if p == action_probs[0][1] else (110, 100, 90), -1)
                _text(img, name, (cx0, py), FG, 0.42)
                _text(img, f"{p:.2f}", (cx0 + 232, py), DIM, 0.42)
                py += 22

        # Uncertainty plot, full panel width, under both columns
        ty = max(ty, py) + 8
        if entropy_hist:
            _text(img, "uncertainty", (INFO_X + 20, ty), DIM, 0.5)
            gx, gy, gw, gh = INFO_X + 20, ty + 8, W - 44 - (INFO_X + 20), 40
            cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), (50, 46, 42), -1)
            pts = entropy_hist[-240:]
            if len(pts) > 1:
                xs = np.linspace(gx, gx + gw, len(pts)).astype(int)
                ys = (gy + gh - np.clip(np.asarray(pts), 0, 1) * (gh - 4) - 2).astype(int)
                cv2.polylines(img, [np.stack([xs, ys], axis=1)], False, ACCENT, 1, cv2.LINE_AA)
            _text(img, f"{pts[-1]:.2f}", (gx + gw - 48, gy + 16), FG, 0.45)
            ty = gy + gh + 24

        col2_anchor = ty   # column 2 is pinned here, independent of object count

        if slots:
            _text(img, "objects", (INFO_X + 20, ty), DIM, 0.5)
            ty += 22
            for s in sorted(slots, key=lambda s: -s.ctrl_prob)[:3]:
                mark = ">" if s.ctrl_prob > 0.7 else " "
                _text(img, f"{mark}#{s.slot_id} v=({s.vx:+.1f},{s.vy:+.1f}) "
                           f"ctrl={s.ctrl_prob:.2f}",
                      (INFO_X + 20, ty), FG if s.ctrl_prob > 0.7 else DIM, 0.45)
                ty += 20

        # Bottom of column 2: live conv activations and the sprite-memory
        # gallery, pinned so they do not jump as objects come and go
        cx0 = INFO_X + 330
        cy = col2_anchor
        if features is not None:
            _text(img, "conv features (live)", (cx0, cy - 6), DIM, 0.45)
            for k, ch in enumerate(features[:4]):
                tx, ty2 = cx0 + k * 66, cy + 4
                if ty2 + 46 > H - 28:
                    break
                tile = cv2.applyColorMap((ch * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                tile = cv2.resize(tile, (60, 46), interpolation=cv2.INTER_NEAREST)
                img[ty2 : ty2 + 46, tx : tx + 60] = tile
            cy += 4 + 52 + 24
        if gallery:
            _text(img, "sprite memory", (cx0, cy - 6), DIM, 0.45)
            for k, (proto, verdict, _cid, _seen) in enumerate(gallery[:4]):
                tx, ty2 = cx0 + k * 66, cy + 4
                if ty2 + 46 > BTN_STOP[1] - 8:   # do not overrun the buttons
                    break
                p = np.clip(proto, 0, 255).astype(np.uint8)
                tile = cv2.cvtColor(cv2.resize(p, (46, 46),
                                               interpolation=cv2.INTER_NEAREST),
                                    cv2.COLOR_GRAY2BGR)
                img[ty2 : ty2 + 46, tx : tx + 46] = tile
                cv2.rectangle(img, (tx - 1, ty2 - 1), (tx + 46, ty2 + 46),
                              VERDICT_COLORS.get(verdict, DIM), 2)
            cy += 4 + 46 + 26

        # Sound events: recent onsets, coloured by cluster id, fading with age
        if audio_events is not None:
            _text(img, "audio events", (cx0, cy - 6), DIM, 0.45)
            for k, (eid, ago, verdict) in enumerate(audio_events[:7]):
                tx, ty2 = cx0 + k * 36, cy + 2
                if ty2 + 24 > BTN_STOP[1] - 8:
                    break
                hue = (eid * 47) % 180
                fade = max(0.25, 1.0 - ago / 120)
                hsv = np.uint8([[[hue, 200, int(230 * fade)]]])
                color = tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
                _rounded_rect(img, tx, ty2, 30, 24, 6, color)
                if verdict != "unknown":   # meaning established: outline it
                    cv2.rectangle(img, (tx - 1, ty2 - 1), (tx + 31, ty2 + 25),
                                  VERDICT_COLORS[verdict], 2)
                _text(img, str(eid), (tx + 8, ty2 + 17), (20, 20, 20), 0.45, 2)

        # Lower half of the panel is the commentary. Width is limited to column
        # 1 and long lines wrap, so they never overrun the sprites on the right.
        if thoughts:
            import textwrap

            th_y = max(380, ty + 24)
            _text(img, "thoughts", (INFO_X + 20, th_y), DIM, 0.5)
            th_y += 24
            for line in thoughts[-11:]:
                for sub in textwrap.wrap(line, 38)[:2]:
                    if th_y > BTN_STOP[1] - 14:
                        break
                    _text(img, sub, (INFO_X + 20, th_y), GREEN, 0.45)
                    th_y += 22

        # Buttons: click, or the m, r and q keys
        for (x1, y1, x2, y2), label, color in (
            (BTN_MUTE, "MUTE (m)", (110, 100, 90)),
            (BTN_RESTART, "RESTART (r)", (70, 140, 190)),
            (BTN_STOP, "STOP (q)", (60, 60, 160)),
        ):
            _rounded_rect(img, x1, y1, x2 - x1, y2 - y1, 6, color)
            _text(img, label, (x1 + 16, y1 + 18), (240, 240, 240), 0.48, 1)
        return img
