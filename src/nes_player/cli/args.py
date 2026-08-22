"""Command-line surface of `nes-player` (spec §19).

Kept apart from the command bodies so that `--help` costs nothing: the heavy
imports (torch, cv2, the emulator) live inside the command functions and only
load once a subcommand actually runs.
"""

import argparse

from nes_player.cli.data import cmd_dataset_build, cmd_tas_replay
from nes_player.cli.explore import cmd_explore
from nes_player.cli.play import cmd_play
from nes_player.cli.train import (
    cmd_collect_branches,
    cmd_improve,
    cmd_train_av,
    cmd_train_bc,
    cmd_train_ego,
    cmd_train_idm,
    cmd_train_slots,
    cmd_train_wm,
)

FEEDBACK_HELP = (
    "where the agent learns that something good or bad happened, which is an "
    "INPUT and not telemetry: 'strict' tells it nothing; 'visual' (default) "
    "sees a death on the screen, which is honest and measured to have no false "
    "alarms in 41,400 frames, but reports no score; 'privileged' reads score "
    "and lives from emulator memory, which is what a teacher may do and a "
    "student may not; 'pixel' reads the on-screen panel and does not work yet")
CORE_HELP = "emulation core: fceumm (default), nestopia, quicknes"
STATE_HELP = ("integration start state ('default' for its own); needed by games "
              "whose title screen cannot be passed from power-on")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nes-player")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("tas-replay", help="replay an .fm2 movie in the emulator")
    p.add_argument("--game", required=True,
                   help="integration id, e.g. SuperMarioBros-Nes-v0")
    p.add_argument("--movie", required=True, help="path to the .fm2 file")
    p.add_argument("--integrations", default=None, help="custom integrations directory")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--window", action="store_true", help="show the game window")
    p.add_argument("--realtime", action="store_true", help="hold ~60 fps in the window")
    p.add_argument("--video-out", default=None, help="write an mp4 with the overlay")
    p.set_defaults(func=cmd_tas_replay)

    d = sub.add_parser("dataset-build", help="record an episode from a TAS movie")
    d.add_argument("--game", required=True)
    d.add_argument("--movie", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--integrations", default=None)
    d.add_argument("--max-frames", type=int, default=None)
    d.set_defaults(func=cmd_dataset_build)

    t = sub.add_parser("train-bc", help="behavioural cloning on recorded episodes")
    t.add_argument("--episode", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--audio", action="store_true", help="multimodal model (video + sound)")
    t.add_argument("--init-from", default=None,
                   help="base checkpoint: reuse the body and audio encoder, retrain the heads")
    t.add_argument("--max-episodes", type=int, default=None)
    t.add_argument("--attn", type=float, default=0.0,
                   help="attention-loss weight: pull conv attention onto tracker boxes")
    t.add_argument("--attn-lead", type=int, nargs="+", default=[0],
                   help="frames ahead to aim the attention target, extrapolated "
                        "from tracker velocity. Several values mark all of those "
                        "places at once: '--attn-lead 0 15' means both where an "
                        "object is and where it will be in a quarter second, "
                        "leaving the network to weight them. A single non-zero "
                        "value replaces now with later, which measured worse")
    t.add_argument("--attn-source", choices=("tracker", "oam"), default="tracker",
                   help="where the attention target comes from: 'tracker' infers "
                        "objects from motion in the pixels, 'oam' reads the "
                        "console's own sprite table, which is exact and works in "
                        "any NES game. Supervision only — the policy still plays "
                        "from pixels and sound")
    t.add_argument("--memory", choices=("short", "wide", "long", "epic"), default="short",
                   help="how far back the frame stack reaches: short 67 ms (four "
                        "consecutive frames), wide 0.53 s, long 2.1 s. The wider "
                        "settings space the samples geometrically, so the window "
                        "grows without the network growing")
    t.set_defaults(func=cmd_train_bc)

    g = sub.add_parser("play", help="run a trained BC policy in the emulator")
    g.add_argument("--game", required=True)
    g.add_argument("--checkpoint", required=True)
    g.add_argument("--integrations", default=None)
    g.add_argument("--max-frames", type=int, default=None,
                   help="frames per episode (default 3600; unlimited under --loop)")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--auto-start", action="store_true",
                   help="get through the menus by itself instead of waiting for the model")
    g.add_argument("--repeat", type=int, default=4,
                   help="hold an action for N frames (long jumps; the policy runs at ~15 Hz)")
    g.add_argument("--loop", action="store_true",
                   help="endless episodes, for streaming")
    g.add_argument("--window", action="store_true")
    g.add_argument("--realtime", action="store_true")
    g.add_argument("--video-out", default=None)
    g.add_argument("--hd", action="store_true", help="1920x1080 window and video")
    g.add_argument("--cam", action=argparse.BooleanOptionalAction, default=True,
                   help="Grad-CAM attention overlay (--no-cam to disable)")
    g.add_argument("--jump-hold", type=int, default=None,
                   help="frames to hold A for a jump (default: from the explore knowledge base)")
    g.add_argument("--planner", action="store_true",
                   help="MPC on top of the ego world model (needs a --ghost checkpoint)")
    g.add_argument("--state", default=None, help=STATE_HELP)
    g.add_argument("--core", default=None, help=CORE_HELP)
    g.add_argument("--sound-loc", default="runs/av_smb",
                   help="AV-align checkpoint: ping the sound source on the panel")
    g.add_argument("--feedback", choices=("strict", "visual", "privileged", "pixel"),
                   default="visual", help=FEEDBACK_HELP)
    g.add_argument("--ghost", default="runs/ego_smb4",
                   help="ego world-model checkpoint for the ghost trajectory ('' disables it)")
    g.set_defaults(func=cmd_play)

    x = sub.add_parser("explore", help="instinct policy: calibration and curiosity")
    x.add_argument("--game", default="SuperMarioBros-Nes-v0")
    x.add_argument("--integrations", default=None)
    x.add_argument("--max-frames", type=int, default=None)
    x.add_argument("--loop", action="store_true")
    x.add_argument("--window", action="store_true")
    x.add_argument("--realtime", action="store_true")
    x.add_argument("--video-out", default=None)
    x.add_argument("--hd", action="store_true", help="1920x1080 window and video")
    x.add_argument("--observer", default=None,
                   help="BC checkpoint that watches and shows CAM/probabilities without playing")
    x.add_argument("--state", default=None, help=STATE_HELP)
    x.add_argument("--core", default=None, help=CORE_HELP)
    x.add_argument("--record", default=None,
                   help="directory: write exploration episodes as Zarr datasets")
    x.add_argument("--feedback", choices=("strict", "visual", "privileged", "pixel"),
                   default="visual", help=FEEDBACK_HELP)
    x.set_defaults(func=cmd_explore)

    wm = sub.add_parser("train-wm",
                        help="legacy action-blind world model, kept to reproduce "
                             "the negative result; for the model play --ghost "
                             "actually loads, use train-ego")
    wm.add_argument("--episode", required=True)
    wm.add_argument("--out", required=True)
    wm.add_argument("--epochs", type=int, default=3)
    wm.set_defaults(func=cmd_train_wm)

    eg = sub.add_parser("train-ego",
                        help="ego world model: where the hero goes for a given "
                             "button sequence; this is what --ghost uses")
    eg.add_argument("--episode", required=True)
    eg.add_argument("--out", required=True)
    eg.add_argument("--epochs", type=int, default=4)
    eg.add_argument("--seed", type=int, default=0)
    eg.add_argument("--branches", default=None,
                    help="counterfactual branch file(s) from collect-branches")
    eg.set_defaults(func=cmd_train_ego)

    cf = sub.add_parser("collect-branches",
                        help="replay the same moments with each candidate "
                             "action, so the model can see what a button does")
    cf.add_argument("--out", required=True)
    cf.add_argument("--checkpoint", default="runs/hero_pre_1")
    cf.add_argument("--game", default="SuperMarioBros-Nes-v0")
    cf.add_argument("--state", default="default")
    cf.add_argument("--frames", type=int, default=6000)
    cf.add_argument("--seed", type=int, default=0)
    cf.set_defaults(func=cmd_collect_branches)

    av = sub.add_parser("train-av", help="contrastive sound↔frame (source localisation)")
    av.add_argument("--episode", required=True)
    av.add_argument("--out", required=True)
    av.add_argument("--epochs", type=int, default=2)
    av.set_defaults(func=cmd_train_av)

    idm = sub.add_parser("train-idm",
                         help="inverse dynamics: guess the button from before/after frames")
    idm.add_argument("--episode", required=True)
    idm.add_argument("--out", required=True)
    idm.add_argument("--epochs", type=int, default=3)
    idm.add_argument("--max-episodes", type=int, default=None)
    idm.set_defaults(func=cmd_train_idm)

    sl = sub.add_parser("train-slots", help="neural slots: slot-attention autoencoder")
    sl.add_argument("--episode", required=True)
    sl.add_argument("--out", required=True)
    sl.add_argument("--epochs", type=int, default=3)
    sl.set_defaults(func=cmd_train_slots)

    im = sub.add_parser("improve", help="self-imitation: retrain on the best rollouts")
    im.add_argument("--checkpoint", required=True)
    im.add_argument("--game", default="SuperMarioBros-Nes-v0")
    im.add_argument("--rounds", type=int, default=5)
    im.add_argument("--rollouts", type=int, default=8)
    im.add_argument("--frames", type=int, default=1500)
    im.add_argument("--integrations", default=None)
    im.add_argument("--visual", action="store_true",
                    help="reward from pixels (camera scroll) instead of RAM — spec §12.9")
    im.add_argument("--start-pulses", type=int, default=1,
                    help="START pulse series at the beginning of a rollout (games with intros)")
    im.add_argument("--state", default=None, help=STATE_HELP)
    im.set_defaults(func=cmd_improve)

    return parser
