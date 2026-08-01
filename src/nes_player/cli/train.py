"""Training commands: one thin wrapper per learning module (spec §19).

Every wrapper does the same three things — import lazily so that `--help` stays
instant, call the trainer, print the one number that says whether it worked.
The baseline is always printed next to the score: an accuracy means nothing
without the majority-class or chance level beside it.
"""

import argparse


def cmd_train_bc(args: argparse.Namespace) -> None:
    """Behavioural cloning on recorded episodes."""
    from nes_player.policy.bc import train_bc

    meta = train_bc(args.episode, args.out, epochs=args.epochs, use_audio=args.audio,
                    init_from=args.init_from, max_episodes=args.max_episodes,
                    attn=args.attn)
    last = meta["history"][-1]
    print(f"modality={meta['modality']} val_acc={last['val_acc']:.3f} "
          f"majority_baseline={meta['val_majority_baseline']:.3f}")


def cmd_train_av(args: argparse.Namespace) -> None:
    """Contrastive audio↔frame alignment (spec §12.4)."""
    from nes_player.perception.av_align import train_av_align

    meta = train_av_align(args.episode, args.out, epochs=args.epochs)
    last = meta["history"][-1]
    print(f"val_top1={last['val_top1']:.3f} chance={last['chance']:.3f}")


def cmd_train_idm(args: argparse.Namespace) -> None:
    """Inverse dynamics: recover the button pressed between two frames (spec §12.2)."""
    from nes_player.policy.idm import train_idm

    meta = train_idm(args.episode, args.out, epochs=args.epochs,
                     max_episodes=args.max_episodes)
    last = meta["history"][-1]
    print(f"val_acc={last['val_acc']:.3f} majority={meta['val_majority_baseline']:.3f}")


def cmd_train_slots(args: argparse.Namespace) -> None:
    """Slot-attention autoencoder — a documented negative result, see docs/experiments.md."""
    from nes_player.perception.slots import train_slots

    meta = train_slots(args.episode, args.out, epochs=args.epochs)
    print(f"recon_mse={meta['history'][-1]['recon_mse']:.5f}")


def cmd_train_wm(args: argparse.Namespace) -> None:
    """Latent world model. `action_advantage` above 1 means actions actually help."""
    from nes_player.world_model.model import train_wm

    meta = train_wm(args.episode, args.out, epochs=args.epochs)
    print(f"action_advantage={meta['action_advantage']:.3f} "
          f"(must be > 1: predictions with actions beat predictions without)")


def cmd_improve(args: argparse.Namespace) -> None:
    """Self-imitation: play, keep the best rollouts, retrain on them."""
    from nes_player.policy.improve import self_imitation

    self_imitation(args.checkpoint, game=args.game, rounds=args.rounds,
                   rollouts_per_round=args.rollouts, frames=args.frames,
                   visual=args.visual, integrations=args.integrations,
                   start_pulses=args.start_pulses)
