"""`nes-player` command-line entry point (spec §19).

The commands live in sibling modules — `play`, `explore`, `train`, `data` — and
the parser in `args`. This file only wires them together.
"""


def main() -> None:
    from nes_player.cli.args import build_parser

    args = build_parser().parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nstopped")
