"""Top-level ``mhcflurry`` CLI dispatcher.

Subcommands are listed in ``_SUBCOMMANDS``; each subcommand module
exposes ``register_subparser(subparser)`` and ``run(args)``. New
subcommands plug in by extending ``_SUBCOMMANDS`` only.
"""
from __future__ import annotations

import argparse
import sys

from . import compare_models, plot_model_comparison


_SUBCOMMANDS = {
    "compare-models": compare_models,
    "plot-model-comparison": plot_model_comparison,
}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="mhcflurry",
        description=(
            "MHCflurry. Use 'mhcflurry <subcommand> --help' for details on "
            "any subcommand."
        ),
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)
    for name, module in _SUBCOMMANDS.items():
        first_line = (module.__doc__ or "").strip().splitlines()
        help_text = first_line[0] if first_line else None
        module.register_subparser(sub.add_parser(name, help=help_text))
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    module = _SUBCOMMANDS[args.subcommand]
    return module.run(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
