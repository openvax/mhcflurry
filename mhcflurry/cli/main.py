# PYTHON_ARGCOMPLETE_OK
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Top-level ``mhcflurry`` CLI dispatcher.

Every subcommand is registered as ``(module path, entry attr, one-line
help)`` and lazy-imported on dispatch, so ``mhcflurry --help`` and
``mhcflurry <subcommand> --help`` only pull in the modules they actually
need. This keeps the torch-import cost off the top-level help path.

Two flavors of subcommand:

* **New under the parent**: ``train``, ``eval``, ``compare-models``,
  ``plot-model-comparison``, ``paper-figures``.
  Each module exposes ``run_argv(argv)`` which does its own argparse.
* **Historical mhcflurry-* commands**: ``predict``, ``predict-scan``,
  ``downloads``, ``calibrate-percentile-ranks``, the ``class1-train-*`` /
  ``class1-select-*`` family, ``pseudosequences``. Each is wrapped by
  invoking the command module's existing ``run(argv)`` (or ``main(argv)``
  for ``pseudosequences``) on the post-subcommand argv. All
  ``mhcflurry-*`` console_scripts in ``setup.py`` remain installed as
  compat shims pointing at the same underlying entry functions.

Shell tab completion (argcomplete) completes subcommand names from
:func:`build_parser`. For the subcommands in ``FLAG_COMPLETION_SUBCOMMANDS``
it also imports that one subcommand's module and attaches its parser, so the
subcommand's own flags complete; see :func:`build_completion_parser`.
"""
from __future__ import annotations

import argparse
import importlib
import sys

import argcomplete

from ..version import __version__
from .completion import completing_subcommand, is_completing
from .help import style_help


# Every subcommand is ``(module_path, entry_attr, one-line help)``. The
# entry function must accept a single ``argv`` argument; it owns its own
# arg parsing.
_SUBCOMMANDS = {
    "train": (
        "mhcflurry.cli.train_command", "run_argv",
        "Release-training workflows."),
    "eval": (
        "mhcflurry.cli.eval_command", "run_argv",
        "Evaluation, model comparison, and paper-figure workflows."),
    "compare-models": (
        "mhcflurry.cli.compare_models", "run_argv",
        "Compare two model ensembles on data_evaluation."),
    "plot-model-comparison": (
        "mhcflurry.cli.plot_model_comparison", "run_argv",
        "Render plots from a compare-models output directory."),
    "paper-figures": (
        "mhcflurry.cli.paper_figures", "run_argv",
        "Render paper-style figures from retraining artifacts."),
    "predict": (
        "mhcflurry.cli.predict_command", "run",
        "Predict affinity, processing and presentation for peptides and MHC alleles."),
    "predict-scan": (
        "mhcflurry.cli.predict_scan_command", "run",
        "Scan protein sequences for predicted MHC I ligands."),
    "downloads": (
        "mhcflurry.cli.downloads_command", "run",
        "Fetch, inspect, resolve data + model downloads."),
    "calibrate-percentile-ranks": (
        "mhcflurry.cli.calibrate_percentile_ranks_command", "run",
        "Calibrate percentile ranks on an existing predictor."),
    "class1-train-allele-specific-models": (
        "mhcflurry.cli.train_allele_specific_models_command", "run",
        "Train Class I allele-specific affinity models."),
    "class1-select-allele-specific-models": (
        "mhcflurry.cli.select_allele_specific_models_command", "run",
        "Select Class I allele-specific models from a candidate pool."),
    "class1-train-pan-allele-models": (
        "mhcflurry.cli.train_pan_allele_models_command", "run",
        "Train Class I pan-allele affinity models."),
    "class1-select-pan-allele-models": (
        "mhcflurry.cli.select_pan_allele_models_command", "run",
        "Select Class I pan-allele models from a candidate pool."),
    "class1-train-processing-models": (
        "mhcflurry.cli.train_processing_models_command", "run",
        "Train Class I antigen-processing models."),
    "class1-select-processing-models": (
        "mhcflurry.cli.select_processing_models_command", "run",
        "Select Class I processing models from a candidate pool."),
    "class1-train-presentation-models": (
        "mhcflurry.cli.train_presentation_models_command", "run",
        "Train the Class I presentation predictor (affinity + processing)."),
    "class1-generate-training-hyperparameters": (
        "mhcflurry.cli.generate_training_hyperparameters", "run_argv",
        "Generate release-training hyperparameter grids."),
    "class1-reassign-mass-spec-training-data": (
        "mhcflurry.cli.reassign_mass_spec_training_data", "run_argv",
        "Reassign mass-spec rows in Class I affinity training data."),
    "pseudosequences": (
        "mhcflurry.pseudosequences", "main",
        "Pseudosequence CSV registry helper (filename / path / list)."),
}


# Display grouping for the friendly help screen. Each subcommand listed
# here must appear in ``_SUBCOMMANDS``; the closing assertion catches
# drift between the two.
_HELP_GROUPS = (
    ("Prediction", (
        "predict", "predict-scan",
    )),
    ("Data management", (
        "downloads",
    )),
    ("Class I training", (
        "class1-train-allele-specific-models",
        "class1-train-pan-allele-models",
        "class1-train-processing-models",
        "class1-train-presentation-models",
    )),
    ("Class I selection", (
        "class1-select-allele-specific-models",
        "class1-select-pan-allele-models",
        "class1-select-processing-models",
    )),
    ("Calibration", (
        "calibrate-percentile-ranks",
    )),
    ("Evaluation and figures", (
        "eval",
    )),
    ("Release training", (
        "train",
    )),
    ("Helpers", (
        "class1-generate-training-hyperparameters",
        "class1-reassign-mass-spec-training-data",
        "pseudosequences",
    )),
    ("Shortcuts for eval subcommands", (
        "compare-models", "plot-model-comparison", "paper-figures",
    )),
)


def _check_help_groups():
    """Fail-fast at import time if _HELP_GROUPS drifts from _SUBCOMMANDS.

    Missing or unknown command names prevent importing this dispatcher.
    Historical console scripts import their command modules directly and
    do not run this check. Duplicate grouping is checked by CLI tests.
    """
    grouped = {name for _, names in _HELP_GROUPS for name in names}
    missing = set(_SUBCOMMANDS) - grouped
    extra = grouped - set(_SUBCOMMANDS)
    # Raise explicitly rather than ``assert`` so the guard survives ``python -O``
    # (which strips assertions): drift must never ship silently.
    if missing:
        raise RuntimeError(
            "subcommands missing from _HELP_GROUPS: %s" % missing)
    if extra:
        raise RuntimeError(
            "_HELP_GROUPS lists unknown subcommands: %s" % extra)


_check_help_groups()


# Subcommands whose flags complete on Tab. Their modules import in tens of
# milliseconds, without torch, numpy or pandas. The other subcommands complete
# by name only: the training, selection, calibration and evaluation modules
# import torch or pandas at module level (seconds per Tab), and most ``train``
# and ``eval`` subcommands shell out to scripts/ in a source checkout.
FLAG_COMPLETION_SUBCOMMANDS = (
    "predict",
    "predict-scan",
    "downloads",
    "pseudosequences",
)


def build_parser(subcommand_parsers=None):
    """Return the top-level parser used by ``--help`` and tooling.

    Subparsers are registered by name only — no per-subcommand arguments
    are added here. The legacy module owns its own ``--help`` output and
    is invoked lazily by :func:`main`. Keeps this function torch-free.

    Parameters
    ----------
    subcommand_parsers : dict, optional
        Maps a subcommand name to that subcommand's own parser. Its
        arguments are copied into the subcommand's entry (as
        ``argparse`` ``parents``), as shell completion needs. Names not
        given are registered by name only.

    Returns
    -------
    argparse.ArgumentParser
    """
    subcommand_parsers = subcommand_parsers or {}
    parser = argparse.ArgumentParser(
        prog="mhcflurry",
        description=(
            "MHCflurry. Use 'mhcflurry <subcommand> --help' for details on "
            "any subcommand."
        ),
    )
    # ``--version`` / ``-V`` is handled by main()'s early-exit before
    # parse_args runs; no argparse action needed here.
    sub = parser.add_subparsers(dest="subcommand", required=True)
    for name, (_module, _entry, help_text) in _SUBCOMMANDS.items():
        parents = [subcommand_parsers[name]] if name in subcommand_parsers else []
        sub.add_parser(name, help=help_text, add_help=False, parents=parents)
    return parser


def load_subcommand_parser(name):
    """Import one subcommand's module and return its argument parser.

    Parameters
    ----------
    name : str
        Subcommand name, a key of the subcommand registry.

    Returns
    -------
    argparse.ArgumentParser
        The module's module-level ``parser`` when it has one (the
        historical command modules do), otherwise ``make_parser()``'s
        result.
    """
    module = importlib.import_module(_SUBCOMMANDS[name][0])
    parser = getattr(module, "parser", None)
    if parser is None:
        parser = module.make_parser()
    return parser


def build_completion_parser(environ=None):
    """Return the parser that answers a shell completion request.

    Parameters
    ----------
    environ : mapping, optional
        Environment holding argcomplete's variables. Defaults to
        ``os.environ``.

    Returns
    -------
    argparse.ArgumentParser
        :func:`build_parser`, plus the parser of the subcommand on the
        completion line when it is in ``FLAG_COMPLETION_SUBCOMMANDS``. Only
        that subcommand's module is imported.
    """
    name = completing_subcommand(_SUBCOMMANDS, environ)
    if name in FLAG_COMPLETION_SUBCOMMANDS:
        return build_parser({name: load_subcommand_parser(name)})
    return build_parser()


def format_help():
    """Build the friendly help screen shown for bare ``mhcflurry`` and ``--help``.

    Replaces argparse's default flat-listing layout with grouped sections,
    a short usage banner, and a couple of example invocations. Pure-string
    formatting; no argparse internals involved.
    """
    lines = [
        "usage: mhcflurry <subcommand> [args]",
        "",
        "MHCflurry %s" % __version__,
        "",
    ]
    name_width = max(
        len(name) for _, names in _HELP_GROUPS for name in names
    )
    for group_name, names in _HELP_GROUPS:
        lines.append("%s:" % group_name)
        for name in names:
            _, _, help_text = _SUBCOMMANDS[name]
            lines.append("  %s  %s" % (name.ljust(name_width), help_text))
        lines.append("")
    lines.extend([
        "Examples:",
        "  mhcflurry predict --alleles HLA-A0201 --peptides SIINFEKL --out out.csv",
        "  mhcflurry eval compare-models --a results/new_run/ --b public --out cmp/",
        "  mhcflurry eval paper-figures run --a results/new_run/ --out eval/",
        "  mhcflurry train pan-allele-release --run-dir runs/2.3.0 --release 2.3.0",
        "  mhcflurry <subcommand> --help",
        "",
        "Historical mhcflurry-* scripts remain installed and run the same code.",
        "",
        "Options:",
        "  -h, --help     show this help message and exit",
        "  --version, -V  show 'mhcflurry %s' and exit" % __version__,
    ])
    return "\n".join(lines)


def _rewrite_parser_prog(parser, prog):
    """Recursively rewrite ``parser.prog`` and its subparsers' progs.

    Returns a list of ``(parser, original_prog)`` pairs that
    :func:`_restore_parser_progs` can use to undo the change. Walking the
    subparser tree is necessary because sub-subcommands (``downloads
    fetch``, ``pseudosequences filename``, …) take their prog from the
    parent at parser-build time, which for legacy modules is import
    time — before the unified dispatcher gets to rename the top-level
    parser.
    """
    saved = []
    if parser is None or not hasattr(parser, "prog"):
        return saved
    saved.append((parser, parser.prog))
    parser.prog = prog
    # argparse._SubParsersAction is technically a private name (underscore
    # prefix), but the class has been stable across Python releases and is
    # the standard handle every CLI tool that walks subparser trees uses
    # (sphinx-autoprogram, click->argparse adapters, etc).
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub_name, sub_parser in action.choices.items():
                saved.extend(_rewrite_parser_prog(
                    sub_parser, "%s %s" % (prog, sub_name)))
    return saved


def _restore_parser_progs(saved):
    """Reverse :func:`_rewrite_parser_prog`; used as a tests-friendly finally."""
    for parser, original_prog in saved:
        parser.prog = original_prog


def _entry_accepts_prog(entry_func):
    """Return whether ``entry_func`` accepts a ``prog=`` display override."""
    # Imported here: it costs several milliseconds, and a Tab press never
    # dispatches.
    import inspect

    try:
        signature = inspect.signature(entry_func)
    except (TypeError, ValueError):
        return False
    return "prog" in signature.parameters


def main(argv=None):
    """Dispatch entry point for the ``mhcflurry`` console script."""
    if is_completing():
        # A shell Tab press: argcomplete prints the completions and exits.
        argcomplete.autocomplete(build_completion_parser())
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        # Bare ``mhcflurry``: treat as "user asked for help". stdout +
        # exit 0 so the screen can be piped into ``less`` and so
        # tab-completion shells that probe ``mhcflurry`` don't see a
        # spurious error.
        print(style_help(format_help(), sys.stdout))
        return 0
    if argv[0] in ("-h", "--help"):
        print(style_help(format_help(), sys.stdout))
        return 0
    if argv[0] in ("-V", "--version"):
        print("mhcflurry %s" % __version__)
        return 0
    subcommand = argv[0]
    remaining = argv[1:]
    if subcommand in _SUBCOMMANDS:
        module_path, entry, _ = _SUBCOMMANDS[subcommand]
        module = importlib.import_module(module_path)
        # Subcommands' parsers normally default ``prog`` to ``sys.argv[0]``
        # (the console script name). When dispatched under the parent, keep
        # display ``prog`` separate from ``sys.argv`` so artifact-producing
        # commands can still record a reproducible command in GENERATE.sh.
        prog = "mhcflurry %s" % subcommand
        saved_progs = _rewrite_parser_prog(
            getattr(module, "parser", None), prog)
        entry_func = getattr(module, entry)
        try:
            if _entry_accepts_prog(entry_func):
                return entry_func(remaining, prog=prog)
            return entry_func(remaining)
        finally:
            _restore_parser_progs(saved_progs)
    # Unknown subcommand — let argparse emit the standard error + usage.
    build_parser().parse_args(argv)
    return 2  # unreachable; parse_args exits


if __name__ == "__main__":
    sys.exit(main() or 0)
