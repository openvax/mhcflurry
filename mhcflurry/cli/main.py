"""Top-level ``mhcflurry`` CLI dispatcher.

Every subcommand is registered as ``(module path, entry attr, one-line
help)`` and lazy-imported on dispatch, so ``mhcflurry --help`` and
``mhcflurry <subcommand> --help`` only pull in the modules they actually
need. This keeps the torch-import cost off the top-level help path.

Two flavors of subcommand:

* **New under the parent**: ``compare-models``, ``plot-model-comparison``.
  Each module exposes ``run_argv(argv)`` which does its own argparse.
* **Legacy mhcflurry-* commands**: ``predict``, ``predict-scan``,
  ``downloads``, ``calibrate-percentile-ranks``, the ``class1-train-*`` /
  ``class1-select-*`` family, ``pseudosequences``. Each is wrapped by
  invoking the legacy module's existing ``run(argv)`` (or ``main(argv)``
  for ``pseudosequences``) on the post-subcommand argv. All
  ``mhcflurry-*`` console_scripts in ``setup.py`` remain installed as
  compat shims pointing at the same underlying entry functions.
"""
from __future__ import annotations

import argparse
import importlib
import sys


# Every subcommand is ``(module_path, entry_attr, one-line help)``. The
# entry function must accept a single ``argv`` argument; it owns its own
# arg parsing. Listed in the order they should appear in ``--help``.
_SUBCOMMANDS = {
    # New under the parent.
    "compare-models": (
        "mhcflurry.cli.compare_models", "run_argv",
        "Compare two model ensembles (run-vs-run or run-vs-public) on "
        "data_evaluation."),
    "plot-model-comparison": (
        "mhcflurry.cli.plot_model_comparison", "run_argv",
        "Render ROC / PR / scatter / delta plots from a compare-models "
        "output directory."),
    # Legacy mhcflurry-* commands (still installed standalone as compat
    # shims; same entry functions).
    "predict": (
        "mhcflurry.predict_command", "run",
        "Predict MHC binding affinities for peptide/allele pairs."),
    "predict-scan": (
        "mhcflurry.predict_scan_command", "run",
        "Scan protein sequences for MHC-binding peptides."),
    "downloads": (
        "mhcflurry.downloads_command", "run",
        "Fetch, inspect, and resolve MHCflurry data + model downloads."),
    "calibrate-percentile-ranks": (
        "mhcflurry.calibrate_percentile_ranks_command", "run",
        "Calibrate percentile ranks on an existing predictor."),
    "class1-train-allele-specific-models": (
        "mhcflurry.train_allele_specific_models_command", "run",
        "Train class1 allele-specific affinity models."),
    "class1-select-allele-specific-models": (
        "mhcflurry.select_allele_specific_models_command", "run",
        "Select class1 allele-specific models from a candidate pool."),
    "class1-train-pan-allele-models": (
        "mhcflurry.train_pan_allele_models_command", "run",
        "Train class1 pan-allele affinity models."),
    "class1-select-pan-allele-models": (
        "mhcflurry.select_pan_allele_models_command", "run",
        "Select class1 pan-allele models from a candidate pool."),
    "class1-train-processing-models": (
        "mhcflurry.train_processing_models_command", "run",
        "Train class1 antigen-processing models."),
    "class1-select-processing-models": (
        "mhcflurry.select_processing_models_command", "run",
        "Select class1 processing models from a candidate pool."),
    "class1-train-presentation-models": (
        "mhcflurry.train_presentation_models_command", "run",
        "Train the class1 presentation predictor (affinity + processing)."),
    "pseudosequences": (
        "mhcflurry.pseudosequences", "main",
        "Pseudosequence CSV registry helper (filename / path / list)."),
}


def build_parser():
    """Return the top-level parser used by ``--help`` and tooling.

    Subparsers are registered by name only — no per-subcommand arguments
    are added here. The legacy module owns its own ``--help`` output and
    is invoked lazily by :func:`main`. Keeps this function torch-free.
    """
    parser = argparse.ArgumentParser(
        prog="mhcflurry",
        description=(
            "MHCflurry. Use 'mhcflurry <subcommand> --help' for details on "
            "any subcommand. Each subcommand under this parent is also "
            "available as a standalone 'mhcflurry-<subcommand>' script."
        ),
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)
    for name, (_module, _entry, help_text) in _SUBCOMMANDS.items():
        sub.add_parser(name, help=help_text, add_help=False)
    return parser


def main(argv=None):
    """Dispatch entry point for the ``mhcflurry`` console script."""
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        build_parser().print_help()
        return 0 if argv else 2
    subcommand = argv[0]
    remaining = argv[1:]
    if subcommand in _SUBCOMMANDS:
        module_path, entry, _ = _SUBCOMMANDS[subcommand]
        module = importlib.import_module(module_path)
        # Subcommands' parsers default ``prog`` to ``sys.argv[0]`` (the
        # console script name). When dispatched under the parent, fix
        # the displayed name so ``--help`` shows ``mhcflurry
        # <subcommand>`` instead of just ``mhcflurry``.
        prog = "mhcflurry %s" % subcommand
        saved_argv0 = sys.argv[0]
        sys.argv[0] = prog
        if hasattr(module, "parser") and hasattr(module.parser, "prog"):
            saved_prog = module.parser.prog
            module.parser.prog = prog
        else:
            saved_prog = None
        try:
            return getattr(module, entry)(remaining)
        finally:
            sys.argv[0] = saved_argv0
            if saved_prog is not None:
                module.parser.prog = saved_prog
    # Unknown subcommand — let argparse emit the standard error + usage.
    build_parser().parse_args(argv)
    return 2  # unreachable; parse_args exits


if __name__ == "__main__":
    sys.exit(main() or 0)
