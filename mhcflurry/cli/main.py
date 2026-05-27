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

from ..version import __version__


# Every subcommand is ``(module_path, entry_attr, one-line help)``. The
# entry function must accept a single ``argv`` argument; it owns its own
# arg parsing.
_SUBCOMMANDS = {
    "compare-models": (
        "mhcflurry.cli.compare_models", "run_argv",
        "Compare two model ensembles on data_evaluation."),
    "plot-model-comparison": (
        "mhcflurry.cli.plot_model_comparison", "run_argv",
        "Render plots from a compare-models output directory."),
    "predict": (
        "mhcflurry.predict_command", "run",
        "Predict MHC binding affinities for peptide/allele pairs."),
    "predict-scan": (
        "mhcflurry.predict_scan_command", "run",
        "Scan protein sequences for MHC-binding peptides."),
    "downloads": (
        "mhcflurry.downloads_command", "run",
        "Fetch, inspect, resolve data + model downloads."),
    "calibrate-percentile-ranks": (
        "mhcflurry.calibrate_percentile_ranks_command", "run",
        "Calibrate percentile ranks on an existing predictor."),
    "class1-train-allele-specific-models": (
        "mhcflurry.train_allele_specific_models_command", "run",
        "Train Class I allele-specific affinity models."),
    "class1-select-allele-specific-models": (
        "mhcflurry.select_allele_specific_models_command", "run",
        "Select Class I allele-specific models from a candidate pool."),
    "class1-train-pan-allele-models": (
        "mhcflurry.train_pan_allele_models_command", "run",
        "Train Class I pan-allele affinity models."),
    "class1-select-pan-allele-models": (
        "mhcflurry.select_pan_allele_models_command", "run",
        "Select Class I pan-allele models from a candidate pool."),
    "class1-train-processing-models": (
        "mhcflurry.train_processing_models_command", "run",
        "Train Class I antigen-processing models."),
    "class1-select-processing-models": (
        "mhcflurry.select_processing_models_command", "run",
        "Select Class I processing models from a candidate pool."),
    "class1-train-presentation-models": (
        "mhcflurry.train_presentation_models_command", "run",
        "Train the Class I presentation predictor (affinity + processing)."),
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
    ("Calibration", (
        "calibrate-percentile-ranks",
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
    ("Model comparison (new in 2.3.0)", (
        "compare-models", "plot-model-comparison",
    )),
    ("Helpers", (
        "pseudosequences",
    )),
)


def _check_help_groups():
    grouped = {name for _, names in _HELP_GROUPS for name in names}
    missing = set(_SUBCOMMANDS) - grouped
    extra = grouped - set(_SUBCOMMANDS)
    assert not missing, "subcommands missing from _HELP_GROUPS: %s" % missing
    assert not extra, "_HELP_GROUPS lists unknown subcommands: %s" % extra


_check_help_groups()


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
            "any subcommand."
        ),
    )
    parser.add_argument(
        "--version", action="version", version="mhcflurry %s" % __version__)
    sub = parser.add_subparsers(dest="subcommand", required=True)
    for name, (_module, _entry, help_text) in _SUBCOMMANDS.items():
        sub.add_parser(name, help=help_text, add_help=False)
    return parser


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
        "Every subcommand is also installed as a standalone "
        "mhcflurry-<subcommand>",
        "script. Both forms run the same underlying entry point.",
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
        "  mhcflurry compare-models --a results/new_run/ --b public --out cmp/",
        "  mhcflurry <subcommand> --help",
        "",
        "Options:",
        "  -h, --help     show this screen and exit",
        "  --version      show 'mhcflurry %s' and exit" % __version__,
    ])
    return "\n".join(lines)


def main(argv=None):
    """Dispatch entry point for the ``mhcflurry`` console script."""
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        # Bare ``mhcflurry``: print the friendly help and exit. Non-zero
        # exit matches argparse's "missing required positional" convention
        # (and how git, kubectl, etc. behave).
        print(format_help())
        return 2
    if argv[0] in ("-h", "--help"):
        print(format_help())
        return 0
    if argv[0] in ("-V", "--version"):
        print("mhcflurry %s" % __version__)
        return 0
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
