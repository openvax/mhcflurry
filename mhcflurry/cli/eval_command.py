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

"""Evaluation command namespace.

``mhcflurry eval`` is the semantic home for model comparison, benchmark score
generation, and paper-style figures. The first release keeps the existing
commands as compatibility entry points while making the preferred shape
available:

* ``mhcflurry eval compare-models`` delegates to ``mhcflurry compare-models``.
* ``mhcflurry eval plot-comparison`` delegates to
  ``mhcflurry plot-model-comparison``.
* ``mhcflurry eval paper-figures render`` delegates to
  ``mhcflurry paper-figures``.
* ``mhcflurry eval paper-figures score-predictions`` derives reusable AUC/PPV
  score tables from saved benchmark prediction tables.
* ``mhcflurry eval paper-figures run`` runs compare-models, paper-figures, and
  plot-model-comparison as one local evaluation/figure pipeline.

Future benchmark-prediction and external-predictor registration commands should
be added under this namespace rather than as new top-level commands.
"""
from __future__ import annotations

import argparse
import os
import sys


def make_parser(prog="mhcflurry eval"):
    """Return the lightweight namespace parser used by docs/help tooling."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="eval_subcommand", required=True)
    sub.add_parser(
        "compare-models",
        help="Compare two model ensembles on data_evaluation.",
        add_help=False,
    )
    sub.add_parser(
        "plot-comparison",
        help="Render plots from a compare-models output directory.",
        add_help=False,
    )
    sub.add_parser(
        "plot-model-comparison",
        help="Compatibility alias for plot-comparison.",
        add_help=False,
    )
    paper = sub.add_parser(
        "paper-figures",
        help="Render or run paper-style evaluation figures.",
        add_help=False,
    )
    paper_sub = paper.add_subparsers(dest="paper_figures_subcommand")
    paper_sub.add_parser(
        "render",
        help="Render paper-style figures from saved inputs.",
        add_help=False,
    )
    paper_sub.add_parser(
        "run",
        help="Compare models, render paper figures, and write plot PDFs.",
        add_help=False,
    )
    paper_sub.add_parser(
        "score-predictions",
        help="Derive reusable score tables from saved predictions.",
        add_help=False,
    )
    return parser


parser = make_parser()


def run_argv(argv, prog="mhcflurry eval"):
    """Dispatch ``mhcflurry eval`` subcommands."""
    argv = list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(format_help(prog))
        return 0

    subcommand = argv[0]
    rest = argv[1:]
    if subcommand == "compare-models":
        from . import compare_models
        return _run_existing_command(
            compare_models, rest, "%s compare-models" % prog)
    if subcommand in ("plot-comparison", "plot-model-comparison"):
        from . import plot_model_comparison
        return _run_existing_command(
            plot_model_comparison, rest, "%s %s" % (prog, subcommand))
    if subcommand == "paper-figures":
        return _run_paper_figures(rest, "%s paper-figures" % prog)

    make_parser(prog).parse_args(argv)
    return 2


def format_help(prog="mhcflurry eval"):
    """Return a compact help screen for the eval namespace."""
    lines = [
        "usage: %s <subcommand> [args]" % prog,
        "",
        "Evaluation and paper-figure workflows.",
        "",
        "Subcommands:",
        "  compare-models          Compare two model ensembles.",
        "  plot-comparison         Render diagnostic plots from compare output.",
        "  paper-figures render    Render paper figures from saved inputs.",
        "  paper-figures score-predictions",
        "                          Derive score tables from saved predictions.",
        "  paper-figures run       Compare, render paper figures, and write PDFs.",
        "",
        "Compatibility:",
        "  mhcflurry compare-models, mhcflurry plot-model-comparison, and",
        "  mhcflurry paper-figures remain supported.",
        "",
        "Options:",
        "  -h, --help  show this help message and exit",
    ]
    return "\n".join(lines)


def _run_existing_command(module, argv, prog):
    command_parser = module.make_parser()
    command_parser.prog = prog
    return module.run(command_parser.parse_args(argv))


def _run_paper_figures(argv, prog):
    if not argv or argv[0] in ("-h", "--help"):
        print(_format_paper_figures_help(prog))
        return 0
    subcommand = argv[0]
    if subcommand == "render":
        from . import paper_figures
        return _run_existing_command(
            paper_figures, argv[1:], "%s render" % prog)
    if subcommand == "score-predictions":
        return _run_score_predictions(
            _make_score_predictions_parser(
                "%s score-predictions" % prog).parse_args(argv[1:]))
    if subcommand == "run":
        return _run_paper_figures_pipeline(
            _make_paper_figures_run_parser("%s run" % prog).parse_args(
                argv[1:]))
    if subcommand.startswith("-"):
        # Compatibility shortcut: ``mhcflurry eval paper-figures --out ...``
        # behaves like the explicit ``render`` form.
        from . import paper_figures
        return _run_existing_command(paper_figures, argv, prog)

    _make_paper_figures_parser(prog).parse_args(argv)
    return 2


def _make_paper_figures_parser(prog):
    paper_parser = argparse.ArgumentParser(prog=prog)
    sub = paper_parser.add_subparsers(dest="paper_figures_subcommand", required=True)
    sub.add_parser("render", add_help=False)
    sub.add_parser("score-predictions", add_help=False)
    sub.add_parser("run", add_help=False)
    return paper_parser


def _format_paper_figures_help(prog):
    return "\n".join([
        "usage: %s <subcommand> [args]" % prog,
        "",
        "Paper-style evaluation figure workflows.",
        "",
        "Subcommands:",
        "  render  Render figures from saved comparison/scores/prediction inputs.",
        "  score-predictions",
        "          Derive reusable score tables from saved predictions.",
        "  run     Run compare-models, render paper figures, and write PDFs.",
        "",
        "Use '%s render --help', '%s score-predictions --help', or "
        "'%s run --help' for arguments." % (prog, prog, prog),
    ])


def _make_score_predictions_parser(prog):
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Derive reusable notebook-style AUC/PPV score tables from a saved "
            "benchmark prediction table. The output can be passed back to "
            "paper-figures through --scores-dir."
        ),
    )
    parser.add_argument(
        "--kind",
        required=True,
        choices=("multiallelic", "monoallelic"),
        help="Benchmark kind. Sets the default grouping column.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Saved benchmark prediction table (CSV or CSV.BZ2).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help=(
            "Output CSV path, typically accuracy_scores.multiallelic.csv or "
            "accuracy_scores.monoallelic.csv."
        ),
    )
    parser.add_argument(
        "--index-column",
        help=(
            "Optional grouping column override. Defaults to sample_id for "
            "multiallelic and auto-detect for monoallelic."
        ),
    )
    parser.add_argument(
        "--external-baselines",
        default=None,
        help=(
            "Comma-separated external predictor comparators used for percent "
            "change columns. Default matches paper-figures render."
        ),
    )
    parser.add_argument(
        "--predictor-info",
        help=(
            "Optional predictor_info.csv with predictor and higher_is_better "
            "columns for custom score columns."
        ),
    )
    return parser


def _run_score_predictions(args):
    from . import paper_figures

    if args.external_baselines:
        external_baselines = paper_figures._parse_external_baselines(
            args.external_baselines)
    else:
        external_baselines = paper_figures.EXTERNAL_BASELINES
    index_column = args.index_column
    if index_column is None and args.kind == "multiallelic":
        index_column = "sample_id"
    predictor_info = None
    if args.predictor_info:
        import pandas
        predictor_info = pandas.read_csv(args.predictor_info)
        if "predictor" in predictor_info.columns:
            predictor_info = predictor_info.set_index("predictor", drop=False)
    scores = paper_figures.score_saved_prediction_table(
        args.input,
        index_column=index_column,
        kind=args.kind,
        predictor_info=predictor_info,
        external_baselines=external_baselines,
    )
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    scores.to_csv(out, index=False)
    print("Wrote: %s" % out)
    return 0


def _make_paper_figures_run_parser(prog):
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Run the local evaluation-to-figures workflow for a trained model. "
            "This composes compare-models, paper-figures, and "
            "plot-model-comparison; remote orchestration remains in the release "
            "wrapper."
        ),
    )
    parser.add_argument(
        "--a", required=True,
        help="Candidate model run directory, 'public', or 'public:<release>'.",
    )
    parser.add_argument(
        "--b", default="public",
        help="Baseline model run directory or public release. Default: %(default)s.",
    )
    parser.add_argument("--a-label", help="Display label for side A.")
    parser.add_argument("--b-label", help="Display label for side B.")
    parser.add_argument(
        "--out", required=True,
        help="Evaluation output directory. compare-models writes here.",
    )
    parser.add_argument(
        "--data-dir",
        help="data_evaluation directory. Defaults to installed data_evaluation.",
    )
    parser.add_argument(
        "--include",
        default="auto",
        help="compare-models component subset. Default: %(default)s.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        help="Smoke-test: only read first N benchmark files.",
    )
    parser.add_argument(
        "--scores-dir",
        help="Saved figure-input directory passed to paper-figures.",
    )
    parser.add_argument(
        "--multiallelic-predictions",
        help="Saved multiallelic prediction table passed to paper-figures.",
    )
    parser.add_argument(
        "--monoallelic-predictions",
        help="Saved monoallelic prediction table passed to paper-figures.",
    )
    parser.add_argument(
        "--paper-figures-out",
        help="Paper-figure output directory. Default: <out>/plots/paper_figures.",
    )
    parser.add_argument(
        "--formats",
        default="svg,pdf,png",
        help="Paper-figure formats. Default: %(default)s.",
    )
    parser.add_argument(
        "--summary-pdf",
        help="Combined diagnostic PDF. Default: <out>/plots/model_comparison_figures.pdf.",
    )
    parser.add_argument(
        "--candidate-predictor",
        help="Candidate predictor name passed to paper-figures.",
    )
    parser.add_argument(
        "--external-baselines",
        help="External baseline predictor list passed to paper-figures.",
    )
    parser.add_argument(
        "--preferred-predictors",
        help="Preferred predictor list passed to paper-figures.",
    )
    parser.add_argument(
        "--presentation-panel-predictors",
        help="Presentation panel candidate predictor list passed to paper-figures.",
    )
    parser.add_argument(
        "--presentation-panel-baselines",
        help="Presentation panel baseline predictor list passed to paper-figures.",
    )
    parser.add_argument(
        "--skip-comparison-plots",
        action="store_true",
        default=False,
        help="Skip plot-model-comparison after paper figures are rendered.",
    )
    return parser


def _run_paper_figures_pipeline(args):
    from . import compare_models
    from . import paper_figures
    from . import plot_model_comparison

    os.makedirs(args.out, exist_ok=True)
    compare_argv = [
        "--a", args.a,
        "--b", args.b,
        "--out", args.out,
        "--include", args.include,
    ]
    for source, flag in [
            (args.a_label, "--a-label"),
            (args.b_label, "--b-label"),
            (args.data_dir, "--data-dir"),
            (args.limit_files, "--limit-files")]:
        if source is not None:
            compare_argv.extend([flag, str(source)])
    compare_status = compare_models.run(
        compare_models.make_parser().parse_args(compare_argv))
    if compare_status:
        return compare_status

    paper_out = args.paper_figures_out or os.path.join(
        args.out, "plots", "paper_figures")
    paper_argv = [
        "--comparison-dir", args.out,
        "--out", paper_out,
        "--formats", args.formats,
    ]
    for source, flag in [
            (args.scores_dir, "--scores-dir"),
            (args.multiallelic_predictions, "--multiallelic-predictions"),
            (args.monoallelic_predictions, "--monoallelic-predictions"),
            (args.candidate_predictor, "--candidate-predictor"),
            (args.external_baselines, "--external-baselines"),
            (args.preferred_predictors, "--preferred-predictors"),
            (args.presentation_panel_predictors,
             "--presentation-panel-predictors"),
            (args.presentation_panel_baselines,
             "--presentation-panel-baselines")]:
        if source:
            paper_argv.extend([flag, source])
    paper_status = paper_figures.run(
        paper_figures.make_parser().parse_args(paper_argv))
    if paper_status:
        return paper_status

    if args.skip_comparison_plots:
        return 0

    summary_pdf = args.summary_pdf or os.path.join(
        args.out, "plots", "model_comparison_figures.pdf")
    plot_argv = [
        "--input", args.out,
        "--summary-pdf", summary_pdf,
        "--paper-figures-out", paper_out,
        "--include-paper-figures-in-summary-pdf",
    ]
    return plot_model_comparison.run(
        plot_model_comparison.make_parser().parse_args(plot_argv))


if __name__ == "__main__":
    sys.exit(run_argv(sys.argv[1:]) or 0)
