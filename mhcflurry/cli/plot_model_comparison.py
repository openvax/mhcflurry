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

"""Plot the metric outputs from ``mhcflurry compare-models``.

Reads the CSVs + JSON written by ``compare-models`` and renders ROC / PR
/ scatter / per-allele delta plots under ``<input>/plots/`` for affinity,
processing, and presentation comparisons. Kept as a separate subcommand so
the metric pipeline doesn't pay the matplotlib import cost.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy
import pandas

from .figure_style import (
    DIAGONAL_COLOR,
    NEGATIVE_DELTA_COLOR,
    POSITIVE_DELTA_COLOR,
    SIDE_A_COLOR,
    SIDE_B_COLOR,
    apply_paper_style as _apply_paper_style,
)
from .model_comparison_constants import (
    METRIC_NAMES,
    PRESENTATION_MODES,
    PRESENTATION_SCORE_KINDS,
    PROCESSING_MODES,
)
from ..common import positive_int_arg


METRIC_DISPLAY_NAMES = {
    "roc_auc": "AUROC",
    "pr_auc": "AUPRC",
    "ppv_at_n": "PPV@N",
}

def make_parser():
    """Return a standalone parser for documentation tooling (autoprogram)."""
    parser = argparse.ArgumentParser(prog="mhcflurry plot-model-comparison")
    register_subparser(parser)
    return parser


def run_argv(argv):
    """Entry point for the lazy ``mhcflurry plot-model-comparison`` dispatcher."""
    return run(make_parser().parse_args(argv))


def register_subparser(parser):
    parser.description = __doc__
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = """
Plotting layers:
  * This command always reads an existing compare-models directory and writes
    diagnostic plots under <input>/plots.
  * --summary-pdf collects those diagnostic plot PDFs into one review packet.
  * The --paper-figures-* flags optionally call ``mhcflurry eval paper-figures
    render`` with saved score/prediction inputs for the broader paper-style
    figure suite.
  * Custom saved-prediction score columns need predictor_info.csv metadata with
    predictor and higher_is_better.
"""
    parser.add_argument(
        "--input", required=True,
        help="Output directory produced by ``mhcflurry compare-models``.",
    )
    parser.add_argument(
        "--max-scatter-points", type=positive_int_arg, default=100_000,
        help="Subsample scatter plots above this many points (default 100k).",
    )
    parser.add_argument(
        "--components", default="auto",
        help=(
            "Comma-separated subset of {affinity, processing, presentation}; "
            "default 'auto' plots whichever components are present in --input."
        ),
    )
    parser.add_argument(
        "--summary-pdf",
        help=(
            "Optional PDF path. When set, generated plots are collected into "
            "a single PDF, preserving vector plots where possible. Use a "
            "top-level file under <input>/plots or a path outside the plot "
            "tree; paper and diagnostic subdirectories are reserved."
        ),
    )
    parser.add_argument(
        "--a-label",
        help=(
            "Override side A label from side_a.json. Useful for regenerating "
            "plots from an existing comparison without recomputing metrics."
        ),
    )
    parser.add_argument(
        "--b-label",
        help=(
            "Override side B label from side_b.json. Useful for regenerating "
            "plots from an existing comparison without recomputing metrics."
        ),
    )
    parser.add_argument(
        "--paper-figures-scores-dir",
        help=(
            "Optional saved figure-input directory passed to "
            "``mhcflurry paper-figures --scores-dir``. This may contain "
            "saved prediction tables, derived score tables, and predictor "
            "metadata such as predictor_info.csv."
        ),
    )
    parser.add_argument(
        "--paper-figures-artifacts-dir",
        help=(
            "Compatibility alias for --paper-figures-scores-dir."
        ),
    )
    parser.add_argument(
        "--paper-figures-out",
        help=(
            "Output directory for paper-style figures. Default: "
            "<input>/plots/paper_figures. Must be a dedicated directory, "
            "not <input>/plots or a diagnostic component directory."
        ),
    )
    parser.add_argument(
        "--paper-figures-formats",
        default="svg,pdf,png",
        help="Comma-separated paper-figure formats. Default: %(default)s.",
    )
    parser.add_argument(
        "--include-paper-figures-in-summary-pdf",
        action="store_true",
        default=False,
        help=(
            "When --summary-pdf and paper-style figures are enabled, append "
            "paper_figures/paper_figures.pdf to the summary PDF."
        ),
    )
    parser.add_argument(
        "--paper-figures-candidate-predictor",
        help="Candidate predictor passed through to ``mhcflurry paper-figures``.",
    )
    parser.add_argument(
        "--paper-figures-external-baselines",
        help=(
            "External baselines passed through to ``mhcflurry paper-figures`` "
            "as PREDICTOR or PREDICTOR:PERCENT_CHANGE_SUFFIX."
        ),
    )
    parser.add_argument(
        "--paper-figures-multiallelic-predictions",
        help=(
            "Saved multiallelic test-set prediction table passed through to "
            "``mhcflurry paper-figures``."
        ),
    )
    parser.add_argument(
        "--paper-figures-monoallelic-predictions",
        help=(
            "Saved monoallelic test-set prediction table passed through to "
            "``mhcflurry paper-figures``."
        ),
    )
    parser.add_argument(
        "--paper-figures-preferred-predictors",
        help=(
            "Comma-separated preferred predictors passed through to "
            "``mhcflurry paper-figures``."
        ),
    )
    parser.add_argument(
        "--paper-figures-presentation-panel-predictors",
        help=(
            "Comma-separated presentation-panel candidate predictors passed "
            "through to ``mhcflurry paper-figures``."
        ),
    )
    parser.add_argument(
        "--paper-figures-presentation-panel-baselines",
        help=(
            "Comma-separated presentation-panel baseline predictors passed "
            "through to ``mhcflurry paper-figures``."
        ),
    )
    return parser


def run(args):
    import matplotlib
    matplotlib.use("Agg")

    _apply_paper_style()
    labels = _load_side_labels(args.input)
    if args.a_label:
        labels["a"] = args.a_label
    if args.b_label:
        labels["b"] = args.b_label
    available = _detect_available_components(args.input)
    if args.components == "auto":
        components = available
    else:
        requested = [p.strip() for p in args.components.split(",")]
        if not requested or any(not component for component in requested):
            raise SystemExit("--components contains an empty component")
        duplicates = sorted({
            component for component in requested
            if requested.count(component) > 1
        })
        if duplicates:
            raise SystemExit("Duplicate --components entries: %s" % (
                ", ".join(duplicates)))
        components = [c for c in requested if c in available]
        missing = sorted(set(requested) - set(available))
        if missing:
            raise SystemExit(
                "Requested component(s) not present in %s: %s" % (
                    args.input, ", ".join(missing)))

    plot_dir = os.path.join(args.input, "plots")
    paper_figures_dir = args.paper_figures_out or os.path.join(
        plot_dir, "paper_figures")
    paper_scores_dir = (
        args.paper_figures_scores_dir or args.paper_figures_artifacts_dir
    )
    _validate_paper_figure_inputs_outside_plot_dir(
        plot_dir,
        {
            "--paper-figures-scores-dir": paper_scores_dir,
            "--paper-figures-multiallelic-predictions": (
                args.paper_figures_multiallelic_predictions
            ),
            "--paper-figures-monoallelic-predictions": (
                args.paper_figures_monoallelic_predictions
            ),
        },
    )
    _validate_plot_output_paths(
        plot_dir, paper_figures_dir, args.summary_pdf)
    _reset_plot_directory(
        plot_dir, preserve_directory=paper_figures_dir)
    if args.summary_pdf:
        try:
            os.unlink(args.summary_pdf)
        except FileNotFoundError:
            pass
    paper_dir = os.path.join(plot_dir, "paper")
    os.makedirs(paper_dir, exist_ok=True)

    for component in components:
        if component == "affinity":
            _plot_affinity(
                args.input, plot_dir, labels, args.max_scatter_points)
        elif component == "processing":
            _plot_processing(
                args.input, plot_dir, labels, args.max_scatter_points)
        elif component == "presentation":
            _plot_presentation(
                args.input, plot_dir, labels, args.max_scatter_points)
    _plot_release_summary(args.input, paper_dir, labels)
    paper_inputs_requested = any([
        paper_scores_dir,
        args.paper_figures_multiallelic_predictions,
        args.paper_figures_monoallelic_predictions,
    ])
    if paper_inputs_requested:
        from . import paper_figures

        paper_argv = [
            "--comparison-dir", args.input,
            "--out", paper_figures_dir,
            "--formats", args.paper_figures_formats,
        ]
        if paper_scores_dir:
            paper_argv.extend(["--scores-dir", paper_scores_dir])
        paper_args = paper_figures.make_parser().parse_args(paper_argv)
        _append_optional_paper_figure_args(args, paper_args)
        status = paper_figures.run(paper_args)
        if status:
            return status
    if args.summary_pdf:
        _write_summary_pdf(
            plot_dir,
            args.summary_pdf,
            include_paper_figures=args.include_paper_figures_in_summary_pdf,
            paper_figures_dir=paper_figures_dir,
        )
        if not os.path.isfile(args.summary_pdf):
            raise RuntimeError(
                "No plot pages were generated for summary PDF: %s" %
                args.summary_pdf)
    return 0


def _reset_plot_directory(plot_dir, preserve_directory=None):
    """Remove stale plots while retaining an existing paper-figure suite."""
    plot_dir = Path(os.path.abspath(plot_dir))
    preserve_directory = (
        Path(os.path.abspath(preserve_directory))
        if preserve_directory is not None else None
    )
    if (
            preserve_directory is not None
            and not _path_is_within(preserve_directory, plot_dir)):
        preserve_directory = None

    if plot_dir.is_dir() and not plot_dir.is_symlink():
        _clean_directory_except(plot_dir, preserve_directory)
    elif plot_dir.exists() or plot_dir.is_symlink():
        plot_dir.unlink()
    plot_dir.mkdir(parents=True, exist_ok=True)


def _validate_paper_figure_inputs_outside_plot_dir(plot_dir, input_paths):
    """Reject paper inputs that diagnostic cleanup would recursively remove."""
    plot_dir = Path(os.path.abspath(plot_dir))
    resolved_plot_dir = plot_dir.resolve(strict=False)
    conflicts = []
    for option, value in input_paths.items():
        if value is None:
            continue
        path = Path(os.path.abspath(value))
        if _path_is_within_location(
                path, plot_dir, resolved_directory=resolved_plot_dir):
            conflicts.append((option, path))
    if conflicts:
        details = ", ".join(
            "%s=%s" % (option, path) for (option, path) in conflicts)
        raise SystemExit(
            "Paper-figure input paths must be outside the diagnostic plot "
            "output directory %s because that directory is cleared before "
            "rendering. Refusing to delete input: %s. Move the input or "
            "choose a different --input comparison directory."
            % (plot_dir, details)
        )


def _validate_plot_output_paths(
        plot_dir, paper_figures_dir, summary_pdf=None):
    """Reject overlapping command-owned output locations before cleanup."""
    plot_dir = Path(os.path.abspath(plot_dir))
    paper_figures_dir = Path(os.path.abspath(paper_figures_dir))
    diagnostic_dirs = tuple(
        plot_dir / name
        for name in ("affinity", "processing", "presentation", "paper")
    )
    comparison_dir = plot_dir.parent

    if _same_location(paper_figures_dir, plot_dir) or any(
            _path_is_within_location(paper_figures_dir, directory)
            for directory in diagnostic_dirs):
        raise SystemExit(
            "--paper-figures-out must be a dedicated directory distinct "
            "from the diagnostic plot directory and its command-owned "
            "component directories. Refusing unsafe output: %s"
            % paper_figures_dir
        )
    if (
            _path_is_within_location(paper_figures_dir, comparison_dir)
            and not _path_is_within_location(paper_figures_dir, plot_dir)):
        raise SystemExit(
            "--paper-figures-out cannot be inside the comparison input tree "
            "but outside its plots directory: %s" % paper_figures_dir
        )

    if summary_pdf is None:
        return
    summary_pdf = Path(os.path.abspath(summary_pdf))
    conflicting_dirs = (paper_figures_dir,) + diagnostic_dirs
    if (
            _same_location(summary_pdf, plot_dir)
            or any(
                _path_is_within_location(summary_pdf, directory)
                for directory in conflicting_dirs)
    ):
        raise SystemExit(
            "--summary-pdf collides with command-owned figure output: %s. "
            "Use a top-level file under %s (for example, %s) or a path "
            "outside the paper and diagnostic output directories."
            % (
                summary_pdf,
                plot_dir,
                plot_dir / "model_comparison_figures.pdf",
            )
        )
    if (
            _path_is_within_location(summary_pdf, comparison_dir)
            and not _path_is_within_location(summary_pdf, plot_dir)):
        raise SystemExit(
            "--summary-pdf cannot overwrite files in the comparison input "
            "tree outside its plots directory: %s" % summary_pdf
        )


def _same_location(left, right):
    left = Path(os.path.abspath(left))
    right = Path(os.path.abspath(right))
    return (
        left == right
        or left.resolve(strict=False) == right.resolve(strict=False)
    )


def _path_is_within_location(path, directory, resolved_directory=None):
    path = Path(os.path.abspath(path))
    directory = Path(os.path.abspath(directory))
    if _path_is_within(path, directory):
        return True
    if resolved_directory is None:
        resolved_directory = directory.resolve(strict=False)
    return _path_is_within(
        path.resolve(strict=False), resolved_directory)


def _clean_directory_except(directory, preserve_directory):
    for child in directory.iterdir():
        child = Path(os.path.abspath(child))
        if preserve_directory is not None and child == preserve_directory:
            continue
        if (
                preserve_directory is not None
                and _path_is_within(preserve_directory, child)
                and child.is_dir()
                and not child.is_symlink()):
            _clean_directory_except(child, preserve_directory)
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _path_is_within(path, directory):
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _append_optional_paper_figure_args(args, paper_args):
    passthrough = {
        "paper_figures_candidate_predictor": "candidate_predictor",
        "paper_figures_external_baselines": "external_baselines",
        "paper_figures_multiallelic_predictions": "multiallelic_predictions",
        "paper_figures_monoallelic_predictions": "monoallelic_predictions",
        "paper_figures_preferred_predictors": "preferred_predictors",
        "paper_figures_presentation_panel_predictors": (
            "presentation_panel_predictors"
        ),
        "paper_figures_presentation_panel_baselines": (
            "presentation_panel_baselines"
        ),
    }
    for source, target in passthrough.items():
        value = getattr(args, source)
        if value:
            setattr(paper_args, target, value)


def _safe_plot(label, func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(
            "WARNING: skipping %s: %s: %s" % (
                label, type(e).__name__, e))
        return None


def _load_side_labels(input_dir):
    labels = {"a": "Side A", "b": "Side B"}
    for letter in ("a", "b"):
        path = os.path.join(input_dir, "side_%s.json" % letter)
        if os.path.isfile(path):
            with open(path) as fd:
                labels[letter] = json.load(fd).get("label", letter)
    return labels


def _detect_available_components(input_dir):
    components = []
    affinity_dir = os.path.join(input_dir, "affinity")
    if any(
            os.path.isfile(os.path.join(affinity_dir, name))
            for name in (
                "predictions.csv.bz2",
                "per_allele.csv",
                "per_length.csv",
                "summary.json",
            )):
        components.append("affinity")
    if os.path.isdir(os.path.join(input_dir, "processing")):
        components.append("processing")
    if os.path.isdir(os.path.join(input_dir, "presentation")):
        components.append("presentation")
    return components


def _read_optional_csv(path):
    try:
        return pandas.read_csv(path)
    except pandas.errors.EmptyDataError:
        return pandas.DataFrame()


# ---------------------------------------------------------------------------
# affinity
# ---------------------------------------------------------------------------


def _plot_affinity(input_dir, plot_dir, labels, max_scatter_points):
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        average_precision_score, precision_recall_curve,
        roc_auc_score, roc_curve,
    )

    sub_dir = os.path.join(plot_dir, "affinity")
    os.makedirs(sub_dir, exist_ok=True)
    paper_dir = os.path.join(plot_dir, "paper")
    os.makedirs(paper_dir, exist_ok=True)

    pred_path = os.path.join(input_dir, "affinity", "predictions.csv.bz2")
    label_a, label_b = labels["a"], labels["b"]
    if os.path.isfile(pred_path):
        df = _read_optional_csv(pred_path)
        if not {"hit", "a_score", "b_score"}.issubset(df.columns):
            df = pandas.DataFrame()
    else:
        df = pandas.DataFrame()
    if not df.empty:
        y = df.hit.values
        a_score = df.a_score.values
        b_score = df.b_score.values

        _save_roc(plt, roc_curve, roc_auc_score,
                  y, a_score, b_score, label_a, label_b,
                  os.path.join(sub_dir, "roc.png"), title="Affinity ROC")
        _save_pr(plt, precision_recall_curve, average_precision_score,
                 y, a_score, b_score, label_a, label_b,
                 os.path.join(sub_dir, "pr.png"), title="Affinity PR")
        _save_scatter(plt, b_score, a_score, label_b, label_a,
                      os.path.join(sub_dir, "scatter.png"),
                      title="Affinity score: %s vs %s" % (label_a, label_b),
                      max_points=max_scatter_points)

    per_allele_path = os.path.join(input_dir, "affinity", "per_allele.csv")
    if os.path.isfile(per_allele_path):
        per_allele = _read_optional_csv(per_allele_path)
        _save_per_allele_delta(plt, per_allele, sub_dir, label_a, label_b)
        _save_metric_scatter_grid(
            plt,
            [("alleles", per_allele)],
            os.path.join(paper_dir, "affinity_per_allele_scatter.png"),
            label_a,
            label_b,
            "Affinity per-allele accuracy",
        )

    per_length_path = os.path.join(input_dir, "affinity", "per_length.csv")
    if os.path.isfile(per_length_path):
        _save_per_length_grid(
            plt,
            [("affinity", pandas.read_csv(per_length_path))],
            os.path.join(paper_dir, "affinity_per_length_macro.png"),
            label_a,
            label_b,
            "Affinity by peptide length",
        )


def _save_per_allele_delta(plt, per_allele, sub_dir, label_a, label_b):
    if "roc_auc_diff" not in per_allele.columns:
        return
    sorted_df = per_allele.sort_values("roc_auc_diff", ascending=False)
    if sorted_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.1, 2.5))
    values = sorted_df["roc_auc_diff"].to_numpy(dtype=float)
    ax.bar(
        numpy.arange(len(sorted_df)),
        values,
        color=_delta_colors(values),
        edgecolor="white",
        linewidth=0.3,
    )
    ax.axhline(0, color=DIAGONAL_COLOR, linewidth=0.8)
    ax.set_xlabel("allele (sorted by ROC-AUC delta)")
    ax.set_ylabel("%s - %s ROC-AUC" % (label_a, label_b))
    ax.set_title("Per-allele ROC-AUC delta")
    fig.tight_layout()
    _save_figure(fig, os.path.join(sub_dir, "per_allele_roc_delta.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# processing
# ---------------------------------------------------------------------------


def _plot_processing(input_dir, plot_dir, labels, max_scatter_points):
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        average_precision_score, precision_recall_curve,
        roc_auc_score, roc_curve,
    )

    sub_dir = os.path.join(plot_dir, "processing")
    os.makedirs(sub_dir, exist_ok=True)
    paper_dir = os.path.join(plot_dir, "paper")
    os.makedirs(paper_dir, exist_ok=True)
    label_a, label_b = labels["a"], labels["b"]

    processing_dir = os.path.join(input_dir, "processing")
    for mode in PROCESSING_MODES:
        pred_path = os.path.join(
            processing_dir, "predictions_%s.csv.bz2" % mode)
        if not os.path.isfile(pred_path):
            continue
        df = pandas.read_csv(pred_path)
        a_score = _score_values(df, "a", "processing_score")
        b_score = _score_values(df, "b", "processing_score")
        y = df.hit.values
        _save_roc(plt, roc_curve, roc_auc_score,
                  y, a_score, b_score, label_a, label_b,
                  os.path.join(sub_dir, "roc_%s.png" % mode),
                  title="%s processing ROC" % _display_identifier(mode))
        _save_pr(plt, precision_recall_curve, average_precision_score,
                 y, a_score, b_score, label_a, label_b,
                 os.path.join(sub_dir, "pr_%s.png" % mode),
                 title="%s processing PR" % _display_identifier(mode))
        _save_scatter(plt, b_score, a_score, label_b, label_a,
                      os.path.join(sub_dir, "scatter_%s.png" % mode),
                      title="%s processing: %s vs %s" % (
                          _display_identifier(mode), label_a, label_b),
                      max_points=max_scatter_points)

    summary_table_path = os.path.join(processing_dir, "summary_table.csv")
    if os.path.isfile(summary_table_path):
        summary = _read_optional_csv(summary_table_path)
        _safe_plot(
            "processing macro bars",
            _save_macro_bars, plt, summary, sub_dir, label_a, label_b,
            "Processing")
    _save_component_paper_plots(
        plt,
        processing_dir,
        paper_dir,
        "processing",
        PROCESSING_MODES,
        "processing_score",
        label_a,
        label_b,
        title_prefix="Processing score",
    )


# ---------------------------------------------------------------------------
# presentation
# ---------------------------------------------------------------------------


def _plot_presentation(input_dir, plot_dir, labels, max_scatter_points):
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        average_precision_score, precision_recall_curve,
        roc_auc_score, roc_curve,
    )

    sub_dir = os.path.join(plot_dir, "presentation")
    os.makedirs(sub_dir, exist_ok=True)
    paper_dir = os.path.join(plot_dir, "paper")
    os.makedirs(paper_dir, exist_ok=True)
    label_a, label_b = labels["a"], labels["b"]

    presentation_dir = os.path.join(input_dir, "presentation")
    for mode in PRESENTATION_MODES:
        pred_path = os.path.join(
            presentation_dir, "predictions_%s.csv.bz2" % mode)
        if not os.path.isfile(pred_path):
            continue
        df = pandas.read_csv(pred_path)
        for score_kind in PRESENTATION_SCORE_KINDS:
            a_score = _score_values(df, "a", score_kind)
            b_score = _score_values(df, "b", score_kind)
            y = df.hit.values
            stub = "%s_%s" % (mode, score_kind)
            _save_roc(plt, roc_curve, roc_auc_score,
                      y, a_score, b_score, label_a, label_b,
                      os.path.join(sub_dir, "roc_%s.png" % stub),
                      title="%s ROC (%s)" % (
                          _display_identifier(mode),
                          _display_score_kind(score_kind)))
            _save_pr(plt, precision_recall_curve, average_precision_score,
                     y, a_score, b_score, label_a, label_b,
                     os.path.join(sub_dir, "pr_%s.png" % stub),
                     title="%s PR (%s)" % (
                         _display_identifier(mode),
                         _display_score_kind(score_kind)))
            _save_scatter(plt, b_score, a_score, label_b, label_a,
                          os.path.join(sub_dir, "scatter_%s.png" % stub),
                          title="%s (%s): %s vs %s" % (
                              _display_identifier(mode),
                              _display_score_kind(score_kind),
                              label_a,
                              label_b),
                          max_points=max_scatter_points)

    summary_table_path = os.path.join(presentation_dir, "summary_table.csv")
    if os.path.isfile(summary_table_path):
        summary = _read_optional_csv(summary_table_path)
        _safe_plot(
            "presentation macro bars",
            _save_macro_bars, plt, summary, sub_dir, label_a, label_b,
            "Presentation")
    for score_kind in PRESENTATION_SCORE_KINDS:
        suffix = "" if score_kind == "presentation_score" else "_%s" % score_kind
        _save_component_paper_plots(
            plt,
            presentation_dir,
            paper_dir,
            "presentation",
            PRESENTATION_MODES,
            score_kind,
            label_a,
            label_b,
            name_suffix=suffix,
            title_prefix=_score_kind_title(score_kind),
        )


def _score_values(df, prefix, score_kind):
    """Higher = better; mirror the convention from compare_models."""
    if score_kind == "presentation_score":
        return df["%s_presentation_score" % prefix].values
    if score_kind == "presentation_percentile":
        return -df["%s_presentation_percentile" % prefix].values
    if score_kind == "processing_score":
        return df["%s_processing_score" % prefix].values
    raise ValueError("Unknown score kind: %s" % score_kind)


def _plot_release_summary(input_dir, paper_dir, labels):
    import matplotlib.pyplot as plt

    summary_path = os.path.join(input_dir, "release_summary.csv")
    if not os.path.isfile(summary_path):
        return

    summary = _read_optional_csv(summary_path)
    if summary.empty:
        return

    required = {"average", "component", "metric", "side_a", "side_b", "diff"}
    if not required.issubset(summary.columns):
        return

    macro = summary.loc[summary.average == "Macro"].copy()
    if macro.empty:
        return

    macro["plot_group"] = macro.apply(_release_summary_group_label, axis=1)
    ordered_groups = list(dict.fromkeys(macro["plot_group"]))

    fig, axes = plt.subplots(
        1, len(METRIC_NAMES),
        figsize=(3.2 * len(METRIC_NAMES), 3.0),
        squeeze=False,
    )
    for ax, metric in zip(axes[0], METRIC_NAMES):
        rows = macro.loc[macro.metric == METRIC_DISPLAY_NAMES[metric]]
        rows = (
            rows.groupby("plot_group", as_index=False)[
                ["side_a", "side_b", "diff"]
            ]
            .mean()
        )
        rows = rows.set_index("plot_group").reindex(ordered_groups)
        x = numpy.arange(len(rows))
        width = 0.36
        ax.bar(
            x - width / 2, rows.side_a, width, label=labels["a"],
            color=SIDE_A_COLOR, edgecolor="white", linewidth=0.5)
        ax.bar(
            x + width / 2, rows.side_b, width, label=labels["b"],
            color=SIDE_B_COLOR, edgecolor="white", linewidth=0.5)
        ax.set_title(METRIC_DISPLAY_NAMES[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(rows.index, rotation=35, ha="right")
        ax.set_ylim(_metric_ylim(metric, rows[["side_a", "side_b"]].values))
        ax.grid(axis="y")
    axes[0][0].set_ylabel("Macro mean")
    axes[0][-1].legend(frameon=False)
    fig.suptitle("Model comparison: macro accuracy by component")
    fig.tight_layout()
    _save_figure(fig, os.path.join(paper_dir, "release_summary_macro.png"))
    plt.close(fig)

    fig, axes = plt.subplots(
        1, len(METRIC_NAMES),
        figsize=(3.2 * len(METRIC_NAMES), 3.0),
        squeeze=False,
    )
    for ax, metric in zip(axes[0], METRIC_NAMES):
        rows = macro.loc[macro.metric == METRIC_DISPLAY_NAMES[metric]]
        rows = (
            rows.groupby("plot_group", as_index=False)[
                ["side_a", "side_b", "diff"]
            ]
            .mean()
        )
        rows = rows.set_index("plot_group").reindex(ordered_groups)
        values = rows["diff"].values
        ax.bar(
            numpy.arange(len(rows)), values, color=_delta_colors(values),
            edgecolor="white", linewidth=0.5)
        ax.axhline(0, color=DIAGONAL_COLOR, linewidth=0.8)
        ax.set_title(METRIC_DISPLAY_NAMES[metric])
        ax.set_xticks(numpy.arange(len(rows)))
        ax.set_xticklabels(rows.index, rotation=35, ha="right")
        ax.set_ylabel("%s - %s" % (labels["a"], labels["b"]))
        ax.grid(axis="y")
    fig.suptitle("Model comparison: macro deltas by component")
    fig.tight_layout()
    _save_figure(fig, os.path.join(paper_dir, "release_summary_macro_delta.png"))
    plt.close(fig)


def _release_summary_group_label(row):
    component = row.get("component", "")
    mode = row.get("flank_mode", "")
    if isinstance(mode, str) and mode:
        return "%s\n%s" % (
            _display_identifier(component),
            _display_identifier(mode))
    return _display_identifier(row.get("eval", component))


def _save_component_paper_plots(
        plt, component_dir, paper_dir, component, modes, score_kind,
        label_a, label_b, name_suffix="", title_prefix=None):
    if title_prefix is None:
        title_prefix = component.title()

    sample_frames = []
    length_frames = []
    for mode in modes:
        sample_path = os.path.join(
            component_dir, "per_sample_%s_%s.csv" % (mode, score_kind))
        if os.path.isfile(sample_path):
            sample = _read_optional_csv(sample_path)
            if not sample.empty:
                sample_frames.append((mode, sample))

        length_path = os.path.join(
            component_dir, "per_length_%s_%s.csv" % (mode, score_kind))
        if os.path.isfile(length_path):
            length_df = _read_optional_csv(length_path)
            if not length_df.empty:
                length_frames.append((mode, length_df))

    if sample_frames:
        _save_metric_scatter_grid(
            plt,
            sample_frames,
            os.path.join(
                paper_dir,
                "%s_per_sample_scatter%s.png" % (component, name_suffix)),
            label_a,
            label_b,
            "%s per-sample accuracy" % title_prefix,
        )
        _save_metric_delta_boxplots(
            plt,
            sample_frames,
            os.path.join(
                paper_dir,
                "%s_per_sample_delta_boxplots%s.png" % (
                    component, name_suffix)),
            "%s per-sample deltas" % title_prefix,
            label_a,
            label_b,
        )

    if length_frames:
        _save_per_length_grid(
            plt,
            length_frames,
            os.path.join(
                paper_dir,
                "%s_per_length_macro%s.png" % (component, name_suffix)),
            label_a,
            label_b,
            "%s by peptide length" % title_prefix,
        )


def _save_metric_scatter_grid(
        plt, frames, out_path, label_a, label_b, title, average=None):
    n_rows = len(frames)
    n_cols = len(METRIC_NAMES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 2.7 * n_rows),
        squeeze=False,
    )
    for row_idx, (row_label, df) in enumerate(frames):
        for col_idx, metric in enumerate(METRIC_NAMES):
            ax = axes[row_idx][col_idx]
            a_col, b_col = _metric_columns(metric, average=average)
            if a_col not in df.columns or b_col not in df.columns:
                ax.axis("off")
                continue
            x = df[b_col].to_numpy(dtype=float)
            y = df[a_col].to_numpy(dtype=float)
            mask = numpy.isfinite(x) & numpy.isfinite(y)
            if not mask.any():
                ax.axis("off")
                continue
            x = x[mask]
            y = y[mask]
            colors = _winner_colors(x, y)
            ax.scatter(
                x, y, s=13, alpha=0.72, linewidths=0.15,
                edgecolors="white", c=colors)
            low, high = _metric_ylim(metric, numpy.concatenate([x, y]))
            ax.plot([low, high], [low, high], color=DIAGONAL_COLOR, linewidth=0.8)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.grid(False)
            ax.set_title(METRIC_DISPLAY_NAMES[metric])
            row_display = _display_identifier(row_label)
            if col_idx == 0:
                ax.set_ylabel("%s\n%s" % (row_display, label_a))
            else:
                ax.set_ylabel(label_a)
            ax.set_xlabel(label_b)
            ax.text(
                0.04, 0.96,
                "n=%d\nmean Δ=%+.3f" % (len(x), numpy.nanmean(y - x)),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.85"},
            )
    fig.suptitle(title)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def _save_metric_delta_boxplots(
        plt, frames, out_path, title, label_a, label_b):
    fig, axes = plt.subplots(
        1, len(METRIC_NAMES),
        figsize=(3.0 * len(METRIC_NAMES), 3.0),
        squeeze=False,
    )
    labels = [_display_identifier(label) for (label, _) in frames]
    for ax, metric in zip(axes[0], METRIC_NAMES):
        a_col, b_col = _metric_columns(metric)
        data = []
        for _, df in frames:
            if a_col not in df.columns or b_col not in df.columns:
                data.append([])
                continue
            values = (df[a_col] - df[b_col]).dropna().values
            data.append(values)
        try:
            artists = ax.boxplot(
                data, tick_labels=labels, showfliers=False,
                patch_artist=True)
        except TypeError:
            artists = ax.boxplot(
                data, labels=labels, showfliers=False,
                patch_artist=True)
        _style_boxplot(artists)
        ax.axhline(0, color=DIAGONAL_COLOR, linewidth=0.8)
        ax.set_title(METRIC_DISPLAY_NAMES[metric])
        ax.set_ylabel("%s - %s" % (label_a, label_b))
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y")
    fig.suptitle(title)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def _save_per_length_grid(
        plt, frames, out_path, label_a, label_b, title):
    n_rows = len(frames)
    n_cols = len(METRIC_NAMES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 2.6 * n_rows),
        squeeze=False,
    )
    legend_handles = None
    legend_labels = None
    for row_idx, (row_label, df) in enumerate(frames):
        if "length" not in df.columns:
            for ax in axes[row_idx]:
                ax.axis("off")
            continue
        x = numpy.arange(len(df))
        lengths = [str(v) for v in df["length"]]
        width = 0.36
        for col_idx, metric in enumerate(METRIC_NAMES):
            ax = axes[row_idx][col_idx]
            a_col, b_col = _metric_columns(metric, average="macro")
            if a_col not in df.columns or b_col not in df.columns:
                ax.axis("off")
                continue
            ax.bar(
                x - width / 2, df[a_col], width, label=label_a,
                color=SIDE_A_COLOR, edgecolor="white", linewidth=0.5)
            ax.bar(
                x + width / 2, df[b_col], width, label=label_b,
                color=SIDE_B_COLOR, edgecolor="white", linewidth=0.5)
            ax.set_title(METRIC_DISPLAY_NAMES[metric])
            ax.set_xticks(x)
            ax.set_xticklabels(lengths)
            ax.set_xlabel("Peptide length")
            ax.set_ylim(_metric_ylim(metric, df[[a_col, b_col]].values))
            ax.grid(axis="y")
            if col_idx == 0:
                ax.set_ylabel("%s\nMacro mean" % (
                    _display_identifier(row_label)))
            else:
                ax.set_ylabel("Macro mean")
            if row_idx == 0 and col_idx == n_cols - 1:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=len(legend_labels),
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.96))
    _save_figure(fig, out_path)
    plt.close(fig)


def _metric_columns(metric, average=None):
    if average is None:
        return "a_%s" % metric, "b_%s" % metric
    return "a_%s_%s" % (average, metric), "b_%s_%s" % (average, metric)


def _winner_colors(x_values, y_values):
    return [
        SIDE_A_COLOR if y_value >= x_value else SIDE_B_COLOR
        for x_value, y_value in zip(x_values, y_values)
    ]


def _delta_colors(values):
    values = numpy.asarray(values, dtype=float)
    return [
        POSITIVE_DELTA_COLOR if value >= 0 else NEGATIVE_DELTA_COLOR
        for value in values
    ]


def _style_boxplot(artists):
    for patch in artists.get("boxes", []):
        patch.set_facecolor(SIDE_A_COLOR)
        patch.set_alpha(0.35)
        patch.set_edgecolor("0.30")
        patch.set_linewidth(0.8)
    for median in artists.get("medians", []):
        median.set_color("0.15")
        median.set_linewidth(1.1)
    for key in ("whiskers", "caps"):
        for artist in artists.get(key, []):
            artist.set_color("0.35")
            artist.set_linewidth(0.8)


def _metric_ylim(metric, values):
    values = numpy.asarray(values, dtype=float)
    values = values[numpy.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    if metric == "roc_auc":
        low = min(0.5, max(0.0, float(values.min()) - 0.03))
    else:
        low = 0.0
    high = min(1.0, max(0.1, float(values.max()) + 0.04))
    return low, high


def _write_summary_pdf(
        plot_dir, out_path, include_paper_figures=False,
        paper_figures_dir=None):
    plot_dir = Path(plot_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdfs = _summary_pdf_paths(
        plot_dir, out_path, include_paper_figures, paper_figures_dir)
    if pdfs:
        try:
            try:
                pdf_reader, pdf_writer = _pdf_reader_writer()
            except ImportError:
                _write_summary_pdf_from_pngs(
                    plot_dir, out_path, include_paper_figures,
                    paper_figures_dir)
                return
            writer = pdf_writer()
            for path in pdfs:
                reader = pdf_reader(str(path))
                for page in reader.pages:
                    writer.add_page(page)
            with open(out_path, "wb") as fd:
                writer.write(fd)
            return
        except Exception as e:
            print(
                "WARNING: PDF merge failed; rebuilding summary from PNGs: "
                "%s: %s" % (type(e).__name__, e))
            try:
                out_path.unlink()
            except FileNotFoundError:
                pass
            _write_summary_pdf_from_pngs(
                plot_dir, out_path, include_paper_figures, paper_figures_dir)
            return
    _write_summary_pdf_from_pngs(
        plot_dir, out_path, include_paper_figures, paper_figures_dir)


def _summary_pdf_paths(
        plot_dir, out_path, include_paper_figures=False,
        paper_figures_dir=None):
    result = []
    paper_pdfs = []
    for path in sorted(Path(plot_dir).rglob("*.pdf")):
        if _include_pdf_in_summary(
                path, plot_dir, out_path, include_paper_figures,
                paper_figures_dir):
            if _is_combined_paper_figures_pdf(
                    path, plot_dir, paper_figures_dir):
                _append_unique_path(paper_pdfs, path)
            else:
                _append_unique_path(result, path)
    if include_paper_figures:
        for paper_dir in _paper_figure_dirs(plot_dir, paper_figures_dir):
            paper_pdf = Path(paper_dir) / "paper_figures.pdf"
            if paper_pdf.is_file():
                _append_unique_path(paper_pdfs, paper_pdf)
    return paper_pdfs + result


def _append_unique_path(paths, path, prepend=False):
    path = Path(path)
    resolved = path.resolve()
    if any(existing.resolve() == resolved for existing in paths):
        return
    if prepend:
        paths.insert(0, path)
    else:
        paths.append(path)


def _paper_figure_dirs(plot_dir, paper_figures_dir=None):
    result = []
    if paper_figures_dir:
        result.append(Path(paper_figures_dir))
    plot_dir = Path(plot_dir)
    result.extend([plot_dir / "paper_figures", plot_dir / "paper_2023"])
    unique = []
    for path in result:
        resolved = path.resolve()
        if not any(existing.resolve() == resolved for existing in unique):
            unique.append(path)
    return tuple(unique)


def _is_combined_paper_figures_pdf(path, plot_dir, paper_figures_dir=None):
    path = Path(path)
    if path.name != "paper_figures.pdf":
        return False
    return any(
        path.parent.resolve() == Path(paper_dir).resolve()
        for paper_dir in _paper_figure_dirs(plot_dir, paper_figures_dir)
    )


def _path_is_relative_to(path, directory):
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
        return True
    except ValueError:
        return False


def _include_pdf_in_summary(
        path, plot_dir, out_path, include_paper_figures,
        paper_figures_dir=None):
    path = Path(path)
    plot_dir = Path(plot_dir)
    out_path = Path(out_path)
    if path.resolve() == out_path.resolve():
        return False
    for paper_dir in _paper_figure_dirs(plot_dir, paper_figures_dir):
        if _path_is_relative_to(path, paper_dir):
            return (
                include_paper_figures
                and path.name == "paper_figures.pdf"
                and path.parent.resolve() == Path(paper_dir).resolve()
            )
    try:
        relative_parts = path.relative_to(plot_dir).parts
    except ValueError:
        return False
    return len(relative_parts) > 1


def _pdf_reader_writer():
    try:
        from pypdf import PdfReader, PdfWriter
        return PdfReader, PdfWriter
    except ImportError:
        try:
            from PyPDF2 import PdfReader, PdfWriter
            return PdfReader, PdfWriter
        except ImportError as e:
            raise ImportError("pypdf or PyPDF2 is required") from e


def _include_png_in_summary(
        path, plot_dir, out_path, include_paper_figures,
        paper_figures_dir=None):
    path = Path(path)
    plot_dir = Path(plot_dir)
    out_path = Path(out_path)
    if path.resolve() == out_path.resolve():
        return False
    for paper_dir in _paper_figure_dirs(plot_dir, paper_figures_dir):
        if _path_is_relative_to(path, paper_dir):
            return include_paper_figures
    try:
        relative_parts = path.relative_to(plot_dir).parts
    except ValueError:
        return False
    return len(relative_parts) > 1


def _write_summary_pdf_from_pngs(
        plot_dir, out_path, include_paper_figures=False,
        paper_figures_dir=None):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plot_dir = Path(plot_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pngs = []
    if include_paper_figures:
        for paper_dir in _paper_figure_dirs(plot_dir, paper_figures_dir):
            for path in sorted(Path(paper_dir).rglob("*.png")):
                if _include_png_in_summary(
                        path, plot_dir, out_path, include_paper_figures,
                        paper_figures_dir):
                    _append_unique_path(pngs, path)
    for path in sorted(plot_dir.rglob("*.png")):
        if _include_png_in_summary(
                path, plot_dir, out_path, include_paper_figures,
                paper_figures_dir):
            _append_unique_path(pngs, path)
    if not pngs:
        return

    paper = [path for path in pngs if "paper" in path.parts]
    rest = [path for path in pngs if path not in paper]
    ordered = paper + rest
    with PdfPages(out_path) as pdf:
        for path in ordered:
            image = plt.imread(path)
            fig = plt.figure(figsize=(11.0, 8.5))
            ax = fig.add_axes([0.04, 0.06, 0.92, 0.84])
            ax.imshow(image)
            ax.axis("off")
            try:
                title = str(path.relative_to(plot_dir))
            except ValueError:
                title = str(path)
            fig.suptitle(title, fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)


def _save_figure(fig, out_path):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def _save_macro_bars(plt, summary, sub_dir, label_a, label_b, title_prefix):
    if summary.empty:
        return
    required = {"mode", "score_kind"}
    for metric in METRIC_NAMES:
        required.add("a_macro_%s" % metric)
        required.add("b_macro_%s" % metric)
    if not required.issubset(summary.columns):
        return
    x_labels = [
        "%s\n%s" % (
            _display_identifier(row.mode),
            _display_score_kind(row.score_kind),
        )
        for row in summary.itertuples()
    ]
    x = numpy.arange(len(summary))
    width = 0.38
    for metric in METRIC_NAMES:
        fig, ax = plt.subplots(figsize=(7.1, 3.0))
        ax.bar(
            x - width / 2,
            summary["a_macro_%s" % metric],
            width,
            label=label_a,
            color=SIDE_A_COLOR,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.bar(
            x + width / 2,
            summary["b_macro_%s" % metric],
            width,
            label=label_b,
            color=SIDE_B_COLOR,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right")
        ax.set_ylabel(METRIC_DISPLAY_NAMES[metric])
        ax.set_title("%s: macro %s over samples" % (
            title_prefix, METRIC_DISPLAY_NAMES[metric]))
        ax.set_ylim(_metric_ylim(
            metric,
            summary[["a_macro_%s" % metric, "b_macro_%s" % metric]].values,
        ))
        ax.grid(axis="y")
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()
        _save_figure(fig, os.path.join(sub_dir, "macro_%s.png" % metric))
        plt.close(fig)


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------


def _save_roc(plt, roc_curve_fn, roc_auc_fn,
              y, a_score, b_score, label_a, label_b, out_path, title):
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    y, a_score, b_score = _shared_finite_curve_values(y, a_score, b_score)
    for label, values, color in (
            (label_a, a_score, SIDE_A_COLOR),
            (label_b, b_score, SIDE_B_COLOR)):
        if len(y) == 0 or len(numpy.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve_fn(y, values)
        auc = roc_auc_fn(y, values)
        ax.plot(
            fpr, tpr, label="%s AUC=%.3f" % (label, auc),
            color=color, linewidth=1.6)
    ax.plot([0, 1], [0, 1], color=DIAGONAL_COLOR, linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    ax.grid(False)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def _save_pr(plt, pr_curve_fn, ap_fn,
             y, a_score, b_score, label_a, label_b, out_path, title):
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    y, a_score, b_score = _shared_finite_curve_values(y, a_score, b_score)
    for label, values, color in (
            (label_a, a_score, SIDE_A_COLOR),
            (label_b, b_score, SIDE_B_COLOR)):
        if len(y) == 0 or len(numpy.unique(y)) < 2:
            continue
        precision, recall, _ = pr_curve_fn(y, values)
        ap = ap_fn(y, values)
        ax.plot(
            recall, precision, label="%s AP=%.3f" % (label, ap),
            color=color, linewidth=1.6)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    ax.grid(False)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def _shared_finite_curve_values(y, a_score, b_score):
    """Return paired finite inputs for an A/B diagnostic curve."""
    y = numpy.asarray(y)
    a_score = numpy.asarray(a_score, dtype=float)
    b_score = numpy.asarray(b_score, dtype=float)
    mask = numpy.isfinite(y) & numpy.isfinite(a_score) & numpy.isfinite(b_score)
    return y[mask], a_score[mask], b_score[mask]


def _save_scatter(plt, x_score, y_score, x_label, y_label,
                  out_path, title, max_points):
    mask = numpy.isfinite(x_score) & numpy.isfinite(y_score)
    idx = numpy.flatnonzero(mask)
    if len(idx) == 0:
        return
    if len(idx) > max_points:
        rng = numpy.random.default_rng(17)
        idx = rng.choice(idx, size=max_points, replace=False)
    x = x_score[idx]
    y = y_score[idx]
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.scatter(
        x,
        y,
        s=7,
        alpha=0.42,
        c=_winner_colors(x, y),
        edgecolors="none",
        rasterized=False,
    )
    _add_identity_line(ax, x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def _add_identity_line(ax, x, y):
    finite = numpy.concatenate([
        numpy.asarray(x, dtype=float),
        numpy.asarray(y, dtype=float),
    ])
    finite = finite[numpy.isfinite(finite)]
    if finite.size == 0:
        return
    low = float(finite.min())
    high = float(finite.max())
    if high <= low:
        pad = max(abs(high) * 0.05, 0.01)
        low -= pad
        high += pad
    else:
        pad = (high - low) * 0.05
        low -= pad
        high += pad
    ax.plot([low, high], [low, high], color=DIAGONAL_COLOR, linewidth=0.8)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")


def _display_identifier(value):
    return str(value).replace("_", " ")


def _display_score_kind(value):
    display = {
        "presentation_score": "presentation score",
        "presentation_percentile": "presentation percentile",
        "processing_score": "processing score",
    }
    return display.get(str(value), _display_identifier(value))


def _score_kind_title(value):
    display = {
        "presentation_score": "Presentation score",
        "presentation_percentile": "Presentation percentile rank",
        "processing_score": "Processing score",
    }
    return display.get(str(value), _display_score_kind(value).title())


# Module-level parser for sphinx autoprogram; behaves like the legacy
# ``mhcflurry-*`` command modules.
parser = make_parser()
