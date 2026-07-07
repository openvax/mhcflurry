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

"""Generate paper-style figures from retraining/evaluation artifacts.

This command ports the figure families from the 2023 retraining notebooks
into a reproducible CLI. It is intentionally artifact-driven: each panel is
generated only when the input table that supported the notebook version is
present. Missing inputs are written to ``missing_inputs.md`` and
``manifest.csv`` so a training run can distinguish "not generated because the
data is absent" from "plotting silently drifted."
"""
from __future__ import annotations

import argparse
import ast
import os
import shutil
from pathlib import Path

import numpy
import pandas


EXTERNAL_BASELINES = (
    ("netmhcpan4.ba", "ba"),
    ("netmhcpan4.el", "el"),
    ("mixmhcpred", "mixmhcpred"),
)

PRESENTATION_PANEL_PREDICTORS = (
    "presentation_without_flanks_presentation_score",
    "presentation_with_flanks_presentation_score",
)

PRESENTATION_PANEL_BASELINES = (
    "netmhcpan4.ba",
    "netmhcpan4.el",
    "mixmhcpred",
    "mhcflurry_production",
)

PREFERRED_PREDICTORS = (
    "netmhcpan4.ba",
    "netmhcpan4.el",
    "mixmhcpred",
    "mhcflurry_production",
    "presentation_without_flanks_presentation_score",
    "presentation_with_flanks_presentation_score",
    "presentation_without_flanks_processing_score",
    "presentation_with_flanks_processing_score",
)

LENGTH_LABEL_ORDER = ("All", "8-mer", "9-mer", "10-mer", "11-mer")
DEFAULT_FORMATS = ("svg", "pdf", "png")


def make_parser():
    """Return a standalone parser for documentation tooling."""
    parser = argparse.ArgumentParser(prog="mhcflurry paper-figures")
    register_subparser(parser)
    return parser


def run_argv(argv):
    """Entry point for the lazy ``mhcflurry paper-figures`` dispatcher."""
    return run(make_parser().parse_args(argv))


def register_subparser(parser):
    parser.description = __doc__
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        help=(
            "Directory containing 2023-style retraining artifacts such as "
            "accuracy_scores.multiallelic.csv and predictor_info.csv."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for paper figures and manifest files.",
    )
    parser.add_argument(
        "--formats",
        default=",".join(DEFAULT_FORMATS),
        help=(
            "Comma-separated output formats. Default: %(default)s. Use SVG/PDF "
            "for publication; PNG is for quick review."
        ),
    )
    parser.add_argument(
        "--combined-pdf",
        default=None,
        help=(
            "Optional multi-page PDF path. Default: <out>/paper_figures.pdf. "
            "Pass 'none' to skip."
        ),
    )
    parser.add_argument(
        "--sample-table",
        help=(
            "Optional sample table with sample_id and sample_group columns. "
            "When present, panels that were recent-sample-only in the 2023 "
            "notebooks use --sample-group."
        ),
    )
    parser.add_argument(
        "--sample-group",
        default="MULTIALLELIC-RECENT",
        help="Sample group for recent-only multiallelic panels.",
    )
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=20_000,
        help="Subsample scatter plots above this many points.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Return a non-zero exit code if any requested figure family skips.",
    )
    return parser


def run(args):
    import matplotlib
    matplotlib.use("Agg")

    artifacts_dir = Path(args.artifacts_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = _parse_formats(args.formats)
    combined_pdf = _combined_pdf_path(args.combined_pdf, out_dir)

    _apply_paper_style()
    writer = FigureWriter(out_dir, formats, combined_pdf)
    try:
        predictor_info = _read_predictor_info(
            artifacts_dir / "predictor_info.csv", writer)
        sample_ids = _read_sample_group_ids(args, artifacts_dir, writer)
        _generate_multiallelic_figures(
            artifacts_dir, predictor_info, sample_ids, args.sample_group,
            args.max_scatter_points, writer)
        _generate_model_selection_figures(artifacts_dir, writer)
        _generate_monoallelic_figures(
            artifacts_dir, predictor_info, args.max_scatter_points, writer)
        _generate_processing_notebook_figures(artifacts_dir, predictor_info, writer)
        _generate_proteasome_figures(artifacts_dir, writer)
        _copy_architecture_figures(artifacts_dir, writer)
    finally:
        writer.close()

    _write_manifest(out_dir, writer.rows)
    _write_missing_inputs(out_dir, writer.rows)
    if args.strict and any(row["status"] == "skipped" for row in writer.rows):
        return 2
    return 0


class FigureWriter:
    """Write one plot to SVG/PDF/PNG and track the output manifest."""

    def __init__(self, out_dir, formats, combined_pdf):
        self.out_dir = Path(out_dir)
        self.formats = tuple(formats)
        self.combined_pdf = Path(combined_pdf) if combined_pdf else None
        self.pdf_pages = None
        self.rows = []

    def save(self, fig, name, family, note=""):
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        paths = []
        for fmt in self.formats:
            fmt_dir = self.out_dir / fmt
            fmt_dir.mkdir(parents=True, exist_ok=True)
            path = fmt_dir / ("%s.%s" % (name, fmt))
            fig.savefig(path, transparent=True, bbox_inches="tight")
            paths.append(str(path))
        if self.combined_pdf:
            self.combined_pdf.parent.mkdir(parents=True, exist_ok=True)
            if self.pdf_pages is None:
                self.pdf_pages = PdfPages(self.combined_pdf)
            self.pdf_pages.savefig(fig, transparent=True, bbox_inches="tight")
        plt.close(fig)
        self.rows.append({
            "family": family,
            "figure": name,
            "status": "generated",
            "paths": ";".join(paths),
            "note": note,
            "missing": "",
        })

    def skip(self, family, figure, missing, note):
        if isinstance(missing, (list, tuple)):
            missing = ";".join(str(item) for item in missing)
        self.rows.append({
            "family": family,
            "figure": figure,
            "status": "skipped",
            "paths": "",
            "note": note,
            "missing": str(missing),
        })

    def close(self):
        if self.pdf_pages is not None:
            self.pdf_pages.close()


def _parse_formats(value):
    formats = tuple(
        part.strip().lower() for part in value.split(",") if part.strip())
    if not formats:
        raise ValueError("--formats must contain at least one format")
    allowed = {"svg", "pdf", "png"}
    unknown = set(formats) - allowed
    if unknown:
        raise ValueError(
            "Unsupported figure formats: %s. Allowed: %s" % (
                ", ".join(sorted(unknown)), ", ".join(sorted(allowed))))
    return formats


def _combined_pdf_path(value, out_dir):
    if value == "none":
        return None
    if value:
        return value
    return str(Path(out_dir) / "paper_figures.pdf")


def _apply_paper_style():
    import matplotlib.pyplot as plt

    try:
        import seaborn
        seaborn.set_context("paper")
        seaborn.set_style("white")
    except ImportError:
        pass
    try:
        plt.style.use("seaborn-v0_8-white")
    except OSError:
        try:
            plt.style.use("seaborn-white")
        except OSError:
            pass
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.usetex": False,
    })


def _read_predictor_info(path, writer):
    if not path.is_file():
        writer.skip(
            "metadata", "predictor_info",
            [path],
            "Predictor labels/colors unavailable; using fallback labels.")
        return pandas.DataFrame(columns=[
            "predictor", "description", "primary", "color", "short", "detail"])
    df = pandas.read_csv(path)
    if "predictor" not in df.columns:
        writer.skip(
            "metadata", "predictor_info",
            [path],
            "predictor_info.csv lacks a predictor column; using fallback labels.")
        return pandas.DataFrame(columns=[
            "predictor", "description", "primary", "color", "short", "detail"])
    return df.set_index("predictor", drop=False)


def _read_sample_group_ids(args, artifacts_dir, writer):
    path = Path(args.sample_table) if args.sample_table else (
        artifacts_dir / "sample_table.csv")
    if not path.is_file():
        writer.skip(
            "sample-groups", args.sample_group,
            [path],
            "Sample-group table absent; recent-only panels use all samples.")
        return None
    df = pandas.read_csv(path)
    sample_col = _first_present(df, ("sample_id", "sample", "id"))
    group_col = _first_present(df, ("sample_group", "group", "category"))
    if sample_col is None or group_col is None:
        writer.skip(
            "sample-groups", args.sample_group,
            [path],
            "Sample table must contain sample_id and sample_group columns.")
        return None
    result = set(df.loc[df[group_col] == args.sample_group, sample_col])
    if not result:
        writer.skip(
            "sample-groups", args.sample_group,
            [path],
            "No rows found for sample group %s." % args.sample_group)
        return None
    return result


def _first_present(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def _generate_multiallelic_figures(
        artifacts_dir, predictor_info, recent_sample_ids, sample_group,
        max_scatter_points, writer):
    path = artifacts_dir / "accuracy_scores.multiallelic.csv"
    if not path.is_file():
        writer.skip(
            "multiallelic", "all",
            [path],
            "Multiallelic benchmark scores are required.")
        return

    scores = _normalize_score_predictors(pandas.read_csv(path))
    required = {
        "sample_id", "length_label", "predictor", "auc", "ppv",
        "percent_change_auc_ba", "percent_change_ppv_ba",
        "percent_change_auc_el", "percent_change_ppv_el",
        "percent_change_auc_mixmhcpred", "percent_change_ppv_mixmhcpred",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        writer.skip(
            "multiallelic", "all",
            [path],
            "Missing required columns: %s" % ", ".join(missing))
        return

    recent_note = (
        "Sample table not supplied; using all samples instead of %s." %
        sample_group if recent_sample_ids is None else
        "Restricted to sample group %s." % sample_group
    )

    _plot_external_scatter_triptych(
        scores, predictor_info, "auc", "AUC",
        "fig.3_scores_plots_multiallelic.scatter.auc.ba",
        max_scatter_points, writer)
    _plot_external_scatter_triptych(
        scores, predictor_info, "ppv", "PPV",
        "fig.3_scores_plots_multiallelic.scatter.ppv.ba",
        max_scatter_points, writer)
    _plot_percent_change_by_length(
        scores, predictor_info, "auc", "AUC",
        "fig.3_scores_plots_multiallelic.bar_by_peptide_length.auc.ba",
        writer)
    _plot_percent_change_bars(
        scores, predictor_info, "auc", "AUC",
        "fig.3_scores_plots_multiallelic.bar.auc.presentation",
        writer)
    _plot_percent_change_bars(
        scores, predictor_info, "ppv", "PPV",
        "fig.3_scores_plots_multiallelic.bar.ppv.presentation",
        writer)
    _plot_mean_ppv_small(
        scores, predictor_info, recent_sample_ids, recent_note,
        "fig.3_scores_plots_multiallelic.mean_ppv_small_plot",
        writer)
    _plot_presentation_scatter_grid(
        scores, predictor_info, recent_sample_ids, recent_note,
        max_scatter_points,
        "fig.3_scores_plots_multiallelic.scatter.ppv.presentation",
        writer)
    _plot_graphical_abstract_logistic_regression(
        predictor_info,
        "fig.3_scores_plots_multiallelic.graphical_abstract_logistic_regression",
        writer)


def _plot_external_scatter_triptych(
        scores, predictor_info, metric, metric_label, name, max_points, writer):
    import matplotlib.pyplot as plt

    candidate = "mhcflurry_production"
    pivot = _pivot_all_lengths(scores, metric)
    needed = [candidate] + [predictor for predictor, _ in EXTERNAL_BASELINES]
    missing = [predictor for predictor in needed if predictor not in pivot.columns]
    if missing:
        writer.skip(
            "multiallelic", name, missing,
            "Required predictors absent from multiallelic scores.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2))
    y_label = _short_label(predictor_info, candidate)
    for ax, (baseline, _suffix) in zip(axes, EXTERNAL_BASELINES):
        sub = pivot[[baseline, candidate]].replace(
            [numpy.inf, -numpy.inf], numpy.nan).dropna()
        _scatter_with_winner_colors(
            ax, sub[baseline].values, sub[candidate].values,
            baseline, candidate, predictor_info, max_points)
        _finish_metric_scatter(
            ax,
            _short_label(predictor_info, baseline),
            y_label,
            "%s vs %s" % (
                y_label, _short_label(predictor_info, baseline)),
            metric_label)
    fig.tight_layout(w_pad=1.0)
    writer.save(fig, name, "multiallelic")


def _plot_percent_change_by_length(
        scores, predictor_info, metric, metric_label, name, writer):
    import matplotlib.pyplot as plt

    predictor = "mhcflurry_production"
    sub = scores.loc[scores["predictor"] == predictor].copy()
    if sub.empty:
        writer.skip(
            "multiallelic", name, [predictor],
            "MHCflurry production scores absent.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.1), sharey=True)
    color = _predictor_color(predictor_info, predictor)
    for ax, (baseline, suffix) in zip(axes, EXTERNAL_BASELINES):
        column = "percent_change_%s_%s" % (metric, suffix)
        if column not in sub.columns:
            ax.set_visible(False)
            continue
        plot_df = (
            sub[["length_label", column]]
            .replace([numpy.inf, -numpy.inf], numpy.nan)
            .dropna()
        )
        values = (
            plot_df.groupby("length_label")[column]
            .mean()
            .reindex(LENGTH_LABEL_ORDER)
            .dropna()
        )
        x = numpy.arange(len(values))
        ax.bar(x, values.values, color=color, edgecolor="white", linewidth=0.6)
        ax.axhline(0, color="0.35", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(values.index, rotation=30, ha="right")
        ax.set_title("vs %s" % _short_label(predictor_info, baseline))
        ax.set_ylabel("%% change in %s" % metric_label)
        _despine(ax)
    fig.tight_layout(w_pad=1.0)
    writer.save(fig, name, "multiallelic")


def _plot_percent_change_bars(
        scores, predictor_info, metric, metric_label, name, writer):
    import matplotlib.pyplot as plt

    sub = _all_length_rows(scores)
    predictors = [
        predictor for predictor in PREFERRED_PREDICTORS
        if predictor in set(sub["predictor"])
    ]
    if not predictors:
        writer.skip(
            "multiallelic", name, list(PREFERRED_PREDICTORS),
            "No preferred predictors found in multiallelic scores.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.3), sharey=True)
    for ax, (baseline, suffix) in zip(axes, EXTERNAL_BASELINES):
        column = "percent_change_%s_%s" % (metric, suffix)
        if column not in sub.columns:
            ax.set_visible(False)
            continue
        means = (
            sub.loc[sub["predictor"].isin(predictors), ["predictor", column]]
            .replace([numpy.inf, -numpy.inf], numpy.nan)
            .dropna()
            .groupby("predictor")[column]
            .mean()
            .reindex(predictors)
            .dropna()
        )
        labels = [_short_label(predictor_info, item) for item in means.index]
        colors = [_predictor_color(predictor_info, item) for item in means.index]
        y = numpy.arange(len(means))
        ax.barh(y, means.values, color=colors, edgecolor="white", linewidth=0.6)
        ax.axvline(0, color="0.35", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title("vs %s" % _short_label(predictor_info, baseline))
        ax.set_xlabel("%% change in %s" % metric_label)
        _despine(ax)
    fig.tight_layout(w_pad=1.0)
    writer.save(fig, name, "multiallelic")


def _plot_mean_ppv_small(
        scores, predictor_info, recent_sample_ids, note, name, writer):
    import matplotlib.pyplot as plt

    sub = _all_length_rows(scores)
    if recent_sample_ids is not None:
        sub = sub.loc[sub["sample_id"].isin(recent_sample_ids)]
    candidates = [
        "mhcflurry_production",
        "presentation_without_flanks_processing_score",
        "presentation_with_flanks_presentation_score",
        "presentation_without_flanks_presentation_score",
    ]
    rows = []
    for predictor in candidates:
        values = sub.loc[sub["predictor"] == predictor, "ppv"].replace(
            [numpy.inf, -numpy.inf], numpy.nan).dropna()
        if len(values):
            rows.append((predictor, values.mean(), _predictor_color(
                predictor_info, predictor)))
    external_values = sub.loc[
        sub["predictor"].isin([p for p, _ in EXTERNAL_BASELINES]), "ppv"
    ].replace([numpy.inf, -numpy.inf], numpy.nan).dropna()
    if len(external_values):
        rows.append(("external_tools", external_values.mean(), (0.45, 0.45, 0.45)))
    if not rows:
        writer.skip(
            "multiallelic", name, candidates,
            "No PPV values available for selected predictors.")
        return

    labels = [
        "External tools" if predictor == "external_tools"
        else _short_label(predictor_info, predictor)
        for predictor, _mean, _color in rows
    ]
    values = [mean for _predictor, mean, _color in rows]
    colors = [color for _predictor, _mean, color in rows]
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    x = numpy.arange(len(values))
    ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean PPV")
    ax.set_ylim(0, max(values) * 1.2 if values else 1.0)
    _despine(ax)
    fig.tight_layout()
    writer.save(fig, name, "multiallelic", note=note)


def _plot_presentation_scatter_grid(
        scores, predictor_info, recent_sample_ids, note, max_points, name, writer):
    import matplotlib.pyplot as plt

    pivot = _pivot_all_lengths(scores, "ppv")
    if recent_sample_ids is not None:
        pivot = pivot.loc[pivot.index.isin(recent_sample_ids)]
    needed = list(PRESENTATION_PANEL_PREDICTORS) + list(PRESENTATION_PANEL_BASELINES)
    missing = [predictor for predictor in needed if predictor not in pivot.columns]
    if missing:
        writer.skip(
            "multiallelic", name, missing,
            "Required presentation-panel predictors absent.")
        return

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9))
    for row_index, candidate in enumerate(PRESENTATION_PANEL_PREDICTORS):
        for col_index, baseline in enumerate(PRESENTATION_PANEL_BASELINES):
            ax = axes[row_index, col_index]
            sub = pivot[[baseline, candidate]].replace(
                [numpy.inf, -numpy.inf], numpy.nan).dropna()
            _scatter_with_winner_colors(
                ax, sub[baseline].values, sub[candidate].values,
                baseline, candidate, predictor_info, max_points)
            ax.set_title(_short_label(predictor_info, baseline))
            if col_index == 0:
                ax.set_ylabel(_short_label(predictor_info, candidate))
            else:
                ax.set_ylabel("")
            if row_index == len(PRESENTATION_PANEL_PREDICTORS) - 1:
                ax.set_xlabel("Baseline PPV")
            else:
                ax.set_xlabel("")
            _set_unit_limits(ax)
            _despine(ax)
    fig.tight_layout(w_pad=0.9, h_pad=0.9)
    writer.save(fig, name, "multiallelic", note=note)


def _plot_graphical_abstract_logistic_regression(
        predictor_info, name, writer):
    import matplotlib.pyplot as plt

    x = numpy.linspace(-5.0, 5.0, 200)
    y = 1.0 / (1.0 + numpy.exp(-x))
    ba_color = _predictor_color(predictor_info, "mhcflurry_production")
    ap_color = _predictor_color(
        predictor_info, "presentation_with_flanks_processing_score")
    ps_color = _predictor_color(
        predictor_info, "presentation_with_flanks_presentation_score")

    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    ax.plot(x, y, color=ps_color, linewidth=2.0)
    ax.fill_between(x, 0, y, color=ps_color, alpha=0.12, linewidth=0)
    ax.scatter([-2.7, -0.7, 1.8], [0.06, 0.34, 0.86],
               color=[ba_color, ap_color, ps_color],
               s=38, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xlabel("Combined affinity + processing evidence")
    ax.set_ylabel("Presentation probability")
    ax.set_xticks([])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylim(-0.02, 1.02)
    _despine(ax)
    fig.tight_layout()
    writer.save(fig, name, "multiallelic")


def _generate_model_selection_figures(artifacts_dir, writer):
    path = _first_existing(
        artifacts_dir,
        ("model_selection_accuracy.csv", "model_selection_accuracy.xlsx"))
    if path is None:
        writer.skip(
            "model-selection",
            "fig.1_model_selection_predictor_accuracy.scores.by_locus",
            [
                artifacts_dir / "model_selection_accuracy.csv",
                artifacts_dir / "model_selection_accuracy.xlsx",
            ],
            "Model-selection allele-level accuracy table absent.")
        return
    df = pandas.read_excel(path) if path.suffix == ".xlsx" else pandas.read_csv(path)
    allele_col = _first_present(df, ("allele", "hla", "mhc_allele"))
    score_col = _first_present(df, ("auc", "AUC", "score", "accuracy"))
    count_col = _first_present(
        df, ("num_peptides", "peptides", "train_peptides", "train_count"))
    binder_col = _first_present(
        df, ("percent_binders", "binder_percent", "binders_percent"))
    if allele_col is None or score_col is None:
        writer.skip(
            "model-selection",
            "fig.1_model_selection_predictor_accuracy.scores.by_locus",
            [path],
            "Table must contain allele and AUC/score columns.")
        return

    import matplotlib.pyplot as plt

    df = df.copy()
    df["locus"] = df[allele_col].map(_allele_locus)
    for locus, label in (
            ("HLA-A", "hla_a"),
            ("HLA-B", "hla_b"),
            ("HLA-C", "hla_c"),
            ("H2", "h2"),
            ("other", "other")):
        sub = df.loc[df["locus"] == locus].sort_values(score_col)
        if sub.empty:
            writer.skip(
                "model-selection",
                "fig.1_model_selection_predictor_accuracy.scores.%s" % label,
                [path],
                "No rows for locus %s." % locus)
            continue
        fig, axes = plt.subplots(
            1, 3 if count_col and binder_col else 1,
            figsize=(7.1, max(1.7, 0.18 * len(sub) + 0.6)),
            squeeze=False)
        ax = axes[0, 0]
        y = numpy.arange(len(sub))
        ax.barh(y, sub[score_col], color=(0.34, 0.46, 0.75))
        ax.set_yticks(y)
        ax.set_yticklabels(sub[allele_col])
        ax.set_xlabel("AUC")
        ax.set_title(locus)
        _despine(ax)
        if count_col:
            ax = axes[0, 1]
            ax.barh(y, sub[count_col], color=(0.55, 0.55, 0.55))
            ax.set_yticks(y)
            ax.set_yticklabels([])
            ax.set_xscale("log")
            ax.set_xlabel("Training peptides")
            _despine(ax)
        if binder_col:
            ax = axes[0, 2]
            ax.barh(y, sub[binder_col], color=(0.65, 0.39, 0.67))
            ax.set_yticks(y)
            ax.set_yticklabels([])
            ax.set_xlabel("% binders")
            _despine(ax)
        fig.tight_layout(w_pad=1.0)
        writer.save(
            fig,
            "fig.1_model_selection_predictor_accuracy.scores.%s" % label,
            "model-selection")


def _generate_monoallelic_figures(
        artifacts_dir, predictor_info, max_scatter_points, writer):
    path = artifacts_dir / "accuracy_scores.monoallelic.csv"
    if path.is_file():
        scores = _normalize_score_predictors(pandas.read_csv(path))
        _plot_monoallelic_scatter(
            scores, predictor_info, "auc", "AUC", max_scatter_points,
            "fig.3_scores_plots_monoallelic.scatter.auc.monoallelic.ba",
            writer)
        _plot_monoallelic_scatter(
            scores, predictor_info, "ppv", "PPV", max_scatter_points,
            "fig.3_scores_plots_monoallelic.scatter.ppv.monoallelic.ba",
            writer)
    else:
        writer.skip(
            "monoallelic",
            "fig.3_scores_plots_monoallelic.scatter.auc.monoallelic.ba",
            [path],
            "Monoallelic accuracy scores absent.")
        writer.skip(
            "monoallelic",
            "fig.3_scores_plots_monoallelic.scatter.ppv.monoallelic.ba",
            [path],
            "Monoallelic accuracy scores absent.")

    novel_path = artifacts_dir / "accuracy_scores.monoallelic.novel_alleles.csv"
    if novel_path.is_file():
        scores = _normalize_score_predictors(pandas.read_csv(novel_path))
        _plot_monoallelic_scatter(
            scores, predictor_info, "auc", "AUC", max_scatter_points,
            "fig.3_scores_plots_monoallelic.scatter.auc.monoallelic.novel_alleles.ba",
            writer,
            preferred_candidate="no_additional_ms_similar")
    else:
        writer.skip(
            "monoallelic",
            "fig.3_scores_plots_monoallelic.scatter.auc.monoallelic.novel_alleles.ba",
            [novel_path],
            "Novel-allele monoallelic accuracy scores absent.")


def _plot_monoallelic_scatter(
        scores, predictor_info, metric, metric_label, max_points, name, writer,
        preferred_candidate="no_additional_ms"):
    candidate = (
        preferred_candidate if preferred_candidate in set(scores["predictor"])
        else "mhcflurry_production"
    )
    if candidate not in set(scores["predictor"]):
        writer.skip(
            "monoallelic", name, [candidate],
            "MHCflurry candidate predictor absent.")
        return
    pivot = scores.pivot_table(
        index=_monoallelic_index_columns(scores),
        columns="predictor",
        values=metric,
        aggfunc="mean")
    needed = [candidate] + [predictor for predictor, _ in EXTERNAL_BASELINES]
    missing = [predictor for predictor in needed if predictor not in pivot.columns]
    if missing:
        writer.skip(
            "monoallelic", name, missing,
            "Required predictors absent from monoallelic scores.")
        return
    _plot_scatter_triptych_from_pivot(
        pivot, predictor_info, candidate, metric_label, max_points, name,
        "monoallelic", writer)


def _generate_processing_notebook_figures(artifacts_dir, predictor_info, writer):
    no_c_path = artifacts_dir / "accuracy_scores.multiallelic.no_C.csv"
    motif_path = artifacts_dir / "antigen_processing.motifs.xlsx"
    correlation_path = artifacts_dir / "correlation.processing_vs_affinity.sampled.csv.bz2"
    training_path = artifacts_dir / "train_data.ap.production.csv"

    if no_c_path.is_file():
        _plot_cysteine_removed_panels(artifacts_dir, no_c_path, predictor_info, writer)
    else:
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.auc.ap.c_removed.scatter",
            [no_c_path],
            "Cysteine-removed multiallelic benchmark scores absent.")
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.auc.ap.c_removed.bar",
            [no_c_path],
            "Cysteine-removed multiallelic benchmark scores absent.")
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.bar.ap_vs_others",
            [no_c_path],
            "AP-vs-other benchmark table absent.")

    if motif_path.is_file():
        _plot_ap_motif_logo(motif_path, writer)
    else:
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.logo.ap",
            [motif_path],
            "Antigen-processing motif workbook absent.")

    if correlation_path.is_file() and training_path.is_file():
        _plot_ap_correlation_panels(correlation_path, training_path, writer)
    else:
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.correlation.ap_by_gene",
            [correlation_path, training_path],
            "AP correlation and training-data tables absent.")
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.extended.ap_correlation",
            [correlation_path, training_path],
            "AP correlation and training-data tables absent.")
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.correlation.included_vs_excluded",
            [correlation_path, training_path],
            "AP correlation and training-data tables absent.")


def _plot_cysteine_removed_panels(artifacts_dir, no_c_path, predictor_info, writer):
    import matplotlib.pyplot as plt

    full_path = artifacts_dir / "accuracy_scores.multiallelic.csv"
    if not full_path.is_file():
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.auc.ap.c_removed.scatter",
            [full_path],
            "Full multiallelic benchmark scores absent.")
        return
    full = _normalize_score_predictors(pandas.read_csv(full_path))
    no_c = _normalize_score_predictors(pandas.read_csv(no_c_path))
    predictor = "presentation_without_flanks_processing_score"
    full_pivot = _pivot_all_lengths(full, "auc")
    no_c_pivot = _pivot_all_lengths(no_c, "auc")
    if predictor not in full_pivot.columns or predictor not in no_c_pivot.columns:
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.auc.ap.c_removed.scatter",
            [predictor],
            "AP predictor absent from full or cysteine-removed benchmark.")
        return
    joined = pandas.DataFrame({
        "full": full_pivot[predictor],
        "c_removed": no_c_pivot[predictor],
    }).dropna()

    fig, ax = plt.subplots(figsize=(2.5, 2.4))
    ax.scatter(joined["full"], joined["c_removed"],
               color=_predictor_color(predictor_info, predictor),
               s=16, alpha=0.8, edgecolor="white", linewidth=0.2)
    _add_diagonal(ax, joined["full"], joined["c_removed"])
    ax.set_xlabel("Full benchmark AUC")
    ax.set_ylabel("Cysteine removed AUC")
    _despine(ax)
    fig.tight_layout()
    writer.save(
        fig,
        "fig.4_processing_predictor_plots.auc.ap.c_removed.scatter",
        "antigen-processing")

    means = pandas.Series({
        "Full": joined["full"].mean(),
        "C removed": joined["c_removed"].mean(),
    })
    fig, ax = plt.subplots(figsize=(2.1, 2.2))
    ax.bar(numpy.arange(len(means)), means.values,
           color=_predictor_color(predictor_info, predictor),
           edgecolor="white", linewidth=0.6)
    ax.set_xticks(numpy.arange(len(means)))
    ax.set_xticklabels(means.index, rotation=25, ha="right")
    ax.set_ylabel("Mean AUC")
    _despine(ax)
    fig.tight_layout()
    writer.save(
        fig,
        "fig.4_processing_predictor_plots.auc.ap.c_removed.bar",
        "antigen-processing")

    _plot_ap_vs_others(no_c, predictor_info, writer)


def _plot_ap_vs_others(scores, predictor_info, writer):
    import matplotlib.pyplot as plt

    sub = _all_length_rows(scores)
    predictors = [
        predictor for predictor in (
            "presentation_without_flanks_processing_score",
            "presentation_with_flanks_processing_score",
            "mhcflurry_production",
            "netmhcpan4.ba",
            "netmhcpan4.el",
            "mixmhcpred",
        )
        if predictor in set(sub["predictor"])
    ]
    if not predictors:
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.bar.ap_vs_others",
            [],
            "No AP comparison predictors found.")
        return
    means = (
        sub.loc[sub["predictor"].isin(predictors)]
        .groupby("predictor")["auc"]
        .mean()
        .reindex(predictors)
        .dropna()
    )
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = numpy.arange(len(means))
    ax.bar(
        x, means.values,
        color=[_predictor_color(predictor_info, p) for p in means.index],
        edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_short_label(predictor_info, p) for p in means.index],
        rotation=30, ha="right")
    ax.set_ylabel("Mean AUC")
    _despine(ax)
    fig.tight_layout()
    writer.save(
        fig, "fig.4_processing_predictor_plots.bar.ap_vs_others",
        "antigen-processing")


def _plot_ap_motif_logo(path, writer):
    import matplotlib.pyplot as plt

    try:
        import logomaker
    except ImportError:
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.logo.ap",
            [path],
            "logomaker is not installed.")
        return
    workbook = pandas.read_excel(path, sheet_name=None)
    sheet_name = next(iter(workbook))
    df = workbook[sheet_name]
    if "position" in df.columns:
        df = df.set_index("position")
    numeric = df.select_dtypes(include=[numpy.number])
    if numeric.empty:
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.logo.ap",
            [path],
            "Motif workbook contains no numeric amino-acid matrix.")
        return
    fig, ax = plt.subplots(figsize=(7.0, 2.0))
    logomaker.Logo(numeric, ax=ax)
    ax.set_xlabel("Position")
    ax.set_ylabel("Weight")
    _despine(ax)
    fig.tight_layout()
    writer.save(
        fig, "fig.4_processing_predictor_plots.logo.ap",
        "antigen-processing")


def _plot_ap_correlation_panels(correlation_path, training_path, writer):
    import matplotlib.pyplot as plt

    corr = pandas.read_csv(correlation_path)
    training = pandas.read_csv(training_path)
    x_col = _first_present(corr, ("affinity_score", "affinity", "ba_score"))
    y_col = _first_present(corr, ("processing_score", "ap_score", "processing"))
    gene_col = _first_present(corr, ("gene", "protein", "protein_id"))
    if x_col is None or y_col is None:
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.correlation.ap_by_gene",
            [correlation_path],
            "Correlation table must contain affinity and processing columns.")
        return

    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    if gene_col is not None:
        top_genes = corr[gene_col].value_counts().head(6).index
        for gene in top_genes:
            sub = corr.loc[corr[gene_col] == gene]
            ax.scatter(sub[x_col], sub[y_col], s=8, alpha=0.6, label=str(gene))
        ax.legend(frameon=False, loc="best", handlelength=1.0)
    else:
        ax.scatter(corr[x_col], corr[y_col], s=8, alpha=0.6)
    ax.set_xlabel("Affinity score")
    ax.set_ylabel("Processing score")
    _despine(ax)
    fig.tight_layout()
    writer.save(
        fig,
        "fig.4_processing_predictor_plots.correlation.ap_by_gene",
        "antigen-processing")

    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    numeric = corr.select_dtypes(include=[numpy.number])
    if len(numeric.columns) >= 2:
        im = ax.imshow(numeric.corr(), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(numpy.arange(len(numeric.columns)))
        ax.set_yticks(numpy.arange(len(numeric.columns)))
        ax.set_xticklabels(numeric.columns, rotation=45, ha="right")
        ax.set_yticklabels(numeric.columns)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
    _despine(ax)
    fig.tight_layout()
    writer.save(
        fig,
        "fig.4_processing_predictor_plots.extended.ap_correlation",
        "antigen-processing")

    if "included" in training.columns and y_col in training.columns:
        fig, ax = plt.subplots(figsize=(2.5, 2.3))
        groups = [
            training.loc[training["included"].astype(bool), y_col].dropna(),
            training.loc[~training["included"].astype(bool), y_col].dropna(),
        ]
        ax.boxplot(groups, labels=["Included", "Excluded"], showfliers=False)
        ax.set_ylabel("Processing score")
        _despine(ax)
        fig.tight_layout()
        writer.save(
            fig,
            "fig.4_processing_predictor_plots.correlation.included_vs_excluded",
            "antigen-processing")
    else:
        writer.skip(
            "antigen-processing",
            "fig.4_processing_predictor_plots.correlation.included_vs_excluded",
            [training_path],
            "Training table lacks included flag or processing score column.")


def _generate_proteasome_figures(artifacts_dir, writer):
    path = _first_existing(
        artifacts_dir,
        ("Additional File 8.csv", "proteasome_mass_spec.csv"))
    if path is None:
        writer.skip(
            "proteasome",
            "fig.1_proteasome_mass_spec.proteosome.ms",
            [
                artifacts_dir / "Additional File 8.csv",
                artifacts_dir / "proteasome_mass_spec.csv",
            ],
            "Proteasome mass-spec source table absent.")
        return
    df = pandas.read_csv(path)
    category_col = _first_present(df, ("sample", "category", "condition", "gene"))
    value_col = _first_present(df, ("count", "spectra", "intensity", "value"))
    if category_col is None or value_col is None:
        writer.skip(
            "proteasome",
            "fig.1_proteasome_mass_spec.proteosome.ms",
            [path],
            "Proteasome table must contain category and numeric value columns.")
        return
    import matplotlib.pyplot as plt

    values = df.groupby(category_col)[value_col].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5.5, 2.7))
    x = numpy.arange(len(values))
    ax.bar(x, values.values, color=(0.34, 0.46, 0.75),
           edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(values.index, rotation=35, ha="right")
    ax.set_ylabel(value_col.replace("_", " ").title())
    _despine(ax)
    fig.tight_layout()
    writer.save(fig, "fig.1_proteasome_mass_spec.proteosome.ms", "proteasome")


def _copy_architecture_figures(artifacts_dir, writer):
    patterns = (
        "*architecture*.svg", "*architecture*.pdf", "*architecture*.png",
        "*model_information*.svg", "*model_information*.pdf",
        "*model_information*.png", "*model_info*.svg", "*model_info*.pdf",
        "*model_info*.png",
    )
    copied = []
    asset_dir = writer.out_dir / "assets"
    for pattern in patterns:
        for path in artifacts_dir.glob(pattern):
            asset_dir.mkdir(parents=True, exist_ok=True)
            target = asset_dir / path.name
            shutil.copy2(path, target)
            copied.append(str(target))
    if copied:
        writer.rows.append({
            "family": "architecture",
            "figure": "architecture_diagrams",
            "status": "generated",
            "paths": ";".join(copied),
            "note": "Copied existing architecture/model-info artwork.",
            "missing": "",
        })
    else:
        writer.skip(
            "architecture",
            "architecture_diagrams",
            [artifacts_dir / "*architecture*", artifacts_dir / "*model_info*"],
            "Architecture/model-information source artwork absent.")


def _plot_scatter_triptych_from_pivot(
        pivot, predictor_info, candidate, metric_label, max_points, name, family,
        writer):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2))
    for ax, (baseline, _suffix) in zip(axes, EXTERNAL_BASELINES):
        sub = pivot[[baseline, candidate]].replace(
            [numpy.inf, -numpy.inf], numpy.nan).dropna()
        _scatter_with_winner_colors(
            ax, sub[baseline].values, sub[candidate].values,
            baseline, candidate, predictor_info, max_points)
        _finish_metric_scatter(
            ax,
            _short_label(predictor_info, baseline),
            _short_label(predictor_info, candidate),
            "%s vs %s" % (
                _short_label(predictor_info, candidate),
                _short_label(predictor_info, baseline)),
            metric_label)
    fig.tight_layout(w_pad=1.0)
    writer.save(fig, name, family)


def _pivot_all_lengths(scores, metric):
    rows = _all_length_rows(scores)
    return rows.pivot_table(
        index="sample_id",
        columns="predictor",
        values=metric,
        aggfunc="mean")


def _normalize_score_predictors(scores):
    if "predictor" not in scores.columns:
        return scores
    scores = scores.copy()
    scores["predictor"] = scores["predictor"].str.replace(
        r"_affinity$", "", regex=True)
    return scores


def _all_length_rows(scores):
    if "length_label" in scores.columns:
        return scores.loc[scores["length_label"] == "All"].copy()
    if "length" in scores.columns:
        return scores.loc[scores["length"].isnull()].copy()
    return scores.copy()


def _monoallelic_index_columns(scores):
    for candidates in (
            ("allele", "peptide_length"),
            ("allele", "length"),
            ("allele",),
            ("sample_id",),
    ):
        if all(column in scores.columns for column in candidates):
            return list(candidates)
    return scores.index


def _scatter_with_winner_colors(
        ax, x, y, x_predictor, y_predictor, predictor_info, max_points):
    x = numpy.asarray(x, dtype=float)
    y = numpy.asarray(y, dtype=float)
    mask = numpy.isfinite(x) & numpy.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) > max_points:
        rng = numpy.random.default_rng(0)
        indices = rng.choice(len(x), size=max_points, replace=False)
        x = x[indices]
        y = y[indices]
    x_color = _predictor_color(predictor_info, x_predictor)
    y_color = _predictor_color(predictor_info, y_predictor)
    colors = [y_color if y_val >= x_val else x_color
              for x_val, y_val in zip(x, y)]
    ax.scatter(
        x, y, c=colors, s=14, alpha=0.78,
        edgecolor="white", linewidth=0.2, rasterized=False)
    _add_diagonal(ax, x, y)


def _finish_metric_scatter(ax, x_label, y_label, title, metric_label):
    ax.set_title(title)
    ax.set_xlabel("%s %s" % (x_label, metric_label))
    ax.set_ylabel("%s %s" % (y_label, metric_label))
    _set_unit_limits(ax)
    _despine(ax)


def _set_unit_limits(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    low = max(0.0, min(xmin, ymin))
    high = min(1.0, max(xmax, ymax))
    if high <= low:
        low, high = 0.0, 1.0
    pad = (high - low) * 0.05
    ax.set_xlim(max(0.0, low - pad), min(1.0, high + pad))
    ax.set_ylim(max(0.0, low - pad), min(1.0, high + pad))
    ax.set_aspect("equal", adjustable="box")


def _add_diagonal(ax, x, y):
    finite = numpy.asarray(list(x) + list(y), dtype=float)
    finite = finite[numpy.isfinite(finite)]
    if len(finite):
        low = max(0.0, float(finite.min()))
        high = min(1.0, float(finite.max()))
    else:
        low, high = 0.0, 1.0
    if high <= low:
        low, high = 0.0, 1.0
    pad = (high - low) * 0.05
    low = max(0.0, low - pad)
    high = min(1.0, high + pad)
    ax.plot([low, high], [low, high], color="0.35", linewidth=0.8, zorder=0)


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def _short_label(predictor_info, predictor):
    if predictor in predictor_info.index:
        row = predictor_info.loc[predictor]
        for column in ("short", "description"):
            value = row.get(column)
            if isinstance(value, str) and value and value != "-":
                return value
    return predictor.replace("_", " ")


def _predictor_color(predictor_info, predictor):
    if predictor in predictor_info.index:
        value = predictor_info.loc[predictor].get("color")
        if isinstance(value, str) and value and value.lower() != "nan":
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, (list, tuple)) and len(parsed) in (3, 4):
                    return tuple(float(item) for item in parsed)
            except (SyntaxError, ValueError, TypeError):
                pass
    fallback = {
        "netmhcpan4.ba": (0.886, 0.290, 0.200),
        "netmhcpan4.el": (1.000, 0.710, 0.722),
        "mixmhcpred": (0.204, 0.541, 0.741),
        "mhcflurry_production": (0.596, 0.557, 0.835),
    }
    if predictor in fallback:
        return fallback[predictor]
    palette = (
        (0.345, 0.467, 0.741),
        (0.459, 0.439, 0.702),
        (0.639, 0.400, 0.667),
        (0.871, 0.443, 0.498),
        (0.922, 0.612, 0.357),
        (0.580, 0.690, 0.392),
        (0.353, 0.612, 0.518),
        (0.306, 0.573, 0.702),
        (0.545, 0.545, 0.545),
        (0.737, 0.506, 0.741),
    )
    index = sum(
        (position + 1) * ord(char)
        for position, char in enumerate(predictor)
    ) % len(palette)
    return palette[index]


def _allele_locus(value):
    from mhcgnomes import parse

    result = parse(str(value), only_class1=True, raise_on_error=False)
    if result is not None:
        species = getattr(result, "species", None)
        prefix = getattr(species, "mhc_prefix", None)
        gene_name = getattr(result, "gene_name", None)
        if prefix == "HLA" and gene_name in ("A", "B", "C"):
            return "HLA-%s" % gene_name
        if prefix == "H2":
            return "H2"
    return "other"


def _first_existing(directory, names):
    for name in names:
        path = directory / name
        if path.is_file():
            return path
    return None


def _write_manifest(out_dir, rows):
    path = Path(out_dir) / "manifest.csv"
    pandas.DataFrame(rows).to_csv(path, index=False)


def _write_missing_inputs(out_dir, rows):
    skipped = [row for row in rows if row["status"] == "skipped"]
    path = Path(out_dir) / "missing_inputs.md"
    with open(path, "w") as fd:
        fd.write("# Missing paper-figure inputs\n\n")
        if not skipped:
            fd.write("All requested figure families were generated.\n")
            return
        for row in skipped:
            fd.write("## %s / %s\n\n" % (row["family"], row["figure"]))
            if row["missing"]:
                fd.write("Missing: `%s`\n\n" % row["missing"].replace(";", "`, `"))
            fd.write("%s\n\n" % row["note"])


# Module-level parser for sphinx autoprogram; behaves like the legacy
# ``mhcflurry-*`` command modules.
parser = make_parser()


if __name__ == "__main__":
    raise SystemExit(run_argv(os.sys.argv[1:]))
