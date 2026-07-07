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
from pathlib import Path

import numpy
import pandas

from .model_comparison_constants import (
    METRIC_NAMES,
    PRESENTATION_MODES,
    PRESENTATION_SCORE_KINDS,
    PROCESSING_MODES,
)


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
    parser.add_argument(
        "--input", required=True,
        help="Output directory produced by ``mhcflurry compare-models``.",
    )
    parser.add_argument(
        "--max-scatter-points", type=int, default=100_000,
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
            "Optional PDF path. When set, all generated PNG plots are also "
            "collected into a single PDF."
        ),
    )
    return parser


def run(args):
    import matplotlib
    matplotlib.use("Agg")

    labels = _load_side_labels(args.input)
    plot_dir = os.path.join(args.input, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    paper_dir = os.path.join(plot_dir, "paper")
    os.makedirs(paper_dir, exist_ok=True)

    available = _detect_available_components(args.input)
    if args.components == "auto":
        components = available
    else:
        requested = [p.strip() for p in args.components.split(",") if p]
        components = [c for c in requested if c in available]
        for missing in set(requested) - set(available):
            print("WARNING: %s not present in %s" % (missing, args.input))

    for component in components:
        if component == "affinity":
            _plot_affinity(args.input, plot_dir, labels, args.max_scatter_points)
        elif component == "processing":
            _plot_processing(args.input, plot_dir, labels, args.max_scatter_points)
        elif component == "presentation":
            _plot_presentation(args.input, plot_dir, labels, args.max_scatter_points)
    _plot_release_summary(args.input, paper_dir, labels)
    if args.summary_pdf:
        _write_summary_pdf(plot_dir, args.summary_pdf)
    return 0


def _load_side_labels(input_dir):
    labels = {"a": "a", "b": "b"}
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
        df = pandas.read_csv(pred_path)
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
        per_allele = pandas.read_csv(per_allele_path)
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
    sorted_df = per_allele.sort_values("roc_auc_diff", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(numpy.arange(len(sorted_df)), sorted_df["roc_auc_diff"])
    ax.axhline(0, color="0.6", linewidth=1)
    ax.set_xlabel("allele (sorted by ROC-AUC delta)")
    ax.set_ylabel("%s − %s ROC-AUC" % (label_a, label_b))
    ax.set_title("Per-allele ROC-AUC delta")
    fig.tight_layout()
    fig.savefig(os.path.join(sub_dir, "per_allele_roc_delta.png"))
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
                  title="%s processing ROC" % mode)
        _save_pr(plt, precision_recall_curve, average_precision_score,
                 y, a_score, b_score, label_a, label_b,
                 os.path.join(sub_dir, "pr_%s.png" % mode),
                 title="%s processing PR" % mode)
        _save_scatter(plt, b_score, a_score, label_b, label_a,
                      os.path.join(sub_dir, "scatter_%s.png" % mode),
                      title="%s processing: %s vs %s" % (
                          mode, label_a, label_b),
                      max_points=max_scatter_points)

    summary_table_path = os.path.join(processing_dir, "summary_table.csv")
    if os.path.isfile(summary_table_path):
        summary = _read_optional_csv(summary_table_path)
        _save_macro_bars(plt, summary, sub_dir, label_a, label_b)
    _save_component_paper_plots(
        plt,
        processing_dir,
        paper_dir,
        "processing",
        PROCESSING_MODES,
        "processing_score",
        label_a,
        label_b,
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
                      title="%s ROC (%s)" % (mode, score_kind))
            _save_pr(plt, precision_recall_curve, average_precision_score,
                     y, a_score, b_score, label_a, label_b,
                     os.path.join(sub_dir, "pr_%s.png" % stub),
                     title="%s PR (%s)" % (mode, score_kind))
            _save_scatter(plt, b_score, a_score, label_b, label_a,
                          os.path.join(sub_dir, "scatter_%s.png" % stub),
                          title="%s (%s): %s vs %s" % (
                              mode, score_kind, label_a, label_b),
                          max_points=max_scatter_points)

    summary_table_path = os.path.join(presentation_dir, "summary_table.csv")
    if os.path.isfile(summary_table_path):
        summary = _read_optional_csv(summary_table_path)
        _save_macro_bars(plt, summary, sub_dir, label_a, label_b)
    _save_component_paper_plots(
        plt,
        presentation_dir,
        paper_dir,
        "presentation",
        PRESENTATION_MODES,
        "presentation_score",
        label_a,
        label_b,
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
        rows = rows.set_index("plot_group").reindex(ordered_groups)
        x = numpy.arange(len(rows))
        width = 0.36
        ax.bar(x - width / 2, rows.side_a, width, label=labels["a"])
        ax.bar(x + width / 2, rows.side_b, width, label=labels["b"])
        ax.set_title(METRIC_DISPLAY_NAMES[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(rows.index, rotation=35, ha="right")
        ax.set_ylim(_metric_ylim(metric, rows[["side_a", "side_b"]].values))
        ax.grid(axis="y", color="0.9", linewidth=0.8)
    axes[0][0].set_ylabel("Macro mean")
    axes[0][-1].legend(frameon=False)
    fig.suptitle("Release-gate macro accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(paper_dir, "release_summary_macro.png"))
    plt.close(fig)

    fig, axes = plt.subplots(
        1, len(METRIC_NAMES),
        figsize=(3.2 * len(METRIC_NAMES), 3.0),
        squeeze=False,
    )
    for ax, metric in zip(axes[0], METRIC_NAMES):
        rows = macro.loc[macro.metric == METRIC_DISPLAY_NAMES[metric]]
        rows = rows.set_index("plot_group").reindex(ordered_groups)
        values = rows["diff"].values
        colors = numpy.where(values >= 0, "#2b8cbe", "#d95f02")
        ax.bar(numpy.arange(len(rows)), values, color=colors)
        ax.axhline(0, color="0.4", linewidth=0.8)
        ax.set_title(METRIC_DISPLAY_NAMES[metric])
        ax.set_xticks(numpy.arange(len(rows)))
        ax.set_xticklabels(rows.index, rotation=35, ha="right")
        ax.set_ylabel("%s - %s" % (labels["a"], labels["b"]))
        ax.grid(axis="y", color="0.9", linewidth=0.8)
    fig.suptitle("Release-gate macro deltas")
    fig.tight_layout()
    fig.savefig(os.path.join(paper_dir, "release_summary_macro_delta.png"))
    plt.close(fig)


def _release_summary_group_label(row):
    component = row.get("component", "")
    mode = row.get("flank_mode", "")
    if isinstance(mode, str) and mode:
        return "%s\n%s" % (component, mode)
    return str(row.get("eval", component))


def _save_component_paper_plots(
        plt, component_dir, paper_dir, component, modes, score_kind,
        label_a, label_b):
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
            os.path.join(paper_dir, "%s_per_sample_scatter.png" % component),
            label_a,
            label_b,
            "%s per-sample accuracy" % component.title(),
        )
        _save_metric_delta_boxplots(
            plt,
            sample_frames,
            os.path.join(paper_dir, "%s_per_sample_delta_boxplots.png" % component),
            "%s per-sample deltas" % component.title(),
            label_a,
            label_b,
        )

    if length_frames:
        _save_per_length_grid(
            plt,
            length_frames,
            os.path.join(paper_dir, "%s_per_length_macro.png" % component),
            label_a,
            label_b,
            "%s by peptide length" % component.title(),
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
            mask = ~(numpy.isnan(x) | numpy.isnan(y))
            if not mask.any():
                ax.axis("off")
                continue
            x = x[mask]
            y = y[mask]
            ax.scatter(x, y, s=12, alpha=0.55, linewidths=0)
            low, high = _metric_ylim(metric, numpy.concatenate([x, y]))
            ax.plot([low, high], [low, high], color="0.65", linewidth=0.8)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.grid(color="0.9", linewidth=0.8)
            ax.set_title(METRIC_DISPLAY_NAMES[metric])
            if col_idx == 0:
                ax.set_ylabel("%s\n%s" % (row_label, label_a))
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
    fig.savefig(out_path)
    plt.close(fig)


def _save_metric_delta_boxplots(
        plt, frames, out_path, title, label_a, label_b):
    fig, axes = plt.subplots(
        1, len(METRIC_NAMES),
        figsize=(3.0 * len(METRIC_NAMES), 3.0),
        squeeze=False,
    )
    labels = [label for (label, _) in frames]
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
            ax.boxplot(data, tick_labels=labels, showfliers=False)
        except TypeError:
            ax.boxplot(data, labels=labels, showfliers=False)
        ax.axhline(0, color="0.4", linewidth=0.8)
        ax.set_title(METRIC_DISPLAY_NAMES[metric])
        ax.set_ylabel("%s - %s" % (label_a, label_b))
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", color="0.9", linewidth=0.8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
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
    for row_idx, (row_label, df) in enumerate(frames):
        x = numpy.arange(len(df))
        lengths = [str(v) for v in df["length"]]
        width = 0.36
        for col_idx, metric in enumerate(METRIC_NAMES):
            ax = axes[row_idx][col_idx]
            a_col, b_col = _metric_columns(metric, average="macro")
            if a_col not in df.columns or b_col not in df.columns:
                ax.axis("off")
                continue
            ax.bar(x - width / 2, df[a_col], width, label=label_a)
            ax.bar(x + width / 2, df[b_col], width, label=label_b)
            ax.set_title(METRIC_DISPLAY_NAMES[metric])
            ax.set_xticks(x)
            ax.set_xticklabels(lengths)
            ax.set_xlabel("Peptide length")
            ax.set_ylim(_metric_ylim(metric, df[[a_col, b_col]].values))
            ax.grid(axis="y", color="0.9", linewidth=0.8)
            if col_idx == 0:
                ax.set_ylabel("%s\nMacro mean" % row_label)
            else:
                ax.set_ylabel("Macro mean")
            if row_idx == 0 and col_idx == n_cols - 1:
                ax.legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _metric_columns(metric, average=None):
    if average is None:
        return "a_%s" % metric, "b_%s" % metric
    return "a_%s_%s" % (average, metric), "b_%s_%s" % (average, metric)


def _metric_ylim(metric, values):
    values = numpy.asarray(values, dtype=float)
    values = values[~numpy.isnan(values)]
    if values.size == 0:
        return 0.0, 1.0
    if metric == "roc_auc":
        low = min(0.5, max(0.0, float(values.min()) - 0.03))
    else:
        low = 0.0
    high = min(1.0, max(0.1, float(values.max()) + 0.04))
    return low, high


def _write_summary_pdf(plot_dir, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plot_dir = Path(plot_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pngs = [
        path for path in sorted(plot_dir.rglob("*.png"))
        if path.resolve() != out_path.resolve()
    ]
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
            fig.suptitle(str(path.relative_to(plot_dir)), fontsize=10)
            pdf.savefig(fig)
            plt.close(fig)


def _save_macro_bars(plt, summary, sub_dir, label_a, label_b):
    if summary.empty:
        return
    x_labels = [
        "%s\n%s" % (
            row.mode,
            row.score_kind.replace("presentation_", "").replace("_score", ""),
        )
        for row in summary.itertuples()
    ]
    x = numpy.arange(len(summary))
    width = 0.38
    for metric in METRIC_NAMES:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - width / 2, summary["a_macro_%s" % metric], width, label=label_a)
        ax.bar(x + width / 2, summary["b_macro_%s" % metric], width, label=label_b)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right")
        ax.set_ylabel(metric)
        ax.set_title("Macro mean over samples: %s" % metric)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(sub_dir, "macro_%s.png" % metric))
        plt.close(fig)


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------


def _save_roc(plt, roc_curve_fn, roc_auc_fn,
              y, a_score, b_score, label_a, label_b, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, values in ((label_a, a_score), (label_b, b_score)):
        mask = ~numpy.isnan(values)
        if not mask.any() or len(numpy.unique(y[mask])) < 2:
            continue
        fpr, tpr, _ = roc_curve_fn(y[mask], values[mask])
        auc = roc_auc_fn(y[mask], values[mask])
        ax.plot(fpr, tpr, label="%s AUC=%.3f" % (label, auc))
    ax.plot([0, 1], [0, 1], color="0.6", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_pr(plt, pr_curve_fn, ap_fn,
             y, a_score, b_score, label_a, label_b, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, values in ((label_a, a_score), (label_b, b_score)):
        mask = ~numpy.isnan(values)
        if not mask.any() or len(numpy.unique(y[mask])) < 2:
            continue
        precision, recall, _ = pr_curve_fn(y[mask], values[mask])
        ap = ap_fn(y[mask], values[mask])
        ax.plot(recall, precision, label="%s AP=%.3f" % (label, ap))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_scatter(plt, x_score, y_score, x_label, y_label,
                  out_path, title, max_points):
    mask = ~(numpy.isnan(x_score) | numpy.isnan(y_score))
    idx = numpy.flatnonzero(mask)
    if len(idx) > max_points:
        rng = numpy.random.default_rng(17)
        idx = rng.choice(idx, size=max_points, replace=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_score[idx], y_score[idx], s=4, alpha=0.25)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# Module-level parser for sphinx autoprogram; behaves like the legacy
# ``mhcflurry-*`` command modules.
parser = make_parser()
