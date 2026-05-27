"""Plot the metric outputs from ``mhcflurry compare-models``.

Reads the CSVs + JSON written by ``compare-models`` and renders ROC / PR
/ scatter / per-allele delta plots under ``<input>/plots/``. Kept as a
separate subcommand so the metric pipeline doesn't pay the matplotlib
import cost.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy
import pandas


def make_parser():
    """Return a standalone parser for documentation tooling (autoprogram)."""
    parser = argparse.ArgumentParser(prog="mhcflurry plot-model-comparison")
    register_subparser(parser)
    return parser


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
            "Comma-separated subset of {affinity, presentation}; default "
            "'auto' plots whichever components are present in --input."
        ),
    )
    return parser


def run(args):
    import matplotlib
    matplotlib.use("Agg")

    labels = _load_side_labels(args.input)
    plot_dir = os.path.join(args.input, "plots")
    os.makedirs(plot_dir, exist_ok=True)

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
        elif component == "presentation":
            _plot_presentation(args.input, plot_dir, labels, args.max_scatter_points)
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
    if os.path.isfile(os.path.join(input_dir, "affinity", "predictions.csv.bz2")):
        components.append("affinity")
    if os.path.isdir(os.path.join(input_dir, "presentation")):
        components.append("presentation")
    return components


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

    pred_path = os.path.join(input_dir, "affinity", "predictions.csv.bz2")
    df = pandas.read_csv(pred_path)
    y = df.hit.values
    a_score = df.a_score.values
    b_score = df.b_score.values
    label_a, label_b = labels["a"], labels["b"]

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
# presentation
# ---------------------------------------------------------------------------


_PRESENTATION_MODES = ("with_flanks", "without_flanks")
_PRESENTATION_SCORE_KINDS = ("presentation_score", "presentation_percentile")
_METRIC_NAMES = ("roc_auc", "pr_auc", "ppv_at_n")


def _plot_presentation(input_dir, plot_dir, labels, max_scatter_points):
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        average_precision_score, precision_recall_curve,
        roc_auc_score, roc_curve,
    )

    sub_dir = os.path.join(plot_dir, "presentation")
    os.makedirs(sub_dir, exist_ok=True)
    label_a, label_b = labels["a"], labels["b"]

    presentation_dir = os.path.join(input_dir, "presentation")
    for mode in _PRESENTATION_MODES:
        pred_path = os.path.join(
            presentation_dir, "predictions_%s.csv.bz2" % mode)
        if not os.path.isfile(pred_path):
            continue
        df = pandas.read_csv(pred_path)
        for score_kind in _PRESENTATION_SCORE_KINDS:
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
        summary = pandas.read_csv(summary_table_path)
        _save_macro_bars(plt, summary, sub_dir, label_a, label_b)


def _score_values(df, prefix, score_kind):
    """Higher = better; mirror the convention from compare_models."""
    if score_kind == "presentation_score":
        return df["%s_presentation_score" % prefix].values
    if score_kind == "presentation_percentile":
        return -df["%s_presentation_percentile" % prefix].values
    raise ValueError("Unknown score kind: %s" % score_kind)


def _save_macro_bars(plt, summary, sub_dir, label_a, label_b):
    if summary.empty:
        return
    x_labels = [
        "%s\n%s" % (
            row.mode,
            row.score_kind.replace("presentation_", ""),
        )
        for row in summary.itertuples()
    ]
    x = numpy.arange(len(summary))
    width = 0.38
    for metric in _METRIC_NAMES:
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
        if not mask.any():
            continue
        fpr, tpr, _ = roc_curve_fn(y[mask], values[mask])
        auc = roc_auc_fn(y[mask], values[mask])
        ax.plot(fpr, tpr, label="%s AUC=%.3f" % (label, auc))
    ax.plot([0, 1], [0, 1], color="0.6", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_pr(plt, pr_curve_fn, ap_fn,
             y, a_score, b_score, label_a, label_b, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, values in ((label_a, a_score), (label_b, b_score)):
        mask = ~numpy.isnan(values)
        if not mask.any():
            continue
        precision, recall, _ = pr_curve_fn(y[mask], values[mask])
        ap = ap_fn(y[mask], values[mask])
        ax.plot(recall, precision, label="%s AP=%.3f" % (label, ap))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
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
