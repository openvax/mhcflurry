#!/usr/bin/env python

"""Render manuscript figures directly from archived release experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy
import pandas

from mhcflurry.cli.figure_style import (
    NEGATIVE_DELTA_COLOR,
    POSITIVE_DELTA_COLOR,
    apply_paper_style,
    despine,
)


AFFINITY_COMPARISONS = (
    (
        "Keras/post-LSUV, mb1024",
        "proposed_release-vs-published_parity/release_summary.csv",
    ),
    (
        "Keras/pre-LSUV, mb1024",
        "pre_activation_lsuv-vs-published_parity/release_summary.csv",
    ),
    (
        "Keras/no-LSUV, mb1024",
        "no_lsuv-vs-published_parity/release_summary.csv",
    ),
    (
        "PyTorch/post-LSUV, mb1024",
        "pytorch_rmsprop-vs-published_parity/release_summary.csv",
    ),
)

PROCESSING_COMPARISONS = (
    (
        "Kaiming + Keras Adam",
        "kaiming_keras_adam-vs-glorot_keras_adam/release_summary.csv",
    ),
    (
        "Glorot + PyTorch Adam",
        "glorot_pytorch_adam-vs-glorot_keras_adam/release_summary.csv",
    ),
    (
        "Kaiming + PyTorch Adam",
        "kaiming_pytorch_adam-vs-glorot_keras_adam/release_summary.csv",
    ),
)

FLANK_LABELS = {
    "with_flanks": "15 aa",
    "no_flank": "0 aa",
    "short_flanks": "5 aa",
}
METRICS = ("AUROC", "AUPRC", "PPV@N")


def make_parser():
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--affinity-ablation-dir", required=True)
    parser.add_argument("--affinity-native128-dir", required=True)
    parser.add_argument("--processing-run-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--formats", default="svg,pdf,png")
    return parser


def sha256_file(path):
    """Return the SHA256 digest for one source or output file."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path, inputs):
    """Validate and record one manuscript input file."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("Missing experiment source file: %s" % path)
    inputs.append({
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    })
    return path


def read_release_summary(path, inputs):
    """Read one standardized release comparison table."""
    path = require_file(path, inputs)
    result = pandas.read_csv(path)
    required = {"metric", "average", "pct_change"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError("%s lacks columns: %s" % (path, ", ".join(missing)))
    return result


def render_heatmap(ax, values, row_labels, column_labels, title):
    """Render an annotated, zero-centered percent-change heatmap."""
    finite = values[numpy.isfinite(values)]
    limit = max(float(numpy.abs(finite).max()) if finite.size else 0.0, 1.0)
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list(
        "release_delta",
        [NEGATIVE_DELTA_COLOR, (0.98, 0.98, 0.98), POSITIVE_DELTA_COLOR],
    )
    image = ax.imshow(
        values,
        aspect="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    ax.set_xticks(numpy.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=35, ha="right")
    ax.set_yticks(numpy.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if numpy.isfinite(value):
                ax.text(
                    column,
                    row,
                    "%+.2f%%" % value,
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def affinity_interaction_figure(
        affinity_dir, native128_dir, inputs, outputs, out_dir, formats):
    """Render optimizer/LSUV/batch interactions from paired affinity runs."""
    records = []
    comparison_root = Path(affinity_dir) / "paired_comparisons"
    for label, relative in AFFINITY_COMPARISONS:
        table = read_release_summary(comparison_root / relative, inputs)
        table["condition"] = label
        records.append(table)
    native_path = (
        Path(native128_dir) / "paired_comparisons" /
        "pytorch_rmsprop_batch128-vs-published_parity" /
        "release_summary.csv"
    )
    native = read_release_summary(native_path, inputs)
    native["condition"] = "PyTorch/post-LSUV, mb128"
    records.append(native)
    table = pandas.concat(records, ignore_index=True)
    row_labels = list(table["condition"].drop_duplicates())

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for axis, average in zip(axes, ("Macro", "Micro")):
        subset = table.loc[table.average == average]
        values = (
            subset.pivot(index="condition", columns="metric", values="pct_change")
            .reindex(index=row_labels, columns=METRICS)
            .to_numpy(dtype=float)
        )
        image = render_heatmap(
            axis, values, row_labels, METRICS,
            "%s change vs Keras/post-LSUV, mb128" % average)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Change (%)")
    fig.tight_layout(w_pad=1.0)
    save_figure(fig, "affinity_recipe_interactions", out_dir, formats, outputs)


def processing_interaction_figure(
        processing_dir, inputs, outputs, out_dir, formats):
    """Render processing initializer/optimizer interactions across flanks."""
    records = []
    comparison_root = Path(processing_dir) / "paired_comparisons"
    for label, relative in PROCESSING_COMPARISONS:
        table = read_release_summary(comparison_root / relative, inputs)
        table["condition"] = label
        records.append(table)
    table = pandas.concat(records, ignore_index=True)
    row_labels = list(table["condition"].drop_duplicates())
    column_keys = [
        (flank, metric)
        for flank in ("with_flanks", "no_flank", "short_flanks")
        for metric in ("AUPRC", "PPV@N")
    ]
    column_labels = [
        "%s\n%s" % (FLANK_LABELS[flank], metric)
        for flank, metric in column_keys
    ]

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.25), sharey=True)
    for axis, average in zip(axes, ("Macro", "Micro")):
        subset = table.loc[table.average == average]
        values = numpy.full((len(row_labels), len(column_keys)), numpy.nan)
        for row_index, condition in enumerate(row_labels):
            for column_index, (flank, metric) in enumerate(column_keys):
                matched = subset.loc[
                    (subset.condition == condition) &
                    (subset.flank_mode == flank) &
                    (subset.metric == metric),
                    "pct_change",
                ]
                if len(matched) == 1:
                    values[row_index, column_index] = matched.iloc[0]
        image = render_heatmap(
            axis, values, row_labels, column_labels,
            "%s change vs Glorot + Keras Adam" % average)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Change (%)")
    fig.tight_layout(w_pad=1.0)
    save_figure(
        fig, "processing_recipe_interactions", out_dir, formats, outputs)


def flank_context_figure(
        processing_dir, inputs, outputs, out_dir, formats):
    """Render the direct 5-aa versus 15-aa comparison and length CIs."""
    comparison = (
        Path(processing_dir) / "direct_comparisons" /
        "best_5aa-vs-best_15aa"
    )
    samples_path = require_file(comparison / "per_sample.csv", inputs)
    summary_path = require_file(comparison / "summary.json", inputs)
    samples = pandas.read_csv(samples_path)
    summary = json.loads(summary_path.read_text())

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.15))
    ax = axes[0]
    ax.scatter(
        samples.pr_auc_b,
        samples.pr_auc_a,
        color=POSITIVE_DELTA_COLOR,
        edgecolor="white",
        linewidth=0.4,
        s=35,
    )
    low = min(samples.pr_auc_a.min(), samples.pr_auc_b.min())
    high = max(samples.pr_auc_a.max(), samples.pr_auc_b.max())
    pad = (high - low) * 0.08
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color="0.45")
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("15-aa Kaiming/PyTorch AUPRC")
    ax.set_ylabel("5-aa Glorot/Keras AUPRC")
    ax.set_title("Ten held-out samples")
    despine(ax)

    ax = axes[1]
    lengths = [8, 9, 10, 11]
    for offset, (metric, label, color) in enumerate((
            ("pr_auc", "AUPRC", POSITIVE_DELTA_COLOR),
            ("ppv_at_n", "PPV@N", (0.345, 0.467, 0.741)))):
        differences = []
        lower = []
        upper = []
        for length in lengths:
            record = summary["per_length_macro_over_samples"][str(length)][metric]
            differences.append(record["diff"])
            lower.append(record["diff_ci95_sample_bootstrap"][0])
            upper.append(record["diff_ci95_sample_bootstrap"][1])
        differences = numpy.asarray(differences)
        errors = numpy.vstack([
            differences - numpy.asarray(lower),
            numpy.asarray(upper) - differences,
        ])
        x = numpy.arange(len(lengths)) + (offset - 0.5) * 0.14
        ax.errorbar(
            x, differences, yerr=errors, fmt="o", color=color,
            capsize=3, label=label)
    ax.axhline(0.0, color="0.45", linewidth=0.8)
    ax.set_xticks(numpy.arange(len(lengths)))
    ax.set_xticklabels(["%d-mer" % length for length in lengths])
    ax.set_ylabel("5-aa minus 15-aa")
    ax.set_title("Sample-bootstrap 95% CIs")
    ax.legend()
    despine(ax)
    fig.tight_layout(w_pad=1.2)
    save_figure(fig, "processing_flank_context", out_dir, formats, outputs)


def save_figure(fig, name, out_dir, formats, outputs):
    """Save one manuscript figure and record all emitted files."""
    import matplotlib.pyplot as plt
    for file_format in formats:
        path = out_dir / ("%s.%s" % (name, file_format))
        metadata = (
            {"Date": None, "Creator": "MHCflurry release experiment renderer"}
            if file_format == "svg" else None
        )
        fig.savefig(
            path, bbox_inches="tight", transparent=True, metadata=metadata)
        outputs.append({
            "figure": name,
            "format": file_format,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    plt.close(fig)


def run(args):
    """Render all currently available source-backed manuscript figures."""
    matplotlib.use("Agg")
    apply_paper_style()
    matplotlib.rcParams["svg.hashsalt"] = "mhcflurry-release-2.3-experiments"
    formats = tuple(part.strip().lower() for part in args.formats.split(","))
    if not formats or any(part not in {"svg", "pdf", "png"} for part in formats):
        raise ValueError("--formats must contain svg, pdf, and/or png")
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = []
    outputs = []
    affinity_interaction_figure(
        args.affinity_ablation_dir,
        args.affinity_native128_dir,
        inputs,
        outputs,
        out_dir,
        formats,
    )
    processing_interaction_figure(
        args.processing_run_dir, inputs, outputs, out_dir, formats)
    flank_context_figure(
        args.processing_run_dir, inputs, outputs, out_dir, formats)
    manifest = {
        "schema_version": 1,
        "inputs": inputs,
        "outputs": outputs,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(make_parser().parse_args()))
