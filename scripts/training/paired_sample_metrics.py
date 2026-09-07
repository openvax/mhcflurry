#!/usr/bin/env python3
"""Estimate paired sample uncertainty from saved per-sample metric tables."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy
import pandas


def comparison_long_frame(frame, *, units, condition, metrics, labels):
    """Convert compare-models paired a_/b_ columns without changing cohorts."""
    if len(labels) != 2 or labels[0] == labels[1]:
        raise ValueError("Require distinct side A and B labels")
    shared = units + [name for name in ("n", "n_pos") if name in frame and name not in units]
    pieces = []
    for prefix, label in zip(("a", "b"), labels):
        columns = {prefix + "_" + metric: metric for metric in metrics}
        piece = frame[shared + list(columns)].rename(columns=columns).copy()
        piece[condition] = label
        pieces.append(piece)
    return pandas.concat(pieces, ignore_index=True)


def paired_summary(frame, *, units, condition, metrics, baseline,
                   replicates=10000, seed=42):
    """Bootstrap complete matched samples; never silently drop unmatched rows."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    keys = units + [condition]
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("Sample/condition identities must be nonmissing and unique")
    wide = frame.set_index(keys)[metrics].unstack(condition).sort_index()
    if baseline not in frame[condition].unique():
        raise ValueError("Baseline condition is missing")
    if len(wide) < 2 or not numpy.isfinite(wide.to_numpy()).all():
        raise ValueError("Require at least two complete, finite matched samples")
    for count in ("n", "n_pos"):
        if count in frame:
            sizes = frame.set_index(keys)[count].unstack(condition)
            if (sizes.nunique(axis=1) != 1).any():
                raise ValueError("Unmatched sample counts: %s" % count)
    rng = numpy.random.default_rng(seed)
    # Resample samples together across conditions and metrics. This estimates
    # uncertainty of equal-sample macro means, not pooled (micro) metrics.
    indices = rng.integers(0, len(wide), size=(replicates, len(wide)))
    summary = []
    differences = []
    draws = {}
    for name in sorted(set(frame[condition]) - {baseline}):
        for metric in metrics:
            delta = wide[(metric, name)] - wide[(metric, baseline)]
            bootstrap = delta.to_numpy()[indices].mean(axis=1)
            lower, upper = numpy.quantile(bootstrap, [0.025, 0.975])
            summary.append({
                "condition": name, "baseline": baseline, "metric": metric,
                "samples": len(delta), "mean": wide[(metric, name)].mean(),
                "baseline_mean": wide[(metric, baseline)].mean(),
                "delta": delta.mean(), "ci_low": lower, "ci_high": upper,
                "samples_improved": int((delta > 0).sum()),
                "samples_regressed": int((delta < 0).sum()),
            })
            values = delta.rename("delta").reset_index()
            values["condition"] = name
            values["metric"] = metric
            differences.append(values)
            draws["%s:%s" % (name, metric)] = bootstrap
    if not summary:
        raise ValueError("Need at least one alternative condition")
    return pandas.DataFrame(summary), pandas.concat(differences), draws


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog=os.environ.get("MHCFLURRY_CLI_PROG"), description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--unit-columns", nargs="+", required=True)
    parser.add_argument("--condition-column", required=True)
    parser.add_argument("--metric-columns", nargs="+", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--comparison-labels", nargs=2, metavar=("A", "B"),
                        help="Read compare-models a_/b_ metric columns with these side labels.")
    parser.add_argument("--scope", help="Select rows whose scope column equals this value")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    source = Path(args.metrics)
    frame = pandas.read_csv(source, dtype={name: str for name in args.unit_columns})
    if args.scope:
        frame = frame.loc[frame.scope == args.scope].copy()
    if args.comparison_labels:
        frame = comparison_long_frame(frame, units=args.unit_columns, condition=args.condition_column,
                                      metrics=args.metric_columns, labels=args.comparison_labels)
    summary, differences, draws = paired_summary(
        frame, units=args.unit_columns, condition=args.condition_column,
        metrics=args.metric_columns, baseline=args.baseline,
        replicates=args.replicates, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "paired_input.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    differences.to_csv(out / "sample_differences.csv", index=False)
    numpy.savez_compressed(out / "bootstrap_deltas.npz", **draws)
    provenance = {
        "arguments": vars(args), "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "method": "paired sample percentile bootstrap, equal-sample macro mean",
        "limitations": "Exploratory, conditional on trained models; no correction for model selection or multiple comparisons.",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot

    fig, axes = pyplot.subplots(
        1, len(args.metric_columns), figsize=(12, max(3, len(draws) * 0.35)),
        squeeze=False, layout="constrained")
    for axis, metric in zip(axes[0], args.metric_columns):
        values = summary.loc[summary.metric == metric].reset_index(drop=True)
        y = numpy.arange(len(values))
        axis.hlines(y, values.ci_low, values.ci_high, color="#26728e", linewidth=2)
        axis.scatter(values.delta, y, color="#173f5f", zorder=3)
        axis.axvline(0, color="gray", linestyle="--", linewidth=1)
        axis.set_yticks(y, labels=values.condition)
        axis.set_title(metric)
        axis.set_xlabel("Macro difference (95% paired sample interval)")
        axis.invert_yaxis()
    fig.suptitle("Reference: %s" % args.baseline)
    for extension in ("svg", "png"):
        fig.savefig(out / ("paired_sample_intervals." + extension), dpi=180)
    pyplot.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
