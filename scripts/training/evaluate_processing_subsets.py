#!/usr/bin/env python3
"""Evaluate every fixed-size subset of a processing ensemble, without selection."""

import argparse
import itertools
import json
import math
import os
from pathlib import Path

import numpy
import pandas

from mhcflurry.cli.processing_affinity_control import (
    AFFINITY_COLUMN, IDENTITY_COLUMNS, score_risk_sets, summarize_metrics,
)
from mhcflurry.common import configure_pytorch
from mhcflurry.experiment_archive import sha256_file
from mhcflurry.processing_matching import MATCHING_POLICY, validate_matching_assignments


def unique_inputs(frame):
    """Return one verified peptide/flank identity per original source row."""
    result = frame[["source_row", *IDENTITY_COLUMNS]].copy()
    for column in ("n_flank", "c_flank"):
        result[column] = result[column].fillna("")
    if result.isna().any().any() or result.empty:
        raise ValueError("Require nonempty identified source rows")
    rows = pandas.to_numeric(result.source_row, errors="raise")
    if not (numpy.isfinite(rows) & (rows >= 0) & (rows == numpy.floor(rows))).all():
        raise ValueError("source_row must contain nonnegative integers")
    result["source_row"] = rows.astype("int64")
    result = result.drop_duplicates()
    if result.source_row.duplicated().any():
        raise ValueError("Inconsistent identities for repeated source_row")
    return result.sort_values("source_row").reset_index(drop=True)


def subsets(model_count, size, max_subsets=10000):
    """Enumerate all combinations independently of labels and predictions."""
    if not 1 <= size <= model_count:
        raise ValueError("Subset size must be between one and the model count")
    count = math.comb(model_count, size)
    if count > max_subsets:
        raise ValueError("Requested %d subsets exceeds limit %d" % (count, max_subsets))
    return list(itertools.combinations(range(model_count), size))


def subset_distribution(summary, names):
    """Summarize composition sensitivity; these are not sampling intervals."""
    selected = summary.set_index("score").loc[names]
    metrics = ["roc_auc", "pr_auc", "ppv_at_n", "macro_roc_auc",
               "macro_pr_auc", "macro_ppv_at_n"]
    records = []
    for metric in metrics:
        values = selected[metric]
        records.append({"metric": metric, "subsets": len(values),
                        "mean": values.mean(), "median": values.median(),
                        "min": values.min(), "max": values.max(),
                        "q25": values.quantile(0.25), "q75": values.quantile(0.75)})
    return pandas.DataFrame(records)


def plot_subset_distribution(summary, names, comparisons, out):
    """Plot composition sensitivity and fixed comparisons, not significance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    table = summary.set_index("score")
    metrics = [("macro_roc_auc", "Macro AUROC"),
               ("macro_pr_auc", "Macro AUPRC"),
               ("macro_ppv_at_n", "Macro PPV@N")]
    colors = ["#262626", "#2166ac", "#c65d16", "#597b39"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, (metric, label) in zip(axes, metrics):
        ax.hist(table.loc[names, metric], bins=12, color="#c8d2de",
                edgecolor="white", label="All %d subsets" % len(names))
        for index, comparison in enumerate(comparisons):
            ax.axvline(table.loc[comparison, metric], color=colors[index % len(colors)],
                       linewidth=2, linestyle="--" if index == 0 else "-",
                       label=comparison.replace("_", " "))
        ax.set_xlabel(label)
        ax.set_ylabel("Number of subsets")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Every fixed-size subset; no held-out selection", fontsize=13)
    for extension in ("png", "svg"):
        fig.savefig(Path(out) / ("subset_distribution." + extension), dpi=180)
    plt.close(fig)


def make_parser():
    """Build the source-checkout experiment command."""
    parser = argparse.ArgumentParser(
        prog=os.environ.get("MHCFLURRY_CLI_PROG"), description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Strict processing-affinity-control matched_predictions table.")
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--subset-size", type=int, default=4)
    parser.add_argument("--max-subsets", type=int, default=10000)
    parser.add_argument("--reference-score", required=True,
                        help="Cached full-ensemble score to verify before subset evaluation.")
    parser.add_argument("--comparison-score", action="append", default=[])
    parser.add_argument("--verification-atol", type=float, default=1e-5)
    parser.add_argument("--member-cache-dir", type=Path,
                        help="Reuse fingerprinted member predictions from a prior run.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--backend", choices=("cpu", "mps", "gpu"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser


def run(args):
    """Cache member predictions, verify full scores, and score every subset."""
    if args.out.exists():
        raise ValueError("Use a fresh output directory to preserve previous results")
    if args.threads < 1 or args.batch_size < 1 or args.max_subsets < 1:
        raise ValueError("Threads, batch size and subset limit must be positive")
    if not numpy.isfinite(args.verification_atol) or args.verification_atol < 0:
        raise ValueError("Verification tolerance must be finite and nonnegative")
    frame = pandas.read_csv(args.input, dtype={"sample_id": str})
    counts = frame.groupby(["sample_id", "risk_set_id"]).size() - 1
    if counts.empty or counts.nunique() != 1:
        raise ValueError("Require a common positive decoys-per-hit count")
    validate_matching_assignments(frame, AFFINITY_COLUMN, int(counts.iloc[0]))
    # Derive this rather than trusting a stale auxiliary column in a cache.
    frame["peptide_len"] = frame.peptide.str.len()
    unique = unique_inputs(frame)
    comparisons = list(dict.fromkeys([args.reference_score] + args.comparison_score))
    for column in comparisons:
        if column not in frame or not numpy.isfinite(frame[column]).all():
            raise ValueError("Missing or nonfinite comparison score: %s" % column)
    manifest_path = args.models_dir / "manifest.csv"
    manifest = pandas.read_csv(manifest_path)
    model_names = manifest.model_name.tolist()
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("Require nonempty unique model names")
    members = []
    for index, name in enumerate(model_names):
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Invalid model name")
        weights = args.models_dir / ("weights_" + name + ".npz")
        members.append({"index": index, "model_name": name,
                        "weights": str(weights.resolve()), "sha256": sha256_file(weights)})
    combinations = subsets(len(members), args.subset_size, args.max_subsets)
    subset_names = ["subset_" + "_".join(map(str, indices)) for indices in combinations]
    if set(comparisons) & set(subset_names + ["reconstructed_full"]):
        raise ValueError("Comparison name collides with a generated score")
    provenance = {
        "design": "processing-exhaustive-ensemble-subsets-v1",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "input_sha256": sha256_file(args.input),
        "manifest_sha256": sha256_file(manifest_path), "members": members,
        "processing_matching_policy": MATCHING_POLICY,
        "rows": len(frame), "unique_source_rows": len(unique),
        "subsets": len(combinations), "selection": "none; enumerate all combinations",
        "training": False, "release_accepted": False,
    }
    args.out.mkdir(parents=True)
    (args.out / "experiment.json").write_text(json.dumps(provenance, indent=2) + "\n")
    manifest.to_csv(args.out / "manifest.csv", index=False)
    unique.to_csv(args.out / "prediction_inputs.csv.bz2", index=False)
    if args.member_cache_dir:
        saved = json.loads((args.member_cache_dir / "experiment.json").read_text())
        for key in ("input_sha256", "manifest_sha256", "members"):
            if saved[key] != provenance[key]:
                raise ValueError("Member cache identity mismatch: %s" % key)
        for key in ("backend", "threads", "batch_size"):
            if saved["arguments"][key] != provenance["arguments"][key]:
                raise ValueError("Member cache execution mismatch: %s" % key)
        with numpy.load(args.member_cache_dir / "member_predictions.npz", allow_pickle=False) as cache:
            if not numpy.array_equal(cache["source_row"], unique.source_row.to_numpy()):
                raise ValueError("Member cache source rows differ")
            matrix = cache["predictions"].copy()
        if matrix.shape != (len(unique), len(members)) or not numpy.isfinite(matrix).all():
            raise ValueError("Invalid member cache predictions")
        provenance["member_cache"] = {
            "path": str(args.member_cache_dir.resolve()),
            "sha256": sha256_file(args.member_cache_dir / "member_predictions.npz")}
        (args.out / "experiment.json").write_text(json.dumps(provenance, indent=2) + "\n")
    else:
        configure_pytorch(backend=args.backend, num_threads=args.threads)
        from mhcflurry import Class1ProcessingPredictor
        from mhcflurry.flanking_encoding import FlankingEncoding

        predictor = Class1ProcessingPredictor.load(str(args.models_dir))
        if len(predictor.models) != len(members):
            raise ValueError("Loaded model count differs from fingerprinted manifest")
        sequences = FlankingEncoding(unique.peptide, unique.n_flank, unique.c_flank)
        matrix = numpy.empty((len(unique), len(members)), dtype="float64")
        for index, network in enumerate(predictor.models):
            print("Scoring member %d/%d on %d source rows" % (
                index + 1, len(members), len(unique)), flush=True)
            values = network.predict_encoded(sequences, batch_size=args.batch_size)
            if not numpy.isfinite(values).all():
                raise ValueError("Nonfinite member predictions")
            matrix[:, index] = values
            pandas.DataFrame({"source_row": unique.source_row, "score": values}).to_csv(
                args.out / ("member_%02d_predictions.csv.bz2" % index), index=False)
    if args.member_cache_dir:
        for index in range(len(members)):
            pandas.DataFrame({"source_row": unique.source_row, "score": matrix[:, index]}).to_csv(
                args.out / ("member_%02d_predictions.csv.bz2" % index), index=False)
    # Individual predictions + the membership table reconstruct all conditions
    # without storing seventy redundant full peptide/flank tables.
    numpy.savez_compressed(args.out / "member_predictions.npz",
                          source_row=unique.source_row.to_numpy(), predictions=matrix)
    indices = pandas.Index(unique.source_row).get_indexer(frame.source_row)
    if (indices < 0).any():
        raise ValueError("Failed to map source rows to cached predictions")
    full = matrix.mean(axis=1)[indices]
    error = numpy.abs(full - frame[args.reference_score].to_numpy())
    verification = {"max_absolute_error": float(error.max()),
                    "mean_absolute_error": float(error.mean()),
                    "absolute_tolerance": args.verification_atol}
    (args.out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    if error.max() > args.verification_atol:
        raise ValueError("Reconstructed ensemble differs from cached reference: %g" % error.max())
    frame["reconstructed_full"] = full
    metrics = [score_risk_sets(frame, comparisons + ["reconstructed_full"])]
    membership = []
    for number, (name, selected) in enumerate(zip(subset_names, combinations), 1):
        frame[name] = matrix[:, selected].mean(axis=1)[indices]
        metrics.append(score_risk_sets(frame, [name]))
        membership.append({"score": name, "indices": list(selected),
                           "model_names": [model_names[i] for i in selected]})
        if number % 10 == 0 or number == len(combinations):
            print("Evaluated %d/%d subsets" % (number, len(combinations)), flush=True)
    metrics = pandas.concat(metrics, ignore_index=True)
    summary, deltas = summarize_metrics(metrics, args.reference_score)
    distribution = subset_distribution(summary, subset_names)
    (args.out / "subset_membership.json").write_text(json.dumps(membership, indent=2) + "\n")
    frame.to_csv(args.out / "matched_predictions.csv.bz2", index=False)
    metrics.to_csv(args.out / "metrics.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    deltas.to_csv(args.out / "comparisons.csv", index=False)
    distribution.to_csv(args.out / "subset_distribution.csv", index=False)
    plot_subset_distribution(summary, subset_names,
                             ["reconstructed_full"] + comparisons[1:], args.out)
    print(distribution.to_string(index=False), flush=True)
    print(summary.loc[summary.score.isin(comparisons)].to_string(index=False), flush=True)
    (args.out / "completed.json").write_text(json.dumps({"complete": True}) + "\n")
    return 0


def main(argv=None):
    return run(make_parser().parse_args(argv))


if __name__ == "__main__":
    main()
