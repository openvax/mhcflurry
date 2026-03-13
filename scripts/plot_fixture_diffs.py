#!/usr/bin/env python
"""
Compare current PyTorch predictions against the TF fixture and generate
per-output figures showing absolute and percentile differences.

Usage:
    python scripts/plot_fixture_diffs.py [--out-dir /tmp/fixture_diffs]
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mhcflurry import Class1AffinityPredictor, Class1PresentationPredictor
from mhcflurry.downloads import (
    configure,
    get_default_class1_models_dir,
    get_default_class1_presentation_models_dir,
)

FIXTURE_CSV = os.path.join(
    os.path.dirname(__file__), os.pardir, "test", "data",
    "master_released_class1_presentation_highscore_rows.csv.gz",
)
FIXTURE_METADATA = os.path.join(
    os.path.dirname(__file__), os.pardir, "test", "data",
    "master_released_class1_presentation_highscore_rows_metadata.json",
)
BASE_COLUMNS = ["row_id", "peptide", "allele", "n_flank", "c_flank"]
STRING_COLUMNS = ["pres_with_best_allele", "pres_without_best_allele"]


def load_fixture():
    df = pd.read_csv(FIXTURE_CSV, keep_default_na=False)
    with open(FIXTURE_METADATA) as f:
        metadata = json.load(f)
    return df, metadata


def generate_predictions(fixture_df):
    configure()
    affinity_predictor = Class1AffinityPredictor.load(
        get_default_class1_models_dir())
    presentation_predictor = Class1PresentationPredictor.load(
        get_default_class1_presentation_models_dir())

    peptides = fixture_df["peptide"].tolist()
    alleles = fixture_df["allele"].tolist()
    n_flanks = fixture_df["n_flank"].tolist()
    c_flanks = fixture_df["c_flank"].tolist()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aff_df = affinity_predictor.predict_to_dataframe(
            peptides=peptides,
            alleles=alleles,
            throw=False,
            include_percentile_ranks=True,
            include_confidence_intervals=True,
            centrality_measure="mean",
            model_kwargs={"batch_size": 4096},
        )

    sample_names = alleles
    allele_map = {allele: [allele] for allele in sorted(set(alleles))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pres_with_df = presentation_predictor.predict(
            peptides=peptides,
            alleles=allele_map,
            sample_names=sample_names,
            n_flanks=n_flanks,
            c_flanks=c_flanks,
            include_affinity_percentile=True,
            verbose=0,
            throw=True,
        ).sort_values("peptide_num")
        pres_without_df = presentation_predictor.predict(
            peptides=peptides,
            alleles=allele_map,
            sample_names=sample_names,
            n_flanks=None,
            c_flanks=None,
            include_affinity_percentile=True,
            verbose=0,
            throw=True,
        ).sort_values("peptide_num")

    predicted = fixture_df[BASE_COLUMNS].copy()
    predicted["affinity_prediction"] = aff_df["prediction"].values
    predicted["affinity_prediction_low"] = aff_df.get(
        "prediction_low", np.nan)
    predicted["affinity_prediction_high"] = aff_df.get(
        "prediction_high", np.nan)
    predicted["affinity_prediction_percentile"] = aff_df.get(
        "prediction_percentile", np.nan)

    predicted["pres_with_affinity"] = pres_with_df["affinity"].values
    predicted["pres_with_best_allele"] = (
        pres_with_df["best_allele"].astype(str).values)
    predicted["pres_with_affinity_percentile"] = (
        pres_with_df["affinity_percentile"].values)
    predicted["processing_with_score"] = (
        pres_with_df["processing_score"].values)
    predicted["pres_with_processing_score"] = (
        pres_with_df["processing_score"].values)
    predicted["pres_with_presentation_score"] = (
        pres_with_df["presentation_score"].values)
    predicted["pres_with_presentation_percentile"] = (
        pres_with_df["presentation_percentile"].values)

    predicted["pres_without_affinity"] = pres_without_df["affinity"].values
    predicted["pres_without_best_allele"] = (
        pres_without_df["best_allele"].astype(str).values)
    predicted["pres_without_affinity_percentile"] = (
        pres_without_df["affinity_percentile"].values)
    predicted["processing_without_score"] = (
        pres_without_df["processing_score"].values)
    predicted["pres_without_processing_score"] = (
        pres_without_df["processing_score"].values)
    predicted["pres_without_presentation_score"] = (
        pres_without_df["presentation_score"].values)
    predicted["pres_without_presentation_percentile"] = (
        pres_without_df["presentation_percentile"].values)

    return predicted


def plot_output(col, tf_vals, pt_vals, out_dir):
    diff = pt_vals - tf_vals
    abs_diff = np.abs(diff)
    pct_diff = np.where(
        tf_vals != 0,
        100.0 * abs_diff / np.abs(tf_vals),
        np.where(abs_diff == 0, 0.0, np.inf),
    )
    pct_diff_finite = pct_diff[np.isfinite(pct_diff)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(col, fontsize=14, fontweight="bold")

    # Top-left: scatter TF vs PyTorch
    ax = axes[0, 0]
    ax.scatter(tf_vals, pt_vals, alpha=0.5, s=15, edgecolors="none")
    lims = [
        min(tf_vals.min(), pt_vals.min()),
        max(tf_vals.max(), pt_vals.max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=0.8, label="y = x")
    ax.set_xlabel("TF (fixture)")
    ax.set_ylabel("PyTorch (current)")
    ax.set_title("TF vs PyTorch")
    ax.legend(fontsize=8)

    # Top-right: histogram of absolute differences
    ax = axes[0, 1]
    ax.hist(diff, bins=50, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Difference (PyTorch − TF)")
    ax.set_ylabel("Count")
    ax.set_title(
        "Absolute diff: mean=%.2e, max=%.2e" % (
            np.mean(abs_diff), np.max(abs_diff)))

    # Bottom-left: absolute difference by allele (boxplot)
    ax = axes[1, 0]
    # Build a dataframe for the boxplot
    tmp = pd.DataFrame({"allele": alleles_global, "abs_diff": abs_diff})
    allele_order = (
        tmp.groupby("allele")["abs_diff"].median()
        .sort_values(ascending=False).index.tolist())
    box_data = [
        tmp.loc[tmp.allele == a, "abs_diff"].values for a in allele_order]
    if len(allele_order) <= 40:
        bp = ax.boxplot(box_data, vert=True, patch_artist=True)
        ax.set_xticks(range(1, len(allele_order) + 1))
        ax.set_xticklabels(
            [a.replace("HLA-", "") for a in allele_order],
            rotation=90, fontsize=6)
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.7)
    else:
        ax.bar(range(len(allele_order)),
               [np.median(d) for d in box_data], color="steelblue")
        ax.set_xticks(range(len(allele_order)))
        ax.set_xticklabels(allele_order, rotation=90, fontsize=5)
    ax.set_ylabel("|Difference|")
    ax.set_title("Absolute diff by allele")

    # Bottom-right: histogram of percent differences
    ax = axes[1, 1]
    if len(pct_diff_finite) > 0:
        clip_val = np.percentile(pct_diff_finite, 99)
        ax.hist(
            np.clip(pct_diff_finite, 0, clip_val),
            bins=50, edgecolor="black", linewidth=0.3)
        ax.set_xlabel("Percent difference (%)")
        ax.set_ylabel("Count")
        median_pct = np.median(pct_diff_finite)
        ax.set_title(
            "Pct diff: median=%.4f%%, 99th=%.4f%%" % (
                median_pct, clip_val))
    else:
        ax.text(0.5, 0.5, "No finite percent diffs",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Percent differences")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname = os.path.join(out_dir, "%s.png" % col)
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return {
        "column": col,
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_abs_diff": float(np.max(abs_diff)),
        "median_pct_diff": float(
            np.median(pct_diff_finite)) if len(pct_diff_finite) else None,
        "p99_pct_diff": float(
            np.percentile(pct_diff_finite, 99)) if len(
                pct_diff_finite) else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", default="/tmp/fixture_diffs",
        help="Output directory for figures (default: /tmp/fixture_diffs)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading fixture...")
    fixture_df, metadata = load_fixture()
    print("  %d rows, %d alleles" % (
        len(fixture_df), fixture_df.allele.nunique()))

    print("Generating PyTorch predictions...")
    predicted_df = generate_predictions(fixture_df)

    numeric_columns = [
        c for c in fixture_df.columns
        if c not in BASE_COLUMNS + STRING_COLUMNS
    ]

    global alleles_global
    alleles_global = fixture_df["allele"].values

    print("Plotting %d numeric outputs..." % len(numeric_columns))
    summary_rows = []
    for col in sorted(numeric_columns):
        tf_vals = fixture_df[col].to_numpy(dtype=np.float64)
        pt_vals = predicted_df[col].to_numpy(dtype=np.float64)
        valid = np.isfinite(tf_vals) & np.isfinite(pt_vals)
        if valid.sum() == 0:
            print("  %s: no valid values, skipping" % col)
            continue
        stats = plot_output(
            col, tf_vals[valid], pt_vals[valid], args.out_dir)
        summary_rows.append(stats)
        print("  %s: mean_abs=%.2e, max_abs=%.2e" % (
            col, stats["mean_abs_diff"], stats["max_abs_diff"]))

    # String columns: report match rate
    for col in STRING_COLUMNS:
        match = (
            predicted_df[col].astype(str).values
            == fixture_df[col].astype(str).values)
        print("  %s: %d/%d match (%.1f%%)" % (
            col, match.sum(), len(match), 100 * match.mean()))

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\nSummary written to %s" % summary_path)
    print("Figures written to %s" % args.out_dir)


if __name__ == "__main__":
    main()
