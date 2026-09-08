#!/usr/bin/env python
"""Summarize paired affinity-factorial comparisons without merging sides."""

import argparse
import csv
import json
from pathlib import Path


METRICS = ("roc_auc", "pr_auc", "ppv_at_n")
PUBLIC_COMPARISON_NAME = "comparison-vs-public-no-additional-ms"
BASELINE_PUBLIC_COMPARISON_NAME = "baseline-vs-public-no-additional-ms"


def side_metrics(summary, side):
    """Return flattened macro/micro metrics for one comparison side."""
    result = {}
    for metric in METRICS:
        result["macro_" + metric] = summary["macro_mean_over_alleles"][
            metric
        ][side]
        result["micro_" + metric] = summary["micro_pooled"][side][metric]
    return result


def benchmark_identity(summary, path):
    """Return the scored-row identity or reject an obsolete comparison."""
    result = summary.get("benchmark_identity")
    if not result or not result.get("sha256"):
        raise ValueError(
            "Direct public comparison lacks benchmark_identity: %s" % path)
    return result


def summarize(factorial_dir):
    """Write and return one record per completed factorial condition."""
    factorial_dir = Path(factorial_dir)
    manifest = json.loads((factorial_dir / "manifest.json").read_text())
    baseline = manifest["baseline_condition"]
    records = []
    public_benchmark_identity = None
    for condition_record in manifest["records"]:
        condition = condition_record["condition"]
        if condition == baseline:
            summary_path = (
                factorial_dir
                / BASELINE_PUBLIC_COMPARISON_NAME
                / "affinity"
                / "summary.json"
            )
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            candidate = side_metrics(summary, "a")
            reference = candidate
            public_candidate = candidate
            public = side_metrics(summary, "b")
            direct_public_summary = summary
            comparison = "baseline_vs_public"
        else:
            summary_path = (
                factorial_dir
                / condition
                / "comparison-vs-baseline"
                / "affinity"
                / "summary.json"
            )
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            candidate = side_metrics(summary, "a")
            reference = side_metrics(summary, "b")
            public_summary_path = (
                factorial_dir
                / condition
                / PUBLIC_COMPARISON_NAME
                / "affinity"
                / "summary.json"
            )
            public = None
            public_candidate = None
            direct_public_summary = None
            if public_summary_path.exists():
                public_summary = json.loads(public_summary_path.read_text())
                public_candidate = side_metrics(public_summary, "a")
                public = side_metrics(public_summary, "b")
                direct_public_summary = public_summary
            comparison = "candidate_vs_baseline"
        direct_identity = None
        if direct_public_summary is not None:
            direct_identity = benchmark_identity(
                direct_public_summary,
                summary_path if condition == baseline else public_summary_path,
            )
            if public_benchmark_identity is None:
                public_benchmark_identity = direct_identity
            elif direct_identity != public_benchmark_identity:
                raise ValueError(
                    "Direct public comparisons used different benchmark rows: "
                    "%s has %s, expected %s" % (
                        condition,
                        direct_identity,
                        public_benchmark_identity,
                    ))
        record = {
            **condition_record,
            "comparison": comparison,
            "n_rows": summary["n_rows"],
            "n_hits": summary["n_hits"],
            "n_alleles_reported": summary["n_alleles_reported"],
            "public_benchmark_identity_sha256": (
                direct_identity["sha256"] if direct_identity else None),
            "public_n_rows": (
                direct_public_summary["n_rows"]
                if direct_public_summary is not None else None),
            "public_n_hits": (
                direct_public_summary["n_hits"]
                if direct_public_summary is not None else None),
        }
        for key, value in candidate.items():
            record[key] = value
            record[key + "_baseline"] = reference[key]
            record[key + "_delta"] = value - reference[key]
            record[key + "_relative_delta"] = (
                (value - reference[key]) / reference[key]
                if reference[key]
                else None
            )
            public_value = public[key] if public is not None else None
            public_candidate_value = (
                public_candidate[key] if public_candidate is not None else None)
            record[key + "_public"] = public_value
            record[key + "_vs_public_candidate"] = public_candidate_value
            record[key + "_vs_public_delta"] = (
                public_candidate_value - public_value
                if public_value is not None else None
            )
            record[key + "_vs_public_relative_delta"] = (
                (public_candidate_value - public_value) / public_value
                if public_value
                else None
            )
        allele_count = summary.get("allele_count", {})
        for metric in METRICS:
            record["alleles_candidate_better_" + metric] = allele_count.get(
                "a_better_" + metric
            )
            record["alleles_baseline_better_" + metric] = allele_count.get(
                "b_better_" + metric
            )
        records.append(record)

    if records:
        fieldnames = list(records[0])
        out_path = factorial_dir / "summary.csv"
        with out_path.open("w", newline="") as fd:
            writer = csv.DictWriter(fd, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    return records


def main(argv=None):
    """Run the summarizer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("factorial_dir")
    args = parser.parse_args(argv)
    records = summarize(args.factorial_dir)
    print(json.dumps(records, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
