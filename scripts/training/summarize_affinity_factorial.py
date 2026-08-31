#!/usr/bin/env python
"""Summarize paired affinity-factorial comparisons without merging sides."""

import argparse
import csv
import json
from pathlib import Path


METRICS = ("roc_auc", "pr_auc", "ppv_at_n")


def side_metrics(summary, side):
    """Return flattened macro/micro metrics for one comparison side."""
    result = {}
    for metric in METRICS:
        result["macro_" + metric] = summary["macro_mean_over_alleles"][
            metric
        ][side]
        result["micro_" + metric] = summary["micro_pooled"][side][metric]
    return result


def summarize(factorial_dir):
    """Write and return one record per completed factorial condition."""
    factorial_dir = Path(factorial_dir)
    manifest = json.loads((factorial_dir / "manifest.json").read_text())
    baseline = manifest["baseline_condition"]
    records = []
    for condition_record in manifest["records"]:
        condition = condition_record["condition"]
        if condition == baseline:
            summary_path = (
                factorial_dir
                / "baseline-vs-public"
                / "affinity"
                / "summary.json"
            )
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            candidate = side_metrics(summary, "a")
            reference = candidate
            public = side_metrics(summary, "b")
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
                / "comparison-vs-public"
                / "affinity"
                / "summary.json"
            )
            public = None
            if public_summary_path.exists():
                public_summary = json.loads(public_summary_path.read_text())
                public = side_metrics(public_summary, "b")
            comparison = "candidate_vs_baseline"
        record = {
            **condition_record,
            "comparison": comparison,
            "n_rows": summary["n_rows"],
            "n_hits": summary["n_hits"],
            "n_alleles_reported": summary["n_alleles_reported"],
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
            record[key + "_public"] = public_value
            record[key + "_vs_public_delta"] = (
                value - public_value if public_value is not None else None
            )
            record[key + "_vs_public_relative_delta"] = (
                (value - public_value) / public_value
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
