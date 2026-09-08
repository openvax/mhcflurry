#!/usr/bin/env python
"""Summarize architecture-stratified affinity-factorial comparisons."""

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


def dominance_label(deltas):
    """Classify strict metric dominance against the matched baseline."""
    if all(value > 0 for value in deltas):
        return "strictly_dominates_baseline"
    if all(value < 0 for value in deltas):
        return "strictly_dominated_by_baseline"
    return "mixed"


def summarize(factorial_dir):
    """Write and return one record per completed condition/architecture."""
    factorial_dir = Path(factorial_dir)
    architecture_dir = factorial_dir / "architecture_evaluation"
    manifest = json.loads((factorial_dir / "manifest.json").read_text())
    baseline = manifest["baseline_condition"]
    records = []
    for condition_record in manifest["records"]:
        condition = condition_record["condition"]
        subset_root = architecture_dir / "subsets" / condition
        for provenance_path in sorted(subset_root.glob(
                "architecture_*/subset_provenance.json")):
            provenance = json.loads(provenance_path.read_text())
            architecture_num = provenance["architecture_num"]
            architecture_name = "architecture_%d" % architecture_num
            if condition == baseline:
                summary_path = (
                    architecture_dir
                    / "baseline-vs-public"
                    / architecture_name
                    / "affinity"
                    / "summary.json"
                )
                comparison = "baseline_vs_public"
            else:
                summary_path = (
                    architecture_dir
                    / "comparisons"
                    / condition
                    / (architecture_name + "-vs-baseline")
                    / "affinity"
                    / "summary.json"
                )
                comparison = "candidate_vs_baseline"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            candidate = side_metrics(summary, "a")
            reference = (
                candidate if condition == baseline else side_metrics(summary, "b")
            )
            architecture = provenance["architecture"]
            record = {
                **condition_record,
                "architecture_num": architecture_num,
                "topology": architecture["topology"],
                "layer_sizes": json.dumps(architecture["layer_sizes"]),
                "dense_layer_l1_regularization": architecture[
                    "dense_layer_l1_regularization"
                ],
                "comparison": comparison,
                "n_rows": summary["n_rows"],
                "n_hits": summary["n_hits"],
                "n_alleles_reported": summary["n_alleles_reported"],
            }
            deltas = []
            for key, value in candidate.items():
                reference_value = reference[key]
                delta = value - reference_value
                record[key] = value
                record[key + "_baseline"] = reference_value
                record[key + "_delta"] = delta
                record[key + "_relative_delta"] = (
                    delta / reference_value if reference_value else None
                )
                if condition != baseline:
                    deltas.append(delta)
            record["metric_dominance"] = (
                "reference" if condition == baseline else dominance_label(deltas)
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
        out_path = architecture_dir / "summary.csv"
        with out_path.open("w", newline="") as fd:
            writer = csv.DictWriter(fd, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return records


def main(argv=None):
    """Run the summarizer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("factorial_dir")
    args = parser.parse_args(argv)
    print(json.dumps(summarize(args.factorial_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
