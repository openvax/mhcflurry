#!/usr/bin/env python3
"""Audit training row identity, multiplicities, order, and novel examples."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas


def compare_rows(reference, candidate, columns):
    """Compare exact parsed rows on explicit columns, including duplicates."""
    reference = reference[columns].fillna("").astype(str)
    candidate = candidate[columns].fillna("").astype(str)
    left = reference.value_counts(sort=False, dropna=False).rename("reference_count")
    right = candidate.value_counts(sort=False, dropna=False).rename("candidate_count")
    counts = pandas.concat([left, right], axis=1).fillna(0).astype("int64")
    shared = counts.min(axis=1)
    summary = {
        "reference_rows": len(reference), "candidate_rows": len(candidate),
        "shared_rows_with_multiplicity": int(shared.sum()),
        "reference_only_rows_with_multiplicity": int((counts.reference_count - shared).sum()),
        "candidate_only_rows_with_multiplicity": int((counts.candidate_count - shared).sum()),
        "same_row_multiset": bool((counts.reference_count == counts.candidate_count).all()),
        "same_ordered_rows": reference.reset_index(drop=True).equals(candidate.reset_index(drop=True)),
        "identity_columns": columns,
    }
    return summary, counts.loc[counts.reference_count != counts.candidate_count].reset_index()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--columns", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    # Preserve literal labels and numbers rather than inventing equivalences.
    # Supply normalized tables explicitly if semantic normalization is desired.
    reference = pandas.read_csv(args.reference, dtype=str, keep_default_na=False)
    candidate = pandas.read_csv(args.candidate, dtype=str, keep_default_na=False)
    summary, differences = compare_rows(reference, candidate, args.columns)
    summary["files"] = {
        name: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for name, path in (("reference", args.reference), ("candidate", args.candidate))
    }
    summary["comparison"] = "exact parsed values on explicit columns; counts preserve duplicates"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    differences.to_csv(out / "row_differences.csv.bz2", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
