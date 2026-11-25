#!/usr/bin/env python3
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

"""Verify that generated pseudosequences preserve every trained allele verbatim."""
from __future__ import annotations

import argparse
import csv
import sys

import pandas

from mhcflurry.common import AlleleKeyResolver


def _load_pseudosequence_csv(path: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as fd:
        reader = csv.reader(fd)
        header = next(reader, None)
        if header is not None and len(header) >= 2:
            col0 = header[0].strip().lower()
            col1 = header[1].strip().lower()
            if col0 not in ("allele", "normalized_allele", "") and col1 not in (
                "sequence",
                "pseudosequence",
            ):
                mapping[header[0].strip()] = header[1].strip().upper()
        for row in reader:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                mapping[row[0].strip()] = row[1].strip().upper()
    return mapping


def verify_trained_allele_parity(
    *,
    baseline_csv: str,
    generated_csv: str,
    training_data_csv: str | None = None,
    expected_length: int = 39,
) -> dict[str, int]:
    """Assert 0 missing alleles and 0 character diffs on every trained allele."""
    baseline = _load_pseudosequence_csv(baseline_csv)
    generated = _load_pseudosequence_csv(generated_csv)

    bad_lengths = [
        (allele, len(seq))
        for allele, seq in generated.items()
        if len(seq) != expected_length
    ]
    if bad_lengths:
        raise ValueError(
            "Generated table %s has %d entries with length != %d (e.g. %r)"
            % (generated_csv, len(bad_lengths), expected_length, bad_lengths[:5])
        )

    if training_data_csv:
        df = pandas.read_csv(training_data_csv, usecols=["allele"])
        resolver = AlleleKeyResolver(baseline)
        trained_keys = sorted(
            {
                resolved
                for raw in df["allele"].dropna().astype(str).unique()
                if (resolved := resolver.resolve_to_key(raw)) is not None
            }
        )
    else:
        trained_keys = sorted(baseline)

    missing = [a for a in trained_keys if a not in generated]
    changed = [
        (a, baseline[a], generated[a])
        for a in trained_keys
        if a in generated and generated[a] != baseline[a]
    ]
    if missing or changed:
        details = []
        if missing:
            details.append(
                "%d trained alleles missing (e.g. %r)" % (len(missing), missing[:10])
            )
        if changed:
            preview = [
                "%s: baseline=%s != generated=%s" % item for item in changed[:10]
            ]
            details.append(
                "%d trained alleles changed (%s)" % (len(changed), "; ".join(preview))
            )
        raise ValueError(
            "Trained-allele pseudosequence parity check failed: " + "; ".join(details)
        )

    return {
        "baseline_alleles": len(baseline),
        "generated_alleles": len(generated),
        "net_added_alleles": len(set(generated) - set(baseline)),
        "trained_alleles_verified": len(trained_keys),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline pseudosequence CSV (e.g. 2.3.0 allele_sequences.csv).",
    )
    parser.add_argument(
        "--generated",
        required=True,
        help="Newly built pseudosequence CSV to verify.",
    )
    parser.add_argument(
        "--training-data",
        default=None,
        help="Optional curated_training_data.csv[.bz2] listing trained alleles.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=39,
        help="Expected pseudosequence length (default: 39).",
    )
    args = parser.parse_args(argv)
    stats = verify_trained_allele_parity(
        baseline_csv=args.baseline,
        generated_csv=args.generated,
        training_data_csv=args.training_data,
        expected_length=args.length,
    )
    print(
        "Parity OK: verified %d/%d trained alleles bit-for-bit identical "
        "(baseline=%d -> generated=%d, +%d net new alleles)."
        % (
            stats["trained_alleles_verified"],
            stats["trained_alleles_verified"],
            stats["baseline_alleles"],
            stats["generated_alleles"],
            stats["net_added_alleles"],
        )
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
