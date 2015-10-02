#!/usr/bin/env python

"""
Combine 2013 Kim/Peters NetMHCpan dataset[*] with more recent IEDB entries

* = "Dataset size and composition impact the reliability..."
"""

from __future__ import (
    print_function,
    division,
    absolute_import,
    unicode_literals
)
from os.path import join
import pickle
from collections import Counter
import argparse

import pandas as pd

from mhcflurry.paths import CLASS1_DATA_DIRECTORY, CLASS1_DATA_CSV_PATH

IEDB_PICKLE_FILENAME = "iedb_human_class1_assay_datasets.pickle"
IEDB_PICKLE_PATH = join(CLASS1_DATA_DIRECTORY, IEDB_PICKLE_FILENAME)

PETERS_CSV_FILENAME = "bdata.20130222.mhci.public.1.txt"
PETERS_CSV_PATH = join(CLASS1_DATA_DIRECTORY, PETERS_CSV_FILENAME)

parser = argparse.ArgumentParser()

parser.add_argument("--ic50-fraction-tolerance",
    default=0.01,
    type=float,
    help=(
        "How much can the IEDB and NetMHCpan IC50 differ and still be"
        " considered compatible (as a fraction of the NetMHCpan value)"))

parser.add_argument("--min-assay-overlap-size",
    type=int,
    default=1,
    help="Minimum number of entries overlapping between IEDB assay and NetMHCpan data")


parser.add_argument("--min-assay-fraction-same",
    type=float,
    help="Minimum fraction of peptides whose IC50 values agree with the NetMHCpan data",
    default=0.9)

parser.add_argument("--iedb-pickle-path",
    default=IEDB_PICKLE_PATH,
    help="Path to .pickle file containing dictionary of IEDB assay datasets")

parser.add_argument("--netmhcpan-csv-path",
    default=PETERS_CSV_PATH,
    help="Path to CSV with NetMHCpan dataset from 2013 Peters paper")

parser.add_argument("--output-csv-path",
    default=CLASS1_DATA_CSV_PATH,
    help="Path to CSV of combined assay results")

parser.add_argument("--extra-dataset-csv-path",
    default=[],
    action="append",
    help="Additional CSV data source with columns (species, mhc, peptide, meas)")

if __name__ == "__main__":
    args = parser.parse_args()

    print("Reading %s..." % args.iedb_pickle_path)
    with open(args.iedb_pickle_path, "r") as f:
        iedb_datasets = pickle.load(f)

    print("Reading %s..." % args.netmhcpan_csv_path)
    nielsen_data = pd.read_csv(args.netmhcpan_csv_path, sep="\t")
    print("Size of 2013 NetMHCpan dataset: %d" % len(nielsen_data))

    new_allele_counts = Counter()
    combined_columns = {
        "species": list(nielsen_data["species"]),
        "mhc": list(nielsen_data["mhc"]),
        "peptide": list(nielsen_data["sequence"]),
        "peptide_length": list(nielsen_data["peptide_length"]),
        "meas": list(nielsen_data["meas"]),
    }

    all_datasets = {
        path: pd.read_csv(path) for path in args.extra_dataset_csv_path
    }
    all_datasets.update(iedb_datasets)
    for assay, assay_dataset in sorted(all_datasets.items(), key=lambda x: len(x[1])):
        joined = nielsen_data.merge(
            assay_dataset,
            left_on=["mhc", "sequence"],
            right_on=["mhc", "peptide"],
            how="outer")

        if len(joined) == 0:
            continue

        # drop NaN binding values and entries without values in both datasets
        left_missing = joined["meas"].isnull()
        right_missing = joined["value"].isnull()
        overlap_filter_mask = ~(left_missing | right_missing)
        filtered = joined[overlap_filter_mask]
        n_overlap = len(filtered)

        if n_overlap < args.min_assay_overlap_size:
            continue
        # let's count what fraction of this IEDB assay is within 1% of the values in the
        # Nielsen dataset
        tolerance = filtered["meas"] * args.ic50_fraction_tolerance
        abs_diff = (filtered["value"] - filtered["meas"]).abs()
        similar_values = abs_diff <= tolerance
        fraction_similar = similar_values.mean()
        print("Assay=%s, count=%d" % (assay, len(assay_dataset)))
        print("  # entries w/ values in both data sets: %d" % n_overlap)
        print("  fraction similar binding values=%0.4f" % fraction_similar)
        new_peptides = joined[left_missing & ~right_missing]
        if fraction_similar > args.min_assay_fraction_same:
            print("---")
            print("\t using assay: %s" % (assay,))
            print("---")
            combined_columns["mhc"].extend(new_peptides["mhc"])
            combined_columns["peptide"].extend(new_peptides["peptide"])
            combined_columns["peptide_length"].extend(new_peptides["peptide"].str.len())
            combined_columns["meas"].extend(new_peptides["value"])
            # TODO: make this work for non-human data
            combined_columns["species"].extend(["human"] * len(new_peptides))
            for allele in new_peptides["mhc"]:
                new_allele_counts[allele] += 1

    combined_df = pd.DataFrame(
        combined_columns,
        columns=["species", "mhc", "peptide", "peptide_length", "meas"])
    print("New entry allele distribution")
    for (allele, count) in new_allele_counts.most_common():
        print("%s: %d" % (allele, count))
    print("Combined DataFrame size: %d (+%d)" % (
            len(combined_df),
            len(combined_df) - len(nielsen_data)))
    print("Writing %s..." % args.output_csv_path)
    combined_df.to_csv(args.output_csv_path, index=False)
