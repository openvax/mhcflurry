#!/usr/bin/env python
#
# Copyright (c) 2015. Mount Sinai School of Medicine
#
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

import argparse

import mhcflurry

from dataset_paths import PETERS2009_CSV_PATH
from allele_similarities import (
    compute_partial_similarities_from_peptide_overlap,
    fill_in_similarities,
    save_csv
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--binding-data-csv",
    default=PETERS2009_CSV_PATH)

parser.add_argument(
    "--min-overlap-weight",
    default=1.0,
    help="Minimum overlap weight between pair of alleles",
    type=float)

parser.add_argument(
    "--max-ic50",
    default=50000.0,
    type=float)

parser.add_argument(
    "--only-human",
    default=False,
    action="store_true")

parser.add_argument(
    "--raw-similarities-output-path",
    help="CSV file which contains incomplete allele similarities")

parser.add_argument(
    "--complete-similarities-output-path",
    help="CSV file which contains allele similarities after matrix completion")

parser.add_argument(
    "--raw-heatmap-output-path",
    help="PNG file to save heatmap of incomplete similarities matrix")

parser.add_argument(
    "--complete-heatmap-output-path",
    help="PNG file to save heatmap of complete similarities matrix")


def print_dataset_sizes(allele_groups):
    print("---\nDataset Sizes\n---")
    for (allele_name, g) in sorted(allele_groups.items()):
        print("%s: total=%d, 8mer=%d, 9mer=%d, 10mer=%d, 11mer=%d" % (
            allele_name,
            len(g),
            sum(len(k) == 8 for k in g.keys()),
            sum(len(k) == 9 for k in g.keys()),
            sum(len(k) == 10 for k in g.keys()),
            sum(len(k) == 11 for k in g.keys()),
        ))
    print("---")


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    df, peptide_column_name = mhcflurry.data.load_dataframe(
        args.binding_data_csv,
        max_ic50=args.max_ic50,
        only_human=args.only_human)
    allele_groups = {
        allele_name: {
            row[peptide_column_name]: row["regression_output"]
            for (_, row) in group.iterrows()
        }
        for (allele_name, group) in df.groupby("mhc")
    }
    print_dataset_sizes(allele_groups)

    raw_sims_dict, overlap_counts, overlap_weights = \
        compute_partial_similarities_from_peptide_overlap(
            allele_groups,
            min_weight=args.min_overlap_weight)

    if args.raw_similarities_output_path:
        save_csv(
            args.raw_similarities_output_path,
            raw_sims_dict,
            overlap_counts,
            overlap_weights)

    complete_sims_dict = fill_in_similarities(
        curried_raw_sims_dict=raw_sims_dict,
        allele_to_peptide_to_affinity=allele_groups,
        raw_sims_heatmap_path=args.raw_heatmap_output_path,
        complete_sims_heatmap_path=args.complete_heatmap_output_path)

    print("-- Added %d/%d allele similarities" % (
        len(complete_sims_dict) - len(raw_sims_dict),
        len(complete_sims_dict)))

    if args.complete_similarities_output_path:
        save_csv(
            args.complete_similarities_output_path,
            complete_sims_dict,
            overlap_counts,
            overlap_weights)
