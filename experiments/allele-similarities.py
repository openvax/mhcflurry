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

from fancyimpute import ConvexSolver
import numpy as np
import mhcflurry
import seaborn

from dataset_paths import PETERS2009_CSV_PATH

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


def compute_similarities(
        allele_groups,
        min_weight=1.0,
        allele_name_length=None):
    sims = {}
    overlaps = {}
    weights = {}
    for a, da in allele_groups.items():
        if allele_name_length and len(a) != allele_name_length:
            continue
        peptide_set_a = set(da.keys())
        for b, db in allele_groups.items():
            if allele_name_length and len(b) != allele_name_length:
                continue
            peptide_set_b = set(db.keys())
            intersection = peptide_set_a.intersection(peptide_set_b)
            overlaps[(a, b)] = len(intersection)
            total = 0.0
            weight = 0.0
            for peptide in intersection:
                ya = da[peptide]
                yb = db[peptide]
                minval = min(ya, yb)
                maxval = max(ya, yb)
                total += minval
                weight += maxval
            weights[(a, b)] = weight
            if weight > min_weight:
                sims[(a, b)] = total / weight
    return sims, overlaps, weights


def save_heatmap(matrix, allele_names, filename):
    seaborn.set_context("paper")

    with seaborn.plotting_context(font_scale=1):
        figure = seaborn.plt.figure(figsize=(20, 18))
        ax = figure.add_axes()
        seaborn.heatmap(
            data=matrix,
            xticklabels=allele_names,
            yticklabels=allele_names,
            linewidths=0,
            annot=False,
            ax=ax,
            fmt=".2g")
        figure.savefig(filename)


def save_csv(filename, sims, overlap_counts, overlap_weights):
    with open(filename, "w") as f:
        f.write("allele_A,allele_B,similarity,count,weight\n")
        for (a, b), s in sorted(sims.items()):
            count = overlap_counts.get((a, b), 0)
            weight = overlap_weights.get((a, b), 0.0)
            f.write("%s,%s,%0.4f,%d,%0.4f\n" % (a, b, s, count, weight))


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


def build_incomplete_similarity_matrix(allele_groups, sims):
    allele_list = list(sorted(allele_groups.keys()))
    allele_order = {
        allele_name: i
        for (i, allele_name) in enumerate(sorted(allele_groups.keys()))
    }
    n_alleles = len(allele_list)
    sims_incomplete_matrix = np.zeros((n_alleles, n_alleles), dtype=float)

    for a, ai in allele_order.items():
        for b, bi in allele_order.items():
            # if allele pair is missing from similarities dict then
            # fill its slot with NaN, indicating missing data
            sims_incomplete_matrix[ai, bi] = sims.get((a, b), np.nan)
    return allele_list, allele_order, sims_incomplete_matrix


def matrix_to_dictionary(sims, allele_list):
    sims_dict = {}
    for i in range(sims.shape[0]):
        a = allele_list[i]
        for j in range(sims.shape[1]):
            b = allele_list[j]
            sims_dict[a, b] = sims[i, j]
    return sims_dict


def fill_in_similarities(
        raw_sims_dict,
        allele_datasets,
        raw_sims_heatmap_path=None,
        complete_sims_heatmap_path=None):
    allele_list, allele_order, sims_matrix = build_incomplete_similarity_matrix(
        allele_datasets,
        sims=raw_sims_dict)

    if raw_sims_heatmap_path:
        save_heatmap(
            sims_matrix,
            allele_list,
            raw_sims_heatmap_path)

    print("Completing %s similarities matrix with %d missing entries" % (
        sims_matrix.shape,
        np.isnan(sims_matrix).sum()))

    solver = ConvexSolver(
        require_symmetric_solution=True,
        min_value=0.0,
        max_value=1.0,
        error_tolerance=0.0001)

    sims_complete_matrix = solver.complete(sims_matrix)

    if complete_sims_heatmap_path:
        save_heatmap(
            sims_complete_matrix,
            allele_list,
            complete_sims_heatmap_path)

    complete_sims_dict = matrix_to_dictionary(
        sims=sims_complete_matrix,
        allele_list=allele_list)
    return complete_sims_dict

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    df = mhcflurry.data_helpers.load_dataframe(
        args.binding_data_csv,
        max_ic50=args.max_ic50,
        only_human=args.only_human)
    allele_groups = {
        allele_name: {
            row["sequence"]: row["regression_output"]
            for (_, row) in group.iterrows()
        }
        for (allele_name, group) in df.groupby("mhc")
    }
    print_dataset_sizes(allele_groups)

    raw_sims_dict, overlap_counts, overlap_weights = compute_similarities(
        allele_groups,
        min_weight=args.min_overlap_weight)

    if args.raw_similarities_output_path:
        save_csv(
            args.raw_similarities_output_path,
            raw_sims_dict,
            overlap_counts,
            overlap_weights)

    complete_sims_dict = fill_in_similarities(
        raw_sims_dict=raw_sims_dict,
        allele_datasets=allele_groups,
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
