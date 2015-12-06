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

"""
Helper functions for computing pairwise similarities between alleles
"""

import numpy as np
import seaborn
from fancyimpute import ConvexSolver

from common import matrix_to_dictionary


def compute_partial_similarities_from_peptide_overlap(
        allele_to_peptide_to_affinity,
        min_weight=1.0,
        allele_name_length=None):
    """
    Determine similarity between pairs of alleles by examining
    affinity values for overlapping peptides
    """
    sims = {}
    overlaps = {}
    weights = {}
    for a, da in allele_to_peptide_to_affinity.items():
        if allele_name_length and len(a) != allele_name_length:
            continue
        peptide_set_a = set(da.keys())
        for b, db in allele_to_peptide_to_affinity.items():
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


def build_incomplete_similarity_matrix(
        allele_to_peptide_to_affinity,
        sims_dict):
    allele_list = list(sorted(allele_to_peptide_to_affinity.keys()))
    allele_order = {
        allele_name: i
        for (i, allele_name) in enumerate(allele_list)
    }
    n_alleles = len(allele_list)
    sims_incomplete_matrix = np.zeros((n_alleles, n_alleles), dtype=float)

    for a, ai in allele_order.items():
        for b, bi in allele_order.items():
            # if allele pair is missing from similarities dict then
            # fill its slot with NaN, indicating missing data
            sims_incomplete_matrix[ai, bi] = sims_dict.get((a, b), np.nan)
    return allele_list, allele_order, sims_incomplete_matrix


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


def fill_in_similarities(
        raw_sims_dict,
        allele_to_peptide_to_affinity,
        raw_sims_heatmap_path=None,
        complete_sims_heatmap_path=None,
        overlap_weights=None,
        scalar_error_tolerance=0.0001):
    """
    Given an incomplete dictionary of pairwise allele similarities and
    a dictionary of binding data, generate the completed similarities
    """

    allele_list, allele_order, sims_matrix = build_incomplete_similarity_matrix(
        allele_to_peptide_to_affinity,
        sims_dict=raw_sims_dict)

    missing = np.isnan(sims_matrix)

    if overlap_weights:
        error_tolerance = np.ones_like(sims_matrix)
        for ((allele_a, allele_b), weight) in overlap_weights.items():
            i = allele_order[allele_a]
            j = allele_order[allele_b]
            error_tolerance[i, j] = 2.0 ** -weight
        print(
            "-- Error tolerance distribution: min %f, max %f, median %f" % (
                error_tolerance[~missing].min(),
                error_tolerance[~missing].max(),
                np.median(error_tolerance[~missing])))
    else:
        error_tolerance = scalar_error_tolerance

    if raw_sims_heatmap_path:
        save_heatmap(
            sims_matrix,
            allele_list,
            raw_sims_heatmap_path)

    print("Completing %s similarities matrix with %d missing entries" % (
        sims_matrix.shape,
        missing.sum()))

    solver = ConvexSolver(
        require_symmetric_solution=True,
        min_value=0.0,
        max_value=1.0,
        error_tolerance=error_tolerance)

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


def save_csv(filename, sims, overlap_counts, overlap_weights):
    """
    Save pairwise allele similarities in CSV file
    """
    with open(filename, "w") as f:
        f.write("allele_A,allele_B,similarity,count,weight\n")
        for (a, b), s in sorted(sims.items()):
            count = overlap_counts.get((a, b), 0)
            weight = overlap_weights.get((a, b), 0.0)
            f.write("%s,%s,%0.4f,%d,%0.4f\n" % (a, b, s, count, weight))


def compute_allele_similarities(allele_to_peptide_to_affinity, min_weight=0.1):
    """
    Compute pairwise allele similarities from binding data,
    complete similarity matrix for alleles which lack sufficient data,
    returns:
        - dictionary of allele similarities
        - dictionary of # overlapping peptides between alleles
        - dictionary of "weight" of overlapping peptides between alleles
    """
    raw_sims_dict, overlap_counts, overlap_weights = \
        compute_partial_similarities_from_peptide_overlap(
            allele_to_peptide_to_affinity,
            min_weight=min_weight)

    complete_sims_dict = fill_in_similarities(
        raw_sims_dict=raw_sims_dict,
        allele_to_peptide_to_affinity=allele_to_peptide_to_affinity,
        overlap_weights=overlap_weights)

    return complete_sims_dict, overlap_counts, overlap_weights
