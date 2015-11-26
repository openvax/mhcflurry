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


def matrix_to_dictionary(sims, allele_list):
    sims_dict = {}
    for i in range(sims.shape[0]):
        a = allele_list[i]
        for j in range(sims.shape[1]):
            b = allele_list[j]
            sims_dict[a, b] = sims[i, j]
    return sims_dict


def compute_pairwise_allele_similarities(
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


def build_incomplete_similarity_matrix(allele_groups, sims_dict):
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
        allele_datasets,
        raw_sims_heatmap_path=None,
        complete_sims_heatmap_path=None):

    allele_list, allele_order, sims_matrix = build_incomplete_similarity_matrix(
        allele_datasets,
        sims_dict=raw_sims_dict)

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
