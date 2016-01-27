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
from fancyimpute import NuclearNormMinimization

from common import matrix_to_dictionary, curry_dictionary


def compute_similarities_for_single_allele_from_peptide_overlap(
        allele_name,
        allele_to_peptide_to_affinity,
        allele_name_length=None,
        min_weight=1.0):
    """
    Parameters
    ----------
    allele_name : str

    allele_to_peptide_to_affinity : dict
        Dictionary from allele anmes to a dictionary of (peptide -> affinity)

    Returns three dictionaries, mapping alleles to
        - similarity
        - number of overlapping peptides
        - total weight of overlapping affinitie
    """
    sims = {}
    overlaps = {}
    weights = {}
    dataset_a = allele_to_peptide_to_affinity[allele_name]
    peptide_set_a = set(dataset_a.keys())
    for other_allele, dataset_b in allele_to_peptide_to_affinity.items():
        if allele_name_length and len(other_allele) != allele_name_length:
            continue
        peptide_set_b = set(dataset_b.keys())
        intersection = peptide_set_a.intersection(peptide_set_b)
        overlaps[other_allele] = len(intersection)
        total = 0.0
        weight = 0.0
        for peptide in intersection:
            ya = dataset_a[peptide]
            yb = dataset_b[peptide]
            minval = min(ya, yb)
            maxval = max(ya, yb)
            total += minval
            weight += maxval
        weights[other_allele] = weight
        if weight > min_weight:
            sims[other_allele] = total / weight
    return sims, overlaps, weights


def compute_partial_similarities_from_peptide_overlap(
        allele_to_peptide_to_affinity,
        min_weight=1.0,
        allele_name_length=None):
    """
    Determine similarity between pairs of alleles by examining
    affinity values for overlapping peptides. Returns curried dictionaries
    mapping allele -> allele -> float, where the final value is one of:
        - similarity between alleles
        - # overlapping peptides
        - sum weight of overlapping affinities
    """
    sims = {}
    overlaps = {}
    weights = {}
    for allele_name in allele_to_peptide_to_affinity.keys():
        if allele_name_length and len(allele_name) != allele_name_length:
            continue
        curr_sims, curr_overlaps, curr_weights = \
            compute_similarities_for_single_allele_from_peptide_overlap(
                allele_name=allele_name,
                allele_to_peptide_to_affinity=allele_to_peptide_to_affinity,
                allele_name_length=allele_name_length,
                min_weight=min_weight)
        sims[allele_name] = curr_sims
        overlaps[allele_name] = curr_overlaps
        weights[allele_name] = curr_weights
    return sims, overlaps, weights


def build_incomplete_similarity_matrix(
        allele_to_peptide_to_affinity,
        curried_sims_dict):
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
            similarity = curried_sims_dict.get(a, {}).get(b, np.nan)
            sims_incomplete_matrix[(ai, bi)] = similarity
    return allele_list, allele_order, sims_incomplete_matrix


def save_heatmap(matrix, allele_names, filename):
    seaborn.set_context("paper")

    with seaborn.plotting_context(font_scale=1):
        figure = seaborn.plt.figure(figsize=(20, 18))
        ax = figure.add_axes()
        heatmap = seaborn.heatmap(
            data=matrix,
            xticklabels=allele_names,
            yticklabels=allele_names,
            linewidths=0,
            annot=False,
            ax=ax,
            fmt=".2g")
        heatmap.set_xticklabels(labels=allele_names, rotation=45)
        heatmap.set_yticklabels(labels=allele_names, rotation=45)
        figure.savefig(filename)


def fill_in_similarities(
        curried_raw_sims_dict,
        allele_to_peptide_to_affinity,
        raw_sims_heatmap_path=None,
        complete_sims_heatmap_path=None,
        curried_overlap_weights=None,
        scalar_error_tolerance=0.0001):
    """
    Given an incomplete dictionary of pairwise allele similarities and
    a dictionary of binding data, generate the completed similarities
    """

    allele_list, allele_order, sims_matrix = build_incomplete_similarity_matrix(
        allele_to_peptide_to_affinity,
        curried_sims_dict=curried_raw_sims_dict)

    missing = np.isnan(sims_matrix)

    if curried_overlap_weights:
        error_tolerance = np.ones_like(sims_matrix)
        for allele_a, a_dict in curried_overlap_weights.items():
            for allele_b, weight in a_dict.items():
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

    solver = NuclearNormMinimization(
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

    complete_sims_dict = curry_dictionary(
        matrix_to_dictionary(
            sims=sims_complete_matrix,
            allele_list=allele_list))
    return complete_sims_dict


def save_csv(filename, sims, overlap_counts, overlap_weights):
    """
    Save pairwise allele similarities in CSV file.

    Assumes the dictionaries sims, overlap_counts, and overlap_weights are
    'curried', meaning that their keys are single allele names which map
    to nested dictionaries from other allele names to numeric values.
    As a type signature, this means the dictionaries are of the form
        (allele -> allele -> number)
    and not
        (allele, allele) -> number
    """
    with open(filename, "w") as f:
        f.write("allele_A,allele_B,similarity,count,weight\n")
        for (a, a_row) in sorted(sims.items()):
            a_counts = overlap_counts.get(a, {})
            a_weights = overlap_weights.get(a, {})
            for (b, similarity) in sorted(a_row.items()):
                count = a_counts.get(b, 0)
                weight = a_weights.get(b, 0.0)
                f.write("%s,%s,%0.4f,%d,%0.4f\n" % (
                    a,
                    b,
                    similarity,
                    count,
                    weight))


def compute_allele_similarities(allele_to_peptide_to_affinity, min_weight=0.1):
    """
    Compute pairwise allele similarities from binding data,
    complete similarity matrix for alleles which lack sufficient data,
    returns:
        - dictionary of allele similarities
        - dictionary of # overlapping peptides between alleles
        - dictionary of "weight" of overlapping peptides between alleles
    """
    assert isinstance(allele_to_peptide_to_affinity, dict), \
        "Wrong type, expected dict but got %s" % (
            type(allele_to_peptide_to_affinity),)
    raw_sims_dict, overlap_counts, overlap_weights = \
        compute_partial_similarities_from_peptide_overlap(
            allele_to_peptide_to_affinity=allele_to_peptide_to_affinity,
            min_weight=min_weight)

    complete_sims_dict = fill_in_similarities(
        curried_raw_sims_dict=raw_sims_dict,
        allele_to_peptide_to_affinity=allele_to_peptide_to_affinity,
        curried_overlap_weights=overlap_weights)

    return complete_sims_dict, overlap_counts, overlap_weights
