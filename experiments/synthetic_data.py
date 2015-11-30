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
Helper functions for creating synthetic pMHC affinities from allele similarities
and true binding data
"""

from collections import defaultdict

import pandas as pd

from common import curry_dictionary


def load_sims_dict(csv_path, allele_pair_keys=True):
    """
    Read in allele-to-allele similarities as DataFrame
    """
    sims_df = pd.read_csv(csv_path)
    # extract dictionary mapping pairs of alleles to similarity
    sims_dict = {
        (row["allele_A"], row["allele_B"]): row["similarity"]
        for (_, row)
        in sims_df.iterrows()
    }
    if not allele_pair_keys:
        return curry_dictionary(sims_dict)
    else:
        return sims_dict


def create_reverse_peptide_affinity_lookup_dict(allele_datasets):
    """
    Create reverse-lookup dictionary mapping each peptide to a list of triplets:
        [(allele, regression_output, weight), ...]
    """
    peptide_affinities_dict = defaultdict(list)
    for allele, dataset in allele_datasets.items():
        for peptide, y, weight in zip(
                dataset.peptides, dataset.Y, dataset.weights):
            entry = (allele, y, weight)
            peptide_affinities_dict[peptide].append(entry)
    return peptide_affinities_dict


def synthesize_affinities_for_single_allele(
        similarities,
        peptide_to_affinities,
        smoothing=0.005,
        exponent=2.0,
        exclude_alleles=[]):
    """
    Parameters
    ----------
    similarities : dict
        Dictionary mapping allele names to their similarity to the
        allele of interest.

    peptide_to_affinities : dict
        Dictionary mapping peptide sequence to list of triplets containing
        the following fields:
            - allele
            - log-scaled affinity normalized to 0.0 = no binding, 1.0 = strong
            - sample weight

    smoothing : float

    exponent : float

    exclude_alleles : collection of allele names
        Don't use these alleles for synthesizing new affinities

    Returns dictionary mapping peptide sequences to affinities.
    """
    assert isinstance(similarities, dict), \
        "Wrong type for similarities: %s" % type(similarities)
    results = {}
    for peptide, affinities in peptide_to_affinities.items():
        total = 0.0
        denom = 0.0
        for (allele, y, sample_weight) in affinities:
            if allele in exclude_alleles:
                continue
            sim = similarities.get(allele, 0)
            if sim == 0:
                continue
            combined_weight = sim ** exponent * sample_weight
            total += combined_weight * y
            denom += combined_weight
        if denom > 0.0:
            results[peptide] = total / (smoothing + denom)
    return results


def synthesize_affinities_for_all_alleles(
        allele_datasets,
        pairwise_allele_similarities,
        smoothing=0.005,
        exponent=2.0):
    """
    allele_datasets : dict
        Maps each allele name to AlleleData object with fields
            - peptides : list of strings
            - Y : list of float affinities between 0 and 1
            - weights : list of float sample weights between 0 and inf

    pairwise_allele_similarities : dict
        Dictionary from allele -> allele -> value between 0..1

    smoothing : float

    exponent : float
    """
    peptide_to_affinities = create_reverse_peptide_affinity_lookup_dict(
        allele_datasets)
    all_predictions = {}
    allele_names = set(pairwise_allele_similarities.keys())
    for allele in allele_names:
        all_predictions[allele] = synthesize_affinities_for_single_allele(
            similarities=pairwise_allele_similarities[allele],
            peptide_to_affinities=peptide_to_affinities,
            smoothing=smoothing,
            exponent=exponent)
    return all_predictions
