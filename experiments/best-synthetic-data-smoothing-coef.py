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
Find best smoothing coefficient for creating synthetic data. The smoothing
coefficient adds artifical weight toward low binding affinities, which dominate
in cases where no similar allele has values for some peptide.

We're determining "best" as most predictive of already known binding affinities,
averaged across all given alleles.
"""

import argparse

import mhcflurry
from scipy import stats
import numpy as np

from common import curry_dictionary
from dataset_paths import PETERS2009_CSV_PATH
from synthetic_data import (
    synthesize_affinities_for_single_allele,
    create_reverse_peptide_affinity_lookup_dict,
)
from allele_similarities import (
    compute_pairwise_allele_similarities,
    fill_in_similarities
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--binding-data-csv",
    default=PETERS2009_CSV_PATH)

parser.add_argument(
    "--max-ic50",
    default=50000.0,
    type=float)

parser.add_argument(
    "--only-human",
    default=False,
    action="store_true")

parser.add_argument(
    "--allele-similarity-csv",
    required=True)

parser.add_argument(
    "--smoothing-coefs",
    default=[10.0 ** -power for power in np.arange(0, 5.0, 0.25)],
    type=lambda s: [float(si.strip()) for si in s.split(",")],
    help="Smoothing value used for peptides with low weight across alleles")

parser.add_argument(
    "--similarity-exponent",
    default=2.0,
    type=float,
    help="Affinities are synthesized by adding up y_ip * sim(i,j) ** exponent")


def evaluate_smoothing_coef(
        true_data,
        curried_allele_similarities,
        smoothing_coef):
    taus = []
    peptide_to_affinities = create_reverse_peptide_affinity_lookup_dict(
        true_data)
    for allele, dataset in true_data.items():
        allele_similarities = curried_allele_similarities[allele]
        true_data_peptide_set = set(dataset.peptides)
        true_data_peptide_list = list(dataset.peptides)
        # create a peptide -> (allele, affinity, weight) dictionary restricted
        # only to the peptides for which we have data for this allele
        restricted_reverse_lookup = {
            peptide: triplet
            for (peptide, triplet) in peptide_to_affinities.items()
            if peptide in true_data_peptide_set
        }
        synthetic_values = synthesize_affinities_for_single_allele(
            similarities=allele_similarities,
            peptide_to_affinities=restricted_reverse_lookup,
            smoothing=smoothing_coef,
            exponent=2.0,
            exclude_alleles=[allele])
        synthetic_peptide_set = set(synthetic_values.keys())
        # ordered list of peptides for which we have both true and synthetic
        # affinity values
        combined_peptide_list = [
            peptide
            for peptide in true_data_peptide_list
            if peptide in synthetic_peptide_set
        ]

        if len(combined_peptide_list) < 2:
            print(
                "-- Skipping evaluation of %s due to insufficient data" % (
                    allele,))
            continue

        synthetic_affinity_list = [
            synthetic_values[peptide]
            for peptide in combined_peptide_list
        ]
        true_affinity_dict = {
            peptide: yi
            for (peptide, yi)
            in zip(dataset.peptides, dataset.Y)
        }
        true_affinity_list = [
            true_affinity_dict[peptide]
            for peptide in combined_peptide_list
        ]
        assert len(true_affinity_list) == len(synthetic_affinity_list)
        if all(x == true_affinity_list[0] for x in true_affinity_list):
            print(
                "-- can't compute Kendall's tau for %s, all affinities same" % (
                    allele,))
            continue

        tau, _ = stats.kendalltau(
            synthetic_affinity_list,
            true_affinity_list)
        assert not np.isnan(tau)
        taus.append(tau)
    return np.median(taus)


def create_curried_similarity_matrix(allele_to_peptide_to_affinity, min_weight=2.0):
    raw_sims_dict, overlap_counts, overlap_weights = compute_pairwise_allele_similarities(
        allele_to_peptide_to_affinity,
        min_weight=min_weight)

    complete_sims_dict = fill_in_similarities(
        raw_sims_dict=raw_sims_dict,
        allele_datasets=allele_to_peptide_to_affinity)
    # the above dictionary has keys that are pairs of alleles,
    # conver this to a dict -> dict -> sim layout
    return curry_dictionary(complete_sims_dict)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    allele_datasets = mhcflurry.data.load_allele_datasets(
        args.binding_data_csv,
        max_ic50=args.max_ic50,
        only_human=args.only_human)
    n_binding_values = sum(
        len(dataset.peptides)
        for dataset in allele_datasets.values())
    print("Loaded %d binding values for %d alleles" % (
        n_binding_values, len(allele_datasets)))

    allele_to_peptide_to_affinity = {
        allele: {
            peptide: normalized_affinity
            for (peptide, normalized_affinity)
            in zip(dataset.peptides, dataset.Y)
        }
        for (allele, dataset)
        in allele_datasets.items()
    }
    curried_sims_dict = create_curried_similarity_matrix(allele_to_peptide_to_affinity)

    print("Generated similarities between %d alleles" % len(curried_sims_dict))

    results = {}
    for smoothing_coef in args.smoothing_coefs:
        median_tau = evaluate_smoothing_coef(
            true_data=allele_datasets,
            curried_allele_similarities=curried_sims_dict,
            smoothing_coef=smoothing_coef)
        print("Coef=%f, median Kendall tau=%f" % (
            smoothing_coef,
            median_tau))
        results[smoothing_coef] = median_tau

    print("===")
    (best_coef, best_tau) = max(results.items(), key=lambda x: x[1])
    print("Best coef = %f (tau = %f)" % (best_coef, best_tau))
