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

from mhcflurry.data import load_allele_dicts
from scipy import stats
import numpy as np
import sklearn.metrics

from common import curry_dictionary
from dataset_paths import PETERS2009_CSV_PATH
from synthetic_data import (
    synthesize_affinities_for_single_allele,
    create_reverse_lookup_from_simple_dicts
)
from allele_similarities import save_csv, compute_allele_similarities


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
    "--save-allele-similarity-csv",
    help="Path to allele similarities",
    required=False)

parser.add_argument(
    "--smoothing-coefs",
    default=[0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.0001, 0],
    type=lambda s: [float(si.strip()) for si in s.split(",")],
    help="Smoothing value used for peptides with low weight across alleles")

parser.add_argument(
    "--similarity-exponents",
    default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    type=lambda s: [float(si.strip()) for si in s.split(",")],
    help="Affinities are synthesized by adding up y_ip * sim(i,j) ** exponent")


def evaluate_synthetic_data(
        true_data,
        curried_allele_similarities,
        smoothing_coef,
        exponent,
        max_ic50):
    taus = []
    f1_scores = []
    aucs = []
    peptide_to_affinities = create_reverse_lookup_from_simple_dicts(true_data)
    for allele, dataset in true_data.items():
        this_allele_similarities = curried_allele_similarities[allele]
        this_allele_peptides = set(dataset.keys())
        # create a peptide -> (allele, affinity, weight) dictionary restricted
        # only to the peptides for which we have data for this allele
        restricted_reverse_lookup = {
            peptide: triplet
            for (peptide, triplet) in peptide_to_affinities.items()
            if peptide in this_allele_peptides
        }
        synthetic_values = synthesize_affinities_for_single_allele(
            similarities=this_allele_similarities,
            peptide_to_affinities=restricted_reverse_lookup,
            smoothing=smoothing_coef,
            exponent=exponent,
            exclude_alleles=[allele])

        synthetic_peptide_set = set(synthetic_values.keys())
        # set of peptides for which we have both true and synthetic
        # affinity values
        combined_peptide_set = synthetic_peptide_set.intersection(
            this_allele_peptides)

        combined_peptide_list = list(sorted(combined_peptide_set))

        # print("-- allele = %s, n_actual = %d, n_synthetic = %d, intersection = %d" % (
        #    allele,
        #    len(dataset),
        #    len(synthetic_peptide_set),
        #    len(combined_peptide_list)))

        if len(combined_peptide_list) < 2:
            continue

        synthetic_affinity_list = [
            synthetic_values[peptide]
            for peptide in combined_peptide_list
        ]

        true_affinity_list = [
            dataset[peptide]
            for peptide in combined_peptide_list
        ]
        assert len(true_affinity_list) == len(synthetic_affinity_list)
        if all(x == true_affinity_list[0] for x in true_affinity_list):
            continue

        tau, _ = stats.kendalltau(
            synthetic_affinity_list,
            true_affinity_list)
        assert not np.isnan(tau)
        taus.append(tau)

        true_ic50s = max_ic50 ** (1.0 - np.array(true_affinity_list))
        predicted_ic50s = max_ic50 ** (1.0 - np.array(synthetic_affinity_list))

        true_binding_label = true_ic50s <= 500
        if true_binding_label.all() or not true_binding_label.any():
            # can't compute AUC or F1 without both negative and positive cases
            continue
        auc = sklearn.metrics.roc_auc_score(
            true_binding_label,
            synthetic_affinity_list)
        aucs.append(auc)

        predicted_binding_label = predicted_ic50s <= 500
        if predicted_binding_label.all() or not predicted_binding_label.any():
            # can't compute F1 without both positive and negative predictions
            continue
        f1_score = sklearn.metrics.f1_score(
            true_binding_label,
            predicted_binding_label)
        f1_scores.append(f1_score)
    return taus, aucs, f1_scores


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    allele_to_peptide_to_affinity = load_allele_dicts(
        args.binding_data_csv,
        max_ic50=args.max_ic50,
        only_human=args.only_human,
        regression_output=True)

    n_binding_values = sum(
        len(allele_dict)
        for allele_dict in
        allele_to_peptide_to_affinity.values()
    )
    print("Loaded %d binding values for %d alleles" % (
        n_binding_values, len(allele_to_peptide_to_affinity)))

    complete_sims_dict, overlap_counts, overlap_weights = \
        compute_allele_similarities(
            allele_to_peptide_to_affinity,
            min_weight=0.1)

    if args.save_allele_similarity_csv:
        save_csv(
            filename=args.save_allele_similarity_csv,
            sims=complete_sims_dict,
            overlap_counts=overlap_counts,
            overlap_weights=overlap_weights)

    # the above dictionary has keys that are pairs of alleles,
    # conver this to a dict -> dict -> sim layout
    curried_sims_dict = curry_dictionary(complete_sims_dict)

    print("Generated similarities between %d alleles" % len(curried_sims_dict))

    results = {}
    for smoothing_coef in args.smoothing_coefs:
        for exponent in args.similarity_exponents:
            taus, aucs, f1_scores = evaluate_synthetic_data(
                true_data=allele_to_peptide_to_affinity,
                curried_allele_similarities=curried_sims_dict,
                smoothing_coef=smoothing_coef,
                exponent=exponent,
                max_ic50=args.max_ic50)
            median_tau = np.median(taus)
            median_f1 = np.median(f1_scores)
            median_auc = np.median(aucs)
            print(
                "Exp=%f, Coef=%f, tau=%0.4f, AUC = %0.4f, F1 = %0.4f" % (
                    exponent,
                    smoothing_coef,
                    median_tau,
                    median_auc,
                    median_f1))
            scores = (median_tau, median_auc, median_f1)
            results[(exponent, smoothing_coef)] = scores

    print("===")
    ((best_exponent, best_coef), (median_tau, median_auc, median_f1)) = max(
        results.items(),
        key=lambda x: x[1][0] * x[1][1] * x[1][2])
    print("Best exponent = %f, coef = %f (tau=%0.4f, AUC=%0.4f, F1=%0.4f)" % (
        best_exponent,
        best_coef,
        median_tau,
        median_auc,
        median_f1))
