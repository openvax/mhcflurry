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
from sklearn.cross_validation import KFold

from dataset_paths import PETERS2009_CSV_PATH
from synthetic_data import (
    synthesize_affinities_for_single_allele,
    create_reverse_lookup_from_simple_dicts
)
from allele_similarities import compute_allele_similarities


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
        allele_to_peptide_to_affinity,
        smoothing_coef,
        exponent,
        max_ic50,
        n_folds=4):
    """
    Use cross-validation over entries of each allele to determine the predictive
    accuracy of data synthesis on known affinities.
    """
    taus = []
    f1_scores = []
    aucs = []

    peptide_to_affinities = create_reverse_lookup_from_simple_dicts(
        allele_to_peptide_to_affinity)
    for allele, dataset in allele_to_peptide_to_affinity.items():
        peptides = list(dataset.keys())
        affinities = list(dataset.values())
        all_indices = np.arange(len(peptides))
        np.random.shuffle(all_indices)
        n_samples = len(peptides)
        cv_aucs = []
        cv_taus = []
        cv_f1_scores = []

        if len(peptides) < n_folds:
            continue

        kfold_iterator = enumerate(
            KFold(n_samples, n_folds=n_folds, shuffle=True))

        for fold_number, (train_indices, test_indices) in kfold_iterator:
            train_peptides = [peptides[i] for i in train_indices]
            train_affinities = [affinities[i] for i in train_indices]
            test_peptide_set = set([peptides[i] for i in test_indices])

            # copy the affinity data for all alleles other than this one
            fold_affinity_dict = {
                allele_key: affinity_dict
                for (allele_key, affinity_dict)
                in allele_to_peptide_to_affinity.items()
                if allele_key != allele
            }
            # include an affinity dictionary for this allele which
            # only uses training data
            fold_affinity_dict[allele] = {
                train_peptide: train_affinity
                for (train_peptide, train_affinity)
                in zip(train_peptides, train_affinities)
            }
            allele_similarities, overlap_counts, overlap_weights = \
                compute_allele_similarities(
                    allele_to_peptide_to_affinity=fold_affinity_dict,
                    min_weight=0.1)
            this_allele_similarities = allele_similarities[allele]
            # create a peptide -> list[(allele, affinity, weight)] dictionary
            # restricted only to the peptides for which we want to test accuracy
            test_reverse_lookup = {
                peptide: triplets
                for (peptide, triplets) in peptide_to_affinities.items()
                if peptide in test_peptide_set
            }
            synthetic_values = synthesize_affinities_for_single_allele(
                similarities=this_allele_similarities,
                peptide_to_affinities=test_reverse_lookup,
                smoothing=smoothing_coef,
                exponent=exponent,
                exclude_alleles=[allele])

            synthetic_peptide_set = set(synthetic_values.keys())
            # set of peptides for which we have both true and synthetic
            # affinity values
            combined_peptide_set = synthetic_peptide_set.intersection(
                test_peptide_set)

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
            print("==> %s (CV fold %d/%d) tau = %f" % (
                allele,
                fold_number + 1,
                n_folds,
                tau))
            cv_taus.append(tau)

            true_ic50s = max_ic50 ** (1.0 - np.array(true_affinity_list))
            predicted_ic50s = max_ic50 ** (1.0 - np.array(synthetic_affinity_list))

            true_binding_label = true_ic50s <= 500
            if true_binding_label.all() or not true_binding_label.any():
                # can't compute AUC or F1 without both negative and positive cases
                continue
            auc = sklearn.metrics.roc_auc_score(
                true_binding_label,
                synthetic_affinity_list)
            print("==> %s (CV fold %d/%d) AUC = %f" % (
                allele,
                fold_number + 1,
                n_folds,
                auc))
            cv_aucs.append(auc)

            predicted_binding_label = predicted_ic50s <= 500
            if predicted_binding_label.all() or not predicted_binding_label.any():
                # can't compute F1 without both positive and negative predictions
                continue
            f1_score = sklearn.metrics.f1_score(
                true_binding_label,
                predicted_binding_label)
            print("==> %s (CV fold %d/%d) F1 = %f" % (
                allele,
                fold_number + 1,
                n_folds,
                f1_score))
            cv_f1_scores.append(f1_score)
        if len(cv_taus) > 0:
            taus.append(np.mean(cv_taus))
        if len(cv_aucs) > 0:
            aucs.append(np.mean(cv_aucs))
        if len(f1_scores) > 0:
            f1_scores.append(np.mean(f1_scores))
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

    results = {}
    for smoothing_coef in args.smoothing_coefs:
        for exponent in args.similarity_exponents:
            taus, aucs, f1_scores = evaluate_synthetic_data(
                allele_to_peptide_to_affinity=allele_to_peptide_to_affinity,
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
