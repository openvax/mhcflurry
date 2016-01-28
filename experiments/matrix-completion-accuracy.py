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

from fancyimpute import (
    SoftImpute,
    IterativeSVD,
    SimilarityWeightedAveraging,
    KNN,
    MICE,
    BiScaler,
    SimpleFill,
)
from fancyimpute.dictionary_helpers import (
    dense_matrix_from_nested_dictionary,
    transpose_nested_dictionary,
)
from mhcflurry.data import load_allele_dicts
import sklearn.metrics
from sklearn.cross_validation import StratifiedKFold
from scipy import stats
import numpy as np
import pandas as pd

from dataset_paths import PETERS2009_CSV_PATH
from score_set import ScoreSet

parser = argparse.ArgumentParser()

parser.add_argument(
    "--binding-data-csv",
    default=PETERS2009_CSV_PATH)

parser.add_argument(
    "--max-ic50",
    default=50000.0,
    type=float)

parser.add_argument(
    "--save-incomplete-affinity-matrix",
    default=None,
    help="Path to CSV which will contains the incomplete affinity matrix")

parser.add_argument(
    "--only-human",
    default=False,
    action="store_true")

parser.add_argument(
    "--output-file",
    default="matrix-completion-accuracy-results.csv")

parser.add_argument(
    "--normalize-columns",
    default=False,
    action="store_true",
    help="Center and rescale columns of affinity matrix")


parser.add_argument(
    "--normalize-rows",
    default=False,
    action="store_true",
    help="Center and rescale rows of affinity matrix")

parser.add_argument(
    "--n-folds",
    default=10,
    type=int,
    help="Number of cross-validation folds")

parser.add_argument(
    "--verbose",
    default=False,
    action="store_true")


def evaluate_predictions(
        y_true,
        y_pred,
        max_ic50):
    """
    Return mean absolute error, Kendall tau, AUC and F1 score
    """
    if len(y_pred) != len(y_true):
        raise ValueError("y_pred must have same number of elements as y_true")

    mae = np.mean(np.abs(y_true - y_pred))

    # if all predictions are the same
    if (y_pred[0] == y_pred).all():
        return (mae, 0, 0.5, 0)

    tau, _ = stats.kendalltau(
        y_pred,
        y_true)

    true_ic50s = max_ic50 ** (1.0 - np.array(y_true))
    predicted_ic50s = max_ic50 ** (1.0 - np.array(y_pred))

    true_binding_label = true_ic50s <= 500
    if true_binding_label.all() or not true_binding_label.any():
        # can't compute AUC or F1 without both negative and positive cases
        return (mae, tau, 0.5, 0)

    auc = sklearn.metrics.roc_auc_score(true_binding_label, y_pred)

    predicted_binding_label = predicted_ic50s <= 500
    if predicted_binding_label.all() or not predicted_binding_label.any():
        # can't compute F1 without both positive and negative predictions
        return (mae, tau, auc, 0)

    f1_score = sklearn.metrics.f1_score(
        true_binding_label,
        predicted_binding_label)

    return mae, tau, auc, f1_score


def create_imputation_methods(
        verbose=False,
        clip_imputed_values=False,
        knn_print_interval=20,
        knn_params=[1, 3, 5],
        softimpute_params=[1, 5, 10],
        svd_params=[5, 10, 20]):
    min_value = 0 if clip_imputed_values else None
    max_value = 1 if clip_imputed_values else None
    result_dict = {
        "meanFill": SimpleFill("mean"),
        "zeroFill": SimpleFill("zero"),
        "mice": MICE(
            n_burn_in=5,
            n_imputations=25,
            min_value=min_value,
            max_value=max_value,
            verbose=verbose),
        "similarityWeightedAveraging": SimilarityWeightedAveraging(
            orientation="columns",
            verbose=verbose),
    }
    for threshold in softimpute_params:
        result_dict["softImpute-%d" % threshold] = SoftImpute(
            threshold,
            verbose=verbose,
            min_value=min_value,
            max_value=max_value)
    for rank in svd_params:
        result_dict["svdImpute-%d" % rank] = IterativeSVD(
            rank,
            verbose=verbose,
            min_value=min_value,
            max_value=max_value)
    for k in knn_params:
        result_dict["knnImpute-%d" % k] = KNN(
            k,
            orientation="columns",
            verbose=verbose,
            print_interval=knn_print_interval)
    return result_dict


def load_data(binding_data_csv, max_ic50, only_human=False, min_allele_size=1):
    allele_to_peptide_to_affinity = load_allele_dicts(
        binding_data_csv,
        max_ic50=max_ic50,
        only_human=only_human,
        regression_output=True,
        min_allele_size=min_allele_size)
    peptide_to_allele_to_affinity = transpose_nested_dictionary(
        allele_to_peptide_to_affinity)
    n_binding_values = sum(
        len(allele_dict)
        for allele_dict in
        allele_to_peptide_to_affinity.values()
    )
    print("Loaded %d binding values for %d alleles" % (
        n_binding_values,
        len(allele_to_peptide_to_affinity)))

    X, peptide_list, allele_list = \
        dense_matrix_from_nested_dictionary(peptide_to_allele_to_affinity)

    missing_mask = np.isnan(X)
    observed_mask = ~missing_mask

    n_observed_per_peptide = observed_mask.sum(axis=1)
    min_observed_per_peptide = n_observed_per_peptide.min()
    min_peptide_indices = np.where(
        n_observed_per_peptide == min_observed_per_peptide)[0]
    print("%d peptides with %d observations" % (
        len(min_peptide_indices),
        min_observed_per_peptide))

    n_observed_per_allele = observed_mask.sum(axis=0)
    min_observed_per_allele = n_observed_per_allele.min()
    min_allele_indices = np.where(
        n_observed_per_allele == min_observed_per_allele)[0]
    print("%d alleles with %d observations: %s" % (
        len(min_allele_indices),
        min_observed_per_allele,
        [allele_list[i] for i in min_allele_indices]))
    return X, missing_mask, observed_mask, peptide_list, allele_list


def index_counts(indices):
    max_index = indices.max()
    counts = np.zeros(max_index + 1, dtype=int)
    for index in indices:
        counts[index] += 1
    return counts


def stratified_cross_validation(X, observed_mask, n_folds=10):
    n_observed = observed_mask.sum()

    (observed_peptide_index, observed_allele_index) = np.where(observed_mask)
    observed_indices = np.ravel_multi_index(
        (observed_peptide_index, observed_allele_index),
        dims=observed_mask.shape)

    assert len(observed_indices) == n_observed

    observed_allele_counts = observed_mask.sum(axis=0)
    print("# observed per allele: %s" % (observed_allele_counts,))
    assert (index_counts(observed_allele_index) == observed_allele_counts).all()

    kfold = StratifiedKFold(
        observed_allele_index,
        n_folds=n_folds,
        shuffle=True)

    for (_, indirect_test_indices) in kfold:
        test_linear_indices = observed_indices[indirect_test_indices]
        test_coords = np.unravel_index(
            test_linear_indices,
            dims=observed_mask.shape)

        test_allele_counts = index_counts(test_coords[1])
        allele_fractions = test_allele_counts / observed_allele_counts.astype(float)
        print("Fraction of each allele in this CV fold: %s" % (allele_fractions,))

        X_test_vector = X[test_coords]

        X_fold = X.copy()
        X_fold[test_coords] = np.nan

        empty_row_mask = np.isfinite(X_fold).sum(axis=1) == 0
        ok_row_mask = ~empty_row_mask
        ok_row_indices = np.where(ok_row_mask)[0]

        empty_col_mask = np.isfinite(X_fold).sum(axis=0) == 0
        ok_col_mask = ~empty_col_mask
        ok_col_indices = np.where(ok_col_mask)[0]

        ok_mesh = np.ix_(ok_row_indices, ok_col_indices)

        print("Dropping %d empty rows, %d empty columns" % (
            empty_row_mask.sum(),
            empty_col_mask.sum()))
        yield (X_fold, ok_mesh, test_coords, X_test_vector)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    imputation_methods = create_imputation_methods(
        verbose=args.verbose,
        clip_imputed_values=not (args.normalize_rows or args.normalize_rows),
    )
    print("Imputation methods: %s" % imputation_methods)

    X, missing_mask, observed_mask, peptide_list, allele_list = load_data(
        binding_data_csv=args.binding_data_csv,
        max_ic50=args.max_ic50,
        only_human=args.only_human,
        min_allele_size=args.n_folds)

    if args.save_incomplete_affinity_matrix:
        print("Saving incomplete data to %s" % args.save_incomplete_affinity_matrix)
        df = pd.DataFrame(X, columns=allele_list, index=peptide_list)
        df.to_csv(args.save_incomplete_affinity_matrix, index_label="peptide")

    scores = ScoreSet()
    kfold = stratified_cross_validation(
        X=X,
        observed_mask=observed_mask,
        n_folds=args.n_folds)
    for fold_idx, (X_fold, ok_mesh, test_coords, X_test_vector) in enumerate(kfold):
        X_fold_reduced = X_fold[ok_mesh]
        biscaler = BiScaler(
            scale_rows=args.normalize_rows,
            center_rows=args.normalize_rows,
            scale_columns=args.normalize_columns,
            center_columns=args.normalize_columns)
        X_fold_reduced_scaled = biscaler.fit_transform(X=X_fold_reduced)
        for (method_name, solver) in sorted(imputation_methods.items()):
            print("CV fold %d/%d, running %s" % (
                fold_idx + 1,
                args.n_folds,
                method_name))
            X_completed_reduced_scaled = solver.complete(X_fold_reduced)
            X_completed_reduced = biscaler.inverse_transform(
                X_completed_reduced_scaled)
            X_completed = np.zeros_like(X)
            X_completed[ok_mesh] = X_completed_reduced
            y_pred = X_completed[test_coords]
            mae, tau, auc, f1_score = evaluate_predictions(
                y_true=X_test_vector, y_pred=y_pred, max_ic50=args.max_ic50)
            scores.add_many(
                method_name,
                mae=mae,
                tau=tau,
                f1_score=f1_score,
                auc=auc)
    scores.to_csv(args.output_file)
