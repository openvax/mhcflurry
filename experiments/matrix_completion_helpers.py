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

from fancyimpute.dictionary_helpers import (
    dense_matrix_from_nested_dictionary,
    transpose_nested_dictionary,
)
from mhcflurry.data import load_allele_dicts
import sklearn.metrics
from sklearn.cross_validation import StratifiedKFold
from scipy import stats
import numpy as np


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


def prune_data(
        X,
        peptide_list,
        allele_list,
        min_observations_per_peptide=1,
        min_observations_per_allele=1):
    observed_mask = np.isfinite(X)
    n_observed_per_peptide = observed_mask.sum(axis=1)
    too_few_peptide_observations = (
        n_observed_per_peptide < min_observations_per_peptide)
    if too_few_peptide_observations.any():
        drop_peptide_indices = np.where(too_few_peptide_observations)[0]
        keep_peptide_indices = np.where(~too_few_peptide_observations)[0]
        print("Dropping %d peptides with <%d observations" % (
            len(drop_peptide_indices),
            min_observations_per_peptide))
        X = X[keep_peptide_indices]
        observed_mask = observed_mask[keep_peptide_indices]
        peptide_list = [peptide_list[i] for i in keep_peptide_indices]

    n_observed_per_allele = observed_mask.sum(axis=0)
    too_few_allele_observations = (
        n_observed_per_allele < min_observations_per_peptide)
    if too_few_peptide_observations.any():
        drop_allele_indices = np.where(too_few_allele_observations)[0]
        keep_allele_indices = np.where(~too_few_allele_observations)[0]

        print("Dropping %d alleles with <%d observations: %s" % (
            len(drop_allele_indices),
            min_observations_per_allele,
            [allele_list[i] for i in drop_allele_indices]))
        X = X[:, keep_allele_indices]
        observed_mask = observed_mask[:, keep_allele_indices]
        allele_list = [allele_list[i] for i in keep_allele_indices]
    return X, peptide_list, allele_list


def load_data(
        binding_data_csv,
        max_ic50,
        only_human=False,
        min_observations_per_allele=1,
        min_observations_per_peptide=1):
    allele_to_peptide_to_affinity = load_allele_dicts(
        binding_data_csv,
        max_ic50=max_ic50,
        only_human=only_human,
        regression_output=True)
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
    return prune_data(
        X,
        peptide_list,
        allele_list,
        min_observations_per_peptide=min_observations_per_peptide,
        min_observations_per_allele=min_observations_per_allele)


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
