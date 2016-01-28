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
from collections import OrderedDict

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

VERBOSE = False


class ScoreSet(object):
    """
    Useful for keeping a collection of score dictionaries
    which map name->score type->list of values.
    """
    def __init__(self, verbose=True):
        self.groups = {}
        self.verbose = verbose

    def add_many(self, group, **kwargs):
        for (k, v) in sorted(kwargs.items()):
            self.add(group, k, v)

    def add(self, group, score_type, value):
        if group not in self.groups:
            self.groups[group] = {}
        if score_type not in self.groups[group]:
            self.groups[group][score_type] = []
        self.groups[group][score_type].append(value)
        if self.verbose:
            print("--> %s:%s %0.4f" % (group, score_type, value))

    def score_types(self):
        result = set([])
        for (g, d) in sorted(self.groups.items()):
            for score_type in sorted(d.keys()):
                result.add(score_type)
        return list(sorted(result))

    def _reduce_scores(self, reduce_fn):
        score_types = self.score_types()
        return {
            group:
                OrderedDict([
                    (score_type, reduce_fn(score_dict[score_type]))
                    for score_type
                    in score_types
                ])
            for (group, score_dict)
            in self.groups.items()
        }

    def averages(self):
        return self._reduce_scores(np.mean)

    def stds(self):
        return self._reduce_scores(np.std)

    def to_csv(self, filename):
        with open(filename, "w") as f:
            header_list = ["name"]
            score_types = self.score_types()
            for score_type in score_types:
                header_list.append(score_type)
                header_list.append(score_type + "_std")

            header_line = ",".join(header_list) + "\n"
            if self.verbose:
                print(header_line)
            f.write(header_line)

            score_averages = self.averages()
            score_stds = self.stds()

            for name in sorted(score_averages.keys()):
                line_elements = [name]
                for score_type in score_types:
                    line_elements.append(
                        "%0.4f" % score_averages[name][score_type])
                    line_elements.append(
                        "%0.4f" % score_stds[name][score_type])
                line = ",".join(line_elements) + "\n"
                if self.verbose:
                    print(line)
                f.write(line)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    imputation_methods = {
        "softImpute": SoftImpute(verbose=VERBOSE),
        "svdImpute-5": IterativeSVD(5, verbose=VERBOSE),
        "svdImpute-10": IterativeSVD(10, verbose=VERBOSE),
        "svdImpute-20": IterativeSVD(20, verbose=VERBOSE),
        "similarityWeightedAveraging": SimilarityWeightedAveraging(
            orientation="columns",
            verbose=VERBOSE),
        "meanFill": SimpleFill("mean"),
        "zeroFill": SimpleFill("zero"),
        "MICE": MICE(
            n_burn_in=5,
            n_imputations=25,
            min_value=None if args.normalize_rows or args.normalize_columns else 0,
            max_value=None if args.normalize_rows or args.normalize_columns else 1,
            verbose=VERBOSE),
        "knnImpute-3": KNN(3, orientation="columns", verbose=VERBOSE, print_interval=1),
        "knnImpute-7": KNN(7, orientation="columns", verbose=VERBOSE, print_interval=1),
        "knnImpute-15": KNN(15, orientation="columns", verbose=VERBOSE, print_interval=1),
    }

    allele_to_peptide_to_affinity = load_allele_dicts(
        args.binding_data_csv,
        max_ic50=args.max_ic50,
        only_human=args.only_human,
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

    X, peptide_order, allele_order = \
        dense_matrix_from_nested_dictionary(peptide_to_allele_to_affinity)

    if args.save_incomplete_affinity_matrix:
        print("Saving incomplete data to %s" % args.save_incomplete_affinity_matrix)
        column_names = [None] * len(allele_order)
        for (name, position) in allele_order.items():
            column_names[position] = name
        row_names = [None] * len(peptide_order)
        for (name, position) in peptide_order.items():
            row_names[position] = name
        df = pd.DataFrame(X, columns=column_names, index=row_names)
        df.to_csv(args.save_incomplete_affinity_matrix, index_label="peptide")

    scores = ScoreSet()

    missing_mask = np.isnan(X)
    observed_mask = ~missing_mask

    n_observed = observed_mask.sum()

    (observed_x, observed_y) = np.where(observed_mask)
    observed_indices = np.ravel_multi_index(
        (observed_x, observed_y),
        dims=observed_mask.shape)

    assert len(observed_indices) == n_observed

    kfold = StratifiedKFold(observed_y, n_folds=5, shuffle=True)

    for fold_idx, (_, indirect_test_indices) in enumerate(kfold):

        test_linear_indices = observed_indices[indirect_test_indices]
        test_coords = np.unravel_index(
            test_linear_indices,
            dims=observed_mask.shape)
        y_true = X[test_coords]

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
                y_true=y_true, y_pred=y_pred, max_ic50=args.max_ic50)

            scores.add_many(
                method_name,
                mae=mae,
                tau=tau,
                f1_score=f1_score,
                auc=auc)

    scores.to_csv(args.output_file)
