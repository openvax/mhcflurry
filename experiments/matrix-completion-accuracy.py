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
import numpy as np
import pandas as pd

from dataset_paths import PETERS2009_CSV_PATH
from score_set import ScoreSet
from matrix_completion_helpers import (
    load_data,
    evaluate_predictions,
    stratified_cross_validation
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
    "--min-observations-per-peptide",
    type=int,
    default=2,
    help="Drop peptide entries with fewer than this number of measured affinities")

parser.add_argument(
    "--n-folds",
    default=10,
    type=int,
    help="Number of cross-validation folds")

parser.add_argument(
    "--verbose",
    default=False,
    action="store_true")


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


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    imputation_methods = create_imputation_methods(
        verbose=args.verbose,
        clip_imputed_values=not (args.normalize_rows or args.normalize_rows),
    )
    print("Imputation methods: %s" % imputation_methods)

    X, peptide_list, allele_list = load_data(
        binding_data_csv=args.binding_data_csv,
        max_ic50=args.max_ic50,
        only_human=args.only_human,
        min_observations_per_allele=args.n_folds,
        min_observations_per_peptide=args.min_observations_per_peptide)
    observed_mask = np.isfinite(X)
    print("Loaded binding data, shape: %s, n_observed=%d/%d (%0.2f%%)" % (
        X.shape,
        observed_mask.sum(),
        X.size,
        100.0 * observed_mask.sum() / X.size))
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
